"""Validate and compare the corrected-rules scalar and distq baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .torch_benchmark_stats import paired_delta_stats

RULESET_ID = "cascadia_research_aaaaa_4p_card_a_no_habitat_bonus_rules_2026_07_16"


def _load(path: Path, source_revision: str) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "pass":
        raise ValueError(f"report is not passing: {path}")
    if report.get("ruleset_id") != RULESET_ID:
        raise ValueError(f"ruleset mismatch in {path}")
    if report.get("source_revision") != source_revision:
        raise ValueError(f"source revision mismatch in {path}")
    if report.get("search", {}).get("market_decision_samples") != 8:
        raise ValueError(f"market decision sample mismatch in {path}")
    return report


def _mean_score(report: dict[str, Any]) -> float:
    return float(report["strategies"]["gumbel-search"]["mean_seat_score"])


def _paired(left: dict[str, Any], right: dict[str, Any], label: str) -> dict[str, Any]:
    if left["seeds"] != right["seeds"]:
        raise ValueError(f"seed mismatch for {label}")
    left_by_seed = {int(row["seed"]): float(row["mean_score_per_seat"]) for row in left["candidate_per_seed"]}
    right_by_seed = {
        int(row["seed"]): float(row["mean_score_per_seat"])
        for row in right["candidate_per_seed"]
    }
    if left_by_seed.keys() != right_by_seed.keys():
        raise ValueError(f"per-seed coverage mismatch for {label}")
    deltas = [left_by_seed[seed] - right_by_seed[seed] for seed in sorted(left_by_seed)]
    return {
        "label": label,
        "left_experiment_id": left["experiment_id"],
        "right_experiment_id": right["experiment_id"],
        "left_mean_seat_score": _mean_score(left),
        "right_mean_seat_score": _mean_score(right),
        "paired_delta_stats": paired_delta_stats(deltas),
    }


def build_comparison(report_dir: Path, source_revision: str) -> dict[str, Any]:
    reports = {
        "cycle4_n256_d4": _load(report_dir / "rules_20260709_cycle4_n256_d4.json", source_revision),
        "distq_k8_n256_d4": _load(
            report_dir / "rules_20260709_distq_k8_n256_d4.json", source_revision
        ),
        "cycle4_n1024_d16": _load(
            report_dir / "rules_20260709_cycle4_n1024_d16.json", source_revision
        ),
        "distq_k8_n1024_d16": _load(
            report_dir / "rules_20260709_distq_k8_n1024_d16.json", source_revision
        ),
    }
    if len({tuple(report["seeds"]) for report in reports.values()}) != 1:
        raise ValueError("reports do not share one seed set")
    for name, report in reports.items():
        expected_n = 256 if "n256" in name else 1024
        expected_d = 4 if "d4" in name else 16
        if report["search"]["n_simulations"] != expected_n:
            raise ValueError(f"simulation budget mismatch in {name}")
        if report["search"]["determinizations"] != expected_d:
            raise ValueError(f"determinization budget mismatch in {name}")

    comparisons = {
        "distq_minus_cycle4_n256_d4": _paired(
            reports["distq_k8_n256_d4"],
            reports["cycle4_n256_d4"],
            "distq_k8 - cycle4 at n256/d4",
        ),
        "distq_minus_cycle4_n1024_d16": _paired(
            reports["distq_k8_n1024_d16"],
            reports["cycle4_n1024_d16"],
            "distq_k8 - cycle4 at n1024/d16",
        ),
        "cycle4_n1024_d16_minus_n256_d4": _paired(
            reports["cycle4_n1024_d16"],
            reports["cycle4_n256_d4"],
            "cycle4 n1024/d16 - n256/d4",
        ),
        "distq_n1024_d16_minus_n256_d4": _paired(
            reports["distq_k8_n1024_d16"],
            reports["distq_k8_n256_d4"],
            "distq_k8 n1024/d16 - n256/d4",
        ),
    }
    return {
        "status": "pass",
        "ruleset_id": RULESET_ID,
        "source_revision": source_revision,
        "seeds": next(iter(reports.values()))["seeds"],
        "reports": {
            name: {
                "experiment_id": report["experiment_id"],
                "mean_seat_score": _mean_score(report),
                "market_decisions": report["market_decisions"],
            }
            for name, report in reports.items()
        },
        "comparisons": comparisons,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Corrected-Rules Rebaseline Verdict",
        "",
        f"Ruleset: `{report['ruleset_id']}`",
        f"Source revision: `{report['source_revision']}`",
        f"Games: `{len(report['seeds'])}` paired seeds per arm",
        "",
        "| Comparison | Left | Right | Delta | 95% t-CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["comparisons"].values():
        stats = row["paired_delta_stats"]
        lines.append(
            f"| {row['label']} | {row['left_mean_seat_score']:.4f} | "
            f"{row['right_mean_seat_score']:.4f} | {stats['mean']:+.4f} | "
            f"[{stats['t_ci_low']:+.4f}, {stats['t_ci_high']:+.4f}] |"
        )
    lines.extend(["", "## Refresh Decisions", ""])
    for name, row in report["reports"].items():
        market = row["market_decisions"]
        lines.append(
            f"- `{name}`: {market['accepted']} accept / {market['declined']} decline "
            f"({market['acceptance_rate_when_available']:.2%} accept when available)"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", default="cascadiav3/reports")
    parser.add_argument("--source-revision", required=True)
    parser.add_argument(
        "--out", default="cascadiav3/reports/rules_20260709_rebaseline_verdict.json"
    )
    parser.add_argument(
        "--summary-out", default="cascadiav3/reports/rules_20260709_rebaseline_verdict.md"
    )
    args = parser.parse_args()
    report = build_comparison(Path(args.report_dir), args.source_revision)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, Path(args.summary_out))
    print(json.dumps(report["comparisons"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
