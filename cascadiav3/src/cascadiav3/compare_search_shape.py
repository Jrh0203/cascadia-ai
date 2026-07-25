"""Direct score comparison between two Gumbel benchmark reports.

The comparator intentionally has no source, artifact, rules-identity,
preregistration, sample-size, or promotion gate. It pairs whatever seed
results the reports have in common and reports the observed score difference
and uncertainty.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .torch_benchmark_stats import paired_delta_stats

DEFAULT_VARIED_KEYS = ("determinizations",)


def _load_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "pass":
        raise ValueError(f"report is not complete: {path}")
    return report


def _scores_by_seed(report: dict[str, Any]) -> dict[int, float]:
    return {
        int(row["seed"]): float(row["mean_score_per_seat"])
        for row in report.get("candidate_per_seed", [])
    }


def _assessment(stats: dict[str, Any]) -> str:
    low = stats.get("t_ci_low")
    high = stats.get("t_ci_high")
    if low is not None and low > 0.0:
        return "candidate_ahead"
    if high is not None and high < 0.0:
        return "baseline_ahead"
    return "uncertain"


def _mean_score(report: dict[str, Any], scores: dict[int, float]) -> float:
    summary = report.get("strategies", {}).get("gumbel-search", {})
    if summary.get("mean_seat_score") is not None:
        return float(summary["mean_seat_score"])
    return sum(scores.values()) / len(scores)


def build_comparison(
    baseline_path: Path,
    candidate_path: Path,
    source_revision: str | None = None,
    varied_keys: tuple[str, ...] = DEFAULT_VARIED_KEYS,
) -> dict[str, Any]:
    """Compare reports.

    ``source_revision`` and ``varied_keys`` remain accepted so old launch
    scripts keep working; they are descriptive only.
    """
    del source_revision
    baseline = _load_report(baseline_path)
    candidate = _load_report(candidate_path)
    baseline_scores = _scores_by_seed(baseline)
    candidate_scores = _scores_by_seed(candidate)
    seeds = sorted(baseline_scores.keys() & candidate_scores.keys())
    if not seeds:
        raise ValueError("reports have no scored seeds in common")

    deltas = [candidate_scores[seed] - baseline_scores[seed] for seed in seeds]
    stats = paired_delta_stats(deltas)
    baseline_timing = baseline.get("strategies", {}).get("gumbel-search", {})
    candidate_timing = candidate.get("strategies", {}).get("gumbel-search", {})
    varied = {
        key: {
            "baseline": baseline.get("search", {}).get(key),
            "candidate": candidate.get("search", {}).get(key),
        }
        for key in varied_keys
    }
    return {
        "status": "pass",
        "matched_seeds": seeds,
        "baseline_report": str(baseline_path),
        "candidate_report": str(candidate_path),
        "baseline_manifest": baseline.get("manifest"),
        "candidate_manifest": candidate.get("manifest"),
        "varied": varied,
        "baseline_mean_seat_score": _mean_score(baseline, baseline_scores),
        "candidate_mean_seat_score": _mean_score(candidate, candidate_scores),
        "paired_score_deltas": [
            {
                "seed": seed,
                "baseline_score": baseline_scores[seed],
                "candidate_score": candidate_scores[seed],
                "delta": delta,
            }
            for seed, delta in zip(seeds, deltas, strict=True)
        ],
        "paired_delta_stats": stats,
        "assessment": _assessment(stats),
        "timing": {
            "baseline_mean_decision_seconds": baseline_timing.get(
                "mean_total_decision_seconds"
            ),
            "candidate_mean_decision_seconds": candidate_timing.get(
                "mean_total_decision_seconds"
            ),
            "baseline_wall_seconds": baseline.get("candidate_wall_seconds"),
            "candidate_wall_seconds": candidate.get("candidate_wall_seconds"),
        },
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    stats = report["paired_delta_stats"]
    lines = [
        "# Strength Comparison",
        "",
        f"Matched games: `{len(report['matched_seeds'])}`",
        f"Baseline mean seat score: `{report['baseline_mean_seat_score']:.4f}`",
        f"Candidate mean seat score: `{report['candidate_mean_seat_score']:.4f}`",
        f"Paired delta: `{stats['mean']:+.4f}`",
    ]
    if stats["t_ci_low"] is not None:
        lines.append(
            f"95% t-CI: `[{stats['t_ci_low']:+.4f}, {stats['t_ci_high']:+.4f}]`"
        )
    lines.extend(
        [
            f"Assessment: `{report['assessment']}`",
            "",
            "This is a visible measurement, not a promotion gate. Add games or "
            "change course whenever useful.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--source-revision", default="", help=argparse.SUPPRESS)
    parser.add_argument("--varied-key", action="append", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    args = parser.parse_args()
    varied_keys = tuple(args.varied_key) if args.varied_key else DEFAULT_VARIED_KEYS
    report = build_comparison(
        Path(args.baseline),
        Path(args.candidate),
        args.source_revision or None,
        varied_keys=varied_keys,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, Path(args.summary_out))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
