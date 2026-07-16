"""Paired verdict for search-shape ablations (worlds, sigma calibration, ...).

The default varied key is `determinizations` (worlds allocation): worlds are
cycled inside the fixed simulation budget, so changing the count alters
evaluations at every ply — like market sample counts and unlike exact-K1,
trace identity is not a validity condition and is not checked. Pass
`--varied-key` (repeatable) to compare arms that differ in other search
knobs (e.g. `c_scale`, `sigma_norm`, `paired_rollouts`); every search
setting outside the varied set must still match exactly. The verdict is the
campaign-standard paired per-seed score delta with a 95% t-CI over >=100
matched seeds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .torch_benchmark_stats import paired_delta_stats

RULESET_ID = "cascadia_research_aaaaa_4p_card_a_no_habitat_bonus_rules_2026_07_16"
DEFAULT_VARIED_KEYS = ("determinizations",)


def _load_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "pass":
        raise ValueError(f"report is not passing: {path}")
    if report.get("ruleset_id") != RULESET_ID:
        raise ValueError(f"ruleset mismatch in {path}")
    if report.get("control", {}).get("kind") != "none":
        raise ValueError(f"search-shape comparison requires control=none in {path}")
    return report


def _search_without_varied(report: dict[str, Any], varied_keys: tuple[str, ...]) -> dict[str, Any]:
    search = dict(report.get("search", {}))
    for key in varied_keys:
        search.pop(key, None)
    return search


def _scores_by_seed(report: dict[str, Any], label: str) -> dict[int, float]:
    scores = {
        int(row["seed"]): float(row["mean_score_per_seat"])
        for row in report.get("candidate_per_seed", [])
    }
    if len(scores) != len(report.get("seeds", [])):
        raise ValueError(f"duplicate or incomplete per-seed scores in {label}")
    return scores


def _score_verdict(stats: dict[str, Any]) -> str:
    low = stats.get("t_ci_low")
    high = stats.get("t_ci_high")
    if low is not None and low > 0.0:
        return "ci_positive"
    if high is not None and high < 0.0:
        return "ci_negative"
    return "inconclusive"


def build_comparison(
    baseline_path: Path,
    candidate_path: Path,
    source_revision: str | None = None,
    varied_keys: tuple[str, ...] = DEFAULT_VARIED_KEYS,
) -> dict[str, Any]:
    if not varied_keys:
        raise ValueError("at least one varied search key is required")
    baseline = _load_report(baseline_path)
    candidate = _load_report(candidate_path)
    baseline_revision = baseline.get("source_revision")
    candidate_revision = candidate.get("source_revision")
    if not baseline_revision or baseline_revision != candidate_revision:
        raise ValueError("reports must share one non-empty source revision")
    if source_revision is not None and baseline_revision != source_revision:
        raise ValueError("reports do not match the required source revision")
    if baseline.get("seeds") != candidate.get("seeds"):
        raise ValueError("seed mismatch between reports")
    if _search_without_varied(baseline, varied_keys) != _search_without_varied(
        candidate, varied_keys
    ):
        raise ValueError(f"search settings differ beyond {'/'.join(varied_keys)}")
    varied: dict[str, dict[str, Any]] = {}
    for key in varied_keys:
        varied[key] = {
            "baseline": baseline.get("search", {}).get(key),
            "candidate": candidate.get("search", {}).get(key),
        }
    if varied_keys == DEFAULT_VARIED_KEYS:
        worlds = varied["determinizations"]
        if (
            not worlds["baseline"]
            or not worlds["candidate"]
            or worlds["baseline"] == worlds["candidate"]
        ):
            raise ValueError("arms must use two distinct positive world counts")
    elif all(entry["baseline"] == entry["candidate"] for entry in varied.values()):
        raise ValueError(f"arms are identical across the varied keys {'/'.join(varied_keys)}")
    baseline_manifest = str(baseline.get("manifest", ""))
    if not baseline_manifest or baseline_manifest != str(candidate.get("manifest", "")):
        raise ValueError("model manifest identity mismatch")

    seeds = [int(seed) for seed in baseline["seeds"]]
    if len(seeds) < 2:
        raise ValueError("search-shape comparison requires at least two paired seeds")
    baseline_scores = _scores_by_seed(baseline, "baseline")
    candidate_scores = _scores_by_seed(candidate, "candidate")
    if baseline_scores.keys() != candidate_scores.keys() or set(seeds) != baseline_scores.keys():
        raise ValueError("per-seed score coverage mismatch")
    deltas = [candidate_scores[seed] - baseline_scores[seed] for seed in seeds]
    stats = paired_delta_stats(deltas)
    verdict = _score_verdict(stats)
    promotion_scale = len(seeds) >= 100

    varied_provenance: dict[str, Any] = {}
    for key, entry in varied.items():
        varied_provenance[f"baseline_{key}"] = entry["baseline"]
        varied_provenance[f"candidate_{key}"] = entry["candidate"]

    baseline_summary = baseline["strategies"]["gumbel-search"]
    candidate_summary = candidate["strategies"]["gumbel-search"]
    return {
        "status": "pass",
        "scientific_eligibility": (
            "promotion_scale_paired_gate" if promotion_scale else "engineering_smoke_only"
        ),
        "ruleset_id": RULESET_ID,
        "source_revision": baseline_revision,
        "manifest_name": Path(baseline_manifest).name,
        "seeds": seeds,
        "varied_keys": list(varied_keys),
        "search": _search_without_varied(baseline, varied_keys) | varied_provenance,
        "baseline_mean_seat_score": float(baseline_summary["mean_seat_score"]),
        "candidate_mean_seat_score": float(candidate_summary["mean_seat_score"]),
        "paired_score_deltas": [
            {"seed": seed, "delta": delta, "baseline_score": baseline_scores[seed]}
            for seed, delta in zip(seeds, deltas, strict=True)
        ],
        "paired_delta_stats": stats,
        "score_verdict": verdict,
        "proceed_to_high_budget": bool(promotion_scale and verdict == "ci_positive"),
        "timing": {
            "baseline_mean_decision_seconds": float(
                baseline_summary["mean_total_decision_seconds"]
            ),
            "candidate_mean_decision_seconds": float(
                candidate_summary["mean_total_decision_seconds"]
            ),
            "baseline_wall_seconds": baseline["candidate_wall_seconds"],
            "candidate_wall_seconds": candidate["candidate_wall_seconds"],
        },
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    stats = report["paired_delta_stats"]
    timing = report["timing"]
    varied_keys = report.get("varied_keys", list(DEFAULT_VARIED_KEYS))
    varied_lines = [
        f"- {key}: `{report['search'].get(f'baseline_{key}')}` -> "
        f"`{report['search'].get(f'candidate_{key}')}`"
        for key in varied_keys
    ]
    lines = [
        "# Search-Shape Verdict",
        "",
        f"Ruleset: `{report['ruleset_id']}`",
        f"Source revision: `{report['source_revision']}`",
        f"Games: `{len(report['seeds'])}` matched seeds",
        f"Scientific eligibility: `{report['scientific_eligibility']}`",
        "",
        "## Varied settings",
        "",
        *varied_lines,
        "",
        "## Score",
        "",
        f"- Baseline mean seat score: `{report['baseline_mean_seat_score']:.4f}`",
        f"- Candidate mean seat score: `{report['candidate_mean_seat_score']:.4f}`",
        f"- Paired delta: `{stats['mean']:+.4f}`",
        f"- 95% t-CI: `[{stats['t_ci_low']:+.4f}, {stats['t_ci_high']:+.4f}]`",
        f"- Verdict: `{report['score_verdict']}`",
        f"- Proceed to high-budget confirmation: `{report['proceed_to_high_budget']}`",
        "",
        "## Cost",
        "",
        f"- Mean decision seconds: `{timing['baseline_mean_decision_seconds']:.4f}` -> "
        f"`{timing['candidate_mean_decision_seconds']:.4f}`",
        f"- Whole-arm wall seconds: `{timing['baseline_wall_seconds']:.1f}` -> "
        f"`{timing['candidate_wall_seconds']:.1f}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--source-revision", default="")
    parser.add_argument(
        "--varied-key",
        action="append",
        default=None,
        help="Search key allowed (and required, jointly) to differ between "
        "arms; repeatable. Default: determinizations.",
    )
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
