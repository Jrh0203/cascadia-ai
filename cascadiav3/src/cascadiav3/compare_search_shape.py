"""Paired verdict for search-shape ablations (worlds/determinizations).

Determinizations are cycled inside the fixed simulation budget, so changing
the world count alters evaluations at every ply — like market sample counts
and unlike exact-K1, trace identity is not a validity condition and is not
checked. The verdict is the campaign-standard paired per-seed score delta
with a 95% t-CI over >=100 matched seeds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .torch_benchmark_stats import paired_delta_stats

RULESET_ID = "cascadia_research_aaaaa_4p_card_a_no_habitat_bonus_rules_2026_07_09"
VARIED_KEY = "determinizations"


def _load_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "pass":
        raise ValueError(f"report is not passing: {path}")
    if report.get("ruleset_id") != RULESET_ID:
        raise ValueError(f"ruleset mismatch in {path}")
    if report.get("control", {}).get("kind") != "none":
        raise ValueError(f"search-shape comparison requires control=none in {path}")
    return report


def _search_without_varied(report: dict[str, Any]) -> dict[str, Any]:
    search = dict(report.get("search", {}))
    search.pop(VARIED_KEY, None)
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
) -> dict[str, Any]:
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
    if _search_without_varied(baseline) != _search_without_varied(candidate):
        raise ValueError(f"search settings differ beyond {VARIED_KEY}")
    baseline_worlds = baseline.get("search", {}).get(VARIED_KEY)
    candidate_worlds = candidate.get("search", {}).get(VARIED_KEY)
    if not baseline_worlds or not candidate_worlds or baseline_worlds == candidate_worlds:
        raise ValueError("arms must use two distinct positive world counts")
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
        "search": _search_without_varied(baseline)
        | {
            "baseline_determinizations": baseline_worlds,
            "candidate_determinizations": candidate_worlds,
        },
        "baseline_mean_seat_score": float(baseline_summary["mean_seat_score"]),
        "candidate_mean_seat_score": float(candidate_summary["mean_seat_score"]),
        "paired_score_deltas": [
            {"seed": seed, "delta": delta} for seed, delta in zip(seeds, deltas, strict=True)
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
    lines = [
        "# Search-Shape (Worlds) Verdict",
        "",
        f"Ruleset: `{report['ruleset_id']}`",
        f"Source revision: `{report['source_revision']}`",
        f"Games: `{len(report['seeds'])}` matched seeds",
        f"Scientific eligibility: `{report['scientific_eligibility']}`",
        "",
        "## Score",
        "",
        f"- det{report['search']['baseline_determinizations']} mean seat score: "
        f"`{report['baseline_mean_seat_score']:.4f}`",
        f"- det{report['search']['candidate_determinizations']} mean seat score: "
        f"`{report['candidate_mean_seat_score']:.4f}`",
        f"- Paired delta: `{stats['mean']:+.4f}`",
        f"- 95% t-CI: `[{stats['t_ci_low']:+.4f}, {stats['t_ci_high']:+.4f}]`",
        f"- Verdict: `{report['score_verdict']}`",
        f"- Proceed to high-budget confirmation: `{report['proceed_to_high_budget']}`",
        "",
        "## Cost (worlds cycle inside the fixed simulation budget)",
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
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    args = parser.parse_args()
    report = build_comparison(
        Path(args.baseline),
        Path(args.candidate),
        args.source_revision or None,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, Path(args.summary_out))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
