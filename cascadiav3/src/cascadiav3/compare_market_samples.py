"""Validate matched market-decision sample-count ablations.

The verdict is the preregistered paired score/speedup gate (t-CI lower bound
>= NONINFERIORITY_MARGIN and mean-decision speedup >= MIN_SPEEDUP). Unlike
the exact-K1 comparator, trace identity is NOT a validity requirement:
market_decision_samples changes interior simulation values at any ply whose
search horizon reaches a refresh chance node (42/100 seeds diverged before
their first root exposure in the 2026-07-10 CUDA gate; the earlier MPS
screens showed the same). Trace divergence is classified descriptively.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .torch_benchmark_stats import paired_delta_stats

RULESET_ID = "cascadia_research_aaaaa_4p_card_a_no_habitat_bonus_rules_2026_07_16"
DEFAULT_BASELINE_SAMPLES = 8
DEFAULT_CANDIDATE_SAMPLES = 4
NONINFERIORITY_MARGIN = -0.25
MIN_SPEEDUP = 1.15


def _load_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "pass":
        raise ValueError(f"report is not passing: {path}")
    if report.get("ruleset_id") != RULESET_ID:
        raise ValueError(f"ruleset mismatch in {path}")
    if report.get("control", {}).get("kind") != "none":
        raise ValueError(f"market-sample comparison requires control=none in {path}")
    return report


def _search_without_market_samples(report: dict[str, Any]) -> dict[str, Any]:
    search = dict(report.get("search", {}))
    search.pop("market_decision_samples", None)
    return search


def _scores_by_seed(report: dict[str, Any], label: str) -> dict[int, float]:
    scores = {
        int(row["seed"]): float(row["mean_score_per_seat"])
        for row in report.get("candidate_per_seed", [])
    }
    if len(scores) != len(report.get("seeds", [])):
        raise ValueError(f"duplicate or incomplete per-seed scores in {label}")
    return scores


def _load_decisions(path: Path) -> dict[int, dict[int, dict[str, Any]]]:
    by_seed: dict[int, dict[int, dict[str, Any]]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw:
            continue
        row = json.loads(raw)
        if row.get("type") != "gumbel_decision":
            continue
        if row.get("ruleset_id") != RULESET_ID:
            raise ValueError(f"decision trace ruleset mismatch in {path}")
        seed = int(row["seed"])
        ply = int(row["ply"])
        seed_rows = by_seed.setdefault(seed, {})
        if ply in seed_rows:
            raise ValueError(f"duplicate decision for seed {seed} ply {ply} in {path}")
        seed_rows[ply] = row
    return by_seed


def _validate_causal_trace(
    seeds: list[int],
    baseline_path: Path,
    candidate_path: Path,
) -> dict[str, Any]:
    baseline = _load_decisions(baseline_path)
    candidate = _load_decisions(candidate_path)
    if baseline.keys() != candidate.keys() or set(seeds) != baseline.keys():
        raise ValueError("decision trace seed coverage mismatch")

    first_exposure_by_seed: dict[str, int | None] = {}
    first_divergence_by_seed: dict[str, int | None] = {}
    causally_changed_seeds = 0
    pre_exposure_divergent_seeds = 0
    for seed in seeds:
        if set(baseline[seed]) != set(range(80)) or set(candidate[seed]) != set(range(80)):
            raise ValueError(f"decision trace for seed {seed} must contain plies 0..79")
        first_exposure = next(
            (
                ply
                for ply in range(80)
                if baseline[seed][ply].get("free_three_of_a_kind_choice")
                in {"accept", "decline"}
                or candidate[seed][ply].get("free_three_of_a_kind_choice")
                in {"accept", "decline"}
            ),
            None,
        )
        first_exposure_by_seed[str(seed)] = first_exposure
        first_divergence = None
        for ply in range(80):
            left = baseline[seed][ply]
            right = candidate[seed][ply]
            same = left.get("chosen_action_id") == right.get("chosen_action_id") and left.get(
                "free_three_of_a_kind_choice"
            ) == right.get("free_three_of_a_kind_choice")
            if same:
                continue
            first_divergence = ply
            # Pre-exposure divergence is expected here, unlike the exact-K1
            # comparator: market_decision_samples changes interior simulation
            # values at any ply whose search horizon reaches a refresh chance
            # node, so trace identity holds only until the first ply where a
            # simulated refresh alters an argmax. Divergences are classified
            # descriptively; the verdict is the paired score/speedup gate.
            if first_exposure is None or ply < first_exposure:
                pre_exposure_divergent_seeds += 1
            else:
                causally_changed_seeds += 1
            break
        first_divergence_by_seed[str(seed)] = first_divergence

    baseline_available = [
        row
        for seed in seeds
        for row in baseline[seed].values()
        if row.get("free_three_of_a_kind_choice") in {"accept", "decline"}
    ]
    candidate_available = [
        row
        for seed in seeds
        for row in candidate[seed].values()
        if row.get("free_three_of_a_kind_choice") in {"accept", "decline"}
    ]
    if not baseline_available or not candidate_available:
        raise ValueError("market-sample comparison observed no optional-refresh opportunity")
    baseline_seconds = sum(float(row.get("decision_seconds", 0.0)) for row in baseline_available)
    candidate_seconds = sum(float(row.get("decision_seconds", 0.0)) for row in candidate_available)
    if baseline_seconds <= 0.0 or candidate_seconds <= 0.0:
        raise ValueError("available-decision timing must be positive")
    baseline_mean_seconds = (
        baseline_seconds / len(baseline_available) if baseline_available else None
    )
    candidate_mean_seconds = (
        candidate_seconds / len(candidate_available) if candidate_available else None
    )
    return {
        "first_exposure_by_seed": first_exposure_by_seed,
        "first_divergence_by_seed": first_divergence_by_seed,
        "causally_changed_seeds": causally_changed_seeds,
        "pre_exposure_divergent_seeds": pre_exposure_divergent_seeds,
        "baseline_available_decisions": len(baseline_available),
        "candidate_available_decisions": len(candidate_available),
        "baseline_available_seconds": baseline_seconds,
        "candidate_available_seconds": candidate_seconds,
        "baseline_available_mean_seconds": baseline_mean_seconds,
        "candidate_available_mean_seconds": candidate_mean_seconds,
        "available_total_seconds_ratio": (
            baseline_seconds / candidate_seconds if candidate_seconds > 0.0 else None
        ),
        "available_decision_speedup": (
            baseline_mean_seconds / candidate_mean_seconds
            if baseline_mean_seconds is not None
            and candidate_mean_seconds is not None
            and candidate_mean_seconds > 0.0
            else None
        ),
    }


def build_comparison(
    baseline_path: Path,
    candidate_path: Path,
    baseline_decisions_path: Path,
    candidate_decisions_path: Path,
    source_revision: str | None = None,
    baseline_samples: int = DEFAULT_BASELINE_SAMPLES,
    candidate_samples: int = DEFAULT_CANDIDATE_SAMPLES,
) -> dict[str, Any]:
    if baseline_samples <= 0 or candidate_samples <= 0:
        raise ValueError("market sample counts must be positive")
    if baseline_samples == candidate_samples:
        raise ValueError("market sample counts must differ")
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
    if _search_without_market_samples(baseline) != _search_without_market_samples(candidate):
        raise ValueError("search settings differ beyond market_decision_samples")
    if baseline.get("search", {}).get("market_decision_samples") != baseline_samples:
        raise ValueError(f"baseline report must use {baseline_samples} market samples")
    if candidate.get("search", {}).get("market_decision_samples") != candidate_samples:
        raise ValueError(f"candidate report must use {candidate_samples} market samples")
    baseline_manifest_path = str(baseline.get("manifest", ""))
    candidate_manifest_path = str(candidate.get("manifest", ""))
    if not baseline_manifest_path or baseline_manifest_path != candidate_manifest_path:
        raise ValueError("model manifest identity mismatch")
    baseline_manifest = Path(baseline_manifest_path).name

    seeds = [int(seed) for seed in baseline["seeds"]]
    if len(seeds) < 2:
        raise ValueError("market-sample comparison requires at least two paired seeds")
    trace = _validate_causal_trace(seeds, baseline_decisions_path, candidate_decisions_path)
    for report, expected, label in (
        (baseline, baseline_samples, "baseline"),
        (candidate, candidate_samples, "candidate"),
    ):
        observed = report.get("market_decisions", {}).get(
            "mean_chance_samples_when_available"
        )
        if observed is None or abs(float(observed) - expected) > 1.0e-9:
            raise ValueError(f"{label} chance-sample telemetry mismatch")

    baseline_scores = _scores_by_seed(baseline, "baseline")
    candidate_scores = _scores_by_seed(candidate, "candidate")
    if baseline_scores.keys() != candidate_scores.keys() or set(seeds) != baseline_scores.keys():
        raise ValueError("per-seed score coverage mismatch")
    deltas = [candidate_scores[seed] - baseline_scores[seed] for seed in seeds]
    stats = paired_delta_stats(deltas)
    t_ci_low = stats.get("t_ci_low")
    noninferior = t_ci_low is not None and t_ci_low >= NONINFERIORITY_MARGIN

    baseline_summary = baseline["strategies"]["gumbel-search"]
    candidate_summary = candidate["strategies"]["gumbel-search"]
    baseline_decision_seconds = float(baseline_summary["mean_total_decision_seconds"])
    candidate_decision_seconds = float(candidate_summary["mean_total_decision_seconds"])
    speedup = (
        baseline_decision_seconds / candidate_decision_seconds
        if candidate_decision_seconds > 0.0
        else None
    )
    game_count = len(seeds)
    promotion_scale = game_count >= 100
    performance_gate_pass = bool(
        promotion_scale and noninferior and speedup is not None and speedup >= MIN_SPEEDUP
    )
    return {
        "status": "pass",
        "scientific_eligibility": (
            "promotion_scale_paired_gate" if promotion_scale else "engineering_smoke_only"
        ),
        "ruleset_id": RULESET_ID,
        "source_revision": baseline_revision,
        "manifest_name": baseline_manifest,
        "seeds": seeds,
        "search": _search_without_market_samples(baseline)
        | {
            "baseline_market_decision_samples": baseline_samples,
            "candidate_market_decision_samples": candidate_samples,
        },
        "baseline_mean_seat_score": float(baseline_summary["mean_seat_score"]),
        "candidate_mean_seat_score": float(candidate_summary["mean_seat_score"]),
        "paired_score_deltas": [
            {"seed": seed, "delta": delta} for seed, delta in zip(seeds, deltas, strict=True)
        ],
        "paired_delta_stats": stats,
        "noninferiority_margin": NONINFERIORITY_MARGIN,
        "score_noninferior": noninferior,
        "minimum_speedup": MIN_SPEEDUP,
        "performance_gate_pass": performance_gate_pass,
        "trace": trace,
        "simulations": {
            "baseline_total": baseline["market_decisions"][
                "total_simulations_including_market_decision"
            ],
            "candidate_total": candidate["market_decisions"][
                "total_simulations_including_market_decision"
            ],
            "baseline_market_overhead": baseline["market_decisions"][
                "market_decision_simulation_overhead"
            ],
            "candidate_market_overhead": candidate["market_decisions"][
                "market_decision_simulation_overhead"
            ],
            "baseline_market_overhead_per_opportunity": (
                baseline["market_decisions"]["market_decision_simulation_overhead"]
                / trace["baseline_available_decisions"]
                if trace["baseline_available_decisions"]
                else None
            ),
            "candidate_market_overhead_per_opportunity": (
                candidate["market_decisions"]["market_decision_simulation_overhead"]
                / trace["candidate_available_decisions"]
                if trace["candidate_available_decisions"]
                else None
            ),
        },
        "timing": {
            "baseline_mean_decision_seconds": baseline_decision_seconds,
            "candidate_mean_decision_seconds": candidate_decision_seconds,
            "mean_decision_speedup": speedup,
            "baseline_p95_decision_seconds": baseline["candidate_decision_seconds_p95"],
            "candidate_p95_decision_seconds": candidate["candidate_decision_seconds_p95"],
            "baseline_wall_seconds": baseline["candidate_wall_seconds"],
            "candidate_wall_seconds": candidate["candidate_wall_seconds"],
        },
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    stats = report["paired_delta_stats"]
    timing = report["timing"]
    trace = report["trace"]
    sims = report["simulations"]
    lines = [
        "# Market-Decision Sample-Count Verdict",
        "",
        f"Ruleset: `{report['ruleset_id']}`",
        f"Source revision: `{report['source_revision']}`",
        f"Games: `{len(report['seeds'])}` matched seeds",
        f"Scientific eligibility: `{report['scientific_eligibility']}`",
        "",
        "## Score",
        "",
        f"- {report['search']['baseline_market_decision_samples']}-sample mean seat score: "
        f"`{report['baseline_mean_seat_score']:.4f}`",
        f"- {report['search']['candidate_market_decision_samples']}-sample mean seat score: "
        f"`{report['candidate_mean_seat_score']:.4f}`",
        f"- Paired delta: `{stats['mean']:+.4f}`",
        f"- 95% t-CI: `[{stats['t_ci_low']:+.4f}, {stats['t_ci_high']:+.4f}]`",
        f"- Noninferior at `{report['noninferiority_margin']:+.2f}`: "
        f"`{report['score_noninferior']}`",
        "",
        "## Trace classification (descriptive — the verdict is score+speedup)",
        "",
        f"- Identical traces: "
        f"`{len(report['seeds']) - trace['causally_changed_seeds'] - trace['pre_exposure_divergent_seeds']}`",
        f"- Diverged at/after first refresh exposure: `{trace['causally_changed_seeds']}`",
        f"- Diverged before first refresh exposure (expected: sample count "
        f"reaches every ply through simulated refresh nodes): "
        f"`{trace['pre_exposure_divergent_seeds']}`",
        "",
        "## Cost",
        "",
        f"- Mean decision seconds: `{timing['baseline_mean_decision_seconds']:.4f}` -> "
        f"`{timing['candidate_mean_decision_seconds']:.4f}` "
        f"(`{timing['mean_decision_speedup']:.3f}x`)",
        f"- Mean available-decision seconds: `{trace['baseline_available_mean_seconds']:.4f}` -> "
        f"`{trace['candidate_available_mean_seconds']:.4f}` "
        f"(`{trace['available_decision_speedup']:.3f}x`)",
        f"- Available-decision counts: `{trace['baseline_available_decisions']}` -> "
        f"`{trace['candidate_available_decisions']}`",
        f"- Total market-decision seconds: `{trace['baseline_available_seconds']:.4f}` -> "
        f"`{trace['candidate_available_seconds']:.4f}`",
        f"- Market simulation overhead per opportunity: "
        f"`{sims['baseline_market_overhead_per_opportunity']:.2f}` -> "
        f"`{sims['candidate_market_overhead_per_opportunity']:.2f}`",
        f"- Total market simulation overhead: `{sims['baseline_market_overhead']}` -> "
        f"`{sims['candidate_market_overhead']}`",
        f"- Performance gate pass: `{report['performance_gate_pass']}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline-decisions", required=True)
    parser.add_argument("--candidate-decisions", required=True)
    parser.add_argument("--source-revision", default="")
    parser.add_argument("--baseline-samples", type=int, default=DEFAULT_BASELINE_SAMPLES)
    parser.add_argument("--candidate-samples", type=int, default=DEFAULT_CANDIDATE_SAMPLES)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    args = parser.parse_args()
    report = build_comparison(
        Path(args.baseline),
        Path(args.candidate),
        Path(args.baseline_decisions),
        Path(args.candidate_decisions),
        args.source_revision or None,
        args.baseline_samples,
        args.candidate_samples,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, Path(args.summary_out))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
