"""Validate and compare matched Gumbel baseline versus exact-K1 reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .torch_benchmark_stats import paired_delta_stats

RULESET_ID = "cascadia_research_aaaaa_4p_card_a_no_habitat_bonus_rules_2026_07_09"


def _load(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "pass":
        raise ValueError(f"report is not passing: {path}")
    if report.get("ruleset_id") != RULESET_ID:
        raise ValueError(f"ruleset mismatch in {path}")
    if report.get("control", {}).get("kind") != "none":
        raise ValueError(f"exact-endgame comparison requires control=none in {path}")
    return report


def _search_without_exact(report: dict[str, Any]) -> dict[str, Any]:
    search = dict(report.get("search", {}))
    search.pop("exact_endgame_turns", None)
    return search


def _scores_by_seed(report: dict[str, Any], label: str) -> dict[int, float]:
    scores = {
        int(row["seed"]): float(row["mean_score_per_seat"])
        for row in report.get("candidate_per_seed", [])
    }
    if len(scores) != len(report.get("seeds", [])):
        raise ValueError(f"duplicate or incomplete per-seed scores in {label}")
    return scores


def _seat_scores_by_seed(report: dict[str, Any], label: str) -> dict[int, list[float]]:
    scores = {
        int(row["seed"]): [float(value) for value in row["seat_scores"]]
        for row in report.get("candidate_per_seed", [])
    }
    if len(scores) != len(report.get("seeds", [])) or any(
        len(values) != 4 for values in scores.values()
    ):
        raise ValueError(f"invalid per-seat scores in {label}")
    return scores


def _score_verdict(stats: dict[str, Any]) -> str:
    low = stats.get("t_ci_low")
    high = stats.get("t_ci_high")
    if low is not None and low > 0.0:
        return "ci_positive"
    if high is not None and high < 0.0:
        return "ci_negative"
    return "inconclusive"


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


def _first_pre_k1_divergent_ply(
    baseline_rows: dict[int, dict[str, Any]],
    exact_rows: dict[int, dict[str, Any]],
) -> int | None:
    for ply in range(76):
        left = baseline_rows[ply]
        right = exact_rows[ply]
        if left.get("chosen_action_id") != right.get("chosen_action_id") or left.get(
            "free_three_of_a_kind_choice"
        ) != right.get("free_three_of_a_kind_choice"):
            return ply
    return None


def _validate_causal_trace(
    seeds: list[int],
    baseline_path: Path,
    exact_path: Path,
    declared_divergent_seeds: frozenset[int] = frozenset(),
) -> dict[str, Any]:
    baseline = _load_decisions(baseline_path)
    exact = _load_decisions(exact_path)
    if baseline.keys() != exact.keys() or set(seeds) != baseline.keys():
        raise ValueError("decision trace seed coverage mismatch")
    final_action_changes = 0
    final_refresh_changes = 0
    baseline_final_seconds = 0.0
    exact_final_seconds = 0.0
    declared_first_divergent_ply: dict[int, int] = {}
    for seed in seeds:
        if set(baseline[seed]) != set(range(80)) or set(exact[seed]) != set(range(80)):
            raise ValueError(f"decision trace for seed {seed} must contain plies 0..79")
        excluded = seed in declared_divergent_seeds
        if excluded:
            first = _first_pre_k1_divergent_ply(baseline[seed], exact[seed])
            if first is None:
                raise ValueError(
                    f"declared divergent seed {seed} has an identical pre-K1 trace; "
                    "refusing the exclusion"
                )
            declared_first_divergent_ply[seed] = first
        else:
            first = _first_pre_k1_divergent_ply(baseline[seed], exact[seed])
            if first is not None:
                raise ValueError(f"pre-K1 action trace diverges at seed {seed} ply {first}")
        for ply in range(76):
            left = baseline[seed][ply]
            right = exact[seed][ply]
            if bool(left.get("exact_endgame")) or bool(right.get("exact_endgame")):
                raise ValueError(f"exact telemetry activated before K1 at seed {seed} ply {ply}")
        for ply in range(76, 80):
            baseline_row = baseline[seed][ply]
            if bool(baseline_row.get("exact_endgame")):
                raise ValueError(f"baseline exact telemetry activated at seed {seed} ply {ply}")
            exact_row = exact[seed][ply]
            if not bool(exact_row.get("exact_endgame")):
                raise ValueError(f"exact telemetry missing at seed {seed} ply {ply}")
            if int(exact_row.get("total_simulations_run", -1)) != 0:
                raise ValueError(f"exact decision ran simulations at seed {seed} ply {ply}")
            if excluded:
                continue
            final_action_changes += int(
                baseline_row.get("chosen_action_id") != exact_row.get("chosen_action_id")
            )
            final_refresh_changes += int(
                baseline_row.get("free_three_of_a_kind_choice")
                != exact_row.get("free_three_of_a_kind_choice")
            )
            baseline_final_seconds += float(baseline_row.get("decision_seconds", 0.0))
            exact_final_seconds += float(exact_row.get("decision_seconds", 0.0))
    return {
        "action_changes": final_action_changes,
        "refresh_choice_changes": final_refresh_changes,
        "baseline_total_seconds": baseline_final_seconds,
        "exact_total_seconds": exact_final_seconds,
        "speedup": (
            baseline_final_seconds / exact_final_seconds if exact_final_seconds > 0.0 else None
        ),
        "declared_first_divergent_ply": declared_first_divergent_ply,
    }


def build_comparison(
    baseline_path: Path,
    exact_path: Path,
    baseline_decisions_path: Path,
    exact_decisions_path: Path,
    source_revision: str | None = None,
    declared_divergent_seeds: list[int] | None = None,
    exclusion_ruling: str | None = None,
) -> dict[str, Any]:
    excluded = frozenset(declared_divergent_seeds or [])
    if excluded and not (exclusion_ruling or "").strip():
        raise ValueError(
            "declared divergent-seed exclusions require an explicit exclusion ruling"
        )
    if not excluded and (exclusion_ruling or "").strip():
        raise ValueError("an exclusion ruling was given but no seeds were declared")
    baseline = _load(baseline_path)
    exact = _load(exact_path)

    baseline_revision = baseline.get("source_revision")
    exact_revision = exact.get("source_revision")
    if not baseline_revision or not exact_revision:
        raise ValueError("both reports must record a source revision")
    if baseline_revision != exact_revision:
        raise ValueError("source revision mismatch between baseline and exact reports")
    if source_revision is not None and baseline_revision != source_revision:
        raise ValueError("reports do not match the required source revision")
    if baseline.get("seeds") != exact.get("seeds"):
        raise ValueError("seed mismatch between baseline and exact reports")
    if _search_without_exact(baseline) != _search_without_exact(exact):
        raise ValueError("search settings differ beyond exact_endgame_turns")
    if baseline.get("search", {}).get("exact_endgame_turns", 0) != 0:
        raise ValueError("baseline report must use exact_endgame_turns=0")
    if exact.get("search", {}).get("exact_endgame_turns") != 1:
        raise ValueError("exact report must use exact_endgame_turns=1")
    baseline_manifest = Path(str(baseline.get("manifest", ""))).name
    exact_manifest = Path(str(exact.get("manifest", ""))).name
    if not baseline_manifest or not exact_manifest or baseline_manifest != exact_manifest:
        raise ValueError("model manifest identity mismatch")

    seeds = [int(seed) for seed in baseline["seeds"]]
    if not excluded <= set(seeds):
        raise ValueError("declared divergent seeds are not all present in the reports")
    retained = [seed for seed in seeds if seed not in excluded]
    if len(retained) < 2:
        raise ValueError("exact-endgame comparison requires at least two paired seeds")
    exact_frontier = _validate_causal_trace(
        seeds, baseline_decisions_path, exact_decisions_path, excluded
    )
    baseline_exact_count = int(
        baseline.get("market_decisions", {}).get("exact_endgame_decisions", -1)
    )
    exact_count = int(exact.get("market_decisions", {}).get("exact_endgame_decisions", -1))
    if baseline_exact_count != 0:
        raise ValueError("baseline telemetry must contain zero exact decisions")
    expected_exact_count = 4 * len(seeds)
    if exact_count != expected_exact_count:
        raise ValueError(
            f"exact telemetry contains {exact_count} decisions; expected {expected_exact_count}"
        )

    baseline_scores = _scores_by_seed(baseline, "baseline")
    exact_scores = _scores_by_seed(exact, "exact")
    if baseline_scores.keys() != exact_scores.keys() or set(seeds) != baseline_scores.keys():
        raise ValueError("per-seed coverage mismatch")
    deltas = [exact_scores[seed] - baseline_scores[seed] for seed in retained]
    baseline_seats = _seat_scores_by_seed(baseline, "baseline")
    exact_seats = _seat_scores_by_seed(exact, "exact")
    seat0_deltas = [exact_seats[seed][0] - baseline_seats[seed][0] for seed in retained]
    if any(delta < 0.0 for delta in seat0_deltas):
        raise ValueError("exact K1 reduced seat 0 despite an identical pre-K1 state")
    stats = paired_delta_stats(deltas)

    baseline_summary = baseline["strategies"]["gumbel-search"]
    exact_summary = exact["strategies"]["gumbel-search"]
    baseline_decision_seconds = float(baseline_summary["mean_total_decision_seconds"])
    exact_decision_seconds = float(exact_summary["mean_total_decision_seconds"])
    baseline_wall = float(baseline["candidate_wall_seconds"])
    exact_wall = float(exact["candidate_wall_seconds"])
    game_count = len(seeds)
    if game_count >= 100:
        eligibility = (
            "promotion_scale_paired_gate_with_declared_exclusions"
            if excluded
            else "promotion_scale_paired_gate"
        )
    else:
        eligibility = "engineering_smoke_only"
    return {
        "status": "pass",
        "scientific_eligibility": eligibility,
        "ruleset_id": RULESET_ID,
        "source_revision": baseline_revision,
        "manifest_name": baseline_manifest,
        "seeds": seeds,
        "declared_exclusions": {
            "seeds": sorted(excluded),
            "ruling": (exclusion_ruling or "").strip() or None,
            "first_divergent_ply": {
                str(seed): ply
                for seed, ply in sorted(
                    exact_frontier["declared_first_divergent_ply"].items()
                )
            },
        },
        "retained_seed_count": len(retained),
        "search": baseline["search"] | {"candidate_exact_endgame_turns": 1},
        "baseline_mean_seat_score": float(baseline_summary["mean_seat_score"]),
        "exact_mean_seat_score": float(exact_summary["mean_seat_score"]),
        "paired_score_deltas": [
            {"seed": seed, "delta": delta}
            for seed, delta in zip(retained, deltas, strict=True)
        ],
        "paired_delta_stats": stats,
        "score_verdict": _score_verdict(stats),
        "seat0_exact_score_deltas": seat0_deltas,
        "seat0_exact_score_mean_delta": sum(seat0_deltas) / len(seat0_deltas),
        "pre_k1_action_trace_match": True,
        "exact_decisions": exact_count,
        "exact_frontier": exact_frontier,
        "timing": {
            "baseline_mean_decision_seconds": baseline_decision_seconds,
            "exact_mean_decision_seconds": exact_decision_seconds,
            "mean_decision_speedup": (
                baseline_decision_seconds / exact_decision_seconds
                if exact_decision_seconds > 0.0
                else None
            ),
            "baseline_p50_decision_seconds": baseline["candidate_decision_seconds_p50"],
            "exact_p50_decision_seconds": exact["candidate_decision_seconds_p50"],
            "baseline_p95_decision_seconds": baseline["candidate_decision_seconds_p95"],
            "exact_p95_decision_seconds": exact["candidate_decision_seconds_p95"],
            "baseline_wall_seconds": baseline_wall,
            "exact_wall_seconds": exact_wall,
            "wall_speedup": baseline_wall / exact_wall if exact_wall > 0.0 else None,
        },
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    stats = report["paired_delta_stats"]
    timing = report["timing"]
    frontier = report["exact_frontier"]
    lines = [
        "# Exact Final-Personal-Turn Verdict",
        "",
        f"Ruleset: `{report['ruleset_id']}`",
        f"Source revision: `{report['source_revision']}`",
        f"Games: `{len(report['seeds'])}` seeds run, "
        f"`{report['retained_seed_count']}` retained for the paired verdict",
        f"Scientific eligibility: `{report['scientific_eligibility']}`",
        "",
        "## Score",
        "",
        f"- Baseline mean seat score: `{report['baseline_mean_seat_score']:.4f}`",
        f"- Exact-K1 mean seat score: `{report['exact_mean_seat_score']:.4f}`",
        f"- Paired delta: `{stats['mean']:+.4f}`",
        f"- 95% t-CI: `[{stats['t_ci_low']:+.4f}, {stats['t_ci_high']:+.4f}]`",
        f"- Verdict: `{report['score_verdict']}`",
        f"- Seat-0 exact-score mean delta: `{report['seat0_exact_score_mean_delta']:+.4f}`",
        f"- Pre-K1 action traces identical (retained seeds): "
        f"`{report['pre_k1_action_trace_match']}`",
    ]
    exclusions = report.get("declared_exclusions", {})
    if exclusions.get("seeds"):
        lines += [
            "",
            "## Declared exclusions",
            "",
            f"- Excluded seeds: `{exclusions['seeds']}` "
            f"(first pre-K1 divergent ply: `{exclusions['first_divergent_ply']}`)",
            f"- Ruling: {exclusions['ruling']}",
        ]
    lines += [
        "",
        "## Cost",
        "",
        f"- Mean decision seconds: `{timing['baseline_mean_decision_seconds']:.4f}` -> "
        f"`{timing['exact_mean_decision_seconds']:.4f}` "
        f"(`{timing['mean_decision_speedup']:.3f}x`)",
        f"- P95 decision seconds: `{timing['baseline_p95_decision_seconds']:.4f}` -> "
        f"`{timing['exact_p95_decision_seconds']:.4f}`",
        f"- Whole-arm wall seconds: `{timing['baseline_wall_seconds']:.1f}` -> "
        f"`{timing['exact_wall_seconds']:.1f}` (`{timing['wall_speedup']:.3f}x`)",
        f"- Exact decisions verified: `{report['exact_decisions']}`",
        f"- Final action changes: `{frontier['action_changes']}` / "
        f"`{report['exact_decisions']}`",
        f"- Exact-frontier seconds: `{frontier['baseline_total_seconds']:.4f}` -> "
        f"`{frontier['exact_total_seconds']:.4f}` (`{frontier['speedup']:.3f}x`)",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--exact", required=True)
    parser.add_argument("--baseline-decisions", required=True)
    parser.add_argument("--exact-decisions", required=True)
    parser.add_argument("--source-revision", default="")
    parser.add_argument(
        "--declared-divergent-seed",
        action="append",
        type=int,
        default=[],
        help=(
            "Seed to exclude from the paired verdict under an explicit ruling. The"
            " seed must actually diverge pre-K1; a causally clean seed is refused."
        ),
    )
    parser.add_argument(
        "--exclusion-ruling",
        default="",
        help="Required with any declared exclusion: who ruled, when, and why.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    args = parser.parse_args()
    report = build_comparison(
        Path(args.baseline),
        Path(args.exact),
        Path(args.baseline_decisions),
        Path(args.exact_decisions),
        args.source_revision or None,
        declared_divergent_seeds=args.declared_divergent_seed,
        exclusion_ruling=args.exclusion_ruling,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, Path(args.summary_out))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
