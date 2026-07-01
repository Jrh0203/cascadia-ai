"""Compare interactive game-pilot reports by paired seed.

This utility exists so follow-up prefilter-only runs can reuse an already
computed full-search baseline on the same seed set. It intentionally depends
only on JSON reports, not Torch.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any


def _score_mean(game: dict[str, Any]) -> float:
    return mean(float(score["total"]) for score in game["scores"])


def _strategy_games(report: dict[str, Any], strategy: str) -> dict[int, dict[str, Any]]:
    games: dict[int, dict[str, Any]] = {}
    for game in report["games"]:
        if game.get("strategy") != strategy:
            continue
        seed = int(game["seed"])
        if seed in games:
            raise ValueError(f"duplicate {strategy} game for seed {seed}")
        games[seed] = game
    return games


def _maybe_strategy_summary(report: dict[str, Any], strategy: str) -> dict[str, Any] | None:
    return report.get("strategies", {}).get(strategy)


def compare_reports(
    *,
    candidate_report: dict[str, Any],
    baseline_report: dict[str, Any],
    candidate_strategy: str,
    baseline_strategy: str,
) -> dict[str, Any]:
    candidate_games = _strategy_games(candidate_report, candidate_strategy)
    baseline_games = _strategy_games(baseline_report, baseline_strategy)
    common_seeds = sorted(set(candidate_games) & set(baseline_games))
    if not common_seeds:
        raise ValueError("reports share no comparable seeds")

    rows = []
    for seed in common_seeds:
        candidate_mean = _score_mean(candidate_games[seed])
        baseline_mean = _score_mean(baseline_games[seed])
        rows.append(
            {
                "seed": seed,
                "candidate_mean_score_per_seat": candidate_mean,
                "baseline_mean_score_per_seat": baseline_mean,
                "delta_candidate_minus_baseline": candidate_mean - baseline_mean,
            }
        )
    deltas = [row["delta_candidate_minus_baseline"] for row in rows]
    candidate_summary = _maybe_strategy_summary(candidate_report, candidate_strategy)
    baseline_summary = _maybe_strategy_summary(baseline_report, baseline_strategy)
    candidate_seconds = (
        float(candidate_summary["mean_total_decision_seconds"])
        if candidate_summary is not None
        else None
    )
    baseline_seconds = (
        float(baseline_summary["mean_total_decision_seconds"])
        if baseline_summary is not None
        else None
    )
    speedup = (
        baseline_seconds / candidate_seconds
        if candidate_seconds is not None and baseline_seconds is not None and candidate_seconds > 0.0
        else None
    )
    time_reduction = (
        1.0 - (candidate_seconds / baseline_seconds)
        if candidate_seconds is not None and baseline_seconds is not None and baseline_seconds > 0.0
        else None
    )
    return {
        "status": "pass",
        "scientific_eligibility": "interactive_prefilter_game_report_comparison",
        "candidate_experiment_id": candidate_report.get("experiment_id"),
        "baseline_experiment_id": baseline_report.get("experiment_id"),
        "candidate_strategy": candidate_strategy,
        "baseline_strategy": baseline_strategy,
        "paired_seed_count": len(rows),
        "candidate_seed_count": len(candidate_games),
        "baseline_seed_count": len(baseline_games),
        "paired_score_deltas": rows,
        "candidate_mean_score_per_seat": mean(row["candidate_mean_score_per_seat"] for row in rows),
        "baseline_mean_score_per_seat": mean(row["baseline_mean_score_per_seat"] for row in rows),
        "mean_delta_candidate_minus_baseline": mean(deltas),
        "median_delta_candidate_minus_baseline": median(deltas),
        "min_delta_candidate_minus_baseline": min(deltas),
        "max_delta_candidate_minus_baseline": max(deltas),
        "stdev_delta_candidate_minus_baseline": stdev(deltas) if len(deltas) > 1 else 0.0,
        "sem_delta_candidate_minus_baseline": (
            stdev(deltas) / math.sqrt(len(deltas))
            if len(deltas) > 1
            else 0.0
        ),
        "candidate_mean_total_decision_seconds": candidate_seconds,
        "baseline_mean_total_decision_seconds": baseline_seconds,
        "speedup_factor": speedup,
        "time_reduction": time_reduction,
    }


def write_markdown_summary(comparison: dict[str, Any], path: Path) -> None:
    lines = [
        "# CRT Prefilter Game Report Comparison",
        "",
        f"Candidate: `{comparison['candidate_experiment_id']}` / `{comparison['candidate_strategy']}`",
        f"Baseline: `{comparison['baseline_experiment_id']}` / `{comparison['baseline_strategy']}`",
        f"Paired seeds: `{comparison['paired_seed_count']}`",
        "",
        "## Score",
        "",
        f"- Candidate mean seat score: `{comparison['candidate_mean_score_per_seat']:.4f}`",
        f"- Baseline mean seat score: `{comparison['baseline_mean_score_per_seat']:.4f}`",
        f"- Mean delta candidate-baseline: `{comparison['mean_delta_candidate_minus_baseline']:.4f}`",
        f"- Median delta: `{comparison['median_delta_candidate_minus_baseline']:.4f}`",
        f"- Min / max delta: `{comparison['min_delta_candidate_minus_baseline']:.4f}` / `{comparison['max_delta_candidate_minus_baseline']:.4f}`",
        f"- Delta SEM: `{comparison['sem_delta_candidate_minus_baseline']:.4f}`",
        "",
        "## Time",
        "",
        f"- Candidate mean decision seconds: `{comparison['candidate_mean_total_decision_seconds']}`",
        f"- Baseline mean decision seconds: `{comparison['baseline_mean_total_decision_seconds']}`",
        f"- Speedup factor: `{comparison['speedup_factor']}`",
        f"- Time reduction: `{comparison['time_reduction']}`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-report", required=True)
    parser.add_argument("--baseline-report", required=True)
    parser.add_argument("--candidate-strategy", default="prefilter-search")
    parser.add_argument("--baseline-strategy", default="full-search")
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    args = parser.parse_args()

    candidate_report = json.loads(Path(args.candidate_report).read_text(encoding="utf-8"))
    baseline_report = json.loads(Path(args.baseline_report).read_text(encoding="utf-8"))
    comparison = compare_reports(
        candidate_report=candidate_report,
        baseline_report=baseline_report,
        candidate_strategy=args.candidate_strategy,
        baseline_strategy=args.baseline_strategy,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown_summary(comparison, Path(args.summary_out))
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
