#!/usr/bin/env python3
"""Aggregate the protected and all-V3 Cascadia V3 final evaluations."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REQUIRED_ANATOMY = {
    "bear",
    "elk",
    "salmon",
    "hawk",
    "fox",
    "wildlife_total",
    "forest",
    "mountain",
    "prairie",
    "wetland",
    "river",
    "terrain_total",
    "nature_tokens",
    "pinecones",
    "overflow_states",
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _summary(values: list[float], *, block_size: int | None = None) -> dict[str, object]:
    if len(values) < 2:
        raise ValueError("score summary requires at least two observations")
    inference_values = values
    if block_size is not None:
        inference_values = [
            statistics.fmean(values[start : start + block_size])
            for start in range(0, len(values), block_size)
            if len(values[start : start + block_size]) == block_size
        ]
        if len(inference_values) < 2:
            raise ValueError("game-block confidence interval has fewer than two blocks")
    mean = statistics.fmean(values)
    standard_error = statistics.stdev(inference_values) / math.sqrt(len(inference_values))
    return {
        "count": len(values),
        "mean": mean,
        "standard_error": standard_error,
        "confidence_interval_95": [mean - 1.96 * standard_error, mean + 1.96 * standard_error],
        "p10": _percentile(values, 0.10),
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "histogram": dict(sorted(Counter(round(value) for value in values).items())),
        "inference_blocks": len(inference_values),
        "block_size": block_size,
    }


def _anatomy(rows: list[dict[str, Any]]) -> dict[str, object]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        anatomy = row.get("anatomy", {})
        if not isinstance(anatomy, dict):
            raise ValueError("score anatomy must be an object")
        missing = REQUIRED_ANATOMY - set(anatomy)
        if missing:
            raise ValueError(f"score anatomy omits required fields: {sorted(missing)}")
        for name, value in anatomy.items():
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"anatomy field {name} is not finite")
            values[name].append(float(value))
        values["overflow_frequency"].append(float(anatomy["overflow_states"]) / 20.0)
    return {
        name: {
            "mean": statistics.fmean(series),
            "p10": _percentile(series, 0.10),
            "p50": _percentile(series, 0.50),
            "p90": _percentile(series, 0.90),
        }
        for name, series in sorted(values.items())
    }


def _all_v3_score_summary(games: list[dict[str, Any]]) -> dict[str, object]:
    if any(not isinstance(game.get("seats"), list) or len(game["seats"]) != 4 for game in games):
        raise ValueError("every all-V3 game must contain exactly four seats")
    scores = [float(seat["score"]) for game in games for seat in game["seats"]]
    summary = _summary(scores)
    block_size_games = 25
    block_means = [
        statistics.fmean(
            float(seat["score"])
            for game in games[start : start + block_size_games]
            for seat in game["seats"]
        )
        for start in range(0, len(games), block_size_games)
        if len(games[start : start + block_size_games]) == block_size_games
    ]
    standard_error = statistics.stdev(block_means) / math.sqrt(len(block_means))
    mean = float(summary["mean"])
    summary.update(
        {
            "standard_error": standard_error,
            "confidence_interval_95": [
                mean - 1.96 * standard_error,
                mean + 1.96 * standard_error,
            ],
            "inference_blocks": len(block_means),
            "block_size_games": block_size_games,
        }
    )
    return summary


def _latency(values: list[float]) -> dict[str, object]:
    if not values or any(value < 0 or not math.isfinite(value) for value in values):
        raise ValueError("latency observations must be finite and non-negative")
    return {
        "observations": len(values),
        "total_seconds": sum(values),
        "mean_seconds": statistics.fmean(values),
        "p10_seconds": _percentile(values, 0.10),
        "p50_seconds": _percentile(values, 0.50),
        "p90_seconds": _percentile(values, 0.90),
    }


def _throughput(count: int, worker_seconds: float) -> dict[str, float | int | None]:
    return {
        "count": count,
        "worker_seconds": worker_seconds,
        "items_per_worker_second": (
            count / worker_seconds if worker_seconds > 0 else None
        ),
    }


def build_report(
    protected: dict[str, Any],
    all_v3: dict[str, Any],
    *,
    campaign_history: list[dict[str, Any]] | None = None,
    resource_observations: dict[str, Any] | None = None,
    champion: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, object]:
    pairs = protected.get("pairs")
    games = all_v3.get("games")
    if not isinstance(pairs, list) or len(pairs) != 250:
        raise ValueError("protected evaluation must contain exactly 250 pairs")
    if not isinstance(games, list) or len(games) not in (1_000, 4_000):
        raise ValueError("all-V3 evaluation must contain 1,000 or 4,000 games")
    treatment_rows = [pair["treatment"] for pair in pairs]
    control_rows = [pair["control"] for pair in pairs]
    treatment = [float(row["score"]) for row in treatment_rows]
    control = [float(row["score"]) for row in control_rows]
    treatment_latency = [float(row["focal_seconds"]) for row in treatment_rows]
    control_latency = [float(row["focal_seconds"]) for row in control_rows]
    deltas = [left - right for left, right in zip(treatment, control, strict=True)]
    paired = _summary(deltas)
    paired_lower, paired_upper = paired["confidence_interval_95"]
    protected_classification = (
        "outperforming"
        if paired["mean"] > 0 and paired_lower > 0
        else "inferior"
        if paired_upper < 0
        else "inconclusive"
    )
    win_tie_loss = {
        "wins": sum(delta > 0 for delta in deltas),
        "ties": sum(delta == 0 for delta in deltas),
        "losses": sum(delta < 0 for delta in deltas),
    }

    seat_rows = [seat for game in games for seat in game["seats"]]
    scores = [float(row["score"]) for row in seat_rows]
    all_v3_latency = [float(row["decision_seconds"]) for row in seat_rows]
    all_v3_summary = _all_v3_score_summary(games)
    lower, upper = all_v3_summary["confidence_interval_95"]
    requires_extension = len(games) == 1_000 and lower < 100 <= upper
    goal_claimed = all_v3_summary["mean"] >= 100 and lower >= 100
    protected_resources = protected.get("resource_metrics", {})
    all_v3_resources = all_v3.get("resource_metrics", {})
    if not isinstance(protected_resources, dict) or not isinstance(all_v3_resources, dict):
        raise ValueError("resource metrics must be objects")
    protected_worker_seconds = float(protected_resources.get("worker_elapsed_seconds", 0.0))
    all_v3_worker_seconds = float(all_v3_resources.get("worker_elapsed_seconds", 0.0))
    if protected_worker_seconds < 0 or all_v3_worker_seconds < 0:
        raise ValueError("worker elapsed time cannot be negative")
    history = campaign_history or []
    if campaign_history is not None and (
        len(history) != 10
        or [int(item.get("cycle", -1)) for item in history] != list(range(1, 11))
    ):
        raise ValueError("campaign history must contain expert cycles 1 through 10")
    return {
        "schema_id": "cascadia-v3-final-report-v1",
        "passed": True,
        "protected_pairs": {
            "pairs": 250,
            "physical_games": 500,
            "treatment": _summary(treatment),
            "control": _summary(control),
            "paired_delta": paired,
            "win_tie_loss": win_tie_loss,
            "classification": protected_classification,
            "treatment_anatomy": _anatomy(treatment_rows),
            "control_anatomy": _anatomy(control_rows),
            "latency": {
                "treatment_focal_game": _latency(treatment_latency),
                "control_focal_game": _latency(control_latency),
            },
            "throughput": _throughput(500, protected_worker_seconds),
        },
        "all_v3": {
            "games": len(games),
            "seat_games": len(scores),
            "score": all_v3_summary,
            "anatomy": _anatomy(seat_rows),
            "requires_4000_game_extension": requires_extension,
            "goal_100_claimed": goal_claimed,
            "latency": {"seat_game": _latency(all_v3_latency)},
            "throughput": _throughput(len(games), all_v3_worker_seconds),
        },
        "recommendation": (
            "goal-achieved"
            if goal_claimed and protected_classification == "outperforming"
            else "extend-to-4000"
            if requires_extension
            else "continue-v3"
            if protected_classification == "outperforming"
            else "retain-qualified-control"
        ),
        "resource_metrics": {
            "protected": protected_resources,
            "all_v3": all_v3_resources,
            "fleet": resource_observations or {},
        },
        "expert_iteration_learning_curves": history,
        "final_champion": champion or {},
        "provenance": provenance or {},
    }


def render_markdown(report: dict[str, Any]) -> str:
    paired = report["protected_pairs"]["paired_delta"]
    all_v3 = report["all_v3"]["score"]
    lines = [
        "# Cascadia V3 Final Campaign Report",
        "",
        f"Recommendation: **{report['recommendation']}**.",
        "",
        "## Final evaluation",
        "",
        "| Domain | N | Mean | 95% CI | P10 | P50 | P90 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| Protected paired delta | {paired['count']} | {paired['mean']:.3f} | "
            f"[{paired['confidence_interval_95'][0]:.3f}, "
            f"{paired['confidence_interval_95'][1]:.3f}] | {paired['p10']:.1f} | "
            f"{paired['p50']:.1f} | {paired['p90']:.1f} |"
        ),
        (
            f"| All-V3 seat score | {all_v3['count']} | {all_v3['mean']:.3f} | "
            f"[{all_v3['confidence_interval_95'][0]:.3f}, "
            f"{all_v3['confidence_interval_95'][1]:.3f}] | {all_v3['p10']:.1f} | "
            f"{all_v3['p50']:.1f} | {all_v3['p90']:.1f} |"
        ),
        "",
        (
            "Protected classification: "
            f"**{report['protected_pairs']['classification']}**; "
            f"W/T/L {report['protected_pairs']['win_tie_loss']['wins']}/"
            f"{report['protected_pairs']['win_tie_loss']['ties']}/"
            f"{report['protected_pairs']['win_tie_loss']['losses']}."
        ),
        "",
        "## Expert-iteration history",
        "",
        "| Cycle | Selected origin | Promotion | Pairs/tier | K32/R600 delta |",
        "|---:|---|---|---:|---:|",
    ]
    for cycle in report.get("expert_iteration_learning_curves", []):
        promotion = cycle["promotion"]
        k600 = promotion.get("tiers", {}).get("k32-r600", {})
        lines.append(
            f"| {cycle['cycle']} | {cycle['selected_origin']} | "
            f"{promotion['verdict']} | {promotion['pairs_per_tier']} | "
            f"{float(k600.get('mean_delta', float('nan'))):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Score anatomy",
            "",
            "| Component | All-V3 mean | P10 | P50 | P90 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, values in report["all_v3"]["anatomy"].items():
        lines.append(
            f"| {name} | {values['mean']:.3f} | {values['p10']:.3f} | "
            f"{values['p50']:.3f} | {values['p90']:.3f} |"
        )
    lines.extend(
        [
            "",
            "The JSON report is authoritative for complete histograms, resource and "
            "latency telemetry, model identity, provenance, and every training-loss sample.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protected", type=Path, required=True)
    parser.add_argument("--all-v3", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_report(_read(args.protected), _read(args.all_v3))
    _write_atomic(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
