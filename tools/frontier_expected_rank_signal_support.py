#!/usr/bin/env python3
"""Independent target-signal audits for ADR 0100 expected-rank supervision."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from cascadia_mlx.graded_oracle_dataset import (
    GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
)
from cascadia_mlx.graded_oracle_frontier_anchor import (
    GRADED_SOURCE_CHAMPION_FRONTIER,
)
from cascadia_mlx.graded_oracle_frontier_expected_rank import (
    EXPECTED_RANK_STUDENT_TEMPERATURE,
    EXPECTED_RANK_TARGET_SCALE,
    EXPERIMENT_ID,
    ExpectedRankDataset,
    build_expected_rank_target_mask,
)

RESIDUAL_RANGES = (0.0, 3.0, 6.0, 12.0)
MASS_THRESHOLDS = (0.50, 0.80, 0.90, 0.95)
TARGET_SCALES = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)


@dataclass
class _SignalAccumulator:
    groups: int = 0
    candidates: int = 0
    labeled_candidates: int = 0
    deployed_targets: int = 0
    entropy_bits: list[float] = field(default_factory=list)
    effective_support: list[float] = field(default_factory=list)
    deployed_target_mass: list[float] = field(default_factory=list)
    target_gradient_fraction: list[float] = field(default_factory=list)
    outside_target_probability_mass: list[float] = field(default_factory=list)
    support_at_mass: dict[float, list[float]] = field(
        default_factory=lambda: {value: [] for value in MASS_THRESHOLDS}
    )
    scale_target_mass: dict[float, list[float]] = field(
        default_factory=lambda: {value: [] for value in TARGET_SCALES}
    )
    scale_entropy_bits: dict[float, list[float]] = field(
        default_factory=lambda: {value: [] for value in TARGET_SCALES}
    )
    reachability_hits: dict[float, int] = field(
        default_factory=lambda: {value: 0 for value in RESIDUAL_RANGES}
    )
    reachability_exact: dict[float, int] = field(
        default_factory=lambda: {value: 0 for value in RESIDUAL_RANGES}
    )
    required_residual_range: list[float] = field(default_factory=list)


def audit_expected_rank_signal(
    dataset: ExpectedRankDataset,
    *,
    analysis: str,
) -> dict[str, Any]:
    """Audit target concentration, uniform-logit gradients, or reachability."""
    if analysis not in {
        "concentration",
        "gradient",
        "reachability",
        "scale-sweep",
    }:
        raise ValueError(f"unknown expected-rank signal analysis: {analysis}")
    accumulator = _SignalAccumulator()
    for batch in dataset.batches(
        1,
        maximum_actions_per_batch=GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
        maximum_group_actions=GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
    ):
        candidate_mask = np.asarray(batch.candidate_mask)
        count = int(np.sum(candidate_mask[0]))
        expected_rank = np.asarray(batch.expected_rank)[0, :count]
        expected_rank_mask = np.asarray(batch.expected_rank_mask)[0, :count]
        source_flags = np.asarray(batch.source_flags)[0, :count]
        action_hashes = np.asarray(batch.action_hash)[0, :count]
        target = build_expected_rank_target_mask(
            expected_rank=np.asarray(batch.expected_rank),
            expected_rank_mask=np.asarray(batch.expected_rank_mask),
            source_flags=np.asarray(batch.source_flags),
            candidate_mask=candidate_mask,
            action_hashes=np.asarray(batch.action_hash),
        )[0, :count]
        frontier = (
            source_flags.astype(np.int64) & GRADED_SOURCE_CHAMPION_FRONTIER
        ) != 0
        eligible = ~frontier
        _accumulate_group(
            accumulator,
            analysis=analysis,
            expected_rank=expected_rank,
            expected_rank_mask=expected_rank_mask,
            target=target,
            eligible=eligible,
            screen=np.asarray(batch.screen_value)[0, :count],
            action_hashes=action_hashes,
        )
    return _build_report(dataset, analysis, accumulator)


def _accumulate_group(
    accumulator: _SignalAccumulator,
    *,
    analysis: str,
    expected_rank: np.ndarray,
    expected_rank_mask: np.ndarray,
    target: np.ndarray,
    eligible: np.ndarray,
    screen: np.ndarray,
    action_hashes: np.ndarray,
) -> None:
    labeled = np.flatnonzero(expected_rank_mask)
    target_indices = np.flatnonzero(target)
    eligible_indices = np.flatnonzero(eligible)
    if not len(labeled) or not len(target_indices) or not len(eligible_indices):
        raise ValueError("expected-rank signal audit encountered an empty cohort")

    accumulator.groups += 1
    accumulator.candidates += len(expected_rank)
    accumulator.labeled_candidates += len(labeled)
    accumulator.deployed_targets += len(target_indices)

    if analysis in {"concentration", "gradient", "scale-sweep"}:
        labeled_rank = expected_rank[labeled].astype(np.float64)
        logits = -(labeled_rank - 1.0) / EXPECTED_RANK_TARGET_SCALE
        logits -= float(np.max(logits))
        probability = np.exp(logits)
        probability /= float(np.sum(probability))
        full_probability = np.zeros(len(expected_rank), dtype=np.float64)
        full_probability[labeled] = probability

        if analysis == "scale-sweep":
            for scale in TARGET_SCALES:
                scale_logits = -(labeled_rank - 1.0) / scale
                scale_logits -= float(np.max(scale_logits))
                scale_probability = np.exp(scale_logits)
                scale_probability /= float(np.sum(scale_probability))
                scale_full = np.zeros(len(expected_rank), dtype=np.float64)
                scale_full[labeled] = scale_probability
                positive = scale_probability[scale_probability > 0.0]
                accumulator.scale_target_mass[scale].append(
                    float(
                        np.clip(
                            np.sum(scale_full[target_indices]),
                            0.0,
                            1.0,
                        )
                    )
                )
                accumulator.scale_entropy_bits[scale].append(
                    -float(np.sum(positive * np.log2(positive)))
                )
        elif analysis == "concentration":
            positive = probability[probability > 0.0]
            entropy = -float(np.sum(positive * np.log2(positive)))
            accumulator.entropy_bits.append(entropy)
            accumulator.effective_support.append(2.0**entropy)
            accumulator.deployed_target_mass.append(
                float(
                    np.clip(
                        np.sum(full_probability[target_indices]),
                        0.0,
                        1.0,
                    )
                )
            )
            ordered = np.sort(probability)[::-1]
            cumulative = np.cumsum(ordered)
            for threshold in MASS_THRESHOLDS:
                support = int(np.searchsorted(cumulative, threshold)) + 1
                accumulator.support_at_mass[threshold].append(float(support))
        else:
            student = np.zeros(len(expected_rank), dtype=np.float64)
            student[eligible_indices] = 1.0 / len(eligible_indices)
            gradient = (
                student - full_probability
            ) / EXPECTED_RANK_STUDENT_TEMPERATURE
            absolute = np.abs(gradient)
            total = float(np.sum(absolute))
            accumulator.target_gradient_fraction.append(
                float(np.sum(absolute[target_indices])) / total
            )
            accumulator.outside_target_probability_mass.append(
                float(
                    np.clip(
                        np.sum(full_probability[eligible & ~target]),
                        0.0,
                        1.0,
                    )
                )
            )
    else:
        non_target = eligible & ~target
        required = max(
            0.0,
            float(np.max(screen[non_target]) - np.min(screen[target])) / 2.0,
        )
        accumulator.required_residual_range.append(required)
        quota = len(target_indices)
        for residual_range in RESIDUAL_RANGES:
            optimistic = screen.copy()
            optimistic[target] += residual_range
            optimistic[non_target] -= residual_range
            ranked = np.asarray(
                sorted(
                    (int(index) for index in eligible_indices),
                    key=lambda index: (
                        -float(optimistic[index]),
                        bytes(action_hashes[index]),
                    ),
                ),
                dtype=np.int32,
            )[:quota]
            recalled = int(np.sum(target[ranked]))
            accumulator.reachability_hits[residual_range] += recalled
            accumulator.reachability_exact[residual_range] += int(
                recalled == quota
            )


def _build_report(
    dataset: ExpectedRankDataset,
    analysis: str,
    accumulator: _SignalAccumulator,
) -> dict[str, Any]:
    scientific: dict[str, Any] = {
        "split": dataset.split,
        "groups": accumulator.groups,
        "candidates": accumulator.candidates,
        "labeled_candidates": accumulator.labeled_candidates,
        "deployed_target_candidates": accumulator.deployed_targets,
    }
    if analysis == "concentration":
        scientific.update(
            {
                "target_entropy_bits": _distribution(accumulator.entropy_bits),
                "target_effective_support": _distribution(
                    accumulator.effective_support
                ),
                "probability_mass_in_deployed_target": _distribution(
                    accumulator.deployed_target_mass
                ),
                "actions_required_for_probability_mass": {
                    str(threshold): _distribution(values)
                    for threshold, values in accumulator.support_at_mass.items()
                },
            }
        )
    elif analysis == "gradient":
        scientific.update(
            {
                "uniform_student_absolute_gradient_fraction_in_deployed_target": (
                    _distribution(accumulator.target_gradient_fraction)
                ),
                "target_probability_mass_outside_deployed_target": _distribution(
                    accumulator.outside_target_probability_mass
                ),
            }
        )
    elif analysis == "scale-sweep":
        scientific["scales"] = {
            str(scale): {
                "probability_mass_in_deployed_target": _distribution(
                    accumulator.scale_target_mass[scale]
                ),
                "target_entropy_bits": _distribution(
                    accumulator.scale_entropy_bits[scale]
                ),
            }
            for scale in TARGET_SCALES
        }
    else:
        scientific.update(
            {
                "required_symmetric_residual_range": _distribution(
                    accumulator.required_residual_range
                ),
                "ceilings": {
                    str(value): {
                        "target_positive_recall": (
                            accumulator.reachability_hits[value]
                            / accumulator.deployed_targets
                        ),
                        "target_set_exact_fraction": (
                            accumulator.reachability_exact[value]
                            / accumulator.groups
                        ),
                    }
                    for value in RESIDUAL_RANGES
                },
            }
        )
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "audit": f"expected-rank-{analysis}",
        "scientific": scientific,
        "test_split_opened": False,
        "gameplay_opened": False,
        "new_teacher_compute_used": False,
        "external_compute_used": False,
    }


def _distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        raise ValueError("cannot summarize an empty distribution")
    return {
        "count": len(array),
        "min": float(np.min(array)),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(np.max(array)),
    }


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "analysis",
        choices=("concentration", "gradient", "reachability", "scale-sweep"),
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_expected_rank_signal(
        ExpectedRankDataset(args.dataset, args.cache),
        analysis=args.analysis,
    )
    _write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
