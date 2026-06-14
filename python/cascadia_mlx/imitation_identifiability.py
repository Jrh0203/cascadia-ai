"""Audit whether MCE evidence identifies stable decision winners."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import blake3
import numpy as np

from cascadia_mlx.imitation_parent_hidden_dataset import (
    ImitationParentHiddenEvidenceDataset,
)

NORMAL_95 = 1.959963984540054
PAIRWISE_VARIANCE_FLOOR = 1.0
PHASES = (
    ("opening", 0, 5),
    ("early", 5, 10),
    ("middle", 10, 15),
    ("late", 15, 20),
)


@dataclass
class AuditAccumulator:
    groups: int = 0
    candidates: int = 0
    scored_candidates: int = 0
    exact_ties: int = 0
    distinguishable_95: int = 0
    separated_intervals_95: int = 0
    selected_parent_top1: int = 0
    selected_parent_top5: int = 0
    selected_parent_top10: int = 0
    selected_parent_top32: int = 0
    selected_immediate_top1: int = 0
    selected_immediate_top5: int = 0
    selected_immediate_top10: int = 0
    selected_immediate_top32: int = 0
    margin: list[float] = field(default_factory=list)
    combined_standard_error: list[float] = field(default_factory=list)
    margin_z_score: list[float] = field(default_factory=list)
    selected_standard_error: list[float] = field(default_factory=list)
    selected_samples: list[float] = field(default_factory=list)
    runner_up_samples: list[float] = field(default_factory=list)
    scored_per_group: list[float] = field(default_factory=list)
    confidence_set_68: list[float] = field(default_factory=list)
    confidence_set_95: list[float] = field(default_factory=list)
    selected_parent_rank: list[float] = field(default_factory=list)
    selected_immediate_rank: list[float] = field(default_factory=list)

    def add(
        self,
        *,
        means: np.ndarray,
        stddev: np.ndarray,
        samples: np.ndarray,
        scored: np.ndarray,
        selected: np.ndarray,
        parent_rank: np.ndarray,
        immediate_rank: np.ndarray,
    ) -> None:
        scored_indices = np.flatnonzero(scored)
        selected_index = int(np.flatnonzero(selected)[0])
        if selected_index not in scored_indices:
            raise ValueError("selected action is not teacher-scored")
        alternatives = scored_indices[scored_indices != selected_index]
        if len(alternatives) == 0:
            raise ValueError("identifiability audit requires two scored actions")
        runner_up = int(alternatives[np.argmax(means[alternatives])])
        standard_error = stddev / np.sqrt(np.maximum(samples, 1.0))
        margin = float(means[selected_index] - means[runner_up])
        combined = float(np.hypot(standard_error[selected_index], standard_error[runner_up]))
        z_score = margin / combined if combined > 0 else float("inf")
        selected_lower = means[selected_index] - NORMAL_95 * standard_error[selected_index]
        alternative_upper = np.max(means[alternatives] + NORMAL_95 * standard_error[alternatives])

        self.groups += 1
        self.candidates += len(means)
        self.scored_candidates += len(scored_indices)
        self.exact_ties += int(margin == 0)
        self.distinguishable_95 += int(margin > NORMAL_95 * combined)
        self.separated_intervals_95 += int(selected_lower > alternative_upper)
        self.margin.append(margin)
        self.combined_standard_error.append(combined)
        self.margin_z_score.append(z_score)
        self.selected_standard_error.append(float(standard_error[selected_index]))
        self.selected_samples.append(float(samples[selected_index]))
        self.runner_up_samples.append(float(samples[runner_up]))
        self.scored_per_group.append(float(len(scored_indices)))
        self.confidence_set_68.append(
            float(
                np.count_nonzero(
                    means[selected_index] - means[scored_indices]
                    <= np.hypot(
                        standard_error[selected_index],
                        standard_error[scored_indices],
                    )
                )
            )
        )
        self.confidence_set_95.append(
            float(
                np.count_nonzero(
                    means[selected_index] - means[scored_indices]
                    <= NORMAL_95
                    * np.hypot(
                        standard_error[selected_index],
                        standard_error[scored_indices],
                    )
                )
            )
        )

        selected_parent_rank = float(parent_rank[selected_index])
        selected_immediate_rank = float(immediate_rank[selected_index])
        self.selected_parent_rank.append(selected_parent_rank)
        self.selected_immediate_rank.append(selected_immediate_rank)
        self.selected_parent_top1 += int(selected_parent_rank <= 1)
        self.selected_parent_top5 += int(selected_parent_rank <= 5)
        self.selected_parent_top10 += int(selected_parent_rank <= 10)
        self.selected_parent_top32 += int(selected_parent_rank <= 32)
        self.selected_immediate_top1 += int(selected_immediate_rank <= 1)
        self.selected_immediate_top5 += int(selected_immediate_rank <= 5)
        self.selected_immediate_top10 += int(selected_immediate_rank <= 10)
        self.selected_immediate_top32 += int(selected_immediate_rank <= 32)

    def report(self) -> dict[str, Any]:
        if self.groups == 0:
            raise ValueError("identifiability audit received no groups")
        margin = np.asarray(self.margin)
        combined = np.asarray(self.combined_standard_error)
        floor_scale = np.sqrt(combined**2 + PAIRWISE_VARIANCE_FLOOR**2)
        return {
            "groups": self.groups,
            "candidates": self.candidates,
            "scored_candidates": self.scored_candidates,
            "mean_scored_candidates": self.scored_candidates / self.groups,
            "top_two_margin": _distribution(self.margin),
            "combined_standard_error": _distribution(self.combined_standard_error),
            "margin_z_score": _distribution(self.margin_z_score),
            "selected_standard_error": _distribution(self.selected_standard_error),
            "selected_samples": _distribution(self.selected_samples),
            "runner_up_samples": _distribution(self.runner_up_samples),
            "scored_candidates_per_group": _distribution(self.scored_per_group),
            "confidence_set_size_68": _distribution(self.confidence_set_68),
            "confidence_set_size_95": _distribution(self.confidence_set_95),
            "exact_tie_fraction": self.exact_ties / self.groups,
            "margin_at_most_0_25_fraction": float(np.mean(margin <= 0.25)),
            "margin_at_most_0_5_fraction": float(np.mean(margin <= 0.5)),
            "margin_at_most_1_fraction": float(np.mean(margin <= 1.0)),
            "margin_at_most_2_fraction": float(np.mean(margin <= 2.0)),
            "margin_within_combined_se_fraction": float(np.mean(margin <= combined)),
            "margin_within_pairwise_floor_scale_fraction": float(np.mean(margin <= floor_scale)),
            "distinguishable_winner_95_fraction": self.distinguishable_95 / self.groups,
            "separated_confidence_intervals_95_fraction": self.separated_intervals_95 / self.groups,
            "selected_parent_rank": _rank_report(
                self.selected_parent_rank,
                self.selected_parent_top1,
                self.selected_parent_top5,
                self.selected_parent_top10,
                self.selected_parent_top32,
                self.groups,
            ),
            "selected_immediate_rank": _rank_report(
                self.selected_immediate_rank,
                self.selected_immediate_top1,
                self.selected_immediate_top5,
                self.selected_immediate_top10,
                self.selected_immediate_top32,
                self.groups,
            ),
        }


def audit_dataset(dataset_path: str | Path) -> dict[str, Any]:
    dataset = ImitationParentHiddenEvidenceDataset(dataset_path)
    overall = AuditAccumulator()
    phases = {name: AuditAccumulator() for name, _, _ in PHASES}

    for batch in dataset.batches(64):
        candidate_masks = np.asarray(batch.candidate_mask)
        means = np.asarray(batch.teacher_mean)
        stddev = np.asarray(batch.teacher_stddev)
        samples = np.asarray(batch.teacher_samples)
        scored = np.asarray(batch.teacher_scored)
        selected = np.asarray(batch.selected)
        parent_rank = np.asarray(batch.parent_rank)
        immediate_rank = np.asarray(batch.immediate_rank)
        turns = np.asarray(batch.turn)
        for index, mask in enumerate(candidate_masks):
            phase = _phase_name(int(turns[index]) // 4)
            values = {
                "means": means[index][mask],
                "stddev": stddev[index][mask],
                "samples": samples[index][mask],
                "scored": scored[index][mask],
                "selected": selected[index][mask],
                "parent_rank": parent_rank[index][mask],
                "immediate_rank": immediate_rank[index][mask],
            }
            overall.add(**values)
            phases[phase].add(**values)

    manifest_path = Path(dataset_path) / "dataset.json"
    return {
        "schema_version": 1,
        "dataset": {
            "path": str(Path(dataset_path).resolve()),
            "dataset_id": dataset.manifest["dataset_id"],
            "split": dataset.split,
            "first_game_index": dataset.manifest["source"]["first_game_index"],
            "games": dataset.manifest["requested_games"],
            "manifest_blake3": _checksum(manifest_path),
            "teacher": dataset.evidence.manifest["teacher"],
        },
        "method": {
            "normal_95": NORMAL_95,
            "pairwise_variance_floor_points": PAIRWISE_VARIANCE_FLOOR,
            "phase_definition": (
                "acting turn = global turn // 4; bins [0,5), [5,10), [10,15), [15,20)"
            ),
            "caveat": (
                "Normal approximations treat candidate estimates as independent; "
                "adaptive allocation and shared game structure can violate that assumption."
            ),
        },
        "overall": overall.report(),
        "phases": {name: accumulator.report() for name, accumulator in phases.items()},
    }


def _phase_name(acting_turn: int) -> str:
    for name, start, end in PHASES:
        if start <= acting_turn < end:
            return name
    raise ValueError(f"acting turn outside [0, 20): {acting_turn}")


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if len(finite) == 0:
        return {
            "mean": float("inf"),
            "p10": float("inf"),
            "p25": float("inf"),
            "p50": float("inf"),
            "p75": float("inf"),
            "p90": float("inf"),
            "p95": float("inf"),
            "max": float("inf"),
        }
    quantiles = np.quantile(finite, [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0])
    return {
        "mean": float(np.mean(finite)),
        "p10": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p90": float(quantiles[4]),
        "p95": float(quantiles[5]),
        "max": float(quantiles[6]),
    }


def _rank_report(
    values: list[float],
    top1: int,
    top5: int,
    top10: int,
    top32: int,
    groups: int,
) -> dict[str, Any]:
    return {
        "distribution": _distribution(values),
        "top1_fraction": top1 / groups,
        "top5_fraction": top5 / groups,
        "top10_fraction": top10 / groups,
        "top32_fraction": top32 / groups,
        "mean_reciprocal_rank": float(np.mean(1.0 / np.asarray(values))),
    }


def _checksum(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_dataset(args.dataset)
    if args.output is not None:
        _write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
