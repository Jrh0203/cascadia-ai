"""Deployment-aligned expected-rank supervision for ADR 0101."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx

from cascadia_mlx.graded_oracle_frontier_expected_rank import (
    EXPECTED_RANK_STUDENT_TEMPERATURE,
    ExpectedRankBatch,
    ExpectedRankDataset,
    build_expected_rank_cache,
    expected_rank_validation_gates,
    frontier_expected_rank_loss,
)
from cascadia_mlx.graded_oracle_model import GradedOracleRanker

EXPERIMENT_ID = "complete-action-frontier-expected-rank-scale16-v1"
TARGET_SCALE = 16.0
STUDENT_TEMPERATURE = EXPECTED_RANK_STUDENT_TEMPERATURE
ADR0100_TRAIN_RECALL = 0.32205540283358003
MATERIAL_TRAIN_RECALL = ADR0100_TRAIN_RECALL + 0.10


class Scale16ExpectedRankDataset(ExpectedRankDataset):
    """Expected-rank dataset bound to the frozen ADR 0101 target contract."""

    def __init__(
        self,
        root: str | Path,
        cache_root: str | Path,
        *,
        verify_checksums: bool = True,
    ):
        super().__init__(
            root,
            cache_root,
            verify_checksums=verify_checksums,
            experiment_id=EXPERIMENT_ID,
            target_scale=TARGET_SCALE,
            student_temperature=STUDENT_TEMPERATURE,
        )


def build_scale16_expected_rank_cache(
    dataset_root: str | Path,
    cache_root: str | Path,
    *,
    workers: int = 8,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build one cache under the exact ADR 0101 manifest contract."""
    return build_expected_rank_cache(
        dataset_root,
        cache_root,
        workers=workers,
        overwrite=overwrite,
        experiment_id=EXPERIMENT_ID,
        target_scale=TARGET_SCALE,
        student_temperature=STUDENT_TEMPERATURE,
    )


def frontier_expected_rank_scale16_loss(
    model: GradedOracleRanker,
    batch: ExpectedRankBatch,
) -> mx.array:
    """Apply the single frozen scale-16 objective."""
    if float(batch.target_scale) != TARGET_SCALE:
        raise ValueError("scale-16 expected-rank batch target scale drifted")
    if float(batch.student_temperature) != STUDENT_TEMPERATURE:
        raise ValueError("scale-16 expected-rank student temperature drifted")
    return frontier_expected_rank_loss(model, batch)


def classify_scale16_expected_rank_pilot(
    gates: dict[str, bool],
    *,
    train_target_recall: float,
) -> str:
    """Apply the frozen ADR 0101 classification precedence."""
    pipeline_names = (
        "cache_identity_passed",
        "optimization_audit_passed",
        "selected_replay_identical",
        "all_groups_and_candidates_scored_once",
        "all_scores_finite",
        "sealed_test_unopened",
        "gameplay_unopened",
        "new_teacher_compute_unused",
        "external_compute_unused",
    )
    if not all(gates.get(name, False) for name in pipeline_names):
        return "scale16_pipeline_invalid"
    if gates.get("pilot_passed", False):
        return "scale16_expected_rank_model_sufficient"
    train_fit = (
        gates["train_expected_rank_target_recall_at_least_0_80"]
        and gates["train_expected_rank_exact_sets_at_least_0_25"]
    )
    if train_fit:
        return "scale16_expected_rank_train_fit_only"
    if train_target_recall >= MATERIAL_TRAIN_RECALL:
        return "scale16_alignment_material_but_underfit"
    return "scale16_alignment_insufficient"


__all__ = [
    "ADR0100_TRAIN_RECALL",
    "EXPERIMENT_ID",
    "MATERIAL_TRAIN_RECALL",
    "STUDENT_TEMPERATURE",
    "TARGET_SCALE",
    "Scale16ExpectedRankDataset",
    "build_scale16_expected_rank_cache",
    "classify_scale16_expected_rank_pilot",
    "expected_rank_validation_gates",
    "frontier_expected_rank_scale16_loss",
]
