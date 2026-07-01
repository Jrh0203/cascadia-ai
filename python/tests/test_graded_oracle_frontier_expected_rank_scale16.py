from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from cascadia_mlx.graded_oracle_frontier_expected_rank import (
    compare_expected_rank_array_payloads,
    expected_rank_loss_from_scores,
)
from cascadia_mlx.graded_oracle_frontier_expected_rank_scale16 import (
    EXPERIMENT_ID,
    MATERIAL_TRAIN_RECALL,
    TARGET_SCALE,
    classify_scale16_expected_rank_pilot,
    frontier_expected_rank_scale16_loss,
)
from cascadia_mlx.graded_oracle_frontier_expected_rank_scale16_train import (
    RUN_KIND,
    frontier_expected_rank_scale16_adapter,
)
from cascadia_mlx.graded_oracle_frontier_expected_rank_train import (
    FrontierExpectedRankTrainingConfig,
)


def _gates() -> dict[str, bool]:
    return {
        "cache_identity_passed": True,
        "optimization_audit_passed": True,
        "selected_replay_identical": True,
        "all_groups_and_candidates_scored_once": True,
        "all_scores_finite": True,
        "sealed_test_unopened": True,
        "gameplay_unopened": True,
        "new_teacher_compute_unused": True,
        "external_compute_unused": True,
        "train_expected_rank_target_recall_at_least_0_80": False,
        "train_expected_rank_exact_sets_at_least_0_25": False,
        "pilot_passed": False,
    }


def _write_cache_payload(root: Path, *, experiment_id: str, scale: float) -> None:
    root.mkdir()
    arrays = {
        "group_ids.npy": np.asarray([1], dtype=np.uint64),
        "candidate_counts.npy": np.asarray([2], dtype=np.uint32),
        "offsets.npy": np.asarray([0, 2], dtype=np.uint64),
        "expected_ranks.npy": np.asarray([1.0, 2.0], dtype=np.float32),
    }
    for name, values in arrays.items():
        np.save(root / name, values, allow_pickle=False)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "target": {"target_scale": scale},
                "ordered_group_action_identity_blake3": "same",
            }
        )
    )


def test_scale16_concentrates_loss_more_than_scale64() -> None:
    ranks = mx.array([[1.0, 16.0, 64.0, 0.0]])
    target = mx.array([[True, True, True, False]])
    eligible = mx.array([[True, True, True, True]])
    scores = mx.array([[4.0, 1.0, -2.0, -4.0]])
    scale16 = expected_rank_loss_from_scores(
        scores,
        ranks,
        target,
        eligible,
        target_scale=TARGET_SCALE,
    )
    scale64 = expected_rank_loss_from_scores(
        scores,
        ranks,
        target,
        eligible,
        target_scale=64.0,
    )
    mx.eval(scale16, scale64)
    assert float(scale16.item()) < float(scale64.item())


def test_scale16_classification_boundaries_are_frozen() -> None:
    gates = _gates()
    assert (
        classify_scale16_expected_rank_pilot(
            gates,
            train_target_recall=MATERIAL_TRAIN_RECALL,
        )
        == "scale16_alignment_material_but_underfit"
    )
    assert (
        classify_scale16_expected_rank_pilot(
            gates,
            train_target_recall=MATERIAL_TRAIN_RECALL - 1e-9,
        )
        == "scale16_alignment_insufficient"
    )
    gates["train_expected_rank_target_recall_at_least_0_80"] = True
    gates["train_expected_rank_exact_sets_at_least_0_25"] = True
    assert (
        classify_scale16_expected_rank_pilot(
            gates,
            train_target_recall=0.80,
        )
        == "scale16_expected_rank_train_fit_only"
    )
    gates["pilot_passed"] = True
    assert (
        classify_scale16_expected_rank_pilot(
            gates,
            train_target_recall=0.80,
        )
        == "scale16_expected_rank_model_sufficient"
    )
    gates["cache_identity_passed"] = False
    assert (
        classify_scale16_expected_rank_pilot(
            gates,
            train_target_recall=0.80,
        )
        == "scale16_pipeline_invalid"
    )


def test_rank_array_comparison_allows_protocol_metadata_change(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    treatment = tmp_path / "treatment"
    _write_cache_payload(source, experiment_id="scale64", scale=64.0)
    _write_cache_payload(
        treatment,
        experiment_id=EXPERIMENT_ID,
        scale=TARGET_SCALE,
    )
    report = compare_expected_rank_array_payloads(source, treatment)
    assert report["all_file_bytes_identical"]
    assert report["ordered_group_action_identity_identical"]
    assert report["left_target_scale"] == 64.0
    assert report["right_target_scale"] == TARGET_SCALE


def test_scale16_adapter_keeps_the_frozen_training_protocol(
    tmp_path: Path,
) -> None:
    train_cache = tmp_path / "train-cache"
    validation_cache = tmp_path / "validation-cache"
    train_cache.mkdir()
    validation_cache.mkdir()
    config = FrontierExpectedRankTrainingConfig(
        train_dataset=tmp_path / "train",
        validation_dataset=tmp_path / "validation",
        run_dir=tmp_path / "run",
        train_target_cache=str(train_cache),
        validation_target_cache=str(validation_cache),
    )
    config.validate()
    adapter = frontier_expected_rank_scale16_adapter(config)
    assert adapter.kind == RUN_KIND
    assert adapter.selection_metric == "expected_rank_target_positive_miss_rate"
    assert adapter.accuracy_metric == "expected_rank_target_set_exact_fraction"


def test_scale16_loss_rejects_a_mismatched_batch_contract() -> None:
    batch = SimpleNamespace(target_scale=64.0, student_temperature=2.0)
    with pytest.raises(ValueError, match="target scale drifted"):
        frontier_expected_rank_scale16_loss(SimpleNamespace(), batch)
