from __future__ import annotations

import json
import struct
from pathlib import Path

import blake3
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest
from cascadia_mlx.checkpoint import (
    TrainerState,
    load_latest_checkpoint_with_factory,
    prune_checkpoints,
    save_checkpoint,
)
from cascadia_mlx.dataset import DatasetError
from cascadia_mlx.legacy_nnue import LegacyNnueError, _validate_derivation
from cascadia_mlx.rollout_value_dataset import (
    ROLLOUT_VALUE_FEATURE_SCHEMA,
    ROLLOUT_VALUE_HEADER_SIZE,
    ROLLOUT_VALUE_RECORD_PREFIX_SIZE,
    ROLLOUT_VALUE_SHARD_MAGIC,
    ROLLOUT_VALUE_TARGET_SCHEMA,
    RolloutValueDataset,
)
from cascadia_mlx.rollout_value_model import (
    RolloutValueNnue,
    RolloutValueNnueConfig,
    rollout_root_ranking_loss,
    rollout_value_loss,
)
from cascadia_mlx.rollout_value_train import _validation_gates

_HEADER = struct.Struct("<8sHHHHIIII8sQ32s32s32s16s")
_PREFIX = struct.Struct("<BBBBHHIQQfff")


def _checksum(path: Path) -> str:
    return blake3.blake3(path.read_bytes()).hexdigest()


def _write_dataset(root: Path) -> Path:
    root.mkdir()
    teacher = {
        "strategy_id": "exact-mlx-rollout-value-k32-r600-trace8-v1",
        "parent_model_manifest_blake3": "a" * 64,
        "weights_blake3": "b" * 64,
        "feature_count": 11_231,
        "candidate_limit": 32,
        "rollouts": 600,
        "trace_modulus": 8,
        "lmr": True,
        "diverse_prefilter": True,
    }
    records = [
        (0, 0, 1, 1, 1, 94_000, 8, 12.0, 80.0, 0.0, [1, 1, 7]),
        (0, 1, 1, 1, 1, 94_000, 16, 13.0, 79.0, 0.0, [9]),
    ]
    for decision in range(80):
        records.append(
            (
                1,
                decision,
                decision // 4 + 1,
                1,
                4,
                94_000,
                0,
                12.0,
                80.0,
                2.0,
                [2, 3],
            )
        )
    shard = root / "shard-00000.nnv"
    with shard.open("wb") as handle:
        handle.write(
            _HEADER.pack(
                ROLLOUT_VALUE_SHARD_MAGIC,
                1,
                ROLLOUT_VALUE_HEADER_SIZE,
                ROLLOUT_VALUE_RECORD_PREFIX_SIZE,
                11_231,
                len(records),
                2,
                80,
                1,
                bytes([0, 4, 0, 0, 0, 0, 0, 0]),
                94_000,
                blake3.blake3(ROLLOUT_VALUE_FEATURE_SCHEMA.encode()).digest(),
                blake3.blake3(ROLLOUT_VALUE_TARGET_SCHEMA.encode()).digest(),
                blake3.blake3(json.dumps(teacher, separators=(",", ":")).encode()).digest(),
                bytes(16),
            )
        )
        for (
            kind,
            decision,
            personal_turn,
            selected,
            samples,
            game_index,
            rollout_seed,
            immediate,
            target,
            stddev,
            features,
        ) in records:
            handle.write(
                _PREFIX.pack(
                    kind,
                    decision,
                    personal_turn,
                    selected,
                    len(features),
                    0,
                    samples,
                    game_index,
                    rollout_seed,
                    immediate,
                    target,
                    stddev,
                )
            )
            handle.write(np.asarray(features, dtype="<u2").tobytes())
    manifest = {
        "schema_version": 1,
        "dataset_id": "rollout-value-test",
        "feature_schema": ROLLOUT_VALUE_FEATURE_SCHEMA,
        "target_schema": ROLLOUT_VALUE_TARGET_SCHEMA,
        "record_prefix_size": ROLLOUT_VALUE_RECORD_PREFIX_SIZE,
        "split": "train",
        "teacher": teacher,
        "first_game_index": 94_000,
        "requested_games": 1,
        "completed_games": 1,
        "total_records": len(records),
        "trajectory_records": 2,
        "root_estimate_records": 80,
        "created_unix_seconds": 0,
        "updated_unix_seconds": 0,
        "provenance": {},
        "shards": [
            {
                "file": shard.name,
                "first_game_index": 94_000,
                "game_count": 1,
                "record_count": len(records),
                "byte_count": shard.stat().st_size,
                "blake3": _checksum(shard),
            }
        ],
    }
    (root / "dataset.json").write_text(json.dumps(manifest))
    return shard


def test_rollout_value_dataset_preserves_duplicates_and_root_groups(tmp_path: Path) -> None:
    _write_dataset(tmp_path / "rollout")
    dataset = RolloutValueDataset(tmp_path / "rollout")
    trajectory = next(dataset.batches(8))
    roots = next(dataset.batches(80, kind="root"))

    assert dataset.trajectory_count == 2
    assert dataset.root_count == 80
    assert np.array_equal(
        np.asarray(trajectory.feature_indices)[0, :3],
        np.asarray([1, 1, 7]),
    )
    offsets, indices = trajectory.exact_csr()
    assert np.array_equal(np.asarray(offsets), np.asarray([0, 3, 4]))
    assert np.array_equal(np.asarray(indices), np.asarray([1, 1, 7, 9]))
    assert roots.size == 80
    assert roots.selected.all()
    grouped = list(dataset.root_groups(7))
    assert sum(batch.group_count for batch in grouped) == 80
    assert all(np.all(np.asarray(batch.candidate_mask).sum(axis=1) == 1) for batch in grouped)
    grouped_left = [
        batch.decision_index.tolist() for batch in dataset.root_groups(7, shuffle=True, seed=19)
    ]
    grouped_right = [
        batch.decision_index.tolist() for batch in dataset.root_groups(7, shuffle=True, seed=19)
    ]
    assert grouped_left == grouped_right
    left = [batch.decision_index.tolist() for batch in dataset.batches(1, shuffle=True, seed=17)]
    right = [batch.decision_index.tolist() for batch in dataset.batches(1, shuffle=True, seed=17)]
    assert left == right


def test_rollout_value_dataset_rejects_checksum_drift(tmp_path: Path) -> None:
    shard = _write_dataset(tmp_path / "rollout")
    with shard.open("ab") as handle:
        handle.write(b"\0")
    with pytest.raises(DatasetError, match="size mismatch"):
        RolloutValueDataset(tmp_path / "rollout")


def test_derived_nnue_accepts_registered_rollout_derivation_kinds() -> None:
    common = {
        "parent_manifest_blake3": "a" * 64,
        "parent_model_blake3": "b" * 64,
        "train_dataset_manifest_blake3": "c" * 64,
        "validation_dataset_manifest_blake3": "d" * 64,
        "run_manifest_blake3": "e" * 64,
        "checkpoint_manifest_blake3": "f" * 64,
        "selected_checkpoint": "step-000000001",
    }
    for kind in (
        "mlx-rollout-return-finetune-v1",
        "mlx-joint-return-ranking-finetune-v1",
    ):
        _validate_derivation({"kind": kind, **common})
    with pytest.raises(LegacyNnueError, match="derivation metadata is invalid"):
        _validate_derivation({"kind": "unknown", **common})


def test_rollout_value_model_counts_duplicate_features_and_backpropagates() -> None:
    config = RolloutValueNnueConfig(features=3, hidden1=2, hidden2=1)
    model = RolloutValueNnue(
        config,
        tensors={
            "w1": mx.array([[0.0, 0.0], [1.0, 2.0], [0.0, 0.0]]),
            "b1": mx.zeros((2,)),
            "w2": mx.array([[1.0], [1.0]]),
            "b2": mx.zeros((1,)),
            "w3": mx.ones((1,)),
            "b3": mx.zeros((1,)),
        },
    )
    indices = mx.array([[1, 1]], dtype=mx.int32)
    mask = mx.array([[True, True]])
    prediction = model(indices, mask)
    mx.eval(prediction)
    assert float(prediction.item()) == 6.0

    batch = type(
        "Batch",
        (),
        {
            "feature_indices": indices,
            "feature_mask": mask,
            "target_remaining": mx.array([8.0]),
        },
    )()
    loss, gradients = nn.value_and_grad(model, rollout_value_loss)(model, batch)
    mx.eval(loss, gradients)
    assert np.isfinite(float(loss.item()))
    assert set(dict(gradients)) == {"w1", "b1", "w2", "b2", "w3", "b3"}


def test_rollout_root_ranking_loss_rewards_correct_ordering() -> None:
    class IndexedModel:
        def __init__(self, scale: float):
            self.scale = scale

        def __call__(self, feature_indices: mx.array, feature_mask: mx.array) -> mx.array:
            del feature_mask
            return feature_indices[:, 0].astype(mx.float32) * self.scale

    batch = type(
        "RootBatch",
        (),
        {
            "feature_indices": mx.array([[[0], [1]]], dtype=mx.int32),
            "feature_mask": mx.array([[[True], [True]]]),
            "candidate_mask": mx.array([[True, True]]),
            "target_remaining": mx.array([[0.0, 1.0]]),
            "immediate_score": mx.zeros((1, 2)),
            "selected": mx.array([[False, True]]),
        },
    )()
    correct = rollout_root_ranking_loss(IndexedModel(1.0), batch)
    reversed_order = rollout_root_ranking_loss(IndexedModel(-1.0), batch)
    mx.eval(correct, reversed_order)
    assert float(correct.item()) < float(reversed_order.item())


def test_rollout_value_checkpoint_round_trip_preserves_optimizer_and_cursor(
    tmp_path: Path,
) -> None:
    config = RolloutValueNnueConfig(features=3, hidden1=2, hidden2=1)
    model = RolloutValueNnue(config)
    optimizer = optim.AdamW(learning_rate=3e-6, weight_decay=0.0)
    batch = type(
        "Batch",
        (),
        {
            "feature_indices": mx.array([[0, 1]], dtype=mx.int32),
            "feature_mask": mx.array([[True, True]]),
            "target_remaining": mx.array([4.0]),
        },
    )()
    loss, gradients = nn.value_and_grad(model, rollout_value_loss)(model, batch)
    optimizer.update(model, gradients)
    mx.eval(model.parameters(), optimizer.state, loss)
    state = TrainerState(
        epoch=2,
        batch_in_epoch=7,
        global_step=19,
        best_validation_loss=1.25,
        best_validation_rmse=2.5,
        value_epochs_without_improvement=1,
    )
    save_checkpoint(tmp_path, model, optimizer, state)

    loaded, loaded_optimizer, loaded_state, _ = load_latest_checkpoint_with_factory(
        tmp_path,
        learning_rate=3e-6,
        weight_decay=0.0,
        model_factory=lambda values: RolloutValueNnue(RolloutValueNnueConfig.from_dict(values)),
    )
    assert loaded_state == state
    for name, value in dict(model.parameters()).items():
        assert np.array_equal(np.asarray(value), np.asarray(dict(loaded.parameters())[name]))
    assert loaded_optimizer.state


def test_rollout_value_checkpoint_retention_preserves_best_latest_and_recent(
    tmp_path: Path,
) -> None:
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    names = [f"step-{step:09d}" for step in range(1, 5)]
    for name in names:
        path = checkpoints / name
        path.mkdir()
        (path / "checkpoint.json").write_text("{}")
    (tmp_path / "best.json").write_text(json.dumps({"checkpoint": names[0]}))
    (tmp_path / "latest.json").write_text(json.dumps({"checkpoint": names[-1]}))

    prune_checkpoints(tmp_path, keep_recent=2)

    assert sorted(path.name for path in checkpoints.iterdir()) == [
        names[0],
        names[2],
        names[3],
    ]


def test_validation_gates_apply_preregistered_thresholds() -> None:
    train = type("Dataset", (), {"trajectory_count": 100_000})()
    validation = type("Dataset", (), {"trajectory_count": 40_000})()
    parent_trajectory = {
        "rmse": 10.0,
        "pearson": 0.40,
        "turn_residual_pearson": 0.40,
        "turn_quartile_rmse": [10.0, 9.0, 8.0, 7.0],
    }
    selected_trajectory = {
        "rmse": 9.7,
        "pearson": 0.43,
        "turn_residual_pearson": 0.43,
        "turn_quartile_rmse": [10.05, 9.0, 7.9, 6.9],
    }
    parent_root = {
        "pairwise_accuracy": 0.60,
        "selected_action_top1": 0.20,
        "conditional_mean_regret": 2.0,
    }
    selected_root = {
        "pairwise_accuracy": 0.61,
        "selected_action_top1": 0.21,
        "conditional_mean_regret": 1.9,
    }
    gates = _validation_gates(
        train,
        validation,
        parent_trajectory,
        selected_trajectory,
        parent_root,
        selected_root,
        True,
    )
    assert gates["passed"]

    selected_trajectory["turn_quartile_rmse"][0] = 10.11
    assert not _validation_gates(
        train,
        validation,
        parent_trajectory,
        selected_trajectory,
        parent_root,
        selected_root,
        True,
    )["passed"]
