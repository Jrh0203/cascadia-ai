from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

import blake3
import cascadia_mlx.r2_map_verify as standalone_verifier
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest
from cascadia_mlx.checkpoint import (
    R2_MAP_POINTER_NAMES,
    R2_MAP_WRITE_STAGES,
    CheckpointError,
    R2MapCheckpointBundle,
    R2MapCheckpointIdentity,
    R2MapResumeState,
    build_r2_map_checkpoint_bundle,
    load_r2_map_checkpoint_bundle,
    load_r2_map_checkpoint_pointer,
    loss_stream_binding,
    prune_r2_map_checkpoints,
    resolve_r2_map_checkpoint_pointer,
    save_r2_map_checkpoint,
    set_r2_map_checkpoint_pointer,
    verify_r2_map_checkpoint_bundle,
    verify_r2_map_checkpoint_files,
)
from cascadia_mlx.r2_map_model import (
    R2MapBatch,
    R2MapMarketDecisionBatch,
    R2MapModel,
    R2MapModelConfig,
    R2MapPublicState,
)
from cascadia_mlx.r2_map_remote_storage import canonical_json, content_sha256
from cascadia_mlx.r2_map_remote_training import John2RemoteCheckpointStore
from cascadia_mlx.r2_map_tensor_contract import (
    BOARD_SLOTS,
    BOARD_TOKEN_CAPACITY,
    GLOBAL_FEATURES,
    MARKET_FEATURES,
    PLAYER_FEATURES,
    TOKEN_FEATURES,
)
from cascadia_mlx.r2_map_train import (
    PRIMARY_VALIDATION_METRIC,
    R2MapAdapterStep,
    R2MapSupervisedBatch,
    R2MapTrainer,
    R2MapTrainerConfig,
    _bootstrap_policy_loss,
    _chunked_bootstrap_policy_value_and_grad,
    append_loss_record,
    project_conflicting_auxiliary_gradients,
    r2_map_loss_components,
    select_best_validation_checkpoint,
    select_best_validation_checkpoint_bundle,
    validate_loss_stream,
    validate_loss_stream_bytes,
)
from cascadia_mlx.r2_map_training_contract import R2MapMarketDecisionSupervision
from cascadia_mlx.r2_map_verify import (
    compare_r2_map_checkpoint_tensors,
    prediction_panel,
    validate_verification_receipt,
    verify_integrity_only,
    verify_r2_map_checkpoint,
    verify_r2_map_checkpoint_bundle_in_memory,
)
from mlx.utils import tree_flatten

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
PROTOCOL = "synthetic-r2-map-adapter-v1"


def _public_state(*, index: int, candidates: int | None = None) -> R2MapPublicState:
    leading = (1,) if candidates is None else (1, candidates)
    token_shape = (*leading, BOARD_SLOTS, BOARD_TOKEN_CAPACITY)
    features = np.zeros((*token_shape, TOKEN_FEATURES), dtype=np.float32)
    mask = np.zeros(token_shape, dtype=np.bool_)
    mask[..., :4] = True
    features[..., :4, :] = (index + 1) / 10.0
    token_types = np.zeros(token_shape, dtype=np.int32)
    token_types[..., :4] = np.asarray([1, 2, 3, 4], dtype=np.int32)
    return R2MapPublicState(
        token_features=mx.array(features),
        token_types=mx.array(token_types),
        token_mask=mx.array(mask),
        market_features=mx.full((*leading, 4, MARKET_FEATURES), index / 20.0),
        market_mask=mx.ones((*leading, 4), dtype=mx.bool_),
        player_features=mx.full((*leading, BOARD_SLOTS, PLAYER_FEATURES), index / 30.0),
        player_mask=mx.ones((*leading, BOARD_SLOTS), dtype=mx.bool_),
        global_features=mx.full((*leading, GLOBAL_FEATURES), index / 40.0),
    )


def _supervised_batch(index: int) -> R2MapSupervisedBatch:
    selected = index % 2
    score_mask = np.zeros((1, 2), dtype=np.bool_)
    score_mask[0, selected] = True
    action_features = np.full((1, 2, 140), index / 100.0, dtype=np.float32)
    action_features[0, 1, 0] += 1.0
    inputs = R2MapBatch(
        parent=_public_state(index=index),
        candidates=_public_state(index=index + 1, candidates=2),
        candidate_mask=mx.ones((1, 2), dtype=mx.bool_),
        action_features=mx.array(action_features),
        exact_afterstate_scores=mx.array([[20.0 + index, 21.0 + index]]),
    )
    return R2MapSupervisedBatch(
        inputs=inputs,
        score_to_go_targets=mx.array([[50.0 - index, 49.0 - index]]),
        score_component_targets=mx.full((1, 2, 11), 3.0 + index / 10.0),
        score_target_mask=mx.array(score_mask),
        selected_action_index=mx.array([selected], dtype=mx.int32),
        bootstrap_policy_mask=mx.array([True]),
        opponent_tile_slot_targets=mx.array([[0, 1, 2]], dtype=mx.int32),
        opponent_wildlife_slot_targets=mx.array([[1, 2, 3]], dtype=mx.int32),
        opponent_draft_kind_targets=mx.array([[0, 1, 0]], dtype=mx.int32),
        opponent_drafted_wildlife_targets=mx.array([[0, 2, 4]], dtype=mx.int32),
        opponent_replace_three_targets=mx.array([[0, 0, 1]], dtype=mx.int32),
        opponent_paid_wipe_count_targets=mx.array([[0, 1, 2]], dtype=mx.int32),
        opponent_paid_wipe_mask_targets=mx.zeros((1, 3, 20), dtype=mx.int32),
        opponent_paid_wipe_mask_valid=mx.zeros((1, 3, 20), dtype=mx.bool_),
        opponent_valid_mask=mx.ones((1, 3), dtype=mx.bool_),
        market_disposition_targets=mx.array([[0, 1, 2, 3]], dtype=mx.int32),
        market_pair_survival_targets=mx.array([[1, 0, 1, 0]], dtype=mx.int32),
        market_final_slot_targets=mx.array([[0, 1, 2, 3]], dtype=mx.int32),
        market_disposition_mask=mx.ones((1, 4), dtype=mx.bool_),
        market_pair_survival_mask=mx.ones((1, 4), dtype=mx.bool_),
        market_final_slot_mask=mx.ones((1, 4), dtype=mx.bool_),
        batch_identity=f"synthetic-batch-{index:04d}",
    )


class SyntheticAdapter:
    protocol_id = PROTOCOL
    dataset_blake3 = DIGEST_B
    group_batch_size = 2
    maximum_candidates_per_batch = 16_384
    dataset_contract: ClassVar[dict[str, Any]] = {
        "schema_version": 1,
        "dataset_blake3": DIGEST_B,
        "d6_schema": "synthetic-d6-v1",
        "d6_cycle_epochs": 12,
        "imitation_subset_schema": "synthetic-imitation-v1",
        "imitation_subset_parts_per_million": 10_000,
        "collection_kind": "synthetic",
        "example_count": 1_000,
        "imitation_example_count": 10,
        "market_decision_count": 1_000,
        "market_policy_target_count": 10,
    }

    def initial_state(self, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
        return {"batch_index": 0}, {"seed": seed, "draw_index": 0}

    def training_batch(
        self, cursor: dict[str, Any], sampler_state: dict[str, Any]
    ) -> R2MapAdapterStep:
        index = int(cursor["batch_index"])
        assert sampler_state["draw_index"] == index
        return R2MapAdapterStep(
            batch=_supervised_batch(index),
            next_cursor={"batch_index": index + 1},
            next_sampler_state={**sampler_state, "draw_index": index + 1},
        )

    def validation_batches(self) -> tuple[R2MapSupervisedBatch, ...]:
        return (_supervised_batch(100), _supervised_batch(101))

    def fixed_prediction_batch(self, panel_id: str) -> R2MapBatch:
        assert panel_id == "r2-map-fixed-panel-v1"
        return _supervised_batch(999).inputs


class SyntheticPackedValueOnlyAdapter(SyntheticAdapter):
    protocol_id = "r2-map-focal-seat-bootstrap-value-pipe-adapter-v2"
    dataset_contract: ClassVar[dict[str, Any]] = {
        **SyntheticAdapter.dataset_contract,
        "collection_kind": "bootstrap",
        "market_policy_target_count": SyntheticAdapter.dataset_contract[
            "market_decision_count"
        ],
        "adapter_contract_schema_version": 2,
        "adapter_protocol_id": protocol_id,
        "pipe_protocol_id": "r2-map-packed-batch-pipe-v1",
        "focal_seat_rule": "global-game-index-mod-4",
        "bootstrap_games": 50,
        "bootstrap_focal_examples": 1_000,
        "one_epoch_steps": 500,
        "one_epoch_plan_blake3": DIGEST_A,
        "expanded_window_files": False,
        "bootstrap_objective": "selected-value-only-v1",
        "bootstrap_policy_loss_weight": 0.0,
    }

    def training_batch(
        self, cursor: dict[str, Any], sampler_state: dict[str, Any]
    ) -> R2MapAdapterStep:
        step = super().training_batch(cursor, sampler_state)
        return replace(
            step,
            batch=replace(step.batch, bootstrap_policy_mask=mx.array([False])),
        )


def _config(run_dir: Path, *, branch: str = "main") -> R2MapTrainerConfig:
    return R2MapTrainerConfig(
        run_dir=run_dir,
        run_id="unit-run",
        branch_id=branch,
        source_blake3=DIGEST_A,
        dataset_blake3=DIGEST_B,
        adapter_protocol_id=PROTOCOL,
        warmup_steps=1,
        schedule_steps=10,
        loss_event_interval_steps=1,
        seed=42,
        panel_id="r2-map-fixed-panel-v1",
    )


def _packed_value_config(run_dir: Path) -> R2MapTrainerConfig:
    return replace(
        _config(run_dir),
        adapter_protocol_id=SyntheticPackedValueOnlyAdapter.protocol_id,
        auxiliary_loss_weights={
            "components": 0.25,
            "bootstrap_policy": 0.0,
            "opponent_next_action": 0.05,
            "market_survival": 0.05,
            "market_decision_policy": 0.10,
        },
    )


def _tensor_bytes(value: Any) -> dict[str, bytes]:
    return {name: np.asarray(array).tobytes(order="C") for name, array in tree_flatten(value)}


def _market_supervision(index: int = 0) -> R2MapMarketDecisionSupervision:
    return R2MapMarketDecisionSupervision(
        inputs=R2MapMarketDecisionBatch(
            public_state=_public_state(index=index),
            action_mask=mx.ones((1, 2), dtype=mx.bool_),
            action_features=mx.array(np.arange(32, dtype=np.float32).reshape(1, 2, 16) / 10.0),
            exact_current_scores=mx.array([25.0], dtype=mx.float32),
        ),
        score_to_go_targets=mx.array([[0.0, 40.0]], dtype=mx.float32),
        score_target_mask=mx.array([[False, True]]),
        selected_action_index=mx.array([1], dtype=mx.int32),
        policy_target_mask=mx.array([True]),
        batch_identity="synthetic-market-decision",
    )


def test_draft_and_market_imitation_losses_are_real_and_explicitly_masked(
    tmp_path: Path,
) -> None:
    model = R2MapModel()
    batch = replace(_supervised_batch(0), market_decisions=_market_supervision())
    losses = r2_map_loss_components(
        model,
        batch,
        normalization=_config(tmp_path).normalized(),
    )
    mx.eval(losses)
    assert float(losses["bootstrap_policy"].item()) == pytest.approx(np.log(2.0))
    assert float(losses["market_decision_policy"].item()) == pytest.approx(np.log(2.0))

    masked_market = replace(_market_supervision(), policy_target_mask=mx.array([False]))
    masked = replace(
        batch,
        bootstrap_policy_mask=mx.array([False]),
        market_decisions=masked_market,
    )
    masked_losses = r2_map_loss_components(
        model,
        masked,
        normalization=_config(tmp_path).normalized(),
    )
    mx.eval(masked_losses)
    assert float(masked_losses["bootstrap_policy"].item()) == 0.0
    assert float(masked_losses["market_decision_policy"].item()) == 0.0

    value_and_grad = nn.value_and_grad(
        model,
        lambda candidate, supervised: (
            r2_map_loss_components(
                candidate,
                supervised,
                normalization=_config(tmp_path).normalized(),
            )["bootstrap_policy"]
            + r2_map_loss_components(
                candidate,
                supervised,
                normalization=_config(tmp_path).normalized(),
            )["market_decision_policy"]
        ),
    )
    _, gradients = value_and_grad(model, batch)
    flattened = dict(tree_flatten(gradients))
    for head in ("bootstrap_policy_head", "market_decision_bootstrap_policy_head"):
        head_gradients = [
            (name.rsplit(".", maxsplit=1)[-1], np.asarray(value))
            for name, value in flattened.items()
            if head in name.split(".")
        ]
        assert [name for name, _ in head_gradients] == ["weight"]
        assert np.any(head_gradients[0][1] != 0.0)


def test_chunked_bootstrap_policy_value_and_gradient_are_exact() -> None:
    mx.random.seed(771)
    model = R2MapModel()
    batch = _supervised_batch(0)
    direct_value_and_grad = nn.value_and_grad(
        model,
        lambda candidate, supervised: _bootstrap_policy_loss(candidate, supervised),
    )
    direct_value, direct_gradients = direct_value_and_grad(model, batch)
    chunked_value, chunked_gradients = _chunked_bootstrap_policy_value_and_grad(model, batch)
    mx.eval(direct_value, direct_gradients, chunked_value, chunked_gradients)
    assert float(chunked_value.item()) == pytest.approx(float(direct_value.item()), abs=1e-6)
    direct = dict(tree_flatten(direct_gradients))
    chunked = dict(tree_flatten(chunked_gradients))
    assert direct.keys() == chunked.keys()
    for name in direct:
        np.testing.assert_allclose(
            np.asarray(chunked[name]),
            np.asarray(direct[name]),
            rtol=2e-5,
            atol=2e-6,
            err_msg=name,
        )


def test_exact_save_reload_and_next_batch_resume(tmp_path: Path) -> None:
    adapter = SyntheticAdapter()
    trainer = R2MapTrainer(_config(tmp_path), adapter)
    first_record = trainer.step()
    expected_next_batch = trainer.peek_next_batch_identity()
    validation = trainer.validation_metrics()
    checkpoint = trainer.save_checkpoint(validation=validation)
    expected_rng = mx.random.normal((8,))
    mx.eval(expected_rng)
    receipt = verify_r2_map_checkpoint(
        checkpoint,
        run_dir=tmp_path,
        adapter=adapter,
        mark_last_verified=True,
    )
    resumed = R2MapTrainer.resume(_config(tmp_path), adapter)
    actual_rng = mx.random.normal((8,))
    mx.eval(actual_rng)

    assert expected_next_batch == "synthetic-batch-0001"
    assert resumed.peek_next_batch_identity() == expected_next_batch
    assert resumed.cursor == {"batch_index": 1}
    assert resumed.sampler_state["draw_index"] == 1
    assert resumed.global_step == 1
    assert (
        resumed.training_counters
        == trainer.training_counters
        == {
            "draft_groups": 1,
            "draft_candidates": 2,
            "padded_draft_candidates": 2,
            "draft_policy_targets": 1,
            "market_groups": 0,
            "market_actions": 0,
            "market_policy_targets": 0,
        }
    )
    assert receipt["exact_prediction_match"] is True
    assert receipt["exact_next_batch_match"] is True
    assert receipt["next_batch_identity"] == expected_next_batch
    assert first_record["record_blake3"] == resumed.loss_head
    assert any(key.endswith("/cosine") for key in first_record["metrics"])
    np.testing.assert_array_equal(np.asarray(actual_rng), np.asarray(expected_rng))
    assert resolve_r2_map_checkpoint_pointer(tmp_path, "latest_complete") == checkpoint
    assert resolve_r2_map_checkpoint_pointer(tmp_path, "last_verified") == checkpoint
    assert not (tmp_path / "best_validation.json").exists()


def test_verification_and_resume_reject_next_batch_identity_drift(tmp_path: Path) -> None:
    adapter = SyntheticAdapter()
    trainer = R2MapTrainer(_config(tmp_path), adapter)
    trainer.step()
    checkpoint = trainer.save_checkpoint()

    class DriftedAdapter(SyntheticAdapter):
        def training_batch(
            self, cursor: dict[str, Any], sampler_state: dict[str, Any]
        ) -> R2MapAdapterStep:
            step = super().training_batch(cursor, sampler_state)
            return replace(
                step,
                batch=replace(step.batch, batch_identity=f"drifted-{step.batch.batch_identity}"),
            )

    drifted = DriftedAdapter()
    with pytest.raises(CheckpointError, match="exact next batch"):
        verify_r2_map_checkpoint(checkpoint, run_dir=tmp_path, adapter=drifted)
    verify_r2_map_checkpoint(
        checkpoint,
        run_dir=tmp_path,
        adapter=adapter,
        mark_last_verified=True,
    )
    with pytest.raises(CheckpointError, match="exact next batch"):
        R2MapTrainer.resume(_config(tmp_path), drifted)

    class ContractDriftAdapter(SyntheticAdapter):
        dataset_contract: ClassVar[dict[str, Any]] = {
            **SyntheticAdapter.dataset_contract,
            "imitation_example_count": 11,
        }

    with pytest.raises(CheckpointError, match="dataset contract differs"):
        R2MapTrainer.resume(_config(tmp_path), ContractDriftAdapter())

    with pytest.raises(ValueError, match="batch-packing contract differs"):
        R2MapTrainer(
            replace(_config(tmp_path / "packing-drift"), group_batch_size=4),
            SyntheticAdapter(),
        )


def test_packed_value_only_checkpoint_round_trip_resume_and_policy_tamper_rejection(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "value-only"
    adapter = SyntheticPackedValueOnlyAdapter()
    config = _packed_value_config(run_dir)
    trainer = R2MapTrainer(config, adapter)
    trainer.step()
    checkpoint = trainer.save_checkpoint()
    verification = verify_r2_map_checkpoint(
        checkpoint,
        run_dir=run_dir,
        adapter=adapter,
        mark_last_verified=True,
    )
    assert verification["exact_prediction_match"] is True
    assert verification["exact_next_batch_match"] is True

    loaded = load_r2_map_checkpoint_pointer(
        run_dir,
        "last_verified",
        model_factory=lambda values: R2MapModel(R2MapModelConfig.from_dict(values)),
        optimizer_factory=lambda: optim.AdamW(learning_rate=3e-5, weight_decay=1e-4),
        loss_stream_path=trainer.loss_path,
    )
    assert loaded.state.dataset_contract["adapter_contract_schema_version"] == 2
    assert loaded.state.dataset_contract["bootstrap_objective"] == "selected-value-only-v1"
    assert loaded.state.dataset_contract["bootstrap_policy_loss_weight"] == 0.0
    assert loaded.state.auxiliary_loss_weights["bootstrap_policy"] == 0.0

    class PolicyModeDriftAdapter(SyntheticPackedValueOnlyAdapter):
        dataset_contract: ClassVar[dict[str, Any]] = {
            **SyntheticPackedValueOnlyAdapter.dataset_contract,
            "bootstrap_objective": "value-plus-greedy-imitation-v1",
            "bootstrap_policy_loss_weight": 0.1,
        }

    with pytest.raises(CheckpointError, match="resume dataset contract"):
        R2MapTrainer.resume(config, PolicyModeDriftAdapter(), pointer="last_verified")

    resumed = R2MapTrainer.resume(config, adapter, pointer="last_verified")
    resumed.step()
    assert resumed.global_step == 2

    drift_dir = tmp_path / "policy-mode-drift"
    drift_trainer = R2MapTrainer(
        _packed_value_config(drift_dir),
        PolicyModeDriftAdapter(),
    )
    drift_trainer.step()
    with pytest.raises(CheckpointError, match="packed value-only dataset contract"):
        drift_trainer.save_checkpoint()


def test_interrupted_and_uninterrupted_second_step_are_byte_exact(tmp_path: Path) -> None:
    adapter = SyntheticAdapter()
    uninterrupted = R2MapTrainer(_config(tmp_path / "continuous"), adapter)
    uninterrupted.step()
    uninterrupted.step()
    continuous_model = _tensor_bytes(uninterrupted.model.trainable_parameters())
    continuous_optimizer = _tensor_bytes(uninterrupted.optimizer.state)

    interrupted_dir = tmp_path / "interrupted"
    interrupted = R2MapTrainer(_config(interrupted_dir), adapter)
    interrupted.step()
    checkpoint = interrupted.save_checkpoint(validation=interrupted.validation_metrics())
    verify_r2_map_checkpoint(
        checkpoint,
        run_dir=interrupted_dir,
        adapter=adapter,
        mark_last_verified=True,
    )
    resumed = R2MapTrainer.resume(_config(interrupted_dir), adapter)
    resumed.step()

    assert _tensor_bytes(resumed.model.trainable_parameters()) == continuous_model
    assert _tensor_bytes(resumed.optimizer.state) == continuous_optimizer
    assert resumed.cursor == uninterrupted.cursor == {"batch_index": 2}
    assert resumed.sampler_state == uninterrupted.sampler_state
    assert (
        resumed.training_counters
        == uninterrupted.training_counters
        == {
            "draft_groups": 2,
            "draft_candidates": 4,
            "padded_draft_candidates": 4,
            "draft_policy_targets": 2,
            "market_groups": 0,
            "market_actions": 0,
            "market_policy_targets": 0,
        }
    )
    assert resumed.peek_next_batch_identity() == uninterrupted.peek_next_batch_identity()
    continuous_losses = validate_loss_stream(uninterrupted.loss_path)
    resumed_losses = validate_loss_stream(resumed.loss_path)
    assert [record["metrics"] for record in resumed_losses] == [
        record["metrics"] for record in continuous_losses
    ]

    continuous_checkpoint = uninterrupted.save_checkpoint(
        validation=uninterrupted.validation_metrics()
    )
    resumed_checkpoint = resumed.save_checkpoint(validation=resumed.validation_metrics())
    parity = compare_r2_map_checkpoint_tensors(continuous_checkpoint, resumed_checkpoint)
    assert parity["exact_match"] is True
    assert parity["bundles"]["model.safetensors"]["exact_match"] is True
    assert parity["bundles"]["optimizer.safetensors"]["exact_match"] is True


def test_in_memory_training_resume_is_exact_and_creates_no_local_run_tree(
    tmp_path: Path,
) -> None:
    adapter = SyntheticAdapter()
    run_dir = tmp_path / "forbidden-local-run"
    trainer = R2MapTrainer(_config(run_dir), adapter, in_memory=True)
    trainer.step()
    validation = trainer.validation_metrics()
    bundle = trainer.checkpoint_bundle(validation=validation)
    loss_content = trainer.loss_content
    receipt = verify_r2_map_checkpoint_bundle_in_memory(
        bundle,
        loss_content=loss_content,
        adapter=adapter,
    )
    expected_next = trainer.peek_next_batch_identity()
    assert not run_dir.exists()
    assert receipt["exact_prediction_match"] is True

    resumed = R2MapTrainer.resume_from_bundle(
        _config(run_dir),
        adapter,
        bundle=bundle,
        loss_content=loss_content,
    )
    assert resumed.peek_next_batch_identity() == expected_next
    assert resumed.loss_content == loss_content
    assert resumed.loss_path is None
    assert resumed.training_counters == trainer.training_counters
    assert not run_dir.exists()
    resumed.step()
    assert len(validate_loss_stream_bytes(resumed.loss_content)) == 2
    assert not run_dir.exists()


class _CheckpointMemoryClient:
    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, str]] = {}
        self.transactions: dict[str, dict[str, Any]] = {}
        self.receipt_sequence = 0

    def _receipt(self) -> dict[str, str]:
        self.receipt_sequence += 1
        identity = f"memory-{self.receipt_sequence}"
        return {
            "storage_receipt_relative": f"control/receipts/req-{identity}.json",
            "storage_receipt_sha256": hashlib.sha256(identity.encode()).hexdigest(),
        }

    def put_bytes(
        self,
        relative: str,
        payload: bytes,
        *,
        expected_current: str = "absent",
        mutable: bool = False,
    ) -> dict[str, Any]:
        current = self.objects.get(relative)
        current_sha256 = None if current is None else content_sha256(current[0])
        if (expected_current == "absent" and current is not None) or (
            expected_current != "absent" and expected_current != current_sha256
        ):
            raise RuntimeError("memory CAS differs")
        mode = "0o600" if mutable else "0o400"
        self.objects[relative] = (bytes(payload), mode)
        return {
            "relative": relative,
            "sha256": content_sha256(payload),
            "size": len(payload),
            "mode": mode,
            "previous_sha256": current_sha256,
            **self._receipt(),
        }

    def begin_transaction(self, manifest: dict[str, Any]) -> dict[str, Any]:
        transaction_id = manifest["transaction_id"]
        self.transactions[transaction_id] = {"manifest": manifest, "objects": {}}
        return {
            "transaction_id": transaction_id,
            "target_relative": manifest["target_relative"],
            "manifest_sha256": manifest["manifest_sha256"],
            **self._receipt(),
        }

    def put_transaction_object(
        self,
        transaction_id: str,
        descriptor: Any,
        chunks: Any,
    ) -> dict[str, Any]:
        payload = b"".join(chunks)
        assert len(payload) == descriptor.size
        assert content_sha256(payload) == descriptor.sha256
        self.transactions[transaction_id]["objects"][descriptor.relative] = payload
        return {**descriptor.to_dict(), **self._receipt()}

    def commit_transaction(self, transaction_id: str, manifest_sha256: str) -> dict[str, Any]:
        transaction = self.transactions[transaction_id]
        manifest = transaction["manifest"]
        assert manifest["manifest_sha256"] == manifest_sha256
        target = manifest["target_relative"]
        for descriptor in manifest["objects"]:
            payload = transaction["objects"][descriptor["relative"]]
            mode = "0o500" if descriptor.get("mode") == "0500" else "0o400"
            self.objects[f"{target}/{descriptor['relative']}"] = (payload, mode)
        self.objects[f"{target}/.r2-map-transaction.json"] = (
            canonical_json(manifest),
            "0o400",
        )
        return {
            "transaction_id": transaction_id,
            "target_relative": target,
            "manifest_sha256": manifest_sha256,
            "object_count": len(manifest["objects"]),
            "committed": True,
            **self._receipt(),
        }

    def abort_transaction(self, transaction_id: str, manifest_sha256: str) -> dict[str, Any]:
        self.transactions.pop(transaction_id, None)
        return {
            "transaction_id": transaction_id,
            "manifest_sha256": manifest_sha256,
            "aborted": True,
            **self._receipt(),
        }

    def open_object_with_receipt(self, relative: str) -> dict[str, Any]:
        payload, mode = self.objects[relative]
        digest = content_sha256(payload)
        return {
            "object_token": {
                "schema_version": 1,
                "schema_id": "cascadia.r2-map.remote-object-token.v1",
                "relative": relative,
                "sha256": digest,
                "size": len(payload),
                "mode": int(mode, 8),
                "token_sha256": hashlib.sha256((relative + digest).encode()).hexdigest(),
            },
            **self._receipt(),
        }

    def iter_object_with_receipts(self, token: dict[str, Any], *, window_bytes: int):
        payload, _ = self.objects[token["relative"]]
        for offset in range(0, len(payload), window_bytes):
            chunk = payload[offset : offset + window_bytes]
            yield {
                "payload": chunk,
                "payload_sha256": content_sha256(chunk),
                "object_token_sha256": token["token_sha256"],
                "offset": offset,
                "length": len(chunk),
                **self._receipt(),
            }


def test_remote_checkpoint_transaction_resume_is_exact_and_filesystem_free(
    tmp_path: Path,
) -> None:
    adapter = SyntheticAdapter()
    run_dir = tmp_path / "forbidden-remote-training-tree"
    trainer = R2MapTrainer(_config(run_dir), adapter, in_memory=True)
    trainer.step()
    bundle = trainer.checkpoint_bundle(validation=trainer.validation_metrics())
    verification = verify_r2_map_checkpoint_bundle_in_memory(
        bundle,
        loss_content=trainer.loss_content,
        adapter=adapter,
    )
    expected_next = trainer.peek_next_batch_identity()
    client = _CheckpointMemoryClient()
    store = John2RemoteCheckpointStore(client, run_id="unit-run")  # type: ignore[arg-type]
    publication = store.publish_checkpoint(
        bundle,
        loss_content=trainer.loss_content,
        verification_receipt=verification,
    )
    artifact = publication.work_artifact(bundle)
    assert artifact["path"].endswith("/checkpoint.json")
    assert (
        artifact["storage_receipt_sha256"]
        == publication.transaction_commit["storage_receipt_sha256"]
    )
    assert [value[1] for key, value in client.objects.items() if key.endswith(".json")]
    assert not run_dir.exists()

    loaded = store.load_checkpoint("last_verified")
    resumed = R2MapTrainer.resume_from_bundle(
        _config(run_dir),
        adapter,
        bundle=loaded.bundle,
        loss_content=loaded.loss_content,
    )
    assert resumed.peek_next_batch_identity() == expected_next
    assert resumed.loss_content == trainer.loss_content
    assert publication.remote_objects
    assert not run_dir.exists()

    verification_relative = f"runs/unit-run/verifications/{bundle.checkpoint_id}.json"
    verification_payload, mode = client.objects[verification_relative]
    tampered = json.loads(verification_payload)
    tampered["exact_prediction_match"] = False
    client.objects[verification_relative] = (canonical_json(tampered) + b"\n", mode)
    with pytest.raises(CheckpointError, match="verification receipt identity"):
        store.load_checkpoint("last_verified")


def test_in_memory_validation_selection_is_order_independent_and_prefers_lower_step(
    tmp_path: Path,
) -> None:
    adapter = SyntheticAdapter()
    run_dir = tmp_path / "forbidden-local-selection"
    trainer = R2MapTrainer(_config(run_dir), adapter, in_memory=True)
    trainer.step()
    first = trainer.checkpoint_bundle(validation={PRIMARY_VALIDATION_METRIC: 0.25})
    first_receipt = verify_r2_map_checkpoint_bundle_in_memory(
        first,
        loss_content=trainer.loss_content,
        adapter=adapter,
    )
    trainer.step()
    second = trainer.checkpoint_bundle(validation={PRIMARY_VALIDATION_METRIC: 0.25})
    second_receipt = verify_r2_map_checkpoint_bundle_in_memory(
        second,
        loss_content=trainer.loss_content,
        adapter=adapter,
    )
    assert (
        select_best_validation_checkpoint_bundle([(first, first_receipt), (second, second_receipt)])
        == first
    )
    assert (
        select_best_validation_checkpoint_bundle([(second, second_receipt), (first, first_receipt)])
        == first
    )
    assert not run_dir.exists()


def test_checkpoint_identity_tamper_and_duplicate_id_are_rejected(tmp_path: Path) -> None:
    adapter = SyntheticAdapter()
    trainer = R2MapTrainer(_config(tmp_path), adapter)
    trainer.step()
    checkpoint = trainer.save_checkpoint()
    with pytest.raises(CheckpointError, match="already exists"):
        trainer.save_checkpoint()
    with pytest.raises(CheckpointError, match="dataset_blake3"):
        verify_r2_map_checkpoint_files(checkpoint, expected_identity={"dataset_blake3": DIGEST_A})
    manifest_path = checkpoint / "checkpoint.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["identity"]["source_blake3"] = DIGEST_B
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(CheckpointError, match="manifest identity hash"):
        verify_r2_map_checkpoint_files(checkpoint)


def test_pointer_roles_are_distinct_and_best_selection_uses_only_validation(
    tmp_path: Path,
) -> None:
    adapter = SyntheticAdapter()
    trainer = R2MapTrainer(_config(tmp_path), adapter)
    trainer.step()
    checkpoint = trainer.save_checkpoint(validation={PRIMARY_VALIDATION_METRIC: 0.25})
    assert (tmp_path / "latest_complete.json").exists()
    assert all(
        not (tmp_path / f"{name}.json").exists()
        for name in set(R2_MAP_POINTER_NAMES) - {"latest_complete"}
    )
    verify_r2_map_checkpoint(
        checkpoint,
        run_dir=tmp_path,
        adapter=adapter,
        mark_last_verified=True,
    )
    set_r2_map_checkpoint_pointer(tmp_path, "incumbent", checkpoint)
    set_r2_map_checkpoint_pointer(tmp_path, "promoted", checkpoint)
    selected = select_best_validation_checkpoint(tmp_path)
    assert selected == checkpoint
    for name in R2_MAP_POINTER_NAMES:
        assert resolve_r2_map_checkpoint_pointer(tmp_path, name) == checkpoint
    with pytest.raises(CheckpointError, match="unknown"):
        set_r2_map_checkpoint_pointer(tmp_path, "best_training_loss", checkpoint)


def test_validation_selection_is_order_independent_and_uses_primary_metric_only(
    tmp_path: Path,
) -> None:
    adapter = SyntheticAdapter()
    trainer = R2MapTrainer(_config(tmp_path), adapter)
    trainer.step()
    first = trainer.save_checkpoint(
        validation={PRIMARY_VALIDATION_METRIC: 0.25, "component_validation_loss": 999.0}
    )
    verify_r2_map_checkpoint(first, run_dir=tmp_path, adapter=adapter)
    trainer.step()
    second = trainer.save_checkpoint(
        validation={PRIMARY_VALIDATION_METRIC: 0.25, "component_validation_loss": 0.0}
    )
    verify_r2_map_checkpoint(second, run_dir=tmp_path, adapter=adapter)
    forward = select_best_validation_checkpoint(tmp_path, checkpoint_paths=[first, second])
    reverse = select_best_validation_checkpoint(tmp_path, checkpoint_paths=[second, first])
    assert forward == reverse == first
    receipt_path = tmp_path / "verifications" / f"{first.name}.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["exact_prediction_match"] = False
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(CheckpointError, match="receipt identity"):
        validate_verification_receipt(receipt_path, checkpoint_path=first)


def test_best_validation_selection_skips_verified_recovery_checkpoint(
    tmp_path: Path,
) -> None:
    adapter = SyntheticAdapter()
    trainer = R2MapTrainer(_config(tmp_path), adapter)
    trainer.step()
    recovery = trainer.save_checkpoint(validation=None)
    verify_r2_map_checkpoint(recovery, run_dir=tmp_path, adapter=adapter)
    trainer.step()
    evaluated = trainer.save_checkpoint(validation={PRIMARY_VALIDATION_METRIC: 0.25})
    verify_r2_map_checkpoint(evaluated, run_dir=tmp_path, adapter=adapter)
    assert select_best_validation_checkpoint(tmp_path) == evaluated


def test_branch_aware_loss_stream_preserves_suffix_and_requires_new_branch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "losses.jsonl"
    first = append_loss_record(
        path,
        branch_id="main",
        global_step=1,
        batch_identity="batch-1",
        metrics={"loss": 1.0},
        parent_record_blake3=None,
    )
    append_loss_record(
        path,
        branch_id="main",
        global_step=2,
        batch_identity="batch-2",
        metrics={"loss": 0.9},
        parent_record_blake3=first["record_blake3"],
    )
    fork = append_loss_record(
        path,
        branch_id="recovery",
        global_step=2,
        batch_identity="batch-2-replay",
        metrics={"loss": 0.91},
        parent_record_blake3=first["record_blake3"],
    )
    records = validate_loss_stream(path)
    assert len(records) == 3
    assert fork["parent_record_blake3"] == first["record_blake3"]
    assert path.read_text().count("\n") == 3

    tampered = path.read_text().replace('"loss":0.9', '"loss":0.8')
    path.write_text(tampered)
    with pytest.raises(CheckpointError, match="hash differs"):
        validate_loss_stream(path)


def test_conflicting_auxiliary_gradient_is_projected_without_modifying_primary() -> None:
    primary = {
        "trunk": {"weight": mx.array([1.0, 0.0])},
        "auxiliary_head": {"weight": mx.array([0.0, 0.0])},
    }
    auxiliary = {
        "trunk": {"weight": mx.array([-1.0, 2.0])},
        "auxiliary_head": {"weight": mx.array([3.0, 4.0])},
    }
    projected, diagnostics = project_conflicting_auxiliary_gradients(primary, auxiliary)
    mx.eval(projected)
    np.testing.assert_array_equal(np.asarray(primary["trunk"]["weight"]), [1.0, 0.0])
    np.testing.assert_allclose(np.asarray(projected["trunk"]["weight"]), [0.0, 2.0])
    np.testing.assert_array_equal(np.asarray(projected["auxiliary_head"]["weight"]), [3.0, 4.0])
    assert diagnostics["trunk"]["projected"] is True
    assert diagnostics["auxiliary_head"]["projected"] is False

    aligned = {
        "trunk": {"weight": mx.array([1.0, 2.0])},
        "auxiliary_head": {"weight": mx.array([3.0, 4.0])},
    }
    unchanged, aligned_diagnostics = project_conflicting_auxiliary_gradients(primary, aligned)
    mx.eval(unchanged)
    np.testing.assert_array_equal(
        np.asarray(unchanged["trunk"]["weight"]),
        np.asarray(aligned["trunk"]["weight"]),
    )
    assert aligned_diagnostics["trunk"]["projected"] is False


def test_resume_with_post_checkpoint_suffix_requires_a_fresh_branch(tmp_path: Path) -> None:
    adapter = SyntheticAdapter()
    main = R2MapTrainer(_config(tmp_path), adapter)
    main.step()
    checkpoint = main.save_checkpoint()
    verify_r2_map_checkpoint(
        checkpoint,
        run_dir=tmp_path,
        adapter=adapter,
        mark_last_verified=True,
    )
    main.step()
    with pytest.raises(CheckpointError, match="fork a branch"):
        R2MapTrainer.resume(_config(tmp_path), adapter)

    recovery = R2MapTrainer.resume(_config(tmp_path, branch="recovery"), adapter)
    assert recovery.peek_next_batch_identity() == "synthetic-batch-0001"
    recovery_record = recovery.step()
    records = validate_loss_stream(recovery.loss_path)
    assert len(records) == 3
    assert recovery_record["branch_id"] == "recovery"
    assert recovery_record["parent_record_blake3"] == records[0]["record_blake3"]


def test_checkpoint_loss_prefix_allows_suffix_but_rejects_prefix_tamper(tmp_path: Path) -> None:
    adapter = SyntheticAdapter()
    trainer = R2MapTrainer(_config(tmp_path), adapter)
    trainer.step()
    checkpoint = trainer.save_checkpoint()
    original = trainer.loss_path.read_bytes()
    append_loss_record(
        trainer.loss_path,
        branch_id="main",
        global_step=2,
        batch_identity="suffix",
        metrics={"loss": 1.0},
        parent_record_blake3=trainer.loss_head,
    )
    verify_r2_map_checkpoint_files(checkpoint, loss_stream_path=trainer.loss_path)
    tampered = bytearray(trainer.loss_path.read_bytes())
    tampered[original.index(b"synthetic-batch")] ^= 1
    trainer.loss_path.write_bytes(tampered)
    with pytest.raises(CheckpointError, match="prefix differs"):
        verify_r2_map_checkpoint_files(checkpoint, loss_stream_path=trainer.loss_path)


class TinyCheckpointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)


def _direct_checkpoint_arguments(run_dir: Path, checkpoint_id: str) -> dict[str, Any]:
    loss_path = run_dir / "losses/loss-stream.jsonl"
    loss_path.parent.mkdir(parents=True, exist_ok=True)
    loss_path.touch()
    model_config = R2MapModelConfig().to_dict()
    config_hash = blake3.blake3(
        json.dumps(model_config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    model = TinyCheckpointModel()
    optimizer = optim.AdamW(learning_rate=1e-3)
    optimizer.init(model.trainable_parameters())
    state = R2MapResumeState(
        global_step=1,
        epoch=0,
        batch_in_epoch=1,
        examples_seen=1,
        cursor={"batch": 1},
        sampler_state={"draw": 1},
        rng_state={"schema_version": 1, "mlx_uint32_keys": [[0, 1]]},
        scheduler_state={"schema_version": 1, "next_step": 1},
        normalization={"score_scale": 100.0},
        auxiliary_loss_weights={"components": 0.25},
        dataset_contract=SyntheticAdapter.dataset_contract,
        training_counters={
            "draft_groups": 1,
            "draft_candidates": 2,
            "padded_draft_candidates": 2,
            "draft_policy_targets": 1,
            "market_groups": 0,
            "market_actions": 0,
            "market_policy_targets": 0,
        },
        loss_stream=loss_stream_binding(loss_path, relative_to=run_dir, head_record_blake3=None),
        next_batch_identity="direct-checkpoint-next-batch",
        validation=None,
    )
    return {
        "run_dir": run_dir,
        "model": model,
        "optimizer": optimizer,
        "identity": R2MapCheckpointIdentity(
            checkpoint_id=checkpoint_id,
            run_id="fault-run",
            branch_id="main",
            source_blake3=DIGEST_A,
            dataset_blake3=DIGEST_B,
            model_config_blake3=config_hash,
            training_config_blake3=DIGEST_A,
            loss_contract_blake3=DIGEST_B,
        ),
        "state": state,
        "model_config": model_config,
        "fixed_prediction_panel": {"probe": mx.array([[1.0]], dtype=mx.float32)},
        "prediction_panel_id": "fault-panel-v1",
    }


def test_resume_state_rejects_training_counter_algebra_drift(tmp_path: Path) -> None:
    state = _direct_checkpoint_arguments(tmp_path, "counter-algebra")["state"]
    state.validate()
    with pytest.raises(CheckpointError, match="counter algebra"):
        replace(
            state,
            training_counters={
                **state.training_counters,
                "padded_draft_candidates": 1,
            },
        ).validate()
    with pytest.raises(CheckpointError, match="counter algebra"):
        replace(state, examples_seen=2).validate()


@pytest.mark.parametrize("stage", R2_MAP_WRITE_STAGES)
def test_fault_at_every_checkpoint_write_stage_never_points_to_partial(
    tmp_path: Path, stage: str
) -> None:
    run_dir = tmp_path / stage
    baseline_arguments = _direct_checkpoint_arguments(run_dir, "baseline")
    baseline = save_r2_map_checkpoint(**baseline_arguments)
    candidate_arguments = _direct_checkpoint_arguments(run_dir, "candidate")

    def inject(observed: str) -> None:
        if observed == stage:
            raise RuntimeError(f"injected at {stage}")

    with pytest.raises(RuntimeError, match="injected"):
        save_r2_map_checkpoint(**candidate_arguments, fault_injector=inject)
    pointed = resolve_r2_map_checkpoint_pointer(run_dir, "latest_complete")
    if stage == "latest-pointer-committed":
        assert pointed.name == "candidate"
    else:
        assert pointed == baseline
    if (run_dir / "checkpoints/candidate").exists():
        verify_r2_map_checkpoint_files(run_dir / "checkpoints/candidate")
    assert not list((run_dir / "checkpoints").glob(".*.tmp"))


def test_in_memory_checkpoint_bundle_round_trip_and_tamper_rejection(tmp_path: Path) -> None:
    arguments = _direct_checkpoint_arguments(tmp_path, "memory-only")
    build_arguments = {key: value for key, value in arguments.items() if key != "run_dir"}
    before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
    bundle = build_r2_map_checkpoint_bundle(**build_arguments)
    manifest, state, panel = verify_r2_map_checkpoint_bundle(bundle, loss_stream=b"")
    loaded = load_r2_map_checkpoint_bundle(
        bundle,
        model_factory=lambda _values: TinyCheckpointModel(),
        optimizer_factory=lambda: optim.AdamW(learning_rate=1e-3),
        loss_stream=b"",
    )
    after = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
    assert before == after
    assert bundle.total_bytes == sum(len(value) for value in bundle.objects.values())
    assert manifest["checkpoint_id"] == "memory-only"
    assert state.global_step == 1
    assert set(panel) == {"probe"}
    assert _tensor_bytes(loaded.model.parameters()) == _tensor_bytes(
        arguments["model"].parameters()
    )
    assert _tensor_bytes(loaded.optimizer.state) == _tensor_bytes(arguments["optimizer"].state)

    corrupted = dict(bundle.objects)
    payload = bytearray(corrupted["model.safetensors"])
    payload[-1] ^= 1
    corrupted["model.safetensors"] = bytes(payload)
    with pytest.raises(CheckpointError, match="failed integrity"):
        verify_r2_map_checkpoint_bundle(
            R2MapCheckpointBundle(bundle.checkpoint_id, bundle.manifest, corrupted)
        )


def test_pruning_preserves_every_pointed_semantic_role(tmp_path: Path) -> None:
    run_dir = tmp_path / "prune"
    first = save_r2_map_checkpoint(**_direct_checkpoint_arguments(run_dir, "first"))
    second = save_r2_map_checkpoint(**_direct_checkpoint_arguments(run_dir, "second"))
    third = save_r2_map_checkpoint(**_direct_checkpoint_arguments(run_dir, "third"))
    set_r2_map_checkpoint_pointer(run_dir, "incumbent", first)
    set_r2_map_checkpoint_pointer(run_dir, "promoted", first)
    set_r2_map_checkpoint_pointer(run_dir, "last_verified", second)
    set_r2_map_checkpoint_pointer(run_dir, "latest_complete", second)
    prune_r2_map_checkpoints(run_dir, keep_recent=0)
    assert first.exists()
    assert second.exists()
    assert not third.exists()


def test_integrity_only_cannot_advance_last_verified(tmp_path: Path) -> None:
    adapter = SyntheticAdapter()
    trainer = R2MapTrainer(_config(tmp_path), adapter)
    trainer.step()
    checkpoint = trainer.save_checkpoint()
    result = verify_integrity_only(checkpoint, run_dir=tmp_path)
    assert result["fixed_prediction_recomputed"] is False
    assert not (tmp_path / "last_verified.json").exists()


def test_standalone_verifier_does_not_import_trainer() -> None:
    source = Path(standalone_verifier.__file__).read_text()
    modules = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.append(node.module)
    assert not any(
        module == "r2_map_train" or module.endswith(".r2_map_train") for module in modules
    )


def test_checkpoint_loader_restores_model_optimizer_scheduler_and_panel(tmp_path: Path) -> None:
    adapter = SyntheticAdapter()
    trainer = R2MapTrainer(_config(tmp_path), adapter)
    trainer.step()
    trainer.save_checkpoint()
    loaded = load_r2_map_checkpoint_pointer(
        tmp_path,
        "latest_complete",
        model_factory=lambda values: R2MapModel(R2MapModelConfig.from_dict(values)),
        optimizer_factory=lambda: optim.AdamW(learning_rate=3e-5, weight_decay=1e-4),
        loss_stream_path=trainer.loss_path,
    )
    expected_panel = prediction_panel(
        loaded.model, adapter.fixed_prediction_batch("r2-map-fixed-panel-v1")
    )
    for name, expected in expected_panel.items():
        np.testing.assert_array_equal(
            np.asarray(loaded.prediction_panel[name]), np.asarray(expected)
        )
    assert loaded.state.scheduler_state["next_step"] == 1
    assert int(loaded.optimizer.state["step"].item()) == 1


def test_adapter_identity_and_checkpoint_config_drift_fail_closed(tmp_path: Path) -> None:
    adapter = SyntheticAdapter()
    with pytest.raises(ValueError, match="dataset identity"):
        R2MapTrainer(replace(_config(tmp_path), dataset_blake3=DIGEST_A), adapter)
    trainer = R2MapTrainer(_config(tmp_path / "valid"), adapter)
    trainer.step()
    checkpoint = trainer.save_checkpoint()
    verify_r2_map_checkpoint(
        checkpoint,
        run_dir=trainer.run_dir,
        adapter=adapter,
        mark_last_verified=True,
    )
    drifted = replace(_config(trainer.run_dir), learning_rate=2e-5)
    with pytest.raises(CheckpointError, match="training_config_blake3"):
        R2MapTrainer.resume(drifted, adapter)
