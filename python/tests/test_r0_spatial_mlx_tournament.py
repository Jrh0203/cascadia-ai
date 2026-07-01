from __future__ import annotations

import json
from pathlib import Path

import cascadia_mlx.r0_spatial_mlx_tournament as tournament
import mlx.core as mx
import numpy as np
import pytest
from cascadia_mlx.r0_spatial_mlx_cache import R0SpatialMlxBatch
from cascadia_mlx.r0_spatial_mlx_model import R0SpatialIsoValueModel
from cascadia_mlx.r0_spatial_mlx_tournament import (
    ADR_ID,
    AUTHORIZED_ARMS,
    EXPERIMENT_ID,
    PROTOCOL_ID,
    R0SpatialMlxTournamentProtocol,
    _calibration_metrics,
    _pack_exact_shape,
    benchmark_model,
    benchmark_training_step,
    validate_authorization,
)


class _FakeCache:
    arm = "hex-radius-4-61"
    corpus_lock_id = "1" * 64
    exporter_executable_blake3 = "2" * 64

    def sample_count(self, split: str) -> int:
        return 4

    def batch(
        self,
        split: str,
        indices: np.ndarray,
        *,
        transform_ids: np.ndarray,
    ) -> R0SpatialMlxBatch:
        count = len(indices)
        tokens = np.zeros((count, 4, 84, 11), dtype=np.int32)
        mask = np.zeros((count, 4, 84), dtype=np.bool_)
        mask[:, :, :6] = True
        tokens[:, :, :6, 4] = 2
        tokens[:, :, :6, 5] = 1
        tokens[:, :, :6, 6] = 5
        tokens[:, :, :6, 9] = 5
        return R0SpatialMlxBatch(
            spatial_tokens=mx.array(tokens),
            spatial_mask=mx.array(mask),
            market_features=mx.zeros((count, 4, 31)),
            market_mask=mx.ones((count, 4), dtype=mx.bool_),
            global_features=mx.zeros((count, 96)),
            targets=mx.zeros((count, 11)),
            game_index=mx.array(indices),
            turn=mx.zeros((count,), dtype=mx.int32),
        )


def _authorization(path: Path) -> dict:
    identity = {
        "protocol_id": PROTOCOL_ID,
        "protocol_blake3": tournament._canonical_blake3(R0SpatialMlxTournamentProtocol().to_dict()),
        "corpus_lock_id": "1" * 64,
        "mlx_source_blake3": "3" * 64,
        "exporter_executable_blake3": "2" * 64,
        "authorized_arms": list(AUTHORIZED_ARMS),
        "approved_by": "test-parent",
        "approved_unix_ms": 1,
    }
    value = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "adr": ADR_ID,
        "approved": True,
        "authorization_id": tournament._canonical_blake3(identity),
        "identity": identity,
    }
    path.write_text(json.dumps(value))
    return value


def test_protocol_rejects_any_optimizer_or_budget_drift() -> None:
    R0SpatialMlxTournamentProtocol().validate()
    with pytest.raises(ValueError, match="protocol drifted"):
        R0SpatialMlxTournamentProtocol(training_steps=501).validate()


def test_authorization_is_bound_to_protocol_source_corpus_and_exporter(
    tmp_path: Path,
) -> None:
    path = tmp_path / "authorization.json"
    expected = _authorization(path)
    assert (
        validate_authorization(
            path,
            cache=_FakeCache(),
            source={"v2_source_blake3": "3" * 64},
        )
        == expected
    )
    drifted = json.loads(path.read_text())
    drifted["identity"]["exporter_executable_blake3"] = "4" * 64
    drifted["authorization_id"] = tournament._canonical_blake3(drifted["identity"])
    path.write_text(json.dumps(drifted))
    with pytest.raises(ValueError, match="does not match"):
        validate_authorization(
            path,
            cache=_FakeCache(),
            source={"v2_source_blake3": "3" * 64},
        )


def test_exact_shape_control_packs_only_active_rows() -> None:
    batch = _FakeCache().batch(
        "validation",
        np.arange(3),
        transform_ids=np.zeros(3, dtype=np.int64),
    )
    packed = _pack_exact_shape(batch)
    tokens = np.asarray(packed.spatial_tokens)
    mask = np.asarray(packed.spatial_mask)
    assert tokens.shape == (3, 4, 23, 11)
    assert np.all(mask.sum(axis=-1) == 6)
    assert np.all(tokens[~mask] == 0)


def test_same_host_inference_and_gradient_calibrations_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tournament, "WARMUP_ITERATIONS", 1)
    monkeypatch.setattr(tournament, "STEADY_ITERATIONS", 1)
    monkeypatch.setattr(tournament, "TRAINING_BENCHMARK_WARMUP_ITERATIONS", 1)
    monkeypatch.setattr(tournament, "TRAINING_BENCHMARK_STEADY_ITERATIONS", 1)
    mx.random.seed(41)
    model = R0SpatialIsoValueModel()
    inference = benchmark_model(model, _FakeCache(), exact_shape_control=True)
    training = benchmark_training_step(model, _FakeCache(), exact_shape_control=True)
    assert inference["spatial_token_capacity"] == 23
    assert inference["steady_examples_per_second"] > 0
    assert inference["inference_actions_per_second"] > 0
    assert training["spatial_token_capacity"] == 23
    assert training["examples_per_second"] > 0


def test_calibration_metrics_handle_constant_predictions_without_nan() -> None:
    metrics = _calibration_metrics(
        2,
        [
            4.0,
            6.0,
            8.0,
            20.0,
            12.0,
        ],
    )
    assert metrics["predicted_total_mean"] == 2.0
    assert metrics["target_total_mean"] == 3.0
    assert metrics["total_correlation"] == 0.0
    assert metrics["calibration_slope"] == 0.0
    assert metrics["calibration_intercept"] == 3.0
