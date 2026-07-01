from __future__ import annotations

import json
from pathlib import Path

import cascadia_mlx.r2_sparse_mlx_tournament as tournament
import mlx.core as mx
import numpy as np
import pytest
from cascadia_mlx.r2_sparse_mlx_cache import (
    BOARD_SLOTS,
    BOARD_TOKEN_CAPACITY,
    GLOBAL_FEATURES,
    GRAPH_MAX_DEGREE,
    MARKET_FEATURES,
    PLAYER_FEATURES,
    TARGET_DIM,
    TOKEN_CAPACITY,
    TOKEN_FEATURES,
    R2SparseMlxBatch,
)
from cascadia_mlx.r2_sparse_mlx_model import R2SparseValueModel
from cascadia_mlx.r2_sparse_mlx_tournament import (
    ADR_ID,
    AUTHORIZED_RUNS,
    EXPERIMENT_ID,
    PROTOCOL_ID,
    RUN_ARCHITECTURES,
    R2SparseMlxTournamentProtocol,
    _ablate_token_type,
    _calibration_metrics,
    benchmark_model,
    benchmark_training_step,
    validate_authorization,
)


class _FakeCache:
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
    ) -> R2SparseMlxBatch:
        count = len(indices)
        mask = np.zeros(
            (count, BOARD_SLOTS, BOARD_TOKEN_CAPACITY),
            dtype=np.bool_,
        )
        mask[:, :, :6] = True
        token_types = np.zeros(mask.shape, dtype=np.int32)
        token_types[:, :, :6] = np.asarray([1, 1, 2, 2, 3, 4])
        return R2SparseMlxBatch(
            token_features=mx.zeros(
                (
                    count,
                    BOARD_SLOTS,
                    BOARD_TOKEN_CAPACITY,
                    TOKEN_FEATURES,
                )
            ),
            token_types=mx.array(token_types),
            token_mask=mx.array(mask),
            graph_neighbors=mx.zeros(
                (
                    count,
                    BOARD_SLOTS,
                    BOARD_TOKEN_CAPACITY,
                    GRAPH_MAX_DEGREE,
                ),
                dtype=mx.int32,
            ),
            graph_neighbor_mask=mx.zeros(
                (
                    count,
                    BOARD_SLOTS,
                    BOARD_TOKEN_CAPACITY,
                    GRAPH_MAX_DEGREE,
                ),
                dtype=mx.bool_,
            ),
            graph_relations=mx.zeros(
                (
                    count,
                    BOARD_SLOTS,
                    BOARD_TOKEN_CAPACITY,
                    GRAPH_MAX_DEGREE,
                ),
                dtype=mx.int32,
            ),
            graph_direction_features=mx.zeros(
                (
                    count,
                    BOARD_SLOTS,
                    BOARD_TOKEN_CAPACITY,
                    GRAPH_MAX_DEGREE,
                    6,
                )
            ),
            market_features=mx.zeros((count, 4, MARKET_FEATURES)),
            market_mask=mx.ones((count, 4), dtype=mx.bool_),
            player_features=mx.zeros(
                (count, BOARD_SLOTS, PLAYER_FEATURES)
            ),
            player_mask=mx.ones((count, BOARD_SLOTS), dtype=mx.bool_),
            global_features=mx.zeros((count, GLOBAL_FEATURES)),
            targets=mx.zeros((count, TARGET_DIM)),
            game_index=mx.array(indices),
            turn=mx.zeros((count,), dtype=mx.int32),
            transform_ids=mx.zeros((count,), dtype=mx.int32),
        )


def _authorization(path: Path, r0_control: Path) -> dict:
    r0_identity = {
        "r0_classification": tournament.R0_COMPLETE_CLASSIFICATION,
        "classification_order_byte_identical": True,
        "r0_selected_stage2_candidate": None,
        "selected_control_arm": tournament.R0_EXACT_CONTROL,
        "validation": {
            "samples": 10_000,
            "mean_component_mae": 2.7,
            "total_mae": 2.65,
            "total_rmse": 3.37,
        },
    }
    r0_value = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "adr": ADR_ID,
        "contract_id": tournament.R0_BINDING_CONTRACT,
        "binding_id": tournament._canonical_blake3(r0_identity),
        "identity": r0_identity,
    }
    r0_control.write_text(json.dumps(r0_value))
    identity = {
        "protocol_id": PROTOCOL_ID,
        "protocol_blake3": tournament._canonical_blake3(
            R2SparseMlxTournamentProtocol().to_dict()
        ),
        "corpus_lock_id": "1" * 64,
        "r0_control_binding_id": r0_value["binding_id"],
        "mlx_source_blake3": "4" * 64,
        "exporter_executable_blake3": "2" * 64,
        "bundle_id": "5" * 64,
        "authorized_runs": list(AUTHORIZED_RUNS),
        "run_architectures": RUN_ARCHITECTURES,
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


def test_protocol_rejects_budget_or_parameter_scope_drift() -> None:
    protocol = R2SparseMlxTournamentProtocol()
    protocol.validate()
    assert "input-adapters" in protocol.parameter_count_scope
    with pytest.raises(ValueError, match="protocol drifted"):
        R2SparseMlxTournamentProtocol(training_steps=501).validate()


def test_authorization_binds_r0_source_corpus_exporter_and_run(tmp_path: Path) -> None:
    path = tmp_path / "authorization.json"
    r0_control = tmp_path / "r0-control.json"
    expected = _authorization(path, r0_control)
    assert (
        validate_authorization(
            path,
            cache=_FakeCache(),
            source={"v2_source_blake3": "4" * 64},
            run_role="graph-primary",
            r0_control=r0_control,
        )
        == expected
    )
    drifted_control = json.loads(r0_control.read_text())
    drifted_control["identity"]["selected_control_arm"] = "hex-radius-4-61"
    drifted_control["binding_id"] = tournament._canonical_blake3(
        drifted_control["identity"]
    )
    r0_control.write_text(json.dumps(drifted_control))
    with pytest.raises(ValueError, match="absent, stale, or incomplete"):
        validate_authorization(
            path,
            cache=_FakeCache(),
            source={"v2_source_blake3": "4" * 64},
            run_role="graph-primary",
            r0_control=r0_control,
        )


def test_inference_and_gradient_benchmarks_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tournament, "WARMUP_ITERATIONS", 1)
    monkeypatch.setattr(tournament, "STEADY_ITERATIONS", 1)
    monkeypatch.setattr(tournament, "TRAINING_BENCHMARK_WARMUP_ITERATIONS", 1)
    monkeypatch.setattr(tournament, "TRAINING_BENCHMARK_STEADY_ITERATIONS", 1)
    model = R2SparseValueModel()
    inference = benchmark_model(model, _FakeCache())
    training = benchmark_training_step(model, _FakeCache())
    assert inference["token_capacity"] == TOKEN_CAPACITY
    assert inference["inference_actions_per_second"] > 0
    assert training["examples_per_second"] > 0


def test_type_ablation_masks_tokens_and_cached_graph_incidence() -> None:
    batch = _FakeCache().batch(
        "validation",
        np.asarray([0]),
        transform_ids=np.asarray([0]),
    )
    token_types = np.asarray(batch.token_types)
    neighbors = np.asarray(batch.graph_neighbors)
    neighbor_mask = np.zeros_like(neighbors, dtype=np.bool_)
    relations = np.zeros_like(neighbors, dtype=np.int32)
    directions = np.zeros((*neighbors.shape, 6), dtype=np.float32)

    neighbors[0, 0, 0, 0] = 2
    neighbor_mask[0, 0, 0, 0] = True
    relations[0, 0, 0, 0] = 1
    neighbors[0, 0, 2, 0] = 0
    neighbor_mask[0, 0, 2, 0] = True
    relations[0, 0, 2, 0] = 1
    batch = R2SparseMlxBatch(
        **{
            **batch.__dict__,
            "graph_neighbors": mx.array(neighbors),
            "graph_neighbor_mask": mx.array(neighbor_mask),
            "graph_relations": mx.array(relations),
            "graph_direction_features": mx.array(directions),
        }
    )

    ablated = _ablate_token_type(batch, 2)
    output_mask = np.asarray(ablated.token_mask)
    output_edges = np.asarray(ablated.graph_neighbor_mask)
    assert np.all(~output_mask[token_types == 2])
    assert output_edges[0, 0, 0, 0] == 0
    assert output_edges[0, 0, 2, 0] == 0


def test_calibration_metrics_handle_constant_predictions_without_nan() -> None:
    metrics = _calibration_metrics(2, [4.0, 6.0, 8.0, 20.0, 12.0])
    assert metrics["predicted_total_mean"] == 2.0
    assert metrics["target_total_mean"] == 3.0
    assert metrics["total_correlation"] == 0.0
    assert metrics["calibration_slope"] == 0.0
    assert metrics["calibration_intercept"] == 3.0
