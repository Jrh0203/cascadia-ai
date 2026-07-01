from __future__ import annotations

import json
from pathlib import Path

import blake3
import mlx.core as mx
import numpy as np
import pytest
from cascadia_mlx.graded_oracle_dataset import decode_graded_action_feature_bytes
from cascadia_mlx.r2_map_market_decision import decode_market_decision_action_bytes
from cascadia_mlx.r2_map_model import (
    R2MapBatch,
    R2MapMarketDecisionBatch,
    R2MapModel,
    R2MapModelConfig,
    R2MapPublicState,
)
from cascadia_mlx.r2_map_numpy import (
    R2MapNumpyError,
    R2MapNumpyModel,
    decode_action_features,
    decode_market_action_features,
)
from cascadia_mlx.r2_map_serve import R2MapProtocolError, _verify_portable_checkpoint
from cascadia_mlx.r2_map_tensor_contract import (
    BOARD_SLOTS,
    BOARD_TOKEN_CAPACITY,
    GLOBAL_FEATURES,
    MARKET_FEATURES,
    PLAYER_FEATURES,
    TOKEN_FEATURES,
)
from mlx.utils import tree_flatten


def _state(rng: np.random.Generator, count: int) -> dict[str, np.ndarray]:
    token_types = np.zeros((count, BOARD_SLOTS, BOARD_TOKEN_CAPACITY), dtype=np.int32)
    token_types[..., :24] = rng.integers(1, 5, size=(count, BOARD_SLOTS, 24))
    token_mask = token_types != 0
    token_features = rng.normal(
        0.0, 0.15, size=(count, BOARD_SLOTS, BOARD_TOKEN_CAPACITY, TOKEN_FEATURES)
    ).astype(np.float32)
    token_features *= token_mask[..., None]
    return {
        "token_features": token_features,
        "token_types": token_types,
        "token_mask": token_mask,
        "market_features": rng.normal(0.0, 0.15, size=(count, 4, MARKET_FEATURES)).astype(
            np.float32
        ),
        "market_mask": np.ones((count, 4), dtype=np.bool_),
        "player_features": rng.normal(0.0, 0.15, size=(count, BOARD_SLOTS, PLAYER_FEATURES)).astype(
            np.float32
        ),
        "player_mask": np.ones((count, BOARD_SLOTS), dtype=np.bool_),
        "global_features": rng.normal(0.0, 0.15, size=(count, GLOBAL_FEATURES)).astype(np.float32),
    }


def _mlx_state(values: dict[str, np.ndarray], *, candidates: bool = False) -> R2MapPublicState:
    if candidates:
        values = {name: value[None, ...] for name, value in values.items()}
    return R2MapPublicState(**{name: mx.array(value) for name, value in values.items()})


def _actions(count: int) -> np.ndarray:
    raw = np.zeros((count, 128), dtype=np.uint8)
    raw[:, 1] = np.arange(count) % 2
    raw[:, 2] = np.arange(count) % 4
    raw[:, 3] = (np.arange(count) + 1) % 4
    raw[:, 4] = 17
    raw[:, 5] = 2
    raw[:, 6] = 255
    raw[:, 7] = 0b10101
    raw[:, 9] = 3
    raw[:, 10] = np.arange(count, dtype=np.int8).view(np.uint8)
    raw[:, 11] = (-np.arange(count, dtype=np.int8)).view(np.uint8)
    raw[:, 12] = np.arange(count) % 6
    raw[:, 13] = 1
    raw[:, 14] = 1
    raw[:, 15] = 255
    raw[:, 38] = 3
    raw[:, 104:106] = np.array([73], dtype="<u2").view(np.uint8)
    raw[:, 106:128] = np.arange(11, dtype="<i2").view(np.uint8)
    return raw


def _portable_model(tmp_path: Path) -> tuple[R2MapModel, R2MapNumpyModel]:
    mx.random.seed(20260619)
    model = R2MapModel()
    # Fresh checkpoints deliberately zero selection heads. Give the golden
    # fixture deterministic nonzero heads so fusion/trunk parity is observable.
    model.score_to_go_head.weight = mx.random.normal(model.score_to_go_head.weight.shape) * 0.03
    model.score_to_go_head.bias = mx.array([0.125], dtype=mx.float32)
    model.bootstrap_policy_head.weight = (
        mx.random.normal(model.bootstrap_policy_head.weight.shape) * 0.03
    )
    model.market_decision_score_to_go_head.weight = (
        mx.random.normal(model.market_decision_score_to_go_head.weight.shape) * 0.03
    )
    model.market_decision_score_to_go_head.bias = mx.array([-0.25], dtype=mx.float32)
    mx.eval(model.parameters())
    path = tmp_path / "model.safetensors"
    mx.save_safetensors(str(path), dict(tree_flatten(model.parameters())))
    return model, R2MapNumpyModel(path, R2MapModelConfig().to_dict(), candidate_chunk_size=2)


def test_portable_action_and_market_decoders_are_exact() -> None:
    actions = _actions(5)
    np.testing.assert_array_equal(
        decode_action_features(actions), decode_graded_action_feature_bytes(actions)
    )
    market = np.asarray(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0, 0],
            [1, 1, 2, 0, 0, 0, 0, 0],
            [1, 1, 3, 5, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(
        decode_market_action_features(market), decode_market_decision_action_bytes(market)
    )


def test_static_opponent_deduplication_preserves_exact_graph(tmp_path: Path) -> None:
    _model, optimized = _portable_model(tmp_path)
    reference = R2MapNumpyModel(
        tmp_path / "model.safetensors",
        R2MapModelConfig().to_dict(),
        candidate_chunk_size=4,
        deduplicate_static_opponents=False,
    )
    rng = np.random.default_rng(42017)
    parent, candidates = _state(rng, 1), _state(rng, 7)
    for name in (
        "token_features",
        "token_types",
        "token_mask",
        "player_features",
        "player_mask",
    ):
        candidates[name][:, 1:] = candidates[name][0, 1:]
    actions = _actions(7)
    exact = np.arange(7, dtype=np.float32) + 60.0
    expected = reference.score_actions(parent, candidates, actions, exact)
    observed = optimized.score_actions(parent, candidates, actions, exact)
    for name in (
        "action_scores",
        "predicted_score_to_go",
        "predicted_score_components_to_go",
        "bootstrap_policy_logits",
    ):
        np.testing.assert_allclose(
            getattr(observed, name), getattr(expected, name), rtol=2e-5, atol=2e-5
        )


@pytest.mark.parametrize("fixture_seed", [7, 918273])
def test_numpy_cpu_logits_match_mlx_fixed_fixtures(tmp_path: Path, fixture_seed: int) -> None:
    model, portable = _portable_model(tmp_path)
    rng = np.random.default_rng(fixture_seed)
    parent, candidates = _state(rng, 1), _state(rng, 3)
    actions = _actions(3)
    exact = np.asarray([72.0, 73.0, 74.0], dtype=np.float32)
    batch = R2MapBatch(
        parent=_mlx_state(parent),
        candidates=_mlx_state(candidates, candidates=True),
        candidate_mask=mx.ones((1, 3), dtype=mx.bool_),
        action_features=mx.array(decode_graded_action_feature_bytes(actions)[None, ...]),
        exact_afterstate_scores=mx.array(exact[None, ...]),
    )
    mlx_prediction = model.score_actions(batch)
    mx.eval(
        mlx_prediction.action_scores,
        mlx_prediction.predicted_score_to_go,
        mlx_prediction.predicted_score_components_to_go,
        mlx_prediction.bootstrap_policy_logits,
    )
    portable_prediction = portable.score_actions(parent, candidates, actions, exact)
    for name in (
        "action_scores",
        "predicted_score_to_go",
        "predicted_score_components_to_go",
        "bootstrap_policy_logits",
    ):
        np.testing.assert_allclose(
            np.asarray(getattr(mlx_prediction, name))[0],
            getattr(portable_prediction, name),
            rtol=2e-5,
            atol=2e-5,
        )

    market = np.asarray([[1, 1, 2, 0, 0, 0, 0, 0], [1, 1, 3, 5, 0, 0, 0, 0]], dtype=np.uint8)
    market_batch = R2MapMarketDecisionBatch(
        public_state=_mlx_state(parent),
        action_mask=mx.ones((1, 2), dtype=mx.bool_),
        action_features=mx.array(decode_market_decision_action_bytes(market)[None, ...]),
        exact_current_scores=mx.array([55.0], dtype=mx.float32),
    )
    mlx_market = model.score_market_decisions(market_batch)
    mx.eval(mlx_market.action_scores, mlx_market.predicted_score_to_go)
    portable_market = portable.score_market_decisions(parent, market, 55.0)
    np.testing.assert_allclose(
        np.asarray(mlx_market.action_scores)[0], portable_market.action_scores, rtol=2e-5, atol=2e-5
    )
    np.testing.assert_allclose(
        np.asarray(mlx_market.predicted_score_to_go)[0],
        portable_market.predicted_score_to_go,
        rtol=2e-5,
        atol=2e-5,
    )


def test_portable_model_rejects_config_and_safetensors_schema(tmp_path: Path) -> None:
    _model, portable = _portable_model(tmp_path)
    assert portable.array_backend == "numpy"
    drifted = R2MapModelConfig().to_dict() | {"hidden_dim": 256}
    with pytest.raises(R2MapNumpyError, match="config"):
        R2MapNumpyModel(tmp_path / "model.safetensors", drifted)
    malformed = tmp_path / "malformed.safetensors"
    malformed.write_bytes(b"short")
    with pytest.raises(R2MapNumpyError, match="truncated"):
        R2MapNumpyModel(malformed, R2MapModelConfig().to_dict())


def test_portable_checkpoint_rejects_model_hash_tampering(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "checkpoints" / "step-000001"
    verification = run_dir / "verifications"
    checkpoint.mkdir(parents=True)
    verification.mkdir()
    model = checkpoint / "model.safetensors"
    model.write_bytes(b"frozen-model-bytes")
    config = R2MapModelConfig().to_dict()
    manifest = {
        "schema_version": 2,
        "schema_id": "r2-map-checkpoint-v2",
        "checkpoint_id": checkpoint.name,
        "model_config": config,
        "identity": {
            "model_config_blake3": blake3.blake3(
                json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        },
        "files": {
            "model.safetensors": {
                "bytes": model.stat().st_size,
                "blake3": blake3.blake3(model.read_bytes()).hexdigest(),
            }
        },
    }
    manifest_path = checkpoint / "checkpoint.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    receipt = {
        "schema_version": 2,
        "schema_id": "r2-map-checkpoint-verification-v2",
        "checkpoint_id": checkpoint.name,
        "checkpoint_manifest_blake3": blake3.blake3(manifest_path.read_bytes()).hexdigest(),
        "exact_prediction_match": True,
        "exact_next_batch_match": True,
        "verification_id": "a" * 64,
    }
    (verification / f"{checkpoint.name}.json").write_text(json.dumps(receipt))
    _verify_portable_checkpoint(run_dir, checkpoint)
    model.write_bytes(b"tampered-model-bytes")
    with pytest.raises(R2MapProtocolError, match="hash differs"):
        _verify_portable_checkpoint(run_dir, checkpoint)
