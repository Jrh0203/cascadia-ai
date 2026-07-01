"""Portable NumPy/OpenBLAS inference for the frozen R2-MAP v1.1 graph.

This module intentionally has no MLX dependency.  It loads the exact float32
``model.safetensors`` produced by native John1 training and mirrors the MLX
public-state encoders, attention blocks, action scorer, and market-decision
head.  Linux/arm64 worker containers use it for local frozen inference.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

BOARD_SLOTS = 4
BOARD_TOKEN_CAPACITY = 139
TOKEN_FEATURES = 60
MARKET_FEATURES = 31
PLAYER_FEATURES = 23
GLOBAL_FEATURES = 96
ACTION_FEATURES = 140
MARKET_ACTION_FEATURES = 16
HIDDEN = 192
HEADS = 4
LATENTS = 16
SCORE_COMPONENTS = 11
ACTION_BYTES = 128
MARKET_ACTION_BYTES = 8
MODEL_ARCHITECTURE = "r2-map-v1.1"
DEFAULT_CANDIDATE_CHUNK_SIZE = 256


class R2MapNumpyError(ValueError):
    """Portable checkpoint, tensor, or graph contract violation."""


@dataclass(frozen=True)
class NumpyActionPrediction:
    action_scores: np.ndarray
    predicted_score_to_go: np.ndarray
    predicted_score_components_to_go: np.ndarray
    bootstrap_policy_logits: np.ndarray


@dataclass(frozen=True)
class NumpyMarketPrediction:
    action_scores: np.ndarray
    predicted_score_to_go: np.ndarray


def _gelu(value: np.ndarray) -> np.ndarray:
    """Float32 GELU with a vectorized erf approximation (<2e-7 error)."""
    x = np.asarray(value, dtype=np.float32)
    absolute = np.abs(x / np.float32(math.sqrt(2.0)))
    t = np.float32(1.0) / (np.float32(1.0) + np.float32(0.3275911) * absolute)
    polynomial = (
        (
            ((np.float32(1.061405429) * t - np.float32(1.453152027)) * t + np.float32(1.421413741))
            * t
            - np.float32(0.284496736)
        )
        * t
        + np.float32(0.254829592)
    ) * t
    erf = np.float32(1.0) - polynomial * np.exp(-(absolute * absolute))
    erf = np.copysign(erf, x)
    return x * np.float32(0.5) * (np.float32(1.0) + erf)


def _layer_norm(
    value: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    mean = np.mean(value, axis=-1, keepdims=True, dtype=np.float32)
    centered = value - mean
    variance = np.mean(centered * centered, axis=-1, keepdims=True, dtype=np.float32)
    normalized = centered / np.sqrt(variance + np.float32(eps))
    return normalized * weight + bias


def _masked_mean(value: np.ndarray, mask: np.ndarray) -> np.ndarray:
    weights = np.asarray(mask, dtype=np.float32)[..., None]
    count = np.maximum(np.sum(weights, axis=1, dtype=np.float32), np.float32(1.0))
    return np.sum(value * weights, axis=1, dtype=np.float32) / count


def _masked_pool(value: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mean = _masked_mean(value, mask)
    masked = np.where(np.asarray(mask, dtype=np.bool_)[..., None], value, np.float32(-1e9))
    maximum = np.max(masked, axis=1)
    maximum = np.where(np.any(mask, axis=1, keepdims=True), maximum, np.float32(0.0))
    return np.concatenate([mean, maximum], axis=-1)


def _one_hot(values: np.ndarray, classes: int) -> np.ndarray:
    values = np.asarray(values)
    valid = values < classes
    clipped = np.where(valid, values, 0)
    return np.eye(classes, dtype=np.float32)[clipped] * valid[..., None]


def _one_hot_with_none(values: np.ndarray, classes: int) -> np.ndarray:
    mapped = np.where(np.asarray(values) < classes, values, classes)
    return np.eye(classes + 1, dtype=np.float32)[mapped]


def _mask_bits(values: np.ndarray, bits: int) -> np.ndarray:
    shifts = np.arange(bits, dtype=np.uint8)
    return ((np.asarray(values, dtype=np.uint8)[..., None] >> shifts) & 1).astype(np.float32)


def decode_action_features(raw_actions: np.ndarray) -> np.ndarray:
    raw = np.ascontiguousarray(raw_actions, dtype=np.uint8)
    if raw.ndim != 2 or raw.shape[1] != ACTION_BYTES or not len(raw):
        raise R2MapNumpyError("R2-MAP action bytes must have shape [N, 128]")
    signed = raw.view(np.int8)
    result = np.zeros((len(raw), ACTION_FEATURES), dtype=np.float32)
    result[:, 0] = raw[:, 0]
    result[:, 1:3] = _one_hot(raw[:, 1], 2)
    result[:, 3:7] = _one_hot(raw[:, 2], 4)
    result[:, 7:11] = _one_hot(raw[:, 3], 4)
    result[:, 11] = raw[:, 4].astype(np.float32) / np.float32(84.0)
    result[:, 12:17] = _one_hot(raw[:, 5], 5)
    result[:, 17:23] = _one_hot_with_none(raw[:, 6], 5)
    result[:, 23:28] = _mask_bits(raw[:, 7], 5)
    result[:, 28] = raw[:, 8]
    result[:, 29:34] = _one_hot(raw[:, 9], 5)
    result[:, 34] = signed[:, 10].astype(np.float32) / np.float32(24.0)
    result[:, 35] = signed[:, 11].astype(np.float32) / np.float32(24.0)
    result[:, 36:42] = _one_hot(raw[:, 12], 6)
    presence = raw[:, 13].astype(np.float32)
    result[:, 42] = presence
    result[:, 43] = signed[:, 14].astype(np.float32) / np.float32(24.0) * presence
    result[:, 44] = signed[:, 15].astype(np.float32) / np.float32(24.0) * presence
    result[:, 45] = raw[:, 16]
    result[:, 46] = raw[:, 17].astype(np.float32) / np.float32(20.0)
    result[:, 47:127] = _mask_bits(raw[:, 18:38], 4).reshape(len(raw), 80)
    result[:, 127] = raw[:, 38].astype(np.float32) / np.float32(20.0)
    result[:, 128] = np.frombuffer(raw[:, 104:106].copy().tobytes(), dtype="<u2").astype(
        np.float32
    ) / np.float32(100.0)
    deltas = np.frombuffer(raw[:, 106:128].copy().tobytes(), dtype="<i2").reshape(len(raw), 11)
    result[:, 129:140] = deltas.astype(np.float32) / np.float32(20.0)
    return result


def decode_market_action_features(raw_actions: np.ndarray) -> np.ndarray:
    raw = np.ascontiguousarray(raw_actions, dtype=np.uint8)
    if raw.ndim != 2 or raw.shape[1] != MARKET_ACTION_BYTES or not len(raw):
        raise R2MapNumpyError("R2-MAP market action bytes must have shape [N, 8]")
    if np.any(raw[:, 0] != 1) or np.any(raw[:, 4:] != 0):
        raise R2MapNumpyError("R2-MAP market action schema or reserved bytes differ")
    decision, action, masks = raw[:, 1], raw[:, 2], raw[:, 3]
    if np.any(decision > 1) or np.any(action > 3) or np.any(masks > 15):
        raise R2MapNumpyError("R2-MAP market action enum or mask is invalid")
    result = np.zeros((len(raw), MARKET_ACTION_FEATURES), dtype=np.float32)
    result[np.arange(len(raw)), decision] = 1.0
    result[np.arange(len(raw)), 2 + action] = 1.0
    result[:, 6:10] = _mask_bits(masks, 4)
    cardinality = np.sum(result[:, 6:10], axis=1).astype(np.int64)
    present = cardinality > 0
    result[np.arange(len(raw))[present], 9 + cardinality[present]] = 1.0
    result[:, 14] = action == 3
    result[:, 15] = (action == 0) | (action == 2)
    return result


class SafeTensorStore:
    """Strict read-only float32 safetensors loader backed by one byte buffer."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        raw = self.path.read_bytes()
        if len(raw) < 8:
            raise R2MapNumpyError("safetensors file is truncated")
        header_bytes = int.from_bytes(raw[:8], "little")
        if header_bytes <= 0 or 8 + header_bytes > len(raw):
            raise R2MapNumpyError("safetensors header length is invalid")
        try:
            header = json.loads(raw[8 : 8 + header_bytes])
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise R2MapNumpyError("safetensors header is invalid") from error
        if not isinstance(header, dict):
            raise R2MapNumpyError("safetensors header must be an object")
        self._raw = raw
        self._base = 8 + header_bytes
        self._header = {name: value for name, value in header.items() if name != "__metadata__"}

    def tensor(self, name: str, shape: tuple[int, ...]) -> np.ndarray:
        descriptor = self._header.get(name)
        if not isinstance(descriptor, dict) or descriptor.get("dtype") != "F32":
            raise R2MapNumpyError(f"missing float32 model tensor {name}")
        if descriptor.get("shape") != list(shape):
            raise R2MapNumpyError(f"model tensor {name} shape differs")
        offsets = descriptor.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(value, int) for value in offsets)
        ):
            raise R2MapNumpyError(f"model tensor {name} offsets are invalid")
        start, stop = self._base + offsets[0], self._base + offsets[1]
        expected = math.prod(shape) * 4
        if start < self._base or stop - start != expected or stop > len(self._raw):
            raise R2MapNumpyError(f"model tensor {name} byte range is invalid")
        return np.frombuffer(self._raw, dtype="<f4", count=math.prod(shape), offset=start).reshape(
            shape
        )


class R2MapNumpyModel:
    """Exact graph mirror used by Linux CPU workers."""

    array_backend = "numpy"

    def __init__(
        self,
        weights: str | Path,
        model_config: dict[str, Any],
        *,
        candidate_chunk_size: int = DEFAULT_CANDIDATE_CHUNK_SIZE,
        deduplicate_static_opponents: bool = True,
    ):
        if (
            model_config.get("architecture") != MODEL_ARCHITECTURE
            or model_config.get("hidden_dim") != HIDDEN
            or model_config.get("attention_heads") != HEADS
            or model_config.get("board_latents") != LATENTS
            or model_config.get("board_latent_blocks") != 1
            or model_config.get("cross_board_blocks") != 1
            or model_config.get("feed_forward_multiplier") != 2
            or model_config.get("action_feature_dim") != ACTION_FEATURES
            or model_config.get("score_component_dim") != SCORE_COMPONENTS
            or model_config.get("precision") != "float32"
        ):
            raise R2MapNumpyError("portable R2-MAP model config differs from v1.1")
        if candidate_chunk_size <= 0:
            raise R2MapNumpyError("candidate chunk size must be positive")
        self.store = SafeTensorStore(weights)
        self.candidate_chunk_size = candidate_chunk_size
        self.deduplicate_static_opponents = deduplicate_static_opponents
        self._validate_live_weights()

    def _weight(self, name: str, shape: tuple[int, ...]) -> np.ndarray:
        return self.store.tensor(name, shape)

    def _linear(self, value: np.ndarray, name: str, out: int, *, bias: bool = True) -> np.ndarray:
        inputs = value.shape[-1]
        weight = self._weight(f"{name}.weight", (out, inputs))
        result = np.matmul(value, weight.T)
        if bias:
            result = result + self._weight(f"{name}.bias", (out,))
        return np.asarray(result, dtype=np.float32)

    def _norm(self, value: np.ndarray, name: str, width: int) -> np.ndarray:
        return _layer_norm(
            value,
            self._weight(f"{name}.weight", (width,)),
            self._weight(f"{name}.bias", (width,)),
        )

    def _projection(self, value: np.ndarray, name: str, inputs: int) -> np.ndarray:
        value = _gelu(self._linear(value, f"{name}.layers.0", HIDDEN))
        return self._norm(value, f"{name}.layers.2", HIDDEN)

    def _attention(
        self,
        queries: np.ndarray,
        keys: np.ndarray,
        values: np.ndarray,
        mask: np.ndarray,
        name: str,
    ) -> np.ndarray:
        batch, query_count, _ = queries.shape
        key_count = keys.shape[1]
        head = HIDDEN // HEADS
        q = (
            self._linear(queries, f"{name}.query_proj", HIDDEN)
            .reshape(batch, query_count, HEADS, head)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self._linear(keys, f"{name}.key_proj", HIDDEN)
            .reshape(batch, key_count, HEADS, head)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self._linear(values, f"{name}.value_proj", HIDDEN)
            .reshape(batch, key_count, HEADS, head)
            .transpose(0, 2, 1, 3)
        )
        logits = np.matmul(q, k.transpose(0, 1, 3, 2)) * np.float32(1.0 / math.sqrt(head))
        logits = np.where(mask[:, None, None, :], logits, np.float32(-1e9))
        logits = logits - np.max(logits, axis=-1, keepdims=True)
        probability = np.exp(logits)
        probability /= np.sum(probability, axis=-1, keepdims=True, dtype=np.float32)
        output = np.matmul(probability, v).transpose(0, 2, 1, 3).reshape(batch, query_count, HIDDEN)
        return self._linear(output, f"{name}.out_proj", HIDDEN)

    def _feed_forward(self, value: np.ndarray, name: str) -> np.ndarray:
        hidden = _gelu(self._linear(value, f"{name}.layers.0", HIDDEN * 2))
        return self._linear(hidden, f"{name}.layers.2", HIDDEN)

    def _cross_block(
        self, latents: np.ndarray, inputs: np.ndarray, mask: np.ndarray, prefix: str
    ) -> np.ndarray:
        query = self._norm(latents, f"{prefix}.query_norm", HIDDEN)
        normalized = self._norm(inputs, f"{prefix}.input_norm", HIDDEN)
        latents = latents + self._attention(
            query, normalized, normalized, mask, f"{prefix}.attention"
        )
        normalized_latents = self._norm(latents, f"{prefix}.feed_forward_norm", HIDDEN)
        return latents + self._feed_forward(normalized_latents, f"{prefix}.feed_forward")

    def _self_block(self, value: np.ndarray, mask: np.ndarray, prefix: str) -> np.ndarray:
        normalized = self._norm(value, f"{prefix}.norm_attention", HIDDEN)
        value = value + self._attention(
            normalized, normalized, normalized, mask, f"{prefix}.attention"
        )
        normalized = self._norm(value, f"{prefix}.norm_feed_forward", HIDDEN)
        value = value + self._feed_forward(normalized, f"{prefix}.feed_forward")
        return value * mask[..., None]

    def _summary_projection(self, value: np.ndarray, prefix: str) -> np.ndarray:
        value = _gelu(self._linear(value, f"{prefix}.layers.0", HIDDEN))
        return self._norm(value, f"{prefix}.layers.2", HIDDEN)

    def _encode_board_rows(
        self,
        token_features: np.ndarray,
        token_types: np.ndarray,
        token_mask: np.ndarray,
        players: np.ndarray,
        player_mask: np.ndarray,
        prefix: str,
    ) -> np.ndarray:
        """Encode one or more board slots without recomputing repeated slots."""
        batch, slots = token_features.shape[:2]
        tokens = (
            self._projection(
                token_features,
                f"{prefix}.common_encoder.token_projection",
                TOKEN_FEATURES,
            )
            * token_mask[..., None]
        )
        flat_tokens = tokens.reshape(batch * slots, BOARD_TOKEN_CAPACITY, HIDDEN)
        flat_types = token_types.reshape(batch * slots, BOARD_TOKEN_CAPACITY)
        flat_mask = token_mask.reshape(batch * slots, BOARD_TOKEN_CAPACITY)
        flat_players = players.reshape(batch * slots, 1, HIDDEN)
        flat_player_mask = player_mask.reshape(batch * slots, 1)
        type_rows, type_masks = [], []
        for token_type in range(1, 5):
            selected = flat_mask & (flat_types == token_type)
            type_rows.append(_masked_mean(flat_tokens, selected))
            type_masks.append(np.any(selected, axis=1))
        type_summaries = np.stack(type_rows, axis=1)
        type_mask = np.stack(type_masks, axis=1)
        inputs = np.concatenate([flat_players, type_summaries, flat_tokens], axis=1)
        input_mask = np.concatenate([flat_player_mask, type_mask, flat_mask], axis=1)
        latents = np.broadcast_to(
            self._weight(f"{prefix}.latents", (LATENTS, HIDDEN))[None, :, :],
            (batch * slots, LATENTS, HIDDEN),
        ).copy()
        latents = self._cross_block(latents, inputs, input_mask, f"{prefix}.perceiver_cross")
        latent_mask = np.ones((batch * slots, LATENTS), dtype=np.bool_)
        latents = self._self_block(latents, latent_mask, f"{prefix}.board_blocks.0")
        return self._summary_projection(
            _masked_pool(latents, latent_mask), f"{prefix}.board_summary_projection"
        ).reshape(batch, slots, HIDDEN)

    @staticmethod
    def _opponent_rows_are_static(state: dict[str, np.ndarray]) -> bool:
        """Return whether the three non-acting slots are identical in the group."""
        if state["token_features"].shape[0] <= 1:
            return False
        for name in (
            "token_features",
            "token_types",
            "token_mask",
            "player_features",
            "player_mask",
        ):
            rows = state[name][:, 1:]
            if not np.array_equal(rows[1:], np.broadcast_to(rows[:1], rows[1:].shape)):
                return False
        return True

    def encode_state(self, state: dict[str, np.ndarray], prefix: str) -> np.ndarray:
        _validate_public_state(state)
        batch = state["token_features"].shape[0]
        market_rows = self._projection(
            state["market_features"], f"{prefix}.common_encoder.market_projection", MARKET_FEATURES
        )
        market = _masked_mean(market_rows, state["market_mask"])
        global_context = self._projection(
            state["global_features"], f"{prefix}.common_encoder.global_projection", GLOBAL_FEATURES
        )
        static_opponents = (
            self.deduplicate_static_opponents
            and prefix == "afterstate_encoder"
            and self._opponent_rows_are_static(state)
        )
        if static_opponents:
            active_players = self._projection(
                state["player_features"][:, :1],
                f"{prefix}.common_encoder.player_projection",
                PLAYER_FEATURES,
            )
            opponent_players_once = self._projection(
                state["player_features"][:1, 1:],
                f"{prefix}.common_encoder.player_projection",
                PLAYER_FEATURES,
            )
            players = np.concatenate(
                [
                    active_players,
                    np.broadcast_to(opponent_players_once, (batch, BOARD_SLOTS - 1, HIDDEN)),
                ],
                axis=1,
            )
            players = players * state["player_mask"][..., None]
            active_board = self._encode_board_rows(
                state["token_features"][:, :1],
                state["token_types"][:, :1],
                state["token_mask"][:, :1],
                players[:, :1],
                state["player_mask"][:, :1],
                prefix,
            )
            opponent_board_once = self._encode_board_rows(
                state["token_features"][:1, 1:],
                state["token_types"][:1, 1:],
                state["token_mask"][:1, 1:],
                players[:1, 1:],
                state["player_mask"][:1, 1:],
                prefix,
            )
            board = np.concatenate(
                [
                    active_board,
                    np.broadcast_to(opponent_board_once, (batch, BOARD_SLOTS - 1, HIDDEN)),
                ],
                axis=1,
            )
        else:
            players = (
                self._projection(
                    state["player_features"],
                    f"{prefix}.common_encoder.player_projection",
                    PLAYER_FEATURES,
                )
                * state["player_mask"][..., None]
            )
            board = self._encode_board_rows(
                state["token_features"],
                state["token_types"],
                state["token_mask"],
                players,
                state["player_mask"],
                prefix,
            )
        context = np.concatenate(
            [global_context[:, None, :], market[:, None, :], board + players], axis=1
        )
        context_mask = np.concatenate(
            [np.ones((batch, 2), dtype=np.bool_), state["player_mask"]], axis=1
        )
        context = self._self_block(context, context_mask, f"{prefix}.cross_board_blocks.0")
        return self._summary_projection(
            _masked_pool(context, context_mask), f"{prefix}.state_summary_projection"
        )

    def score_actions(
        self,
        parent: dict[str, np.ndarray],
        candidates: dict[str, np.ndarray],
        action_bytes: np.ndarray,
        exact_scores: np.ndarray,
    ) -> NumpyActionPrediction:
        return self.score_action_features(
            parent,
            candidates,
            decode_action_features(action_bytes),
            exact_scores,
        )

    def score_action_features(
        self,
        parent: dict[str, np.ndarray],
        candidates: dict[str, np.ndarray],
        action_features: np.ndarray,
        exact_scores: np.ndarray,
    ) -> NumpyActionPrediction:
        """Score already-decoded actions for MLX/NumPy checkpoint parity."""
        _validate_public_state(parent)
        _validate_public_state(candidates)
        count = candidates["token_features"].shape[0]
        action_features = np.asarray(action_features, dtype=np.float32)
        if parent["token_features"].shape[0] != 1 or action_features.shape != (
            count,
            ACTION_FEATURES,
        ):
            raise R2MapNumpyError("portable action scorer expects one exhaustive group")
        parent_encoded = self.encode_state(parent, "parent_encoder")[0]
        predicted = np.empty(count, dtype=np.float32)
        components = np.empty((count, SCORE_COMPONENTS), dtype=np.float32)
        policy = np.empty(count, dtype=np.float32)
        for start in range(0, count, self.candidate_chunk_size):
            stop = min(start + self.candidate_chunk_size, count)
            after = self.encode_state(
                {name: value[start:stop] for name, value in candidates.items()},
                "afterstate_encoder",
            )
            parent_rows = np.broadcast_to(parent_encoded, after.shape)
            action = _gelu(
                self._linear(
                    action_features[start:stop],
                    "action_projection.layers.0",
                    HIDDEN * 2,
                )
            )
            action = self._norm(action, "action_projection.layers.2", HIDDEN * 2)
            action = _gelu(self._linear(action, "action_projection.layers.3", HIDDEN))
            fused = np.concatenate(
                [parent_rows, after, after - parent_rows, after * parent_rows, action], axis=-1
            )
            fused = _gelu(self._linear(fused, "action_fusion.layers.0", HIDDEN * 8))
            fused = self._norm(fused, "action_fusion.layers.2", HIDDEN * 8)
            fused = _gelu(self._linear(fused, "action_fusion.layers.3", HIDDEN))
            fused = self._norm(fused, "action_fusion.layers.5", HIDDEN)
            hidden = _gelu(self._linear(fused, "action_trunk.layers.0", HIDDEN * 2))
            hidden = self._norm(hidden, "action_trunk.layers.2", HIDDEN * 2)
            hidden = _gelu(self._linear(hidden, "action_trunk.layers.3", HIDDEN))
            predicted[start:stop] = self._linear(hidden, "score_to_go_head", 1)[:, 0]
            action_multi = np.tanh(self._linear(hidden, "multitask_projection", 24))
            candidate_multi = np.tanh(self._linear(after, "multitask_projection", 24))
            policy[start:stop] = self._linear(action_multi, "bootstrap_policy_head", 1, bias=False)[
                :, 0
            ]
            components[start:stop] = self._linear(
                candidate_multi, "score_component_head", SCORE_COMPONENTS
            )
        scores = np.asarray(exact_scores, dtype=np.float32) + predicted
        if not all(np.all(np.isfinite(value)) for value in (scores, predicted, components, policy)):
            raise R2MapNumpyError("portable R2-MAP inference produced non-finite output")
        return NumpyActionPrediction(scores, predicted, components, policy)

    def score_market_decisions(
        self,
        parent: dict[str, np.ndarray],
        action_bytes: np.ndarray,
        exact_current_score: float,
    ) -> NumpyMarketPrediction:
        encoded = self.encode_state(parent, "parent_encoder")
        count = len(action_bytes)
        parent_rows = np.broadcast_to(encoded, (count, HIDDEN))
        action = _gelu(
            self._linear(
                decode_market_action_features(action_bytes),
                "market_decision_action_projection.layers.0",
                HIDDEN,
            )
        )
        action = self._norm(action, "market_decision_action_projection.layers.2", HIDDEN)
        hidden = np.concatenate([parent_rows, action, parent_rows * action], axis=-1)
        hidden = _gelu(self._linear(hidden, "market_decision_fusion.layers.0", HIDDEN * 2))
        hidden = self._norm(hidden, "market_decision_fusion.layers.2", HIDDEN * 2)
        hidden = _gelu(self._linear(hidden, "market_decision_fusion.layers.3", HIDDEN))
        hidden = self._norm(hidden, "market_decision_fusion.layers.5", HIDDEN)
        predicted = self._linear(hidden, "market_decision_score_to_go_head", 1)[:, 0]
        scores = np.float32(exact_current_score) + predicted
        if not np.all(np.isfinite(scores)):
            raise R2MapNumpyError("portable market inference produced non-finite output")
        return NumpyMarketPrediction(scores, predicted)

    def _validate_live_weights(self) -> None:
        # Force all graph-defining boundary tensors through the strict loader;
        # internal tensors are validated lazily with exact shapes at first use.
        for prefix in ("parent_encoder", "afterstate_encoder"):
            self._weight(f"{prefix}.latents", (LATENTS, HIDDEN))
        self._weight("score_to_go_head.weight", (1, HIDDEN))
        self._weight("score_to_go_head.bias", (1,))
        self._weight("market_decision_score_to_go_head.weight", (1, HIDDEN))
        self._weight("market_decision_score_to_go_head.bias", (1,))


def _validate_public_state(state: dict[str, np.ndarray]) -> None:
    required = {
        "token_features",
        "token_types",
        "token_mask",
        "market_features",
        "market_mask",
        "player_features",
        "player_mask",
        "global_features",
    }
    if set(state) != required:
        raise R2MapNumpyError("portable public state field set differs")
    batch = state["token_features"].shape[0]
    expected = {
        "token_features": (batch, BOARD_SLOTS, BOARD_TOKEN_CAPACITY, TOKEN_FEATURES),
        "token_types": (batch, BOARD_SLOTS, BOARD_TOKEN_CAPACITY),
        "token_mask": (batch, BOARD_SLOTS, BOARD_TOKEN_CAPACITY),
        "market_features": (batch, 4, MARKET_FEATURES),
        "market_mask": (batch, 4),
        "player_features": (batch, BOARD_SLOTS, PLAYER_FEATURES),
        "player_mask": (batch, BOARD_SLOTS),
        "global_features": (batch, GLOBAL_FEATURES),
    }
    if batch <= 0 or any(state[name].shape != shape for name, shape in expected.items()):
        raise R2MapNumpyError("portable public state tensor shape differs")
    if np.any((state["token_mask"] != 0) & (state["token_mask"] != 1)):
        raise R2MapNumpyError("portable token mask is not boolean")
