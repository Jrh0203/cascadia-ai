"""Iso-parameter MLX ranker for the ADR 0150 action-edit comparison."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import blake3
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from cascadia_mlx.dataset import ENTITY_DIM
from cascadia_mlx.graded_oracle_dataset import (
    GRADED_ORACLE_ACTION_DIM,
    GRADED_ORACLE_PRIOR_DIM,
)
from cascadia_mlx.graded_oracle_model import (
    GRADED_ORACLE_RESIDUAL_RANGE,
    GRADED_ORACLE_UNCERTAINTY_FLOOR,
)
from cascadia_mlx.r2_sparse_mlx_cache import (
    BOARD_SLOTS,
    BOARD_TOKEN_CAPACITY,
)
from cascadia_mlx.r2_sparse_mlx_model import (
    CommonStateEncoder,
    MaskedAttentionBlock,
    PerceiverCrossBlock,
    masked_pool,
    type_summary_tokens,
)
from cascadia_mlx.r3_action_edit_mlx_cache import (
    ARMS,
    R3_TOKEN_FEATURES,
)
from cascadia_mlx.s1_exact_supply_mlx_cache import (
    ARCHETYPE_COUNT,
    EXACT_SUPPLY_DIM,
    FRONTIER_FEATURE_DIM,
)

MODEL_SCHEMA_VERSION = 1
ARCHITECTURE = "r3-action-edit-iso-independent-ranker-v1"


@dataclass(frozen=True)
class R3ActionEditModelConfig:
    """The only model graph admitted to the ADR 0150 comparison."""

    schema_version: int = MODEL_SCHEMA_VERSION
    architecture: str = ARCHITECTURE
    arm: str = ARMS[0]
    hidden_dim: int = 64
    attention_heads: int = 4
    parent_perceiver_latents: int = 16
    candidate_perceiver_latents: int = 8
    parent_latent_blocks: int = 1
    candidate_latent_blocks: int = 1
    cross_board_blocks: int = 1
    staged_market_blocks: int = 1
    feed_forward_multiplier: int = 2

    def validate(self) -> None:
        if self.schema_version != MODEL_SCHEMA_VERSION or self.architecture != ARCHITECTURE:
            raise ValueError("unsupported R3 action-edit model schema")
        if self.arm not in ARMS:
            raise ValueError("R3 model names an unknown comparison arm")
        if self.hidden_dim != 64 or self.attention_heads != 4:
            raise ValueError("ADR 0150 freezes hidden width 64 and four heads")
        if self.parent_perceiver_latents != 16 or self.candidate_perceiver_latents != 8:
            raise ValueError("ADR 0150 Perceiver latent counts drifted")
        if (
            self.parent_latent_blocks != 1
            or self.candidate_latent_blocks != 1
            or self.cross_board_blocks != 1
            or self.staged_market_blocks != 1
        ):
            raise ValueError("ADR 0150 attention-block counts drifted")
        if self.feed_forward_multiplier != 2:
            raise ValueError("ADR 0150 freezes feed-forward multiplier two")

    def to_dict(self) -> dict[str, int | str]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> R3ActionEditModelConfig:
        config = cls(**values)
        config.validate()
        return config


@dataclass(frozen=True)
class R3ActionEditPrediction:
    """Independent candidate scores and calibrated standard-error estimates."""

    scores: mx.array
    residuals: mx.array
    standard_errors: mx.array


@dataclass(frozen=True)
class R3ActionEditEncoding:
    """Independent candidate hidden states before the frozen output heads."""

    hidden: mx.array
    candidate_mask: mx.array


class R2PerceiverParentEncoder(nn.Module):
    """The accepted R2 fixed-latent trunk without its value output head."""

    def __init__(self, config: R3ActionEditModelConfig):
        super().__init__()
        hidden = config.hidden_dim
        self.config = config
        self.common_encoder = CommonStateEncoder(hidden)
        self.latents = mx.random.normal((config.parent_perceiver_latents, hidden)) * 0.02
        self.perceiver_cross = PerceiverCrossBlock(
            hidden,
            config.attention_heads,
            config.feed_forward_multiplier,
        )
        self.perceiver_blocks = [
            MaskedAttentionBlock(
                hidden,
                config.attention_heads,
                config.feed_forward_multiplier,
            )
            for _ in range(config.parent_latent_blocks)
        ]
        self.board_summary_projection = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.cross_board_blocks = [
            MaskedAttentionBlock(
                hidden,
                config.attention_heads,
                config.feed_forward_multiplier,
            )
            for _ in range(config.cross_board_blocks)
        ]
        self.state_summary_projection = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )

    def __call__(self, batch: object) -> mx.array:
        tokens, players, market, global_context = self.common_encoder(
            batch.token_features,
            batch.token_mask,
            batch.market_features,
            batch.market_mask,
            batch.player_features,
            batch.player_mask,
            batch.global_features,
        )
        batch_size = tokens.shape[0]
        hidden = self.config.hidden_dim
        flat_tokens = tokens.reshape(
            batch_size * BOARD_SLOTS,
            BOARD_TOKEN_CAPACITY,
            hidden,
        )
        flat_types = batch.token_types.reshape(
            batch_size * BOARD_SLOTS,
            BOARD_TOKEN_CAPACITY,
        )
        flat_mask = batch.token_mask.reshape(
            batch_size * BOARD_SLOTS,
            BOARD_TOKEN_CAPACITY,
        )
        flat_players = players.reshape(batch_size * BOARD_SLOTS, 1, hidden)
        player_mask = batch.player_mask.reshape(batch_size * BOARD_SLOTS, 1)
        type_summaries, type_mask = type_summary_tokens(
            flat_tokens,
            flat_types,
            flat_mask,
        )
        inputs = mx.concatenate(
            [flat_players, type_summaries, flat_tokens],
            axis=1,
        )
        input_mask = mx.concatenate(
            [player_mask, type_mask, flat_mask],
            axis=1,
        )
        latents = mx.broadcast_to(
            self.latents[None, :, :],
            (
                batch_size * BOARD_SLOTS,
                self.config.parent_perceiver_latents,
                hidden,
            ),
        )
        latents = self.perceiver_cross(latents, inputs, input_mask)
        latent_mask = mx.ones(
            (
                batch_size * BOARD_SLOTS,
                self.config.parent_perceiver_latents,
            ),
            dtype=mx.bool_,
        )
        for block in self.perceiver_blocks:
            latents = block(latents, latent_mask)
        board_pooled = masked_pool(latents, latent_mask)
        board_summaries = self.board_summary_projection(board_pooled).reshape(
            batch_size,
            BOARD_SLOTS,
            hidden,
        )
        context = mx.concatenate(
            [
                global_context[:, None, :],
                market[:, None, :],
                board_summaries + players,
            ],
            axis=1,
        )
        context_mask = mx.concatenate(
            [
                mx.ones((batch_size, 2), dtype=mx.bool_),
                batch.player_mask,
            ],
            axis=1,
        )
        for block in self.cross_board_blocks:
            context = block(context, context_mask)
        return self.state_summary_projection(masked_pool(context, context_mask))


class CandidatePerceiverEncoder(nn.Module):
    """One shared fixed-latent encoder over control or R3 candidate tokens."""

    def __init__(self, config: R3ActionEditModelConfig):
        super().__init__()
        hidden = config.hidden_dim
        self.config = config
        self.token_projection = nn.Sequential(
            nn.Linear(R3_TOKEN_FEATURES, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.latents = mx.random.normal((config.candidate_perceiver_latents, hidden)) * 0.02
        self.cross = PerceiverCrossBlock(
            hidden,
            config.attention_heads,
            config.feed_forward_multiplier,
        )
        self.blocks = [
            MaskedAttentionBlock(
                hidden,
                config.attention_heads,
                config.feed_forward_multiplier,
            )
            for _ in range(config.candidate_latent_blocks)
        ]
        self.output = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )

    def __call__(
        self,
        token_features: mx.array,
        token_mask: mx.array,
    ) -> mx.array:
        groups, candidates, tokens, _ = token_features.shape
        hidden = self.config.hidden_dim
        flat_features = token_features.reshape(
            groups * candidates,
            tokens,
            R3_TOKEN_FEATURES,
        )
        flat_mask = token_mask.reshape(groups * candidates, tokens)
        encoded = self.token_projection(flat_features) * flat_mask[..., None]
        latents = mx.broadcast_to(
            self.latents[None, :, :],
            (
                groups * candidates,
                self.config.candidate_perceiver_latents,
                hidden,
            ),
        )
        latents = self.cross(latents, encoded, flat_mask)
        latent_mask = mx.ones(
            (
                groups * candidates,
                self.config.candidate_perceiver_latents,
            ),
            dtype=mx.bool_,
        )
        for block in self.blocks:
            latents = block(latents, latent_mask)
        return self.output(masked_pool(latents, latent_mask)).reshape(
            groups,
            candidates,
            hidden,
        )


class R3ActionEditRanker(nn.Module):
    """One parent encode and independent candidate scoring for every arm."""

    def __init__(self, config: R3ActionEditModelConfig | None = None):
        super().__init__()
        config = config or R3ActionEditModelConfig()
        config.validate()
        self.config = config
        hidden = config.hidden_dim
        self.parent_encoder = R2PerceiverParentEncoder(config)
        self.candidate_encoder = CandidatePerceiverEncoder(config)
        self.action_projection = nn.Sequential(
            nn.Linear(GRADED_ORACLE_ACTION_DIM, hidden * 2),
            nn.GELU(),
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
        )
        self.prior_projection = nn.Sequential(
            nn.Linear(GRADED_ORACLE_PRIOR_DIM, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.staged_market_projection = nn.Sequential(
            nn.Linear(ENTITY_DIM, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.staged_market_blocks = [
            MaskedAttentionBlock(
                hidden,
                config.attention_heads,
                config.feed_forward_multiplier,
            )
            for _ in range(config.staged_market_blocks)
        ]
        self.staged_market_summary = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.supply_projection = nn.Sequential(
            nn.Linear(EXACT_SUPPLY_DIM, hidden * 2),
            nn.GELU(),
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
        )
        self.staged_supply_projection = nn.Sequential(
            nn.Linear(EXACT_SUPPLY_DIM, hidden * 2),
            nn.GELU(),
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
        )
        self.archetype_embedding = nn.Embedding(ARCHETYPE_COUNT, hidden)
        self.frontier_projection = nn.Sequential(
            nn.Linear(FRONTIER_FEATURE_DIM, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.relation_projection = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2),
            nn.GELU(),
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
        )
        self.candidate_fusion = nn.Sequential(
            nn.Linear(hidden * 10, hidden * 4),
            nn.GELU(),
            nn.LayerNorm(hidden * 4),
            nn.Linear(hidden * 4, hidden),
            nn.GELU(),
        )
        self.output_trunk = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
        )
        self.residual_head = nn.Linear(hidden, 1)
        self.standard_error_head = nn.Linear(hidden, 1)
        self.residual_head.weight = mx.zeros_like(self.residual_head.weight)
        self.residual_head.bias = mx.zeros_like(self.residual_head.bias)
        self.standard_error_head.weight = mx.zeros_like(self.standard_error_head.weight)
        self.standard_error_head.bias = (
            mx.ones_like(self.standard_error_head.bias) * 0.541324854612918
        )

    def encode_parent(self, batch: object) -> mx.array:
        """Encode the exact public parent once per decision."""
        return self.parent_encoder(batch.parent)

    def encode_candidates(
        self,
        batch: object,
        *,
        candidate_slice: slice | None = None,
        parent_state: mx.array | None = None,
    ) -> R3ActionEditEncoding:
        parent = self.encode_parent(batch) if parent_state is None else parent_state
        selected = candidate_slice or slice(None)
        base = batch.base
        candidate_mask = base.candidate_mask[:, selected]
        spatial = self.candidate_encoder(
            batch.candidate_token_features[:, selected],
            batch.candidate_token_mask[:, selected],
        )
        action = self.action_projection(base.action_features[:, selected])
        prior = self.prior_projection(base.prior_features[:, selected])

        staged_market = self.staged_market_projection(base.staged_market_entities[:, selected])
        groups, candidates, market_width, hidden = staged_market.shape
        flat_market = staged_market.reshape(
            groups * candidates,
            market_width,
            hidden,
        )
        flat_market_mask = base.staged_market_mask[:, selected].reshape(
            groups * candidates,
            market_width,
        )
        for block in self.staged_market_blocks:
            flat_market = block(flat_market, flat_market_mask)
        staged_market_summary = self.staged_market_summary(
            masked_pool(flat_market, flat_market_mask)
        ).reshape(groups, candidates, hidden)

        supply = self.supply_projection(batch.supply_vector)
        supply_candidates = mx.broadcast_to(
            supply[:, None, :],
            (groups, candidates, hidden),
        )
        staged_supply = self.staged_supply_projection(batch.staged_supply_vector[:, selected])
        relation = self.relation_projection(
            mx.concatenate(
                [
                    self.archetype_embedding(batch.selected_archetype[:, selected]),
                    self.frontier_projection(batch.frontier_features[:, selected]),
                ],
                axis=-1,
            )
        )
        parent_candidates = mx.broadcast_to(
            parent[:, None, :],
            (groups, candidates, hidden),
        )
        fused = self.candidate_fusion(
            mx.concatenate(
                [
                    parent_candidates,
                    spatial,
                    action,
                    prior,
                    staged_market_summary,
                    supply_candidates,
                    staged_supply,
                    relation,
                    parent_candidates * spatial,
                    action * spatial,
                ],
                axis=-1,
            )
        )
        output = self.output_trunk(fused) * candidate_mask[..., None]
        return R3ActionEditEncoding(
            hidden=output,
            candidate_mask=candidate_mask,
        )

    def predict_from_encoding(
        self,
        batch: object,
        encoding: R3ActionEditEncoding,
        *,
        candidate_slice: slice | None = None,
    ) -> R3ActionEditPrediction:
        selected = candidate_slice or slice(None)
        groups, candidates, _ = encoding.hidden.shape
        if encoding.candidate_mask.shape != (groups, candidates):
            raise ValueError("R3 candidate encoding mask shape differs")
        raw_residual = self.residual_head(encoding.hidden).reshape(groups, candidates)
        residuals = (
            GRADED_ORACLE_RESIDUAL_RANGE
            * mx.tanh(raw_residual)
            * encoding.candidate_mask
        )
        standard_errors = (
            nn.softplus(
                self.standard_error_head(encoding.hidden).reshape(groups, candidates)
            )
            + 1e-4
        ) * encoding.candidate_mask
        return R3ActionEditPrediction(
            scores=batch.base.screen_value[:, selected] + residuals,
            residuals=residuals,
            standard_errors=standard_errors,
        )

    def predict(
        self,
        batch: object,
        *,
        candidate_slice: slice | None = None,
        parent_state: mx.array | None = None,
    ) -> R3ActionEditPrediction:
        encoding = self.encode_candidates(
            batch,
            candidate_slice=candidate_slice,
            parent_state=parent_state,
        )
        return self.predict_from_encoding(
            batch,
            encoding,
            candidate_slice=candidate_slice,
        )

    def __call__(self, batch: object) -> R3ActionEditPrediction:
        return self.predict(batch)


def r3_action_edit_loss_components(
    model: R3ActionEditRanker,
    batch: object,
) -> dict[str, mx.array]:
    """The frozen graded-oracle objective with no representation auxiliary."""
    prediction = model(batch)
    base = batch.base
    residuals = prediction.residuals
    scores = prediction.scores
    r1200_regression = _uncertainty_weighted_huber(
        residuals,
        base.r1200_mean - base.screen_value,
        base.r1200_stddev,
        base.r1200_samples,
        base.r1200_mask,
    )
    r4800_regression = _uncertainty_weighted_huber(
        residuals,
        base.r4800_mean - base.screen_value,
        base.r4800_stddev,
        base.r4800_samples,
        base.r4800_mask,
    )
    r1200_listwise = _masked_soft_target_cross_entropy(
        scores,
        base.r1200_mean,
        base.r1200_mask,
        temperature=2.0,
    )
    r4800_winner = _hard_cross_entropy(
        scores,
        base.candidate_mask,
        base.selected_index,
        temperature=1.0,
    )
    highest_mean = mx.where(
        base.r4800_mask,
        base.r4800_mean,
        mx.where(base.r1200_mask, base.r1200_mean, base.r600_mean),
    )
    highest_stddev = mx.where(
        base.r4800_mask,
        base.r4800_stddev,
        mx.where(base.r1200_mask, base.r1200_stddev, base.r600_stddev),
    )
    highest_samples = mx.where(
        base.r4800_mask,
        base.r4800_samples,
        mx.where(base.r1200_mask, base.r1200_samples, base.r600_samples),
    )
    scored_mask = base.r4800_mask | base.r1200_mask | base.r600_mask
    teacher_standard_error = _teacher_standard_error(
        highest_stddev,
        highest_samples,
    )
    predicted_variance = mx.maximum(prediction.standard_errors**2, 1e-8)
    gaussian = mx.log(mx.maximum(prediction.standard_errors, 1e-4)) + (
        teacher_standard_error**2 + (scores - highest_mean) ** 2
    ) / (2.0 * predicted_variance)
    standard_error_calibration = _masked_mean(gaussian, scored_mask)
    screen_only_regularization = _masked_mean(
        residuals**2,
        base.candidate_mask & ~scored_mask,
    )
    return {
        "r1200_huber": r1200_regression,
        "r4800_huber": r4800_regression,
        "r1200_listwise": r1200_listwise,
        "r4800_winner": r4800_winner,
        "standard_error_calibration": standard_error_calibration,
        "screen_only_regularization": screen_only_regularization,
    }


def r3_action_edit_loss(
    model: R3ActionEditRanker,
    batch: object,
) -> mx.array:
    components = r3_action_edit_loss_components(model, batch)
    return (
        components["r1200_huber"]
        + 4.0 * components["r4800_huber"]
        + 0.5 * components["r1200_listwise"]
        + components["r4800_winner"]
        + 0.1 * components["standard_error_calibration"]
        + 0.01 * components["screen_only_regularization"]
    )


def parameter_count(model: R3ActionEditRanker) -> int:
    return sum(int(value.size) for _, value in tree_flatten(model.trainable_parameters()))


def parameter_layout_blake3(model: R3ActionEditRanker) -> str:
    layout = [
        {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        for name, value in tree_flatten(model.trainable_parameters())
    ]
    payload = json.dumps(
        layout,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return blake3.blake3(payload).hexdigest()


def parameter_tensor_blake3(model: R3ActionEditRanker) -> str:
    digest = blake3.blake3()
    for name, value in tree_flatten(model.trainable_parameters()):
        array = mx.asarray(value)
        mx.eval(array)
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
        digest.update(bytes(memoryview(array)))
    return digest.hexdigest()


def _teacher_standard_error(
    stddev: mx.array,
    samples: mx.array,
) -> mx.array:
    empirical_variance = stddev**2 / mx.maximum(samples, 1.0)
    return mx.sqrt(empirical_variance + GRADED_ORACLE_UNCERTAINTY_FLOOR**2)


def _uncertainty_weighted_huber(
    prediction: mx.array,
    target: mx.array,
    stddev: mx.array,
    samples: mx.array,
    mask: mx.array,
) -> mx.array:
    uncertainty = _teacher_standard_error(stddev, samples)
    error = (prediction - target) / uncertainty
    absolute = mx.abs(error)
    huber = mx.where(
        absolute <= 1.0,
        0.5 * error**2,
        absolute - 0.5,
    )
    return _masked_mean(huber, mask)


def _masked_soft_target_cross_entropy(
    scores: mx.array,
    targets: mx.array,
    mask: mx.array,
    *,
    temperature: float,
) -> mx.array:
    masked_scores = mx.where(mask, scores / temperature, -1e9)
    masked_targets = mx.where(mask, targets / temperature, -1e9)
    target_probabilities = mx.softmax(masked_targets, axis=-1)
    log_probabilities = masked_scores - mx.logsumexp(
        masked_scores,
        axis=-1,
        keepdims=True,
    )
    per_group = -mx.sum(
        mx.where(mask, target_probabilities * log_probabilities, 0.0),
        axis=-1,
    )
    return _masked_mean(per_group, mx.any(mask, axis=-1))


def _hard_cross_entropy(
    scores: mx.array,
    mask: mx.array,
    selected_index: mx.array,
    *,
    temperature: float,
) -> mx.array:
    masked = mx.where(mask, scores / temperature, -1e9)
    selected_mask = mx.arange(scores.shape[-1])[None, :] == selected_index[:, None]
    selected_score = mx.sum(mx.where(selected_mask, masked, 0.0), axis=-1)
    return mx.mean(mx.logsumexp(masked, axis=-1) - selected_score)


def _masked_mean(values: mx.array, mask: mx.array) -> mx.array:
    weights = mask.astype(values.dtype)
    return mx.sum(values * weights) / mx.maximum(mx.sum(weights), 1.0)
