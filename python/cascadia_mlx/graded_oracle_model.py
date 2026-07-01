"""Frozen MLX residual ranker for complete-action graded-oracle supervision."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from cascadia_mlx.dataset import ENTITY_DIM, GLOBAL_DIM
from cascadia_mlx.graded_oracle_dataset import (
    GRADED_ORACLE_ACTION_DIM,
    GRADED_ORACLE_PRIOR_DIM,
    GRADED_ORACLE_PRIOR_SCHEMA,
    GRADED_ORACLE_PUBLIC_SUPPLY_SIZE,
)
from cascadia_mlx.model import SetAttentionBlock, _masked_pool

GRADED_ORACLE_ARCHITECTURE = "complete-action-graded-residual-v1"
GRADED_ORACLE_MODEL_SCHEMA_VERSION = 2
GRADED_ORACLE_RESIDUAL_RANGE = 12.0
GRADED_ORACLE_UNCERTAINTY_FLOOR = 1.0
GRADED_ORACLE_CANDIDATE_FACTOR_NAMES = (
    "action",
    "prior",
    "parent",
    "staged",
    "board_cross",
    "staged_cross",
    "action_parent_product",
)


@dataclass(frozen=True)
class GradedOracleModelConfig:
    """Serializable ADR 0081 architecture."""

    schema_version: int = GRADED_ORACLE_MODEL_SCHEMA_VERSION
    architecture: str = GRADED_ORACLE_ARCHITECTURE
    prior_feature_schema: str = GRADED_ORACLE_PRIOR_SCHEMA
    hidden_dim: int = 192
    attention_heads: int = 6
    board_blocks: int = 3
    market_blocks: int = 2
    feed_forward_multiplier: int = 4

    def validate(self) -> None:
        if (
            self.schema_version != GRADED_ORACLE_MODEL_SCHEMA_VERSION
            or self.architecture != GRADED_ORACLE_ARCHITECTURE
            or self.prior_feature_schema != GRADED_ORACLE_PRIOR_SCHEMA
        ):
            raise ValueError("unsupported graded-oracle model configuration")
        if self.hidden_dim <= 0 or self.hidden_dim % self.attention_heads:
            raise ValueError("hidden_dim must be positive and divisible by attention_heads")
        if self.board_blocks < 0 or self.market_blocks < 0:
            raise ValueError("block counts cannot be negative")
        if self.feed_forward_multiplier <= 0:
            raise ValueError("feed_forward_multiplier must be positive")

    def to_dict(self) -> dict[str, int | str]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> GradedOracleModelConfig:
        config = cls(**values)
        config.validate()
        return config


@dataclass(frozen=True)
class GradedOraclePrediction:
    """Point scores, bounded residuals, and positive rollout standard errors."""

    scores: mx.array
    residuals: mx.array
    standard_errors: mx.array


class GradedOracleRanker(nn.Module):
    """Score all legal actions in linear candidate-set memory."""

    def __init__(self, config: GradedOracleModelConfig | None = None):
        super().__init__()
        config = config or GradedOracleModelConfig()
        config.validate()
        self.config = config
        hidden = config.hidden_dim

        self.board_projection = nn.Sequential(
            nn.Linear(ENTITY_DIM, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.market_projection = nn.Sequential(
            nn.Linear(ENTITY_DIM, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.seat_embedding = nn.Embedding(4, hidden)
        self.board_blocks = [
            SetAttentionBlock(
                hidden,
                config.attention_heads,
                config.feed_forward_multiplier,
            )
            for _ in range(config.board_blocks)
        ]
        self.market_blocks = [
            SetAttentionBlock(
                hidden,
                config.attention_heads,
                config.feed_forward_multiplier,
            )
            for _ in range(config.market_blocks)
        ]
        self.global_projection = nn.Sequential(
            nn.Linear(GLOBAL_DIM, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
        )
        self.supply_projection = nn.Sequential(
            nn.Linear(GRADED_ORACLE_PUBLIC_SUPPLY_SIZE, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.parent_projection = nn.Sequential(
            nn.Linear(hidden * 12, hidden * 2),
            nn.GELU(),
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
        )
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
        self.staged_projection = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 2),
            nn.GELU(),
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
        )
        self.action_query_norm = nn.LayerNorm(hidden)
        self.board_cross_norm = nn.LayerNorm(hidden)
        self.staged_market_cross_norm = nn.LayerNorm(hidden)
        self.board_cross_attention = nn.MultiHeadAttention(
            hidden,
            config.attention_heads,
            bias=True,
        )
        self.staged_market_cross_attention = nn.MultiHeadAttention(
            hidden,
            config.attention_heads,
            bias=True,
        )
        self.candidate_projection = nn.Sequential(
            nn.Linear(hidden * 7, hidden * 3),
            nn.GELU(),
            nn.LayerNorm(hidden * 3),
            nn.Linear(hidden * 3, hidden),
            nn.GELU(),
        )
        self.output_trunk = nn.Sequential(
            nn.Linear(hidden * 4, hidden * 3),
            nn.GELU(),
            nn.LayerNorm(hidden * 3),
            nn.Linear(hidden * 3, hidden),
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

    def encode_candidate_factors(
        self,
        board_entities: mx.array,
        board_mask: mx.array,
        market_entities: mx.array,
        market_mask: mx.array,
        global_features: mx.array,
        public_supply: mx.array,
        action_features: mx.array,
        prior_features: mx.array,
        staged_market_entities: mx.array,
        staged_market_mask: mx.array,
        staged_public_supply: mx.array,
        screen_value: mx.array,
        candidate_mask: mx.array,
    ) -> tuple[mx.array, ...]:
        """Return the seven exact inputs to candidate_projection."""
        groups, candidates = screen_value.shape
        hidden = self.config.hidden_dim

        boards = self.board_projection(board_entities)
        boards = boards + self.seat_embedding(mx.arange(4))[None, :, None, :]
        boards = boards.reshape(groups * 4, 23, hidden)
        flat_board_mask = board_mask.reshape(groups * 4, 23)
        boards = boards * flat_board_mask[..., None]
        for block in self.board_blocks:
            boards = block(boards, flat_board_mask)
        board_summary = _masked_pool(boards, flat_board_mask).reshape(groups, hidden * 8)
        board_tokens = boards.reshape(groups, 4 * 23, hidden)
        board_token_mask = board_mask.reshape(groups, 4 * 23)

        market = self.market_projection(market_entities)
        for block in self.market_blocks:
            market = block(market, market_mask)
        market_summary = _masked_pool(market, market_mask)
        parent = self.parent_projection(
            mx.concatenate(
                [
                    board_summary,
                    market_summary,
                    self.global_projection(global_features),
                    self.supply_projection(public_supply),
                ],
                axis=-1,
            )
        )

        staged_market = self.market_projection(staged_market_entities)
        flat_staged_market = staged_market.reshape(groups * candidates, 4, hidden)
        flat_staged_mask = staged_market_mask.reshape(groups * candidates, 4)
        for block in self.market_blocks:
            flat_staged_market = block(flat_staged_market, flat_staged_mask)
        staged_market = flat_staged_market.reshape(groups, candidates, 4, hidden)
        staged_summary = _masked_pool(
            flat_staged_market,
            flat_staged_mask,
        ).reshape(groups, candidates, hidden * 2)
        staged = self.staged_projection(
            mx.concatenate(
                [
                    staged_summary,
                    self.supply_projection(staged_public_supply),
                ],
                axis=-1,
            )
        )

        action = self.action_projection(action_features)
        prior = self.prior_projection(prior_features)
        parent_candidates = mx.broadcast_to(
            parent[:, None, :],
            (groups, candidates, hidden),
        )
        query = self.action_query_norm(action + prior + parent_candidates + staged)

        board_attention_mask = mx.where(
            board_token_mask[:, None, None, :],
            0.0,
            -1e9,
        )
        board_cross = self.board_cross_attention(
            query,
            self.board_cross_norm(board_tokens),
            self.board_cross_norm(board_tokens),
            mask=board_attention_mask,
        )

        flat_query = query.reshape(groups * candidates, 1, hidden)
        staged_attention_mask = mx.where(
            flat_staged_mask[:, None, None, :],
            0.0,
            -1e9,
        )
        normalized_staged_market = self.staged_market_cross_norm(flat_staged_market)
        staged_cross = self.staged_market_cross_attention(
            flat_query,
            normalized_staged_market,
            normalized_staged_market,
            mask=staged_attention_mask,
        ).reshape(groups, candidates, hidden)

        return (
            action,
            prior,
            parent_candidates,
            staged,
            board_cross,
            staged_cross,
            action * parent_candidates,
        )

    def encode_candidates(
        self,
        board_entities: mx.array,
        board_mask: mx.array,
        market_entities: mx.array,
        market_mask: mx.array,
        global_features: mx.array,
        public_supply: mx.array,
        action_features: mx.array,
        prior_features: mx.array,
        staged_market_entities: mx.array,
        staged_market_mask: mx.array,
        staged_public_supply: mx.array,
        screen_value: mx.array,
        candidate_mask: mx.array,
        *,
        return_prepool: bool = False,
    ) -> mx.array:
        """Return either pre-pool candidates or the exact vectors used by both heads."""
        factors = self.encode_candidate_factors(
            board_entities,
            board_mask,
            market_entities,
            market_mask,
            global_features,
            public_supply,
            action_features,
            prior_features,
            staged_market_entities,
            staged_market_mask,
            staged_public_supply,
            screen_value,
            candidate_mask,
        )
        candidate = self.candidate_projection(mx.concatenate(factors, axis=-1))
        candidate = candidate * candidate_mask[..., None]
        if return_prepool:
            return candidate
        return self.encode_output_from_prepool(candidate, candidate_mask)

    def encode_output_from_prepool(
        self,
        candidate: mx.array,
        candidate_mask: mx.array,
    ) -> mx.array:
        """Apply the unchanged candidate-set pooling and output trunk."""
        hidden = self.config.hidden_dim
        candidate_pool = _masked_pool(candidate, candidate_mask)
        candidate_mean = mx.broadcast_to(
            candidate_pool[:, None, :hidden],
            candidate.shape,
        )
        candidate_maximum = mx.broadcast_to(
            candidate_pool[:, None, hidden:],
            candidate.shape,
        )
        output = self.output_trunk(
            mx.concatenate(
                [
                    candidate,
                    candidate_mean,
                    candidate_maximum,
                    candidate - candidate_mean,
                ],
                axis=-1,
            )
        )
        return output

    def __call__(
        self,
        board_entities: mx.array,
        board_mask: mx.array,
        market_entities: mx.array,
        market_mask: mx.array,
        global_features: mx.array,
        public_supply: mx.array,
        action_features: mx.array,
        prior_features: mx.array,
        staged_market_entities: mx.array,
        staged_market_mask: mx.array,
        staged_public_supply: mx.array,
        screen_value: mx.array,
        candidate_mask: mx.array,
    ) -> GradedOraclePrediction:
        groups, candidates = screen_value.shape
        output = self.encode_candidates(
            board_entities,
            board_mask,
            market_entities,
            market_mask,
            global_features,
            public_supply,
            action_features,
            prior_features,
            staged_market_entities,
            staged_market_mask,
            staged_public_supply,
            screen_value,
            candidate_mask,
        )
        raw_residual = self.residual_head(output).reshape(groups, candidates)
        residual = (
            GRADED_ORACLE_RESIDUAL_RANGE
            * mx.tanh(raw_residual)
            * candidate_mask
        )
        standard_error = (
            nn.softplus(self.standard_error_head(output).reshape(groups, candidates))
            + 1e-4
        )
        standard_error = standard_error * candidate_mask
        return GradedOraclePrediction(
            scores=screen_value + residual,
            residuals=residual,
            standard_errors=standard_error,
        )


def predict_graded_oracle_batch(
    model: GradedOracleRanker,
    batch: object,
) -> GradedOraclePrediction:
    """Run the full frozen feature path for one grouped batch."""
    return model(
        batch.board_entities,
        batch.board_mask,
        batch.market_entities,
        batch.market_mask,
        batch.global_features,
        batch.public_supply,
        batch.action_features,
        batch.prior_features,
        batch.staged_market_entities,
        batch.staged_market_mask,
        batch.staged_public_supply,
        batch.screen_value,
        batch.candidate_mask,
    )


def encode_graded_oracle_batch(
    model: GradedOracleRanker,
    batch: object,
) -> mx.array:
    """Return pre-head candidate embeddings without changing model semantics."""
    return model.encode_candidates(
        batch.board_entities,
        batch.board_mask,
        batch.market_entities,
        batch.market_mask,
        batch.global_features,
        batch.public_supply,
        batch.action_features,
        batch.prior_features,
        batch.staged_market_entities,
        batch.staged_market_mask,
        batch.staged_public_supply,
        batch.screen_value,
        batch.candidate_mask,
    )


def encode_graded_oracle_prepool_batch(
    model: GradedOracleRanker,
    batch: object,
) -> mx.array:
    """Return exact post-candidate-projection vectors before set pooling."""
    return model.encode_candidates(
        batch.board_entities,
        batch.board_mask,
        batch.market_entities,
        batch.market_mask,
        batch.global_features,
        batch.public_supply,
        batch.action_features,
        batch.prior_features,
        batch.staged_market_entities,
        batch.staged_market_mask,
        batch.staged_public_supply,
        batch.screen_value,
        batch.candidate_mask,
        return_prepool=True,
    )


def encode_graded_oracle_factor_batch(
    model: GradedOracleRanker,
    batch: object,
) -> mx.array:
    """Return the exact seven typed inputs to candidate_projection."""
    factors = model.encode_candidate_factors(
        batch.board_entities,
        batch.board_mask,
        batch.market_entities,
        batch.market_mask,
        batch.global_features,
        batch.public_supply,
        batch.action_features,
        batch.prior_features,
        batch.staged_market_entities,
        batch.staged_market_mask,
        batch.staged_public_supply,
        batch.screen_value,
        batch.candidate_mask,
    )
    return mx.stack(factors, axis=-2)


def score_graded_oracle_batch(model: GradedOracleRanker, batch: object) -> mx.array:
    """Return one finite point score per complete legal action."""
    return predict_graded_oracle_batch(model, batch).scores


def graded_oracle_loss_components(
    model: GradedOracleRanker,
    batch: object,
) -> dict[str, mx.array]:
    """Compute the six frozen ADR 0081 objective terms."""
    prediction = predict_graded_oracle_batch(model, batch)
    scores = prediction.scores
    residuals = prediction.residuals

    r1200_regression = _uncertainty_weighted_huber(
        residuals,
        batch.r1200_mean - batch.screen_value,
        batch.r1200_stddev,
        batch.r1200_samples,
        batch.r1200_mask,
    )
    r4800_regression = _uncertainty_weighted_huber(
        residuals,
        batch.r4800_mean - batch.screen_value,
        batch.r4800_stddev,
        batch.r4800_samples,
        batch.r4800_mask,
    )
    r1200_listwise = _masked_soft_target_cross_entropy(
        scores,
        batch.r1200_mean,
        batch.r1200_mask,
        temperature=2.0,
    )
    r4800_winner = _hard_cross_entropy(
        scores,
        batch.candidate_mask,
        batch.selected_index,
        temperature=1.0,
    )

    highest_mean = mx.where(
        batch.r4800_mask,
        batch.r4800_mean,
        mx.where(batch.r1200_mask, batch.r1200_mean, batch.r600_mean),
    )
    highest_stddev = mx.where(
        batch.r4800_mask,
        batch.r4800_stddev,
        mx.where(batch.r1200_mask, batch.r1200_stddev, batch.r600_stddev),
    )
    highest_samples = mx.where(
        batch.r4800_mask,
        batch.r4800_samples,
        mx.where(batch.r1200_mask, batch.r1200_samples, batch.r600_samples),
    )
    scored_mask = batch.r4800_mask | batch.r1200_mask | batch.r600_mask
    teacher_standard_error = _teacher_standard_error(
        highest_stddev,
        highest_samples,
    )
    predicted_variance = mx.maximum(prediction.standard_errors**2, 1e-8)
    gaussian = (
        mx.log(mx.maximum(prediction.standard_errors, 1e-4))
        + (
            teacher_standard_error**2
            + (scores - highest_mean) ** 2
        )
        / (2.0 * predicted_variance)
    )
    standard_error_calibration = _masked_mean(gaussian, scored_mask)

    screen_only = batch.candidate_mask & ~scored_mask
    screen_only_regularization = _masked_mean(residuals**2, screen_only)
    return {
        "r1200_huber": r1200_regression,
        "r4800_huber": r4800_regression,
        "r1200_listwise": r1200_listwise,
        "r4800_winner": r4800_winner,
        "standard_error_calibration": standard_error_calibration,
        "screen_only_regularization": screen_only_regularization,
    }


def graded_oracle_loss(model: GradedOracleRanker, batch: object) -> mx.array:
    """Apply the frozen ADR 0081 term weights."""
    components = graded_oracle_loss_components(model, batch)
    return (
        components["r1200_huber"]
        + 4.0 * components["r4800_huber"]
        + 0.5 * components["r1200_listwise"]
        + components["r4800_winner"]
        + 0.1 * components["standard_error_calibration"]
        + 0.01 * components["screen_only_regularization"]
    )


def load_promoted_graded_oracle_model(model_dir: str | Path) -> GradedOracleRanker:
    """Load a promoted graded-oracle model with its serialized architecture."""
    model_dir = Path(model_dir)
    manifest = json.loads((model_dir / "model.json").read_text())
    if manifest.get("status") != "promoted" or manifest.get("kind") != (
        "graded-oracle-ranking"
    ):
        raise ValueError("unsupported promoted graded-oracle model")
    model = GradedOracleRanker(
        GradedOracleModelConfig.from_dict(manifest["model_config"])
    )
    model.load_weights(str(model_dir / manifest["model"]["file"]))
    model.eval()
    return model


def _teacher_standard_error(stddev: mx.array, samples: mx.array) -> mx.array:
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
    huber = mx.where(absolute <= 1.0, 0.5 * error**2, absolute - 0.5)
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
    valid = mx.any(mask, axis=-1)
    return _masked_mean(per_group, valid)


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
