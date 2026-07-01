"""Width-192 exact-R2 Multitask Action Perceiver (R2-MAP).

The model consumes only the accepted public exact-R2 tensors.  It scores every
complete action from an exact full afterstate and freezes the gameplay equation
as ``exact_afterstate_score + predicted_score_to_go``.  Search, materialization,
loss routing, and checkpoint concerns intentionally live outside this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from cascadia_mlx.graded_oracle_dataset import (
    GRADED_ORACLE_ACTION_DIM,
    GRADED_ORACLE_MAX_WILDLIFE_WIPES,
)
from cascadia_mlx.opponent_intent_dataset import MARKET_SLOTS, OPPONENT_COUNT
from cascadia_mlx.r2_map_market_decision import MARKET_DECISION_FEATURE_DIM
from cascadia_mlx.r2_map_tensor_contract import (
    BOARD_OWNERSHIP_ENCODING,
    BOARD_SLOTS,
    BOARD_TOKEN_CAPACITY,
    GLOBAL_FEATURES,
    MARKET_FEATURES,
    PLAYER_FEATURES,
    TARGET_DIM,
    TOKEN_FEATURES,
)
from cascadia_mlx.r2_sparse_mlx_model import (
    CommonStateEncoder,
    MaskedAttentionBlock,
    PerceiverCrossBlock,
    masked_pool,
    type_summary_tokens,
)

MODEL_SCHEMA_VERSION = 3
ARCHITECTURE = "r2-map-v1.1"
MULTITASK_SCHEMA = "r2-map-selected-afterstate-complete-opponent-action-v3"
REFERENCE_PRECISION = "float32"
MULTITASK_DIM = 24
PARAMETER_MIN = 4_000_000
PARAMETER_MAX = 8_000_000
EXACT_PARAMETER_COUNT = 4_531_853

# This is a mechanical capacity-sizing width, not an attention topology change.
# The accepted Perceiver feed-forward multiplier remains exactly two.
ACTION_FUSION_EXPANSION = 8

PUBLIC_STATE_TENSOR_NAMES = (
    "token_features",
    "token_types",
    "token_mask",
    "market_features",
    "market_mask",
    "player_features",
    "player_mask",
    "global_features",
)
PUBLIC_ACTION_TENSOR_NAMES = (
    "candidate_mask",
    "action_features",
    "exact_afterstate_scores",
)
PUBLIC_MARKET_DECISION_TENSOR_NAMES = (
    "action_mask",
    "action_features",
    "exact_current_scores",
)
FORBIDDEN_MODEL_INPUT_NAMES = frozenset(
    {
        "hidden_order",
        "future_refill",
        "policy_id",
        "policy_code",
        "game_id",
        "game_index",
        "host",
        "host_id",
        "split",
        "split_id",
    }
)


@dataclass(frozen=True)
class R2MapModelConfig:
    """Frozen graph identity for the first expert-iteration baseline."""

    schema_version: int = MODEL_SCHEMA_VERSION
    architecture: str = ARCHITECTURE
    multitask_schema: str = MULTITASK_SCHEMA
    hidden_dim: int = 192
    attention_heads: int = 4
    board_latents: int = 16
    board_latent_blocks: int = 1
    cross_board_blocks: int = 1
    feed_forward_multiplier: int = 2
    action_fusion_expansion: int = ACTION_FUSION_EXPANSION
    multitask_dim: int = MULTITASK_DIM
    precision: str = REFERENCE_PRECISION
    board_ownership_encoding: str = BOARD_OWNERSHIP_ENCODING
    action_feature_dim: int = GRADED_ORACLE_ACTION_DIM
    score_component_dim: int = TARGET_DIM
    opponent_count: int = OPPONENT_COUNT
    market_slots: int = MARKET_SLOTS
    score_to_go_enabled: bool = True
    score_components_enabled: bool = True
    bootstrap_policy_enabled: bool = True
    opponent_next_action_enabled: bool = True
    market_survival_enabled: bool = True
    market_decision_value_enabled: bool = True
    uncertainty_enabled: bool = False
    legal_affordance_enabled: bool = False
    public_transition_enabled: bool = False

    def validate(self) -> None:
        if (
            self.schema_version != MODEL_SCHEMA_VERSION
            or self.architecture != ARCHITECTURE
            or self.multitask_schema != MULTITASK_SCHEMA
        ):
            raise ValueError("unsupported R2-MAP model schema")
        if self.hidden_dim != 192 or self.attention_heads != 4:
            raise ValueError("R2-MAP freezes width 192 and four attention heads")
        if self.board_latents != 16:
            raise ValueError("R2-MAP freezes 16 board latents")
        if self.board_latent_blocks != 1 or self.cross_board_blocks != 1:
            raise ValueError("R2-MAP freezes one board block and one cross-board block")
        if self.feed_forward_multiplier != 2:
            raise ValueError("R2-MAP freezes the accepted feed-forward multiplier at two")
        if self.action_fusion_expansion != ACTION_FUSION_EXPANSION:
            raise ValueError("R2-MAP action-fusion sizing drifted")
        if self.multitask_dim != MULTITASK_DIM:
            raise ValueError("R2-MAP freezes the multitask bottleneck at 24")
        if self.precision != REFERENCE_PRECISION:
            raise ValueError("R2-MAP reference inference is float32 only")
        if self.board_ownership_encoding != BOARD_OWNERSHIP_ENCODING:
            raise ValueError("R2-MAP requires explicit relative-seat board ownership")
        if self.action_feature_dim != 140 or self.score_component_dim != 11:
            raise ValueError("R2-MAP lossless action or score-component dimensions drifted")
        if self.opponent_count != 3 or self.market_slots != 4:
            raise ValueError("R2-MAP freezes three ordered opponents and four market slots")
        if not all(
            (
                self.score_to_go_enabled,
                self.score_components_enabled,
                self.bootstrap_policy_enabled,
                self.opponent_next_action_enabled,
                self.market_survival_enabled,
                self.market_decision_value_enabled,
            )
        ):
            raise ValueError("an active R2-MAP v1 head was disabled")
        if any(
            (
                self.uncertainty_enabled,
                self.legal_affordance_enabled,
                self.public_transition_enabled,
            )
        ):
            raise ValueError("a deferred R2-MAP head was enabled without a matched ablation")

    def to_dict(self) -> dict[str, int | str | bool]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> R2MapModelConfig:
        config = cls(**values)
        config.validate()
        return config


@dataclass(frozen=True, slots=True)
class R2MapPublicState:
    """The complete and exclusive public exact-R2 model input surface.

    Parent tensors have leading shape ``[groups]``. Candidate tensors have
    leading shape ``[groups, candidates]``. No identity or future-information
    fields can be attached because the dataclass is frozen and slotted.
    """

    token_features: mx.array
    token_types: mx.array
    token_mask: mx.array
    market_features: mx.array
    market_mask: mx.array
    player_features: mx.array
    player_mask: mx.array
    global_features: mx.array

    def validate(self, *, candidates: bool) -> tuple[int, ...]:
        leading_rank = 2 if candidates else 1
        expected_token_rank = leading_rank + 3
        if self.token_features.ndim != expected_token_rank:
            label = "candidate" if candidates else "parent"
            raise ValueError(f"R2-MAP {label} token tensor rank drifted")
        leading = tuple(self.token_features.shape[:leading_rank])
        expected_shapes = {
            "token_features": (*leading, BOARD_SLOTS, BOARD_TOKEN_CAPACITY, TOKEN_FEATURES),
            "token_types": (*leading, BOARD_SLOTS, BOARD_TOKEN_CAPACITY),
            "token_mask": (*leading, BOARD_SLOTS, BOARD_TOKEN_CAPACITY),
            "market_features": (*leading, MARKET_SLOTS, MARKET_FEATURES),
            "market_mask": (*leading, MARKET_SLOTS),
            "player_features": (*leading, BOARD_SLOTS, PLAYER_FEATURES),
            "player_mask": (*leading, BOARD_SLOTS),
            "global_features": (*leading, GLOBAL_FEATURES),
        }
        for name, expected in expected_shapes.items():
            observed = tuple(getattr(self, name).shape)
            if observed != expected:
                raise ValueError(f"R2-MAP public tensor {name} shape {observed} != {expected}")
        for name in ("token_features", "market_features", "player_features", "global_features"):
            if getattr(self, name).dtype != mx.float32:
                raise ValueError(f"R2-MAP public tensor {name} must be float32")
        if self.token_types.dtype != mx.int32:
            raise ValueError("R2-MAP token_types must be int32")
        for name in ("token_mask", "market_mask", "player_mask"):
            if getattr(self, name).dtype != mx.bool_:
                raise ValueError(f"R2-MAP public tensor {name} must be bool")
        return leading

    def flatten_candidates(self) -> R2MapPublicState:
        groups, candidates = self.validate(candidates=True)

        def flatten(value: mx.array) -> mx.array:
            return value.reshape(groups * candidates, *value.shape[2:])

        return R2MapPublicState(
            token_features=flatten(self.token_features),
            token_types=flatten(self.token_types),
            token_mask=flatten(self.token_mask),
            market_features=flatten(self.market_features),
            market_mask=flatten(self.market_mask),
            player_features=flatten(self.player_features),
            player_mask=flatten(self.player_mask),
            global_features=flatten(self.global_features),
        )


@dataclass(frozen=True, slots=True)
class R2MapBatch:
    """One decision group with its exhaustive exact full-afterstate set."""

    parent: R2MapPublicState
    candidates: R2MapPublicState
    candidate_mask: mx.array
    action_features: mx.array
    exact_afterstate_scores: mx.array

    def validate(self) -> tuple[int, int]:
        parent_leading = self.parent.validate(candidates=False)
        candidate_leading = self.candidates.validate(candidates=True)
        groups = parent_leading[0]
        if candidate_leading[0] != groups:
            raise ValueError("R2-MAP parent and candidate group counts differ")
        candidates = candidate_leading[1]
        expected = (groups, candidates)
        if tuple(self.candidate_mask.shape) != expected or self.candidate_mask.dtype != mx.bool_:
            raise ValueError("R2-MAP candidate_mask shape or dtype drifted")
        if tuple(self.action_features.shape) != (*expected, GRADED_ORACLE_ACTION_DIM):
            raise ValueError("R2-MAP lossless complete-action feature shape drifted")
        if self.action_features.dtype != mx.float32:
            raise ValueError("R2-MAP complete-action features must be float32")
        if tuple(self.exact_afterstate_scores.shape) != expected:
            raise ValueError("R2-MAP exact afterstate score shape drifted")
        if self.exact_afterstate_scores.dtype != mx.float32:
            raise ValueError("R2-MAP exact afterstate scores must be float32")
        if not bool(mx.all(mx.any(self.candidate_mask, axis=1)).item()):
            raise ValueError("every R2-MAP decision must contain at least one legal candidate")
        return groups, candidates


@dataclass(frozen=True, slots=True)
class R2MapMarketDecisionBatch:
    """Public pre-refill decision groups and every currently legal action."""

    public_state: R2MapPublicState
    action_mask: mx.array
    action_features: mx.array
    exact_current_scores: mx.array

    def validate(self) -> tuple[int, int]:
        leading = self.public_state.validate(candidates=False)
        groups = leading[0]
        if self.action_mask.ndim != 2 or self.action_mask.shape[0] != groups:
            raise ValueError("R2-MAP market-decision action mask shape drifted")
        actions = self.action_mask.shape[1]
        if self.action_mask.dtype != mx.bool_ or not bool(
            mx.all(mx.any(self.action_mask, axis=1)).item()
        ):
            raise ValueError("every market decision must contain a legal action")
        if tuple(self.action_features.shape) != (groups, actions, MARKET_DECISION_FEATURE_DIM):
            raise ValueError("R2-MAP market-decision feature shape drifted")
        if self.action_features.dtype != mx.float32:
            raise ValueError("R2-MAP market-decision features must be float32")
        if tuple(self.exact_current_scores.shape) != (groups,):
            raise ValueError("R2-MAP market-decision current score shape drifted")
        if self.exact_current_scores.dtype != mx.float32:
            raise ValueError("R2-MAP market-decision current scores must be float32")
        return groups, actions


@dataclass(frozen=True)
class R2MapOpponentNextAction:
    """Factorized ordered next-public-action logits for all three opponents."""

    tile_slot_logits: mx.array
    wildlife_slot_logits: mx.array
    draft_kind_logits: mx.array
    drafted_wildlife_logits: mx.array
    replace_three_logits: mx.array
    paid_wipe_count_logits: mx.array
    paid_wipe_mask_logits: mx.array


@dataclass(frozen=True)
class R2MapMarketSurvival:
    """Four-tile disposition, pair-survival, and final-slot logits."""

    disposition_logits: mx.array
    pair_survival_logits: mx.array
    final_slot_logits: mx.array


@dataclass(frozen=True)
class R2MapPrediction:
    """Active v1 outputs; deferred heads are absent by construction."""

    action_scores: mx.array
    predicted_score_to_go: mx.array
    predicted_score_components_to_go: mx.array
    bootstrap_policy_logits: mx.array
    opponent_next_action: R2MapOpponentNextAction
    market_survival: R2MapMarketSurvival
    candidate_mask: mx.array


@dataclass(frozen=True)
class R2MapActionPrediction:
    """Live inference outputs; training-only auxiliaries are not evaluated."""

    action_scores: mx.array
    predicted_score_to_go: mx.array
    predicted_score_components_to_go: mx.array
    bootstrap_policy_logits: mx.array
    candidate_mask: mx.array


@dataclass(frozen=True)
class R2MapMarketDecisionPrediction:
    """Scalar public decision values before any hidden refill is observed."""

    action_scores: mx.array
    predicted_score_to_go: mx.array
    bootstrap_policy_logits: mx.array
    action_mask: mx.array


class ExactR2PerceiverEncoder(nn.Module):
    """Accepted exact-R2 adapters and fixed-latent topology at width 192."""

    def __init__(self, config: R2MapModelConfig):
        super().__init__()
        hidden = config.hidden_dim
        self.config = config
        self.common_encoder = CommonStateEncoder(
            hidden,
            board_token_capacity=BOARD_TOKEN_CAPACITY,
        )
        self.latents = mx.random.normal((config.board_latents, hidden)) * 0.02
        self.perceiver_cross = PerceiverCrossBlock(
            hidden,
            config.attention_heads,
            config.feed_forward_multiplier,
        )
        self.board_blocks = [
            MaskedAttentionBlock(
                hidden,
                config.attention_heads,
                config.feed_forward_multiplier,
            )
            for _ in range(config.board_latent_blocks)
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

    def __call__(self, state: R2MapPublicState) -> mx.array:
        state.validate(candidates=False)
        tokens, players, market, global_context = self.common_encoder(
            state.token_features,
            state.token_mask,
            state.market_features,
            state.market_mask,
            state.player_features,
            state.player_mask,
            state.global_features,
        )
        batch_size = tokens.shape[0]
        hidden = self.config.hidden_dim
        flat_tokens = tokens.reshape(
            batch_size * BOARD_SLOTS,
            BOARD_TOKEN_CAPACITY,
            hidden,
        )
        flat_types = state.token_types.reshape(
            batch_size * BOARD_SLOTS,
            BOARD_TOKEN_CAPACITY,
        )
        flat_mask = state.token_mask.reshape(
            batch_size * BOARD_SLOTS,
            BOARD_TOKEN_CAPACITY,
        )
        flat_players = players.reshape(batch_size * BOARD_SLOTS, 1, hidden)
        player_mask = state.player_mask.reshape(batch_size * BOARD_SLOTS, 1)
        type_summaries, type_mask = type_summary_tokens(flat_tokens, flat_types, flat_mask)
        inputs = mx.concatenate([flat_players, type_summaries, flat_tokens], axis=1)
        input_mask = mx.concatenate([player_mask, type_mask, flat_mask], axis=1)
        latents = mx.broadcast_to(
            self.latents[None, :, :],
            (batch_size * BOARD_SLOTS, self.config.board_latents, hidden),
        )
        latents = self.perceiver_cross(latents, inputs, input_mask)
        latent_mask = mx.ones(
            (batch_size * BOARD_SLOTS, self.config.board_latents),
            dtype=mx.bool_,
        )
        for block in self.board_blocks:
            latents = block(latents, latent_mask)
        board_summaries = self.board_summary_projection(masked_pool(latents, latent_mask)).reshape(
            batch_size, BOARD_SLOTS, hidden
        )
        context = mx.concatenate(
            [global_context[:, None, :], market[:, None, :], board_summaries + players],
            axis=1,
        )
        context_mask = mx.concatenate(
            [mx.ones((batch_size, 2), dtype=mx.bool_), state.player_mask],
            axis=1,
        )
        for block in self.cross_board_blocks:
            context = block(context, context_mask)
        return self.state_summary_projection(masked_pool(context, context_mask))


class R2MapModel(nn.Module):
    """Independent complete-action scorer with training-only auxiliary heads."""

    def __init__(self, config: R2MapModelConfig | None = None):
        super().__init__()
        config = config or R2MapModelConfig()
        config.validate()
        self.config = config
        hidden = config.hidden_dim
        self.parent_encoder = ExactR2PerceiverEncoder(config)
        self.afterstate_encoder = ExactR2PerceiverEncoder(config)
        self.action_projection = nn.Sequential(
            nn.Linear(config.action_feature_dim, hidden * 2),
            nn.GELU(),
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
        )
        fusion_hidden = hidden * config.action_fusion_expansion
        self.action_fusion = nn.Sequential(
            nn.Linear(hidden * 5, fusion_hidden),
            nn.GELU(),
            nn.LayerNorm(fusion_hidden),
            nn.Linear(fusion_hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.action_trunk = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
        )
        self.score_to_go_head = nn.Linear(hidden, 1)

        self.multitask_projection = nn.Linear(hidden, config.multitask_dim)
        self.bootstrap_policy_head = nn.Linear(config.multitask_dim, 1, bias=False)
        self.score_component_head = nn.Linear(config.multitask_dim, config.score_component_dim)
        self.opponent_embedding = nn.Embedding(config.opponent_count, config.multitask_dim)
        self.opponent_tile_slot_head = nn.Linear(config.multitask_dim, MARKET_SLOTS)
        self.opponent_wildlife_slot_head = nn.Linear(config.multitask_dim, MARKET_SLOTS)
        self.opponent_draft_kind_head = nn.Linear(config.multitask_dim, 2)
        self.opponent_drafted_wildlife_head = nn.Linear(config.multitask_dim, 5)
        self.opponent_replace_three_head = nn.Linear(config.multitask_dim, 2)
        self.opponent_paid_wipe_count_head = nn.Linear(
            config.multitask_dim, GRADED_ORACLE_MAX_WILDLIFE_WIPES + 1
        )
        self.opponent_wipe_ordinal_embedding = nn.Embedding(
            GRADED_ORACLE_MAX_WILDLIFE_WIPES, config.multitask_dim
        )
        self.opponent_paid_wipe_mask_head = nn.Linear(config.multitask_dim, 16)
        self.market_slot_embedding = nn.Embedding(config.market_slots, config.multitask_dim)
        self.market_disposition_head = nn.Linear(config.multitask_dim, 4)
        self.market_pair_survival_head = nn.Linear(config.multitask_dim, 2)
        self.market_final_slot_head = nn.Linear(config.multitask_dim, MARKET_SLOTS)

        self.market_decision_action_projection = nn.Sequential(
            nn.Linear(MARKET_DECISION_FEATURE_DIM, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.market_decision_fusion = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 2),
            nn.GELU(),
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.market_decision_score_to_go_head = nn.Linear(hidden, 1)
        self.market_decision_bootstrap_policy_head = nn.Linear(hidden, 1, bias=False)

        # A fresh model starts as the exact score baseline; training must earn
        # every predicted remaining point and policy preference.
        self.score_to_go_head.weight = mx.zeros_like(self.score_to_go_head.weight)
        self.score_to_go_head.bias = mx.zeros_like(self.score_to_go_head.bias)
        self.bootstrap_policy_head.weight = mx.zeros_like(self.bootstrap_policy_head.weight)
        self.market_decision_score_to_go_head.weight = mx.zeros_like(
            self.market_decision_score_to_go_head.weight
        )
        self.market_decision_score_to_go_head.bias = mx.zeros_like(
            self.market_decision_score_to_go_head.bias
        )
        self.market_decision_bootstrap_policy_head.weight = mx.zeros_like(
            self.market_decision_bootstrap_policy_head.weight
        )

        parameters = parameter_count(self)
        if parameters != EXACT_PARAMETER_COUNT:
            raise ValueError(
                f"R2-MAP v1.1 parameter count {parameters} differs from {EXACT_PARAMETER_COUNT}"
            )

    def encode(self, batch: R2MapBatch) -> tuple[mx.array, mx.array, mx.array]:
        groups, candidates = batch.validate()
        parent = self.parent_encoder(batch.parent)
        candidate = self.afterstate_encoder(batch.candidates.flatten_candidates()).reshape(
            groups, candidates, self.config.hidden_dim
        )
        parent_expanded = mx.broadcast_to(
            parent[:, None, :],
            (groups, candidates, self.config.hidden_dim),
        )
        action = self.action_projection(batch.action_features)
        fusion = self.action_fusion(
            mx.concatenate(
                [
                    parent_expanded,
                    candidate,
                    candidate - parent_expanded,
                    candidate * parent_expanded,
                    action,
                ],
                axis=-1,
            )
        )
        hidden = self.action_trunk(fusion) * batch.candidate_mask[..., None]
        return parent, candidate, hidden

    def __call__(
        self,
        batch: R2MapBatch,
        *,
        selected_auxiliary_index: mx.array | None = None,
    ) -> R2MapPrediction:
        _parent, candidate, hidden = self.encode(batch)
        groups, candidates = batch.candidate_mask.shape
        valid = batch.candidate_mask
        predicted_to_go = self.score_to_go_head(hidden).reshape(groups, candidates) * valid
        action_scores = mx.where(
            valid,
            estimated_final_score(batch.exact_afterstate_scores, predicted_to_go),
            -mx.inf,
        )
        action_multitask = self.multitask_state(hidden)
        bootstrap_policy = mx.where(
            valid,
            self.bootstrap_policy_head(action_multitask).reshape(groups, candidates),
            -mx.inf,
        )

        candidate_multitask = self.multitask_state(candidate)
        auxiliary_valid = valid
        if selected_auxiliary_index is not None:
            if tuple(selected_auxiliary_index.shape) != (groups,):
                raise ValueError("R2-MAP selected auxiliary index shape drifted")
            selection = mx.arange(candidates)[None, :] == selected_auxiliary_index[:, None]
            candidate_multitask = mx.sum(
                candidate_multitask * selection[..., None], axis=1, keepdims=True
            )
            auxiliary_valid = mx.ones((groups, 1), dtype=mx.bool_)
        components = self.score_component_head(candidate_multitask) * auxiliary_valid[..., None]
        opponents = (
            candidate_multitask[:, :, None, :]
            + self.opponent_embedding(mx.arange(self.config.opponent_count))[None, None, :, :]
        )
        wipe_states = (
            opponents[:, :, :, None, :]
            + self.opponent_wipe_ordinal_embedding(mx.arange(GRADED_ORACLE_MAX_WILDLIFE_WIPES))[
                None, None, None, :, :
            ]
        )
        market = (
            candidate_multitask[:, :, None, :]
            + self.market_slot_embedding(mx.arange(self.config.market_slots))[None, None, :, :]
        )
        candidate_aux_mask = auxiliary_valid[..., None, None]
        return R2MapPrediction(
            action_scores=action_scores,
            predicted_score_to_go=predicted_to_go,
            predicted_score_components_to_go=components,
            bootstrap_policy_logits=bootstrap_policy,
            opponent_next_action=R2MapOpponentNextAction(
                tile_slot_logits=self.opponent_tile_slot_head(opponents) * candidate_aux_mask,
                wildlife_slot_logits=self.opponent_wildlife_slot_head(opponents)
                * candidate_aux_mask,
                draft_kind_logits=self.opponent_draft_kind_head(opponents) * candidate_aux_mask,
                drafted_wildlife_logits=self.opponent_drafted_wildlife_head(opponents)
                * candidate_aux_mask,
                replace_three_logits=self.opponent_replace_three_head(opponents)
                * candidate_aux_mask,
                paid_wipe_count_logits=self.opponent_paid_wipe_count_head(opponents)
                * candidate_aux_mask,
                paid_wipe_mask_logits=self.opponent_paid_wipe_mask_head(wipe_states)
                * auxiliary_valid[..., None, None, None],
            ),
            market_survival=R2MapMarketSurvival(
                disposition_logits=self.market_disposition_head(market) * candidate_aux_mask,
                pair_survival_logits=self.market_pair_survival_head(market) * candidate_aux_mask,
                final_slot_logits=self.market_final_slot_head(market) * candidate_aux_mask,
            ),
            candidate_mask=valid,
        )

    def score_actions(self, batch: R2MapBatch) -> R2MapActionPrediction:
        """Score a complete legal draft screen without training-only heads."""
        _parent, candidate, hidden = self.encode(batch)
        groups, candidates = batch.candidate_mask.shape
        valid = batch.candidate_mask
        predicted = self.score_to_go_head(hidden).reshape(groups, candidates) * valid
        action_multitask = self.multitask_state(hidden)
        candidate_multitask = self.multitask_state(candidate)
        return R2MapActionPrediction(
            action_scores=mx.where(
                valid,
                estimated_final_score(batch.exact_afterstate_scores, predicted),
                -mx.inf,
            ),
            predicted_score_to_go=predicted,
            predicted_score_components_to_go=(
                self.score_component_head(candidate_multitask) * valid[..., None]
            ),
            bootstrap_policy_logits=mx.where(
                valid,
                self.bootstrap_policy_head(action_multitask).reshape(groups, candidates),
                -mx.inf,
            ),
            candidate_mask=valid,
        )

    def multitask_state(self, encoded: mx.array) -> mx.array:
        """Project any shared width-192 state into the frozen 24-d tanh subspace."""
        if encoded.shape[-1] != self.config.hidden_dim:
            raise ValueError("R2-MAP multitask input width drifted")
        return mx.tanh(self.multitask_projection(encoded))

    def score_market_decisions(
        self, batch: R2MapMarketDecisionBatch
    ) -> R2MapMarketDecisionPrediction:
        """Score every legal pre-refill action from public information only."""
        groups, actions = batch.validate()
        parent = self.parent_encoder(batch.public_state)
        parent = mx.broadcast_to(parent[:, None, :], (groups, actions, self.config.hidden_dim))
        action = self.market_decision_action_projection(batch.action_features)
        hidden = (
            self.market_decision_fusion(mx.concatenate([parent, action, parent * action], axis=-1))
            * batch.action_mask[..., None]
        )
        predicted = self.market_decision_score_to_go_head(hidden).reshape(groups, actions)
        predicted = predicted * batch.action_mask
        scores = mx.where(
            batch.action_mask,
            batch.exact_current_scores[:, None] + predicted,
            -mx.inf,
        )
        policy = mx.where(
            batch.action_mask,
            self.market_decision_bootstrap_policy_head(hidden).reshape(groups, actions),
            -mx.inf,
        )
        return R2MapMarketDecisionPrediction(
            action_scores=scores,
            predicted_score_to_go=predicted,
            bootstrap_policy_logits=policy,
            action_mask=batch.action_mask,
        )


def estimated_final_score(
    exact_afterstate_score: mx.array,
    predicted_score_to_go: mx.array,
) -> mx.array:
    """The frozen inference identity used to rank complete legal actions."""
    if exact_afterstate_score.shape != predicted_score_to_go.shape:
        raise ValueError("exact afterstate and predicted score-to-go shapes differ")
    return exact_afterstate_score + predicted_score_to_go


def parameter_count(model: nn.Module) -> int:
    return sum(int(value.size) for _, value in tree_flatten(model.trainable_parameters()))


def tensor_contract_manifest(config: R2MapModelConfig | None = None) -> dict[str, Any]:
    """Return the public-only, versioned tensor and output contract."""
    config = config or R2MapModelConfig()
    config.validate()
    return {
        "schema_version": MODEL_SCHEMA_VERSION,
        "architecture": ARCHITECTURE,
        "precision": REFERENCE_PRECISION,
        "parameter_count": EXACT_PARAMETER_COUNT,
        "parent_state_tensors": list(PUBLIC_STATE_TENSOR_NAMES),
        "candidate_afterstate_tensors": list(PUBLIC_STATE_TENSOR_NAMES),
        "action_tensors": list(PUBLIC_ACTION_TENSOR_NAMES),
        "market_decision_tensors": list(PUBLIC_MARKET_DECISION_TENSOR_NAMES),
        "forbidden_inputs": sorted(FORBIDDEN_MODEL_INPUT_NAMES),
        "state_shapes": {
            "token_features": [BOARD_SLOTS, BOARD_TOKEN_CAPACITY, TOKEN_FEATURES],
            "token_types": [BOARD_SLOTS, BOARD_TOKEN_CAPACITY],
            "token_mask": [BOARD_SLOTS, BOARD_TOKEN_CAPACITY],
            "market_features": [MARKET_SLOTS, MARKET_FEATURES],
            "market_mask": [MARKET_SLOTS],
            "player_features": [BOARD_SLOTS, PLAYER_FEATURES],
            "player_mask": [BOARD_SLOTS],
            "global_features": [GLOBAL_FEATURES],
        },
        "action_feature_dim": config.action_feature_dim,
        "market_decision_action_feature_dim": MARKET_DECISION_FEATURE_DIM,
        "opponent_paid_wipe_maximum": GRADED_ORACLE_MAX_WILDLIFE_WIPES,
        "auxiliary_conditioning": "candidate-afterstate-selected-in-loss",
        "live_response_tensors": [
            "action_scores",
            "predicted_score_to_go",
            "predicted_score_components_to_go",
            "bootstrap_policy_logits",
        ],
        "score_component_dim": config.score_component_dim,
        "active_heads": [
            "score-to-go",
            "score-to-go-components-11",
            "bootstrap-policy-preference",
            "ordered-opponent-next-action",
            "four-tile-market-survival",
            "public-market-decision-value-and-bootstrap-policy",
        ],
        "deferred_heads": ["uncertainty", "legal-affordance", "public-transition"],
        "inference_equation": "exact_afterstate_score + predicted_score_to_go",
        "market_decision_inference_equation": (
            "exact_current_score + predicted_market_decision_score_to_go"
        ),
    }
