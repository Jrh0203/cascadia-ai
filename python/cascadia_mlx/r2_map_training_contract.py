"""Trainer-independent supervised-batch contract for R2-MAP."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

import mlx.core as mx
import numpy as np

from cascadia_mlx.graded_oracle_dataset import GRADED_ORACLE_MAX_WILDLIFE_WIPES
from cascadia_mlx.r2_map_model import R2MapBatch, R2MapMarketDecisionBatch


@dataclass(frozen=True, slots=True)
class R2MapMarketDecisionSupervision:
    """Observed-return supervision for replay-reconstructed pre-refill choices."""

    inputs: R2MapMarketDecisionBatch
    score_to_go_targets: mx.array
    score_target_mask: mx.array
    selected_action_index: mx.array
    policy_target_mask: mx.array
    batch_identity: str

    def validate(self) -> tuple[int, int]:
        groups, actions = self.inputs.validate()
        expected = (groups, actions)
        if tuple(self.score_to_go_targets.shape) != expected:
            raise ValueError("market-decision score target shape drifted")
        if self.score_to_go_targets.dtype != mx.float32:
            raise ValueError("market-decision score targets must be float32")
        if (
            tuple(self.score_target_mask.shape) != expected
            or self.score_target_mask.dtype != mx.bool_
        ):
            raise ValueError("market-decision score target mask shape or dtype drifted")
        if tuple(self.selected_action_index.shape) != (groups,):
            raise ValueError("market-decision selected action shape drifted")
        if self.selected_action_index.dtype != mx.int32:
            raise ValueError("market-decision selected actions must be int32")
        if (
            tuple(self.policy_target_mask.shape) != (groups,)
            or self.policy_target_mask.dtype != mx.bool_
        ):
            raise ValueError("market-decision policy target mask shape or dtype drifted")
        selected = np.asarray(self.selected_action_index)
        legal = np.asarray(self.inputs.action_mask)
        target_mask = np.asarray(self.score_target_mask)
        if np.any(selected < 0) or np.any(selected >= actions):
            raise ValueError("market-decision selected action is out of range")
        if not np.all(legal[np.arange(groups), selected]):
            raise ValueError("market-decision selected action is not legal")
        if not np.all(target_mask.sum(axis=1) == 1) or not np.all(
            target_mask[np.arange(groups), selected]
        ):
            raise ValueError("market-decision observed target must select only the played action")
        policy_mask = np.asarray(self.policy_target_mask)
        if np.any(policy_mask & (np.asarray(self.inputs.action_mask).sum(axis=1) < 2)):
            raise ValueError("market-decision policy target requires a choice")
        if not self.batch_identity:
            raise ValueError("market-decision supervision requires an immutable identity")
        return groups, actions


@dataclass(frozen=True, slots=True)
class R2MapSupervisedBatch:
    """Public model inputs plus complete-game supervision targets."""

    inputs: R2MapBatch
    score_to_go_targets: mx.array
    score_component_targets: mx.array
    score_target_mask: mx.array
    selected_action_index: mx.array
    bootstrap_policy_mask: mx.array
    opponent_tile_slot_targets: mx.array
    opponent_wildlife_slot_targets: mx.array
    opponent_draft_kind_targets: mx.array
    opponent_drafted_wildlife_targets: mx.array
    opponent_replace_three_targets: mx.array
    opponent_paid_wipe_count_targets: mx.array
    opponent_paid_wipe_mask_targets: mx.array
    opponent_paid_wipe_mask_valid: mx.array
    opponent_valid_mask: mx.array
    market_disposition_targets: mx.array
    market_pair_survival_targets: mx.array
    market_final_slot_targets: mx.array
    market_disposition_mask: mx.array
    market_pair_survival_mask: mx.array
    market_final_slot_mask: mx.array
    batch_identity: str
    market_decisions: R2MapMarketDecisionSupervision | None = None

    def validate(self) -> tuple[int, int]:
        groups, candidates = self.inputs.validate()
        candidate_shape = (groups, candidates)
        shapes = {
            "score_to_go_targets": candidate_shape,
            "score_component_targets": (*candidate_shape, 11),
            "score_target_mask": candidate_shape,
            "selected_action_index": (groups,),
            "bootstrap_policy_mask": (groups,),
            "opponent_tile_slot_targets": (groups, 3),
            "opponent_wildlife_slot_targets": (groups, 3),
            "opponent_draft_kind_targets": (groups, 3),
            "opponent_drafted_wildlife_targets": (groups, 3),
            "opponent_replace_three_targets": (groups, 3),
            "opponent_paid_wipe_count_targets": (groups, 3),
            "opponent_paid_wipe_mask_targets": (
                groups,
                3,
                GRADED_ORACLE_MAX_WILDLIFE_WIPES,
            ),
            "opponent_paid_wipe_mask_valid": (
                groups,
                3,
                GRADED_ORACLE_MAX_WILDLIFE_WIPES,
            ),
            "opponent_valid_mask": (groups, 3),
            "market_disposition_targets": (groups, 4),
            "market_pair_survival_targets": (groups, 4),
            "market_final_slot_targets": (groups, 4),
            "market_disposition_mask": (groups, 4),
            "market_pair_survival_mask": (groups, 4),
            "market_final_slot_mask": (groups, 4),
        }
        for name, expected in shapes.items():
            if tuple(getattr(self, name).shape) != expected:
                raise ValueError(f"R2-MAP supervised tensor {name} shape drifted")
        for name in ("score_to_go_targets", "score_component_targets"):
            if getattr(self, name).dtype != mx.float32:
                raise ValueError(f"R2-MAP supervised tensor {name} must be float32")
        boolean_names = {
            "score_target_mask",
            "bootstrap_policy_mask",
            "opponent_valid_mask",
            "opponent_paid_wipe_mask_valid",
            "market_disposition_mask",
            "market_pair_survival_mask",
            "market_final_slot_mask",
        }
        for name in boolean_names:
            if getattr(self, name).dtype != mx.bool_:
                raise ValueError(f"R2-MAP supervised tensor {name} must be bool")
        for name in set(shapes) - {
            "score_to_go_targets",
            "score_component_targets",
            *boolean_names,
        }:
            if getattr(self, name).dtype != mx.int32:
                raise ValueError(f"R2-MAP supervised tensor {name} must be int32")
        if not self.batch_identity:
            raise ValueError("R2-MAP supervised batch requires an immutable identity")
        if self.market_decisions is not None:
            self.market_decisions.validate()
        selected = np.asarray(self.selected_action_index)
        legal = np.asarray(self.inputs.candidate_mask)
        if np.any(selected < 0) or np.any(selected >= candidates):
            raise ValueError("R2-MAP selected action target is out of range")
        if not np.all(legal[np.arange(groups), selected]):
            raise ValueError("R2-MAP selected action target names an illegal candidate")
        score_mask = np.asarray(self.score_target_mask)
        if not np.all(score_mask.sum(axis=1) == 1) or not np.all(
            score_mask[np.arange(groups), selected]
        ):
            raise ValueError("R2-MAP observed-return mask must select only the played action")
        policy_mask = np.asarray(self.bootstrap_policy_mask)
        if np.any(policy_mask & (legal.sum(axis=1) < 2)):
            raise ValueError("R2-MAP bootstrap policy target requires a complete choice")
        return groups, candidates


@dataclass(frozen=True)
class R2MapAdapterStep:
    batch: R2MapSupervisedBatch
    next_cursor: dict[str, Any]
    next_sampler_state: dict[str, Any]


class R2MapTrainingAdapter(Protocol):
    """Versioned, deterministic boundary around compact replay streams."""

    protocol_id: str
    dataset_blake3: str
    dataset_contract: dict[str, Any]
    group_batch_size: int
    maximum_candidates_per_batch: int

    def initial_state(self, seed: int) -> tuple[dict[str, Any], dict[str, Any]]: ...

    def training_batch(
        self,
        cursor: dict[str, Any],
        sampler_state: dict[str, Any],
    ) -> R2MapAdapterStep: ...

    def validation_batches(self) -> Iterable[R2MapSupervisedBatch]: ...

    def fixed_prediction_batch(self, panel_id: str) -> R2MapBatch: ...
