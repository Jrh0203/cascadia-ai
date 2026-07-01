"""Deterministic fake-model process fixture for Rust/Python R2MP integration tests."""

from __future__ import annotations

import sys

import mlx.core as mx

from cascadia_mlx.r2_map_model import (
    R2MapMarketDecisionPrediction,
    R2MapMarketSurvival,
    R2MapOpponentNextAction,
    R2MapPrediction,
)
from cascadia_mlx.r2_map_serve import (
    R2MapCheckpointRegistry,
    R2MapRegistryEntry,
    serve_r2_map,
)

MODEL_IDENTITY = {
    "checkpoint_id": "cross-language-fixture",
    "checkpoint_manifest_blake3": "1" * 64,
    "model_config_blake3": "2" * 64,
    "model_weights_blake3": "3" * 64,
    "verification_id": "4" * 64,
}


class ExactFixtureModel:
    """Return exact afterstate scores while exercising every response head."""

    def __call__(self, batch: object) -> R2MapPrediction:
        groups, candidates = batch.validate()
        valid = batch.candidate_mask
        zeros = mx.zeros((groups, candidates), dtype=mx.float32)
        return R2MapPrediction(
            action_scores=mx.where(valid, batch.exact_afterstate_scores, -mx.inf),
            predicted_score_to_go=zeros,
            predicted_score_components_to_go=mx.zeros(
                (groups, candidates, 11), dtype=mx.float32
            ),
            bootstrap_policy_logits=mx.where(valid, zeros, -mx.inf),
            opponent_next_action=R2MapOpponentNextAction(
                tile_slot_logits=mx.zeros((groups, candidates, 3, 4)),
                wildlife_slot_logits=mx.zeros((groups, candidates, 3, 4)),
                draft_kind_logits=mx.zeros((groups, candidates, 3, 2)),
                drafted_wildlife_logits=mx.zeros((groups, candidates, 3, 5)),
                replace_three_logits=mx.zeros((groups, candidates, 3, 2)),
                paid_wipe_count_logits=mx.zeros((groups, candidates, 3, 21)),
                paid_wipe_mask_logits=mx.zeros((groups, candidates, 3, 20, 16)),
            ),
            market_survival=R2MapMarketSurvival(
                disposition_logits=mx.zeros((groups, candidates, 4, 4)),
                pair_survival_logits=mx.zeros((groups, candidates, 4, 2)),
                final_slot_logits=mx.zeros((groups, candidates, 4, 4)),
            ),
            candidate_mask=valid,
        )

    def score_actions(self, batch: object) -> R2MapPrediction:
        return self(batch)

    def score_market_decisions(self, batch: object) -> R2MapMarketDecisionPrediction:
        groups, actions = batch.validate()
        valid = batch.action_mask
        zeros = mx.zeros((groups, actions), dtype=mx.float32)
        return R2MapMarketDecisionPrediction(
            action_scores=mx.where(
                valid,
                batch.exact_current_scores[:, None] + zeros,
                -mx.inf,
            ),
            predicted_score_to_go=zeros,
            bootstrap_policy_logits=mx.where(valid, zeros, -mx.inf),
            action_mask=valid,
        )


def main() -> None:
    registry = R2MapCheckpointRegistry(capacity=1)
    registry.register_model(R2MapRegistryEntry(model=ExactFixtureModel(), **MODEL_IDENTITY))
    serve_r2_map(registry, sys.stdin.buffer, sys.stdout.buffer)


if __name__ == "__main__":
    main()
