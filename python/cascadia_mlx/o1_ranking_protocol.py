"""Frozen protocol constants for ADR 0188."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from cascadia_mlx.o1_ranking_intent_cache import ARMS

EXPERIMENT_ID = "o1-high-regret-draft-ranking-integration-v1"
PROTOCOL_ID = "o1-intent-conditioned-exact-r2-reranker-v1"
ADR_ID = "0188"
TRAINING_SEED = 2026061719
TRAINING_STEPS = 2_000
GROUPS_PER_STEP = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
CHECKPOINT_STEPS = 250
METRIC_STEPS = 100
MLX_CACHE_LIMIT_BYTES = 1_073_741_824
MAX_SMOKE_STEPS = 10

WAVE_HOSTS = {
    "primary": {
        ARMS[0]: "john1",
        ARMS[1]: "john2",
        ARMS[2]: "john3",
        ARMS[3]: "john4",
    },
    "rotated": {
        ARMS[0]: "john2",
        ARMS[1]: "john3",
        ARMS[2]: "john4",
        ARMS[3]: "john1",
    },
}
HOST_ALIASES = {
    "Johns-Mac-mini": "john1",
}


@dataclass(frozen=True)
class O1RankingTrainingProtocol:
    """Every scientific and optimization constant held equal across arms."""

    protocol_id: str = PROTOCOL_ID
    seed: int = TRAINING_SEED
    optimizer: str = "adamw"
    training_steps: int = TRAINING_STEPS
    groups_per_step: int = GROUPS_PER_STEP
    candidates_per_group: int = 64
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    checkpoint_steps: int = CHECKPOINT_STEPS
    metric_steps: int = METRIC_STEPS
    geometry: str = "canonical-only"
    schedule: str = "blake3-keyed-cyclic-permutation-over-frozen-group-ids"
    warm_start: str = "accepted-exact-r2-step-3000"
    trainable_parameters: str = "intent-adapter-only"
    validation_during_training: bool = False
    early_stopping: bool = False
    loss: str = (
        "r1200_huber+4*r4800_huber+0.5*r1200_listwise+"
        "r4800_winner+0.1*standard_error_calibration+"
        "0.01*screen_only_regularization"
    )

    def validate(self) -> None:
        if self != O1RankingTrainingProtocol():
            raise ValueError("O1 ranking protocol drifted from ADR 0188")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)


def normalize_host(host: str) -> str:
    return HOST_ALIASES.get(host, host)
