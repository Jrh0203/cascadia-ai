"""Frozen constants for the exact-R2 opportunity query factorial."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from cascadia_mlx.opportunity_cross_attention_mlx_model import ARMS

EXPERIMENT_ID = "opportunity-cross-attention-mlx-tournament-v1"
PROTOCOL_ID = "exact-r2-opportunity-query-factorial-v1"
ADR_ID = "0166"
TRAINING_SEED = 2026061718
TRAINING_STEPS = 2000
GROUPS_PER_STEP = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
CHECKPOINT_STEPS = 250
METRIC_STEPS = 100
VALIDATION_PROBE_GROUPS = 24
MLX_CACHE_LIMIT_BYTES = 1_073_741_824
MAX_SMOKE_STEPS = 10
RELATIONAL_DATA_ARM = "c0-exact-r2"
ARM_HOSTS = {
    ARMS[0]: "john1",
    ARMS[1]: "john2",
    ARMS[2]: "john3",
    ARMS[3]: "john4",
}
HOST_ALIASES = {"Johns-Mac-mini": "john1"}


@dataclass(frozen=True)
class OpportunityCrossAttentionTrainingProtocol:
    """Every non-query variable held equal across the four arms."""

    protocol_id: str = PROTOCOL_ID
    seed: int = TRAINING_SEED
    optimizer: str = "adamw"
    training_steps: int = TRAINING_STEPS
    groups_per_step: int = GROUPS_PER_STEP
    train_candidate_cap: int = 512
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    checkpoint_steps: int = CHECKPOINT_STEPS
    metric_steps: int = METRIC_STEPS
    validation_probe_groups: int = VALIDATION_PROBE_GROUPS
    warm_start: str = "final-authorized-c0-exact-r2-checkpoint"
    trainable_scope: str = "opportunity-adapters-only"
    early_stopping: bool = False
    schedule: str = (
        "three-independent-all-group-permutations-plus-alternating-"
        "low-supply-and-independent-winner-permutations"
    )
    d6_schedule: str = (
        "blake3-of-seed-step-slot-over-rust-d6-ids-0-through-11"
    )
    factual_surface: str = (
        "exact-r2-parent-plus-exact-s1-supply-tokens-for-all-arms"
    )
    arm_factorial: str = (
        "parent-or-candidate-query-crossed-over-supply-and-frontier-memory"
    )
    loss: str = (
        "r1200_huber+4*r4800_huber+0.5*r1200_listwise+"
        "r4800_winner+0.1*standard_error_calibration+"
        "0.01*screen_only_regularization"
    )

    def validate(self) -> None:
        if self != OpportunityCrossAttentionTrainingProtocol():
            raise ValueError("opportunity query protocol drifted from ADR 0166")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)


def normalize_host(host: str) -> str:
    """Map the local macOS hostname onto its cluster alias."""
    short = host.removesuffix(".local")
    return HOST_ALIASES.get(short, short)


__all__ = [
    "ADR_ID",
    "ARM_HOSTS",
    "CHECKPOINT_STEPS",
    "EXPERIMENT_ID",
    "LEARNING_RATE",
    "MAX_SMOKE_STEPS",
    "METRIC_STEPS",
    "MLX_CACHE_LIMIT_BYTES",
    "PROTOCOL_ID",
    "RELATIONAL_DATA_ARM",
    "TRAINING_SEED",
    "TRAINING_STEPS",
    "VALIDATION_PROBE_GROUPS",
    "WEIGHT_DECAY",
    "OpportunityCrossAttentionTrainingProtocol",
    "normalize_host",
]
