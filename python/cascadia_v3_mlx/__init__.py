"""Cascadia V3 radius-7 Stockfish-style NNUE training package."""

from .contracts import (
    ARCHITECTURE_ID,
    BASE_FEATURE_ROWS,
    FC0_OUTPUTS,
    FC1_INPUTS,
    FC1_OUTPUTS,
    PHASE_BUCKETS,
    TRANSFORM_WIDTH,
    V3MlxConfig,
)
from .model import SparseBatch, V3Nnue, v3_loss

__all__ = [
    "ARCHITECTURE_ID",
    "BASE_FEATURE_ROWS",
    "FC0_OUTPUTS",
    "FC1_INPUTS",
    "FC1_OUTPUTS",
    "PHASE_BUCKETS",
    "TRANSFORM_WIDTH",
    "SparseBatch",
    "V3MlxConfig",
    "V3Nnue",
    "v3_loss",
]
