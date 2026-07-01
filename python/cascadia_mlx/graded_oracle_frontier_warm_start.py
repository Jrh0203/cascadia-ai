"""Verified loading for the selected ADR 0089 frontier-ranker checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import blake3
import mlx.core as mx

from cascadia_mlx.graded_oracle_model import (
    GradedOracleModelConfig,
    GradedOracleRanker,
)

EXPECTED_WARM_START_CHECKPOINT = "step-000003592-epoch-0008-batch-000000"
EXPECTED_WARM_START_MANIFEST_BLAKE3 = (
    "a8c2d6f5e932f3eb2957b6ceb6f3bc31c6734aa5c31f93f079136622cf76e812"
)
EXPECTED_WARM_START_MODEL_BLAKE3 = (
    "2c061a85c03ddfc0ad43004e404f8af4964340212ae255f826c2114572587adb"
)


def load_frontier_warm_start(checkpoint_dir: str | Path) -> GradedOracleRanker:
    """Load the exact selected checkpoint with full model-file verification."""
    checkpoint_dir = Path(checkpoint_dir)
    if checkpoint_dir.name != EXPECTED_WARM_START_CHECKPOINT:
        raise ValueError("frontier warm-start checkpoint drifted")
    manifest_path = checkpoint_dir / "checkpoint.json"
    if checksum(manifest_path) != EXPECTED_WARM_START_MANIFEST_BLAKE3:
        raise ValueError("frontier warm-start manifest drifted")
    manifest = json.loads(manifest_path.read_text())
    model_path = checkpoint_dir / "model.safetensors"
    metadata = manifest["files"]["model.safetensors"]
    if (
        model_path.stat().st_size != int(metadata["bytes"])
        or checksum(model_path) != EXPECTED_WARM_START_MODEL_BLAKE3
        or metadata["blake3"] != EXPECTED_WARM_START_MODEL_BLAKE3
    ):
        raise ValueError("frontier warm-start model drifted")
    model = GradedOracleRanker(
        GradedOracleModelConfig.from_dict(manifest["model_config"])
    )
    model.load_weights(str(model_path))
    mx.eval(model.parameters())
    return model


def checksum(path: Path) -> str:
    """Return a streaming BLAKE3 file identity."""
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
