"""Deterministic source and runtime identity for every V3 MLX lineage."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import blake3
import cascadia_mlx.checkpoint as checkpoint_module
import mlx.core as mx


def training_source_identity() -> dict[str, object]:
    """Hash the complete training implementation and its runtime contract."""
    repository = Path(__file__).resolve().parents[2]
    sources = set(Path(__file__).resolve().parent.glob("*.py"))
    sources.update(
        (
            Path(checkpoint_module.__file__).resolve(),
            repository / "pyproject.toml",
            repository / "uv.lock",
        )
    )
    records = []
    aggregate = blake3.blake3(b"cascadia-v3-training-sources-v1")
    for path in sorted(sources):
        payload = path.read_bytes()
        relative = path.relative_to(repository).as_posix()
        digest = blake3.blake3(payload).hexdigest()
        records.append({"path": relative, "bytes": len(payload), "blake3": digest})
        encoded = relative.encode()
        aggregate.update(len(encoded).to_bytes(8, "little"))
        aggregate.update(encoded)
        aggregate.update(len(payload).to_bytes(8, "little"))
        aggregate.update(payload)
    return {
        "schema_id": "cascadia-v3-training-source-identity-v1",
        "blake3": aggregate.hexdigest(),
        "files": records,
        "runtime": {
            "python": sys.version,
            "mlx": mx.__version__,
            "platform": platform.platform(),
        },
    }
