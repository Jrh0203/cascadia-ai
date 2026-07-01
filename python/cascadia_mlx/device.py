"""Probe the local MLX device with an evaluated GPU workload."""

from __future__ import annotations

import json
import platform
import time
from importlib.metadata import version
from typing import Any

import mlx.core as mx


def probe_device(vector_size: int = 1_000_000) -> dict[str, Any]:
    """Run a deterministic reduction and return a machine-readable device report."""
    if vector_size <= 0:
        raise ValueError("vector_size must be positive")

    values = mx.arange(vector_size, dtype=mx.float32)
    started = time.perf_counter()
    result = mx.sum(values * values)
    mx.eval(result)
    elapsed_seconds = time.perf_counter() - started

    expected = (vector_size - 1) * vector_size * (2 * vector_size - 1) / 6
    relative_error = abs(float(result.item()) - expected) / expected

    return {
        "schema_version": 1,
        "mlx_version": version("mlx"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device": str(mx.default_device()),
        "vector_size": vector_size,
        "elapsed_seconds": elapsed_seconds,
        "relative_error": relative_error,
    }


def main() -> None:
    """Print the device report as stable, script-friendly JSON."""
    print(json.dumps(probe_device(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
