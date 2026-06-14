from __future__ import annotations

import platform

import pytest
from cascadia_mlx.device import probe_device


def test_probe_rejects_non_positive_vector_size() -> None:
    with pytest.raises(ValueError, match="positive"):
        probe_device(0)


@pytest.mark.skipif(platform.system() != "Darwin", reason="MLX Metal probe requires macOS")
def test_probe_executes_on_apple_gpu() -> None:
    report = probe_device(10_000)

    assert report["schema_version"] == 1
    assert report["machine"] == "arm64"
    assert "gpu" in report["device"].lower()
    assert report["relative_error"] < 1e-5
