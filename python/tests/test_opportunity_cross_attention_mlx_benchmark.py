from __future__ import annotations

from pathlib import Path

import blake3
import pytest
from cascadia_mlx.complete_decision_mlx_benchmark import (
    combine_complete_decisions_with_r6,
)
from cascadia_mlx.opportunity_cross_attention_mlx_benchmark import (
    _valid_binary_identity,
)


def test_opportunity_binary_identity_requires_file_and_checksum(
    tmp_path: Path,
) -> None:
    binary = tmp_path / "r6-replay"
    binary.write_bytes(b"opportunity-r6")
    checksum = blake3.blake3(binary.read_bytes()).hexdigest()

    assert _valid_binary_identity(
        {"path": str(binary), "blake3": checksum}
    )
    assert not _valid_binary_identity(
        {"path": str(binary), "blake3": "0" * 64}
    )
    assert not _valid_binary_identity(
        {"path": str(tmp_path), "blake3": checksum}
    )


def test_complete_decision_r6_combination_preserves_alignment() -> None:
    result = combine_complete_decisions_with_r6(
        {
            "combined_complete_decisions": {
                "actions": 300,
                "latency_samples_milliseconds": [10.0, 20.0, 30.0],
            }
        },
        {
            "samples": [
                {"row": 0, "nanoseconds": 1_000_000},
                {"row": 1, "nanoseconds": 2_000_000},
                {"row": 2, "nanoseconds": 3_000_000},
            ]
        },
    )

    assert result["groups"] == 3
    assert result["latency_samples_milliseconds"] == [11.0, 22.0, 33.0]
    assert result["elapsed_seconds"] == pytest.approx(0.066)
