from __future__ import annotations

from pathlib import Path

import blake3
from cascadia_mlx.relational_substrate_mlx_benchmark import (
    _valid_binary_identity,
)


def test_binary_identity_rejects_missing_or_nonfile_paths(
    tmp_path: Path,
) -> None:
    assert not _valid_binary_identity(None)
    assert not _valid_binary_identity({"path": "", "blake3": "0" * 64})
    assert not _valid_binary_identity(
        {"path": str(tmp_path), "blake3": "0" * 64}
    )


def test_binary_identity_requires_exact_checksum(tmp_path: Path) -> None:
    binary = tmp_path / "r6-replay"
    binary.write_bytes(b"exact-r6-binary")
    checksum = blake3.blake3(binary.read_bytes()).hexdigest()

    assert _valid_binary_identity(
        {"path": str(binary), "blake3": checksum}
    )
    assert not _valid_binary_identity(
        {"path": str(binary), "blake3": "0" * 64}
    )
