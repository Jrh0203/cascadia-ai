from __future__ import annotations

from pathlib import Path

from cascadia_mlx.r2_map_local_write_guard import (
    JOHN1_MLX_INTERPRETER,
    SANDBOX_PROFILE,
    SANDBOX_PROFILE_SHA256,
    john1_attestation_publication_receipt_relative,
)
from r2_map_john1_no_write_launcher import (
    _attestation_relative,
    _snapshot_paths,
)


def test_zero_write_profile_and_attestation_paths_are_frozen() -> None:
    assert "(deny file-write*)" in SANDBOX_PROFILE
    assert '(allow file-write* (literal "/dev/null"))' in SANDBOX_PROFILE
    assert len(SANDBOX_PROFILE_SHA256) == 64
    assert JOHN1_MLX_INTERPRETER == (
        "/Users/johnherrick/.local/share/uv/python/"
        "cpython-3.12.13-macos-aarch64-none/bin/python3.12"
    )
    assert _attestation_relative("packing-sweep", "run-1") == (
        "reports/w2-w3/run-1/local-write-attestation.json"
    )
    assert _attestation_relative("train", "run-1") == ("runs/run-1/local-write-attestation.json")
    assert john1_attestation_publication_receipt_relative("a" * 64) == (
        f"control/receipts/req-john1-attestation-{'a' * 32}.json"
    )


def test_scoped_snapshot_detects_metadata_mutation(tmp_path: Path) -> None:
    child = tmp_path / "value.txt"
    child.write_text("before")
    before = _snapshot_paths((tmp_path,))
    child.write_text("after-longer")
    after = _snapshot_paths((tmp_path,))
    assert before != after


def test_snapshot_root_symlink_is_lexical_and_never_traversed(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    (target / "secret.txt").write_text("must-not-be-inventoried")
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    snapshot = _snapshot_paths((link,))
    assert snapshot == [
        {
            "path": str(link),
            "state": "present",
            "entries": 1,
            "apparent_bytes": 0,
            "sha256": snapshot[0]["sha256"],
        }
    ]
    (target / "secret.txt").write_text("changed-behind-symlink")
    assert _snapshot_paths((link,)) == snapshot
