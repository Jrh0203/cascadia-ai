from __future__ import annotations

import json
import subprocess
from argparse import Namespace
from copy import deepcopy
from pathlib import Path

import pytest
from r2_d0.canonical import D0Error, canonical_json
from r2_d0.signing import public_key_from_private, sign_stdin, signature_bytes
from r2_d0_maintenance_window import render_open, render_restore, verify


@pytest.fixture
def key(tmp_path: Path) -> tuple[Path, Path]:
    private = tmp_path / "key"
    subprocess.run(
        ["/usr/bin/ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(private)],
        check=True,
    )
    private.chmod(0o600)
    public = tmp_path / "key.pub.normalized"
    public.write_bytes(public_key_from_private(private))
    return private, public


def _write(path: Path, value: dict[str, object]) -> Path:
    path.write_bytes(canonical_json(value))
    return path


def _open_rows(tmp_path: Path) -> list[Path]:
    result = []
    homes = {"john1": "/Users/johnherrick", "john2": "/Users/john2", "john3": "/Users/john3"}
    for index, (host, home) in enumerate(homes.items(), 1):
        result.append(
            _write(
                tmp_path / f"{host}-open.json",
                {
                    "host": host,
                    "authorized_keys_path": f"{home}/.ssh/authorized_keys",
                    "backup_path": f"{home}/.ssh/authorized_keys.cascadia-v2-original",
                    "original_sha256": f"{index:064x}",
                    "original_size": 100 + index,
                    "active_sha256": f"{index + 10:064x}",
                    "active_size": 150 + index,
                    "mode": "0600",
                    "source_ip_denies": ["100.98.107.61", "100.98.16.59"],
                    "changed_unix_ms": 1000 + index,
                    "original_preserved": True,
                    "active_differs": True,
                    "status": "pass",
                },
            )
        )
    return result


def _restore_rows(tmp_path: Path, opened: dict[str, object]) -> list[Path]:
    result = []
    for index, row in enumerate(opened["hosts"], 1):
        result.append(
            _write(
                tmp_path / f"{row['host']}-restore.json",
                {
                    "host": row["host"],
                    "authorized_keys_path": row["authorized_keys_path"],
                    "restored_sha256": row["original_sha256"],
                    "restored_size": row["original_size"],
                    "mode": "0600",
                    "source_ip_denies_absent": True,
                    "backup_absent": True,
                    "restored_unix_ms": 2000 + index,
                    "status": "pass",
                },
            )
        )
    return result


def test_signed_restore_gate_requires_exact_three_host_original_state(
    tmp_path: Path, key: tuple[Path, Path]
) -> None:
    private, public = key
    opened_path = tmp_path / "opened.json"
    opened = render_open(
        Namespace(
            host=_open_rows(tmp_path),
            supersedes_receipt_sha256="f" * 64,
            out=opened_path,
        )
    )
    open_signature = tmp_path / "opened-signature.json"
    open_signature.write_bytes(signature_bytes(sign_stdin(private, opened_path.read_bytes())))
    restore_paths = _restore_rows(tmp_path, opened)
    restore_path = tmp_path / "restored.json"
    restored = render_restore(
        Namespace(
            open_receipt=opened_path,
            open_signature=open_signature,
            public_key=public,
            host=restore_paths,
            out=restore_path,
        )
    )
    restore_signature = tmp_path / "restored-signature.json"
    restore_signature.write_bytes(signature_bytes(sign_stdin(private, restore_path.read_bytes())))
    result = verify(
        Namespace(
            open_receipt=opened_path,
            open_signature=open_signature,
            restore_receipt=restore_path,
            restore_signature=restore_signature,
            public_key=public,
        )
    )
    assert result["restore_gate_satisfied"] is True
    assert result["restore_receipt_sha256"] == restored["receipt_sha256"]

    changed = deepcopy(json.loads(restore_paths[0].read_bytes()))
    changed["backup_absent"] = False
    restore_paths[0].write_bytes(canonical_json(changed))
    with pytest.raises(D0Error, match="original state"):
        render_restore(
            Namespace(
                open_receipt=opened_path,
                open_signature=open_signature,
                public_key=public,
                host=restore_paths,
                out=tmp_path / "must-not-exist.json",
            )
        )
