from __future__ import annotations

import json
import os
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import r2_d0.bootstrap as bootstrap_module
from r2_d0.authorization import authorize_work_packet
from r2_d0.bootstrap import (
    apply_bootstrap,
    build_helper_archive,
    install_bootstrap_artifacts,
    verify_helper_archive,
)
from r2_d0.canonical import D0Error, canonical_json, sha256_bytes
from r2_d0.signing import (
    public_key_fingerprint,
    public_key_from_private,
    sign_stdin,
    signature_bytes,
    verify_stdin,
)
from r2_d0_test_support import rendered_bootstrap, rendered_work


@pytest.fixture
def signing_key(tmp_path: Path) -> Path:
    key = tmp_path / "test-ed25519"
    subprocess.run(
        ["/usr/bin/ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)],
        check=True,
        capture_output=True,
    )
    os.chmod(key, 0o600)
    return key


def test_openssh_stdin_signature_roundtrip_and_tamper(signing_key: Path) -> None:
    payload = b"canonical-packet-bytes"
    public_key = public_key_from_private(signing_key)
    signature = sign_stdin(signing_key, payload)
    verify_stdin(public_key, payload, signature)
    with pytest.raises(D0Error):
        verify_stdin(public_key, payload + b"!", signature)
    changed = dict(signature)
    changed["signature_armored"] = changed["signature_armored"].replace("A", "B", 1)
    with pytest.raises(D0Error):
        verify_stdin(public_key, payload, changed)


def test_private_key_mode_and_type_are_fail_closed(signing_key: Path, tmp_path: Path) -> None:
    os.chmod(signing_key, 0o644)
    with pytest.raises(D0Error, match="mode"):
        sign_stdin(signing_key, b"payload")
    with pytest.raises(D0Error, match="regular"):
        sign_stdin(tmp_path, b"payload")


def test_signed_work_authorization_binds_host_phase_time_helper_and_key(
    signing_key: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "r2_d0.authorization.pwd.getpwuid",
        lambda _uid: SimpleNamespace(pw_name="john2"),
    )
    public_key = public_key_from_private(signing_key)
    helper_archive, helper_receipt = build_helper_archive(Path("tools"))
    now = 1_800_000_000_000
    packet = rendered_work(
        "john2",
        "preflight",
        helper_sha256=helper_receipt["archive_sha256"],
        fingerprint=public_key_fingerprint(public_key),
        now=now,
    )
    signature = signature_bytes(sign_stdin(signing_key, packet))
    authorized = authorize_work_packet(
        packet,
        signature,
        public_key,
        expected_phase="preflight",
        required_operation="preflight-audit",
        helper_sha256=sha256_bytes(helper_archive),
        current_user="john2",
        now_unix_ms=now + 1,
    )
    assert authorized["host"] == "john2"
    with pytest.raises(D0Error, match="kernel UID"):
        authorize_work_packet(
            packet,
            signature,
            public_key,
            expected_phase="preflight",
            required_operation="preflight-audit",
            helper_sha256=sha256_bytes(helper_archive),
            current_user="john3",
            now_unix_ms=now + 1,
        )
    with pytest.raises(D0Error, match="validity"):
        authorize_work_packet(
            packet,
            signature,
            public_key,
            expected_phase="preflight",
            required_operation="preflight-audit",
            helper_sha256=sha256_bytes(helper_archive),
            current_user="john2",
            now_unix_ms=now + 7_200_001,
        )
    with pytest.raises(D0Error, match="full execution"):
        authorize_work_packet(
            packet,
            signature,
            public_key,
            expected_phase="preflight",
            required_operation="preflight-audit",
            helper_sha256=sha256_bytes(helper_archive),
            current_user="john2",
            now_unix_ms=now + 3_570_001,
            require_full_execution_window=True,
        )
    with pytest.raises(D0Error, match="helper"):
        authorize_work_packet(
            packet,
            signature,
            public_key,
            expected_phase="preflight",
            required_operation="preflight-audit",
            helper_sha256="0" * 64,
            current_user="john2",
            now_unix_ms=now + 1,
        )


def test_helper_archive_is_reproducible_complete_and_tamper_evident() -> None:
    first, first_receipt = build_helper_archive(Path("tools"))
    second, second_receipt = build_helper_archive(Path("tools"))
    assert first == second
    assert first_receipt == second_receipt
    assert verify_helper_archive(first)["file_count"] >= 10
    tampered = bytearray(first)
    tampered[1024] ^= 1
    with pytest.raises(D0Error):
        verify_helper_archive(bytes(tampered))


def test_apple_python39_builds_verifies_extracts_and_imports_helper_closure(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "helper.tar"
    source_root = Path("tools").resolve()
    driver = (
        "import pathlib,sys;"
        f"sys.path.insert(0,{str(source_root)!r});"
        "from r2_d0.bootstrap import build_helper_archive,verify_helper_archive;"
        f"a,r=build_helper_archive(pathlib.Path({str(source_root)!r}));"
        "v=verify_helper_archive(a);"
        "assert v['archive_sha256']==r['archive_sha256'];"
        f"pathlib.Path({str(archive_path)!r}).write_bytes(a)"
    )
    built = subprocess.run(
        ["/usr/bin/python3", "-I", "-S", "-B", "-c", driver],
        check=False,
        capture_output=True,
        text=True,
    )
    assert built.returncode == 0, built.stderr
    archive = archive_path.read_bytes()
    assert verify_helper_archive(archive)["archive_sha256"] == sha256_bytes(archive)

    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with tarfile.open(archive_path, mode="r:") as bundle:
        for member in bundle:
            stream = bundle.extractfile(member)
            assert stream is not None
            target = extracted / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(stream.read())
    import_driver = (
        "import sys;"
        f"sys.path.insert(0,{str(extracted)!r});"
        "import r2_d0.bootstrap,r2_d0.bundle,r2_d0.cli,r2_d0.runtime;"
        "assert r2_d0.bootstrap.HELPER_ENTRYPOINT=='r2_map_d0_runtime.py'"
    )
    imported = subprocess.run(
        ["/usr/bin/python3", "-I", "-S", "-B", "-c", import_driver],
        check=False,
        capture_output=True,
        text=True,
    )
    assert imported.returncode == 0, imported.stderr
    cli = subprocess.run(
        [
            "/usr/bin/python3",
            "-I",
            "-S",
            "-B",
            str(extracted / "r2_map_d0_runtime.py"),
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert cli.returncode == 0, cli.stderr
    assert "usage: r2-map-d0-runtime" in cli.stdout


def test_bootstrap_artifact_transaction_installs_exact_modes(
    signing_key: Path,
    tmp_path: Path,
) -> None:
    helper, _ = build_helper_archive(Path("tools"))
    public_key = public_key_from_private(signing_key)
    receipt_path = tmp_path / "owner/.config/cascadia-r2-d0/bootstrap-receipt.json"
    receipt_payload = b'{"status":"test"}'
    installed = install_bootstrap_artifacts(
        helper_archive=helper,
        public_key=public_key,
        helper_destination=tmp_path / "owner/.local/libexec/cascadia-r2-d0/v1",
        key_destination=tmp_path / "owner/.config/cascadia-r2-d0/public-key",
        receipt_destination=receipt_path,
        receipt_payload=receipt_payload,
    )
    entrypoint = Path(installed["helper_destination"]) / "r2_map_d0_runtime.py"
    key_path = Path(installed["public_key_destination"])
    assert entrypoint.read_bytes().startswith(b"#!/usr/bin/env -S python3 -I -S -B")
    assert entrypoint.stat().st_mode & 0o777 == 0o555
    assert key_path.read_bytes() == public_key
    assert key_path.stat().st_mode & 0o777 == 0o400
    assert receipt_path.read_bytes() == receipt_payload
    with pytest.raises(D0Error, match="already exists"):
        install_bootstrap_artifacts(
            helper_archive=helper,
            public_key=public_key,
            helper_destination=Path(installed["helper_destination"]),
            key_destination=key_path,
            receipt_destination=receipt_path,
            receipt_payload=receipt_payload,
        )


def test_bootstrap_expiry_precedes_mutation_and_partial_key_transaction_rolls_back(
    signing_key: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper, helper_receipt = build_helper_archive(Path("tools"))
    public_key = public_key_from_private(signing_key)
    now = 1_800_000_000_000
    packet = rendered_bootstrap(
        host="john1",
        helper_sha256=helper_receipt["archive_sha256"],
        helper_size=len(helper),
        public_key_sha256=sha256_bytes(public_key),
        fingerprint=public_key_fingerprint(public_key),
        now=now,
    )
    with pytest.raises(D0Error, match="validity"):
        apply_bootstrap(
            packet,
            authorized_packet_sha256=sha256_bytes(packet),
            helper_archive=helper,
            public_key=public_key,
            now_unix_ms=now + 60_001,
        )

    original_atomic_key = bootstrap_module._atomic_key
    calls = 0

    def fail_receipt(path: Path, value: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise D0Error("injected receipt failure")
        original_atomic_key(path, value)

    monkeypatch.setattr(bootstrap_module, "_atomic_key", fail_receipt)
    owner = tmp_path / "owner"
    helper_destination = owner / ".local/libexec/cascadia-r2-d0/v1"
    key_destination = owner / ".config/cascadia-r2-d0/public-key"
    receipt_destination = owner / ".config/cascadia-r2-d0/bootstrap-receipt.json"
    with pytest.raises(D0Error, match="injected"):
        install_bootstrap_artifacts(
            helper_archive=helper,
            public_key=public_key,
            helper_destination=helper_destination,
            key_destination=key_destination,
            receipt_destination=receipt_destination,
            receipt_payload=b"receipt",
        )
    assert not helper_destination.exists()
    assert not key_destination.exists()
    assert not receipt_destination.exists()


def test_signature_bundle_is_canonical_json(signing_key: Path) -> None:
    encoded = signature_bytes(sign_stdin(signing_key, b"payload"))
    assert canonical_json(json.loads(encoded)) == encoded
