from __future__ import annotations

import base64
import hashlib
import io
import json
import multiprocessing
import os
import signal
import stat
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import replace
from pathlib import Path

import blake3
import pytest
from cascadia_mlx import r2_map_remote_storage as remote_storage
from cascadia_mlx import r2_map_remote_worker as remote_worker
from cascadia_mlx.r2_map_remote_storage import (
    REMOTE_HOST_ALIAS,
    REMOTE_HOSTNAME,
    REMOTE_IDENTITY_SHA256,
    REMOTE_ROOT,
    RemoteProtocolError,
    RemoteResult,
    RemoteStorageClient,
    SshTransport,
    TransactionObject,
    _validate_relative,
    build_john1_runtime_manifest,
    build_transaction_manifest,
    canonical_json,
    content_sha256,
    document_sha256,
)


def _test_contract(tmp_path: Path) -> remote_worker.WorkerContract:
    root = tmp_path / "r2-map-v1"
    root.mkdir(mode=0o700)
    details = root.stat()
    contract = replace(
        remote_worker.PRODUCTION_CONTRACT,
        root=root,
        expected_uid=details.st_uid,
        expected_gid=details.st_gid,
        expected_root_device=details.st_dev,
        expected_root_inode=details.st_ino,
        min_free_bytes=0,
        max_campaign_bytes=1 << 30,
        max_data_bytes=(1 << 30) - (16 << 20),
        receipt_budget_bytes=16 << 20,
        max_receipt_bytes=64 << 10,
        max_receipt_entries=10_000,
    )
    for relative in (
        "control/locks",
        "control/transactions",
        "control/receipts",
        "control/receipt-reservations",
        "control/data-reservations",
        "reports",
        "checkpoints",
        "source",
        "logs",
        "tmp",
    ):
        path = root / relative
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.chmod(0o700)
    return contract


def _manifest(transaction_id: str, target: str, objects: list[dict]) -> dict:
    result = {
        "schema_version": 1,
        "schema_id": remote_worker.TRANSACTION_SCHEMA,
        "transaction_id": transaction_id,
        "target_relative": target,
        "objects": sorted(objects, key=lambda item: item["relative"]),
    }
    result["manifest_sha256"] = remote_worker.document_sha256(result, "manifest_sha256")
    return result


def _worker_request(operation: str, arguments: dict, request_id: str) -> dict:
    request = {
        "schema_version": 1,
        "schema_id": remote_worker.COMMAND_SCHEMA,
        "request_id": request_id,
        "issued_unix_ms": 1,
        "root": str(remote_worker.PRODUCTION_ROOT),
        "worker_sha256": "aa" * 32,
        "operation": operation,
        "arguments": arguments,
        "semantic_sha256": remote_worker.request_semantic_sha256(operation, arguments),
    }
    request["command_sha256"] = remote_worker.request_command_sha256(request)
    return request


def _test_storage_proof(contract: remote_worker.WorkerContract) -> dict:
    capacity = remote_worker._storage_capacity_state(contract)
    return {
        "schema_id": remote_worker.CAPACITY_PROOF_SCHEMA,
        "schema_version": remote_worker.PROTOCOL_VERSION,
        "protocol_sha256": remote_worker.PROTOCOL_SHA256,
        "root": str(contract.root),
        "root_mode": "0700",
        "root_uid": contract.expected_uid,
        "root_gid": contract.expected_gid,
        "root_device": contract.expected_root_device,
        "root_inode": contract.expected_root_inode,
        "host_identity_sha256": contract.expected_identity_sha256,
        "filesystem": "apfs",
        "protocol": "Apple Fabric",
        "internal": True,
        "removable": False,
        "solid_state": True,
        **capacity,
    }


def test_endpoint_and_ssh_transport_are_frozen_to_john2() -> None:
    transport = SshTransport()
    assert REMOTE_HOST_ALIAS == "john2"
    assert REMOTE_HOSTNAME == "100.100.43.38"
    assert str(REMOTE_ROOT) == "/Users/john2/cascadia-bench/r2-map-v1"
    assert len(REMOTE_IDENTITY_SHA256) == 64
    rendered = " ".join(transport.base_argv)
    for required in (
        "BatchMode=yes",
        "ClearAllForwardings=yes",
        "ControlMaster=no",
        "ControlPath=none",
        "UpdateHostKeys=no",
        "Compression=no",
        "StrictHostKeyChecking=yes",
        "GlobalKnownHostsFile=/dev/null",
        "HostKeyAlgorithms=ssh-ed25519",
        "HostKeyAlias=100.100.43.38",
        "IdentitiesOnly=yes",
        "PubkeyAcceptedAlgorithms=ssh-ed25519",
        f"UserKnownHostsFile={remote_storage.JOHN2_SSH_KNOWN_HOSTS}",
        "PasswordAuthentication=no",
        "KbdInteractiveAuthentication=no",
        "RequestTTY=no",
    ):
        assert required in rendered
    with pytest.raises(ValueError, match="only the frozen john2"):
        SshTransport("john4")
    assert "Compression=yes" in " ".join(SshTransport(compression=True).base_argv)
    with pytest.raises(TypeError, match="explicit boolean"):
        SshTransport(compression=1)  # type: ignore[arg-type]


def test_remote_request_id_can_be_deterministically_frozen() -> None:
    client = RemoteStorageClient()
    request_id = "req-john1-attestation-" + "a" * 32
    request = client._request("put-file", {}, request_id=request_id)
    assert request["request_id"] == request_id
    assert request["command_sha256"] == remote_worker.request_command_sha256(request)
    with pytest.raises(ValueError, match="request id"):
        client._request("put-file", {}, request_id="caller-chosen")


def test_ssh_configuration_verification_rejects_persistent_control_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = "\n".join(
        (
            "hostname 100.100.43.38",
            "user john2",
            f"identityfile {remote_storage.JOHN2_SSH_IDENTITY}",
            "controlmaster false",
            "controlpath none",
            "updatehostkeys false",
            "hostkeyalias 100.100.43.38",
            "hostkeyalgorithms ssh-ed25519",
            "pubkeyacceptedalgorithms ssh-ed25519",
            "preferredauthentications publickey",
            f"userknownhostsfile {remote_storage.JOHN2_SSH_KNOWN_HOSTS}",
            "globalknownhostsfile /dev/null",
            "identitiesonly yes",
        )
    )

    def completed(stdout: str) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 0, stdout=stdout, stderr="")

    monkeypatch.setattr(
        remote_storage.subprocess,
        "run",
        lambda *args, **kwargs: completed(base),
    )
    monkeypatch.setattr(
        remote_storage,
        "_provision_and_verify_ssh_trust",
        lambda: {
            "known_hosts": {"sha256": remote_storage.JOHN2_SSH_KNOWN_HOSTS_SHA256},
            "host_key_fingerprint": remote_storage.JOHN2_SSH_HOST_KEY_FINGERPRINT,
            "public_key_fingerprint": remote_storage.JOHN1_JOHN2_KEY_FINGERPRINT,
        },
    )
    verified = SshTransport().verify_local_configuration()
    assert verified["controlmaster"] == "no"
    assert verified["controlpath"] == "none"
    assert verified["updatehostkeys"] == "no"

    monkeypatch.setattr(
        remote_storage.subprocess,
        "run",
        lambda *args, **kwargs: completed(
            base.replace("controlmaster false", "controlmaster true")
        ),
    )
    with pytest.raises(remote_storage.RemoteTransportError, match="no-persistence"):
        SshTransport().verify_local_configuration()


@pytest.mark.parametrize(
    "output",
    (
        'designated => identifier "runtime"',
        '# designated => cdhash H"813b48b5ad9c7ca8c61b0c627ff29ad2c719414c"',
        'noise\n  # designated => identifier "runtime"\nmore noise',
    ),
)
def test_codesign_designated_requirement_parser_accepts_supported_macos_formats(
    output: str,
) -> None:
    expected = output.split("=>", 1)[1].splitlines()[0].strip()
    assert remote_worker._parse_designated_requirement(output) == expected
    assert remote_storage._parse_designated_requirement(output) == expected


def test_codesign_designated_requirement_parser_fails_closed() -> None:
    assert remote_worker._parse_designated_requirement("identifier=runtime") == ""
    assert remote_storage._parse_designated_requirement("identifier=runtime") == ""


def test_dashboard_api_deployment_is_transaction_bound_atomic_and_fixed_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir(mode=0o700)
    destination = repository / "target/release/cascadia-api"
    monkeypatch.setattr(remote_storage, "JOHN1_REPOSITORY_ROOT", repository)
    monkeypatch.setattr(remote_storage, "JOHN1_DASHBOARD_API_PATH", destination)
    monkeypatch.setattr(remote_storage.socket, "gethostname", lambda: remote_storage.JOHN1_HOSTNAME)
    monkeypatch.setattr(
        remote_storage.pwd,
        "getpwuid",
        lambda _uid: type("Owner", (), {"pw_name": remote_storage.JOHN1_USER})(),
    )
    monkeypatch.setattr(remote_storage, "JOHN1_UID", os.getuid())
    monkeypatch.setattr(remote_storage, "JOHN1_GID", os.getgid())
    executable = b"thin arm64 dashboard api"
    bundle = "bundles/dashboard-api-test-v1"
    provenance = {
        "schema_version": 1,
        "schema_id": remote_storage.DASHBOARD_API_BUNDLE_SCHEMA,
        "bundle_target_relative": bundle,
        "source_freeze": {
            "target_relative": "source/dashboard-api-source-test-v1",
            "manifest_sha256": "11" * 32,
            "storage_receipt_relative": "control/receipts/req-source.json",
            "storage_receipt_sha256": "22" * 32,
        },
        "build_script_freeze": {
            "target_relative": "source/dashboard-api-build-test-v1",
            "manifest_sha256": "33" * 32,
            "storage_receipt_relative": "control/receipts/req-build-source.json",
            "storage_receipt_sha256": "44" * 32,
        },
        "build": {
            "run_id": "dashboard-api-build-test-v1",
            "storage_receipt_relative": "control/receipts/req-build.json",
            "storage_receipt_sha256": "55" * 32,
        },
        "inspection": {
            "storage_receipt_relative": "control/receipts/req-inspection.json",
            "storage_receipt_sha256": "66" * 32,
        },
        "executable": {
            "relative": "cascadia-api",
            "sha256": content_sha256(executable),
            "blake3": blake3.blake3(executable).hexdigest(),
            "size": len(executable),
            "mach_o_arches": ["arm64"],
            "codesign": {"verified": True},
        },
    }
    provenance["manifest_sha256"] = document_sha256(provenance, "manifest_sha256")
    provenance_bytes = canonical_json(provenance)
    transaction = _manifest(
        "dashboard-api-test-v1",
        bundle,
        [
            {
                "relative": "cascadia-api",
                "size": len(executable),
                "sha256": content_sha256(executable),
                "mode": "0500",
            },
            {
                "relative": "dashboard-api-manifest.json",
                "size": len(provenance_bytes),
                "sha256": content_sha256(provenance_bytes),
            },
        ],
    )
    transaction_bytes = canonical_json(transaction)

    def token(relative: str, payload: bytes, mode: int) -> dict:
        value = {
            "schema_version": 1,
            "schema_id": remote_worker.OBJECT_TOKEN_SCHEMA,
            "relative": relative,
            "sha256": content_sha256(payload),
            "size": len(payload),
            "device": 1,
            "inode": 2,
            "mtime_ns": 3,
            "ctime_ns": 4,
            "mode": mode,
        }
        value["token_sha256"] = document_sha256(value, "token_sha256")
        return value

    payloads = {
        f"{bundle}/.r2-map-transaction.json": (transaction_bytes, 0o400),
        f"{bundle}/cascadia-api": (executable, 0o500),
        f"{bundle}/dashboard-api-manifest.json": (provenance_bytes, 0o400),
    }

    class Client:
        def __init__(self):
            self.publications = []

        def open_object_with_receipt(self, relative):
            payload, mode = payloads[relative]
            return {
                "object_token": token(relative, payload, mode),
                "storage_receipt_relative": "control/receipts/req-open.json",
                "storage_receipt_sha256": "ab" * 32,
            }

        def read_range_with_receipt(self, object_token, offset, length, *, max_bytes):
            payload = payloads[object_token["relative"]][0][offset : offset + length]
            return {
                "payload": payload,
                "payload_sha256": content_sha256(payload),
                "object_token_sha256": object_token["token_sha256"],
                "offset": offset,
                "length": length,
                "storage_receipt_relative": "control/receipts/req-read.json",
                "storage_receipt_sha256": "cd" * 32,
            }

        def iter_object_with_receipts(self, object_token):
            yield self.read_range_with_receipt(
                object_token, 0, object_token["size"], max_bytes=object_token["size"]
            )

        def put_bytes(self, relative, payload):
            self.publications.append((relative, payload))
            return {
                "relative": relative,
                "size": len(payload),
                "sha256": content_sha256(payload),
                "storage_receipt_relative": "control/receipts/req-deployment.json",
                "storage_receipt_sha256": "ef" * 32,
            }

    client = Client()
    result = remote_storage.deploy_dashboard_api(
        client,
        bundle_relative=f"{bundle}/cascadia-api",
        expected_sha256=content_sha256(executable),
    )
    assert destination.read_bytes() == executable
    assert stat.S_IMODE(destination.stat().st_mode) == 0o500
    assert result["destination"] == str(destination)
    assert result["bundle_provenance_sha256"] == provenance["manifest_sha256"]
    assert result["executable_range_receipts"][0]["storage_receipt_sha256"] == "cd" * 32
    assert result["deployment_sha256"] == document_sha256(
        {key: value for key, value in result.items() if key != "deployment_remote"},
        "deployment_sha256",
    )
    assert result["deployment_remote"]["storage_receipt_sha256"] == "ef" * 32
    assert client.publications[0][0].startswith("control/dashboard-deployments/")

    with pytest.raises(ValueError, match="immutable dashboard-api"):
        remote_storage.deploy_dashboard_api(
            Client(),
            bundle_relative="build/run-x/cascadia-api",
            expected_sha256=content_sha256(executable),
        )


@pytest.mark.parametrize(
    "value",
    ("/absolute", "../escape", "a/../escape", "a//b", "a\\b", "a\x00b", ""),
)
def test_relative_paths_fail_closed(value: str) -> None:
    with pytest.raises(ValueError):
        _validate_relative(value, "test")


def test_transaction_manifest_is_canonical_sorted_and_hash_bound() -> None:
    alpha = b"alpha"
    beta = b"beta"
    manifest = build_transaction_manifest(
        "checkpoint-001",
        "checkpoints/checkpoint-001",
        [
            TransactionObject("z.bin", len(beta), content_sha256(beta)),
            TransactionObject("a.bin", len(alpha), content_sha256(alpha)),
        ],
    )
    assert [item["relative"] for item in manifest["objects"]] == ["a.bin", "z.bin"]
    assert manifest["manifest_sha256"] == document_sha256(manifest, "manifest_sha256")
    with pytest.raises(ValueError, match="unique"):
        build_transaction_manifest(
            "checkpoint-001",
            "checkpoints/checkpoint-001",
            [
                TransactionObject("a.bin", len(alpha), content_sha256(alpha)),
                TransactionObject("a.bin", len(alpha), content_sha256(alpha)),
            ],
        )


def _fake_source_freeze() -> dict:
    return {
        "target_relative": "source/runtime-source-freeze",
        "manifest_sha256": "91" * 32,
        "storage_receipt_relative": "control/receipts/req-source-freeze.json",
        "storage_receipt_sha256": "92" * 32,
    }


def test_run_relative_checkpoint_transaction_shape_and_executable_mode() -> None:
    payload = b"signed executable"
    manifest = build_transaction_manifest(
        "checkpoint-002",
        "runs/bootstrap/checkpoints/checkpoint-002",
        [
            TransactionObject(
                "runtime",
                len(payload),
                content_sha256(payload),
                mode=0o500,
            )
        ],
    )
    assert manifest["objects"][0]["mode"] == "0500"
    assert remote_worker._validate_transaction_manifest(canonical_json(manifest)) == manifest
    invalid = {**manifest, "target_relative": "runs/bootstrap/not-checkpoints/checkpoint-002"}
    invalid["manifest_sha256"] = document_sha256(invalid, "manifest_sha256")
    with pytest.raises(remote_worker.RemoteWorkerError, match="runs/RUN/checkpoints"):
        remote_worker._validate_transaction_manifest(canonical_json(invalid))


def test_john1_runtime_manifest_binds_signed_arm64_packet_and_stream_limits() -> None:
    inspection = {
        "schema_version": 1,
        "schema_id": "cascadia.r2-map.ephemeral-executable-inspection.v1",
        "relative": "build/run-build/runtime",
        "sha256": "ab" * 32,
        "blake3": "ac" * 32,
        "size": 4096,
        "mode": "0500",
        "mach_o_arches": ["arm64"],
        "file_description": "Mach-O 64-bit executable arm64",
        "codesign": {
            "verified": True,
            "strict": True,
            "cdhash": "12" * 20,
            "identifier": "cascadia-r2-runtime",
            "team_identifier": "not set",
            "signature": "adhoc",
            "designated_requirement": 'identifier "cascadia-r2-runtime"',
            "designated_requirement_sha256": content_sha256(b'identifier "cascadia-r2-runtime"'),
            "verify_output_sha256": "34" * 32,
            "detail_output_sha256": "56" * 32,
            "portable_detail_sha256": "78" * 32,
        },
        "inspection_receipt_sha256": "cd" * 32,
        "storage_receipt_relative": "control/receipts/req-inspection.json",
    }
    manifest = build_john1_runtime_manifest(
        packet_id="bootstrap-runtime-001",
        executable_relative="bundles/bootstrap-runtime-001/cascadia-r2-runtime",
        inspection=inspection,
        source_freeze=_fake_source_freeze(),
        build_receipt_relative="control/receipts/req-build.json",
        build_receipt_sha256="ef" * 32,
        output_prefix_relative="logs/generation/bootstrap-runtime-001",
        stdout_max_bytes=1 << 20,
        stderr_max_bytes=1 << 16,
        created_unix_ms=1_781_789_600_000,
    )
    assert manifest["manifest_sha256"] == document_sha256(manifest, "manifest_sha256")
    assert manifest["executable"]["mach_o_arches"] == ["arm64"]
    assert manifest["combined_packet_max_bytes"] == 64 * (1 << 20)
    with pytest.raises(ValueError, match="bundles"):
        build_john1_runtime_manifest(
            packet_id="bootstrap-runtime-001",
            executable_relative="build/runtime",
            inspection=inspection,
            source_freeze=_fake_source_freeze(),
            build_receipt_relative="control/receipts/req-build.json",
            build_receipt_sha256="ef" * 32,
            output_prefix_relative="logs/generation/bootstrap-runtime-001",
            stdout_max_bytes=1 << 20,
            stderr_max_bytes=1 << 16,
            created_unix_ms=1,
        )


def _fake_runtime_manifest(payload: bytes, packet_id: str = "stale-packet-001") -> dict:
    inspection = {
        "schema_version": 1,
        "schema_id": "cascadia.r2-map.ephemeral-executable-inspection.v1",
        "relative": "build/run-build/runtime",
        "sha256": content_sha256(payload),
        "blake3": blake3.blake3(payload).hexdigest(),
        "size": len(payload),
        "mode": "0500",
        "mach_o_arches": ["arm64"],
        "file_description": "Mach-O 64-bit executable arm64",
        "codesign": {
            "verified": True,
            "strict": True,
            "cdhash": "12" * 20,
            "identifier": "runtime",
            "team_identifier": "not set",
            "signature": "adhoc",
            "designated_requirement": 'identifier "runtime"',
            "designated_requirement_sha256": content_sha256(b'identifier "runtime"'),
            "verify_output_sha256": "34" * 32,
            "detail_output_sha256": "56" * 32,
            "portable_detail_sha256": "78" * 32,
        },
        "inspection_receipt_sha256": "cd" * 32,
        "storage_receipt_relative": "control/receipts/req-inspection.json",
    }
    return build_john1_runtime_manifest(
        packet_id=packet_id,
        executable_relative=f"bundles/{packet_id}/cascadia-r2-runtime",
        inspection=inspection,
        source_freeze=_fake_source_freeze(),
        build_receipt_relative="control/receipts/req-build.json",
        build_receipt_sha256="ef" * 32,
        output_prefix_relative=f"logs/generation/{packet_id}",
        stdout_max_bytes=1 << 20,
        stderr_max_bytes=1 << 16,
        created_unix_ms=1,
    )


def test_startup_cleanup_accepts_only_exact_two_file_signed_packet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"fake signed Mach-O"
    packet_id = "stale-packet-001"
    directory = tmp_path / f"cascadia-r2-map-runtime-{packet_id}"
    directory.mkdir(mode=0o700)
    monkeypatch.setattr(remote_storage, "JOHN1_STAGING_ROOT", tmp_path)
    monkeypatch.setattr(remote_storage, "JOHN1_UID", os.getuid())
    monkeypatch.setattr(remote_storage, "JOHN1_GID", os.getgid())
    manifest = _fake_runtime_manifest(payload, packet_id)
    manifest_path = directory / "runtime-manifest.json"
    executable = directory / "cascadia-r2-runtime"
    manifest_path.write_bytes(canonical_json(manifest))
    executable.write_bytes(payload)
    manifest_path.chmod(0o400)
    executable.chmod(0o500)
    monkeypatch.setattr(
        remote_storage,
        "_codesign_fields",
        lambda _path: {
            "mach_o_arches": ["arm64"],
            "cdhash": "12" * 20,
            "designated_requirement": 'identifier "runtime"',
            "designated_requirement_sha256": content_sha256(b'identifier "runtime"'),
            "portable_detail_sha256": "78" * 32,
        },
    )
    monkeypatch.setattr(remote_storage, "_verify_john1_staging_host", lambda: {"safe": True})
    monkeypatch.setattr(
        remote_storage.subprocess,
        "run",
        lambda *_args, **_kwargs: type("Completed", (), {"stdout": ""})(),
    )

    class FakeClient:
        def __init__(self) -> None:
            self.publications = []

        def put_bytes(self, relative, content):
            self.publications.append((relative, content))
            return {
                "storage_receipt_relative": "control/receipts/req-startup-cleanup.json",
                "storage_receipt_sha256": "ab" * 32,
            }

    client = FakeClient()
    receipts = remote_storage.cleanup_stale_john1_runtime_directories(client)
    assert len(receipts) == 1
    assert not directory.exists()
    assert len(client.publications) == 1
    assert receipts[0]["receipt"]["cleanup"]["removed"] is True


@pytest.mark.parametrize("unsafe_kind", ("symlink", "fifo"))
def test_startup_cleanup_refuses_symlink_and_special_file(tmp_path: Path, unsafe_kind: str) -> None:
    directory = tmp_path / "cascadia-r2-map-runtime-unsafe-001"
    directory.mkdir(mode=0o700)
    manifest = directory / "runtime-manifest.json"
    manifest.write_text("{}")
    manifest.chmod(0o400)
    executable = directory / "cascadia-r2-runtime"
    if unsafe_kind == "symlink":
        executable.symlink_to(manifest)
    else:
        os.mkfifo(executable, 0o500)
    try:
        with pytest.raises(remote_storage.RemoteStorageError):
            remote_storage._validate_exact_staging_directory(directory)
    finally:
        executable.unlink()
        manifest.unlink()
        directory.rmdir()


class _RuntimeClient:
    def __init__(
        self,
        manifest: dict,
        executable: bytes,
        *,
        download_fault: str | None = None,
        stream_fault: str | None = None,
        omit_receipt_for: str | None = None,
    ) -> None:
        self.manifest = manifest
        self.manifest_bytes = canonical_json(manifest)
        self.executable = executable
        self.download_fault = download_fault
        self.stream_fault = stream_fault
        self.omit_receipt_for = omit_receipt_for
        self.publications: list[tuple[str, bytes]] = []
        self.streams: list[tuple[str, bytes]] = []

    def open_object(self, relative: str) -> dict:
        if relative == "bundles/runtime-test/runtime-manifest.json":
            payload = self.manifest_bytes
            mode = 0o400
        elif relative == self.manifest["executable"]["relative"]:
            payload = self.executable
            mode = 0o500
        else:
            raise AssertionError(f"unexpected remote object: {relative}")
        return {
            "relative": relative,
            "size": len(payload),
            "sha256": content_sha256(payload),
            "mode": mode,
            "token_sha256": "01" * 32,
        }

    def open_object_with_receipt(self, relative: str) -> dict:
        return {
            "object_token": self.open_object(relative),
            "storage_receipt_relative": "control/receipts/req-open-object.json",
            "storage_receipt_sha256": "10" * 32,
        }

    def read_range(self, _token, offset, length, *, max_bytes):
        assert offset == 0
        assert length == len(self.manifest_bytes)
        assert max_bytes == remote_worker.MAX_EPHEMERAL_MANIFEST_BYTES
        return self.manifest_bytes

    def read_range_with_receipt(self, token, offset, length, *, max_bytes):
        payload = self.read_range(token, offset, length, max_bytes=max_bytes)
        return {
            "payload": payload,
            "payload_sha256": content_sha256(payload),
            "object_token_sha256": token["token_sha256"],
            "offset": offset,
            "length": length,
            "storage_receipt_relative": "control/receipts/req-read-range.json",
            "storage_receipt_sha256": "20" * 32,
        }

    def iter_object(self, _token):
        if self.download_fault == "mid-download":
            yield self.executable[: max(1, len(self.executable) // 2)]
            raise OSError("injected download interruption")
        if self.download_fault == "hash":
            yield b"x" * len(self.executable)
            return
        yield self.executable

    def iter_object_with_receipts(self, token):
        offset = 0
        for payload in self.iter_object(token):
            yield {
                "payload": payload,
                "payload_sha256": content_sha256(payload),
                "object_token_sha256": token["token_sha256"],
                "offset": offset,
                "length": len(payload),
                "storage_receipt_relative": "control/receipts/req-read-executable.json",
                "storage_receipt_sha256": "30" * 32,
            }
            offset += len(payload)

    def put_bytes(self, relative: str, payload: bytes) -> dict:
        self.publications.append((relative, payload))
        result = {
            "relative": relative,
            "size": len(payload),
            "sha256": content_sha256(payload),
        }
        if self.omit_receipt_for is None or self.omit_receipt_for not in relative:
            result["storage_receipt_relative"] = "control/receipts/req-put-bytes.json"
            result["storage_receipt_sha256"] = "ab" * 32
        return result

    def put_unknown_stream(self, relative: str, chunks, *, max_bytes: int) -> dict:
        payload = b"".join(chunks)
        assert len(payload) <= max_bytes
        self.streams.append((relative, payload))
        stream_name = "stdout" if relative.endswith(".stdout") else "stderr"
        if self.stream_fault == stream_name:
            raise OSError(f"injected {stream_name} upload failure")
        result = {
            "relative": relative,
            "size": len(payload),
            "sha256": content_sha256(payload),
        }
        if self.omit_receipt_for != stream_name:
            result["storage_receipt_relative"] = f"control/receipts/req-{stream_name}-stream.json"
            result["storage_receipt_sha256"] = "cd" * 32
        return result


def _runtime_codesign_fields(manifest: dict) -> dict:
    signed = manifest["executable"]["codesign"]
    return {
        "file_description": "Mach-O 64-bit executable arm64",
        "mach_o_arches": ["arm64"],
        "cdhash": signed["cdhash"],
        "identifier": signed["identifier"],
        "team_identifier": signed["team_identifier"],
        "signature": signed["signature"],
        "designated_requirement": signed["designated_requirement"],
        "designated_requirement_sha256": signed["designated_requirement_sha256"],
        "verify_output_sha256": signed["verify_output_sha256"],
        "detail_output_sha256": signed["detail_output_sha256"],
        "portable_detail_sha256": signed["portable_detail_sha256"],
    }


def _runtime_harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    download_fault: str | None = None,
    identity_fault: str | None = None,
    stream_fault: str | None = None,
    omit_receipt_for: str | None = None,
) -> tuple[_RuntimeClient, dict]:
    monkeypatch.setattr(remote_storage, "JOHN1_STAGING_ROOT", tmp_path)
    monkeypatch.setattr(remote_storage, "JOHN1_UID", os.getuid())
    monkeypatch.setattr(remote_storage, "JOHN1_GID", os.getgid())
    monkeypatch.setattr(
        remote_storage,
        "_verify_john1_staging_host",
        lambda: {"safe": True, "staging_root": str(tmp_path)},
    )
    executable = b"fake signed arm64 Mach-O runtime"
    manifest = _fake_runtime_manifest(executable, "runtime-test")
    manifest["executable"]["relative"] = "bundles/runtime-test/cascadia-r2-runtime"
    manifest["manifest_sha256"] = document_sha256(manifest, "manifest_sha256")
    client = _RuntimeClient(
        manifest,
        executable,
        download_fault=download_fault,
        stream_fault=stream_fault,
        omit_receipt_for=omit_receipt_for,
    )
    codesign = _runtime_codesign_fields(manifest)
    if identity_fault == "codesign":
        codesign["cdhash"] = "ff" * 20
    monkeypatch.setattr(remote_storage, "_codesign_fields", lambda _path: dict(codesign))
    xattrs = "com.apple.quarantine\n" if identity_fault == "xattr" else ""
    monkeypatch.setattr(
        remote_storage.subprocess,
        "run",
        lambda *_args, **_kwargs: type("Completed", (), {"stdout": xattrs})(),
    )
    return client, manifest


def test_stage_runtime_success_proves_exact_two_file_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, manifest = _runtime_harness(tmp_path, monkeypatch)
    chown_calls = []
    fchown_calls = []
    real_chown = os.chown
    real_fchown = os.fchown

    def tracked_chown(path, uid, gid, **kwargs):
        chown_calls.append((Path(path), uid, gid))
        return real_chown(path, uid, gid, **kwargs)

    def tracked_fchown(descriptor, uid, gid):
        fchown_calls.append((uid, gid))
        return real_fchown(descriptor, uid, gid)

    monkeypatch.setattr(remote_storage.os, "chown", tracked_chown)
    monkeypatch.setattr(remote_storage.os, "fchown", tracked_fchown)
    staged = remote_storage.stage_john1_runtime(
        client, "bundles/runtime-test/runtime-manifest.json"
    )
    assert {entry.name for entry in staged.directory.iterdir()} == {
        "runtime-manifest.json",
        "cascadia-r2-runtime",
    }
    assert [
        entry["relative"] for entry in staged.john1_staging_proof["staged_inventory"]["entries"]
    ] == ["cascadia-r2-runtime", "runtime-manifest.json"]
    assert staged.manifest["manifest_sha256"] == manifest["manifest_sha256"]
    assert (staged.directory, os.getuid(), os.getgid()) in chown_calls
    assert fchown_calls == [(os.getuid(), os.getgid()), (os.getuid(), os.getgid())]
    cleanup = remote_storage.cleanup_john1_runtime_directory(staged.directory)
    assert cleanup["removed"] is True
    assert not list(tmp_path.glob("cascadia-r2-map-runtime-*"))


def test_exact_stage_validation_rejects_hardlinks_and_wrong_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _manifest = _runtime_harness(tmp_path, monkeypatch)
    staged = remote_storage.stage_john1_runtime(
        client, "bundles/runtime-test/runtime-manifest.json"
    )
    external_link = tmp_path / "runtime-hardlink"
    os.link(staged.executable, external_link)
    try:
        with pytest.raises(remote_storage.RemoteStorageError, match="mode/size"):
            remote_storage._validate_exact_staging_directory(staged.directory)
    finally:
        external_link.unlink()

    real_lstat = os.lstat

    def wrong_owner_lstat(path):
        details = real_lstat(path)
        if Path(path) == staged.manifest_path:
            fields = list(details)
            fields[4] = details.st_uid + 1
            return os.stat_result(fields)
        return details

    with monkeypatch.context() as scoped:
        scoped.setattr(remote_storage.os, "lstat", wrong_owner_lstat)
        with pytest.raises(remote_storage.RemoteStorageError, match="mode/size"):
            remote_storage._validate_exact_staging_directory(staged.directory)
    cleanup = remote_storage.cleanup_john1_runtime_directory(staged.directory)
    assert cleanup["removed"] is True


@pytest.mark.parametrize(
    ("download_fault", "identity_fault"),
    (("mid-download", None), ("hash", None), (None, "codesign"), (None, "xattr")),
)
def test_stage_runtime_faults_clean_exact_namespace_and_publish_authenticated_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    download_fault: str | None,
    identity_fault: str | None,
) -> None:
    client, _manifest = _runtime_harness(
        tmp_path,
        monkeypatch,
        download_fault=download_fault,
        identity_fault=identity_fault,
    )
    with pytest.raises((OSError, RemoteProtocolError)):
        remote_storage.stage_john1_runtime(client, "bundles/runtime-test/runtime-manifest.json")
    assert not list(tmp_path.glob("cascadia-r2-map-runtime-*"))
    assert len(client.publications) == 1
    relative, encoded = client.publications[0]
    assert relative.startswith("control/staging-cleanups/stage-failure-runtime-test-")
    receipt = json.loads(encoded)
    assert receipt["cleanup"]["removed"] is True
    assert receipt["cleanup_receipt_sha256"] == document_sha256(receipt, "cleanup_receipt_sha256")


class _RuntimeProcess:
    def __init__(self, *, timeout: bool = False) -> None:
        self.stdout = io.BytesIO(b"runtime stdout\n")
        self.stderr = io.BytesIO(b"runtime stderr\n")
        self.timeout = timeout
        self.killed = False
        self.wait_count = 0

    def wait(self, timeout=None):
        self.wait_count += 1
        if self.timeout and timeout is not None and not self.killed:
            raise remote_storage.subprocess.TimeoutExpired("runtime", timeout)
        return -9 if self.killed else 0

    def kill(self) -> None:
        self.killed = True


@pytest.mark.parametrize("fault", (None, "launch", "timeout", "stdout", "stderr"))
def test_execute_runtime_success_and_faults_always_clean_and_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str | None,
) -> None:
    client, _manifest = _runtime_harness(
        tmp_path,
        monkeypatch,
        stream_fault=fault if fault in {"stdout", "stderr"} else None,
    )
    process = _RuntimeProcess(timeout=fault == "timeout")
    observed_prelaunch: list[tuple[str, int]] = []

    def fake_popen(*_args, **kwargs):
        directory = Path(kwargs["cwd"])
        observed_prelaunch.extend(
            sorted(
                (entry.name, stat.S_IMODE(entry.stat().st_mode)) for entry in directory.iterdir()
            )
        )
        if fault == "launch":
            raise OSError("injected launch failure")
        return process

    monkeypatch.setattr(remote_storage.subprocess, "Popen", fake_popen)
    if fault == "launch":
        expected_error = OSError
    elif fault in {"stdout", "stderr"}:
        expected_error = remote_storage.RemoteStorageError
    else:
        expected_error = None
    if expected_error is None:
        result = remote_storage.execute_john1_runtime(
            client,
            manifest_relative="bundles/runtime-test/runtime-manifest.json",
            run_id="fault-test",
            timeout_seconds=1,
        )
        assert result["execution"]["timed_out"] is (fault == "timeout")
        assert result["cleanup"]["cleanup"]["removed"] is True
        assert result["execution_remote"]["storage_receipt_sha256"] == "ab" * 32
        assert result["cleanup_remote"]["storage_receipt_sha256"] == "ab" * 32
    else:
        with pytest.raises(expected_error):
            remote_storage.execute_john1_runtime(
                client,
                manifest_relative="bundles/runtime-test/runtime-manifest.json",
                run_id="fault-test",
                timeout_seconds=1,
            )
    assert observed_prelaunch == [
        ("cascadia-r2-runtime", 0o500),
        ("runtime-manifest.json", 0o400),
    ]
    assert not list(tmp_path.glob("cascadia-r2-map-runtime-*"))
    cleanup_publications = [
        json.loads(payload)
        for relative, payload in client.publications
        if relative.startswith("control/staging-cleanups/")
    ]
    assert len(cleanup_publications) == 1
    assert cleanup_publications[0]["cleanup"]["removed"] is True


@pytest.mark.parametrize(
    ("omit_receipt_for", "error_match"),
    (("stdout", "output streaming failed"), ("staging-cleanups", "receipt publication failed")),
)
def test_execute_runtime_rejects_missing_stream_or_cleanup_storage_receipt_after_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    omit_receipt_for: str,
    error_match: str,
) -> None:
    client, _manifest = _runtime_harness(
        tmp_path,
        monkeypatch,
        omit_receipt_for=omit_receipt_for,
    )
    monkeypatch.setattr(
        remote_storage.subprocess, "Popen", lambda *_args, **_kwargs: _RuntimeProcess()
    )
    with pytest.raises(remote_storage.RemoteStorageError, match=error_match):
        remote_storage.execute_john1_runtime(
            client,
            manifest_relative="bundles/runtime-test/runtime-manifest.json",
            run_id="missing-receipt",
            timeout_seconds=1,
        )
    assert not list(tmp_path.glob("cascadia-r2-map-runtime-*"))
    assert any(
        relative.startswith("control/staging-cleanups/")
        for relative, _payload in client.publications
    )


def test_ssh_terminate_kills_descendants_after_process_leader_has_exited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExitedLeader:
        pid = 424242

        @staticmethod
        def wait(timeout=None):
            del timeout
            return 0

        @staticmethod
        def poll():
            return 0

    group_exists = True
    signals: list[int] = []

    def killpg(_pid: int, sent_signal: int) -> None:
        nonlocal group_exists
        if not group_exists:
            raise ProcessLookupError
        if sent_signal != 0:
            signals.append(sent_signal)
        if sent_signal == remote_storage.signal.SIGKILL:
            group_exists = False

    monkeypatch.setattr(remote_storage.os, "killpg", killpg)
    SshTransport._terminate_and_reap(ExitedLeader())  # type: ignore[arg-type]
    assert signals == [remote_storage.signal.SIGTERM, remote_storage.signal.SIGKILL]


def test_worker_atomic_put_cas_and_object_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _test_contract(tmp_path)

    def storage_proof(_contract: object, *, measure_size: bool = True) -> dict[str, int]:
        assert measure_size is True
        return {
            "campaign_apparent_bytes": remote_worker._campaign_apparent_size(contract),
            "campaign_data_apparent_bytes": remote_worker._campaign_apparent_size(contract),
            "free_bytes": 1 << 40,
            "max_campaign_bytes": contract.max_campaign_bytes,
            "receipt_apparent_bytes": 0,
            "receipt_entries": 0,
        }

    monkeypatch.setattr(remote_worker, "verify_root", storage_proof)
    payload = b"immutable remote bytes"
    arguments = {
        "relative": "reports/result.bin",
        "size": len(payload),
        "sha256": content_sha256(payload),
        "expected_current": "absent",
        "mutable": False,
    }
    receipt = remote_worker._put_file(contract, arguments, io.BytesIO(payload))
    path = contract.root / "reports/result.bin"
    assert path.read_bytes() == payload
    assert stat.S_IMODE(path.stat().st_mode) == 0o400
    assert receipt["previous_sha256"] is None
    with pytest.raises(remote_worker.RemoteWorkerError, match="precondition"):
        remote_worker._put_file(contract, arguments, io.BytesIO(payload))

    replacement = b"replacement"
    replaced = remote_worker._put_file(
        contract,
        {
            **arguments,
            "size": len(replacement),
            "sha256": content_sha256(replacement),
            "expected_current": content_sha256(payload),
            "mutable": True,
        },
        io.BytesIO(replacement),
    )
    assert path.read_bytes() == replacement
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert replaced["previous_sha256"] == content_sha256(payload)
    token = remote_worker._object_token(contract, "reports/result.bin")
    assert token["sha256"] == content_sha256(replacement)
    assert token["token_sha256"] == remote_worker.document_sha256(token, "token_sha256")


def test_put_journal_recovers_lost_response_and_finalizes_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _test_contract(tmp_path)
    monkeypatch.setattr(
        remote_worker,
        "verify_root",
        lambda _contract, *, measure_size=True: _test_storage_proof(contract),
    )
    payload = b"crash-recoverable immutable payload"
    arguments = {
        "relative": "reports/recoverable.bin",
        "size": len(payload),
        "sha256": content_sha256(payload),
        "expected_current": "absent",
        "mutable": False,
    }
    request = _worker_request("put-file", arguments, "req-put-recoverable")
    context: dict = {}
    result = remote_worker._put_file(
        contract,
        arguments,
        io.BytesIO(payload),
        request=request,
        commit_context=context,
    )
    target = contract.root / arguments["relative"]
    before = target.stat()
    assert context["journal"].exists()
    assert result["storage_transaction"]["campaign_data_apparent_bytes"] == (
        result["projected_data_bytes"] + result["transaction_overhead_bytes"]
    )

    recovered = remote_worker._recovered_put_result(contract, context)
    recovered["payload_size"] = 0
    recovered["payload_sha256"] = content_sha256(b"")
    receipt = remote_worker._receipt(
        request,
        contract.expected_identity_sha256,
        "ok",
        recovered,
    )
    receipt["root"] = str(contract.root)
    receipt["receipt_sha256"] = remote_worker.document_sha256(receipt, "receipt_sha256")
    lock_fd, _ = remote_worker._global_lock(contract)
    try:
        remote_worker._persist_receipt_locked(contract, receipt)
        assert remote_worker._load_replay_receipt_locked(contract, request) == receipt
        remote_worker._finalize_put_commit(context)
    finally:
        os.close(lock_fd)

    after = target.stat()
    assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
    assert target.read_bytes() == payload
    assert not context["journal"].exists()
    assert not context["backup"].exists()


def test_receipt_query_recovers_pending_put_without_resending_body(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _test_contract(tmp_path)
    monkeypatch.setattr(
        remote_worker,
        "verify_root",
        lambda _contract, *, measure_size=True: _test_storage_proof(contract),
    )
    payload = b"recover from durable staged object without replaying this body" * 2048
    arguments = {
        "relative": "reports/query-recovered.bin",
        "size": len(payload),
        "sha256": content_sha256(payload),
        "expected_current": "absent",
        "mutable": False,
    }
    request = _worker_request("put-file", arguments, "req-put-query-recovery")
    lock_fd, _ = remote_worker._global_lock(contract)
    try:
        remote_worker._reserve_receipt_capacity_locked(contract, request)
    finally:
        os.close(lock_fd)
    context: dict = {}
    remote_worker._put_file(
        contract,
        arguments,
        io.BytesIO(payload),
        request=request,
        commit_context=context,
    )
    assert context["journal"].exists()
    assert context["data_reservation"].exists()

    query = remote_worker._query_receipt(
        contract,
        {
            "request_id": request["request_id"],
            "semantic_sha256": request["semantic_sha256"],
            "command_sha256": request["command_sha256"],
            "operation": request["operation"],
        },
    )

    assert query["found"] is True
    assert query["receipt"]["result"]["sha256"] == content_sha256(payload)
    assert query["journal_present"] is False
    assert query["receipt_reservation_present"] is False
    assert query["data_reservation_present"] is False
    assert (contract.root / arguments["relative"]).read_bytes() == payload
    assert not context["journal"].exists()
    assert not context["data_reservation"].exists()


@pytest.mark.parametrize("reserved", ("receipt-reservations", "data-reservations"))
def test_public_uploads_cannot_poison_capacity_reservation_namespaces(
    tmp_path: Path,
    reserved: str,
) -> None:
    contract = _test_contract(tmp_path)
    payload = b"poison"
    arguments = {
        "relative": f"control/{reserved}/poison.json",
        "size": len(payload),
        "sha256": content_sha256(payload),
        "expected_current": "absent",
        "mutable": False,
    }
    with pytest.raises(remote_worker.RemoteWorkerError, match="reserved control namespace"):
        remote_worker._put_file(contract, arguments, io.BytesIO(payload))
    with pytest.raises(remote_worker.RemoteWorkerError, match="reserved control namespace"):
        remote_worker._put_unknown_stream(
            contract,
            {"relative": arguments["relative"], "max_bytes": 64},
            io.BytesIO(payload),
        )


def test_put_journal_rolls_back_every_pre_receipt_crash_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _test_contract(tmp_path)
    monkeypatch.setattr(
        remote_worker,
        "verify_root",
        lambda _contract, *, measure_size=True: _test_storage_proof(contract),
    )
    target = contract.root / "reports/mutable.json"
    target.write_bytes(b"old")
    target.chmod(0o600)
    replacement = b"new"
    arguments = {
        "relative": "reports/mutable.json",
        "size": len(replacement),
        "sha256": content_sha256(replacement),
        "expected_current": content_sha256(b"old"),
        "mutable": True,
    }
    request = _worker_request("publish-status", arguments, "req-put-mutable-crash")
    staging = remote_worker._stage_stream(
        target,
        io.BytesIO(replacement),
        len(replacement),
        content_sha256(replacement),
        0o600,
    )
    context = remote_worker._put_commit_context(
        contract,
        request,
        target,
        staging,
        expected_current=content_sha256(b"old"),
        current_size=3,
        expected_sha256=content_sha256(replacement),
        size=3,
        mode=0o600,
        storage_precommit=_test_storage_proof(contract),
        storage_staged=_test_storage_proof(contract),
        receipt_reservation_apparent_bytes=0,
        data_reservation_apparent_bytes=0,
    )
    # Crash after journal fsync, before the first rename.
    remote_worker._rollback_put_commit(context)
    assert target.read_bytes() == b"old"
    assert not staging.exists() and not context["journal"].exists()

    staging = remote_worker._stage_stream(
        target,
        io.BytesIO(replacement),
        len(replacement),
        content_sha256(replacement),
        0o600,
    )
    context = remote_worker._put_commit_context(
        contract,
        request,
        target,
        staging,
        expected_current=content_sha256(b"old"),
        current_size=3,
        expected_sha256=content_sha256(replacement),
        size=3,
        mode=0o600,
        storage_precommit=_test_storage_proof(contract),
        storage_staged=_test_storage_proof(contract),
        receipt_reservation_apparent_bytes=0,
        data_reservation_apparent_bytes=0,
    )
    os.rename(target, context["backup"])
    # Crash after old->backup, before staged->target.
    remote_worker._rollback_put_commit(context)
    assert target.read_bytes() == b"old"
    assert not context["backup"].exists() and not context["journal"].exists()


def test_put_reserves_receipt_capacity_before_target_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _test_contract(tmp_path)
    proof = _test_storage_proof(contract)
    proof["receipt_entries"] = contract.max_receipt_entries
    monkeypatch.setattr(
        remote_worker,
        "verify_root",
        lambda _contract, *, measure_size=True: dict(proof),
    )
    payload = b"must not commit"
    arguments = {
        "relative": "reports/no-capacity.bin",
        "size": len(payload),
        "sha256": content_sha256(payload),
        "expected_current": "absent",
        "mutable": False,
    }
    with pytest.raises(remote_worker.RemoteWorkerError, match="receipt capacity"):
        remote_worker._put_file(contract, arguments, io.BytesIO(payload))
    assert not (contract.root / arguments["relative"]).exists()


def test_worker_rejects_symlink_and_device_escape(tmp_path: Path) -> None:
    contract = _test_contract(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (contract.root / "reports/link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(remote_worker.RemoteWorkerError, match="symlink"):
        remote_worker._safe_path(contract, "reports/link/value", "test")


def test_worker_symlink_diagnostics_distinguish_dangling_and_escape_without_target_leak(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "campaign"
    run = campaign / "tmp/run-diagnostics"
    run.mkdir(parents=True)

    dangling = run / "dangling-link"
    dangling.symlink_to(run / "external-sensitive-missing-target")
    with pytest.raises(
        remote_worker.RemoteWorkerError,
        match=r"campaign symlink is dangling: tmp/run-diagnostics/dangling-link$",
    ) as dangling_error:
        remote_worker._apparent_size(
            run,
            allowed_symlink_prefixes=(run,),
            diagnostic_root=campaign,
        )
    assert "external-sensitive-missing-target" not in str(dangling_error.value)

    dangling.unlink()
    external = tmp_path / "external-sensitive-existing-target"
    external.write_bytes(b"outside\n")
    escaping = run / "escaping-link"
    escaping.symlink_to(external)
    with pytest.raises(
        remote_worker.RemoteWorkerError,
        match=r"campaign symlink escapes its run boundary: tmp/run-diagnostics/escaping-link$",
    ) as escaping_error:
        remote_worker._apparent_size(
            run,
            allowed_symlink_prefixes=(run,),
            diagnostic_root=campaign,
        )
    assert "external-sensitive-existing-target" not in str(escaping_error.value)


def test_unknown_stream_is_bounded_atomic_and_hash_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _test_contract(tmp_path)
    monkeypatch.setattr(
        remote_worker,
        "verify_root",
        lambda _contract, *, measure_size=True: _test_storage_proof(contract),
    )
    payload = b"headless stdout\n" * 8
    arguments = {
        "relative": "logs/headless/turn-1.log",
        "max_bytes": len(payload),
        "expected_current": "absent",
    }
    request = _worker_request("put-stream", arguments, "req-stream-turn-1")
    lock_fd, _ = remote_worker._global_lock(contract)
    try:
        remote_worker._reserve_receipt_capacity_locked(contract, request)
    finally:
        os.close(lock_fd)
    context: dict = {}
    result = remote_worker._put_unknown_stream(
        contract,
        arguments,
        io.BytesIO(payload),
        request=request,
        commit_context=context,
    )
    assert result["sha256"] == content_sha256(payload)
    assert result["size"] == len(payload)
    assert (contract.root / "logs/headless/turn-1.log").read_bytes() == payload
    query = remote_worker._query_receipt(
        contract,
        {
            "request_id": request["request_id"],
            "semantic_sha256": request["semantic_sha256"],
            "command_sha256": request["command_sha256"],
            "operation": "put-stream",
        },
    )
    assert query["found"] is True
    assert query["journal_present"] is False
    assert query["data_reservation_present"] is False
    rejected_arguments = {
        "relative": "logs/headless/turn-2.log",
        "max_bytes": len(payload) - 1,
        "expected_current": "absent",
    }
    rejected_request = _worker_request(
        "put-stream", rejected_arguments, "req-stream-turn-2"
    )
    with pytest.raises(remote_worker.RemoteWorkerError, match="exceeded"):
        remote_worker._put_unknown_stream(
            contract,
            rejected_arguments,
            io.BytesIO(payload),
            request=rejected_request,
        )
    assert not (contract.root / "logs/headless/turn-2.log").exists()


def test_lease_lock_requires_owner_token_and_supports_renew_release(tmp_path: Path) -> None:
    contract = _test_contract(tmp_path)
    acquire_arguments = {
        "name": "controller",
        "owner": "root-orchestrator",
        "lease_seconds": 60,
        "lease_epoch": "acquire-1",
    }
    acquire_request = _worker_request("lock-acquire", acquire_arguments, "req-lock-acquire-1")
    acquired = remote_worker._lease_operation(
        contract,
        "lock-acquire",
        acquire_arguments,
        acquire_request,
    )
    assert remote_worker._lease_operation(
        contract, "lock-acquire", acquire_arguments, acquire_request
    ) == acquired
    other_arguments = {
        "name": "controller",
        "owner": "other",
        "lease_seconds": 60,
        "lease_epoch": "acquire-other-1",
    }
    with pytest.raises(remote_worker.RemoteWorkerError, match="held"):
        remote_worker._lease_operation(
            contract,
            "lock-acquire",
            other_arguments,
            _worker_request("lock-acquire", other_arguments, "req-lock-acquire-other"),
        )
    renew_arguments = {
        "name": "controller",
        "owner": "root-orchestrator",
        "token": acquired["token"],
        "lease_seconds": 60,
        "lease_epoch": "renew-1",
    }
    renew_request = _worker_request("lock-renew", renew_arguments, "req-lock-renew-1")
    renewed = remote_worker._lease_operation(
        contract,
        "lock-renew",
        renew_arguments,
        renew_request,
    )
    assert renewed["revision"] == acquired["revision"] + 1
    assert remote_worker._lease_operation(
        contract, "lock-renew", renew_arguments, renew_request
    ) == renewed
    release_arguments = {
        "name": "controller",
        "owner": "root-orchestrator",
        "token": acquired["token"],
        "lease_seconds": 60,
        "lease_epoch": "release-1",
    }
    release_request = _worker_request(
        "lock-release", release_arguments, "req-lock-release-1"
    )
    released = remote_worker._lease_operation(
        contract,
        "lock-release",
        release_arguments,
        release_request,
    )
    assert released["released"] is True
    assert released["active"] is False
    assert remote_worker._lease_operation(
        contract, "lock-release", release_arguments, release_request
    ) == released
    assert remote_worker._lease_operation(
        contract, "lock-renew", renew_arguments, renew_request
    ) == renewed
    current = remote_worker._load_lease_document(
        contract,
        remote_worker._lock_path(contract, "controller"),
        "controller",
    )
    assert current == released
    history = remote_worker._lease_history_path(contract, "controller", "renew-1")
    assert remote_worker._load_lease_document(
        contract,
        history,
        "controller",
        request=renew_request,
        lease_epoch="renew-1",
    ) == renewed


def test_lease_write_ahead_transition_recovers_before_exact_retry(tmp_path: Path) -> None:
    contract = _test_contract(tmp_path)
    acquire_arguments = {
        "name": "controller",
        "owner": "root-orchestrator",
        "lease_seconds": 60,
        "lease_epoch": "acquire-pending",
    }
    acquire_request = _worker_request(
        "lock-acquire", acquire_arguments, "req-lock-acquire-pending"
    )
    document = {
        "schema_id": "cascadia.r2-map.remote-lease.v2",
        "name": "controller",
        "owner": "root-orchestrator",
        "token": "a" * 64,
        "revision": 1,
        "active": True,
        "released": False,
        "lease_epoch": "acquire-pending",
        "request_id": acquire_request["request_id"],
        "semantic_sha256": acquire_request["semantic_sha256"],
        "command_sha256": acquire_request["command_sha256"],
        "issued_unix_ms": 1000,
        "expires_unix_ms": 61000,
    }
    document["lease_sha256"] = document_sha256(document, "lease_sha256")
    pending = remote_worker._lease_pending_path(contract, "controller")
    remote_worker._atomic_write(pending, canonical_json(document), 0o600)

    recovered = remote_worker._lease_operation(
        contract,
        "lock-acquire",
        acquire_arguments,
        acquire_request,
    )
    assert recovered == document
    assert not pending.exists()
    assert remote_worker._load_lease_document(
        contract,
        remote_worker._lock_path(contract, "controller"),
        "controller",
    ) == document
    assert remote_worker._load_lease_document(
        contract,
        remote_worker._lease_history_path(
            contract, "controller", "acquire-pending"
        ),
        "controller",
        request=acquire_request,
        lease_epoch="acquire-pending",
    ) == document


def test_checkpoint_transaction_is_immutable_atomic_and_complete(tmp_path: Path) -> None:
    contract = _test_contract(tmp_path)
    model = b"model tensors"
    optimizer = b"optimizer tensors"
    objects = [
        {"relative": "model.bin", "size": len(model), "sha256": content_sha256(model)},
        {
            "relative": "state/optimizer.bin",
            "size": len(optimizer),
            "sha256": content_sha256(optimizer),
        },
    ]
    manifest = _manifest("checkpoint-001", "checkpoints/checkpoint-001", objects)
    manifest_payload = canonical_json(manifest)
    begun = remote_worker._transaction_begin(
        contract,
        {"size": len(manifest_payload), "sha256": content_sha256(manifest_payload)},
        io.BytesIO(manifest_payload),
    )
    assert begun["object_count"] == 2
    for descriptor in manifest["objects"]:
        # Match payload by canonical descriptor order.
        payload = optimizer if descriptor["relative"].startswith("state/") else model
        put_arguments = {"transaction_id": "checkpoint-001", **descriptor}
        remote_worker._transaction_put(
            contract,
            put_arguments,
            io.BytesIO(payload),
            _worker_request(
                "transaction-put",
                put_arguments,
                "req-transaction-put-{}".format(descriptor["relative"].replace("/", "-")),
            ),
        )
    committed = remote_worker._transaction_commit(
        contract,
        {
            "transaction_id": "checkpoint-001",
            "manifest_sha256": manifest["manifest_sha256"],
        },
    )
    target = contract.root / "checkpoints/checkpoint-001"
    assert committed["committed"] is True
    assert (target / "model.bin").read_bytes() == model
    assert (target / "state/optimizer.bin").read_bytes() == optimizer
    assert json.loads((target / ".r2-map-transaction.json").read_bytes()) == manifest
    assert stat.S_IMODE(target.stat().st_mode) == 0o500
    assert stat.S_IMODE((target / "model.bin").stat().st_mode) == 0o400
    repeated = remote_worker._transaction_commit(
        contract,
        {
            "transaction_id": "checkpoint-001",
            "manifest_sha256": manifest["manifest_sha256"],
        },
    )
    assert repeated["committed"] is True
    assert repeated["recovered"] is True
    assert repeated["outcome_sha256"] == committed["outcome_sha256"]


def test_upload_preflight_failures_release_exact_reservations_and_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _test_contract(tmp_path)
    monkeypatch.setattr(
        remote_worker,
        "verify_root",
        lambda _contract, *, measure_size=True: _test_storage_proof(contract),
    )
    payload = b"reservation-cleanup" * 1024
    original_reserve = remote_worker._reserve_data_capacity_locked
    injected_staging: list[Path] = []

    def reserve_then_poison(*args, **kwargs):
        reservation = original_reserve(*args, **kwargs)
        assert injected_staging
        injected_staging[-1].symlink_to(tmp_path / "outside-staging")
        return reservation

    monkeypatch.setattr(
        remote_worker, "_reserve_data_capacity_locked", reserve_then_poison
    )

    put_arguments = {
        "relative": "reports/poisoned-put.bin",
        "size": len(payload),
        "sha256": content_sha256(payload),
        "expected_current": "absent",
        "mutable": False,
    }
    put_request = _worker_request("put-file", put_arguments, "req-poisoned-put")
    put_staging = contract.root / "reports/.poisoned-put.bin.req-poisoned-put.put-staging"
    injected_staging.append(put_staging)
    with pytest.raises(remote_worker.RemoteWorkerError, match="staging"):
        remote_worker._put_file(
            contract,
            put_arguments,
            io.BytesIO(payload),
            request=put_request,
        )
    assert not os.path.lexists(put_staging)
    assert not remote_worker._data_reservation_path(contract, put_request["request_id"]).exists()

    stream_arguments = {
        "relative": "logs/poisoned-stream.bin",
        "max_bytes": len(payload),
        "expected_current": "absent",
    }
    stream_request = _worker_request(
        "put-stream", stream_arguments, "req-poisoned-stream"
    )
    stream_staging = (
        contract.root / "logs/.poisoned-stream.bin.req-poisoned-stream.stream-staging"
    )
    injected_staging.append(stream_staging)
    with pytest.raises(remote_worker.RemoteWorkerError, match="staging"):
        remote_worker._put_unknown_stream(
            contract,
            stream_arguments,
            io.BytesIO(payload),
            request=stream_request,
        )
    assert not os.path.lexists(stream_staging)
    assert not remote_worker._data_reservation_path(
        contract, stream_request["request_id"]
    ).exists()

    object_descriptor = {
        "relative": "object.bin",
        "size": len(payload),
        "sha256": content_sha256(payload),
    }
    manifest = _manifest(
        "poisoned-transaction",
        "checkpoints/poisoned-transaction",
        [object_descriptor],
    )
    encoded = canonical_json(manifest)
    remote_worker._transaction_begin(
        contract,
        {"size": len(encoded), "sha256": content_sha256(encoded)},
        io.BytesIO(encoded),
    )
    transaction_arguments = {
        "transaction_id": "poisoned-transaction",
        **object_descriptor,
    }
    transaction_request = _worker_request(
        "transaction-put", transaction_arguments, "req-poisoned-transaction"
    )
    tree = remote_worker._transaction_tree(contract, manifest)
    transaction_staging = (
        tree / ".object.bin.req-poisoned-transaction.transaction-staging"
    )
    injected_staging.append(transaction_staging)
    with pytest.raises(remote_worker.RemoteWorkerError, match="staging"):
        remote_worker._transaction_put(
            contract,
            transaction_arguments,
            io.BytesIO(payload),
            transaction_request,
        )
    assert not os.path.lexists(transaction_staging)
    assert not remote_worker._data_reservation_path(
        contract, transaction_request["request_id"]
    ).exists()


def test_atomic_receipt_and_reservation_partials_are_exactly_recovered(
    tmp_path: Path,
) -> None:
    contract = _test_contract(tmp_path)
    arguments = {
        "relative": "reports/recoverable.bin",
        "size": 1,
        "sha256": content_sha256(b"x"),
        "expected_current": "absent",
        "mutable": False,
    }
    request = _worker_request("put-file", arguments, "req-recoverable-partials")
    receipt_reservation = remote_worker._receipt_reservation_path(
        contract, request["request_id"]
    )
    receipt_partial = receipt_reservation.parent / (
        f".{receipt_reservation.name}.pending.tmp"
    )
    receipt_partial.write_bytes(b"partial")
    receipt_partial.chmod(0o600)
    lock_fd, _ = remote_worker._global_lock(contract)
    try:
        assert remote_worker._reserve_receipt_capacity_locked(
            contract, request
        ) == receipt_reservation
    finally:
        os.close(lock_fd)
    assert receipt_reservation.exists()
    assert not receipt_partial.exists()

    data_reservation = remote_worker._data_reservation_path(
        contract, request["request_id"]
    )
    data_partial = data_reservation.parent / (
        f".{data_reservation.name}.pending.tmp"
    )
    data_partial.write_bytes(b"partial")
    data_partial.chmod(0o600)
    lock_fd, _ = remote_worker._global_lock(contract)
    try:
        storage = remote_worker._storage_capacity_state(contract)
        assert remote_worker._reserve_data_capacity_locked(
            contract, request, storage
        ) == data_reservation
    finally:
        os.close(lock_fd)
    assert data_reservation.exists()
    assert not data_partial.exists()

    receipt = remote_worker._receipt(
        request,
        contract.expected_identity_sha256,
        "ok",
        {"payload_size": 0, "payload_sha256": content_sha256(b"")},
        contract=contract,
    )
    receipt_path = contract.root / (
        "control/receipts/{}.json".format(request["request_id"])
    )
    command_partial = receipt_path.parent / (f".{receipt_path.name}.pending.tmp")
    command_partial.write_bytes(b"partial")
    command_partial.chmod(0o400)
    lock_fd, _ = remote_worker._global_lock(contract)
    try:
        installed = remote_worker._persist_receipt_locked(contract, receipt)
    finally:
        os.close(lock_fd)
    assert installed["disposition"] == "installed"
    assert receipt_path.exists()
    assert not command_partial.exists()


def test_commit_validation_failure_does_not_terminalize_and_abort_remains_legal(
    tmp_path: Path,
) -> None:
    contract = _test_contract(tmp_path)
    payload = b"commit validation"
    descriptor = {
        "relative": "model.bin",
        "size": len(payload),
        "sha256": content_sha256(payload),
    }
    manifest = _manifest("invalid-commit", "checkpoints/invalid-commit", [descriptor])
    encoded = canonical_json(manifest)
    remote_worker._transaction_begin(
        contract,
        {"size": len(encoded), "sha256": content_sha256(encoded)},
        io.BytesIO(encoded),
    )
    arguments = {"transaction_id": "invalid-commit", **descriptor}
    remote_worker._transaction_put(
        contract,
        arguments,
        io.BytesIO(payload),
        _worker_request("transaction-put", arguments, "req-invalid-commit-put"),
    )
    object_path = remote_worker._transaction_tree(contract, manifest) / "model.bin"
    object_path.chmod(0o600)
    with pytest.raises(remote_worker.RemoteWorkerError, match="commit verification"):
        remote_worker._transaction_commit(
            contract,
            {
                "transaction_id": "invalid-commit",
                "manifest_sha256": manifest["manifest_sha256"],
            },
        )
    assert not remote_worker._transaction_outcome_path(
        contract, "invalid-commit"
    ).exists()
    aborted = remote_worker._transaction_abort(
        contract,
        {
            "transaction_id": "invalid-commit",
            "manifest_sha256": manifest["manifest_sha256"],
        },
    )
    assert aborted["aborted"] is True


def test_abort_validates_cleanup_tree_before_terminal_outcome(tmp_path: Path) -> None:
    contract = _test_contract(tmp_path)
    payload = b"abort validation"
    manifest = _manifest(
        "unsafe-abort",
        "checkpoints/unsafe-abort",
        [{"relative": "model.bin", "size": len(payload), "sha256": content_sha256(payload)}],
    )
    encoded = canonical_json(manifest)
    remote_worker._transaction_begin(
        contract,
        {"size": len(encoded), "sha256": content_sha256(encoded)},
        io.BytesIO(encoded),
    )
    tree = remote_worker._transaction_tree(contract, manifest)
    alias = tree / "unsafe-link"
    alias.symlink_to(tmp_path / "outside")
    arguments = {
        "transaction_id": "unsafe-abort",
        "manifest_sha256": manifest["manifest_sha256"],
    }
    with pytest.raises(remote_worker.RemoteWorkerError, match="unsafe entry"):
        remote_worker._transaction_abort(contract, arguments)
    assert not remote_worker._transaction_outcome_path(contract, "unsafe-abort").exists()
    alias.unlink()
    assert remote_worker._transaction_abort(contract, arguments)["aborted"] is True


def test_transaction_outcome_temp_and_postrename_states_recover_exactly(
    tmp_path: Path,
) -> None:
    contract = _test_contract(tmp_path)
    payload = b"postrename recovery"
    descriptor = {
        "relative": "model.bin",
        "size": len(payload),
        "sha256": content_sha256(payload),
    }
    manifest = _manifest("postrename", "checkpoints/postrename", [descriptor])
    outcome_path = remote_worker._transaction_outcome_path(contract, "postrename")
    orphan = outcome_path.parent / (f".{outcome_path.name}.pending.tmp")
    orphan.write_bytes(b"partial")
    orphan.chmod(0o400)
    encoded = canonical_json(manifest)
    remote_worker._transaction_begin(
        contract,
        {"size": len(encoded), "sha256": content_sha256(encoded)},
        io.BytesIO(encoded),
    )
    assert not orphan.exists()
    put_arguments = {"transaction_id": "postrename", **descriptor}
    remote_worker._transaction_put(
        contract,
        put_arguments,
        io.BytesIO(payload),
        _worker_request("transaction-put", put_arguments, "req-postrename-put"),
    )
    tree = remote_worker._transaction_tree(contract, manifest)
    target = contract.root / manifest["target_relative"]
    outcome = remote_worker._transaction_outcome(manifest, "commit")
    remote_worker._atomic_write(outcome_path, canonical_json(outcome), 0o400)
    remote_worker._atomic_write(
        tree / ".r2-map-transaction.json", canonical_json(manifest), 0o400
    )
    os.rename(tree, target)
    target.chmod(0o700)
    recovered = remote_worker._transaction_commit(
        contract,
        {
            "transaction_id": "postrename",
            "manifest_sha256": manifest["manifest_sha256"],
        },
    )
    assert recovered["recovered"] is True
    assert stat.S_IMODE(target.stat().st_mode) == 0o500
    assert not remote_worker._transaction_root(contract, "postrename").exists()


def test_same_target_transaction_loser_abort_preserves_winner_and_cleans_itself(
    tmp_path: Path,
) -> None:
    contract = _test_contract(tmp_path)
    winner_payload = b"winner"
    loser_payload = b"loser"
    target = "checkpoints/shared-target"
    winner = _manifest(
        "shared-winner",
        target,
        [
            {
                "relative": "model.bin",
                "size": len(winner_payload),
                "sha256": content_sha256(winner_payload),
            }
        ],
    )
    loser = _manifest(
        "shared-loser",
        target,
        [
            {
                "relative": "model.bin",
                "size": len(loser_payload),
                "sha256": content_sha256(loser_payload),
            }
        ],
    )
    for manifest, payload in ((winner, winner_payload), (loser, loser_payload)):
        encoded = canonical_json(manifest)
        remote_worker._transaction_begin(
            contract,
            {"size": len(encoded), "sha256": content_sha256(encoded)},
            io.BytesIO(encoded),
        )
        arguments = {"transaction_id": manifest["transaction_id"], **manifest["objects"][0]}
        remote_worker._transaction_put(
            contract,
            arguments,
            io.BytesIO(payload),
            _worker_request(
                "transaction-put", arguments, "req-put-{}".format(manifest["transaction_id"])
            ),
        )
    remote_worker._transaction_commit(
        contract,
        {
            "transaction_id": winner["transaction_id"],
            "manifest_sha256": winner["manifest_sha256"],
        },
    )
    with pytest.raises(
        remote_worker.RemoteWorkerError,
        match=r"target and transaction tree both exist|provenance differs",
    ):
        remote_worker._transaction_commit(
            contract,
            {
                "transaction_id": loser["transaction_id"],
                "manifest_sha256": loser["manifest_sha256"],
            },
        )
    aborted = remote_worker._transaction_abort(
        contract,
        {
            "transaction_id": loser["transaction_id"],
            "manifest_sha256": loser["manifest_sha256"],
        },
    )
    assert aborted["target_owned_by_other_transaction"] is True
    visible = contract.root / target
    assert (visible / "model.bin").read_bytes() == winner_payload
    assert not remote_worker._transaction_root(contract, loser["transaction_id"]).exists()
    assert not remote_worker._transaction_tree(contract, loser).exists()


def _run_fixture_contract(tmp_path: Path) -> remote_worker.WorkerContract:
    contract = _test_contract(tmp_path)
    for relative in ("build", "cache/runs", "source/run-fixture"):
        path = contract.root / relative
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return contract


def _run_request(arguments: dict, request_id: str) -> dict:
    request = _worker_request("run-command", arguments, request_id)
    request["worker_sha256"] = content_sha256(
        Path(remote_worker.__file__).read_bytes()
    )
    request["command_sha256"] = remote_worker.request_command_sha256(request)
    return request


def test_durable_run_supervisor_finalizes_and_exact_retry_never_respawns(
    tmp_path: Path,
) -> None:
    contract = _run_fixture_contract(tmp_path)
    arguments = {
        "run_id": "durable-success",
        "cwd_relative": "source/run-fixture",
        "argv": ["/usr/bin/git", "--version"],
        "environment": {},
        "python_path_relatives": [],
        "timeout_seconds": 20,
        "output_relative": "reports/run-durable-success",
        "_test_max_run_bytes": 1 << 20,
    }
    request = _run_request(arguments, "req-run-durable-success")
    result = remote_worker._run_command(contract, arguments, request)
    assert result["exit_code"] == 0
    assert result["resource_exceeded"] is False
    assert result["stdout_size"] > 0
    assert result["temporary_cleaned"] is True
    assert not remote_worker._data_reservation_path(
        contract, request["request_id"]
    ).exists()
    assert not remote_worker._run_supervisor_path(
        contract, request["request_id"]
    ).exists()
    state = remote_worker._load_run_state(
        contract,
        remote_worker._run_state_path(contract, request["request_id"]),
        request,
    )
    assert state["phase"] == "finalized"
    assert remote_worker._run_command(contract, arguments, request) == result


def test_durable_run_watchdog_stops_aggregate_growth_and_releases_reservation(
    tmp_path: Path,
) -> None:
    contract = _run_fixture_contract(tmp_path)
    writer = contract.root / "source/run-fixture/grow.py"
    writer.write_text(
        "#!/usr/bin/python3\n"
        "import os\n"
        "root=os.environ['CARGO_TARGET_DIR']\n"
        "os.makedirs(root, exist_ok=True)\n"
        "i=0\n"
        "while True:\n"
        " p=os.path.join(root, f'chunk-{i:08d}')\n"
        " with open(p, 'wb') as f: f.write(b'x'*4096)\n"
        " i += 1\n"
    )
    writer.chmod(0o500)
    arguments = {
        "run_id": "durable-cap",
        "cwd_relative": "source/run-fixture",
        "argv": [str(writer)],
        "environment": {},
        "python_path_relatives": [],
        "timeout_seconds": 20,
        "output_relative": "reports/run-durable-cap",
        "_test_max_run_bytes": 256 * 1024,
    }
    request = _run_request(arguments, "req-run-durable-cap")
    result = remote_worker._run_command(contract, arguments, request)
    assert result["resource_exceeded"] is True
    assert result["exit_code"] == 125
    assert result["aggregate_monitor_samples"] >= 1
    assert not remote_worker._data_reservation_path(
        contract, request["request_id"]
    ).exists()
    assert not (contract.root / "build/run-durable-cap").exists()
    assert not (contract.root / "cache/runs/run-durable-cap").exists()


def test_reserved_run_intent_recovers_partial_mkdir_without_respawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _run_fixture_contract(tmp_path)
    arguments = {
        "run_id": "reserved-crash",
        "cwd_relative": "source/run-fixture",
        "argv": ["/usr/bin/git", "--version"],
        "environment": {},
        "python_path_relatives": [],
        "timeout_seconds": 20,
        "output_relative": "reports/run-reserved-crash",
        "_test_max_run_bytes": 1 << 20,
    }
    request = _run_request(arguments, "req-run-reserved-crash")
    original_transition = remote_worker._transition_run_state

    def fail_prepared(contract_value, state_path, request_value, expected, **changes):
        if changes.get("phase") == "prepared":
            raise OSError("injected post-mkdir crash")
        return original_transition(
            contract_value, state_path, request_value, expected, **changes
        )

    monkeypatch.setattr(remote_worker, "_transition_run_state", fail_prepared)
    with pytest.raises(OSError, match="post-mkdir crash"):
        remote_worker._run_command(contract, arguments, request)
    state_path = remote_worker._run_state_path(contract, request["request_id"])
    state = remote_worker._load_run_state(contract, state_path, request)
    assert state["phase"] == "reserved"
    assert (contract.root / "tmp/run-reserved-crash").exists()
    monkeypatch.setattr(remote_worker, "_transition_run_state", original_transition)

    recovered = remote_worker._run_command(contract, arguments, request)
    assert recovered["exit_code"] == 126
    assert recovered["supervisor_interrupted"] is True
    assert not (contract.root / "tmp/run-reserved-crash").exists()
    assert not (contract.root / "build/run-reserved-crash").exists()
    assert not (contract.root / "cache/runs/run-reserved-crash").exists()
    assert not remote_worker._data_reservation_path(
        contract, request["request_id"]
    ).exists()


def test_completed_run_recovers_half_moved_output_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _run_fixture_contract(tmp_path)
    arguments = {
        "run_id": "half-output",
        "cwd_relative": "source/run-fixture",
        "argv": ["/usr/bin/git", "--version"],
        "environment": {},
        "python_path_relatives": [],
        "timeout_seconds": 20,
        "output_relative": "reports/run-half-output",
        "_test_max_run_bytes": 1 << 20,
    }
    request = _run_request(arguments, "req-run-half-output")
    original_rename = remote_worker.os.rename
    injected = False

    def fail_second_log(source, destination):
        nonlocal injected
        if (
            not injected
            and Path(source).name == "stderr.log"
            and str(destination).endswith("run-output-staging/stderr.log")
        ):
            injected = True
            raise OSError("injected half-output crash")
        return original_rename(source, destination)

    monkeypatch.setattr(remote_worker.os, "rename", fail_second_log)
    with pytest.raises(OSError, match="half-output crash"):
        remote_worker._run_command(contract, arguments, request)
    state_path = remote_worker._run_state_path(contract, request["request_id"])
    assert remote_worker._load_run_state(contract, state_path, request)["phase"] == "completed"
    staging = contract.root / (
        "reports/.run-half-output.{}.run-output-staging".format(request["request_id"])
    )
    assert (staging / "stdout.log").exists()
    assert not (staging / "stderr.log").exists()

    monkeypatch.setattr(remote_worker.os, "rename", original_rename)
    recovered = remote_worker._run_command(contract, arguments, request)
    assert recovered["exit_code"] == 0
    assert (contract.root / "reports/run-half-output/stdout.log").exists()
    assert (contract.root / "reports/run-half-output/stderr.log").exists()


def test_finalized_run_retry_finishes_reservation_and_config_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _run_fixture_contract(tmp_path)
    arguments = {
        "run_id": "finalized-cleanup",
        "cwd_relative": "source/run-fixture",
        "argv": ["/usr/bin/git", "--version"],
        "environment": {},
        "python_path_relatives": [],
        "timeout_seconds": 20,
        "output_relative": "reports/run-finalized-cleanup",
        "_test_max_run_bytes": 1 << 20,
    }
    request = _run_request(arguments, "req-run-finalized-cleanup")
    original_release = remote_worker._release_matching_data_reservation
    injected = False

    def fail_release(*args, **kwargs):
        nonlocal injected
        if not injected:
            injected = True
            raise OSError("injected post-finalized crash")
        return original_release(*args, **kwargs)

    monkeypatch.setattr(
        remote_worker, "_release_matching_data_reservation", fail_release
    )
    with pytest.raises(OSError, match="post-finalized crash"):
        remote_worker._run_command(contract, arguments, request)
    state_path = remote_worker._run_state_path(contract, request["request_id"])
    assert remote_worker._load_run_state(contract, state_path, request)["phase"] == "finalized"
    assert remote_worker._data_reservation_path(
        contract, request["request_id"]
    ).exists()
    assert remote_worker._run_supervisor_path(
        contract, request["request_id"]
    ).exists()

    recovered = remote_worker._run_command(contract, arguments, request)
    assert recovered["exit_code"] == 0
    assert not remote_worker._data_reservation_path(
        contract, request["request_id"]
    ).exists()
    assert not remote_worker._run_supervisor_path(
        contract, request["request_id"]
    ).exists()


def test_worker_death_leaves_supervisor_to_finish_and_retry_adopts_result(
    tmp_path: Path,
) -> None:
    contract = _run_fixture_contract(tmp_path)
    sleeper = contract.root / "source/run-fixture/sleep.py"
    sleeper.write_text(
        "#!/usr/bin/python3\n"
        "import time\n"
        "print('supervisor-start', flush=True)\n"
        "time.sleep(2)\n"
        "print('supervisor-finish', flush=True)\n"
    )
    sleeper.chmod(0o500)
    arguments = {
        "run_id": "worker-death",
        "cwd_relative": "source/run-fixture",
        "argv": [str(sleeper)],
        "environment": {},
        "python_path_relatives": [],
        "timeout_seconds": 20,
        "output_relative": "reports/run-worker-death",
        "_test_max_run_bytes": 1 << 20,
    }
    request = _run_request(arguments, "req-run-worker-death")

    context = multiprocessing.get_context("fork")
    worker_process = context.Process(
        target=remote_worker._run_command,
        args=(contract, arguments, request),
    )
    worker_process.start()
    state_path = remote_worker._run_state_path(contract, request["request_id"])
    deadline = time.monotonic() + 10
    observed = None
    while time.monotonic() < deadline:
        try:
            observed = remote_worker._load_run_state(contract, state_path, request)
        except FileNotFoundError:
            time.sleep(0.05)
            continue
        if observed["phase"] == "running":
            break
        time.sleep(0.05)
    assert observed is not None and observed["phase"] == "running"
    assert worker_process.pid is not None
    os.kill(worker_process.pid, signal.SIGKILL)
    worker_process.join(timeout=5)
    assert not worker_process.is_alive()

    recovered = remote_worker._run_command(contract, arguments, request)
    assert recovered["exit_code"] == 0
    assert recovered["supervisor_interrupted"] is False
    output = contract.root / "reports/run-worker-death/stdout.log"
    assert output.read_bytes() == b"supervisor-start\nsupervisor-finish\n"
    final_state = remote_worker._load_run_state(contract, state_path, request)
    assert final_state["phase"] == "finalized"
    assert remote_worker._process_identity_sha256(
        int(final_state["supervisor_pid"])
    ) is None


def test_leader_gone_marked_process_group_is_still_reaped() -> None:
    marker = "req-leader-gone-group"
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import os,time; child=os.fork(); "
                "time.sleep(60) if child==0 else time.sleep(0.5)"
            ),
        ],
        env={**os.environ, "CASCADIA_R2_RUN_MARKER": marker},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    identity = remote_worker._process_identity_sha256(process.pid)
    assert identity is not None
    process.wait(timeout=5)
    assert remote_worker._process_identity_sha256(process.pid) is None
    assert remote_worker._process_group_exists(process.pid)
    try:
        assert remote_worker._terminate_exact_process_group(
            process.pid,
            identity,
            "leader-gone fixture",
            marker=marker,
        ) is True
        assert not remote_worker._process_group_exists(process.pid)
    finally:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)


def test_ephemeral_run_window_cleanup_is_two_output_token_bound_and_cas_atomic(
    tmp_path: Path,
) -> None:
    contract = _test_contract(tmp_path)
    run_id = "window-001"
    build = contract.root / f"build/run-{run_id}"
    cache = contract.root / f"cache/runs/run-{run_id}"
    build.mkdir(mode=0o700, parents=True)
    cache.mkdir(mode=0o700, parents=True)
    manifest_path = build / "window.json"
    dataset_path = build / "window.r2map"
    manifest_path.write_bytes(b'{"rows":2}\n')
    dataset_path.write_bytes(b"R2MAP\x00payload")
    (cache / "scratch.bin").write_bytes(b"temporary cache")
    manifest_token = remote_worker._object_token(contract, f"build/run-{run_id}/window.json")
    dataset_token = remote_worker._object_token(contract, f"build/run-{run_id}/window.r2map")
    prepared = remote_worker._prepare_run_cleanup(
        contract,
        {
            "run_id": run_id,
            "manifest_object_token": manifest_token,
            "dataset_object_token": dataset_token,
        },
    )
    cleanup_token = prepared["cleanup_token"]
    assert cleanup_token["cleanup_token_sha256"] == remote_worker.document_sha256(
        cleanup_token, "cleanup_token_sha256"
    )
    dataset_path.write_bytes(b"R2MAP\x00changed")
    with pytest.raises(remote_worker.RemoteWorkerError, match="changed after token"):
        remote_worker._commit_run_cleanup(
            contract,
            {"cleanup_token": cleanup_token},
        )
    assert build.exists() and cache.exists()

    dataset_token = remote_worker._object_token(contract, f"build/run-{run_id}/window.r2map")
    cleanup_token = remote_worker._prepare_run_cleanup(
        contract,
        {
            "run_id": run_id,
            "manifest_object_token": manifest_token,
            "dataset_object_token": dataset_token,
        },
    )["cleanup_token"]
    committed = remote_worker._commit_run_cleanup(
        contract,
        {"cleanup_token": cleanup_token},
    )
    assert committed["build_removed"] is True
    assert committed["cache_removed"] is True
    assert committed["cleanup_token_sha256"] == cleanup_token["cleanup_token_sha256"]
    assert not build.exists() and not cache.exists()
    recovered = remote_worker._commit_run_cleanup(
        contract,
        {"cleanup_token": cleanup_token},
    )
    assert recovered["build_already_removed"] is True
    assert recovered["cache_already_removed"] is True


def test_failed_run_cleanup_is_nofollow_three_tree_cas_and_idempotent(
    tmp_path: Path,
) -> None:
    contract = _test_contract(tmp_path)
    run_id = "failed-001"
    trees = {
        "tmp": contract.root / f"tmp/run-{run_id}",
        "build": contract.root / f"build/run-{run_id}",
        "cache": contract.root / f"cache/runs/run-{run_id}",
    }
    for path in trees.values():
        path.mkdir(mode=0o700, parents=True)
    (trees["tmp"] / "stdout.log").write_bytes(b"partial output")
    (trees["build"] / "partial.bin").write_bytes(b"partial build")
    outside = tmp_path / "outside-interpreter"
    outside.write_bytes(b"must survive")
    (trees["cache"] / "python").symlink_to(outside)

    token = remote_worker._prepare_failed_run_cleanup(contract, {"run_id": run_id})["cleanup_token"]
    assert token["schema_id"] == remote_worker.FAILED_RUN_CLEANUP_TOKEN_SCHEMA
    assert token["cleanup_token_sha256"] == remote_worker.document_sha256(
        token, "cleanup_token_sha256"
    )
    assert token["trees"]["cache"]["entry_count"] == 1

    (trees["build"] / "changed.bin").write_bytes(b"changed after prepare")
    with pytest.raises(remote_worker.RemoteWorkerError, match="CAS inventory changed"):
        remote_worker._commit_failed_run_cleanup(contract, {"cleanup_token": token})
    assert all(path.exists() for path in trees.values())

    token = remote_worker._prepare_failed_run_cleanup(contract, {"run_id": run_id})["cleanup_token"]
    committed = remote_worker._commit_failed_run_cleanup(contract, {"cleanup_token": token})
    assert committed["all_absent"] is True
    assert committed["removed"] == {"tmp": True, "build": True, "cache": True}
    assert outside.read_bytes() == b"must survive"
    assert not any(path.exists() for path in trees.values())

    repeated = remote_worker._commit_failed_run_cleanup(contract, {"cleanup_token": token})
    assert repeated["all_absent"] is True
    assert repeated["removed"] == {"tmp": False, "build": False, "cache": False}


def test_failed_run_cleanup_resumes_from_partially_deleted_atomic_tombstone(
    tmp_path: Path,
) -> None:
    contract = _test_contract(tmp_path)
    run_id = "failed-tombstone"
    trees = {
        "tmp": contract.root / f"tmp/run-{run_id}",
        "build": contract.root / f"build/run-{run_id}",
        "cache": contract.root / f"cache/runs/run-{run_id}",
    }
    for path in trees.values():
        path.mkdir(mode=0o700, parents=True)
        (path / "first.bin").write_bytes(b"first")
        (path / "second.bin").write_bytes(b"second")
    token = remote_worker._prepare_failed_run_cleanup(
        contract, {"run_id": run_id}
    )["cleanup_token"]
    tombstone = remote_worker._run_cleanup_tombstone(
        trees["build"], token["cleanup_token_sha256"]
    )
    os.rename(trees["build"], tombstone)
    (tombstone / "first.bin").unlink()

    recovered = remote_worker._commit_failed_run_cleanup(
        contract, {"cleanup_token": token}
    )
    assert recovered["all_absent"] is True
    assert not tombstone.exists()
    assert not any(path.exists() for path in trees.values())


def test_controller_run_is_bound_to_frozen_source_and_narrow_subcommands(tmp_path: Path) -> None:
    contract = _test_contract(tmp_path)
    source_root = contract.root / "source/controller-freeze"
    tool = source_root / "tools/r2_map_expert_iteration.py"
    python_root = source_root / "python"
    tool.parent.mkdir(mode=0o700, parents=True)
    python_root.mkdir(mode=0o700)
    tool.write_bytes(b"#!/usr/bin/env python3\n")
    tool.chmod(0o500)
    module = python_root / "module.py"
    module.write_bytes(b"VALUE = 1\n")
    module.chmod(0o400)
    manifest = _manifest(
        "controller-freeze",
        "source/controller-freeze",
        [
            {
                "relative": "python/module.py",
                "size": module.stat().st_size,
                "sha256": content_sha256(module.read_bytes()),
            },
            {
                "relative": "tools/r2_map_expert_iteration.py",
                "size": tool.stat().st_size,
                "sha256": content_sha256(tool.read_bytes()),
                "mode": "0500",
            },
        ],
    )
    (source_root / ".r2-map-transaction.json").write_bytes(canonical_json(manifest))
    validated = remote_worker._validate_controller_source(
        contract,
        tool,
        source_root,
        {"source_manifest_sha256": manifest["manifest_sha256"]},
        [str(tool), "show-state"],
    )
    assert validated == source_root
    with pytest.raises(remote_worker.RemoteWorkerError, match="subcommand"):
        remote_worker._validate_controller_source(
            contract,
            tool,
            source_root,
            {"source_manifest_sha256": manifest["manifest_sha256"]},
            [str(tool), "init"],
        )


def test_read_and_lock_client_apis_expose_authenticated_storage_receipts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "read_bytes", lambda _path: b"worker")
    client = RemoteStorageClient()
    token = {
        "schema_version": 1,
        "schema_id": remote_worker.OBJECT_TOKEN_SCHEMA,
        "relative": "build/run-window-001/window.r2map",
        "sha256": "11" * 32,
        "size": 3,
        "device": 1,
        "inode": 2,
        "mtime_ns": 3,
        "ctime_ns": 4,
        "mode": 0o400,
    }
    token["token_sha256"] = document_sha256(token, "token_sha256")
    receipt_sha256 = "ab" * 32

    def fake_execute(operation, _arguments, *_args, **_kwargs):
        if operation == "open-object":
            result = {"object_token": token}
            payload = b""
        elif operation == "read-range":
            payload = b"abc"
            result = {
                "payload_sha256": content_sha256(payload),
                "object_token_sha256": token["token_sha256"],
                "offset": 0,
                "length": 3,
            }
        else:
            payload = b""
            result = {"operation": operation}
        return RemoteResult(
            payload=payload,
            receipt={
                "request_id": f"req-{operation}",
                "receipt_sha256": receipt_sha256,
                "result": result,
            },
            input_sha256=content_sha256(b""),
            input_size=0,
        )

    monkeypatch.setattr(client, "execute", fake_execute)
    opened = client.open_object_with_receipt(token["relative"])
    assert opened["storage_receipt_sha256"] == receipt_sha256
    read = client.read_range_with_receipt(token, 0, 3)
    assert read["payload"] == b"abc"
    assert read["storage_receipt_sha256"] == receipt_sha256
    acquired = client.acquire_lock("controller", "root", lease_epoch="acquire-root-1")
    renewed = client.renew_lock(
        "controller", "root", "token", lease_epoch="renew-root-1"
    )
    released = client.release_lock(
        "controller", "root", "token", lease_epoch="release-root-1"
    )
    assert all(
        value["storage_receipt_sha256"] == receipt_sha256 for value in (acquired, renewed, released)
    )


def _framed_response(
    client: RemoteStorageClient,
    request: dict,
    payload: bytes,
    result: dict,
    *,
    status: str = "ok",
) -> bytes:
    complete_result = {
        **result,
        "payload_size": len(payload),
        "payload_sha256": content_sha256(payload),
    }
    receipt = {
        "schema_version": 1,
        "schema_id": remote_worker.RECEIPT_SCHEMA,
        "request_id": request["request_id"],
        "semantic_sha256": request["semantic_sha256"],
        "command_sha256": request["command_sha256"],
        "operation": request["operation"],
        "status": status,
        "host": REMOTE_HOST_ALIAS,
        "host_identity_sha256": REMOTE_IDENTITY_SHA256,
        "root": str(REMOTE_ROOT),
        "completed_unix_ms": 1,
        "result": complete_result,
    }
    receipt["receipt_sha256"] = document_sha256(receipt, "receipt_sha256")
    header = {
        "schema_id": "cascadia.r2-map.remote-frame.v1",
        "status": status,
        "response_disposition": "committed",
        "payload_size": len(payload),
        "payload_sha256": content_sha256(payload),
        "receipt_size": len(canonical_json(receipt)),
    }
    return remote_worker.encode_frame(header, payload, receipt)


def test_frame_is_bound_to_request_payload_host_and_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "read_bytes", lambda _path: b"worker")
    client = RemoteStorageClient()
    request = client._request("preflight", {})
    encoded = _framed_response(client, request, b"payload", {"proof": True})
    result = client._decode_response(
        encoded,
        request,
        input_sha256=content_sha256(b""),
        input_size=0,
    )
    assert result.payload == b"payload"
    tampered = bytearray(encoded)
    header_size = remote_worker.FRAME_PREFIX.unpack_from(encoded)[1]
    tampered[remote_worker.FRAME_PREFIX.size + header_size] ^= 1
    with pytest.raises(RemoteProtocolError):
        client._decode_response(
            bytes(tampered),
            request,
            input_sha256=content_sha256(b""),
            input_size=0,
        )


def test_request_is_canonical_hash_bound_and_worker_address_is_content_addressed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "read_bytes", lambda _path: b"worker bytes")
    client = RemoteStorageClient()
    request = client._request("open-object", {"relative": "reports/result.json"})
    assert request["command_sha256"] == remote_worker.request_command_sha256(request)
    assert client.worker_sha256 == hashlib.sha256(b"worker bytes").hexdigest()
    assert client.worker_sha256 in str(client.worker_remote_path)
    encoded = base64.urlsafe_b64encode(canonical_json(request)).rstrip(b"=")
    assert json.loads(base64.urlsafe_b64decode(encoded + b"=" * (-len(encoded) % 4))) == request


def test_unknown_stream_requires_independent_local_remote_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "read_bytes", lambda _path: b"worker")
    client = RemoteStorageClient()
    payload = b"abc"
    receipt = {
        "result": {"sha256": content_sha256(b"different"), "size": len(payload)},
    }
    response = RemoteResult(
        payload=b"",
        receipt=receipt,
        input_sha256=content_sha256(payload),
        input_size=len(payload),
    )
    monkeypatch.setattr(client, "execute", lambda *_args, **_kwargs: response)
    with pytest.raises(RemoteProtocolError, match="differs"):
        client.put_unknown_stream("logs/headless/turn.log", [payload], max_bytes=100)
