from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import os
import tarfile
from pathlib import Path

import pytest
import r2_d0.runtime as runtime_module
from r2_d0.canonical import D0Error, _john2_artifact_pending, render_document
from r2_d0.inventory import selected_runtime_paths
from r2_d0.runtime import (
    DOCKER,
    DOCKER_ACCOUNTING_COMMAND_MAX_BYTES,
    EGRESS_CONTROL_DOCUMENT_TIMEOUT_SECONDS,
    EGRESS_CONTROL_PROCESS_CLEANUP_SCRIPT,
    EGRESS_CONTROL_RECEIVE_TIMEOUT_SECONDS,
    EGRESS_CONTROL_SOCKET_QUIESCENCE_SCRIPT,
    EXPECTED_ENGINE_VERSION,
    NFT_FAILSAFE_CANCEL_SCRIPT,
    NFT_FAILSAFE_CLEANUP_SCRIPT,
    NFT_FAILSAFE_LAUNCH_SCRIPT,
    NFT_GUARD_INSTALL_SCRIPT,
    NFT_SCHEMA_INVENTORY_MAX_BYTES,
    SCANNER_ATTESTATION_CLEANUP_SCRIPT,
    SCANNER_EXPORT_SHA256,
    SCANNER_EXPORT_SIZE,
    SCANNER_LOCAL_REFERENCE,
    SCANNER_REGISTRY_CA_CERT,
    SCANNER_REGISTRY_CA_PATH,
    SCANNER_REGISTRY_CA_SHA256,
    SCANNER_REGISTRY_CLEANUP_SCRIPT,
    SCANNER_REGISTRY_HOST,
    SCANNER_REGISTRY_PORT,
    SCANNER_REGISTRY_PREPARE_SCRIPT,
    SCANNER_REGISTRY_PROCESS_MARKER,
    SCANNER_REGISTRY_REPOSITORY,
    SCANNER_REGISTRY_RESOLVER_HOST,
    SCANNER_REGISTRY_SERVER_CERT,
    SCANNER_REGISTRY_SERVER_CERT_DER_SHA256,
    SCANNER_REGISTRY_SERVER_CERT_SHA256,
    SCANNER_REGISTRY_SERVER_KEY,
    SCANNER_REGISTRY_SERVER_KEY_SHA256,
    SCANNER_REGISTRY_SERVER_SCRIPT,
    SCANNER_REGISTRY_TLS_CLIENT_SCRIPT,
    SCANNER_REGISTRY_TRUST_CLEANUP_SCRIPT,
    SCANNER_REGISTRY_TRUST_INSTALL_SCRIPT,
    SCANNER_SOCKET_SAMPLER_LAUNCH_SCRIPT,
    SCANNER_SOCKET_SAMPLER_STOP_SCRIPT,
    CommandRunner,
    Completed,
    _bounded_completed_evidence,
    _bounded_raw_nft_inventory,
    _bounded_state_differences,
    _buildkit_egress_program,
    _classify_egress_trace_delta,
    _colima_status,
    _converged_positive_runtime_activity,
    _docker_context_snapshot,
    _egress_socket_observations,
    _egress_socket_state_is_packet_capable,
    _egress_trace_delta,
    _egress_trace_projection,
    _egress_trace_tuple_fields,
    _flatten_probe_oci,
    _guest_listener_allowlist,
    _guest_network_projection,
    _nft_install_argv,
    _probe_oci_attestation_inventory,
    _probe_oci_graph,
    _probe_oci_inventory,
    _reject_counter_delta,
    _require_empty_daemon_accounting,
    _require_exact_logical_image_references,
    _require_reject_counter_transition,
    _require_single_logical_image,
    _scanner_attestation_residue_identity,
    _scanner_local_generator_reference,
    _scanner_registry_descriptor,
    _scanner_resolver_context,
    _trace_counter_comparison,
    _validate_buildkit_egress_table,
    _validate_default_private_pid_mode,
    _validate_failsafe_launch_receipt,
    _validate_guest_binfmt_inventory,
    _validate_guest_nested_virtualization,
    _validate_guest_package_license_inventory,
    _validate_inactive_runtime_activity,
    _validate_network_lease_transition,
    _validate_positive_runtime_activity,
    _validate_probe_oci_attachment_contract,
    _validate_probe_spdx_predicate,
    _validate_scanner_registry_cleanup,
    _validate_smoke_volume_inspects,
    _wait_preexisting_egress_socket_absent,
    global_dependency_environment,
    hardened_flags,
    homebrew_plan,
    rollback_plan,
    runtime_environment,
    validate_explicit_runtime_environment,
    verify_buildx_inspect,
    verify_daemon_config,
    verify_engine_info,
    verify_probe_oci,
)
from r2_d0_campaign_plan import ordered_transactions
from r2_d0_test_support import work_spec


def _packet(host: str, phase: str) -> dict[str, object]:
    return json.loads(render_document(work_spec(host, phase), kind="work"))


def test_absent_baseline_contains_only_campaign_owned_mutable_state() -> None:
    home = Path("/Users/john2")
    assert selected_runtime_paths(home) == [
        home / ".local/share/cascadia-r2/colima",
        home / "Library/Caches/cascadia-r2/colima",
        home / ".config/cascadia-r2/docker",
    ]
    assert all(not str(path).startswith("/opt/homebrew") for path in selected_runtime_paths(home))


def test_global_dependency_probe_environment_cannot_create_campaign_state() -> None:
    environment = global_dependency_environment("john2")
    assert environment["HOME"] == "/Users/john2"
    assert not any(key.startswith("COLIMA") or key == "DOCKER_CONFIG" for key in environment)
    assert not any(
        key in environment for key in ("HOMEBREW_CACHE", "HOMEBREW_LOGS", "HOMEBREW_TEMP")
    )


def test_recovery_environment_requires_exact_explicit_isolated_homes() -> None:
    packet = _packet("john2", "verify")
    environment = runtime_environment(packet)
    assert validate_explicit_runtime_environment(packet, environment) == environment
    assert environment["HOME"] == "/Users/john2"
    assert environment["COLIMA_HOME"] == "/Users/john2/.local/share/cascadia-r2/colima"
    assert environment["DOCKER_CONFIG"] == "/Users/john2/.config/cascadia-r2/docker"
    assert environment["DOCKER_HOST"] == (
        "unix:///Users/john2/.local/share/cascadia-r2/colima/cascadia-r2/docker.sock"
    )


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        ("remove", "COLIMA_HOME"),
        ("remove", "DOCKER_CONFIG"),
        ("remove", "DOCKER_HOST"),
        ("replace_colima", "/Users/john2/.colima"),
        ("replace_docker_config", "/Users/john2/.docker"),
        ("replace_docker_host", "unix:///Users/john2/.colima/default/docker.sock"),
        ("extra", "CONTAINER_HOST"),
    ],
)
def test_recovery_environment_fails_closed_before_subprocess(mutation: str, value: str) -> None:
    packet = _packet("john2", "verify")
    environment = runtime_environment(packet)
    if mutation == "remove":
        environment.pop(value)
    elif mutation == "replace_colima":
        environment["COLIMA_HOME"] = value
    elif mutation == "replace_docker_config":
        environment["DOCKER_CONFIG"] = value
    elif mutation == "replace_docker_host":
        environment["DOCKER_HOST"] = value
    else:
        environment[value] = "unexpected"
    with pytest.raises(D0Error, match="explicit isolated runtime environment differs"):
        validate_explicit_runtime_environment(packet, environment)


def test_guest_egress_installer_never_reads_root_only_failsafe_state() -> None:
    assert "with open(state" not in NFT_GUARD_INSTALL_SCRIPT
    assert "capture_output=True,text=True" in NFT_GUARD_INSTALL_SCRIPT
    assert "failsafe=json.loads(launched.stdout)" in NFT_GUARD_INSTALL_SCRIPT
    assert "len(launched.stdout)>4096" in NFT_GUARD_INSTALL_SCRIPT
    assert "sys.stdout.write(json.dumps(value" in NFT_FAILSAFE_LAUNCH_SCRIPT


def test_failsafe_scripts_compile_and_preserve_root_only_state() -> None:
    for name, script in (
        ("guard", NFT_GUARD_INSTALL_SCRIPT),
        ("launch", NFT_FAILSAFE_LAUNCH_SCRIPT),
        ("cancel", NFT_FAILSAFE_CANCEL_SCRIPT),
        ("cleanup", NFT_FAILSAFE_CLEANUP_SCRIPT),
    ):
        compile(script, name, "exec")
    assert "os.O_EXCL|os.O_NOFOLLOW,0o600" in NFT_FAILSAFE_LAUNCH_SCRIPT
    assert "stat.S_IMODE(st.st_mode)==0o600" in NFT_FAILSAFE_CANCEL_SCRIPT
    assert "stat.S_IMODE(st.st_mode)==0o600" in NFT_FAILSAFE_CLEANUP_SCRIPT
    assert "os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600" in NFT_FAILSAFE_LAUNCH_SCRIPT
    assert "expected=bytes.fromhex('66697265640a')" in NFT_FAILSAFE_CLEANUP_SCRIPT
    assert "os.unlink(marker); marker_removed=True" in NFT_FAILSAFE_CLEANUP_SCRIPT


def test_failsafe_launcher_receipt_accepts_only_exact_privileged_stdout() -> None:
    receipt = {
        "deadline_unix_ms": 1,
        "pid": 123,
        "state_path": "/run/exact.json",
        "status": "armed",
        "table": "exact_table",
    }
    assert (
        _validate_failsafe_launch_receipt(
            receipt, table="exact_table", state_path="/run/exact.json"
        )
        == receipt
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("table", "wrong"),
        ("state_path", "/run/wrong.json"),
        ("status", "clean"),
        ("pid", True),
        ("pid", 1),
        ("deadline_unix_ms", True),
        ("deadline_unix_ms", 0),
        ("extra", "tampered"),
    ],
)
def test_failsafe_launcher_receipt_rejects_stdout_tamper(field: str, value: object) -> None:
    receipt: dict[str, object] = {
        "deadline_unix_ms": 1,
        "pid": 123,
        "state_path": "/run/exact.json",
        "status": "armed",
        "table": "exact_table",
    }
    receipt[field] = value
    with pytest.raises(D0Error, match="failsafe receipt"):
        _validate_failsafe_launch_receipt(
            receipt, table="exact_table", state_path="/run/exact.json"
        )


def test_install_plan_verifies_global_formulae_without_mutating_them() -> None:
    commands = homebrew_plan(_packet("john2", "install"))
    argv = [command.argv for command in commands]
    assert argv[:2] == [
        ("/opt/homebrew/bin/brew", "--version"),
        ("/opt/homebrew/bin/brew", "--repository"),
    ]
    assert argv[2:] == [
        ("/opt/homebrew/bin/brew", "list", "--versions", "lima"),
        ("/opt/homebrew/bin/brew", "list", "--versions", "colima"),
        ("/opt/homebrew/bin/brew", "list", "--versions", "docker"),
        ("/opt/homebrew/bin/brew", "list", "--versions", "docker-buildx"),
    ]
    assert not any("install" in command for command in argv)


def test_rollback_plan_deletes_only_the_exact_campaign_profile() -> None:
    argv = [command.argv for command in rollback_plan(_packet("john2", "rollback"))]
    assert argv == [
        ("/opt/homebrew/bin/colima", "stop", "--profile", "cascadia-r2"),
        (
            "/opt/homebrew/bin/colima",
            "delete",
            "--profile",
            "cascadia-r2",
            "--data",
            "--force",
        ),
    ]
    assert not any("uninstall" in command for command in argv)


def test_hardened_container_uses_docker_default_private_pid_namespace() -> None:
    flags = hardened_flags("contract-test")
    assert "--pid" not in flags
    assert _validate_default_private_pid_mode({"PidMode": ""}) == {
        "effective_pid_mode": "default-private",
        "host_pid_namespace_shared": False,
        "container_pid_namespace_shared": False,
        "status": "pass",
    }


BUILDX_INSPECT = b"""Name:   default
Driver: docker

Nodes:
Name:             default
Endpoint:         default
Status:           running
BuildKit version: v0.30.0
Platforms:        linux/arm64
"""


def test_buildx_inspect_accepts_exact_frozen_integrated_driver_identity() -> None:
    assert verify_buildx_inspect(BUILDX_INSPECT) == {
        "driver": "docker",
        "buildkit_version": "v0.30.0",
        "platforms": "linux/arm64",
        "output_sha256": ("f6998677e06d839b3bd6e89bf19226fd231f283f4e2b1b77deaefc99404b18b3"),
    }


@pytest.mark.parametrize(
    ("old", "new"),
    (
        (b"Driver: docker", b"Driver: docker-container"),
        (b"BuildKit version: v0.30.0", b"BuildKit version: v0.29.0"),
        (b"Platforms:        linux/arm64", b"Platforms:        linux/amd64"),
    ),
)
def test_buildx_inspect_rejects_wrong_driver_version_or_platform(
    old: bytes,
    new: bytes,
) -> None:
    with pytest.raises(D0Error, match="integrated-driver identity"):
        verify_buildx_inspect(BUILDX_INSPECT.replace(old, new))


@pytest.mark.parametrize(
    "duplicate",
    (
        b"Driver: docker\n",
        b"BuildKit version: v0.30.0\n",
        b"Platforms:        linux/arm64\n",
    ),
)
def test_buildx_inspect_rejects_duplicate_identity_labels(duplicate: bytes) -> None:
    with pytest.raises(D0Error, match="integrated-driver identity"):
        verify_buildx_inspect(BUILDX_INSPECT + duplicate)


def test_nft_install_transport_keeps_program_guest_local() -> None:
    program = b"add table inet cascadia_r2_d0\n"
    assert _nft_install_argv(program) == [
        "/bin/sh",
        "-c",
        'printf "%s" "$1" | /usr/bin/sudo -n /usr/sbin/nft -f -',
        "cascadia-r2-d0-nft",
        program.decode("ascii"),
    ]


@pytest.mark.parametrize(
    "program",
    (
        b"missing-newline",
        b"nul\0byte\n",
        b"x" * 4096 + b"\n",
        b"non-ascii-\xff\n",
    ),
)
def test_nft_install_transport_rejects_unbounded_or_ambiguous_program(
    program: bytes,
) -> None:
    with pytest.raises(D0Error, match="nftables program"):
        _nft_install_argv(program)


MANAGEMENT_FLOW = {
    "family": "ip",
    "client_address": "192.168.5.2",
    "client_port": 49360,
    "server_address": "192.168.5.15",
    "server_port": 22,
}


def test_egress_program_allows_only_established_sshd_replies() -> None:
    program = _buildkit_egress_program("cascadia_r2_d0", MANAGEMENT_FLOW).decode()
    assert "tcp sport 22 ct state established accept" in program
    assert "ct original" not in program
    assert "tcp dport 22" not in program
    assert "tcp dport 49360" not in program
    assert "tcp sport 49360" not in program


def test_diagnostic_egress_program_records_tuples_without_permitting_them() -> None:
    program = _buildkit_egress_program(
        "cascadia_r2_d0",
        MANAGEMENT_FLOW,
        trace=True,
    ).decode()
    for name in ("tcp4", "udp4", "other4", "tcp6", "udp6", "other6"):
        assert f"add set inet cascadia_r2_d0 {name}" in program
        assert f"update @{name}" in program
        assert f'comment "cascadia-d0-trace-{name}"' in program
    assert program.rstrip().endswith('comment "cascadia-d0-reject"')
    assert program.count(" accept ") == 2
    assert program.count(" reject ") == 1
    assert "meta skuid" not in program
    assert "uid . ifname" not in program


@pytest.mark.parametrize(
    "flow",
    (
        {},
        {**MANAGEMENT_FLOW, "family": "inet"},
        {**MANAGEMENT_FLOW, "client_port": 0},
        {**MANAGEMENT_FLOW, "client_port": True},
    ),
)
def test_egress_program_rejects_malformed_management_tuple(flow: dict[str, object]) -> None:
    with pytest.raises(D0Error, match="management flow"):
        _buildkit_egress_program("cascadia_r2_d0", flow)


def test_guest_guard_scripts_fail_closed_on_env_sessions_and_failsafe_state() -> None:
    assert "len(parts)!=4" in NFT_GUARD_INSTALL_SCRIPT
    assert "server_port!=22" in NFT_GUARD_INSTALL_SCRIPT
    assert "normalized!=[expected]" in NFT_GUARD_INSTALL_SCRIPT
    assert "os.O_EXCL|os.O_NOFOLLOW" in NFT_FAILSAFE_LAUNCH_SCRIPT
    assert "raise SystemExit(91)" in NFT_FAILSAFE_LAUNCH_SCRIPT
    assert "state+'.fired'" in NFT_FAILSAFE_CANCEL_SCRIPT
    assert "raise SystemExit(104)" in NFT_FAILSAFE_CANCEL_SCRIPT
    compile(EGRESS_CONTROL_SOCKET_QUIESCENCE_SCRIPT, "egress-quiescence", "exec")
    assert "['/usr/bin/ss','-Hntoa']" in EGRESS_CONTROL_SOCKET_QUIESCENCE_SCRIPT
    assert "'status':'timed-out'" in EGRESS_CONTROL_SOCKET_QUIESCENCE_SCRIPT
    assert "state=='TIME-WAIT'" in EGRESS_CONTROL_SOCKET_QUIESCENCE_SCRIPT
    assert "timer_fields[0]=='timewait'" in EGRESS_CONTROL_SOCKET_QUIESCENCE_SCRIPT
    assert "consecutive_absent>=3" in EGRESS_CONTROL_SOCKET_QUIESCENCE_SCRIPT
    assert "consecutive_absent+1 if not matches" in EGRESS_CONTROL_SOCKET_QUIESCENCE_SCRIPT
    compile(runtime_module._EGRESS_SERVER_SCRIPT, "egress-server", "exec")
    compile(runtime_module._EGRESS_CLIENT_SCRIPT, "egress-client", "exec")
    assert "SO_LINGER" in runtime_module._EGRESS_CLIENT_SCRIPT
    assert "struct.pack('ii',1,0)" in runtime_module._EGRESS_CLIENT_SCRIPT
    assert "outcome['abortive_close']=True" in runtime_module._EGRESS_CLIENT_SCRIPT
    assert "SO_LINGER" in runtime_module._EGRESS_SERVER_SCRIPT
    assert "struct.pack('ii',1,0)" in runtime_module._EGRESS_SERVER_SCRIPT
    assert "'abortive_close':abortive_close" in runtime_module._EGRESS_SERVER_SCRIPT
    compile(EGRESS_CONTROL_PROCESS_CLEANUP_SCRIPT, "egress-process-cleanup", "exec")
    assert "marker.encode() not in parts" in EGRESS_CONTROL_PROCESS_CLEANUP_SCRIPT
    assert "os.kill(pid,signal.SIGTERM)" in EGRESS_CONTROL_PROCESS_CLEANUP_SCRIPT
    assert "os.kill(pid,signal.SIGKILL)" in EGRESS_CONTROL_PROCESS_CLEANUP_SCRIPT
    assert "if os.path.isdir(proc): raise SystemExit(76)" in EGRESS_CONTROL_PROCESS_CLEANUP_SCRIPT


def test_egress_socket_observations_preserve_exact_state_timer_and_identity() -> None:
    descriptor = {
        "host_address": "198.18.134.1",
        "peer_address": "198.18.134.2",
        "port": 43022,
    }
    line = (
        "CLOSING 1 2 198.18.134.1:49670 198.18.134.2:43022 "
        "timer:(persist,237ms,1) ino:0 sk:5007 "
        "cgroup:/user.slice/user-501.slice/session-4.scope ---"
    )
    unrelated = "ESTAB 0 0 127.0.0.1:1 127.0.0.1:2 uid:501 ino:22 sk:1"
    assert _egress_socket_observations(f"{line}\n{unrelated}\n".encode(), descriptor) == [
        {
            "cgroup": "/user.slice/user-501.slice/session-4.scope",
            "inode": 0,
            "line": line,
            "packet_capable": True,
            "process_users_available": False,
            "socket": "5007",
            "state": "CLOSING",
            "timer": ["persist", "237ms", "1"],
            "uid": None,
        }
    ]


@pytest.mark.parametrize(
    ("state", "timer", "expected"),
    (
        ("CLOSING", ["persist", "237ms", "1"], True),
        ("ESTAB", None, True),
        ("FIN-WAIT-1", ["on", "1sec", "2"], True),
        ("TIME-WAIT", ["timewait", "59sec", "0"], False),
        ("TIME-WAIT", None, False),
        ("TIME-WAIT", ["persist", "1sec", "1"], True),
    ),
)
def test_egress_socket_terminal_state_semantics(
    state: str,
    timer: list[str] | None,
    expected: bool,
) -> None:
    assert _egress_socket_state_is_packet_capable(state, timer) is expected


def test_egress_quiescence_rejects_persistent_packet_capable_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = {
        "line": "CLOSING 198.18.134.1:49670 198.18.134.2:43022 timer:(persist,1sec,1)",
        "packet_capable": True,
        "state": "CLOSING",
        "timer": ["persist", "1sec", "1"],
    }
    payload = json.dumps(
        {
            "last_matches": [row],
            "last_output_sha256": "0" * 64,
            "packet_capable_matches": [row],
            "samples": 100,
            "status": "timed-out",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    monkeypatch.setattr(
        runtime_module,
        "_guest",
        lambda *_args, **_kwargs: Completed((), 0, payload, b""),
    )
    with pytest.raises(D0Error, match="did not quiesce"):
        _wait_preexisting_egress_socket_absent(
            object(),
            {
                "host_address": "198.18.134.1",
                "peer_address": "198.18.134.2",
                "port": 43022,
            },
        )


def test_probe_oci_inventory_preserves_image_only_export_without_accepting_it() -> None:
    config = b"{}"
    config_digest = hashlib.sha256(config).hexdigest()
    manifest = json.dumps(
        {
            "config": {
                "digest": f"sha256:{config_digest}",
                "mediaType": "application/vnd.oci.image.config.v1+json",
                "size": len(config),
            },
            "layers": [],
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "schemaVersion": 2,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    manifest_digest = hashlib.sha256(manifest).hexdigest()
    index = json.dumps(
        {
            "manifests": [
                {
                    "digest": f"sha256:{manifest_digest}",
                    "mediaType": "application/vnd.oci.image.manifest.v1+json",
                    "size": len(manifest),
                }
            ],
            "schemaVersion": 2,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    layout = b'{"imageLayoutVersion":"1.0.0"}'
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name, payload in (
            ("oci-layout", layout),
            ("index.json", index),
            (f"blobs/sha256/{manifest_digest}", manifest),
            (f"blobs/sha256/{config_digest}", config),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o444
            info.mtime = 0
            archive.addfile(info, io.BytesIO(payload))
    inventory = _probe_oci_inventory(output.getvalue())
    assert inventory["attestation_descriptor_count"] == 0
    assert inventory["unreferenced_members"] == []
    assert inventory["manifest_descriptors"][0]["digest_matches"] is True
    assert inventory["manifest_descriptors"][0]["size_matches"] is True
    assert inventory["manifest_descriptors"][0]["references"][0]["digest_matches"] is True


def _probe_inventory_archive(
    manifest_value: dict[str, object],
    *,
    annotations: dict[str, str] | None = None,
) -> bytes:
    manifest = json.dumps(
        manifest_value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    manifest_digest = hashlib.sha256(manifest).hexdigest()
    descriptor: dict[str, object] = {
        "digest": f"sha256:{manifest_digest}",
        "mediaType": "application/vnd.oci.image.manifest.v1+json",
        "size": len(manifest),
    }
    if annotations is not None:
        descriptor["annotations"] = annotations
    index = json.dumps(
        {"manifests": [descriptor], "schemaVersion": 2},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name, payload in (
            ("oci-layout", b'{"imageLayoutVersion":"1.0.0"}'),
            ("index.json", index),
            (f"blobs/sha256/{manifest_digest}", manifest),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o444
            info.mtime = 0
            archive.addfile(info, io.BytesIO(payload))
    return output.getvalue()


def _canonical_test_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _test_tar(members: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name, payload in sorted(members.items()):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o444
            info.uid = 0
            info.gid = 0
            info.mtime = 0
            archive.addfile(info, io.BytesIO(payload))
    return output.getvalue()


def _oci_descriptor(payload: bytes, media_type: str) -> dict[str, object]:
    return {
        "digest": f"sha256:{hashlib.sha256(payload).hexdigest()}",
        "mediaType": media_type,
        "size": len(payload),
    }


def _nested_probe_oci_archive(
    *,
    attestation_reference: str | None = None,
    empty_subjects: bool = True,
    provenance_statement_override: bytes | None = None,
    spdx_statement_override: bytes | None = None,
    subject_digest: str | None = None,
) -> tuple[bytes, dict[str, object]]:
    layer = _test_tar({"probe.txt": runtime_module.PROBE_PAYLOAD})
    layer_descriptor = _oci_descriptor(layer, "application/vnd.oci.image.layer.v1.tar")
    image_config = _canonical_test_json(
        {
            "architecture": "arm64",
            "config": {
                "Labels": {"org.opencontainers.image.title": ("cascadia-r2-d0-buildkit-probe")}
            },
            "os": "linux",
            "rootfs": {
                "diff_ids": [f"sha256:{hashlib.sha256(layer).hexdigest()}"],
                "type": "layers",
            },
        }
    )
    image_config_descriptor = _oci_descriptor(
        image_config, "application/vnd.oci.image.config.v1+json"
    )
    image_manifest = _canonical_test_json(
        {
            "config": image_config_descriptor,
            "layers": [layer_descriptor],
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "schemaVersion": 2,
        }
    )
    image_descriptor = _oci_descriptor(image_manifest, "application/vnd.oci.image.manifest.v1+json")
    image_descriptor["platform"] = {"architecture": "arm64", "os": "linux"}
    image_digest = str(image_descriptor["digest"])
    image_hex = image_digest.split(":", 1)[1]
    bound_subject = subject_digest or image_hex
    subject = (
        []
        if empty_subjects and subject_digest is None
        else [{"digest": {"sha256": bound_subject}, "name": "probe"}]
    )
    spdx_statement = spdx_statement_override or _canonical_test_json(
        {
            "_type": "https://in-toto.io/Statement/v0.1",
            "predicate": {
                "SPDXID": "SPDXRef-DOCUMENT",
                "creationInfo": {
                    "created": "2026-06-19T11:40:41Z",
                    "creators": [
                        "Organization: Anchore, Inc",
                        "Tool: syft-v1.42.3",
                        "Tool: buildkit-v0.30.0",
                    ],
                    "licenseListVersion": "3.28",
                },
                "dataLicense": "CC0-1.0",
                "documentNamespace": (
                    "https://anchore.com/syft/dir/sbom-12345678-1234-4abc-8abc-123456789abc"
                ),
                "name": "sbom",
                "packages": [
                    {
                        "SPDXID": "SPDXRef-DocumentRoot-Directory-sbom",
                        "copyrightText": "NOASSERTION",
                        "downloadLocation": "NOASSERTION",
                        "filesAnalyzed": False,
                        "licenseConcluded": "NOASSERTION",
                        "licenseDeclared": "NOASSERTION",
                        "name": "sbom",
                        "primaryPackagePurpose": "FILE",
                        "supplier": "NOASSERTION",
                    }
                ],
                "relationships": [
                    {
                        "relatedSpdxElement": ("SPDXRef-DocumentRoot-Directory-sbom"),
                        "relationshipType": "DESCRIBES",
                        "spdxElementId": "SPDXRef-DOCUMENT",
                    }
                ],
                "spdxVersion": "SPDX-2.3",
            },
            "predicateType": "https://spdx.dev/Document",
            "subject": subject,
        }
    )
    session_uri = "http://buildkit-session/fixture123"
    provenance_statement = provenance_statement_override or _canonical_test_json(
        {
            "_type": "https://in-toto.io/Statement/v0.1",
            "predicate": {
                "buildDefinition": {
                    "buildType": (
                        "https://github.com/moby/buildkit/blob/master/docs/"
                        "attestations/slsa-definitions.md"
                    ),
                    "externalParameters": {
                        "configSource": {
                            "digest": {"sha256": runtime_module.PROBE_ARCHIVE_SHA256},
                            "path": "Dockerfile",
                            "uri": session_uri,
                        },
                        "request": {
                            "args": {
                                "force-network-mode": "none",
                                "no-cache": "",
                            },
                            "compatibilityVersion": 20,
                            "frontend": "dockerfile.v0",
                            "root": {
                                "configSource": {
                                    "digest": {"sha256": runtime_module.PROBE_ARCHIVE_SHA256},
                                    "path": "Dockerfile",
                                    "uri": session_uri,
                                },
                                "request": {
                                    "args": {
                                        "force-network-mode": "none",
                                        "no-cache": "",
                                    }
                                },
                            },
                        },
                    },
                    "internalParameters": {
                        "buildConfig": {
                            "digestMapping": {"sha256:" + "1" * 64: "step0"},
                            "llbDefinition": [{"id": "step0", "op": {"Op": {}}}],
                        },
                        "builderPlatform": "linux/arm64",
                        "dockerfileVersion": "1.24.0",
                    },
                    "resolvedDependencies": [
                        {
                            "digest": {
                                "sha256": runtime_module.SCANNER_IMAGE["manifest_digest"].split(
                                    ":", 1
                                )[1]
                            },
                            "uri": (
                                "pkg:docker/localhost%3A5047/cascadia/"
                                "buildkit-syft-scanner?digest="
                                + runtime_module.SCANNER_IMAGE["manifest_digest"]
                            ),
                        },
                        {
                            "digest": {"sha256": runtime_module.PROBE_ARCHIVE_SHA256},
                            "uri": session_uri,
                        },
                    ],
                },
                "runDetails": {
                    "builder": {"id": ""},
                    "metadata": {
                        "buildkit_completeness": {
                            "request": True,
                            "resolvedDependencies": True,
                        },
                        "buildkit_hermetic": True,
                        "buildkit_metadata": {
                            "layers": {"step0:0": [[layer_descriptor]]},
                            "source": {
                                "infos": [
                                    {
                                        "data": base64.b64encode(
                                            runtime_module.PROBE_DOCKERFILE
                                        ).decode("ascii"),
                                        "digestMapping": {"sha256:" + "1" * 64: "step0"},
                                        "filename": "Dockerfile",
                                        "language": "Dockerfile",
                                        "llbDefinition": [{"id": "step0", "op": {"Op": {}}}],
                                    }
                                ],
                                "locations": {"step0": {}},
                            },
                        },
                        "finishedOn": "2026-06-19T08:11:37.324411748-04:00",
                        "invocationId": "fixture123",
                        "startedOn": "2026-06-19T08:11:36.756935067-04:00",
                    },
                },
            },
            "predicateType": "https://slsa.dev/provenance/v1",
            "subject": subject,
        }
    )
    attestation_config = b"{}"
    attestation_layers: list[dict[str, object]] = []
    for predicate, statement in (
        ("https://spdx.dev/Document", spdx_statement),
        ("https://slsa.dev/provenance/v1", provenance_statement),
    ):
        descriptor = _oci_descriptor(statement, "application/vnd.in-toto+json")
        descriptor["annotations"] = {"in-toto.io/predicate-type": predicate}
        attestation_layers.append(descriptor)
    attestation_manifest = _canonical_test_json(
        {
            "config": _oci_descriptor(
                attestation_config,
                "application/vnd.oci.empty.v1+json",
            ),
            "layers": attestation_layers,
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "schemaVersion": 2,
        }
    )
    attestation_descriptor = _oci_descriptor(
        attestation_manifest, "application/vnd.oci.image.manifest.v1+json"
    )
    attestation_descriptor["annotations"] = {
        "vnd.docker.reference.digest": attestation_reference or image_digest,
        "vnd.docker.reference.type": "attestation-manifest",
    }
    attestation_descriptor["platform"] = {"architecture": "unknown", "os": "unknown"}
    nested_index = _canonical_test_json(
        {
            "manifests": [image_descriptor, attestation_descriptor],
            "mediaType": "application/vnd.oci.image.index.v1+json",
            "schemaVersion": 2,
        }
    )
    nested_descriptor = _oci_descriptor(nested_index, "application/vnd.oci.image.index.v1+json")
    root_index = _canonical_test_json(
        {
            "manifests": [nested_descriptor],
            "mediaType": "application/vnd.oci.image.index.v1+json",
            "schemaVersion": 2,
        }
    )
    blobs = {
        f"blobs/sha256/{hashlib.sha256(payload).hexdigest()}": payload
        for payload in (
            layer,
            image_config,
            image_manifest,
            attestation_config,
            spdx_statement,
            provenance_statement,
            attestation_manifest,
            nested_index,
        )
    }
    archive = _test_tar(
        {
            **blobs,
            "index.json": root_index,
            "oci-layout": b'{"imageLayoutVersion":"1.0.0"}',
        }
    )
    return archive, {
        "attestation_manifest_digest": attestation_descriptor["digest"],
        "image_manifest_digest": image_digest,
        "nested_index_digest": nested_descriptor["digest"],
    }


FIXTURE_BUILD_STARTED_UNIX_NS = 1_781_871_096_528_396_000
FIXTURE_BUILD_FINISHED_UNIX_NS = 1_781_871_097_348_178_000


def _verify_probe_archive(archive: bytes) -> dict[str, object]:
    return verify_probe_oci(
        archive,
        build_finished_unix_ns=FIXTURE_BUILD_FINISHED_UNIX_NS,
        build_started_unix_ns=FIXTURE_BUILD_STARTED_UNIX_NS,
    )


def test_verify_probe_oci_accepts_recursive_buildx_v32_shape() -> None:
    archive, identities = _nested_probe_oci_archive()
    verified = _verify_probe_archive(archive)
    assert verified["archive_sha256"] == hashlib.sha256(archive).hexdigest()
    assert verified["image_manifest_digest"] == identities["image_manifest_digest"]
    assert verified["recursive_attestation_descriptor_count"] == 1
    assert verified["recursive_graph"]["status"] == "pass"
    assert [node["kind"] for node in verified["recursive_graph"]["nodes"]] == [
        "index",
        "manifest",
        "manifest",
    ]
    assert set(verified["predicate_types"]) == {
        "https://slsa.dev/provenance/v1",
        "https://spdx.dev/Document",
    }
    assert verified["attestation_binding"]["binding"] == "docker-index-descriptor"
    assert (
        verified["attestation_binding"]["image_manifest_digest"]
        == identities["image_manifest_digest"]
    )
    spdx = next(
        item
        for item in verified["predicates"]
        if item["predicate_type"] == "https://spdx.dev/Document"
    )["predicate_validation"]
    assert spdx == {
        "content_binding": "spdx-document-root-package",
        "document_name": "sbom",
        "document_namespace": (
            "https://anchore.com/syft/dir/sbom-12345678-1234-4abc-8abc-123456789abc"
        ),
        "file_count": 0,
        "files_field_present": False,
        "package_count": 1,
        "probe_checksum_algorithms": [],
        "probe_file_count": 0,
        "relationship_count": 1,
        "spdx_version": "SPDX-2.3",
        "status": "pass",
    }
    slsa = next(
        item
        for item in verified["predicates"]
        if item["predicate_type"] == "https://slsa.dev/provenance/v1"
    )["predicate_validation"]
    assert slsa == {
        "build_type": (
            "https://github.com/moby/buildkit/blob/master/docs/attestations/slsa-definitions.md"
        ),
        "builder_id": "",
        "builder_platform": "linux/arm64",
        "context_sha256": runtime_module.PROBE_ARCHIVE_SHA256,
        "dockerfile_version": "1.24.0",
        "hermetic": True,
        "resolved_dependency_count": 2,
        "run_duration_unix_ns": 567_476_681,
        "run_finished_unix_ns": 1_781_871_097_324_411_748,
        "run_started_unix_ns": 1_781_871_096_756_935_067,
        "scanner_manifest_digest": runtime_module.SCANNER_IMAGE["manifest_digest"],
        "status": "pass",
        "timestamp_tolerance_unix_ns": 0,
    }
    inventory = _probe_oci_inventory(archive)
    assert inventory["attestation_descriptor_count"] == 1
    assert inventory["recursive_graph"]["status"] == "pass"
    assert inventory["unreferenced_members"] == []


def test_verify_probe_oci_accepts_matching_nonempty_statement_subjects() -> None:
    archive, _ = _nested_probe_oci_archive(empty_subjects=False)
    verified = _verify_probe_archive(archive)
    assert all(item["subject_count"] == 1 for item in verified["predicates"])


def test_flat_probe_oci_rejects_empty_subject_without_recursive_attachment() -> None:
    archive, _ = _nested_probe_oci_archive()
    flat, _ = _flatten_probe_oci(archive)
    with pytest.raises(D0Error, match="lacks exact Docker attachment binding"):
        runtime_module._verify_flat_probe_oci(
            flat,
            build_finished_unix_ns=FIXTURE_BUILD_FINISHED_UNIX_NS,
            build_started_unix_ns=FIXTURE_BUILD_STARTED_UNIX_NS,
        )


def _attachment_graph_fixture() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    archive, _ = _nested_probe_oci_archive()
    graph = copy.deepcopy(_probe_oci_graph(archive))
    manifests = [node for node in graph["nodes"] if node["kind"] == "manifest"]
    attestation = next(
        node
        for node in manifests
        if node["descriptor"].get("annotations", {}).get("vnd.docker.reference.type")
        == "attestation-manifest"
    )
    image = next(node for node in manifests if node is not attestation)
    return graph, image, attestation


def test_attachment_contract_rejects_wrong_attestation_platform() -> None:
    graph, _, attestation = _attachment_graph_fixture()
    attestation["descriptor"]["platform"] = {"architecture": "arm64", "os": "linux"}
    with pytest.raises(D0Error, match="annotations or platform differ"):
        _validate_probe_oci_attachment_contract(graph)


def test_attachment_contract_rejects_extra_descriptor_annotation() -> None:
    graph, _, attestation = _attachment_graph_fixture()
    attestation["descriptor"]["annotations"]["untrusted.example/key"] = "value"
    with pytest.raises(D0Error, match="annotations or platform differ"):
        _validate_probe_oci_attachment_contract(graph)


def test_attachment_contract_rejects_cross_index_image_reference() -> None:
    graph, image, _ = _attachment_graph_fixture()
    image_digest = image["descriptor"]["digest"]
    image_edge = next(edge for edge in graph["edges"] if edge["child_digest"] == image_digest)
    image_edge["parent_digest"] = None
    with pytest.raises(D0Error, match="cross-index or ambiguous"):
        _validate_probe_oci_attachment_contract(graph)


def test_attachment_contract_rejects_ambiguous_image_parent() -> None:
    graph, image, _ = _attachment_graph_fixture()
    image_digest = image["descriptor"]["digest"]
    image_edge = next(edge for edge in graph["edges"] if edge["child_digest"] == image_digest)
    graph["edges"].append(copy.deepcopy(image_edge))
    with pytest.raises(D0Error, match="cross-index or ambiguous"):
        _validate_probe_oci_attachment_contract(graph)


def test_attachment_contract_rejects_shared_statement_layer() -> None:
    graph, image, attestation = _attachment_graph_fixture()
    image["document"]["layers"].append(copy.deepcopy(attestation["document"]["layers"][0]))
    with pytest.raises(D0Error, match="not exclusively attached"):
        _validate_probe_oci_attachment_contract(graph)


def _live_provenance_statement() -> dict[str, object]:
    archive, _ = _nested_probe_oci_archive()
    members = runtime_module._safe_tar(archive)
    graph = _probe_oci_graph(archive)
    attestation = next(
        node
        for node in graph["nodes"]
        if node["kind"] == "manifest"
        and node["descriptor"].get("annotations", {}).get("vnd.docker.reference.type")
        == "attestation-manifest"
    )
    layer = next(
        item
        for item in attestation["document"]["layers"]
        if item["annotations"]["in-toto.io/predicate-type"] == "https://slsa.dev/provenance/v1"
    )
    return json.loads(members[f"blobs/sha256/{layer['digest'].split(':', 1)[1]}"])


def test_slsa_v1_rejects_nonempty_builder_identity() -> None:
    statement = _live_provenance_statement()
    statement["predicate"]["runDetails"]["builder"]["id"] = "untrusted-builder"
    archive, _ = _nested_probe_oci_archive(
        provenance_statement_override=_canonical_test_json(statement)
    )
    with pytest.raises(D0Error, match="maximal provenance identity differs"):
        _verify_probe_archive(archive)


def test_slsa_v1_rejects_nonhermetic_metadata() -> None:
    statement = _live_provenance_statement()
    statement["predicate"]["runDetails"]["metadata"]["buildkit_hermetic"] = False
    archive, _ = _nested_probe_oci_archive(
        provenance_statement_override=_canonical_test_json(statement)
    )
    with pytest.raises(D0Error, match="run metadata differs"):
        _verify_probe_archive(archive)


def test_slsa_v1_rejects_wrong_context_digest() -> None:
    statement = _live_provenance_statement()
    statement["predicate"]["buildDefinition"]["externalParameters"]["configSource"]["digest"][
        "sha256"
    ] = "0" * 64
    archive, _ = _nested_probe_oci_archive(
        provenance_statement_override=_canonical_test_json(statement)
    )
    with pytest.raises(D0Error, match="external parameters differ"):
        _verify_probe_archive(archive)


def test_slsa_v1_rejects_wrong_embedded_dockerfile() -> None:
    statement = _live_provenance_statement()
    statement["predicate"]["runDetails"]["metadata"]["buildkit_metadata"]["source"]["infos"][0][
        "data"
    ] = base64.b64encode(b"FROM untrusted\n").decode("ascii")
    archive, _ = _nested_probe_oci_archive(
        provenance_statement_override=_canonical_test_json(statement)
    )
    with pytest.raises(D0Error, match="BuildKit metadata differs"):
        _verify_probe_archive(archive)


def test_rfc3339_parser_preserves_nanoseconds_and_offsets() -> None:
    assert (
        runtime_module._parse_rfc3339_unix_ns("2026-06-19T08:11:36.756935067-04:00")
        == 1_781_871_096_756_935_067
    )
    assert (
        runtime_module._parse_rfc3339_unix_ns("2026-06-19T12:11:36.756935067Z")
        == 1_781_871_096_756_935_067
    )


@pytest.mark.parametrize(
    "value",
    (
        "2026-06-19 12:11:36Z",
        "2026-06-19T12:11:36.1234567890Z",
        "2026-02-30T12:11:36Z",
        "2026-06-19T12:11:36+24:00",
    ),
)
def test_rfc3339_parser_rejects_invalid_values(value: str) -> None:
    with pytest.raises(D0Error, match="timestamp"):
        runtime_module._parse_rfc3339_unix_ns(value)


def test_slsa_v1_rejects_reversed_run_timestamps() -> None:
    statement = _live_provenance_statement()
    metadata = statement["predicate"]["runDetails"]["metadata"]
    metadata["startedOn"], metadata["finishedOn"] = (
        metadata["finishedOn"],
        metadata["startedOn"],
    )
    archive, _ = _nested_probe_oci_archive(
        provenance_statement_override=_canonical_test_json(statement)
    )
    with pytest.raises(D0Error, match="escape the build envelope"):
        _verify_probe_archive(archive)


def test_slsa_v1_rejects_timestamp_before_build_envelope() -> None:
    statement = _live_provenance_statement()
    statement["predicate"]["runDetails"]["metadata"]["startedOn"] = (
        "2026-06-19T08:11:35.999999999-04:00"
    )
    archive, _ = _nested_probe_oci_archive(
        provenance_statement_override=_canonical_test_json(statement)
    )
    with pytest.raises(D0Error, match="escape the build envelope"):
        _verify_probe_archive(archive)


def test_slsa_v1_rejects_timestamp_after_build_envelope() -> None:
    statement = _live_provenance_statement()
    statement["predicate"]["runDetails"]["metadata"]["finishedOn"] = (
        "2026-06-19T08:11:38.000000000-04:00"
    )
    archive, _ = _nested_probe_oci_archive(
        provenance_statement_override=_canonical_test_json(statement)
    )
    with pytest.raises(D0Error, match="escape the build envelope"):
        _verify_probe_archive(archive)


def test_probe_verifier_rejects_extreme_outer_build_duration() -> None:
    archive, _ = _nested_probe_oci_archive()
    with pytest.raises(D0Error, match="escape the build envelope"):
        verify_probe_oci(
            archive,
            build_finished_unix_ns=(FIXTURE_BUILD_STARTED_UNIX_NS + 901 * 1_000_000_000),
            build_started_unix_ns=FIXTURE_BUILD_STARTED_UNIX_NS,
        )


def test_probe_oci_attestation_inventory_commits_exact_raw_statements() -> None:
    archive, identities = _nested_probe_oci_archive()
    inventory = _probe_oci_attestation_inventory(archive)
    assert inventory["archive_sha256"] == hashlib.sha256(archive).hexdigest()
    assert inventory["attestation_manifest_count"] == 1
    assert inventory["statement_count"] == 2
    assert inventory["graph"]["status"] == "pass"
    assert inventory["graph"]["unreferenced_members"] == []
    assert {item["predicate_type_statement"] for item in inventory["statements"]} == {
        "https://spdx.dev/Document",
        "https://slsa.dev/provenance/v1",
    }
    for statement in inventory["statements"]:
        raw = base64.b64decode(statement["statement_raw_base64"], validate=True)
        assert len(raw) == statement["statement_size"]
        assert hashlib.sha256(raw).hexdigest() == statement["statement_sha256"]
        assert json.loads(raw) == statement["statement"]
        predicate = _canonical_test_json(statement["predicate"])
        assert len(predicate) == statement["predicate_canonical_size"]
        assert hashlib.sha256(predicate).hexdigest() == statement["predicate_canonical_sha256"]
        assert statement["subject"] == []
    annotations = inventory["attestation_manifests"][0]["descriptor"]["annotations"]
    assert annotations["vnd.docker.reference.digest"] == identities["image_manifest_digest"]


def test_probe_oci_attestation_inventory_preserves_malformed_statement() -> None:
    malformed = b'{"predicate":'
    archive, _ = _nested_probe_oci_archive(spdx_statement_override=malformed)
    inventory = _probe_oci_attestation_inventory(archive)
    statement = next(
        item
        for item in inventory["statements"]
        if item["predicate_type_annotation"] == "https://spdx.dev/Document"
    )
    assert base64.b64decode(statement["statement_raw_base64"], validate=True) == malformed
    assert statement["statement_sha256"] == hashlib.sha256(malformed).hexdigest()
    assert statement["statement_size"] == len(malformed)
    assert statement["decode_error"] == "JSONDecodeError"
    assert statement["statement"] is None
    assert statement["predicate"] is None
    with pytest.raises(D0Error, match="statement is invalid"):
        _verify_probe_archive(archive)


def _syft_spdx_predicate() -> dict[str, object]:
    return {
        "SPDXID": "SPDXRef-DOCUMENT",
        "creationInfo": {
            "created": "2026-06-19T11:40:41Z",
            "creators": [
                "Organization: Anchore, Inc",
                "Tool: syft-v1.42.3",
                "Tool: buildkit-v0.30.0",
            ],
            "licenseListVersion": "3.28",
        },
        "dataLicense": "CC0-1.0",
        "documentNamespace": (
            "https://anchore.com/syft/dir/sbom-12345678-1234-4abc-8abc-123456789abc"
        ),
        "name": "sbom",
        "packages": [
            {
                "SPDXID": "SPDXRef-DocumentRoot-Directory-sbom",
                "copyrightText": "NOASSERTION",
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": False,
                "licenseConcluded": "NOASSERTION",
                "licenseDeclared": "NOASSERTION",
                "name": "sbom",
                "primaryPackagePurpose": "FILE",
                "supplier": "NOASSERTION",
            }
        ],
        "relationships": [
            {
                "relatedSpdxElement": "SPDXRef-DocumentRoot-Directory-sbom",
                "relationshipType": "DESCRIBES",
                "spdxElementId": "SPDXRef-DOCUMENT",
            }
        ],
        "spdxVersion": "SPDX-2.3",
    }


def test_spdx_validator_accepts_exact_live_syft_shape() -> None:
    verified = _validate_probe_spdx_predicate(_syft_spdx_predicate())
    assert verified["files_field_present"] is False
    assert verified["file_count"] == 0
    assert verified["content_binding"] == "spdx-document-root-package"


def test_spdx_validator_rejects_non_live_files_array() -> None:
    predicate = _syft_spdx_predicate()
    predicate["files"] = [
        {
            "SPDXID": "SPDXRef-File-probe.txt-deadbeef",
            "checksums": [
                {
                    "algorithm": "SHA256",
                    "checksumValue": runtime_module.PROBE_PAYLOAD_SHA256,
                }
            ],
            "fileName": "probe.txt",
        }
    ]
    with pytest.raises(D0Error, match="required structure differs"):
        _validate_probe_spdx_predicate(predicate)


@pytest.mark.parametrize(
    ("field", "replacement", "error"),
    (
        ("packages", [], "required structure differs"),
        ("relationships", [], "required structure differs"),
        ("documentNamespace", "https://example.invalid/spdx", "required structure differs"),
    ),
)
def test_spdx_validator_rejects_malformed_required_syft_document_fields(
    field: str, replacement: object, error: str
) -> None:
    predicate = _syft_spdx_predicate()
    predicate[field] = replacement
    with pytest.raises(D0Error, match=error):
        _validate_probe_spdx_predicate(predicate)


def test_probe_oci_graph_rejects_dangling_nested_descriptor() -> None:
    archive, identities = _nested_probe_oci_archive()
    members = runtime_module._safe_tar(archive)
    digest = str(identities["attestation_manifest_digest"]).split(":", 1)[1]
    del members[f"blobs/sha256/{digest}"]
    with pytest.raises(D0Error, match="descriptor is dangling"):
        _probe_oci_graph(_test_tar(members))


def test_probe_oci_graph_rejects_nested_descriptor_size_mismatch() -> None:
    archive, _ = _nested_probe_oci_archive()
    members = runtime_module._safe_tar(archive)
    index = json.loads(members["index.json"])
    index["manifests"][0]["size"] += 1
    members["index.json"] = _canonical_test_json(index)
    with pytest.raises(D0Error, match="digest or size differs"):
        _probe_oci_graph(_test_tar(members))


def test_probe_oci_graph_rejects_nested_descriptor_digest_mismatch() -> None:
    archive, identities = _nested_probe_oci_archive()
    members = runtime_module._safe_tar(archive)
    digest = str(identities["nested_index_digest"]).split(":", 1)[1]
    members[f"blobs/sha256/{digest}"] += b"\n"
    with pytest.raises(D0Error, match="digest or size differs"):
        _probe_oci_graph(_test_tar(members))


def test_probe_oci_graph_rejects_descriptor_cycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = "sha256:" + "0" * 64
    child = {
        "digest": digest,
        "mediaType": "application/vnd.oci.image.index.v1+json",
        "size": 0,
    }
    while True:
        payload = _canonical_test_json(
            {
                "manifests": [child],
                "mediaType": "application/vnd.oci.image.index.v1+json",
                "schemaVersion": 2,
            }
        )
        if child["size"] == len(payload):
            break
        child["size"] = len(payload)
    root = _canonical_test_json(
        {
            "manifests": [child],
            "mediaType": "application/vnd.oci.image.index.v1+json",
            "schemaVersion": 2,
        }
    )
    archive = _test_tar(
        {
            f"blobs/sha256/{'0' * 64}": payload,
            "index.json": root,
            "oci-layout": b'{"imageLayoutVersion":"1.0.0"}',
        }
    )
    monkeypatch.setattr(runtime_module, "sha256_bytes", lambda _value: "0" * 64)
    with pytest.raises(D0Error, match="descriptor cycle"):
        _probe_oci_graph(archive)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    (
        (
            {"attestation_reference": "sha256:" + "f" * 64},
            "cross-index or ambiguous",
        ),
        ({"subject_digest": "f" * 64}, "subject does not bind"),
    ),
)
def test_verify_probe_oci_rejects_mismatched_attestation_binding(
    kwargs: dict[str, str], error: str
) -> None:
    archive, _ = _nested_probe_oci_archive(**kwargs)
    with pytest.raises(D0Error, match=error):
        _verify_probe_archive(archive)


@pytest.mark.parametrize(
    ("manifest", "annotations", "shape", "attestations"),
    (
        (
            {"mediaType": "application/vnd.oci.image.manifest.v1+json"},
            None,
            "absent",
            0,
        ),
        (
            {
                "layers": None,
                "mediaType": "application/vnd.oci.image.manifest.v1+json",
                "subject": {
                    "digest": "sha256:" + "0" * 64,
                    "mediaType": "application/vnd.oci.image.manifest.v1+json",
                    "size": 1,
                },
            },
            {"vnd.docker.reference.type": "attestation-manifest"},
            "null",
            1,
        ),
    ),
)
def test_probe_oci_inventory_handles_absent_or_null_manifest_layers(
    manifest: dict[str, object],
    annotations: dict[str, str] | None,
    shape: str,
    attestations: int,
) -> None:
    inventory = _probe_oci_inventory(_probe_inventory_archive(manifest, annotations=annotations))
    assert inventory["attestation_descriptor_count"] == attestations
    assert inventory["manifest_descriptors"][0]["manifest_layers_shape"] == shape
    assert inventory["manifest_descriptors"][0]["references"] == []


def test_probe_oci_inventory_rejects_non_object_manifest_descriptor() -> None:
    index = b'{"manifests":["malformed"],"schemaVersion":2}'
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name, payload in (
            ("oci-layout", b'{"imageLayoutVersion":"1.0.0"}'),
            ("index.json", index),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o444
            info.mtime = 0
            archive.addfile(info, io.BytesIO(payload))
    with pytest.raises(D0Error, match="descriptor is not an object"):
        _probe_oci_inventory(output.getvalue())


def test_egress_socket_observations_fail_closed_on_oversized_row() -> None:
    descriptor = {
        "host_address": "198.18.134.1",
        "peer_address": "198.18.134.2",
        "port": 43022,
    }
    with pytest.raises(D0Error, match="row exceeds"):
        _egress_socket_observations(
            (b"CLOSING 198.18.134.1:1 198.18.134.2:43022 " + b"x" * 4096),
            descriptor,
        )


def test_egress_guard_runs_after_conntrack_before_matching_established() -> None:
    program = _buildkit_egress_program(
        "exact_table",
        {
            "family": "ip",
            "client_address": "100.64.0.1",
            "client_port": 49152,
            "server_address": "100.64.0.2",
        },
    ).decode("ascii")
    chain = "type filter hook output priority 0; policy drop;"
    assert chain in program
    assert chain in NFT_GUARD_INSTALL_SCRIPT
    assert "priority -200" not in program
    assert "priority -200" not in NFT_GUARD_INSTALL_SCRIPT
    assert "tcp sport 22 ct state established accept" in program


def _nft_table_fixture(state_op: str, state_right: object) -> dict[str, object]:
    table = "cascadia_r2_d0"
    binding = {"family": "inet", "table": table, "chain": "output"}
    return {
        "nftables": [
            {"metainfo": {"json_schema_version": 1}},
            {"table": {"family": "inet", "name": table}},
            {
                "chain": {
                    "family": "inet",
                    "table": table,
                    "name": "output",
                    "type": "filter",
                    "hook": "output",
                    "prio": 0,
                    "policy": "drop",
                }
            },
            {
                "rule": {
                    **binding,
                    "comment": "cascadia-d0-loopback",
                    "expr": [
                        {
                            "match": {
                                "op": "==",
                                "left": {"meta": {"key": "oifname"}},
                                "right": "lo",
                            }
                        },
                        {"accept": None},
                    ],
                }
            },
            {
                "rule": {
                    **binding,
                    "comment": "cascadia-d0-ssh-reply",
                    "expr": [
                        {
                            "match": {
                                "op": "==",
                                "left": {"payload": {"protocol": "tcp", "field": "sport"}},
                                "right": 22,
                            }
                        },
                        {
                            "match": {
                                "op": state_op,
                                "left": {"ct": {"key": "state"}},
                                "right": state_right,
                            }
                        },
                        {"accept": None},
                    ],
                }
            },
            {
                "rule": {
                    **binding,
                    "comment": "cascadia-d0-reject",
                    "expr": [
                        {"counter": {"packets": 0, "bytes": 0}},
                        {"reject": None},
                    ],
                }
            },
        ]
    }


def _nft_trace_table_fixture() -> dict[str, object]:
    value = _nft_table_fixture("in", "established")
    entries = value["nftables"]
    assert isinstance(entries, list)
    table = "cascadia_r2_d0"
    schemas = {
        "tcp4": [
            "ifname",
            "ipv4_addr",
            "inet_service",
            "ipv4_addr",
            "inet_service",
        ],
        "udp4": [
            "ifname",
            "ipv4_addr",
            "inet_service",
            "ipv4_addr",
            "inet_service",
        ],
        "other4": ["ifname", "ipv4_addr", "ipv4_addr", "inet_proto"],
        "tcp6": [
            "ifname",
            "ipv6_addr",
            "inet_service",
            "ipv6_addr",
            "inet_service",
        ],
        "udp6": [
            "ifname",
            "ipv6_addr",
            "inet_service",
            "ipv6_addr",
            "inet_service",
        ],
        "other6": ["ifname", "ipv6_addr", "ipv6_addr", "inet_proto"],
    }
    set_entries = [
        {
            "set": {
                "family": "inet",
                "flags": ["dynamic"],
                "name": name,
                "size": 65_535,
                "stmt": [{"counter": None}],
                "table": table,
                "type": nft_type,
            }
        }
        for name, nft_type in schemas.items()
    ]
    operands = {
        "tcp4": [
            {"meta": {"key": "oifname"}},
            {"payload": {"field": "saddr", "protocol": "ip"}},
            {"payload": {"field": "sport", "protocol": "tcp"}},
            {"payload": {"field": "daddr", "protocol": "ip"}},
            {"payload": {"field": "dport", "protocol": "tcp"}},
        ],
        "udp4": [
            {"meta": {"key": "oifname"}},
            {"payload": {"field": "saddr", "protocol": "ip"}},
            {"payload": {"field": "sport", "protocol": "udp"}},
            {"payload": {"field": "daddr", "protocol": "ip"}},
            {"payload": {"field": "dport", "protocol": "udp"}},
        ],
        "other4": [
            {"meta": {"key": "oifname"}},
            {"payload": {"field": "saddr", "protocol": "ip"}},
            {"payload": {"field": "daddr", "protocol": "ip"}},
            {"meta": {"key": "l4proto"}},
        ],
        "tcp6": [
            {"meta": {"key": "oifname"}},
            {"payload": {"field": "saddr", "protocol": "ip6"}},
            {"payload": {"field": "sport", "protocol": "tcp"}},
            {"payload": {"field": "daddr", "protocol": "ip6"}},
            {"payload": {"field": "dport", "protocol": "tcp"}},
        ],
        "udp6": [
            {"meta": {"key": "oifname"}},
            {"payload": {"field": "saddr", "protocol": "ip6"}},
            {"payload": {"field": "sport", "protocol": "udp"}},
            {"payload": {"field": "daddr", "protocol": "ip6"}},
            {"payload": {"field": "dport", "protocol": "udp"}},
        ],
        "other6": [
            {"meta": {"key": "oifname"}},
            {"payload": {"field": "saddr", "protocol": "ip6"}},
            {"payload": {"field": "daddr", "protocol": "ip6"}},
            {"meta": {"key": "l4proto"}},
        ],
    }
    trace_rules = [
        {
            "rule": {
                "chain": "output",
                "comment": f"cascadia-d0-trace-{name}",
                "expr": [
                    {
                        "set": {
                            "elem": {"concat": operands[name]},
                            "op": "update",
                            "set": f"@{name}",
                        }
                    }
                ],
                "family": "inet",
                "table": table,
            }
        }
        for name in schemas
    ]
    entries[2:2] = set_entries
    entries[-1:-1] = trace_rules
    return value


def test_nft_trace_validator_and_tuple_delta_are_exact() -> None:
    before = _nft_trace_table_fixture()
    assert _validate_buildkit_egress_table(
        before,
        table="cascadia_r2_d0",
        management_flow=MANAGEMENT_FLOW,
        trace=True,
    ) == {"reject_rules": 1, "rejected_packets": 0, "rejected_bytes": 0}
    after = json.loads(json.dumps(before))
    entries = after["nftables"]
    assert isinstance(entries, list)
    tcp4 = next(entry["set"] for entry in entries if entry.get("set", {}).get("name") == "tcp4")
    tcp4["elem"] = [
        {
            "elem": {
                "counter": {"bytes": 60, "packets": 1},
                "val": {
                    "concat": [
                        "",
                        "100.0.0.0",
                        50706,
                        "158.28.0.0",
                        50706,
                    ]
                },
            }
        }
    ]
    before_projection = _egress_trace_projection(before, table="cascadia_r2_d0")
    after_projection = _egress_trace_projection(after, table="cascadia_r2_d0")
    assert _egress_trace_delta(before_projection, after_projection) == {
        "bytes": 60,
        "packets": 1,
        "tuples": [
            {
                "bytes": 60,
                "packets": 1,
                "set": "tcp4",
                "socket_uid": None,
                "socket_uid_available": False,
                "tuple": ["", "100.0.0.0", 50706, "158.28.0.0", 50706],
            }
        ],
    }


def test_nft_trace_projection_rejects_inferred_uid_or_untyped_values() -> None:
    value = _nft_trace_table_fixture()
    entries = value["nftables"]
    assert isinstance(entries, list)
    tcp4 = next(entry["set"] for entry in entries if entry.get("set", {}).get("name") == "tcp4")
    tcp4["elem"] = [
        {
            "elem": {
                "counter": {"bytes": 60, "packets": 1},
                "val": {"concat": [0, "", "100.0.0.0", 50706, "158.28.0.0", 50706]},
            }
        }
    ]
    with pytest.raises(D0Error, match="trace tuple or counter"):
        _egress_trace_projection(value, table="cascadia_r2_d0")
    tcp4["elem"][0]["elem"]["val"]["concat"] = [
        "",
        "100.0.0.0",
        True,
        "158.28.0.0",
        50706,
    ]
    with pytest.raises(D0Error, match="typed trace tuple"):
        _egress_trace_projection(value, table="cascadia_r2_d0")


def test_nft_trace_existing_element_counter_delta_matches_reject_delta() -> None:
    before = _nft_trace_table_fixture()
    entries = before["nftables"]
    assert isinstance(entries, list)
    tcp4 = next(entry["set"] for entry in entries if entry.get("set", {}).get("name") == "tcp4")
    exact_tuple = ["c2d0h85d3319e", "198.18.134.1", 49670, "198.18.134.2", 43022]
    tcp4["elem"] = [
        {
            "elem": {
                "counter": {"bytes": 212, "packets": 4},
                "val": {"concat": exact_tuple},
            }
        }
    ]
    after = json.loads(json.dumps(before))
    after_entries = after["nftables"]
    after_tcp4 = next(
        entry["set"] for entry in after_entries if entry.get("set", {}).get("name") == "tcp4"
    )
    after_tcp4["elem"][0]["elem"]["counter"] = {"bytes": 383, "packets": 7}
    trace_delta = _egress_trace_delta(
        _egress_trace_projection(before, table="cascadia_r2_d0"),
        _egress_trace_projection(after, table="cascadia_r2_d0"),
    )
    assert trace_delta["packets"] == 3
    assert trace_delta["bytes"] == 171
    assert (
        _trace_counter_comparison(
            trace_delta,
            {"rejected_bytes": 171, "rejected_packets": 3},
        )["equal"]
        is True
    )


@pytest.mark.parametrize(
    ("set_name", "values", "destination", "protocol"),
    (
        (
            "tcp4",
            ["eth0", "198.51.100.1", 49152, "203.0.113.9", 443],
            "203.0.113.9",
            None,
        ),
        (
            "other4",
            ["eth0", "198.51.100.1", "203.0.113.9", "icmp"],
            "203.0.113.9",
            "icmp",
        ),
        (
            "other6",
            ["eth0", "2001:db8::1", "2001:db8::2", "ipv6-icmp"],
            "2001:db8::2",
            "ipv6-icmp",
        ),
    ),
)
def test_egress_trace_tuple_schema_never_treats_protocol_as_destination(
    set_name: str,
    values: list[object],
    destination: str,
    protocol: str | None,
) -> None:
    fields = _egress_trace_tuple_fields({"set": set_name, "tuple": values})
    assert fields["destination_address"] == destination
    if protocol is not None:
        assert fields["protocol"] == protocol
        assert fields["destination_address"] != protocol


@pytest.mark.parametrize(
    "row",
    (
        {"set": "other6", "tuple": ["eth0", "2001:db8::1", "2001:db8::2", ""]},
        {"set": "other4", "tuple": ["eth0", "2001:db8::1", "2001:db8::2", "icmp"]},
        {"set": "other6", "tuple": ["eth0", "2001:db8::1", "ipv6-icmp", "ipv6-icmp"]},
    ),
)
def test_egress_trace_tuple_schema_rejects_malformed_other_protocol_rows(
    row: dict[str, object],
) -> None:
    with pytest.raises(D0Error, match=r"trace (protocol|address)"):
        _egress_trace_tuple_fields(row)


def test_scoped_egress_accounting_classifies_exact_v32_local_control_denials() -> None:
    delta = {
        "bytes": 416,
        "packets": 5,
        "tuples": [
            {
                "bytes": 192,
                "packets": 2,
                "set": "other6",
                "tuple": [
                    "docker0",
                    "fe80::c814:ebff:fe23:1d2e",
                    "ff02::16",
                    "ipv6-icmp",
                ],
            },
            {
                "bytes": 152,
                "packets": 2,
                "set": "other6",
                "tuple": ["veth578df0d", "::", "ff02::16", "ipv6-icmp"],
            },
            {
                "bytes": 72,
                "packets": 1,
                "set": "other6",
                "tuple": [
                    "veth578df0d",
                    "::",
                    "ff02::1:ffa1:bb7f",
                    "ipv6-icmp",
                ],
            },
        ],
    }
    scoped = _classify_egress_trace_delta(delta)
    assert scoped["external_reject_zero"] is True
    assert scoped["external_or_unclassified_denied"] == {
        "bytes": 0,
        "packets": 0,
        "tuples": [],
    }
    local = scoped["local_control_denied"]
    assert (local["packets"], local["bytes"]) == (5, 416)
    assert {row["destination_family"] for row in local["tuples"]} == {6}
    assert {row["destination_multicast_scope"] for row in local["tuples"]} == {2}
    assert {row["protocol"] for row in local["tuples"]} == {"ipv6-icmp"}
    assert {row["output_interface"] for row in local["tuples"]} == {
        "docker0",
        "veth578df0d",
    }
    assert scoped["local_control_policy"] == "denied-and-accounted-not-whitelisted"


@pytest.mark.parametrize(
    "values",
    (
        ["eth0", "192.0.2.4", "198.51.100.8", "icmp"],
        ["eth0", "2001:db8::1", "2001:db8::2", "ipv6-icmp"],
        ["eth0", "::", "ff02::16", "ipv6-icmp"],
        ["docker0", "::", "ff05::16", "ipv6-icmp"],
        ["docker0", "::", "ff02::16", "udp"],
    ),
)
def test_scoped_egress_accounting_keeps_nonexact_traffic_external(
    values: list[object],
) -> None:
    set_name = "other4" if ":" not in str(values[2]) else "other6"
    delta = {
        "bytes": 64,
        "packets": 1,
        "tuples": [
            {
                "bytes": 64,
                "packets": 1,
                "set": set_name,
                "tuple": values,
            }
        ],
    }
    scoped = _classify_egress_trace_delta(delta)
    assert scoped["external_reject_zero"] is False
    assert scoped["local_control_denied"]["packets"] == 0
    assert scoped["external_or_unclassified_denied"]["packets"] == 1


def test_trace_counter_mismatch_is_diagnostic_data_not_an_exception() -> None:
    assert _trace_counter_comparison(
        {"bytes": 342, "packets": 6},
        {"rejected_bytes": 212, "rejected_packets": 4},
    ) == {
        "byte_difference": 130,
        "bytes_equal": False,
        "equal": False,
        "packet_difference": 2,
        "packets_equal": False,
    }
    assert (
        _trace_counter_comparison(
            {"bytes": 212, "packets": 4},
            {"rejected_bytes": 212, "rejected_packets": 4},
        )["equal"]
        is True
    )


def test_raw_nft_inventory_preserves_unknown_element_shape_without_typing_it() -> None:
    raw = json.dumps(
        {
            "nftables": [
                {"metainfo": {"json_schema_version": 1}},
                {
                    "set": {
                        "elem": [
                            {
                                "elem": {
                                    "counter": {"bytes": 342, "packets": 6},
                                    "val": {
                                        "concat": [
                                            {"prefix": {"addr": 0, "len": 32}},
                                            "eth0",
                                            "198.18.134.1",
                                            50408,
                                            "198.18.134.2",
                                            43022,
                                        ]
                                    },
                                }
                            }
                        ],
                        "name": "tcp4",
                    }
                },
                {
                    "rule": {
                        "expr": [
                            {
                                "set": {
                                    "elem": {
                                        "concat": [
                                            {"meta": {"key": "skuid"}},
                                            {"meta": {"key": "oifname"}},
                                        ]
                                    },
                                    "op": "update",
                                    "set": "@tcp4",
                                }
                            }
                        ]
                    }
                },
            ]
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    result = _bounded_raw_nft_inventory(raw)
    assert result == {
        "entry_count": 3,
        "raw_json": raw.decode("ascii"),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "raw_size": len(raw),
        "status": "bounded-raw-inventory",
    }


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"[]",
        b'{"nftables":{}}',
        b'{"nftables":[]}',
        b'{"nftables":[0]}',
        b'{"extra":1,"nftables":[{}]}',
        b"\xff",
        b"{" + b" " * NFT_SCHEMA_INVENTORY_MAX_BYTES,
    ],
)
def test_raw_nft_inventory_rejects_unbounded_or_non_envelope_data(raw: bytes) -> None:
    with pytest.raises(D0Error):
        _bounded_raw_nft_inventory(raw)


def test_docker_accounting_command_evidence_is_exact_and_bounded() -> None:
    completed = Completed(
        argv=(DOCKER, "system", "df", "--format", "{{json .}}"),
        returncode=0,
        stdout=b'{"Type":"Images","TotalCount":"1"}\n',
        stderr=b"",
    )
    evidence = _bounded_completed_evidence(completed)
    assert evidence["argv"] == list(completed.argv)
    assert evidence["returncode"] == 0
    assert evidence["stdout"] == completed.stdout.decode()
    assert evidence["stdout_size"] == len(completed.stdout)
    assert evidence["stdout_sha256"] == hashlib.sha256(completed.stdout).hexdigest()
    with pytest.raises(D0Error, match="exceeds its bound"):
        _bounded_completed_evidence(
            Completed(
                argv=("oversized",),
                returncode=0,
                stdout=b"x" * (DOCKER_ACCOUNTING_COMMAND_MAX_BYTES + 1),
                stderr=b"",
            )
        )


def test_scanner_attestation_residue_identity_and_empty_daemon_gate_are_exact() -> None:
    manifest = "sha256:" + "1" * 64
    config = "sha256:" + "2" * 64
    provenance = "sha256:" + "3" * 64
    spdx = "sha256:" + "4" * 64
    reference, digests = _scanner_attestation_residue_identity(
        {
            "attestation_config_digest": config,
            "attestation_manifest_digest": manifest,
            "provenance_digest": provenance,
            "spdx_digest": spdx,
        }
    )
    assert reference == f"moby-dangling@{manifest}"
    assert digests == tuple(sorted((manifest, config, provenance, spdx)))
    clean = {
        "build_cache": 0,
        "containers": 0,
        "images": 0,
        "info_containers": 0,
        "info_images": 0,
        "volumes": 0,
    }
    _require_empty_daemon_accounting(clean)
    with pytest.raises(D0Error, match="baseline is not empty"):
        _require_empty_daemon_accounting({**clean, "images": 1, "info_images": 1})
    compile(SCANNER_ATTESTATION_CLEANUP_SCRIPT, "scanner-attestation-cleanup", "exec")
    assert "validate-precondition" in SCANNER_ATTESTATION_CLEANUP_SCRIPT
    assert "mutation_started" in SCANNER_ATTESTATION_CLEANUP_SCRIPT
    assert "commands" in SCANNER_ATTESTATION_CLEANUP_SCRIPT
    assert "['images','remove','--sync',reference]" in SCANNER_ATTESTATION_CLEANUP_SCRIPT
    assert "content_after" in SCANNER_ATTESTATION_CLEANUP_SCRIPT


@pytest.mark.parametrize(
    ("state_op", "state_right"),
    [
        ("in", "established"),
        ("in", ["established"]),
        ("==", "established"),
    ],
)
def test_nft_validator_accepts_exact_established_state_encodings(
    state_op: str, state_right: object
) -> None:
    assert _validate_buildkit_egress_table(
        _nft_table_fixture(state_op, state_right),
        table="cascadia_r2_d0",
        management_flow=MANAGEMENT_FLOW,
    ) == {"reject_rules": 1, "rejected_packets": 0, "rejected_bytes": 0}


@pytest.mark.parametrize(
    ("state_op", "state_right"),
    [
        ("in", "new"),
        ("in", ["established", "related"]),
        ("==", ["established"]),
        ("!=", "established"),
        ("in", 1),
    ],
)
def test_nft_validator_rejects_broader_or_malformed_state_encodings(
    state_op: str, state_right: object
) -> None:
    with pytest.raises(D0Error, match="SSH-reply rule differs"):
        _validate_buildkit_egress_table(
            _nft_table_fixture(state_op, state_right),
            table="cascadia_r2_d0",
            management_flow=MANAGEMENT_FLOW,
        )


def test_reject_counter_delta_uses_nonzero_install_baseline() -> None:
    assert _reject_counter_delta(
        {"reject_rules": 1, "rejected_packets": 7, "rejected_bytes": 420},
        {"reject_rules": 1, "rejected_packets": 8, "rejected_bytes": 480},
    ) == {"rejected_packets": 1, "rejected_bytes": 60}


@pytest.mark.parametrize(
    ("before", "after"),
    [
        (
            {"reject_rules": 1, "rejected_packets": 2, "rejected_bytes": 20},
            {"reject_rules": 1, "rejected_packets": 1, "rejected_bytes": 20},
        ),
        (
            {"reject_rules": 1, "rejected_packets": 2, "rejected_bytes": 20},
            {"reject_rules": 2, "rejected_packets": 3, "rejected_bytes": 30},
        ),
        (
            {"reject_rules": 1, "rejected_packets": True, "rejected_bytes": 20},
            {"reject_rules": 1, "rejected_packets": 3, "rejected_bytes": 30},
        ),
    ],
)
def test_reject_counter_delta_rejects_regression_or_shape_drift(
    before: dict[str, object], after: dict[str, object]
) -> None:
    with pytest.raises(D0Error, match=r"counter|cardinality"):
        _reject_counter_delta(before, after)


def test_reject_counter_transition_requires_both_control_counters_to_increase() -> None:
    baseline = {"reject_rules": 1, "rejected_packets": 7, "rejected_bytes": 420}
    assert _require_reject_counter_transition(
        baseline,
        {"reject_rules": 1, "rejected_packets": 8, "rejected_bytes": 480},
        expect_positive=True,
    ) == {"rejected_packets": 1, "rejected_bytes": 60}
    for after in (
        baseline,
        {"reject_rules": 1, "rejected_packets": 8, "rejected_bytes": 420},
        {"reject_rules": 1, "rejected_packets": 7, "rejected_bytes": 480},
    ):
        with pytest.raises(D0Error, match="did not increment"):
            _require_reject_counter_transition(baseline, after, expect_positive=True)


def test_reject_counter_transition_requires_offline_build_delta_zero() -> None:
    baseline = {"reject_rules": 1, "rejected_packets": 8, "rejected_bytes": 480}
    assert _require_reject_counter_transition(baseline, baseline, expect_positive=False) == {
        "rejected_packets": 0,
        "rejected_bytes": 0,
    }
    with pytest.raises(D0Error, match="BuildKit attempted guest egress"):
        _require_reject_counter_transition(
            baseline,
            {"reject_rules": 1, "rejected_packets": 9, "rejected_bytes": 540},
            expect_positive=False,
        )


def test_negative_control_server_timeout_precedes_document_deadline() -> None:
    assert EGRESS_CONTROL_RECEIVE_TIMEOUT_SECONDS == 5
    assert EGRESS_CONTROL_DOCUMENT_TIMEOUT_SECONDS == 15
    assert EGRESS_CONTROL_RECEIVE_TIMEOUT_SECONDS < EGRESS_CONTROL_DOCUMENT_TIMEOUT_SECONDS
    assert "connection.settimeout(receive_timeout)" in runtime_module._EGRESS_SERVER_SCRIPT
    assert "connection.settimeout(60)" not in runtime_module._EGRESS_SERVER_SCRIPT


@pytest.mark.parametrize(
    "pid_mode",
    ("host", "container:abc123", "private", None, 0),
)
def test_default_private_pid_namespace_rejects_sharing_or_unexpected_config(
    pid_mode: object,
) -> None:
    with pytest.raises(D0Error, match="PID namespace mode"):
        _validate_default_private_pid_mode({"PidMode": pid_mode})


def _colima_status_document(packet: dict[str, object]) -> dict[str, object]:
    colima_home = Path(packet["paths"]["colima_home"])  # type: ignore[index]
    return {
        "display_name": "colima [profile=cascadia-r2]",
        "driver": "macOS Virtualization.Framework",
        "arch": "aarch64",
        "runtime": "docker",
        "mount_type": "virtiofs",
        "docker_socket": f"unix://{colima_home / 'cascadia-r2' / 'docker.sock'}",
        "containerd_socket": f"unix://{colima_home / 'cascadia-r2' / 'containerd.sock'}",
        "kubernetes": False,
        "cpu": 10,
        "memory": 14 * 1024**3,
        "disk": 13 * 1024**3,
    }


def test_colima_status_accepts_exact_modern_and_legacy_running_schemas() -> None:
    packet = _packet("john1", "verify")
    modern = _colima_status_document(packet)
    result = _colima_status(json.dumps(modern).encode(), packet)
    assert result["status"] == "running"
    assert result["schema"] == "modern-statusless"
    assert result["observed_status"] is None
    for status in ("running", "started", "RUNNING"):
        legacy = {**modern, "status": status}
        result = _colima_status(json.dumps(legacy).encode(), packet)
        assert result["schema"] == "legacy-explicit-status"
        assert result["observed_status"] == status.lower()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("display_name", "colima"),
        ("driver", "qemu"),
        ("arch", "x86_64"),
        ("runtime", "containerd"),
        ("mount_type", "sshfs"),
        ("docker_socket", "unix:///tmp/docker.sock"),
        ("containerd_socket", "unix:///tmp/containerd.sock"),
        ("kubernetes", True),
        ("cpu", 9),
        ("memory", 13 * 1024**3),
        ("disk", 12 * 1024**3),
    ),
)
def test_colima_status_rejects_effective_configuration_drift(field: str, value: object) -> None:
    packet = _packet("john1", "verify")
    document = _colima_status_document(packet)
    document[field] = value
    with pytest.raises(D0Error, match="effective configuration"):
        _colima_status(json.dumps(document).encode(), packet)


def test_colima_status_rejects_missing_extra_or_stopped_fields() -> None:
    packet = _packet("john1", "verify")
    document = _colima_status_document(packet)
    missing = dict(document)
    missing.pop("driver")
    extra = {**document, "unexpected": True}
    for invalid in (missing, extra):
        with pytest.raises(D0Error, match="effective field set"):
            _colima_status(json.dumps(invalid).encode(), packet)
    for status in ("stopped", "error", ""):
        with pytest.raises(D0Error, match="not running"):
            _colima_status(json.dumps({**document, "status": status}).encode(), packet)


def _engine_info(**overrides: object) -> bytes:
    value: dict[str, object] = {
        "OSType": "linux",
        "Architecture": "aarch64",
        "ServerVersion": EXPECTED_ENGINE_VERSION,
        "Swarm": {"LocalNodeState": "inactive"},
        "ID": "sealed-daemon-id",
        "Name": "colima-cascadia-r2",
        "OperatingSystem": "Ubuntu 24.04 LTS",
        "KernelVersion": "6.8.0-generic",
        "DockerRootDir": "/var/lib/docker",
        "Driver": "overlayfs",
        "CgroupDriver": "cgroupfs",
        "CgroupVersion": "2",
        "HttpProxy": "",
        "HttpsProxy": "",
        "NoProxy": "",
        "RegistryConfig": {
            "InsecureRegistryCIDRs": ["127.0.0.0/8", "::1/128"],
            "IndexConfigs": {
                "docker.io": {
                    "Name": "docker.io",
                    "Mirrors": [],
                    "Secure": True,
                    "Official": True,
                }
            },
            "Mirrors": [],
        },
        "SecurityOptions": [
            "name=apparmor",
            "name=seccomp,profile=builtin",
            "name=cgroupns",
        ],
    }
    value.update(overrides)
    return json.dumps(value).encode("utf-8")


def test_engine_info_matches_the_sealed_daemon_configuration() -> None:
    result = verify_engine_info(_engine_info())
    assert result["driver"] == "overlayfs"
    assert result["cgroup_driver"] == "cgroupfs"
    daemon = verify_daemon_config(
        json.dumps(
            {
                "exec-opts": ["native.cgroupdriver=cgroupfs"],
                "features": {"buildkit": True, "containerd-snapshotter": True},
            }
        ),
        result,
    )
    assert daemon["status"] == "pass"


@pytest.mark.parametrize(
    ("field", "legacy_value"),
    (("Driver", "overlay2"), ("CgroupDriver", "systemd")),
)
def test_engine_info_rejects_values_contradicting_the_sealed_daemon_configuration(
    field: str, legacy_value: str
) -> None:
    with pytest.raises(D0Error, match="Docker Engine configuration differs"):
        verify_engine_info(_engine_info(**{field: legacy_value}))


def test_engine_registry_config_canonicalizes_only_cidr_order() -> None:
    first = verify_engine_info(_engine_info())
    reversed_config = {
        "InsecureRegistryCIDRs": ["::1/128", "127.0.0.0/8"],
        "IndexConfigs": {
            "docker.io": {
                "Name": "docker.io",
                "Mirrors": [],
                "Secure": True,
                "Official": True,
            }
        },
        "Mirrors": [],
    }
    second = verify_engine_info(_engine_info(RegistryConfig=reversed_config))
    assert first["registry_config_sha256"] == second["registry_config_sha256"]


@pytest.mark.parametrize(
    "registry",
    (
        {
            "InsecureRegistryCIDRs": ["127.0.0.0/8", "127.0.0.0/8"],
            "IndexConfigs": {},
            "Mirrors": [],
        },
        {
            "InsecureRegistryCIDRs": ["127.0.0.0/8", "10.0.0.0/8"],
            "IndexConfigs": {},
            "Mirrors": [],
        },
        {
            "InsecureRegistryCIDRs": ["127.0.0.0/8", "::1/128"],
            "IndexConfigs": {},
            "Mirrors": ["https://mirror.invalid"],
        },
        {
            "InsecureRegistryCIDRs": ["127.0.0.0/8", "::1/128"],
            "IndexConfigs": {"docker.io": {"Name": "wrong"}},
            "Mirrors": [],
        },
        {
            "InsecureRegistryCIDRs": ["127.0.0.0/8", "::1/128"],
            "IndexConfigs": {},
            "Mirrors": [],
            "Extra": True,
        },
    ),
)
def test_engine_registry_config_rejects_duplicate_extra_or_wrong_values(
    registry: dict[str, object],
) -> None:
    with pytest.raises(D0Error, match="registry configuration differs"):
        verify_engine_info(_engine_info(RegistryConfig=registry))


@pytest.mark.parametrize(
    "daemon_config",
    (
        {
            "exec-opts": ["native.cgroupdriver=systemd"],
            "features": {"buildkit": True, "containerd-snapshotter": True},
        },
        {
            "exec-opts": ["native.cgroupdriver=cgroupfs"],
            "features": {"buildkit": True, "containerd-snapshotter": False},
        },
    ),
)
def test_daemon_config_rejects_semantics_that_disagree_with_engine_contract(
    daemon_config: dict[str, object],
) -> None:
    with pytest.raises(D0Error, match="guest Docker daemon configuration differs"):
        verify_daemon_config(json.dumps(daemon_config), verify_engine_info(_engine_info()))


class _ContextRunner:
    def __init__(self, outputs: dict[tuple[str, ...], bytes]) -> None:
        self.outputs = outputs

    def run(self, argv: list[str], **_kwargs: object) -> Completed:
        key = tuple(argv)
        return Completed(key, 0, self.outputs[key], b"")


def _context_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, object], str, str, dict[tuple[str, ...], bytes]]:
    config = tmp_path / "docker"
    name = "colima-cascadia-r2"
    endpoint = f"unix://{tmp_path}/colima/cascadia-r2/docker.sock"
    directory = __import__("hashlib").sha256(name.encode()).hexdigest()
    metadata = {
        "Name": name,
        "Metadata": {"Description": "colima [profile=cascadia-r2]"},
        "Endpoints": {"docker": {"Host": endpoint, "SkipTLSVerify": False}},
    }
    named_root = config / "contexts" / "meta" / directory
    named_root.mkdir(parents=True)
    os.chmod(config / "contexts", 0o755)
    os.chmod(config / "contexts" / "meta", 0o755)
    os.chmod(named_root, 0o755)
    metadata_path = named_root / "meta.json"
    metadata_path.write_text(json.dumps(metadata, separators=(",", ":")))
    os.chmod(metadata_path, 0o644)
    packet: dict[str, object] = {"paths": {"docker_config": str(config)}}
    monkeypatch.setattr(
        runtime_module, "runtime_environment", lambda _packet: {"DOCKER_HOST": endpoint}
    )
    rows = [
        {
            "Current": False,
            "Description": "colima [profile=cascadia-r2]",
            "DockerEndpoint": endpoint,
            "Error": "",
            "Name": name,
        },
        {
            "Current": True,
            "Description": "Current DOCKER_HOST based configuration",
            "DockerEndpoint": endpoint,
            "Error": "",
            "Name": "default",
        },
    ]
    inspected = [
        {
            "Name": "default",
            "Metadata": {},
            "Endpoints": {"docker": {"Host": endpoint, "SkipTLSVerify": False}},
            "TLSMaterial": {},
            "Storage": {"MetadataPath": "<IN MEMORY>", "TLSPath": "<IN MEMORY>"},
        },
        {
            **metadata,
            "TLSMaterial": {},
            "Storage": {
                "MetadataPath": str(named_root),
                "TLSPath": str(config / "contexts" / "tls" / directory),
            },
        },
    ]
    outputs = {
        (DOCKER, "context", "show"): b"default\n",
        (DOCKER, "context", "ls", "--format", "{{json .}}"): b"".join(
            json.dumps(row, separators=(",", ":")).encode() + b"\n" for row in rows
        ),
        (DOCKER, "context", "inspect", "default", name): json.dumps(inspected).encode(),
    }
    return packet, name, endpoint, outputs


def test_context_contract_accepts_only_default_plus_exact_colima_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet, name, endpoint, outputs = _context_fixture(tmp_path, monkeypatch)
    result = _docker_context_snapshot(packet, _ContextRunner(outputs))
    assert result["context_names"] == ["default", name]
    assert result["effective_docker_host"] == endpoint
    assert result["named_context"]["tls_storage_absent"] is True


def test_context_contract_rejects_extra_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet, _name, endpoint, outputs = _context_fixture(tmp_path, monkeypatch)
    key = (DOCKER, "context", "ls", "--format", "{{json .}}")
    extra = {
        "Current": False,
        "Description": "unauthorized",
        "DockerEndpoint": endpoint,
        "Error": "",
        "Name": "extra",
    }
    outputs[key] += json.dumps(extra).encode() + b"\n"
    with pytest.raises(D0Error, match="Docker context inventory differs"):
        _docker_context_snapshot(packet, _ContextRunner(outputs))


def test_context_contract_rejects_wrong_endpoint_or_activation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet, _name, _endpoint, outputs = _context_fixture(tmp_path, monkeypatch)
    key = (DOCKER, "context", "ls", "--format", "{{json .}}")
    outputs[key] = outputs[key].replace(b"docker.sock", b"wrong.sock", 1)
    with pytest.raises(D0Error, match="Docker context inventory differs"):
        _docker_context_snapshot(packet, _ContextRunner(outputs))

    packet, name, _endpoint, outputs = _context_fixture(tmp_path / "activation", monkeypatch)
    outputs[(DOCKER, "context", "show")] = f"{name}\n".encode()
    with pytest.raises(D0Error, match="current context"):
        _docker_context_snapshot(packet, _ContextRunner(outputs))


def test_context_contract_rejects_tls_or_credential_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet, _name, _endpoint, outputs = _context_fixture(tmp_path, monkeypatch)
    config = Path(packet["paths"]["docker_config"])  # type: ignore[index]
    (config / "contexts" / "tls").mkdir()
    with pytest.raises(D0Error, match="unexpected state"):
        _docker_context_snapshot(packet, _ContextRunner(outputs))


def _runtime_activity_fixture(tmp_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    home = "/Users/johnherrick"
    colima = tmp_path / "colima"
    instance = colima / "_lima" / "colima-cascadia-r2"
    network = colima / "_lima" / "_networks" / "user-v2"
    usernet_pid, hostagent_pid, ssh_pid = 101, 102, 103
    packet: dict[str, object] = {
        "host": "john1",
        "paths": {"colima_home": str(colima)},
    }
    activity: dict[str, object] = {
        "processes": [
            (
                f"{usernet_pid} 1 johnherrick /opt/homebrew/bin/limactl usernet "
                f"-p {network}/usernet_user-v2.pid -e {network}/user-v2_ep.sock "
                f"--listen-qemu {network}/user-v2_qemu.sock --listen "
                f"{network}/user-v2_fd.sock --subnet 192.168.5.0/24"
            ),
            (
                f"{hostagent_pid} 1 johnherrick /opt/homebrew/bin/limactl hostagent "
                f"--pidfile {instance}/ha.pid --socket {instance}/ha.sock "
                "--guestagent /opt/homebrew/share/lima/"
                "lima-guestagent.Linux-aarch64.gz colima-cascadia-r2"
            ),
            f"{ssh_pid} 1 johnherrick ssh: {instance}/ssh.sock [mux]",
        ],
        "observer_ancestors": [
            (
                "200 1 johnherrick /Users/johnherrick/cascadia/.venv/bin/python "
                "tools/r2_map_d0_dashboard_watch.py --watch --interval-seconds 5"
            )
        ],
        "launchd": [],
        "mounts": [],
        "active_unix_sockets": [
            f"limactl {usernet_pid} johnherrick 6u unix 0x1 0t0 {network}/user-v2_ep.sock",
            f"limactl {usernet_pid} johnherrick 7u unix 0x2 0t0 {network}/user-v2_qemu.sock",
            f"limactl {usernet_pid} johnherrick 8u unix 0x3 0t0 {network}/user-v2_fd.sock",
            f"limactl {usernet_pid} johnherrick 9u unix 0x4 0t0 ->0x5",
            f"limactl {hostagent_pid} johnherrick 6u unix 0x6 0t0 {instance}/ha.sock",
            f"ssh {ssh_pid} johnherrick 4u unix 0x7 0t0 {instance}/ssh.sock.token",
            f"ssh {ssh_pid} johnherrick 6u unix 0x8 0t0 {colima}/cascadia-r2/docker.sock",
            f"ssh {ssh_pid} johnherrick 7u unix 0x9 0t0 {colima}/cascadia-r2/containerd.sock",
            f"ssh {ssh_pid} johnherrick 8u unix 0xa 0t0 /tmp/lima-psl-127.0.0.1-53-42/sock",
        ],
        "active_tcp_listeners": [
            f"limactl {usernet_pid} johnherrick 13u IPv4 0x1 0t0 TCP 127.0.0.1:60000 (LISTEN)",
            f"limactl {hostagent_pid} johnherrick 11u IPv6 0x2 0t0 TCP *:53 (LISTEN)",
        ],
        "inactive": False,
    }
    assert home.endswith("johnherrick")
    return packet, activity


def test_positive_runtime_activity_accepts_only_exact_lima_roles(tmp_path: Path) -> None:
    packet, activity = _runtime_activity_fixture(tmp_path)
    result = _validate_positive_runtime_activity(packet, activity)
    assert result["host_runtime_tcp_listener_roles"] == [
        "hostagent-dns",
        "usernet-loopback",
    ]
    assert result["runtime_observer_ancestor_count"] == 1
    assert result["runtime_observer_process_count"] == 0


def test_positive_runtime_activity_accepts_exact_dashboard_observer_chain(
    tmp_path: Path,
) -> None:
    packet, activity = _runtime_activity_fixture(tmp_path)
    activity["processes"].extend(  # type: ignore[union-attr]
        [
            "201 200 johnherrick /opt/homebrew/bin/colima status --profile cascadia-r2",
            ("202 201 johnherrick /opt/homebrew/bin/limactl list colima-cascadia-r2 --json"),
            "203 201 johnherrick (limactl)",
        ]
    )
    result = _validate_positive_runtime_activity(packet, activity)
    assert result["runtime_process_count"] == 3
    assert result["runtime_observer_process_count"] == 3
    assert result["runtime_observer_processes_exact"] is True


def _inactive_activity_fixture(tmp_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    packet, activity = _runtime_activity_fixture(tmp_path)
    activity["processes"] = []
    activity["active_unix_sockets"] = []
    activity["active_tcp_listeners"] = []
    activity["inactive"] = True
    return packet, activity


def test_inactive_runtime_activity_accepts_exact_dashboard_watcher(
    tmp_path: Path,
) -> None:
    packet, activity = _inactive_activity_fixture(tmp_path)
    result = _validate_inactive_runtime_activity(packet, activity)
    assert result["inactive"] is True
    assert result["runtime_daemons_inactive"] is True
    assert result["runtime_observer_ancestor_count"] == 1
    assert result["runtime_observer_process_count"] == 0


def test_inactive_runtime_activity_accepts_exact_transient_observer_chain(
    tmp_path: Path,
) -> None:
    packet, activity = _inactive_activity_fixture(tmp_path)
    activity["processes"] = [
        "201 200 johnherrick /opt/homebrew/bin/colima status --profile cascadia-r2",
        ("202 201 johnherrick /opt/homebrew/bin/limactl list colima-cascadia-r2 --json"),
        "203 201 johnherrick (limactl)",
    ]
    activity["inactive"] = False
    result = _validate_inactive_runtime_activity(packet, activity)
    assert result["inactive"] is True
    assert result["runtime_observer_process_count"] == 3


@pytest.mark.parametrize(
    "field,value",
    (
        (
            "processes",
            ["299 1 johnherrick /opt/homebrew/bin/limactl unauthorized"],
        ),
        (
            "active_unix_sockets",
            ["limactl 299 johnherrick 6u unix 0x1 0t0 /tmp/unauthorized.sock"],
        ),
        (
            "active_tcp_listeners",
            ["limactl 299 johnherrick 6u IPv4 0x1 0t0 TCP *:9999 (LISTEN)"],
        ),
        ("launchd", ["299 0 com.example.colima"]),
        ("mounts", ["colima on /tmp/unauthorized"]),
    ),
)
def test_inactive_runtime_activity_rejects_non_observer_activity(
    tmp_path: Path,
    field: str,
    value: list[str],
) -> None:
    packet, activity = _inactive_activity_fixture(tmp_path)
    activity[field] = value
    with pytest.raises(D0Error, match=r"runtime (activity|process) remains"):
        _validate_inactive_runtime_activity(packet, activity)


def _guest_package(name: str) -> dict[str, str]:
    return {"name": name, "version": "1.0", "architecture": "arm64"}


def _guest_copyright(
    name: str,
    *,
    present: bool,
    requested_exists: bool | None = None,
) -> dict[str, object]:
    base = name.split(":", 1)[0]
    path = f"/usr/share/doc/{base}/copyright"
    return {
        "package": name,
        "requested": path,
        "resolved": path,
        "doc_dir": f"/usr/share/doc/{base}",
        "doc_dir_exists": present if requested_exists is None else requested_exists,
        "doc_dir_is_symlink": False,
        "doc_dir_symlink_target": None,
        "requested_exists": present if requested_exists is None else requested_exists,
        "requested_is_symlink": False,
        "requested_symlink_target": None,
        "exists": present,
        "present": present,
        "size": 8 if present else 0,
        "sha256": "a" * 64 if present else None,
    }


def _guest_package_license_fixture() -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    names = ("docker-ce", "containerd.io", "base-files")
    return (
        [_guest_package(name) for name in names],
        [
            _guest_copyright("docker-ce", present=False),
            _guest_copyright("containerd.io", present=True),
            _guest_copyright("base-files", present=True),
        ],
    )


def test_guest_license_inventory_accepts_honest_present_and_absent_records() -> None:
    packages, licenses = _guest_package_license_fixture()
    names, by_package, runtime_licenses = _validate_guest_package_license_inventory(
        packages,
        licenses,
    )
    assert names == {"docker-ce", "containerd.io", "base-files"}
    assert by_package["docker-ce"]["exists"] is False
    assert {item["package"] for item in runtime_licenses} == {
        "docker-ce",
        "containerd.io",
    }


@pytest.mark.parametrize(
    "mutation,error",
    (
        ("missing-record", "cardinality"),
        ("duplicate-record", "cardinality"),
        ("unknown-package", "package names or licenses"),
        ("false-absence", "absent copyright identity"),
        ("false-presence", "present copyright identity"),
        ("unmarked-alias", "path aliasing"),
    ),
)
def test_guest_license_inventory_rejects_incomplete_or_false_records(
    mutation: str,
    error: str,
) -> None:
    packages, licenses = _guest_package_license_fixture()
    if mutation == "missing-record":
        licenses.pop()
    elif mutation == "duplicate-record":
        licenses.append(dict(licenses[-1]))
    elif mutation == "unknown-package":
        licenses[-1]["package"] = "unknown"
    elif mutation == "false-absence":
        licenses[0]["exists"] = True
    elif mutation == "false-presence":
        licenses[1]["exists"] = False
    elif mutation == "unmarked-alias":
        licenses[-1]["resolved"] = "/usr/share/doc/other/copyright"
    with pytest.raises(D0Error, match=error):
        _validate_guest_package_license_inventory(packages, licenses)


def test_guest_license_inventory_accepts_explicit_symlink_alias() -> None:
    packages, licenses = _guest_package_license_fixture()
    licenses[-1].update(
        {
            "doc_dir_is_symlink": True,
            "doc_dir_symlink_target": "other",
            "resolved": "/usr/share/doc/other/copyright",
        }
    )
    _validate_guest_package_license_inventory(packages, licenses)


def _guest_network_json() -> str:
    return json.dumps(
        [
            {
                "ifname": "lo",
                "addr_info": [
                    {"family": "inet", "local": "127.0.0.1", "prefixlen": 8},
                    {"family": "inet6", "local": "::1", "prefixlen": 128},
                ],
            },
            {
                "ifname": "eth0",
                "addr_info": [
                    {
                        "family": "inet",
                        "local": "192.168.5.1",
                        "prefixlen": 24,
                        "scope": "global",
                    },
                    {
                        "family": "inet6",
                        "local": "fe80::5055:55ff:fe5d:c82d",
                        "prefixlen": 64,
                        "scope": "link",
                    },
                ],
            },
            {
                "ifname": "docker0",
                "addr_info": [{"family": "inet", "local": "172.17.0.1", "prefixlen": 16}],
            },
        ]
    )


def test_guest_listener_allowlist_is_bound_to_exact_interface_addresses() -> None:
    network = _guest_network_projection(_guest_network_json())
    listeners = _guest_listener_allowlist(
        "\n".join(
            [
                "LISTEN 0 4096 0.0.0.0:22 0.0.0.0:*",
                "LISTEN 0 32 127.0.0.1:53 0.0.0.0:*",
                "LISTEN 0 32 192.168.5.1:53 0.0.0.0:*",
                "LISTEN 0 32 [::1]:53 [::]:*",
                "LISTEN 0 32 [fe80::5055:55ff:fe5d:c82d]%eth0:53 [::]:*",
                "LISTEN 0 4096 [::]:22 [::]:*",
            ]
        ),
        network,
    )
    assert {item["endpoint"] for item in listeners} == {
        "0.0.0.0:22",
        "127.0.0.1:53",
        "192.168.5.1:53",
        "[::1]:53",
        "[fe80::5055:55ff:fe5d:c82d]%eth0:53",
        "[::]:22",
    }


@pytest.mark.parametrize("endpoint", ("0.0.0.0:53", "172.17.0.1:53", "*:53"))
def test_guest_listener_allowlist_rejects_unbound_dns_endpoints(endpoint: str) -> None:
    network = _guest_network_projection(_guest_network_json())
    with pytest.raises(D0Error, match="not allowlisted"):
        _guest_listener_allowlist(f"LISTEN 0 32 {endpoint} 0.0.0.0:*", network)


def test_guest_network_projection_rejects_extra_interface() -> None:
    value = json.loads(_guest_network_json())
    value.append({"ifname": "eth1", "addr_info": []})
    with pytest.raises(D0Error, match="interface inventory"):
        _guest_network_projection(json.dumps(value))


def test_guest_network_projection_accepts_valid_warm_docker_bridge() -> None:
    value = json.loads(_guest_network_json())
    docker0 = next(item for item in value if item["ifname"] == "docker0")
    docker0["addr_info"].append(
        {
            "family": "inet6",
            "local": "fe80::a070:b2ff:fe1d:d6ed",
            "prefixlen": 64,
            "scope": "link",
        }
    )
    projection = _guest_network_projection(json.dumps(value))
    assert projection["docker_network_lifecycle"] == "warm"
    assert projection["docker0_ipv6_link_local"] == "fe80::a070:b2ff:fe1d:d6ed"


def _guest_binfmt_json(
    *,
    name: str = "python3.12",
    interpreter: str = "/usr/bin/python3.12",
) -> str:
    return json.dumps(
        {
            "control": ["register", "status"],
            "handlers": {
                name: "\n".join(
                    [
                        "enabled",
                        f"interpreter {interpreter}",
                        "flags: ",
                        "offset 0",
                        "magic cb0d0d0a",
                        "",
                    ]
                )
            },
        }
    )


def test_guest_binfmt_inventory_accepts_only_native_python_handler() -> None:
    result = _validate_guest_binfmt_inventory(_guest_binfmt_json())
    assert result["handlers"]["python3.12"] == {
        "interpreter": "/usr/bin/python3.12",
        "magic": "cb0d0d0a",
        "enabled": True,
        "foreign_architecture": False,
    }


@pytest.mark.parametrize(
    "value,error",
    (
        (_guest_binfmt_json(name="rosetta"), "unauthorized"),
        (_guest_binfmt_json(name="qemu-aarch64"), "unauthorized"),
        (_guest_binfmt_json(interpreter="/usr/bin/qemu-aarch64-static"), "native Python"),
        (
            json.dumps({"control": ["register"], "handlers": {}}),
            "inventory differs",
        ),
    ),
)
def test_guest_binfmt_inventory_rejects_foreign_or_drifted_handlers(
    value: str,
    error: str,
) -> None:
    with pytest.raises(D0Error, match=error):
        _validate_guest_binfmt_inventory(value)


def test_guest_nested_virtualization_requires_module_without_usable_device() -> None:
    result = _validate_guest_nested_virtualization(
        json.dumps({"dev_kvm": False, "modules": ["/sys/module/kvm"]})
    )
    assert result == {
        "usable_kvm_device_present": False,
        "kernel_kvm_module_present": True,
        "architecture_specific_kvm_modules": [],
        "nested_virtualization_enabled": False,
        "status": "pass",
    }


@pytest.mark.parametrize(
    "value",
    (
        {"dev_kvm": True, "modules": ["/sys/module/kvm"]},
        {"dev_kvm": False, "modules": []},
        {
            "dev_kvm": False,
            "modules": ["/sys/module/kvm", "/sys/module/kvm_intel"],
        },
    ),
)
def test_guest_nested_virtualization_rejects_capability_or_identity_drift(
    value: dict[str, object],
) -> None:
    with pytest.raises(D0Error, match="capability differs"):
        _validate_guest_nested_virtualization(json.dumps(value))


@pytest.mark.parametrize(
    "process",
    (
        "201 200 johnherrick /opt/homebrew/bin/colima start --profile cascadia-r2",
        "201 200 johnherrick /opt/homebrew/bin/docker run alpine:3.22.1",
        "201 1 johnherrick /opt/homebrew/bin/colima status --profile cascadia-r2",
    ),
)
def test_positive_runtime_activity_rejects_non_observer_runtime_clients(
    tmp_path: Path, process: str
) -> None:
    packet, activity = _runtime_activity_fixture(tmp_path)
    activity["processes"].append(process)  # type: ignore[union-attr]
    with pytest.raises(D0Error, match="runtime process set differs"):
        _validate_positive_runtime_activity(packet, activity)


def test_positive_runtime_activity_rejects_unbound_observer_child(tmp_path: Path) -> None:
    packet, activity = _runtime_activity_fixture(tmp_path)
    activity["processes"].append(  # type: ignore[union-attr]
        "202 200 johnherrick /opt/homebrew/bin/limactl list colima-cascadia-r2 --json"
    )
    with pytest.raises(D0Error, match="runtime process set differs"):
        _validate_positive_runtime_activity(packet, activity)


@pytest.mark.parametrize(
    "watcher",
    (
        (
            "200 1 johnherrick /Users/johnherrick/cascadia/.venv/bin/python "
            "tools/r2_map_d0_dashboard_watch.py --watch --interval-seconds 10"
        ),
        (
            "200 1 intruder /Users/johnherrick/cascadia/.venv/bin/python "
            "tools/r2_map_d0_dashboard_watch.py --watch --interval-seconds 5"
        ),
        (
            "200 99 johnherrick /Users/johnherrick/cascadia/.venv/bin/python "
            "tools/r2_map_d0_dashboard_watch.py --watch --interval-seconds 5"
        ),
    ),
)
def test_positive_runtime_activity_rejects_dashboard_ancestry_drift(
    tmp_path: Path, watcher: str
) -> None:
    packet, activity = _runtime_activity_fixture(tmp_path)
    activity["observer_ancestors"] = [watcher]
    with pytest.raises(D0Error, match="dashboard observer ancestry"):
        _validate_positive_runtime_activity(packet, activity)


def test_positive_runtime_activity_rejects_wrong_profile_observer(tmp_path: Path) -> None:
    packet, activity = _runtime_activity_fixture(tmp_path)
    activity["processes"].append(  # type: ignore[union-attr]
        "201 200 johnherrick /opt/homebrew/bin/colima status --profile default"
    )
    with pytest.raises(D0Error, match="runtime process set differs"):
        _validate_positive_runtime_activity(packet, activity)


def _activity_sampler(items: list[dict[str, object]]):
    iterator = iter(items)
    return lambda: next(iterator)


def _remap_activity_pids(activity: dict[str, object]) -> dict[str, object]:
    encoded = json.dumps(activity)
    for old, new in (("101", "111"), ("102", "112"), ("103", "113")):
        encoded = encoded.replace(old, new)
    return json.loads(encoded)


def test_activity_convergence_accepts_transient_invalid_then_two_stable(
    tmp_path: Path,
) -> None:
    packet, stable = _runtime_activity_fixture(tmp_path)
    transient = json.loads(json.dumps(stable))
    transient["processes"].append(  # type: ignore[union-attr]
        "299 1 johnherrick /opt/homebrew/bin/limactl unauthorized"
    )
    result = _converged_positive_runtime_activity(
        packet,
        sampler=_activity_sampler([transient, stable, stable]),
        deadline_seconds=100,
        interval_seconds=0,
        max_samples=3,
        sleeper=lambda _seconds: None,
    )
    convergence = result["stability_convergence"]
    assert convergence["attempt_count"] == 3
    assert [item["status"] for item in convergence["sample_sequence"]] == [
        "invalid",
        "valid",
        "valid",
    ]


def test_activity_convergence_rejects_persistent_extra_with_diagnostics(
    tmp_path: Path,
) -> None:
    packet, activity = _runtime_activity_fixture(tmp_path)
    activity["processes"].append(  # type: ignore[union-attr]
        "299 1 johnherrick /opt/homebrew/bin/limactl unauthorized"
    )
    with pytest.raises(D0Error, match="failed bounded convergence") as captured:
        _converged_positive_runtime_activity(
            packet,
            sampler=_activity_sampler([activity, activity, activity]),
            deadline_seconds=100,
            interval_seconds=0,
            max_samples=3,
            sleeper=lambda _seconds: None,
        )
    message = str(captured.value)
    assert '"sample_sequence"' in message
    assert "limactl unauthorized" in message
    assert '"rows_sha256"' in message


def test_activity_convergence_rejects_alternating_valid_instability(tmp_path: Path) -> None:
    packet, first = _runtime_activity_fixture(tmp_path)
    second = _remap_activity_pids(first)
    with pytest.raises(D0Error, match="failed bounded convergence"):
        _converged_positive_runtime_activity(
            packet,
            sampler=_activity_sampler([first, second, first, second]),
            deadline_seconds=100,
            interval_seconds=0,
            max_samples=4,
            sleeper=lambda _seconds: None,
        )


def test_activity_convergence_accepts_exact_dashboard_observer_race(tmp_path: Path) -> None:
    packet, without_observer = _runtime_activity_fixture(tmp_path)
    with_observer = json.loads(json.dumps(without_observer))
    with_observer["processes"].extend(  # type: ignore[union-attr]
        [
            "201 200 johnherrick /opt/homebrew/bin/colima status --profile cascadia-r2",
            ("202 201 johnherrick /opt/homebrew/bin/limactl list colima-cascadia-r2 --json"),
        ]
    )
    result = _converged_positive_runtime_activity(
        packet,
        sampler=_activity_sampler([without_observer, with_observer]),
        deadline_seconds=100,
        interval_seconds=0,
        max_samples=2,
        sleeper=lambda _seconds: None,
    )
    sequence = result["stability_convergence"]["sample_sequence"]
    assert sequence[0]["stable_projection_sha256"] == sequence[1]["stable_projection_sha256"]


def test_positive_runtime_activity_rejects_extra_listener_or_socket(tmp_path: Path) -> None:
    packet, activity = _runtime_activity_fixture(tmp_path)
    activity["active_tcp_listeners"].append(  # type: ignore[union-attr]
        "limactl 102 johnherrick 12u IPv6 0x3 0t0 TCP *:8080 (LISTEN)"
    )
    with pytest.raises(D0Error, match="outside the exact Lima allowlist"):
        _validate_positive_runtime_activity(packet, activity)

    packet, activity = _runtime_activity_fixture(tmp_path / "socket")
    activity["active_unix_sockets"].append(  # type: ignore[union-attr]
        "ssh 103 johnherrick 9u unix 0xb 0t0 /tmp/unauthorized.sock"
    )
    with pytest.raises(D0Error, match="Unix-socket path differs"):
        _validate_positive_runtime_activity(packet, activity)


def test_positive_runtime_activity_rejects_non_loopback_or_process_drift(
    tmp_path: Path,
) -> None:
    packet, activity = _runtime_activity_fixture(tmp_path)
    activity["active_tcp_listeners"][0] = (  # type: ignore[index]
        "limactl 101 johnherrick 13u IPv4 0x1 0t0 TCP *:60000 (LISTEN)"
    )
    with pytest.raises(D0Error, match="outside the exact Lima allowlist"):
        _validate_positive_runtime_activity(packet, activity)

    packet, activity = _runtime_activity_fixture(tmp_path / "process")
    activity["processes"].append(  # type: ignore[union-attr]
        "104 1 johnherrick /opt/homebrew/bin/limactl unauthorized"
    )
    with pytest.raises(D0Error, match="runtime process set differs"):
        _validate_positive_runtime_activity(packet, activity)


@pytest.mark.parametrize("field", ("launchd", "mounts"))
def test_positive_runtime_activity_rejects_startup_or_mount_drift(
    tmp_path: Path, field: str
) -> None:
    packet, activity = _runtime_activity_fixture(tmp_path)
    activity[field] = ["unauthorized"]
    with pytest.raises(D0Error, match="startup-item or host-mount"):
        _validate_positive_runtime_activity(packet, activity)


def test_containerd_image_store_normalizes_duplicate_cli_rows() -> None:
    row = {
        "Repository": "alpine",
        "Tag": "3.22.1-d0",
        "ID": "sha256:" + "4" * 64,
    }
    result = _require_single_logical_image(
        [row, dict(row)],
        repository="alpine",
        tag="3.22.1-d0",
        image_id="sha256:" + "4" * 64,
    )
    assert result["logical_image_count"] == 1
    assert result["duplicate_cli_rows_normalized"] == 1


def test_scanner_generator_is_local_tag_plus_immutable_manifest_digest() -> None:
    digest = "sha256:" + "4" * 64
    assert _scanner_local_generator_reference({"manifest_digest": digest}) == (
        f"{SCANNER_LOCAL_REFERENCE}@{digest}"
    )
    assert SCANNER_LOCAL_REFERENCE.startswith("localhost/")


@pytest.mark.parametrize(
    "digest",
    (None, "", "sha256:short", "sha512:" + "4" * 64, "sha256:" + "G" * 64),
)
def test_scanner_generator_rejects_noncanonical_manifest_digest(digest: object) -> None:
    with pytest.raises(D0Error, match="scanner manifest digest"):
        _scanner_local_generator_reference({"manifest_digest": digest})


def test_scanner_store_requires_source_and_local_resolver_tags_only() -> None:
    image_id = "sha256:" + "4" * 64
    source = "docker/buildkit-syft-scanner:stable-1"
    rows = [
        {"Repository": "docker/buildkit-syft-scanner", "Tag": "stable-1", "ID": image_id},
        {
            "Repository": "localhost/cascadia-r2-buildkit-syft-scanner",
            "Tag": "stable-1",
            "ID": image_id,
        },
    ]
    result = _require_exact_logical_image_references(
        rows,
        references=(source, SCANNER_LOCAL_REFERENCE),
        image_id=image_id,
    )
    assert result["logical_reference_count"] == 2
    assert result["logical_image_count"] == 1

    with pytest.raises(D0Error, match="reference set differs"):
        _require_exact_logical_image_references(
            [*rows, {"Repository": "extra", "Tag": "latest", "ID": image_id}],
            references=(source, SCANNER_LOCAL_REFERENCE),
            image_id=image_id,
        )


def test_loopback_scanner_registry_is_digest_qualified_and_read_only() -> None:
    manifest = "sha256:" + "4" * 64
    descriptor = _scanner_registry_descriptor(
        {
            "manifest_digest": manifest,
            "manifest_size": 481,
            "config_digest": "sha256:" + "5" * 64,
            "config_size": 801,
            "layer_digest": "sha256:" + "6" * 64,
            "layer_size": 43_158_689,
        }
    )
    assert descriptor["host"] == SCANNER_REGISTRY_HOST == "127.0.0.1"
    assert descriptor["resolver_host"] == SCANNER_REGISTRY_RESOLVER_HOST == "localhost"
    assert descriptor["port"] == SCANNER_REGISTRY_PORT
    assert descriptor["repository"] == SCANNER_REGISTRY_REPOSITORY
    assert descriptor["generator_reference"] == (
        f"localhost:{SCANNER_REGISTRY_PORT}/{SCANNER_REGISTRY_REPOSITORY}@{manifest}"
    )
    for name, script in (
        ("prepare", SCANNER_REGISTRY_PREPARE_SCRIPT),
        ("trust-install", SCANNER_REGISTRY_TRUST_INSTALL_SCRIPT),
        ("trust-cleanup", SCANNER_REGISTRY_TRUST_CLEANUP_SCRIPT),
        ("tls-client", SCANNER_REGISTRY_TLS_CLIENT_SCRIPT),
        ("socket-sampler-launch", SCANNER_SOCKET_SAMPLER_LAUNCH_SCRIPT),
        ("socket-sampler-stop", SCANNER_SOCKET_SAMPLER_STOP_SCRIPT),
        ("server", SCANNER_REGISTRY_SERVER_SCRIPT),
        ("cleanup", SCANNER_REGISTRY_CLEANUP_SCRIPT),
    ):
        compile(script, name, "exec")
    assert "def do_GET(self): self.dispatch(True)" in SCANNER_REGISTRY_SERVER_SCRIPT
    assert "def do_HEAD(self): self.dispatch(False)" in SCANNER_REGISTRY_SERVER_SCRIPT
    assert "def do_PUT(self): self.reject()" in SCANNER_REGISTRY_SERVER_SCRIPT
    assert "def do_DELETE(self): self.reject()" in SCANNER_REGISTRY_SERVER_SCRIPT
    assert "ssl.PROTOCOL_TLS_SERVER" in SCANNER_REGISTRY_SERVER_SCRIPT
    assert "ssl.TLSVersion.TLSv1_2" in SCANNER_REGISTRY_SERVER_SCRIPT
    assert "Docker-Content-Digest" in SCANNER_REGISTRY_SERVER_SCRIPT
    assert SCANNER_REGISTRY_PROCESS_MARKER in SCANNER_REGISTRY_SERVER_SCRIPT
    assert "marker in parts and root.encode() in parts" in SCANNER_REGISTRY_CLEANUP_SCRIPT
    assert "candidate not in ancestors" in SCANNER_REGISTRY_CLEANUP_SCRIPT
    assert "['/usr/bin/ss','-Hntoape']" in SCANNER_SOCKET_SAMPLER_LAUNCH_SCRIPT
    assert "['/usr/bin/ss','-Hntop']" not in SCANNER_SOCKET_SAMPLER_LAUNCH_SCRIPT
    assert "os.O_EXCL|os.O_NOFOLLOW,0o600" in SCANNER_REGISTRY_PREPARE_SCRIPT
    assert str(SCANNER_EXPORT_SIZE) in SCANNER_REGISTRY_PREPARE_SCRIPT
    assert SCANNER_EXPORT_SHA256 in SCANNER_REGISTRY_PREPARE_SCRIPT
    assert "'blobs':('directory',0,0o755,None)" in SCANNER_REGISTRY_PREPARE_SCRIPT
    assert "'blobs/sha256':('directory',0,0o755,None)" in SCANNER_REGISTRY_PREPARE_SCRIPT
    assert "seen!=set(expected)" in SCANNER_REGISTRY_PREPARE_SCRIPT
    assert "member.uid!=0 or member.gid!=0" in SCANNER_REGISTRY_PREPARE_SCRIPT
    assert "while view:" in SCANNER_REGISTRY_PREPARE_SCRIPT


def test_loopback_scanner_registry_freezes_bounded_tls_identity() -> None:
    assert SCANNER_REGISTRY_CA_PATH == (
        "/usr/local/share/ca-certificates/cascadia-r2-d0-loopback-registry.crt"
    )
    assert SCANNER_REGISTRY_CA_SHA256 == (
        "ea3dbcc664345bcd06fb268613984814cb906a765b7c84f3c360a5808ac57d62"
    )
    assert SCANNER_REGISTRY_SERVER_CERT_SHA256 == (
        "22553d21cdec62c66e56ffbac5038b8d953f05c116c538f1353ef9633ce9ec5d"
    )
    assert SCANNER_REGISTRY_SERVER_KEY_SHA256 == (
        "d35c78a4324b7ff9e2a2e417f1266537f7e655266c7c6cb0e73d3b7c82a1f5b3"
    )
    assert SCANNER_REGISTRY_SERVER_CERT_DER_SHA256 == (
        "2962586a27879ae7b1797363c3334331aa3ca56cf49ee2d538fd30602bde05cd"
    )
    assert b"BEGIN PRIVATE KEY" not in SCANNER_REGISTRY_CA_CERT
    assert b"BEGIN CERTIFICATE" in SCANNER_REGISTRY_CA_CERT
    assert b"BEGIN CERTIFICATE" in SCANNER_REGISTRY_SERVER_CERT
    assert b"BEGIN PRIVATE KEY" in SCANNER_REGISTRY_SERVER_KEY
    assert "update-ca-certificates" in SCANNER_REGISTRY_TRUST_INSTALL_SCRIPT
    assert "rows!=baseline" in SCANNER_REGISTRY_TRUST_CLEANUP_SCRIPT
    assert "ssl.create_default_context()" in SCANNER_REGISTRY_TLS_CLIENT_SCRIPT


def test_scanner_resolver_tls_context_is_deterministic_and_nonexecuting() -> None:
    reference = "localhost:5047/cascadia/buildkit-syft-scanner@sha256:" + "4" * 64
    first, first_receipt = _scanner_resolver_context(reference)
    second, second_receipt = _scanner_resolver_context(reference)
    assert first == second
    assert first_receipt == second_receipt
    assert first_receipt["archive_bytes"] == 10_240
    with tarfile.open(fileobj=io.BytesIO(first), mode="r:") as archive:
        members = archive.getmembers()
        assert [item.name for item in members] == ["Dockerfile"]
        stream = archive.extractfile(members[0])
        assert stream is not None
        assert stream.read() == f"FROM {reference}\n".encode("ascii")


def test_bounded_network_state_diff_reports_exact_paths() -> None:
    before = {
        "addresses": [
            {
                "addr_info": [
                    {
                        "local": "192.168.5.1",
                        "preferred_life_time": 1831,
                        "valid_life_time": 1831,
                    }
                ],
                "ifname": "eth0",
            }
        ],
        "ruleset": {"nftables": []},
    }
    after = {
        "addresses": [
            {
                "addr_info": [
                    {
                        "local": "192.168.5.1",
                        "preferred_life_time": 1829,
                        "valid_life_time": 1829,
                    }
                ],
                "ifname": "eth0",
            }
        ],
        "ruleset": {"nftables": []},
    }
    assert _bounded_state_differences(before, after) == [
        {
            "after": 1829,
            "before": 1831,
            "path": "$.addresses[0].addr_info[0].preferred_life_time",
            "status": "changed",
        },
        {
            "after": 1829,
            "before": 1831,
            "path": "$.addresses[0].addr_info[0].valid_life_time",
            "status": "changed",
        },
    ]


def _lease_snapshot(*, finite: int, local: str = "192.168.5.1") -> dict[str, object]:
    return {
        "captured_monotonic_ns": 1_000_000_000,
        "lease_timers": [
            {
                "family": "inet",
                "ifindex": 1,
                "ifname": "lo",
                "local": "127.0.0.1",
                "prefixlen": 8,
                "preferred_life_time": 0xFFFFFFFF,
                "valid_life_time": 0xFFFFFFFF,
            },
            {
                "family": "inet",
                "ifindex": 2,
                "ifname": "eth0",
                "local": local,
                "prefixlen": 24,
                "preferred_life_time": finite,
                "valid_life_time": finite,
            },
        ],
    }


def test_network_lease_transition_accepts_only_bounded_countdown() -> None:
    before = _lease_snapshot(finite=1910)
    after = _lease_snapshot(finite=1908)
    after["captured_monotonic_ns"] = 3_000_000_000
    assert _validate_network_lease_transition(before, after) == {
        "elapsed_milliseconds": 2000,
        "finite_timer_count": 2,
        "identity_count": 2,
        "maximum_allowed_decrement": 7,
        "maximum_observed_decrement": 2,
        "status": "pass",
    }


@pytest.mark.parametrize("mutation", ["addition", "removal", "increase", "identity"])
def test_network_lease_transition_rejects_shape_reset_or_identity_drift(
    mutation: str,
) -> None:
    before = _lease_snapshot(finite=1910)
    after = _lease_snapshot(finite=1908)
    after["captured_monotonic_ns"] = 3_000_000_000
    timers = after["lease_timers"]
    assert isinstance(timers, list)
    if mutation == "addition":
        timers.append(dict(timers[-1], local="192.168.5.2"))
    elif mutation == "removal":
        timers.pop()
    elif mutation == "increase":
        timers[-1]["preferred_life_time"] = 1911
    else:
        timers[-1]["local"] = "192.168.5.2"
    with pytest.raises(D0Error, match="network lease"):
        _validate_network_lease_transition(before, after)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("manifest_size", None),
        ("manifest_size", True),
        ("config_size", 0),
        ("layer_size", -1),
        ("config_digest", "sha256:short"),
        ("layer_digest", None),
    ],
)
def test_loopback_scanner_registry_requires_complete_typed_blob_schema(
    field: str, value: object
) -> None:
    verified: dict[str, object] = {
        "manifest_digest": "sha256:" + "4" * 64,
        "manifest_size": 481,
        "config_digest": "sha256:" + "5" * 64,
        "config_size": 801,
        "layer_digest": "sha256:" + "6" * 64,
        "layer_size": 43_158_689,
    }
    verified[field] = value
    with pytest.raises(D0Error, match="registry input schema"):
        _scanner_registry_descriptor(verified)


def test_loopback_scanner_registry_cleanup_rejects_tampered_or_extra_requests() -> None:
    descriptor = {
        "host": "127.0.0.1",
        "port": SCANNER_REGISTRY_PORT,
    }
    valid = {
        "host": "127.0.0.1",
        "listener_absent": True,
        "orphan_pids": [],
        "pid": 123,
        "port": SCANNER_REGISTRY_PORT,
        "record_count": 2,
        "record_sha256": "4" * 64,
        "requests_valid": True,
        "root_absent": True,
        "served_paths": ["/v2/", "/v2/exact/manifests/sha256:abc"],
        "status": "clean",
        "unexpected_requests": [],
    }
    assert _validate_scanner_registry_cleanup(valid, descriptor, require_requests=True) == valid
    abort_cleanup = {
        **valid,
        "pid": None,
        "record_count": 0,
        "requests_valid": False,
        "served_paths": [],
    }
    assert (
        _validate_scanner_registry_cleanup(abort_cleanup, descriptor, require_requests=False)
        == abort_cleanup
    )
    with pytest.raises(D0Error, match="registry"):
        _validate_scanner_registry_cleanup(abort_cleanup, descriptor, require_requests=True)
    for invalid_abort in (
        {**abort_cleanup, "record_count": 1},
        {**abort_cleanup, "served_paths": ["/v2/"]},
        {**abort_cleanup, "unexpected_requests": [{"method": "GET"}]},
        {**abort_cleanup, "listener_absent": False},
        {**abort_cleanup, "orphan_pids": [999]},
    ):
        with pytest.raises(D0Error, match=r"registry|pre-launch"):
            _validate_scanner_registry_cleanup(invalid_abort, descriptor, require_requests=False)
    for mutation in (
        {**valid, "requests_valid": False},
        {**valid, "unexpected_requests": [{"method": "PUT"}]},
        {**valid, "record_count": 0},
        {**valid, "record_sha256": "tampered"},
        {**valid, "port": SCANNER_REGISTRY_PORT + 1},
    ):
        with pytest.raises(D0Error, match="registry"):
            _validate_scanner_registry_cleanup(mutation, descriptor, require_requests=True)


@pytest.mark.parametrize(
    "rows",
    (
        [
            {"Repository": "alpine", "Tag": "3.22.1-d0", "ID": "sha256:" + "4" * 64},
            {"Repository": "extra", "Tag": "latest", "ID": "sha256:" + "5" * 64},
        ],
        [{"Repository": "alpine", "Tag": "3.22.1-d0", "ID": "sha256:" + "6" * 64}],
    ),
)
def test_containerd_image_store_rejects_extra_or_wrong_digest(
    rows: list[dict[str, str]],
) -> None:
    with pytest.raises(D0Error, match="logical image-store identity differs"):
        _require_single_logical_image(
            rows,
            repository="alpine",
            tag="3.22.1-d0",
            image_id="sha256:" + "4" * 64,
        )


def _volume(name: str, run_id: str = "run") -> dict[str, object]:
    return {
        "Name": name,
        "Driver": "local",
        "Labels": {"cascadia.r2-map.d0.run": run_id},
        "Options": None,
        "Mountpoint": f"/var/lib/docker/volumes/{name}/_data",
    }


def test_volume_inspects_normalize_order_and_require_exact_names() -> None:
    result = _validate_smoke_volume_inspects(
        [_volume("output"), _volume("input")],
        expected_names=("input", "output"),
        run_id="run",
    )
    assert [item["Name"] for item in result] == ["input", "output"]


@pytest.mark.parametrize(
    "volumes",
    (
        [_volume("input")],
        [_volume("input"), _volume("output"), _volume("extra")],
        [_volume("input"), _volume("wrong")],
    ),
)
def test_volume_inspects_reject_missing_extra_or_wrong_name(
    volumes: list[dict[str, object]],
) -> None:
    with pytest.raises(D0Error, match="inspected-volume name set differs"):
        _validate_smoke_volume_inspects(
            volumes,
            expected_names=("input", "output"),
            run_id="run",
        )


def test_command_runner_nonzero_error_is_attributable_and_bounded() -> None:
    runner = CommandRunner(
        {"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C"},
        timeout_seconds=5,
        output_max_bytes=4096,
    )
    with pytest.raises(D0Error) as captured:
        runner.run(
            [
                "/usr/bin/python3",
                "-I",
                "-S",
                "-B",
                "-c",
                "import sys;sys.stderr.write('bounded-detail');raise SystemExit(7)",
            ]
        )
    message = str(captured.value)
    assert '"/usr/bin/python3"' in message
    assert "returncode=7" in message
    assert "stderr_sha256=" in message
    assert "bounded-detail" in message


def test_john2_progressive_acquisition_nullability_has_a_legal_prefix() -> None:
    producers = (
        "render-runtime-supply",
        "acquire-scanner",
        "acquire-smoke",
        "render-runtime-supply",
    )
    operations = (
        "acquire-core",
        "acquire-homebrew-artifacts",
        "acquire-scanner",
        "acquire-smoke",
        "render-runtime-supply",
        "render-probe-context",
    )
    for operation_index, operation in enumerate(operations):
        for producer in producers:
            producer_index = operations.index(producer)
            assert _john2_artifact_pending("john2", "install", [operation], producer) is (
                operation_index <= producer_index
            )
    assert not _john2_artifact_pending(
        "john1", "install", ["materialize-runtime-supply"], "render-runtime-supply"
    )


def _dynamic_identity(field: str, paths: dict[str, str]) -> dict[str, object]:
    if field == "smoke_oci":
        return {
            "name": "alpine-3.22.1-arm64-oci",
            "size": 4_200_000,
            "sha256": "5" * 64,
            "source": paths["smoke_oci"],
        }
    if field == "scanner_oci":
        return {
            "name": "buildkit-syft-scanner-v1.11.0-arm64-oci",
            "size": 44_000_000,
            "sha256": "6" * 64,
            "source": paths["scanner_oci"],
        }
    if field == "homebrew_closure":
        return {
            "name": "homebrew-closure-arm64-tahoe-v1",
            "size": 1_000_000,
            "sha256": "7" * 64,
            "source": paths["homebrew_closure"],
        }
    if field == "runtime_supply":
        return {
            "name": "worker-runtime-supply-v1",
            "size": 400_000_000,
            "sha256": "a" * 64,
            "source": paths["runtime_supply"],
        }
    raise AssertionError(field)


def _expected_dynamic_presence(host: str, phase: str, operation: str) -> dict[str, bool]:
    fields = ("smoke_oci", "scanner_oci", "homebrew_closure", "runtime_supply")
    if phase == "preflight":
        return {field: False for field in fields}
    if host == "john2":
        if phase != "install":
            return {field: True for field in fields}
        order = (
            "acquire-core",
            "acquire-homebrew-artifacts",
            "acquire-scanner",
            "acquire-smoke",
            "render-runtime-supply",
            "render-probe-context",
            "install-runtime",
        )
        index = order.index(operation)
        return {
            "smoke_oci": index > order.index("acquire-smoke"),
            "scanner_oci": index > order.index("acquire-scanner"),
            "homebrew_closure": index > order.index("render-runtime-supply"),
            "runtime_supply": index > order.index("render-runtime-supply"),
        }
    if phase == "install" and operation == "materialize-runtime-supply":
        return {
            "smoke_oci": False,
            "scanner_oci": False,
            "homebrew_closure": False,
            "runtime_supply": True,
        }
    return {
        "smoke_oci": True,
        "scanner_oci": False,
        "homebrew_closure": True,
        "runtime_supply": True,
    }


def test_every_transaction_node_has_exact_dynamic_artifact_prefix_state() -> None:
    """Validate both cycles and reject every early or missing derived output."""

    for node in ordered_transactions():
        host = node["host"]
        phase = node["phase"]
        operation = node["operation"]
        operations = (
            ["buildkit-probe", "verify-runtime"]
            if host == "john2" and phase == "verify"
            else [operation]
        )
        specification = work_spec(
            host,
            phase,
            cycle_id=node["cycle_id"],
            operations=operations,
        )
        expected = _expected_dynamic_presence(host, phase, operation)
        for field, present in expected.items():
            specification["artifacts"][field] = (
                _dynamic_identity(field, specification["paths"]) if present else None
            )
        render_document(specification, kind="work")

        for field, present in expected.items():
            invalid = json.loads(json.dumps(specification))
            invalid["artifacts"][field] = (
                None if present else _dynamic_identity(field, invalid["paths"])
            )
            with pytest.raises(D0Error, match="production boundary"):
                render_document(invalid, kind="work")
