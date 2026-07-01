from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cascadia_mlx.r2_map_apfs_workspace as workspace_contract
import pytest
from cascadia_mlx.r2_map_apfs_lifecycle import (
    ApfsLifecycleError,
    ApfsLifecyclePaths,
    ApfsWorkspaceLifecycle,
    CommandResult,
    build_apfs_bootstrap_safety_receipt,
    build_host_safety_receipt,
    host_dashboard_receipt,
    validate_apfs_bootstrap_safety,
    validate_host_safety,
    write_apfs_bootstrap_safety,
    write_host_safety,
)
from cascadia_mlx.r2_map_apfs_workspace import (
    APFS_WORKSPACE_BUDGET_BYTES,
    APFS_WORKSPACE_CAPACITY_BYTES,
    APFS_WORKSPACE_MIN_BACKING_FREE_BYTES,
    APFS_WORKSPACE_SCHEMA,
    ApfsWorkspaceSpec,
)


class FakeMachine:
    def __init__(self, spec: ApfsWorkspaceSpec):
        self.spec = spec
        self.mounted = False
        self.attached = False
        self.commands: list[tuple[str, ...]] = []
        self.contract = {
            "backing_bundle": str(spec.backing_bundle),
            "mountpoint": str(spec.mountpoint),
            "volume_name": spec.volume_name,
            "volume_uuid": "01234567-89AB-CDEF-0123-456789ABCDEF",
            "filesystem": "apfs",
            "capacity_bytes": APFS_WORKSPACE_CAPACITY_BYTES,
            "free_bytes": APFS_WORKSPACE_BUDGET_BYTES,
            "owner_uid": os.getuid(),
            "owner_gid": os.getgid(),
            "mode": "0700",
            "read_only": False,
            "symlink_components": [],
            "backing_free_bytes": APFS_WORKSPACE_MIN_BACKING_FREE_BYTES,
        }

    def run(self, argv: tuple[str, ...]) -> CommandResult:
        self.commands.append(argv)
        if argv[1] == "create":
            self.spec.backing_bundle.mkdir(parents=True)
        elif argv[1] == "attach":
            self.mounted = self.attached = True
        elif argv[0].endswith("diskutil") and argv[1] == "unmount":
            self.mounted = False
            for path in (self.spec.marker, *self.spec.work_paths()):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
        elif argv[1] == "detach":
            self.mounted = self.attached = False
        return CommandResult(argv=argv, returncode=0)

    def observe(self, _spec: ApfsWorkspaceSpec) -> dict[str, Any] | None:
        if not self.mounted:
            return None
        return {"device": "/dev/disk42s1", "contract": dict(self.contract)}


def _fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ApfsWorkspaceLifecycle, FakeMachine]:
    campaign = tmp_path / "John_1/cascadia-cluster/r2-map-v1"
    campaign.mkdir(parents=True)
    mount_namespace = tmp_path / "Volumes/CascadiaR2MapV1"
    mount_namespace.parent.mkdir()
    monkeypatch.setattr(workspace_contract, "APFS_WORKSPACE_MOUNTPOINT", mount_namespace)
    spec = ApfsWorkspaceSpec.historical_fixture(
        campaign_root=campaign,
        mountpoint=mount_namespace,
    )
    spec.mountpoint.mkdir(parents=True)
    paths = ApfsLifecyclePaths.for_spec(spec)
    lifecycle = ApfsWorkspaceLifecycle(
        spec,
        paths=paths,
        mounted_checker=lambda _path: machine.mounted,
        current_uid=os.getuid(),
        current_gid=os.getgid(),
    )
    machine = FakeMachine(spec)
    safe = build_host_safety_receipt(
        status="safe",
        observed_unix_ms=1,
        syspolicyd_rss_bytes=100,
        system_swap_baseline_bytes=200,
        system_swap_observed_bytes=200,
        quiet_window_passed=True,
        detail="60-second quiet window passed",
    )
    write_host_safety(paths.host_safety, safe)
    return lifecycle, machine


def test_plan_is_exact_and_has_no_filesystem_or_runner_side_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lifecycle, machine = _fixture(tmp_path, monkeypatch)
    plan = lifecycle.plan()
    assert plan["create_argv"][0:2] == ["/usr/bin/hdiutil", "create"]
    assert plan["attach_argv"][0:2] == ["/usr/bin/hdiutil", "attach"]
    assert plan["backing_bundle"].endswith("storage/r2-build.sparsebundle")
    assert plan["mountpoint"].endswith("Volumes/CascadiaR2MapV1")
    assert plan["physical_backing_root"].endswith("John_1/cascadia-cluster/r2-map-v1")
    assert plan["mount_namespace_only"] is True
    assert machine.commands == []
    assert not lifecycle.spec.backing_bundle.exists()


def test_create_attach_verify_detach_are_idempotent_and_transactional(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lifecycle, machine = _fixture(tmp_path, monkeypatch)
    created = lifecycle.create(machine.run)
    repeated_create = lifecycle.create(machine.run)
    assert created["state"] == repeated_create["state"] == "created-detached"
    assert repeated_create["idempotent"] is True
    assert sum(command[1] == "create" for command in machine.commands) == 1

    attached = lifecycle.attach(machine.run, machine.observe)
    repeated_attach = lifecycle.attach(machine.run, machine.observe)
    assert attached["state"] == repeated_attach["state"] == "verified-mounted"
    assert attached["physical_backing_root"] == str(lifecycle.spec.campaign_root)
    assert attached["mount_namespace_only"] is True
    assert repeated_attach["idempotent"] is True
    marker = json.loads(lifecycle.spec.marker.read_text())
    assert marker["schema_id"] == APFS_WORKSPACE_SCHEMA
    assert all(path.is_dir() for path in lifecycle.spec.work_paths())
    assert lifecycle.verify(machine.observe)["state"] == "verified-mounted"

    detached = lifecycle.detach(machine.run, machine.observe)
    repeated_detach = lifecycle.detach(machine.run, machine.observe)
    assert detached["state"] == repeated_detach["state"] == "created-detached"
    assert repeated_detach["idempotent"] is True
    assert machine.mounted is False and machine.attached is False
    assert not lifecycle.paths.journal.exists()


@pytest.mark.parametrize(
    "stage",
    ["create-after-journal", "create-after-command", "create-after-receipt"],
)
def test_create_faults_recover_without_duplicate_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    lifecycle, machine = _fixture(tmp_path, monkeypatch)

    def fault(observed: str) -> None:
        if observed == stage:
            raise RuntimeError(stage)

    with pytest.raises(RuntimeError, match=stage):
        lifecycle.create(machine.run, fault=fault)
    recovered = lifecycle.recover(machine.run, machine.observe)
    assert recovered["state"] in {"absent", "created-detached"}
    assert not lifecycle.paths.journal.exists()
    assert sum(command[1] == "create" for command in machine.commands) <= 1


@pytest.mark.parametrize(
    "stage",
    [
        "attach-after-journal",
        "attach-after-command",
        "attach-after-observation",
        "attach-after-marker",
    ],
)
def test_attach_faults_recover_or_leave_clean_detached_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    lifecycle, machine = _fixture(tmp_path, monkeypatch)
    lifecycle.create(machine.run)

    def fault(observed: str) -> None:
        if observed == stage:
            raise RuntimeError(stage)

    with pytest.raises(RuntimeError, match=stage):
        lifecycle.attach(machine.run, machine.observe, fault=fault)
    recovered = lifecycle.recover(machine.run, machine.observe)
    if machine.mounted:
        assert recovered["state"] == "verified-mounted"
        lifecycle.detach(machine.run, machine.observe)
    else:
        assert recovered["state"] == "created-detached"
    assert not lifecycle.paths.journal.exists()


@pytest.mark.parametrize(
    "stage", ["detach-after-journal", "detach-after-unmount", "detach-after-command"]
)
def test_detach_faults_recover_to_detached_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    lifecycle, machine = _fixture(tmp_path, monkeypatch)
    lifecycle.create(machine.run)
    lifecycle.attach(machine.run, machine.observe)

    def fault(observed: str) -> None:
        if observed == stage:
            raise RuntimeError(stage)

    with pytest.raises(RuntimeError, match=stage):
        lifecycle.detach(machine.run, machine.observe, fault=fault)
    recovered = lifecycle.recover(machine.run, machine.observe)
    assert recovered["state"] == "created-detached"
    assert machine.mounted is False and machine.attached is False
    assert not lifecycle.paths.journal.exists()


def test_attach_identity_mismatch_immediately_unwinds_owned_mount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lifecycle, machine = _fixture(tmp_path, monkeypatch)
    lifecycle.create(machine.run)
    machine.contract["filesystem"] = "exfat"
    with pytest.raises(ApfsLifecycleError, match="identity"):
        lifecycle.attach(machine.run, machine.observe)
    assert machine.mounted is False and machine.attached is False
    assert not lifecycle.paths.journal.exists()
    assert any(command[0].endswith("diskutil") for command in machine.commands)
    assert any(
        command[0].endswith("hdiutil") and command[1] == "detach" for command in machine.commands
    )


def test_recovery_refuses_foreign_mount_and_never_detaches_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lifecycle, machine = _fixture(tmp_path, monkeypatch)
    lifecycle.create(machine.run)
    lifecycle._start_journal("attach", stage="after-command")
    machine.mounted = machine.attached = True
    machine.contract["backing_bundle"] = str(tmp_path / "foreign.sparsebundle")
    before = list(machine.commands)
    with pytest.raises(ApfsLifecycleError, match="foreign"):
        lifecycle.recover(machine.run, machine.observe)
    assert machine.commands == before
    assert machine.mounted is True


def test_blocked_host_refuses_create_and_attach_but_detach_remains_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lifecycle, machine = _fixture(tmp_path, monkeypatch)
    lifecycle.create(machine.run)
    lifecycle.attach(machine.run, machine.observe)
    blocked = build_host_safety_receipt(
        status="blocked-host-recovery",
        observed_unix_ms=2,
        syspolicyd_rss_bytes=5 * (1 << 30),
        system_swap_baseline_bytes=100,
        system_swap_observed_bytes=100,
        quiet_window_passed=False,
        detail="syspolicyd exceeds stop threshold",
    )
    write_host_safety(lifecycle.paths.host_safety, blocked)
    with pytest.raises(ApfsLifecycleError, match="APFS-bootstrap-safe"):
        lifecycle.create(machine.run)
    with pytest.raises(ApfsLifecycleError, match="APFS-bootstrap-safe"):
        lifecycle.attach(machine.run, machine.observe)
    assert lifecycle.detach(machine.run, machine.observe)["state"] == "created-detached"


def test_short_lived_bootstrap_receipt_authorizes_only_image_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lifecycle, machine = _fixture(tmp_path, monkeypatch)
    blocked = build_host_safety_receipt(
        status="blocked-host-recovery",
        observed_unix_ms=2,
        syspolicyd_rss_bytes=2 * (1 << 30),
        system_swap_baseline_bytes=100,
        system_swap_observed_bytes=100,
        quiet_window_passed=False,
        detail="strict runtime gate remains blocked",
    )
    write_host_safety(lifecycle.paths.host_safety, blocked)
    now_ms = time.time_ns() // 1_000_000
    bootstrap = build_apfs_bootstrap_safety_receipt(
        status="apfs-bootstrap-safe",
        started_unix_ms=now_ms - 60_000,
        completed_unix_ms=now_ms,
        maximum_syspolicyd_rss_bytes=2 * (1 << 30),
        system_swap_baseline_bytes=100,
        maximum_system_swap_used_bytes=100,
        maximum_memory_pressure_level=1,
        minimum_observed_backing_free_bytes=200 * (1 << 30),
        detail="infrastructure-only window passed; runtime remains unauthorized",
    )
    assert validate_apfs_bootstrap_safety(bootstrap)["runtime_authorized"] is False
    write_apfs_bootstrap_safety(lifecycle.paths.bootstrap_safety, bootstrap)
    monkeypatch.setattr(
        "cascadia_mlx.r2_map_apfs_lifecycle.shutil.disk_usage",
        lambda _path: SimpleNamespace(free=200 * (1 << 30)),
    )
    assert lifecycle.create(machine.run)["state"] == "created-detached"
    assert lifecycle.attach(machine.run, machine.observe)["state"] == "verified-mounted"


def test_stale_or_unsafe_bootstrap_receipt_never_authorizes_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lifecycle, machine = _fixture(tmp_path, monkeypatch)
    blocked = build_host_safety_receipt(
        status="blocked-host-recovery",
        observed_unix_ms=2,
        syspolicyd_rss_bytes=2 * (1 << 30),
        system_swap_baseline_bytes=100,
        system_swap_observed_bytes=100,
        quiet_window_passed=False,
        detail="strict runtime gate remains blocked",
    )
    write_host_safety(lifecycle.paths.host_safety, blocked)
    stale = build_apfs_bootstrap_safety_receipt(
        status="apfs-bootstrap-safe",
        started_unix_ms=1,
        completed_unix_ms=60_001,
        maximum_syspolicyd_rss_bytes=2 * (1 << 30),
        system_swap_baseline_bytes=100,
        maximum_system_swap_used_bytes=100,
        maximum_memory_pressure_level=1,
        minimum_observed_backing_free_bytes=200 * (1 << 30),
        detail="old infrastructure-only receipt",
    )
    write_apfs_bootstrap_safety(lifecycle.paths.bootstrap_safety, stale)
    with pytest.raises(ApfsLifecycleError, match="stale"):
        lifecycle.create(machine.run)

    invalid = replace_receipt(
        stale,
        status="apfs-bootstrap-safe",
        completed_unix_ms=lifecycle.now_ms(),
        maximum_system_swap_used_bytes=101,
        system_swap_delta_bytes=1,
    )
    with pytest.raises(ApfsLifecycleError, match="classification"):
        validate_apfs_bootstrap_safety(invalid)


def test_host_safety_receipt_hash_classification_and_dashboard_projection() -> None:
    blocked = build_host_safety_receipt(
        status="blocked-host-recovery",
        observed_unix_ms=3,
        syspolicyd_rss_bytes=4_461_944_832,
        system_swap_baseline_bytes=2_000_000_000,
        system_swap_observed_bytes=2_000_000_000,
        quiet_window_passed=False,
        detail="host stop remains active",
    )
    assert validate_host_safety(blocked) == blocked
    dashboard = host_dashboard_receipt("control", blocked)
    assert dashboard["intent"] == "control"
    assert dashboard["detail"].startswith("blocked-host-recovery")
    assert dashboard["rss_bytes"] == blocked["rss_bytes"]
    assert "swap_baseline=2000000000" in dashboard["detail"]

    tampered = replace_receipt(blocked, status="safe")
    with pytest.raises(ApfsLifecycleError, match="classification"):
        validate_host_safety(tampered)


def replace_receipt(value: dict[str, Any], **updates: Any) -> dict[str, Any]:
    result = {**value, **updates}
    from cascadia_mlx.r2_map_contracts import content_sha256

    result["receipt_sha256"] = content_sha256(result, hash_field="receipt_sha256")
    return result


def test_spec_path_drift_still_fails_before_lifecycle_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lifecycle, machine = _fixture(tmp_path, monkeypatch)
    with pytest.raises(Exception, match="mount namespace"):
        ApfsWorkspaceLifecycle(
            replace(lifecycle.spec, mountpoint=tmp_path / "internal"),
            mounted_checker=lambda _path: False,
            current_uid=os.getuid(),
            current_gid=os.getgid(),
        )
    assert machine.commands == []
