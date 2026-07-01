"""Retired transactional lifecycle retained for historical receipt validation.

Active campaign operations must use john2's native disk through
``r2_map_remote_storage``.  The command-line lifecycle entry point is disabled;
this implementation remains solely so old evidence and isolated tests can be
decoded without rewriting history.

The implementation is command-runner driven. Importing it performs no I/O,
and tests inject a fake runner and observation provider. The production CLI is
the only layer that supplies subprocess execution, and mutating operations are
blocked unless an explicit host-safety receipt has passed recovery gates.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from cascadia_mlx.r2_map_apfs_workspace import (
    APFS_WORKSPACE_SCHEMA,
    ApfsWorkspaceContractError,
    ApfsWorkspaceSpec,
    validate_marker_and_mount_observation,
)
from cascadia_mlx.r2_map_contracts import CAMPAIGN_ID, content_sha256

LIFECYCLE_RECEIPT_SCHEMA = "cascadia.r2-map.apfs-lifecycle-receipt.v2"
LIFECYCLE_JOURNAL_SCHEMA = "cascadia.r2-map.apfs-lifecycle-journal.v1"
HOST_SAFETY_SCHEMA = "cascadia.r2-map.host-safety.v1"
APFS_BOOTSTRAP_SAFETY_SCHEMA = "cascadia.r2-map.apfs-bootstrap-safety.v2"
APFS_BOOTSTRAP_RECEIPT_MAX_AGE_MS = 5 * 60 * 1000
_DEVICE = re.compile(r"/dev/disk[0-9]+(?:s[0-9]+)?")


class ApfsLifecycleError(RuntimeError):
    """A lifecycle transition, identity, command, or recovery invariant failed."""


@dataclass(frozen=True, slots=True)
class CommandResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str = ""
    stderr: str = ""


class CommandRunner(Protocol):
    def __call__(self, argv: Sequence[str]) -> CommandResult: ...


ObservationProvider = Callable[[ApfsWorkspaceSpec], Mapping[str, Any] | None]
MountedChecker = Callable[[Path], bool]
FaultInjector = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class ApfsLifecyclePaths:
    journal: Path
    receipt: Path
    host_safety: Path
    bootstrap_safety: Path

    @classmethod
    def for_spec(cls, spec: ApfsWorkspaceSpec) -> ApfsLifecyclePaths:
        control = spec.campaign_root / "control"
        return cls(
            journal=control / "apfs-workspace-operation.json",
            receipt=control / "apfs-workspace-status.json",
            host_safety=control / "host-safety.json",
            bootstrap_safety=control / "apfs-bootstrap-safety.json",
        )


class ApfsWorkspaceLifecycle:
    """Idempotent sparsebundle lifecycle with a durable operation journal."""

    def __init__(
        self,
        spec: ApfsWorkspaceSpec,
        *,
        paths: ApfsLifecyclePaths | None = None,
        mounted_checker: MountedChecker,
        current_uid: int,
        current_gid: int,
        now_ms: Callable[[], int] | None = None,
    ):
        spec.validate()
        self.spec = spec
        self.paths = paths or ApfsLifecyclePaths.for_spec(spec)
        self.mounted_checker = mounted_checker
        self.current_uid = current_uid
        self.current_gid = current_gid
        self.now_ms = now_ms or (lambda: time.time_ns() // 1_000_000)

    def plan(self) -> dict[str, Any]:
        """Return exact commands and immutable paths without mutating anything."""
        spec = self.spec
        return {
            "schema_version": 1,
            "schema_id": "cascadia.r2-map.apfs-lifecycle-plan.v1",
            "campaign_id": CAMPAIGN_ID,
            "backing_bundle": str(spec.backing_bundle),
            "mountpoint": str(spec.mountpoint),
            "physical_backing_root": str(spec.campaign_root),
            "mount_namespace_only": True,
            "marker": str(spec.marker),
            "capacity_bytes": spec.capacity_bytes,
            "budget_bytes": spec.budget_bytes,
            "minimum_backing_free_bytes": spec.minimum_backing_free_bytes,
            "minimum_mount_free_bytes": spec.minimum_mount_free_bytes,
            "create_argv": list(self._create_argv()),
            "attach_argv": list(self._attach_argv()),
            "work_paths": [str(path) for path in spec.work_paths()],
            "journal": str(self.paths.journal),
            "status_receipt": str(self.paths.receipt),
        }

    def status(self, observe: ObservationProvider) -> dict[str, Any]:
        journal = _read_optional_json(self.paths.journal)
        receipt = _read_optional_json(self.paths.receipt)
        observation = observe(self.spec) if self.mounted_checker(self.spec.mountpoint) else None
        state = "absent"
        valid = False
        detail = None
        if observation is not None:
            try:
                self._validate_mounted(observation, require_marker=True)
                state, valid = "verified-mounted", True
            except ApfsLifecycleError as error:
                state, detail = "invalid-mounted", str(error)
        elif self.spec.backing_bundle.exists():
            state = "created-detached"
        return {
            "schema_version": 1,
            "schema_id": "cascadia.r2-map.apfs-lifecycle-status.v1",
            "campaign_id": CAMPAIGN_ID,
            "state": state,
            "valid": valid,
            "detail": detail,
            "journal": journal,
            "latest_receipt": receipt,
        }

    def create(
        self,
        runner: CommandRunner,
        *,
        fault: FaultInjector | None = None,
    ) -> dict[str, Any]:
        self._require_infrastructure_safe()
        with _exclusive_lock(self.paths.journal):
            if self.mounted_checker(self.spec.mountpoint):
                raise ApfsLifecycleError("cannot create while the APFS workspace is mounted")
            if self.spec.backing_bundle.exists():
                return self._write_receipt("created-detached", operation="create", idempotent=True)
            self.spec.backing_bundle.parent.mkdir(parents=True, exist_ok=True)
            if self.spec.mountpoint.exists():
                self._require_empty_mountpoint()
            journal = self._start_journal("create", stage="before-command")
            _fault(fault, "create-after-journal")
            result = _run_checked(runner, self._create_argv())
            self._update_journal(journal, stage="after-command", command=result)
            _fault(fault, "create-after-command")
            if not self.spec.backing_bundle.is_dir():
                raise ApfsLifecycleError("hdiutil create did not produce the exact sparsebundle")
            receipt = self._write_receipt("created-detached", operation="create")
            _fault(fault, "create-after-receipt")
            self._clear_journal()
            return receipt

    def attach(
        self,
        runner: CommandRunner,
        observe: ObservationProvider,
        *,
        fault: FaultInjector | None = None,
    ) -> dict[str, Any]:
        self._require_infrastructure_safe()
        with _exclusive_lock(self.paths.journal):
            if self.mounted_checker(self.spec.mountpoint):
                observation = _require_observation(observe(self.spec))
                mounted = self._validate_mounted(observation, require_marker=True)
                return self._write_receipt(
                    "verified-mounted",
                    operation="attach",
                    device=mounted["device"],
                    idempotent=True,
                )
            if not self.spec.backing_bundle.is_dir():
                raise ApfsLifecycleError("cannot attach before the exact sparsebundle exists")
            if self.spec.mountpoint.exists():
                self._require_empty_mountpoint()
            journal = self._start_journal("attach", stage="before-command")
            _fault(fault, "attach-after-journal")
            try:
                result = _run_checked(runner, self._attach_argv())
            except Exception:
                observation = (
                    observe(self.spec) if self.mounted_checker(self.spec.mountpoint) else None
                )
                if observation is not None:
                    self._detach_owned_observation(runner, observation)
                    self._write_receipt("created-detached", operation="attach-abort-unwind")
                    self._clear_journal()
                raise
            self._update_journal(journal, stage="after-command", command=result)
            _fault(fault, "attach-after-command")
            observation = _require_observation(observe(self.spec))
            try:
                mounted = self._validate_mounted(
                    observation, require_marker=False, allow_uninitialized_mode=True
                )
                self._update_journal(journal, stage="observed", device=mounted["device"])
                _fault(fault, "attach-after-observation")
                self._initialize_mounted_workspace(mounted["contract"])
                initialized_observation = _require_observation(observe(self.spec))
                initialized = self._validate_mounted(
                    initialized_observation, require_marker=False
                )
                _atomic_write_json(
                    self.spec.marker,
                    _marker_from_observation(self.spec, initialized["contract"]),
                    mode=0o600,
                )
                self._validate_mounted(initialized_observation, require_marker=True)
            except Exception:
                self._detach_owned_observation(runner, observation)
                self._write_receipt("created-detached", operation="attach-unwind")
                self._clear_journal()
                raise
            _fault(fault, "attach-after-marker")
            receipt = self._write_receipt(
                "verified-mounted", operation="attach", device=mounted["device"]
            )
            self._clear_journal()
            return receipt

    def verify(self, observe: ObservationProvider) -> dict[str, Any]:
        observation = _require_observation(observe(self.spec))
        mounted = self._validate_mounted(observation, require_marker=True)
        return self._write_receipt("verified-mounted", operation="verify", device=mounted["device"])

    def detach(
        self,
        runner: CommandRunner,
        observe: ObservationProvider,
        *,
        fault: FaultInjector | None = None,
    ) -> dict[str, Any]:
        with _exclusive_lock(self.paths.journal):
            if not self.mounted_checker(self.spec.mountpoint):
                self._clear_journal()
                return self._write_receipt("created-detached", operation="detach", idempotent=True)
            mounted = self._validate_mounted(
                _require_observation(observe(self.spec)), require_marker=True
            )
            journal = self._start_journal(
                "detach", stage="before-unmount", device=mounted["device"]
            )
            _fault(fault, "detach-after-journal")
            unmount = _run_checked(
                runner, ("/usr/sbin/diskutil", "unmount", str(self.spec.mountpoint))
            )
            self._update_journal(journal, stage="after-unmount", command=unmount)
            _fault(fault, "detach-after-unmount")
            detached = _run_checked(runner, ("/usr/bin/hdiutil", "detach", mounted["device"]))
            self._update_journal(journal, stage="after-detach", command=detached)
            _fault(fault, "detach-after-command")
            if self.mounted_checker(self.spec.mountpoint) or observe(self.spec) is not None:
                raise ApfsLifecycleError("workspace remains mounted after detach")
            receipt = self._write_receipt("created-detached", operation="detach")
            self._clear_journal()
            return receipt

    def recover(
        self,
        runner: CommandRunner,
        observe: ObservationProvider,
    ) -> dict[str, Any]:
        """Finish or safely unwind only a journaled exact lifecycle operation."""
        with _exclusive_lock(self.paths.journal):
            journal = _read_optional_json(self.paths.journal)
            observation = observe(self.spec) if self.mounted_checker(self.spec.mountpoint) else None
            if journal is None:
                if observation is None:
                    state = "created-detached" if self.spec.backing_bundle.exists() else "absent"
                    return self._write_receipt(state, operation="recover", idempotent=True)
                mounted = self._validate_mounted(observation, require_marker=True)
                return self._write_receipt(
                    "verified-mounted",
                    operation="recover",
                    device=mounted["device"],
                    idempotent=True,
                )
            journal = _validate_journal(journal)
            operation = journal["operation"]
            if operation == "create":
                if observation is not None:
                    raise ApfsLifecycleError(
                        "create recovery found an unexpected mounted workspace"
                    )
                state = "created-detached" if self.spec.backing_bundle.is_dir() else "absent"
                receipt = self._write_receipt(state, operation="recover")
                self._clear_journal()
                return receipt
            if observation is None:
                if operation == "detach" and isinstance(journal.get("device"), str):
                    if _DEVICE.fullmatch(journal["device"]) is None:
                        raise ApfsLifecycleError("detach journal device is invalid")
                    _run_checked(runner, ("/usr/bin/hdiutil", "detach", journal["device"]))
                state = "created-detached" if self.spec.backing_bundle.exists() else "absent"
                receipt = self._write_receipt(state, operation="recover")
                self._clear_journal()
                return receipt
            if operation == "attach":
                try:
                    mounted = self._validate_mounted(
                        observation, require_marker=False, allow_uninitialized_mode=True
                    )
                except ApfsLifecycleError:
                    self._detach_owned_observation(runner, observation)
                    receipt = self._write_receipt("created-detached", operation="recover-unwind")
                    self._clear_journal()
                    return receipt
                self._initialize_mounted_workspace(mounted["contract"])
                initialized_observation = _require_observation(observe(self.spec))
                initialized = self._validate_mounted(
                    initialized_observation, require_marker=False
                )
                _atomic_write_json(
                    self.spec.marker,
                    _marker_from_observation(self.spec, initialized["contract"]),
                    mode=0o600,
                )
                self._validate_mounted(initialized_observation, require_marker=True)
                receipt = self._write_receipt(
                    "verified-mounted",
                    operation="recover",
                    device=mounted["device"],
                )
                self._clear_journal()
                return receipt
            if operation == "detach":
                mounted = self._validate_mounted(observation, require_marker=True)
                _run_checked(runner, ("/usr/sbin/diskutil", "unmount", str(self.spec.mountpoint)))
                _run_checked(runner, ("/usr/bin/hdiutil", "detach", mounted["device"]))
                if self.mounted_checker(self.spec.mountpoint) or observe(self.spec) is not None:
                    raise ApfsLifecycleError("detach recovery could not clear the exact mount")
                receipt = self._write_receipt("created-detached", operation="recover")
                self._clear_journal()
                return receipt
            raise ApfsLifecycleError("journal operation is unsupported")

    def _create_argv(self) -> tuple[str, ...]:
        return (
            "/usr/bin/hdiutil",
            "create",
            "-type",
            "SPARSEBUNDLE",
            "-fs",
            "APFS",
            "-size",
            "64g",
            "-volname",
            self.spec.volume_name,
            str(self.spec.backing_bundle),
        )

    def _attach_argv(self) -> tuple[str, ...]:
        return (
            "/usr/bin/hdiutil",
            "attach",
            "-plist",
            "-owners",
            "on",
            "-mountpoint",
            str(self.spec.mountpoint),
            str(self.spec.backing_bundle),
        )

    def _require_empty_mountpoint(self) -> None:
        if self.spec.mountpoint.exists() and any(self.spec.mountpoint.iterdir()):
            raise ApfsLifecycleError("exact APFS mountpoint must be empty before attach")

    def _require_infrastructure_safe(self) -> None:
        receipt = _read_optional_json(self.paths.host_safety)
        if receipt is not None:
            validate_host_safety(receipt)
            if receipt["status"] == "safe" and receipt["quiet_window_passed"] is True:
                return
        bootstrap = _read_optional_json(self.paths.bootstrap_safety)
        if bootstrap is None:
            raise ApfsLifecycleError(
                "john1 lacks a strict runtime-safe or APFS-bootstrap-safe receipt"
            )
        bootstrap = validate_apfs_bootstrap_safety(bootstrap)
        age_ms = self.now_ms() - bootstrap["completed_unix_ms"]
        if not 0 <= age_ms <= APFS_BOOTSTRAP_RECEIPT_MAX_AGE_MS:
            raise ApfsLifecycleError("APFS bootstrap safety receipt is stale or future-dated")
        if bootstrap["status"] != "apfs-bootstrap-safe":
            raise ApfsLifecycleError("john1 remains blocked-host-recovery")
        current_backing_free = shutil.disk_usage(self.spec.campaign_root).free
        if current_backing_free < self.spec.minimum_backing_free_bytes:
            raise ApfsLifecycleError("backing SSD reserve floor failed before APFS bootstrap")

    def _validate_mounted(
        self,
        observation: Mapping[str, Any],
        *,
        require_marker: bool,
        allow_uninitialized_mode: bool = False,
    ) -> dict[str, Any]:
        mounted = _validate_mounted_wrapper(observation)
        contract = mounted["contract"]
        if contract["backing_bundle"] != str(self.spec.backing_bundle) or contract[
            "mountpoint"
        ] != str(self.spec.mountpoint):
            raise ApfsLifecycleError("mounted image is foreign to the exact workspace paths")
        marker = None
        if require_marker:
            marker = _read_optional_json(self.spec.marker)
            if marker is None:
                raise ApfsLifecycleError("verified APFS workspace marker is absent")
        else:
            marker = _marker_from_observation(self.spec, contract)
        try:
            validate_marker_and_mount_observation(
                self.spec,
                marker,
                contract,
                planned_bytes=self.spec.budget_bytes,
                allow_uninitialized_mode=allow_uninitialized_mode,
            )
        except ApfsWorkspaceContractError as error:
            raise ApfsLifecycleError(f"mounted APFS identity failed: {error}") from error
        return mounted

    def _initialize_mounted_workspace(self, observation: Mapping[str, Any]) -> None:
        if (
            observation["owner_uid"] != self.current_uid
            or observation["owner_gid"] != self.current_gid
        ):
            raise ApfsLifecycleError("mounted APFS owner differs from current campaign owner")
        self.spec.mountpoint.chmod(self.spec.required_mode)
        for path in self.spec.work_paths():
            path.mkdir(parents=False, exist_ok=True)
            path.chmod(self.spec.required_mode)

    def _detach_owned_observation(
        self, runner: CommandRunner, observation: Mapping[str, Any]
    ) -> None:
        mounted = _validate_mounted_wrapper(observation)
        contract = mounted["contract"]
        if contract.get("backing_bundle") != str(self.spec.backing_bundle) or contract.get(
            "mountpoint"
        ) != str(self.spec.mountpoint):
            raise ApfsLifecycleError("recovery refuses to detach a foreign mount")
        _run_checked(runner, ("/usr/sbin/diskutil", "unmount", str(self.spec.mountpoint)))
        _run_checked(runner, ("/usr/bin/hdiutil", "detach", mounted["device"]))

    def _start_journal(self, operation: str, *, stage: str, device: str | None = None) -> dict:
        value = {
            "schema_version": 1,
            "schema_id": LIFECYCLE_JOURNAL_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "operation": operation,
            "stage": stage,
            "backing_bundle": str(self.spec.backing_bundle),
            "mountpoint": str(self.spec.mountpoint),
            "device": device,
            "command": None,
        }
        value["journal_sha256"] = content_sha256(value, hash_field="journal_sha256")
        _atomic_write_json(self.paths.journal, value)
        return value

    def _update_journal(self, journal: dict, *, stage: str, **updates: Any) -> None:
        journal.update(stage=stage, **updates)
        if isinstance(journal.get("command"), CommandResult):
            result = journal["command"]
            journal["command"] = {
                "argv": list(result.argv),
                "returncode": result.returncode,
            }
        journal["journal_sha256"] = content_sha256(journal, hash_field="journal_sha256")
        _atomic_write_json(self.paths.journal, journal)

    def _clear_journal(self) -> None:
        self.paths.journal.unlink(missing_ok=True)

    def _write_receipt(
        self,
        state: str,
        *,
        operation: str,
        device: str | None = None,
        idempotent: bool = False,
    ) -> dict[str, Any]:
        value = {
            "schema_version": 1,
            "schema_id": LIFECYCLE_RECEIPT_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "state": state,
            "operation": operation,
            "backing_bundle": str(self.spec.backing_bundle),
            "mountpoint": str(self.spec.mountpoint),
            "physical_backing_root": str(self.spec.campaign_root),
            "mount_namespace_only": True,
            "device": device,
            "idempotent": idempotent,
        }
        value["receipt_sha256"] = content_sha256(value, hash_field="receipt_sha256")
        _atomic_write_json(self.paths.receipt, value)
        return value


def build_host_safety_receipt(
    *,
    status: str,
    observed_unix_ms: int,
    syspolicyd_rss_bytes: int,
    system_swap_baseline_bytes: int,
    system_swap_observed_bytes: int,
    quiet_window_passed: bool,
    detail: str,
) -> dict[str, Any]:
    value = {
        "schema_version": 1,
        "schema_id": HOST_SAFETY_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "host": "john1",
        "status": status,
        "observed_unix_ms": observed_unix_ms,
        "process": "syspolicyd",
        "rss_bytes": syspolicyd_rss_bytes,
        "hard_stop_rss_bytes": 4 * (1 << 30),
        "recovery_rss_threshold_bytes": 256 * (1 << 20),
        "system_swap_baseline_bytes": system_swap_baseline_bytes,
        "system_swap_observed_bytes": system_swap_observed_bytes,
        "quiet_window_seconds": 60,
        "quiet_window_passed": quiet_window_passed,
        "detail": detail,
    }
    value["receipt_sha256"] = content_sha256(value, hash_field="receipt_sha256")
    return validate_host_safety(value)


def validate_host_safety(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "schema_id",
        "campaign_id",
        "host",
        "status",
        "observed_unix_ms",
        "process",
        "rss_bytes",
        "hard_stop_rss_bytes",
        "recovery_rss_threshold_bytes",
        "system_swap_baseline_bytes",
        "system_swap_observed_bytes",
        "quiet_window_seconds",
        "quiet_window_passed",
        "detail",
        "receipt_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ApfsLifecycleError("host-safety receipt schema differs")
    if (
        value["schema_version"] != 1
        or value["schema_id"] != HOST_SAFETY_SCHEMA
        or value["campaign_id"] != CAMPAIGN_ID
        or value["host"] != "john1"
        or value["status"] not in {"blocked-host-recovery", "safe"}
        or value["process"] != "syspolicyd"
        or value["hard_stop_rss_bytes"] != 4 * (1 << 30)
        or value["recovery_rss_threshold_bytes"] != 256 * (1 << 20)
        or value["quiet_window_seconds"] != 60
        or any(
            not isinstance(value[name], int) or value[name] < 0
            for name in (
                "observed_unix_ms",
                "rss_bytes",
                "system_swap_baseline_bytes",
                "system_swap_observed_bytes",
            )
        )
        or not isinstance(value["quiet_window_passed"], bool)
        or not isinstance(value["detail"], str)
        or not value["detail"]
    ):
        raise ApfsLifecycleError("host-safety receipt identity or metric differs")
    safe = (
        value["quiet_window_passed"]
        and value["rss_bytes"] <= value["recovery_rss_threshold_bytes"]
        and value["system_swap_observed_bytes"] <= value["system_swap_baseline_bytes"]
    )
    if (value["status"] == "safe") != safe:
        raise ApfsLifecycleError("host-safety classification disagrees with recovery gates")
    if value["receipt_sha256"] != content_sha256(value, hash_field="receipt_sha256"):
        raise ApfsLifecycleError("host-safety receipt hash differs")
    return dict(value)


def write_host_safety(path: Path, receipt: Mapping[str, Any]) -> None:
    _atomic_write_json(path, validate_host_safety(receipt))


def build_apfs_bootstrap_safety_receipt(
    *,
    status: str,
    started_unix_ms: int,
    completed_unix_ms: int,
    maximum_syspolicyd_rss_bytes: int,
    system_swap_baseline_bytes: int,
    maximum_system_swap_used_bytes: int,
    maximum_memory_pressure_level: int,
    minimum_observed_backing_free_bytes: int,
    detail: str,
) -> dict[str, Any]:
    """Build the narrow, short-lived gate for image creation/attachment only.

    This receipt never authorizes a build, model load, gameplay process, or any
    other campaign runtime. Those operations continue to require the strict
    ``host-safety`` receipt.
    """
    value = {
        "schema_version": 1,
        "schema_id": APFS_BOOTSTRAP_SAFETY_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "host": "john1",
        "status": status,
        "started_unix_ms": started_unix_ms,
        "completed_unix_ms": completed_unix_ms,
        "window_seconds": 60,
        "process": "syspolicyd",
        "maximum_syspolicyd_rss_bytes": maximum_syspolicyd_rss_bytes,
        "hard_stop_rss_bytes": 4 * (1 << 30),
        "system_swap_baseline_bytes": system_swap_baseline_bytes,
        "maximum_system_swap_used_bytes": maximum_system_swap_used_bytes,
        "system_swap_delta_bytes": max(
            maximum_system_swap_used_bytes - system_swap_baseline_bytes, 0
        ),
        "maximum_memory_pressure_level": maximum_memory_pressure_level,
        "required_memory_pressure_level": 1,
        "minimum_backing_free_bytes": 140 * (1 << 30),
        "minimum_observed_backing_free_bytes": minimum_observed_backing_free_bytes,
        "authorized_operations": ["hdiutil-create", "hdiutil-attach", "verify"],
        "runtime_authorized": False,
        "detail": detail,
    }
    value["receipt_sha256"] = content_sha256(value, hash_field="receipt_sha256")
    return validate_apfs_bootstrap_safety(value)


def validate_apfs_bootstrap_safety(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "schema_id",
        "campaign_id",
        "host",
        "status",
        "started_unix_ms",
        "completed_unix_ms",
        "window_seconds",
        "process",
        "maximum_syspolicyd_rss_bytes",
        "hard_stop_rss_bytes",
        "system_swap_baseline_bytes",
        "maximum_system_swap_used_bytes",
        "system_swap_delta_bytes",
        "maximum_memory_pressure_level",
        "required_memory_pressure_level",
        "minimum_backing_free_bytes",
        "minimum_observed_backing_free_bytes",
        "authorized_operations",
        "runtime_authorized",
        "detail",
        "receipt_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ApfsLifecycleError("APFS bootstrap safety receipt schema differs")
    integer_fields = (
        "started_unix_ms",
        "completed_unix_ms",
        "maximum_syspolicyd_rss_bytes",
        "system_swap_baseline_bytes",
        "maximum_system_swap_used_bytes",
        "system_swap_delta_bytes",
        "maximum_memory_pressure_level",
        "minimum_observed_backing_free_bytes",
    )
    if (
        value["schema_version"] != 1
        or value["schema_id"] != APFS_BOOTSTRAP_SAFETY_SCHEMA
        or value["campaign_id"] != CAMPAIGN_ID
        or value["host"] != "john1"
        or value["status"] not in {"apfs-bootstrap-safe", "blocked-host-recovery"}
        or value["window_seconds"] != 60
        or value["process"] != "syspolicyd"
        or value["hard_stop_rss_bytes"] != 4 * (1 << 30)
        or value["required_memory_pressure_level"] != 1
        or value["minimum_backing_free_bytes"] != 140 * (1 << 30)
        or value["authorized_operations"] != [
            "hdiutil-create",
            "hdiutil-attach",
            "verify",
        ]
        or value["runtime_authorized"] is not False
        or any(not isinstance(value[name], int) or value[name] < 0 for name in integer_fields)
        or value["completed_unix_ms"] < value["started_unix_ms"]
        or value["system_swap_delta_bytes"]
        != max(
            value["maximum_system_swap_used_bytes"]
            - value["system_swap_baseline_bytes"],
            0,
        )
        or not isinstance(value["detail"], str)
        or not value["detail"]
    ):
        raise ApfsLifecycleError("APFS bootstrap safety identity or metric differs")
    safe = (
        value["maximum_syspolicyd_rss_bytes"] < value["hard_stop_rss_bytes"]
        and value["system_swap_delta_bytes"] == 0
        and value["maximum_memory_pressure_level"]
        == value["required_memory_pressure_level"]
        and value["minimum_observed_backing_free_bytes"]
        >= value["minimum_backing_free_bytes"]
    )
    if (value["status"] == "apfs-bootstrap-safe") != safe:
        raise ApfsLifecycleError("APFS bootstrap safety classification disagrees with gates")
    if value["receipt_sha256"] != content_sha256(value, hash_field="receipt_sha256"):
        raise ApfsLifecycleError("APFS bootstrap safety receipt hash differs")
    return dict(value)


def write_apfs_bootstrap_safety(path: Path, receipt: Mapping[str, Any]) -> None:
    _atomic_write_json(path, validate_apfs_bootstrap_safety(receipt))


def host_dashboard_receipt(intent: str, safety: Mapping[str, Any]) -> dict[str, Any]:
    safety = validate_host_safety(safety)
    swap_delta = max(safety["system_swap_observed_bytes"] - safety["system_swap_baseline_bytes"], 0)
    return {
        "intent": intent,
        "detail": (
            f"{safety['status']}: syspolicyd_rss={safety['rss_bytes']} "
            f"hard_stop={safety['hard_stop_rss_bytes']} "
            f"recovery_threshold={safety['recovery_rss_threshold_bytes']} "
            f"swap_baseline={safety['system_swap_baseline_bytes']} "
            f"swap_observed={safety['system_swap_observed_bytes']}"
        ),
        "generation_games_completed": 0,
        "generation_games_target": None,
        "generation_seed_prefix": None,
        "benchmark_pairs_completed": 0,
        "benchmark_pairs_total": None,
        "eta_seconds": None,
        "throughput_games_per_second": None,
        "rss_bytes": safety["rss_bytes"],
        "swap_delta_bytes": swap_delta,
    }


def _marker_from_observation(
    spec: ApfsWorkspaceSpec, observation: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "schema_id": APFS_WORKSPACE_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "backing_bundle": str(spec.backing_bundle),
        "mountpoint": str(spec.mountpoint),
        "volume_name": observation["volume_name"],
        "volume_uuid": observation["volume_uuid"],
        "filesystem": observation["filesystem"],
        "capacity_bytes": observation["capacity_bytes"],
        "budget_bytes": spec.budget_bytes,
        "owner_uid": observation["owner_uid"],
        "owner_gid": observation["owner_gid"],
        "mode": observation["mode"],
        "physical_backing_root": str(spec.campaign_root),
        "mount_namespace_only": True,
    }


def _validate_mounted_wrapper(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"device", "contract"}:
        raise ApfsLifecycleError("mount observation wrapper schema differs")
    if not isinstance(value["device"], str) or _DEVICE.fullmatch(value["device"]) is None:
        raise ApfsLifecycleError("mount observation device is invalid")
    if not isinstance(value["contract"], Mapping):
        raise ApfsLifecycleError("mount observation contract is invalid")
    return {"device": value["device"], "contract": dict(value["contract"])}


def _require_observation(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if value is None:
        raise ApfsLifecycleError("exact mounted APFS observation is absent")
    return value


def _run_checked(runner: CommandRunner, argv: Sequence[str]) -> CommandResult:
    result = runner(tuple(argv))
    if result.argv != tuple(argv) or result.returncode != 0:
        raise ApfsLifecycleError(
            "lifecycle command failed or runner identity drifted: "
            f"argv={tuple(argv)!r} returncode={result.returncode} "
            f"stderr={result.stderr.strip()!r}"
        )
    return result


def _validate_journal(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "schema_id",
        "campaign_id",
        "operation",
        "stage",
        "backing_bundle",
        "mountpoint",
        "device",
        "command",
        "journal_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ApfsLifecycleError("APFS lifecycle journal schema differs")
    if (
        value["schema_version"] != 1
        or value["schema_id"] != LIFECYCLE_JOURNAL_SCHEMA
        or value["campaign_id"] != CAMPAIGN_ID
        or value["operation"] not in {"create", "attach", "detach"}
        or value["journal_sha256"] != content_sha256(value, hash_field="journal_sha256")
    ):
        raise ApfsLifecycleError("APFS lifecycle journal identity differs")
    return dict(value)


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ApfsLifecycleError(f"cannot read lifecycle object {path}") from error
    if not isinstance(value, dict):
        raise ApfsLifecycleError(f"lifecycle object {path} is not a JSON object")
    return value


class _exclusive_lock:
    def __init__(self, path: Path):
        self.path = path.with_name(f".{path.name}.lock")
        self.handle: Any = None

    def __enter__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+")
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)

    def __exit__(self, *_: object) -> None:
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()


def _atomic_write_json(path: Path, value: Mapping[str, Any], *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        temporary = Path(name)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fchmod(handle.fileno(), mode)
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            with suppress(FileNotFoundError):
                temporary.unlink()


def _fault(fault: FaultInjector | None, stage: str) -> None:
    if fault is not None:
        fault(stage)
