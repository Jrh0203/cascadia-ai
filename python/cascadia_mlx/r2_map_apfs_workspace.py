"""Read-only validation contracts for the retired John1 APFS workspace.

The active campaign uses john2's native disk through the strict remote-storage
transport.  This module remains only to validate historical receipts; it must
never supply an active campaign default or authorize a mount operation.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cascadia_mlx.r2_map_contracts import CAMPAIGN_ID, GIB, MIN_FREE_BYTES

APFS_WORKSPACE_SCHEMA = "cascadia.r2-map.apfs-workspace.v2"
APFS_WORKSPACE_CAPACITY_BYTES = 64 * GIB
APFS_WORKSPACE_BUDGET_BYTES = 40 * GIB
APFS_WORKSPACE_MIN_BACKING_FREE_BYTES = MIN_FREE_BYTES + APFS_WORKSPACE_BUDGET_BYTES
APFS_WORKSPACE_MIN_MOUNT_FREE_BYTES = APFS_WORKSPACE_BUDGET_BYTES
APFS_WORKSPACE_VOLUME_NAME = "CascadiaR2Build"
APFS_WORKSPACE_MOUNTPOINT = Path("/Volumes/CascadiaR2MapV1")
APFS_WORKSPACE_MODE = 0o700
APFS_WORKSPACE_DEPRECATED = True
_UUID = re.compile(r"[0-9A-F]{8}(?:-[0-9A-F]{4}){3}-[0-9A-F]{12}")


class ApfsWorkspaceContractError(ValueError):
    """The contained build workspace identity or storage budget differs."""


@dataclass(frozen=True)
class ApfsWorkspaceSpec:
    campaign_root: Path
    backing_bundle: Path
    mountpoint: Path
    marker: Path
    cargo_target: Path
    temporary: Path
    cache: Path
    volume_name: str = APFS_WORKSPACE_VOLUME_NAME
    capacity_bytes: int = APFS_WORKSPACE_CAPACITY_BYTES
    budget_bytes: int = APFS_WORKSPACE_BUDGET_BYTES
    minimum_backing_free_bytes: int = APFS_WORKSPACE_MIN_BACKING_FREE_BYTES
    minimum_mount_free_bytes: int = APFS_WORKSPACE_MIN_MOUNT_FREE_BYTES
    required_mode: int = APFS_WORKSPACE_MODE

    @classmethod
    def campaign_default(cls) -> ApfsWorkspaceSpec:
        raise ApfsWorkspaceContractError(
            "the John1 nested APFS workspace is retired; use john2 remote storage"
        )

    @classmethod
    def historical_fixture(
        cls, *, campaign_root: Path, mountpoint: Path
    ) -> ApfsWorkspaceSpec:
        """Build an explicit validator spec without selecting a live default."""
        backing = campaign_root / "storage/r2-build.sparsebundle"
        return cls(
            campaign_root=campaign_root,
            backing_bundle=backing,
            mountpoint=mountpoint,
            marker=mountpoint / ".r2-map-apfs-workspace.json",
            cargo_target=mountpoint / "cargo-target",
            temporary=mountpoint / "tmp",
            cache=mountpoint / "cache",
        )

    def validate(self) -> None:
        if not self.campaign_root.is_absolute():
            raise ApfsWorkspaceContractError("historical campaign root must be absolute")
        if self.backing_bundle.suffix != ".sparsebundle":
            raise ApfsWorkspaceContractError("APFS backing path must be a sparsebundle")
        _require_contained("backing bundle", self.backing_bundle, self.campaign_root)
        if self.mountpoint != APFS_WORKSPACE_MOUNTPOINT:
            raise ApfsWorkspaceContractError("APFS mount namespace differs from the frozen path")
        if self.mountpoint.parent != APFS_WORKSPACE_MOUNTPOINT.parent or self.mountpoint.suffix:
            raise ApfsWorkspaceContractError("APFS mount namespace must be one exact direct child")
        if self.marker.parent != self.mountpoint:
            raise ApfsWorkspaceContractError("workspace marker must be at the mount root")
        if any(path.parent != self.mountpoint for path in self.work_paths()):
            raise ApfsWorkspaceContractError("work directories must be direct mount children")
        if len(set(self.work_paths())) != 3:
            raise ApfsWorkspaceContractError("workspace paths must be distinct")
        if (
            self.volume_name != APFS_WORKSPACE_VOLUME_NAME
            or self.capacity_bytes != APFS_WORKSPACE_CAPACITY_BYTES
            or self.budget_bytes != APFS_WORKSPACE_BUDGET_BYTES
            or self.minimum_backing_free_bytes != APFS_WORKSPACE_MIN_BACKING_FREE_BYTES
            or self.minimum_mount_free_bytes != APFS_WORKSPACE_MIN_MOUNT_FREE_BYTES
            or self.required_mode != APFS_WORKSPACE_MODE
        ):
            raise ApfsWorkspaceContractError(
                "workspace capacity, reserve, or ownership mode drifted"
            )
        if self.budget_bytes > self.capacity_bytes:
            raise ApfsWorkspaceContractError("workspace budget exceeds image capacity")

    def work_paths(self) -> tuple[Path, Path, Path]:
        return self.cargo_target, self.temporary, self.cache


def validate_marker_and_mount_observation(
    spec: ApfsWorkspaceSpec,
    marker: Mapping[str, Any],
    observation: Mapping[str, Any],
    *,
    planned_bytes: int,
    allow_uninitialized_mode: bool = False,
) -> None:
    """Validate already-collected mount facts without touching or mounting it."""
    spec.validate()
    if planned_bytes < 0 or planned_bytes > spec.budget_bytes:
        raise ApfsWorkspaceContractError("planned bytes exceed the bounded workspace budget")
    required_marker = {
        "schema_version",
        "schema_id",
        "campaign_id",
        "backing_bundle",
        "mountpoint",
        "volume_name",
        "volume_uuid",
        "filesystem",
        "capacity_bytes",
        "budget_bytes",
        "owner_uid",
        "owner_gid",
        "mode",
        "physical_backing_root",
        "mount_namespace_only",
    }
    if set(marker) != required_marker:
        raise ApfsWorkspaceContractError("workspace marker field set differs")
    if (
        marker["schema_version"] != 1
        or marker["schema_id"] != APFS_WORKSPACE_SCHEMA
        or marker["campaign_id"] != CAMPAIGN_ID
        or marker["backing_bundle"] != str(spec.backing_bundle)
        or marker["mountpoint"] != str(spec.mountpoint)
        or marker["volume_name"] != spec.volume_name
        or marker["filesystem"] != "apfs"
        or marker["capacity_bytes"] != spec.capacity_bytes
        or marker["budget_bytes"] != spec.budget_bytes
        or marker["mode"]
        not in ({"0700", "0755"} if allow_uninitialized_mode else {"0700"})
        or marker["physical_backing_root"] != str(spec.campaign_root)
        or marker["mount_namespace_only"] is not True
        or not isinstance(marker["owner_uid"], int)
        or not isinstance(marker["owner_gid"], int)
        or not isinstance(marker["volume_uuid"], str)
        or _UUID.fullmatch(marker["volume_uuid"]) is None
    ):
        raise ApfsWorkspaceContractError("workspace marker identity differs")
    required_observation = {
        "backing_bundle",
        "mountpoint",
        "volume_name",
        "volume_uuid",
        "filesystem",
        "capacity_bytes",
        "free_bytes",
        "owner_uid",
        "owner_gid",
        "mode",
        "read_only",
        "symlink_components",
        "backing_free_bytes",
    }
    if set(observation) != required_observation:
        raise ApfsWorkspaceContractError("mount observation field set differs")
    compared = (
        "backing_bundle",
        "mountpoint",
        "volume_name",
        "volume_uuid",
        "filesystem",
        "capacity_bytes",
        "owner_uid",
        "owner_gid",
        "mode",
    )
    if any(observation[name] != marker[name] for name in compared):
        raise ApfsWorkspaceContractError("mount observation and marker identity differ")
    if observation["read_only"] is not False or observation["symlink_components"] != []:
        raise ApfsWorkspaceContractError("workspace is read-only or traverses a symlink")
    if observation["free_bytes"] < max(spec.minimum_mount_free_bytes, planned_bytes):
        raise ApfsWorkspaceContractError("APFS workspace free-space floor failed")
    if observation["backing_free_bytes"] < spec.minimum_backing_free_bytes:
        raise ApfsWorkspaceContractError("backing SSD reserve floor failed")


def _require_contained(label: str, path: Path, root: Path) -> None:
    if not path.is_absolute():
        raise ApfsWorkspaceContractError(f"{label} path must be absolute")
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ApfsWorkspaceContractError(f"{label} escapes the campaign root") from error
    if not relative.parts or ".." in relative.parts:
        raise ApfsWorkspaceContractError(f"{label} is not a campaign child")
