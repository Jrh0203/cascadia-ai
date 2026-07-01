from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from cascadia_mlx.r2_map_apfs_workspace import (
    APFS_WORKSPACE_BUDGET_BYTES,
    APFS_WORKSPACE_CAPACITY_BYTES,
    APFS_WORKSPACE_MIN_BACKING_FREE_BYTES,
    APFS_WORKSPACE_MOUNTPOINT,
    APFS_WORKSPACE_SCHEMA,
    ApfsWorkspaceContractError,
    ApfsWorkspaceSpec,
    validate_marker_and_mount_observation,
)


def _spec() -> ApfsWorkspaceSpec:
    return ApfsWorkspaceSpec.historical_fixture(
        campaign_root=Path("/legacy/r2-map-v1"),
        mountpoint=APFS_WORKSPACE_MOUNTPOINT,
    )


def _marker(spec: ApfsWorkspaceSpec) -> dict[str, object]:
    return {
        "schema_version": 1,
        "schema_id": APFS_WORKSPACE_SCHEMA,
        "campaign_id": "r2-map-expert-iteration-v1",
        "backing_bundle": str(spec.backing_bundle),
        "mountpoint": str(spec.mountpoint),
        "volume_name": spec.volume_name,
        "volume_uuid": "01234567-89AB-CDEF-0123-456789ABCDEF",
        "filesystem": "apfs",
        "capacity_bytes": APFS_WORKSPACE_CAPACITY_BYTES,
        "budget_bytes": APFS_WORKSPACE_BUDGET_BYTES,
        "owner_uid": 501,
        "owner_gid": 20,
        "mode": "0700",
        "physical_backing_root": str(spec.campaign_root),
        "mount_namespace_only": True,
    }


def _observation(spec: ApfsWorkspaceSpec) -> dict[str, object]:
    value = dict(_marker(spec))
    for name in (
        "schema_version",
        "schema_id",
        "campaign_id",
        "budget_bytes",
        "physical_backing_root",
        "mount_namespace_only",
    ):
        del value[name]
    value.update(
        {
            "free_bytes": APFS_WORKSPACE_BUDGET_BYTES,
            "read_only": False,
            "symlink_components": [],
            "backing_free_bytes": APFS_WORKSPACE_MIN_BACKING_FREE_BYTES,
        }
    )
    return value


def test_historical_workspace_has_physically_contained_backing_and_exact_namespace() -> None:
    spec = _spec()
    spec.validate()
    validate_marker_and_mount_observation(
        spec,
        _marker(spec),
        _observation(spec),
        planned_bytes=APFS_WORKSPACE_BUDGET_BYTES,
    )
    assert spec.backing_bundle.is_relative_to(spec.campaign_root)
    assert spec.mountpoint == Path("/Volumes/CascadiaR2MapV1")
    assert all(spec.mountpoint in path.parents for path in spec.work_paths())


def test_active_campaign_default_is_retired() -> None:
    with pytest.raises(ApfsWorkspaceContractError, match="retired"):
        ApfsWorkspaceSpec.campaign_default()


def test_workspace_rejects_backing_outside_campaign() -> None:
    spec = _spec()
    drifted = replace(spec, backing_bundle=spec.campaign_root.parent / "escaped.sparsebundle")
    with pytest.raises(ApfsWorkspaceContractError, match="campaign root"):
        drifted.validate()


def test_workspace_rejects_mount_namespace_or_work_root_drift() -> None:
    spec = _spec()
    with pytest.raises(ApfsWorkspaceContractError, match="mount namespace"):
        replace(spec, mountpoint=Path("/Volumes/CascadiaR2MapV1-1")).validate()
    with pytest.raises(ApfsWorkspaceContractError, match="direct mount children"):
        replace(spec, cargo_target=spec.campaign_root / "cargo-target").validate()


@pytest.mark.parametrize(
    ("target", "field", "value", "message"),
    [
        ("marker", "filesystem", "exfat", "marker identity"),
        ("marker", "volume_uuid", "not-a-uuid", "marker identity"),
        ("observation", "mountpoint", "/tmp/r2-map", "observation and marker"),
        ("observation", "symlink_components", ["apfs-work"], "symlink"),
        ("observation", "read_only", True, "read-only"),
        ("observation", "free_bytes", 1, "free-space"),
        ("observation", "backing_free_bytes", 1, "reserve"),
    ],
)
def test_workspace_rejects_identity_storage_and_symlink_drift(
    target: str, field: str, value: object, message: str
) -> None:
    spec = _spec()
    marker = _marker(spec)
    observation = _observation(spec)
    (marker if target == "marker" else observation)[field] = value
    with pytest.raises(ApfsWorkspaceContractError, match=message):
        validate_marker_and_mount_observation(spec, marker, observation, planned_bytes=1024)
