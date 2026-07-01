"""No-follow host inventories and exact pre/post ledger comparison."""

from __future__ import annotations

import contextlib
import errno
import hashlib
import json
import os
import selectors
import signal
import stat
import subprocess
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .canonical import (
    CAMPAIGN_ID,
    FROZEN_RUNTIME,
    INVENTORY_SCHEMA,
    LEDGER_COMPARISON_SCHEMA,
    D0Error,
    document_sha256,
    sha256_bytes,
)

SAMPLE_BYTES = 64 * 1024
DEFAULT_FULL_HASH_LIMIT = 2 * 1024**3
DEFAULT_MAX_ENTRIES = 50_000
R2_MARKERS = (b"cascadia-r2", b"r2-map", b"r2_map")
RUNTIME_PROCESS_MARKERS = (
    "colima",
    "lima",
    "docker",
    "buildkit",
    "containerd",
    "podman",
    "gvproxy",
    "vfkit",
    "krunkit",
    "qemu-system",
)
DASHBOARD_WATCH_MARKER = "tools/r2_map_d0_dashboard_watch.py --watch"


@dataclass(frozen=True)
class InventoryPolicy:
    full_hash_limit: int = DEFAULT_FULL_HASH_LIMIT
    max_entries: int = DEFAULT_MAX_ENTRIES
    sample_bytes: int = SAMPLE_BYTES
    require_owner_uid: int | None = None
    reject_device_crossing: bool = True


def secure_owner_directory(path: Path, *, mode: int = 0o700) -> list[Path]:
    """Create an owner-private path through no-follow directory descriptors."""

    if not path.is_absolute() or path == Path("/"):
        raise D0Error("owner directory path is not an absolute descendant")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open("/", flags)
    created: list[Path] = []
    current = Path("/")
    try:
        for component in path.parts[1:]:
            current = current / component
            made = False
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except OSError as error:
                if error.errno != errno.ENOENT:
                    raise D0Error(f"owner directory component is unsafe: {current}") from error
                try:
                    os.mkdir(component, mode=mode, dir_fd=descriptor)
                    made = True
                    child = os.open(component, flags, dir_fd=descriptor)
                except OSError as create_error:
                    raise D0Error(
                        f"owner directory component could not be created safely: {current}"
                    ) from create_error
            try:
                observed = os.fstat(child)
                if not stat.S_ISDIR(observed.st_mode):
                    raise D0Error(f"owner directory component is not a directory: {current}")
                if made:
                    os.fchmod(child, mode)
                    observed = os.fstat(child)
                    if observed.st_uid != os.getuid() or stat.S_IMODE(observed.st_mode) != mode:
                        raise D0Error(f"created owner directory metadata differs: {current}")
                    created.append(current)
                if current == path and observed.st_uid != os.getuid():
                    raise D0Error("owner directory final component has the wrong owner")
            except BaseException:
                os.close(child)
                raise
            os.close(descriptor)
            descriptor = child
    finally:
        os.close(descriptor)
    return created


def _sha256_stream(descriptor: int, size: int) -> str:
    digest = hashlib.sha256()
    position = 0
    while position < size:
        chunk = os.pread(descriptor, min(1024 * 1024, size - position), position)
        if not chunk:
            raise D0Error("short read while hashing inventory file")
        digest.update(chunk)
        position += len(chunk)
    return digest.hexdigest()


def _sample_offsets(size: int, width: int) -> tuple[int, ...]:
    if size <= width:
        return (0,)
    maximum = size - width
    values = {0, maximum, maximum // 4, maximum // 2, (3 * maximum) // 4}
    return tuple(sorted(values))


def _sample_digest(descriptor: int, size: int, width: int) -> tuple[str, list[int]]:
    digest = hashlib.sha256()
    offsets = list(_sample_offsets(size, width))
    for offset in offsets:
        chunk = os.pread(descriptor, min(width, size - offset), offset)
        if len(chunk) != min(width, size - offset):
            raise D0Error("short read while sampling inventory file")
        digest.update(offset.to_bytes(8, "big"))
        digest.update(len(chunk).to_bytes(8, "big"))
        digest.update(chunk)
    return digest.hexdigest(), offsets


def _xattrs(path: Path) -> list[dict[str, Any]]:
    listxattr = getattr(os, "listxattr", None)
    getxattr = getattr(os, "getxattr", None)
    if listxattr is None or getxattr is None:
        return []
    try:
        names = sorted(listxattr(path, follow_symlinks=False))
    except OSError as error:
        raise D0Error(f"cannot enumerate xattrs for {path}") from error
    result: list[dict[str, Any]] = []
    for name in names:
        try:
            value = getxattr(path, name, follow_symlinks=False)
        except OSError as error:
            raise D0Error(f"cannot read xattr for {path}") from error
        result.append(
            {
                "name": name,
                "size": len(value),
                "sha256": hashlib.sha256(value).hexdigest(),
            }
        )
    return result


def _entry(path: Path, relative: str, root_device: int, policy: InventoryPolicy) -> dict[str, Any]:
    try:
        observed = path.lstat()
    except OSError as error:
        raise D0Error(f"cannot lstat inventory path: {path}") from error
    if policy.reject_device_crossing and observed.st_dev != root_device:
        raise D0Error(f"inventory crosses a device boundary: {path}")
    if policy.require_owner_uid is not None and observed.st_uid != policy.require_owner_uid:
        raise D0Error(f"inventory path has the wrong owner: {path}")
    kind = (
        "directory"
        if stat.S_ISDIR(observed.st_mode)
        else "file"
        if stat.S_ISREG(observed.st_mode)
        else "symlink"
        if stat.S_ISLNK(observed.st_mode)
        else "socket"
        if stat.S_ISSOCK(observed.st_mode)
        else "fifo"
        if stat.S_ISFIFO(observed.st_mode)
        else "device"
        if stat.S_ISCHR(observed.st_mode) or stat.S_ISBLK(observed.st_mode)
        else "other"
    )
    value: dict[str, Any] = {
        "relative": relative,
        "type": kind,
        "mode": f"{stat.S_IMODE(observed.st_mode):04o}",
        "uid": observed.st_uid,
        "gid": observed.st_gid,
        "device": observed.st_dev,
        "inode": observed.st_ino,
        "nlink": observed.st_nlink,
        "size": observed.st_size,
        "allocated_bytes": observed.st_blocks * 512,
        "mtime_ns": observed.st_mtime_ns,
        "ctime_ns": observed.st_ctime_ns,
        "birthtime_ns": int(getattr(observed, "st_birthtime", 0) * 1_000_000_000),
        "xattrs": _xattrs(path),
    }
    if kind == "symlink":
        value["target"] = os.readlink(path)
        return value
    if kind != "file":
        return value
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise D0Error(f"cannot no-follow open inventory file: {path}") from error
    try:
        reopened = os.fstat(descriptor)
        if (reopened.st_dev, reopened.st_ino, reopened.st_size) != (
            observed.st_dev,
            observed.st_ino,
            observed.st_size,
        ):
            raise D0Error(f"inventory file changed during open: {path}")
        if observed.st_size <= policy.full_hash_limit:
            value["content_sha256"] = _sha256_stream(descriptor, observed.st_size)
            value["sample_sha256"] = None
            value["sample_offsets"] = []
            scan = os.pread(descriptor, min(observed.st_size, 4 * 1024 * 1024), 0).lower()
        else:
            sample, offsets = _sample_digest(descriptor, observed.st_size, policy.sample_bytes)
            value["content_sha256"] = None
            value["sample_sha256"] = sample
            value["sample_offsets"] = offsets
            scan = b"".join(
                os.pread(descriptor, min(policy.sample_bytes, observed.st_size - offset), offset)
                for offset in offsets
            ).lower()
        value["inspectable_r2_marker"] = any(marker in scan for marker in R2_MARKERS)
    finally:
        os.close(descriptor)
    return value


def _validate_root(path: Path) -> tuple[Path, os.stat_result | None]:
    if not path.is_absolute() or PurePosixPath(str(path)).as_posix() != str(path):
        raise D0Error("inventory root is not canonical and absolute")
    current = Path("/")
    for part in path.parts[1:]:
        current /= part
        try:
            observed = current.lstat()
        except FileNotFoundError:
            return path, None
        except OSError as error:
            raise D0Error(f"cannot inspect inventory root ancestor: {current}") from error
        if stat.S_ISLNK(observed.st_mode) and current != path:
            raise D0Error(f"inventory root ancestor is a symlink: {current}")
        if current != path and not stat.S_ISDIR(observed.st_mode):
            raise D0Error(f"inventory root ancestor is not a directory: {current}")
    try:
        return path, path.lstat()
    except FileNotFoundError:
        return path, None


def inventory_roots(
    roots: Sequence[Path],
    *,
    label: str,
    policy: InventoryPolicy | None = None,
) -> dict[str, Any]:
    selected = policy or InventoryPolicy()
    normalized = sorted({str(path) for path in roots})
    if len(normalized) != len(roots):
        raise D0Error("inventory roots are duplicated")
    root_reports: list[dict[str, Any]] = []
    total_entries = 0
    for raw in normalized:
        root, root_stat = _validate_root(Path(raw))
        if root_stat is None:
            root_reports.append({"root": raw, "present": False, "entries": []})
            continue
        root_device = root_stat.st_dev
        entries = [_entry(root, ".", root_device, selected)]
        if stat.S_ISDIR(root_stat.st_mode):
            pending = [root]
            while pending:
                directory = pending.pop()
                try:
                    children = sorted(
                        os.scandir(directory), key=lambda item: os.fsencode(item.name)
                    )
                except OSError as error:
                    raise D0Error(f"cannot scan inventory directory: {directory}") from error
                for child in children:
                    path = Path(child.path)
                    relative = path.relative_to(root).as_posix()
                    item = _entry(path, relative, root_device, selected)
                    entries.append(item)
                    if item["type"] == "directory":
                        pending.append(path)
                    total_entries += 1
                    if total_entries > selected.max_entries:
                        raise D0Error("inventory exceeds its entry limit")
        entries.sort(key=lambda item: item["relative"])
        root_reports.append({"root": raw, "present": True, "entries": entries})
    totals = {
        "roots": len(root_reports),
        "present_roots": sum(1 for root in root_reports if root["present"]),
        "entries": sum(len(root["entries"]) for root in root_reports),
        "apparent_bytes": sum(entry["size"] for root in root_reports for entry in root["entries"]),
        "allocated_bytes": sum(
            entry["allocated_bytes"] for root in root_reports for entry in root["entries"]
        ),
        "r2_marker_entries": sum(
            1
            for root in root_reports
            for entry in root["entries"]
            if entry.get("inspectable_r2_marker")
            or any(marker.decode("ascii") in entry["relative"].lower() for marker in R2_MARKERS)
        ),
    }
    report: dict[str, Any] = {
        "schema_id": INVENTORY_SCHEMA,
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "label": label,
        "collected_unix_ms": time.time_ns() // 1_000_000,
        "hash_policy": {
            "algorithm": "sha256",
            "full_hash_limit": selected.full_hash_limit,
            "large_file_sample_bytes": selected.sample_bytes,
            "large_file_sample_positions": "start,quarter,middle,three-quarter,end",
        },
        "roots": root_reports,
        "totals": totals,
    }
    report["inventory_sha256"] = document_sha256(report, "inventory_sha256")
    return report


def _contained_path(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def inventory_managed_homebrew_link(
    requested_path: Path,
    *,
    managed_link: Path,
    cellar_root: Path,
    formula: str,
    version: str,
    managed_target_relative: Path,
    requested_suffix: Path,
    installed_file_relative: Path,
    managed_link_target: str,
    requested_link_target: str,
    install_receipt_sha256: str,
    installed_file_sha256: str,
    label: str,
    policy: InventoryPolicy | None = None,
) -> dict[str, Any]:
    """Inventory one explicitly pinned Homebrew-managed symlink chain.

    Generic inventory continues to reject every symlink ancestor.  This narrow
    path records the public Homebrew symlink itself and inventories its resolved
    immutable keg only after proving the exact relative targets, formula,
    version, ownership receipt, containment, and absence of any extra symlink
    ancestor.
    """

    selected = policy or InventoryPolicy()
    keg_root = cellar_root / formula / version
    expected_managed_target = keg_root / managed_target_relative
    expected_installed_file = keg_root / installed_file_relative
    if (
        not requested_path.is_absolute()
        or not managed_link.is_absolute()
        or not cellar_root.is_absolute()
        or requested_path != managed_link / requested_suffix
        or PurePosixPath(managed_link_target).is_absolute()
        or PurePosixPath(requested_link_target).is_absolute()
        or any(part in {"", ".", ".."} for part in managed_target_relative.parts)
        or any(part in {"", ".", ".."} for part in requested_suffix.parts)
        or any(part in {"", ".", ".."} for part in installed_file_relative.parts)
    ):
        raise D0Error("managed Homebrew link authorization differs")
    parent, parent_stat = _validate_root(managed_link.parent)
    if parent_stat is None or parent != managed_link.parent:
        raise D0Error("managed Homebrew link parent is absent")
    keg, keg_stat = _validate_root(keg_root)
    if keg_stat is None or keg != keg_root or not stat.S_ISDIR(keg_stat.st_mode):
        raise D0Error("managed Homebrew keg root is absent or unsafe")

    def resolve_strict(path: Path, label_value: str) -> Path:
        try:
            return path.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise D0Error(f"managed Homebrew {label_value} is dangling or cyclic") from error

    def state() -> dict[str, Any]:
        link_stat = managed_link.lstat()
        if not stat.S_ISLNK(link_stat.st_mode):
            raise D0Error("managed Homebrew public entry is not a symlink")
        link_target = os.readlink(managed_link)
        if link_target != managed_link_target or PurePosixPath(link_target).is_absolute():
            raise D0Error("managed Homebrew public symlink target differs")
        resolved_managed = resolve_strict(managed_link, "public symlink")
        if (
            resolved_managed != expected_managed_target
            or not _contained_path(resolved_managed, keg_root)
        ):
            raise D0Error("managed Homebrew public symlink escapes its pinned keg")

        current = resolved_managed
        for component in requested_suffix.parts[:-1]:
            current /= component
            observed = current.lstat()
            if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
                raise D0Error("managed Homebrew resolved chain has an extra symlink")
        physical_requested = resolved_managed / requested_suffix
        requested_stat = physical_requested.lstat()
        if not stat.S_ISLNK(requested_stat.st_mode):
            raise D0Error("managed Homebrew requested entry is not a symlink")
        final_target = os.readlink(physical_requested)
        if final_target != requested_link_target or PurePosixPath(final_target).is_absolute():
            raise D0Error("managed Homebrew requested symlink target differs")
        resolved_file = resolve_strict(physical_requested, "requested symlink")
        if (
            resolved_file != expected_installed_file
            or not _contained_path(resolved_file, keg_root)
        ):
            raise D0Error("managed Homebrew requested symlink escapes its pinned keg")
        resolved_stat = resolved_file.lstat()
        if not stat.S_ISREG(resolved_stat.st_mode):
            raise D0Error("managed Homebrew requested target is not a regular file")
        receipt = keg_root / "INSTALL_RECEIPT.json"
        receipt_stat = receipt.lstat()
        if not stat.S_ISREG(receipt_stat.st_mode):
            raise D0Error("managed Homebrew install receipt is not a regular file")
        owners = {
            link_stat.st_uid,
            keg_stat.st_uid,
            requested_stat.st_uid,
            resolved_stat.st_uid,
            receipt_stat.st_uid,
        }
        if owners != {os.getuid()}:
            raise D0Error("managed Homebrew link or keg owner differs")
        if any(
            observed.st_mode & 0o022
            for observed in (link_stat, keg_stat, requested_stat, resolved_stat, receipt_stat)
        ):
            raise D0Error("managed Homebrew link or keg is group/world writable")
        link_entry = _entry(managed_link, ".", parent_stat.st_dev, selected)
        requested_entry = _entry(
            physical_requested,
            requested_suffix.as_posix(),
            keg_stat.st_dev,
            selected,
        )
        resolved_entry = _entry(
            resolved_file,
            installed_file_relative.as_posix(),
            keg_stat.st_dev,
            selected,
        )
        receipt_entry = _entry(receipt, "INSTALL_RECEIPT.json", keg_stat.st_dev, selected)
        if (
            receipt_entry.get("content_sha256") != install_receipt_sha256
            or resolved_entry.get("content_sha256") != installed_file_sha256
        ):
            raise D0Error("managed Homebrew receipt or installed file identity differs")
        return {
            "public_symlink": link_entry,
            "public_symlink_target_sha256": sha256_bytes(link_target.encode("utf-8")),
            "resolved_public_target": str(resolved_managed),
            "requested_symlink": requested_entry,
            "requested_symlink_target_sha256": sha256_bytes(final_target.encode("utf-8")),
            "resolved_installed_file": resolved_entry,
            "install_receipt": receipt_entry,
        }

    before = state()
    resolved_tree = inventory_roots(
        [keg_root],
        label=f"{label}-resolved-keg",
        policy=selected,
    )
    after = state()
    if before != after:
        raise D0Error("managed Homebrew link identity changed during inventory")
    result: dict[str, Any] = {
        "schema_id": "cascadia.r2-map.d0-managed-homebrew-link.v1",
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "label": label,
        "requested_path": str(requested_path),
        "formula": formula,
        "version": version,
        "cellar_root": str(cellar_root),
        "keg_root": str(keg_root),
        "before": before,
        "after": after,
        "identity_stable": True,
        "resolved_keg_inventory": resolved_tree,
        "status": "pass",
    }
    result["managed_link_sha256"] = document_sha256(result, "managed_link_sha256")
    return result


def inventory_identity(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("schema_id") != INVENTORY_SCHEMA:
        raise D0Error("inventory schema differs")
    return {
        "hash_policy": report.get("hash_policy"),
        "roots": report.get("roots"),
        "totals": report.get("totals"),
        "managed_homebrew_links": report.get("managed_homebrew_links"),
    }


def compare_inventories(
    before: dict[str, Any], after: dict[str, Any], *, label: str
) -> dict[str, Any]:
    before_identity = inventory_identity(before)
    after_identity = inventory_identity(after)
    matches = before_identity == after_identity
    report: dict[str, Any] = {
        "schema_id": LEDGER_COMPARISON_SCHEMA,
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "label": label,
        "before_inventory_sha256": before.get("inventory_sha256"),
        "after_inventory_sha256": after.get("inventory_sha256"),
        "identity_stable": matches,
        "status": "pass" if matches else "fail",
    }
    report["comparison_sha256"] = document_sha256(report, "comparison_sha256")
    return report


def compare_homebrew_ledger(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    allowed_new_roots: Iterable[str],
    label: str,
) -> dict[str, Any]:
    allowed = tuple(sorted(allowed_new_roots))
    before_roots = {item["root"]: item for item in before.get("roots", [])}
    after_roots = {item["root"]: item for item in after.get("roots", [])}
    if set(before_roots) != set(after_roots):
        raise D0Error("Homebrew ledger root set differs")
    drift: list[str] = []
    created: list[str] = []

    def stable_root(value: dict[str, Any]) -> dict[str, Any]:
        normalized = {"root": value["root"], "present": value["present"], "entries": []}
        for entry in value["entries"]:
            item = dict(entry)
            for field in ("inode", "mtime_ns", "ctime_ns", "birthtime_ns"):
                item.pop(field, None)
            normalized["entries"].append(item)
        return normalized

    for root in sorted(before_roots):
        old = before_roots[root]
        new = after_roots[root]
        if stable_root(old) == stable_root(new):
            continue
        if (
            not old["present"]
            and new["present"]
            and any(
                root == prefix or root.startswith(prefix.rstrip("/") + "/") for prefix in allowed
            )
        ):
            created.append(root)
            continue
        drift.append(root)
    report: dict[str, Any] = {
        "schema_id": LEDGER_COMPARISON_SCHEMA,
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "label": label,
        "before_inventory_sha256": before.get("inventory_sha256"),
        "after_inventory_sha256": after.get("inventory_sha256"),
        "allowed_new_roots": list(allowed),
        "created_roots": created,
        "drift_roots": drift,
        "status": "pass" if not drift else "fail",
    }
    report["comparison_sha256"] = document_sha256(report, "comparison_sha256")
    return report


def compare_homebrew_install(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    allowed_formulae: Iterable[str],
    allowed_isolated_roots: Iterable[str],
    label: str,
) -> dict[str, Any]:
    """Permit only projections of pinned kegs plus isolated D0 roots."""

    formulae = tuple(sorted(allowed_formulae))
    if not formulae or any(name not in FROZEN_RUNTIME for name in formulae):
        raise D0Error("Homebrew install comparison formula set differs")
    isolated = frozenset(allowed_isolated_roots)
    before_roots = {item["root"]: item for item in before.get("roots", [])}
    after_roots = {item["root"]: item for item in after.get("roots", [])}
    if set(before_roots) != set(after_roots):
        raise D0Error("Homebrew install ledger root set differs")
    changed_existing: list[str] = []
    disallowed_new: list[str] = []
    allowed_new: list[str] = []

    def normalized_existing(entry: Mapping[str, Any]) -> dict[str, Any]:
        value = dict(entry)
        if value.get("type") == "directory":
            for field in (
                "inode",
                "nlink",
                "size",
                "allocated_bytes",
                "mtime_ns",
                "ctime_ns",
                "birthtime_ns",
            ):
                value.pop(field, None)
        return value

    def projected_symlink(root: str, relative: str, entry: Mapping[str, Any]) -> bool:
        if entry.get("type") != "symlink" or not isinstance(entry.get("target"), str):
            return False
        target = entry["target"]
        parent = os.path.dirname(os.path.join(root, relative))
        resolved = os.path.normpath(
            target if target.startswith("/") else os.path.join(parent, target)
        )
        return any(
            resolved == f"/opt/homebrew/Cellar/{formula}/{FROZEN_RUNTIME[formula]['version']}"
            or resolved.startswith(
                f"/opt/homebrew/Cellar/{formula}/{FROZEN_RUNTIME[formula]['version']}/"
            )
            for formula in formulae
        )

    for root in sorted(before_roots):
        old_root = before_roots[root]
        new_root = after_roots[root]
        old_entries = {item["relative"]: item for item in old_root["entries"]}
        new_entries = {item["relative"]: item for item in new_root["entries"]}
        for relative, old_entry in old_entries.items():
            new_entry = new_entries.get(relative)
            if new_entry is None or normalized_existing(old_entry) != normalized_existing(
                new_entry
            ):
                changed_existing.append(f"{root}:{relative}")
        created = {
            relative: entry
            for relative, entry in new_entries.items()
            if relative not in old_entries
        }
        if root in isolated:
            allowed_new.extend(f"{root}:{relative}" for relative in sorted(created))
            continue
        for relative, entry in sorted(created.items()):
            permitted = False
            if root == "/opt/homebrew/Cellar":
                permitted = any(
                    relative == formula
                    or relative == f"{formula}/{FROZEN_RUNTIME[formula]['version']}"
                    or relative.startswith(f"{formula}/{FROZEN_RUNTIME[formula]['version']}/")
                    for formula in formulae
                )
            elif entry.get("type") == "directory":
                descendants = [
                    child
                    for child_relative, child in created.items()
                    if child_relative.startswith(relative.rstrip("/") + "/")
                    and child.get("type") != "directory"
                ]
                permitted = bool(descendants) and all(
                    projected_symlink(root, child_relative, child)
                    for child_relative, child in created.items()
                    if child_relative.startswith(relative.rstrip("/") + "/")
                    and child.get("type") != "directory"
                )
            else:
                permitted = projected_symlink(root, relative, entry)
            target = allowed_new if permitted else disallowed_new
            target.append(f"{root}:{relative}")
    cellar = after_roots.get("/opt/homebrew/Cellar")
    if cellar is None:
        raise D0Error("Homebrew install ledger omits the Cellar root")
    cellar_entries = {item["relative"]: item for item in cellar["entries"]}
    for formula in formulae:
        version_root = f"{formula}/{FROZEN_RUNTIME[formula]['version']}"
        direct_versions = {
            relative
            for relative in cellar_entries
            if relative.startswith(formula + "/") and relative.count("/") == 1
        }
        if (
            formula not in cellar_entries
            or version_root not in cellar_entries
            or direct_versions != {version_root}
            or cellar_entries[formula].get("type") != "directory"
            or cellar_entries[version_root].get("type") != "directory"
        ):
            disallowed_new.append(f"/opt/homebrew/Cellar:{formula}:version-set-differs")
    status = "pass" if not changed_existing and not disallowed_new else "fail"
    report: dict[str, Any] = {
        "schema_id": LEDGER_COMPARISON_SCHEMA,
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "label": label,
        "before_inventory_sha256": before.get("inventory_sha256"),
        "after_inventory_sha256": after.get("inventory_sha256"),
        "allowed_formulae": list(formulae),
        "allowed_isolated_roots": sorted(isolated),
        "allowed_new_entries": allowed_new,
        "changed_existing_entries": changed_existing,
        "disallowed_new_entries": disallowed_new,
        "status": status,
    }
    report["comparison_sha256"] = document_sha256(report, "comparison_sha256")
    return report


def _run_lines(
    argv: list[str],
    *,
    allowed_returncodes: frozenset[int] = frozenset({0}),
) -> list[str]:
    maximum = 8 * 1024 * 1024
    deadline = time.monotonic() + 30
    try:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={"LC_ALL": "C", "PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
            start_new_session=True,
        )
    except OSError as error:
        raise D0Error(f"cannot execute inventory command: {argv[0]}") from error
    assert process.stdout is not None
    assert process.stderr is not None
    output = bytearray()
    error_output = bytearray()
    selector = selectors.DefaultSelector()
    failure: D0Error | None = None
    try:
        for stream, target in ((process.stdout, output), (process.stderr, error_output)):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, target)
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                failure = D0Error(f"inventory command timed out: {argv[0]}")
                break
            for key, _mask in selector.select(min(remaining, 0.25)):
                stream = key.fileobj
                try:
                    chunk = os.read(stream.fileno(), 64 * 1024)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(stream)
                    stream.close()
                    continue
                if len(output) + len(error_output) + len(chunk) > maximum:
                    failure = D0Error(f"inventory command output is too large: {argv[0]}")
                    break
                key.data.extend(chunk)
            if failure is not None:
                break
        if failure is None:
            try:
                process.wait(timeout=max(0.001, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                failure = D0Error(f"inventory command timed out: {argv[0]}")
    finally:
        selector.close()
        if failure is not None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        for stream in (process.stdout, process.stderr):
            if not stream.closed:
                stream.close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    if failure is not None:
        raise failure
    if process.returncode not in allowed_returncodes or error_output:
        raise D0Error(f"inventory command failed: {argv[0]}")
    try:
        text = output.decode("utf-8")
    except UnicodeDecodeError as error:
        raise D0Error(f"inventory command output is not UTF-8: {argv[0]}") from error
    return [line.strip() for line in text.splitlines() if line.strip()]


def runtime_activity() -> dict[str, Any]:
    process_lines = _run_lines(["/bin/ps", "-axo", "pid=,ppid=,user=,command="])
    processes = [
        line
        for line in process_lines
        if any(marker in line.lower() for marker in RUNTIME_PROCESS_MARKERS)
    ]
    observer_ancestors = [
        line for line in process_lines if DASHBOARD_WATCH_MARKER in line
    ]
    launchd = [
        line
        for line in _run_lines(["/bin/launchctl", "list"])
        if any(marker in line.lower() for marker in RUNTIME_PROCESS_MARKERS)
    ]
    sockets = [
        line
        for line in _run_lines(["/usr/sbin/lsof", "-nP", "-U"])
        if any(marker in line.lower() for marker in RUNTIME_PROCESS_MARKERS)
    ]
    tcp_listeners = [
        line
        for line in _run_lines(
            ["/usr/sbin/lsof", "-nP", "-iTCP", "-sTCP:LISTEN"],
            allowed_returncodes=frozenset({0, 1}),
        )
        if any(marker in line.lower() for marker in RUNTIME_PROCESS_MARKERS)
    ]
    mounts = [
        line
        for line in _run_lines(["/sbin/mount"])
        if any(marker in line.lower() for marker in RUNTIME_PROCESS_MARKERS)
    ]
    return {
        "processes": processes,
        "observer_ancestors": observer_ancestors,
        "launchd": launchd,
        "active_unix_sockets": sockets,
        "active_tcp_listeners": tcp_listeners,
        "mounts": mounts,
        "inactive": (
            not processes and not launchd and not sockets and not tcp_listeners and not mounts
        ),
    }


def selected_runtime_paths(home: Path) -> list[Path]:
    """Return only campaign-owned mutable runtime state.

    The pinned Homebrew formulae under ``/opt/homebrew`` are immutable,
    pre-existing installer inputs shared by D0 epochs.  They are positively
    identified by the runtime helper and deliberately do not belong to the
    absent-baseline or rollback namespace.
    """

    return [
        home / ".local/share/cascadia-r2/colima",
        home / "Library/Caches/cascadia-r2/colima",
        home / ".config/cascadia-r2/docker",
    ]


def podman_baseline_paths(home: Path) -> list[Path]:
    """Return disclosed Podman installation plus post-cleanup state roots."""

    return [
        Path("/opt/homebrew/Cellar/podman"),
        Path("/opt/homebrew/opt/podman"),
        Path("/opt/homebrew/bin/podman"),
        home / ".config/containers",
        home / ".local/share/containers",
    ]


PODMAN_EMPTY_CONNECTIONS = b'{"Connection":{},"Farm":{}}\n'


def podman_negative_control(
    home: Path,
    *,
    installation_paths: Sequence[Path] | None = None,
) -> dict[str, Any]:
    """Prove the frozen post-cleanup no-machine/no-storage Podman state.

    The unrelated Homebrew formula/CLI is disclosed but not treated as absent.
    The semantic gate intentionally does not compare the deleted VM tree or
    inode/timestamp identities from before the user-authorized cleanup.
    """

    config_root = home / ".config/containers"
    local_root = home / ".local/share/containers"
    expected = {
        config_root: {
            ".": ("directory", "0755", None),
            "podman": ("directory", "0755", None),
            "podman/machine": ("directory", "0755", None),
            "podman/machine/applehv": ("directory", "0755", None),
            "podman-connections.json": (
                "file",
                "0644",
                sha256_bytes(PODMAN_EMPTY_CONNECTIONS),
            ),
            "podman-connections.json.lock": ("file", "0644", sha256_bytes(b"")),
        },
        local_root: {
            ".": ("directory", "0755", None),
            "cache": ("directory", "0700", None),
            "podman": ("directory", "0755", None),
            "podman/machine": ("directory", "0755", None),
            "podman/machine/applehv": ("directory", "0755", None),
            "podman/machine/applehv/cache": ("directory", "0755", None),
        },
    }
    state_inventory = inventory_roots(
        list(expected),
        label=f"{CAMPAIGN_ID}-john1-podman-post-cleanup-state",
        policy=InventoryPolicy(full_hash_limit=1024 * 1024, max_entries=64),
    )
    projections: list[dict[str, Any]] = []
    by_root = {Path(item["root"]): item for item in state_inventory["roots"]}
    for root, required in expected.items():
        root_report = by_root.get(root)
        if root_report is None or root_report.get("present") is not True:
            raise D0Error(f"post-cleanup Podman state root is absent: {root}")
        observed = {entry["relative"]: entry for entry in root_report["entries"]}
        if set(observed) != set(required):
            raise D0Error("post-cleanup Podman state contains an unexpected path")
        for relative, (kind, mode, digest) in required.items():
            entry = observed[relative]
            if (
                entry.get("type") != kind
                or entry.get("mode") != mode
                or entry.get("uid") != os.getuid()
                or (kind == "file" and entry.get("content_sha256") != digest)
            ):
                raise D0Error("post-cleanup Podman state semantics differ")
            projections.append(
                {
                    "root": str(root),
                    "relative": relative,
                    "type": kind,
                    "mode": mode,
                    "uid": entry["uid"],
                    "content_sha256": digest,
                }
            )
    try:
        connection = json.loads((config_root / "podman-connections.json").read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D0Error("post-cleanup Podman connection document is invalid") from error
    if connection != {"Connection": {}, "Farm": {}}:
        raise D0Error("post-cleanup Podman connection document is not empty")
    disclosed = inventory_roots(
        list(
            installation_paths
            if installation_paths is not None
            else (
                Path("/opt/homebrew/Cellar/podman"),
                Path("/opt/homebrew/opt/podman"),
                Path("/opt/homebrew/bin/podman"),
            )
        ),
        label=f"{CAMPAIGN_ID}-john1-podman-installation-disclosure",
        policy=InventoryPolicy(full_hash_limit=64 * 1024 * 1024, max_entries=50_000),
    )
    semantic = {
        "connection_maps_empty": True,
        "machine_records": 0,
        "machine_disks": 0,
        "socket_entries": 0,
        "storage_payload_files": 0,
        "state_projection": sorted(
            projections,
            key=lambda item: (item["root"], item["relative"]),
        ),
    }
    return {
        "schema_id": "cascadia.r2-map.d0-podman-negative-control.v2",
        "schema_version": 2,
        "campaign_id": CAMPAIGN_ID,
        "installation_disclosure": disclosed,
        "state_inventory": state_inventory,
        "semantic": semantic,
        "semantic_sha256": document_sha256(semantic, "semantic_sha256"),
        "status": "pass",
    }
