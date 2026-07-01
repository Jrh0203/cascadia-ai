#!/usr/bin/env python3
"""Plan and operate the exact contained APFS workspace lifecycle.

Importing or running `plan`/`status` performs no mount mutation. Mutating
commands require `--execute`; create and attach additionally require a verified
safe host receipt. This tool never chooses alternative paths or devices.
"""

from __future__ import annotations

import argparse
import json
import os
import plistlib
import re
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPOSITORY / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from cascadia_mlx.r2_map_apfs_lifecycle import (  # noqa: E402
    ApfsLifecycleError,
    ApfsWorkspaceLifecycle,
    CommandResult,
)
from cascadia_mlx.r2_map_apfs_workspace import ApfsWorkspaceSpec  # noqa: E402

_SWAP = re.compile(r"\bused\s*=\s*([0-9]+(?:\.[0-9]+)?)([KMG])\b")
_UNIT = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}


def _syspolicyd_rss_bytes() -> int:
    completed = subprocess.run(
        ["/bin/ps", "-axo", "rss=,command="],
        check=True,
        capture_output=True,
        text=True,
    )
    values = []
    for line in completed.stdout.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) == 2 and fields[1] == "/usr/libexec/syspolicyd":
            values.append(int(fields[0]) * 1024)
    if len(values) > 1:
        raise ApfsLifecycleError("multiple syspolicyd processes were observed")
    return values[0] if values else 0


def _swap_used_bytes() -> int:
    completed = subprocess.run(
        ["/usr/sbin/sysctl", "-n", "vm.swapusage"],
        check=True,
        capture_output=True,
        text=True,
    )
    match = _SWAP.search(completed.stdout)
    if match is None:
        raise ApfsLifecycleError("vm.swapusage output is unrecognized")
    return round(float(match.group(1)) * _UNIT[match.group(2)])


def _memory_pressure_level() -> int:
    completed = subprocess.run(
        ["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        return int(completed.stdout.strip())
    except ValueError as error:
        raise ApfsLifecycleError("memory pressure level output is unrecognized") from error


class ApfsBootstrapMonitor:
    """Abort built-in image operations at the frozen infrastructure stops."""

    def __init__(self, spec: ApfsWorkspaceSpec):
        self.spec = spec
        self.swap_baseline_bytes = _swap_used_bytes()

    def check(self) -> None:
        rss = _syspolicyd_rss_bytes()
        swap = _swap_used_bytes()
        pressure = _memory_pressure_level()
        backing_free = shutil.disk_usage(self.spec.campaign_root).free
        failures = []
        if rss >= 4 * (1 << 30):
            failures.append(f"syspolicyd RSS {rss} reached the 4-GiB stop")
        if swap > self.swap_baseline_bytes:
            failures.append(
                f"system swap grew from {self.swap_baseline_bytes} to {swap} bytes"
            )
        if pressure != 1:
            failures.append(f"memory pressure level became {pressure}")
        if backing_free < self.spec.minimum_backing_free_bytes:
            failures.append(f"backing free space fell to {backing_free} bytes")
        if failures:
            raise ApfsLifecycleError("; ".join(failures))


class SystemRunner:
    def __init__(self, spec: ApfsWorkspaceSpec):
        self.spec = spec

    def __call__(self, argv: tuple[str, ...]) -> CommandResult:
        monitored = argv[:2] in {
            ("/usr/bin/hdiutil", "create"),
            ("/usr/bin/hdiutil", "attach"),
        }
        if not monitored:
            completed = subprocess.run(argv, check=False, capture_output=True, text=True)
            return CommandResult(
                argv=argv,
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
        monitor = ApfsBootstrapMonitor(self.spec)
        try:
            monitor.check()
        except ApfsLifecycleError as error:
            return CommandResult(argv=argv, returncode=70, stderr=str(error))
        process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        abort: str | None = None
        while process.poll() is None:
            time.sleep(0.25)
            try:
                monitor.check()
            except ApfsLifecycleError as error:
                abort = str(error)
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                break
        stdout, stderr = process.communicate()
        if abort is None:
            try:
                monitor.check()
            except ApfsLifecycleError as error:
                abort = str(error)
        returncode = process.returncode if abort is None else 70
        return CommandResult(
            argv=argv,
            returncode=returncode,
            stdout=stdout,
            stderr=(stderr + (f"\nAPFS bootstrap monitor: {abort}" if abort else "")),
        )


class MacOSObservationProvider:
    """Collect already-mounted facts; never attach, detach, or repair."""

    def __call__(self, spec: ApfsWorkspaceSpec) -> dict[str, Any] | None:
        if not os.path.ismount(spec.mountpoint):
            return None
        hdiutil = _plist_command(("/usr/bin/hdiutil", "info", "-plist"))
        image, entity = _find_image_entity(hdiutil, spec)
        disk = _plist_command(("/usr/sbin/diskutil", "info", "-plist", str(spec.mountpoint)))
        metadata = spec.mountpoint.stat()
        usage = shutil.disk_usage(spec.mountpoint)
        backing_usage = shutil.disk_usage(spec.campaign_root)
        mountpoint = str(spec.mountpoint)
        device = str(entity.get("dev-entry") or disk.get("DeviceNode") or "")
        observation = {
            "backing_bundle": str(Path(image["image-path"])),
            "mountpoint": str(disk.get("MountPoint") or entity.get("mount-point") or ""),
            "volume_name": str(disk.get("VolumeName") or ""),
            "volume_uuid": str(disk.get("VolumeUUID") or ""),
            "filesystem": str(disk.get("FilesystemType") or "").lower(),
            "capacity_bytes": int(disk.get("TotalSize") or usage.total),
            "free_bytes": int(disk.get("FreeSpace") or usage.free),
            "owner_uid": metadata.st_uid,
            "owner_gid": metadata.st_gid,
            "mode": f"{stat.S_IMODE(metadata.st_mode):04o}",
            "read_only": bool(disk.get("ReadOnly", False)),
            "symlink_components": _symlink_components(spec.mountpoint, stop=spec.campaign_root),
            "backing_free_bytes": backing_usage.free,
        }
        if observation["mountpoint"] != mountpoint:
            raise ApfsLifecycleError("diskutil mountpoint differs from the exact workspace")
        return {"device": device, "contract": observation}


def _plist_command(argv: tuple[str, ...]) -> dict[str, Any]:
    completed = subprocess.run(argv, check=False, capture_output=True)
    if completed.returncode != 0:
        raise ApfsLifecycleError(f"observation command failed: {argv!r}")
    try:
        value = plistlib.loads(completed.stdout)
    except plistlib.InvalidFileException as error:
        raise ApfsLifecycleError(f"observation command returned invalid plist: {argv!r}") from error
    if not isinstance(value, dict):
        raise ApfsLifecycleError("observation plist is not a dictionary")
    return value


def _find_image_entity(
    info: dict[str, Any], spec: ApfsWorkspaceSpec
) -> tuple[dict[str, Any], dict[str, Any]]:
    matches = []
    for image in info.get("images", []):
        if not isinstance(image, dict) or image.get("image-path") != str(spec.backing_bundle):
            continue
        for entity in image.get("system-entities", []):
            if isinstance(entity, dict) and entity.get("mount-point") == str(spec.mountpoint):
                matches.append((image, entity))
    if len(matches) != 1:
        raise ApfsLifecycleError("hdiutil does not name exactly one expected image mount")
    return matches[0]


def _symlink_components(path: Path, *, stop: Path) -> list[str]:
    components = []
    cursor = path
    while True:
        if cursor.is_symlink():
            components.append(str(cursor))
        if cursor == stop or cursor == cursor.parent:
            break
        cursor = cursor.parent
    return list(reversed(components))


def _lifecycle() -> tuple[ApfsWorkspaceLifecycle, MacOSObservationProvider]:
    spec = ApfsWorkspaceSpec.campaign_default()
    return (
        ApfsWorkspaceLifecycle(
            spec,
            mounted_checker=os.path.ismount,
            current_uid=os.getuid(),
            current_gid=os.getgid(),
        ),
        MacOSObservationProvider(),
    )


def _json(value: object) -> None:
    print(json.dumps(value, sort_keys=True, indent=2))


def _require_execute(arguments: argparse.Namespace) -> None:
    if not arguments.execute:
        raise ApfsLifecycleError("mutating lifecycle commands require --execute")


def command_plan(_arguments: argparse.Namespace) -> None:
    lifecycle, _ = _lifecycle()
    _json(lifecycle.plan())


def command_status(_arguments: argparse.Namespace) -> None:
    lifecycle, observe = _lifecycle()
    _json(lifecycle.status(observe))


def command_create(arguments: argparse.Namespace) -> None:
    _require_execute(arguments)
    lifecycle, _ = _lifecycle()
    _json(lifecycle.create(SystemRunner(lifecycle.spec)))


def command_attach(arguments: argparse.Namespace) -> None:
    _require_execute(arguments)
    lifecycle, observe = _lifecycle()
    _json(lifecycle.attach(SystemRunner(lifecycle.spec), observe))


def command_verify(_arguments: argparse.Namespace) -> None:
    lifecycle, observe = _lifecycle()
    _json(lifecycle.verify(observe))


def command_detach(arguments: argparse.Namespace) -> None:
    _require_execute(arguments)
    lifecycle, observe = _lifecycle()
    _json(lifecycle.detach(SystemRunner(lifecycle.spec), observe))


def command_recover(arguments: argparse.Namespace) -> None:
    _require_execute(arguments)
    lifecycle, observe = _lifecycle()
    _json(lifecycle.recover(SystemRunner(lifecycle.spec), observe))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    for name, function in (
        ("plan", command_plan),
        ("status", command_status),
        ("verify", command_verify),
    ):
        command = commands.add_parser(name)
        command.set_defaults(function=function)
    for name, function in (
        ("create", command_create),
        ("attach", command_attach),
        ("detach", command_detach),
        ("recover", command_recover),
    ):
        command = commands.add_parser(name)
        command.add_argument("--execute", action="store_true")
        command.set_defaults(function=function)
    return result


def main() -> int:
    parser().parse_args()
    raise SystemExit(
        "R2-MAP nested APFS lifecycle is retired; active storage is "
        "john2:/Users/john2/cascadia-bench/r2-map-v1"
    )


if __name__ == "__main__":
    raise SystemExit(main())
