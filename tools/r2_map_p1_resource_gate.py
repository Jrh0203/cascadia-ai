#!/usr/bin/env python3
"""Run the John2 P1 gate and fail closed on its frozen resource limits."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

SCHEMA = "cascadia.r2-map.p1-resource-gate.v1"
MAX_RSS_BYTES = 4 * 1024 * 1024 * 1024
SWAP_PATTERN = re.compile(r"\bused\s*=\s*([0-9]+(?:\.[0-9]+)?)([KMGT]?)", re.I)
TIME_PATTERN = re.compile(r"^\s*([0-9]+)\s+(.+?)\s*$")
UNIT_BYTES = {
    "": 1,
    "K": 1024,
    "M": 1024**2,
    "G": 1024**3,
    "T": 1024**4,
}


class ResourceGateError(RuntimeError):
    """The resource evidence is absent, malformed, or outside the gate."""


def swap_usage() -> tuple[str, int]:
    completed = subprocess.run(
        ["/usr/sbin/sysctl", "vm.swapusage"],
        check=True,
        capture_output=True,
        text=True,
    )
    raw = completed.stdout.strip()
    match = SWAP_PATTERN.search(raw)
    if match is None:
        raise ResourceGateError(f"cannot parse vm.swapusage: {raw!r}")
    used = round(float(match.group(1)) * UNIT_BYTES[match.group(2).upper()])
    return raw, used


def time_metrics(path: Path) -> tuple[int, int]:
    maximum_rss = None
    process_swaps = None
    for line in path.read_text(encoding="utf-8").splitlines():
        match = TIME_PATTERN.match(line)
        if match is None:
            continue
        value = int(match.group(1))
        label = match.group(2)
        if label == "maximum resident set size":
            maximum_rss = value
        elif label == "swaps":
            process_swaps = value
    if maximum_rss is None or process_swaps is None:
        raise ResourceGateError(
            "time -l evidence lacks maximum resident set size or process swaps"
        )
    return maximum_rss, process_swaps


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--metrics", type=Path, required=True)
    value.add_argument("--max-rss-bytes", type=int, default=MAX_RSS_BYTES)
    value.add_argument("command", nargs=argparse.REMAINDER)
    return value


def main() -> int:
    arguments = parser().parse_args()
    command = arguments.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        raise ResourceGateError("resource gate command is empty")
    if arguments.max_rss_bytes <= 0:
        raise ResourceGateError("maximum RSS limit must be positive")

    arguments.metrics.parent.mkdir(parents=True, exist_ok=True)
    arguments.metrics.unlink(missing_ok=True)
    pre_raw, pre_used = swap_usage()
    completed = subprocess.run(
        ["/usr/bin/time", "-l", "-o", str(arguments.metrics), *command],
        check=False,
    )
    post_raw, post_used = swap_usage()
    maximum_rss, process_swaps = time_metrics(arguments.metrics)
    swap_delta = post_used - pre_used
    passed = (
        completed.returncode == 0
        and process_swaps == 0
        and swap_delta == 0
        and maximum_rss <= arguments.max_rss_bytes
    )
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "command_exit_code": completed.returncode,
                "maximum_resident_set_size_bytes": maximum_rss,
                "maximum_resident_set_size_limit_bytes": arguments.max_rss_bytes,
                "process_swaps": process_swaps,
                "system_swap_pre": pre_raw,
                "system_swap_post": post_raw,
                "system_swap_pre_used_bytes": pre_used,
                "system_swap_post_used_bytes": post_used,
                "system_swap_delta_bytes": swap_delta,
                "passed": passed,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        file=sys.stderr,
        flush=True,
    )
    if completed.returncode != 0:
        return completed.returncode
    return 0 if passed else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ResourceGateError, subprocess.SubprocessError) as error:
        print(f"R2-MAP P1 resource gate refused execution: {error}", file=sys.stderr)
        raise SystemExit(2) from error
