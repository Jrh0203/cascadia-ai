"""Fail-closed John1 resource gates for R2-MAP MLX training.

The monitor is intentionally filesystem-free. It samples process RSS/swap and
the host's existing swap allocation, then stops training on any positive swap
growth or a process RSS above the measured five-GiB ceiling.  The ceiling is
just above the 4.66-GB observed worst-case exact 12,168-action screen.
"""

from __future__ import annotations

import platform
import re
import resource
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

MAX_PROCESS_RSS_BYTES = 5 * (1 << 30)
RESOURCE_RECEIPT_FIELDS = (
    "maximum_rss_bytes",
    "process_swaps",
    "system_swap_baseline_bytes",
    "maximum_system_swap_bytes",
    "system_swap_delta_bytes",
    "sample_count",
)
_SWAP = re.compile(r"used = ([0-9.]+)([MG])")
_UNIT = {"M": 1 << 20, "G": 1 << 30}


class R2MapTrainingResourceError(RuntimeError):
    """John1 crossed a frozen memory or zero-swap stop gate."""


def system_swap_used_bytes(
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> int:
    completed = runner(
        ["/usr/sbin/sysctl", "-n", "vm.swapusage"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    match = _SWAP.search(completed.stdout)
    if match is None:
        raise R2MapTrainingResourceError("vm.swapusage output is unrecognized")
    return round(float(match.group(1)) * _UNIT[match.group(2)])


def process_resource_sample(
    usage_getter: Callable[[int], Any] = resource.getrusage,
    *,
    system_name: str | None = None,
) -> tuple[int, int]:
    usage = usage_getter(resource.RUSAGE_SELF)
    rss_bytes = int(usage.ru_maxrss)
    if (platform.system() if system_name is None else system_name) != "Darwin":
        rss_bytes *= 1024
    return rss_bytes, int(getattr(usage, "ru_nswap", 0))


@dataclass
class R2MapTrainingResourceMonitor:
    baseline_system_swap_bytes: int
    maximum_rss_bytes: int = 0
    maximum_process_swaps: int = 0
    maximum_system_swap_bytes: int = 0
    sample_count: int = 0

    @classmethod
    def start(
        cls,
        *,
        swap_reader: Callable[[], int] = system_swap_used_bytes,
    ) -> R2MapTrainingResourceMonitor:
        baseline = swap_reader()
        if baseline < 0:
            raise R2MapTrainingResourceError("system swap baseline is negative")
        return cls(
            baseline_system_swap_bytes=baseline,
            maximum_system_swap_bytes=baseline,
        )

    def sample(
        self,
        *,
        swap_reader: Callable[[], int] = system_swap_used_bytes,
        process_reader: Callable[[], tuple[int, int]] = process_resource_sample,
    ) -> dict[str, int]:
        rss_bytes, process_swaps = process_reader()
        system_swap_bytes = swap_reader()
        if min(rss_bytes, process_swaps, system_swap_bytes) < 0:
            raise R2MapTrainingResourceError("resource sample contains a negative metric")
        self.maximum_rss_bytes = max(self.maximum_rss_bytes, rss_bytes)
        self.maximum_process_swaps = max(self.maximum_process_swaps, process_swaps)
        self.maximum_system_swap_bytes = max(
            self.maximum_system_swap_bytes,
            system_swap_bytes,
        )
        self.sample_count += 1
        sample = {
            "rss_bytes": rss_bytes,
            "process_swaps": process_swaps,
            "system_swap_bytes": system_swap_bytes,
            "system_swap_delta_bytes": max(
                system_swap_bytes - self.baseline_system_swap_bytes,
                0,
            ),
        }
        if rss_bytes > MAX_PROCESS_RSS_BYTES:
            raise R2MapTrainingResourceError("R2-MAP training exceeded the 5-GiB RSS gate")
        if process_swaps > 0:
            raise R2MapTrainingResourceError("R2-MAP training process swapped")
        if sample["system_swap_delta_bytes"] > 0:
            raise R2MapTrainingResourceError("system swap grew during R2-MAP training")
        return sample

    def receipt(self) -> dict[str, int]:
        if self.sample_count <= 0:
            raise R2MapTrainingResourceError("training resource monitor has no samples")
        return {
            "maximum_rss_bytes": self.maximum_rss_bytes,
            "process_swaps": self.maximum_process_swaps,
            "system_swap_baseline_bytes": self.baseline_system_swap_bytes,
            "maximum_system_swap_bytes": self.maximum_system_swap_bytes,
            "system_swap_delta_bytes": max(
                self.maximum_system_swap_bytes - self.baseline_system_swap_bytes,
                0,
            ),
            "sample_count": self.sample_count,
        }


def validate_training_resource_receipt(value: dict[str, Any]) -> dict[str, int]:
    if set(value) != set(RESOURCE_RECEIPT_FIELDS) or any(
        not isinstance(value[name], int) or isinstance(value[name], bool) or value[name] < 0
        for name in RESOURCE_RECEIPT_FIELDS
    ):
        raise R2MapTrainingResourceError("training resource receipt schema differs")
    if (
        value["maximum_rss_bytes"] > MAX_PROCESS_RSS_BYTES
        or value["process_swaps"] != 0
        or value["system_swap_delta_bytes"] != 0
        or value["maximum_system_swap_bytes"] > value["system_swap_baseline_bytes"]
        or value["sample_count"] == 0
    ):
        raise R2MapTrainingResourceError("training resource receipt violates a stop gate")
    return {name: int(value[name]) for name in RESOURCE_RECEIPT_FIELDS}
