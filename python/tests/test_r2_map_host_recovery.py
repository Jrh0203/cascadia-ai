from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import tools.r2_map_host_recovery as recovery


def test_real_window_gate_requires_cpu_rss_and_flat_swap() -> None:
    processes = iter(
        [
            {"pid": 7, "cpu_percent": 0.0, "rss_bytes": 100},
            {"pid": 7, "cpu_percent": 4.9, "rss_bytes": 200},
            {"pid": 7, "cpu_percent": 0.1, "rss_bytes": 300},
        ]
    )
    swaps = iter([1_000, 1_000, 999, 1_000])
    with (
        patch.object(recovery, "_syspolicyd", side_effect=lambda: next(processes)),
        patch.object(recovery, "_swap_used_bytes", side_effect=lambda: next(swaps)),
        patch.object(recovery.time, "sleep"),
        patch.object(recovery.time, "time_ns", side_effect=range(1_000_000, 2_000_000)),
    ):
        proof, receipt = recovery.observe(window_seconds=60, interval_seconds=30)
    assert proof["quiet_window_passed"] is True
    assert proof["system_swap_delta_bytes"] == 0
    assert receipt["status"] == "safe"
    assert receipt["quiet_window_passed"] is True


def test_any_cpu_or_swap_growth_keeps_host_blocked() -> None:
    processes = iter(
        [
            {"pid": 7, "cpu_percent": 0.0, "rss_bytes": 100},
            {"pid": 7, "cpu_percent": 5.1, "rss_bytes": 100},
        ]
    )
    swaps = iter([1_000, 1_000, 1_001])
    with (
        patch.object(recovery, "_syspolicyd", side_effect=lambda: next(processes)),
        patch.object(recovery, "_swap_used_bytes", side_effect=lambda: next(swaps)),
        patch.object(recovery.time, "sleep"),
    ):
        proof, receipt = recovery.observe(window_seconds=60, interval_seconds=60)
    assert proof["quiet_window_passed"] is False
    assert proof["system_swap_delta_bytes"] == 1
    assert receipt["status"] == "blocked-host-recovery"


def test_apfs_bootstrap_gate_allows_high_cpu_but_never_authorizes_runtime() -> None:
    processes = iter(
        [
            {"pid": 7, "cpu_percent": 140.0, "rss_bytes": 2 * (1 << 30)},
            {"pid": 7, "cpu_percent": 99.0, "rss_bytes": 2 * (1 << 30)},
            {"pid": 7, "cpu_percent": 50.0, "rss_bytes": 2 * (1 << 30)},
        ]
    )
    swaps = iter([1_000, 1_000, 999, 1_000])
    with (
        patch.object(recovery, "_syspolicyd", side_effect=lambda: next(processes)),
        patch.object(recovery, "_swap_used_bytes", side_effect=lambda: next(swaps)),
        patch.object(recovery, "_memory_pressure_level", return_value=1),
        patch.object(
            recovery.shutil,
            "disk_usage",
            return_value=SimpleNamespace(free=200 * (1 << 30)),
        ),
        patch.object(recovery.time, "sleep"),
    ):
        proof, receipt = recovery.observe_apfs_bootstrap(
            window_seconds=60, interval_seconds=30
        )
    assert proof["quiet_window_passed"] is True
    assert proof["runtime_authorized"] is False
    assert receipt["status"] == "apfs-bootstrap-safe"
    assert receipt["runtime_authorized"] is False


def test_apfs_bootstrap_gate_blocks_memory_pressure_or_swap_growth() -> None:
    processes = iter(
        [
            {"pid": 7, "cpu_percent": 0.0, "rss_bytes": 100},
            {"pid": 7, "cpu_percent": 0.0, "rss_bytes": 100},
        ]
    )
    swaps = iter([1_000, 1_000, 1_001])
    pressures = iter([1, 2])
    with (
        patch.object(recovery, "_syspolicyd", side_effect=lambda: next(processes)),
        patch.object(recovery, "_swap_used_bytes", side_effect=lambda: next(swaps)),
        patch.object(recovery, "_memory_pressure_level", side_effect=lambda: next(pressures)),
        patch.object(
            recovery.shutil,
            "disk_usage",
            return_value=SimpleNamespace(free=200 * (1 << 30)),
        ),
        patch.object(recovery.time, "sleep"),
    ):
        proof, receipt = recovery.observe_apfs_bootstrap(
            window_seconds=60, interval_seconds=60
        )
    assert proof["quiet_window_passed"] is False
    assert proof["system_swap_delta_bytes"] == 1
    assert receipt["status"] == "blocked-host-recovery"
