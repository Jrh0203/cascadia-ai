from __future__ import annotations

from types import SimpleNamespace

import pytest
from cascadia_mlx.r2_map_training_resources import (
    MAX_PROCESS_RSS_BYTES,
    R2MapTrainingResourceError,
    R2MapTrainingResourceMonitor,
    process_resource_sample,
    system_swap_used_bytes,
    validate_training_resource_receipt,
)


def test_resource_monitor_accepts_bounded_zero_growth_samples() -> None:
    monitor = R2MapTrainingResourceMonitor.start(swap_reader=lambda: 1 << 20)
    sample = monitor.sample(
        swap_reader=lambda: 1 << 20,
        process_reader=lambda: (512 << 20, 0),
    )
    assert sample["system_swap_delta_bytes"] == 0
    assert validate_training_resource_receipt(monitor.receipt())["sample_count"] == 1


@pytest.mark.parametrize("failure", ("rss", "process-swap", "system-swap"))
def test_resource_monitor_stops_on_every_frozen_gate(failure: str) -> None:
    monitor = R2MapTrainingResourceMonitor.start(swap_reader=lambda: 100)
    process = {
        "rss": (MAX_PROCESS_RSS_BYTES + 1, 0),
        "process-swap": (1, 1),
        "system-swap": (1, 0),
    }[failure]
    observed_swap = 101 if failure == "system-swap" else 100
    with pytest.raises(R2MapTrainingResourceError):
        monitor.sample(
            swap_reader=lambda: observed_swap,
            process_reader=lambda: process,
        )


def test_swap_parser_and_platform_rss_units() -> None:
    completed = SimpleNamespace(stdout="total = 4.00G  used = 512.50M  free = 3.50G")
    assert system_swap_used_bytes(runner=lambda *_args, **_kwargs: completed) == round(
        512.5 * (1 << 20)
    )
    usage = SimpleNamespace(ru_maxrss=123, ru_nswap=0)
    assert process_resource_sample(lambda _who: usage, system_name="Darwin") == (123, 0)
    assert process_resource_sample(lambda _who: usage, system_name="Linux") == (123 * 1024, 0)
