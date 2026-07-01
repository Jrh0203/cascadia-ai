from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import cascadia_v3_mlx.cycle_train as cycle_train
import pytest
from cascadia_v3_mlx.cycle_train import (
    SOURCE_REPLAY_EPOCHS,
    CycleTrainingError,
    active_source_schedule,
    source_quotas,
    source_thread_allocations,
)


def test_cycle_one_reassigns_unavailable_recent_replay_to_older_data() -> None:
    value = source_quotas(1)
    assert value == {
        "current_broad": 120_000,
        "current_teacher": 80_000,
        "recent": 0,
        "older_broad": 100_000,
        "older_teacher": 100_000,
    }
    assert sum(value.values()) == 400_000


def test_later_cycle_uses_registered_50_30_20_mix() -> None:
    value = source_quotas(7)
    assert value["current_broad"] + value["current_teacher"] == 200_000
    assert value["recent"] == 120_000
    assert value["older_broad"] + value["older_teacher"] == 80_000
    assert all(count % 32 == 0 for count in value.values())


def test_cycle_quota_domain_is_bounded() -> None:
    with pytest.raises(CycleTrainingError):
        source_quotas(11)


def test_cycle_one_allocates_threads_by_measured_preprocessing_cost() -> None:
    allocations = source_thread_allocations(source_quotas(1))
    assert allocations == {
        "current_broad": 2,
        "current_teacher": 2,
        "older_broad": 2,
        "older_teacher": 3,
    }
    assert sum(allocations.values()) == 9


def test_cycle_one_pass_schedule_skips_zero_quota_recent_source() -> None:
    quotas = source_quotas(1)
    allocations = source_thread_allocations(quotas)
    assert active_source_schedule(quotas, allocations) == (
        ("current_broad", 120_000, 2),
        ("current_teacher", 80_000, 2),
        ("older_broad", 100_000, 2),
        ("older_teacher", 100_000, 3),
    )


def test_active_source_schedule_rejects_missing_or_inactive_allocations() -> None:
    quotas = source_quotas(1)
    with pytest.raises(CycleTrainingError, match="has no thread allocation"):
        active_source_schedule(quotas, {"current_broad": 8})
    allocations = source_thread_allocations(quotas)
    allocations["recent"] = 1
    with pytest.raises(CycleTrainingError, match="inactive source"):
        active_source_schedule(quotas, allocations)


def test_later_cycle_balances_nine_threads_from_live_tail_traces() -> None:
    allocations = source_thread_allocations(source_quotas(7))
    assert allocations == {
        "current_broad": 1,
        "current_teacher": 2,
        "recent": 2,
        "older_broad": 2,
        "older_teacher": 2,
    }
    assert sum(allocations.values()) == 9


def test_thread_allocation_is_independent_of_output_quota_magnitude() -> None:
    quotas = source_quotas(7)
    scaled = {source: quota * 10 for source, quota in quotas.items()}
    assert source_thread_allocations(scaled) == source_thread_allocations(quotas)


def test_thread_budget_refuses_less_than_one_thread_per_source() -> None:
    with pytest.raises(CycleTrainingError):
        source_thread_allocations(source_quotas(7), total_threads=4)


def test_cycle_sources_can_replay_across_the_complete_d6_group() -> None:
    assert SOURCE_REPLAY_EPOCHS == 12


def test_cycle_stream_uses_d6_replay_but_keeps_the_exact_quota(monkeypatch) -> None:
    captured = {}

    def fake_stream(*args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(cycle_train, "RustBatchStream", fake_stream)
    args = Namespace(
        batch_stream_binary=Path("/tmp/v3-batch-stream"),
        batch_size=8192,
        campaign_state=Path("/tmp/campaign-state.json"),
        cycle=1,
        seed=90011,
    )
    result = cycle_train._stream(
        args=args,
        config=object(),
        source="current_teacher",
        paths=[Path("/tmp/current-teacher.v3l")],
        examples=80_000,
        pass_index=1,
        boundaries=None,
        expansion_threads=8,
    )

    assert result is not None
    assert captured["epochs"] == 12
    assert captured["max_examples"] == 80_000
    assert captured["d6_cycle"] is True
