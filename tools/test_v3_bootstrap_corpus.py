from __future__ import annotations

import v3_bootstrap_corpus as corpus


def plan() -> dict[str, object]:
    items = []
    components = [
        ("greedy", 50, 1_000_000_000),
        ("v1-direct", 100, 1_100_000_000),
        ("mixed-frozen", 50, 1_300_000_000),
        ("rare-softmax", 50, 1_400_000_000),
    ]
    for component, count, first in components:
        for index in range(count):
            items.append(
                {
                    "key": f"{component}-{index:03d}",
                    "application_metadata": {
                        "component": component,
                        "games": "2000",
                        "first_game_index": str(first + index * 2000),
                    },
                }
            )
    return {
        "schema_id": "cascadia-v3-bacalhau-collection-plan-v1",
        "phase": "bootstrap_collecting",
        "games": 500_000,
        "work_items": 250,
        "scheduler_owns_placement": True,
        "manual_host_sharding": False,
        "items": items,
    }


def test_expected_plan_is_exact_and_disjoint() -> None:
    items = corpus.expected_items(plan())
    assert len(items) == 250
    intervals = [{"key": key, **value} for key, value in items.items()]
    corpus._validate_intervals(intervals)


def test_overlap_is_rejected() -> None:
    intervals = [
        {"key": "a", "first_game_index": 10, "games": 10},
        {"key": "b", "first_game_index": 19, "games": 10},
    ]
    try:
        corpus._validate_intervals(intervals)
    except corpus.CorpusError:
        pass
    else:
        raise AssertionError("overlapping scientific game domains were accepted")
