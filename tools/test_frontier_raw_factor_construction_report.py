from __future__ import annotations

import json
from pathlib import Path

from frontier_raw_factor_construction_report import (
    event_window,
    normalize_host,
    render_markdown,
    scientific_blake3,
    validate_execution_memory,
)


def test_scientific_hash_is_order_independent() -> None:
    assert scientific_blake3({"a": 1, "b": 2}) == scientific_blake3({"b": 2, "a": 1})


def test_john1_hostname_alias_is_normalized() -> None:
    assert normalize_host("Johns-Mac-mini") == "john1"
    assert normalize_host("john2.local") == "john2"


def test_event_window_requires_successful_complete_pair(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "started",
                        "started_unix_seconds": 10,
                        "queued_seconds": 0.5,
                    }
                ),
                json.dumps(
                    {
                        "event": "finished",
                        "ended_unix_seconds": 14,
                        "elapsed_seconds": 4,
                        "return_code": 0,
                    }
                ),
            ]
        )
    )
    assert event_window(path) == {
        "started_unix_seconds": 10.0,
        "ended_unix_seconds": 14.0,
        "elapsed_seconds": 4.0,
        "queued_seconds": 0.5,
    }


def test_allocator_gate_requires_zero_cache_after_clear() -> None:
    execution = {
        "mlx_memory_before_clear": {
            "peak_active_memory_bytes": 1024,
            "cache_memory_bytes": 1024,
        },
        "mlx_memory_after_clear": {
            "peak_active_memory_bytes": 1024,
            "cache_memory_bytes": 0,
        },
    }
    assert validate_execution_memory(execution)["passed"]
    execution["mlx_memory_after_clear"]["cache_memory_bytes"] = 1
    assert not validate_execution_memory(execution)["passed"]


def test_rejection_markdown_reports_no_selected_construction() -> None:
    report = {
        "classification": "raw_factor_construction_insufficient",
        "selected_kind": None,
        "probes": {
            kind: {
                "train": {
                    "target_positive_recall": 0.1,
                    "target_set_exact_fraction": 0.0,
                },
                "validation": {
                    "target_positive_recall": 0.1,
                    "target_set_exact_fraction": 0.0,
                },
            }
            for kind in (
                "complete-raw-flat",
                "exact-local-relation",
                "explicit-market-transition",
                "fresh-entity-cross",
            )
        },
        "execution": {
            "probe_and_replay_wall_seconds": 10.0,
            "hypotheses_per_hour": 4.0,
        },
    }
    markdown = render_markdown(report)
    assert "Selected construction: `none`" in markdown
    assert "next experiment must audit target learnability" in markdown
