from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from frontier_fit_interference_report import (
    render_markdown,
    summarize_campaign_events,
    validate_replay_comparisons,
    validate_source_identities,
)


def _report(arm: str, scientific: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment_id": "complete-action-frontier-fit-interference-audit-v1",
        "scientific": {"arm": arm, **scientific},
    }


def test_validate_source_identities_requires_four_identical_hosts(
    tmp_path: Path,
) -> None:
    paths = []
    for host in ("john1", "john2", "john3", "john4"):
        path = tmp_path / f"{host}.json"
        path.write_text(
            json.dumps(
                {
                    "host": host,
                    "files": 108,
                    "bundle_sha256": "a" * 64,
                }
            )
        )
        paths.append(path)
    assert validate_source_identities(paths) == {
        "hosts": ["john1", "john2", "john3", "john4"],
        "files": 108,
        "bundle_sha256": "a" * 64,
    }
    paths[-1].write_text(
        json.dumps(
            {
                "host": "john4",
                "files": 108,
                "bundle_sha256": "b" * 64,
            }
        )
    )
    with pytest.raises(ValueError, match="not identical"):
        validate_source_identities(paths)


def test_render_markdown_contains_decision_evidence() -> None:
    metric = {
        "target_positive_recall": 0.5,
        "target_set_exact_fraction": 0.25,
        "r4800_winner_retention": 0.75,
        "mean_objective": 1.25,
    }
    nested = _report(
        "nested-subset",
        {
            "variants": {
                size: {"final": metric}
                for size in ("1", "4", "16", "64")
            }
        },
    )
    capacity = _report(
        "capacity-scaling",
        {
            "variants": {
                width: {
                    "parameter_count": 100,
                    "final": metric,
                }
                for width in ("96", "192", "288")
            }
        },
    )
    gradient = _report(
        "gradient-conflict",
        {
            "selected": {
                "scopes": {
                    "full_model": {
                        "off_diagonal_at_most_negative_0_10_fraction": 0.4,
                        "cosine_to_other_gradient_sum": {
                            "negative_fraction": 0.8,
                            "distribution": {"median": -0.2},
                        },
                    }
                }
            }
        },
    )
    error = _report(
        "error-anatomy",
        {
            "initial": metric,
            "independent": {"aggregate": metric},
            "shared": {"aggregate": metric},
        },
    )
    telemetry = {
        "host": "john1",
        "elapsed_seconds": 10.0,
        "peak_process_rss_bytes": 1024,
        "process_swaps": 0,
        "system_swap_delta_bytes": 0,
    }
    combined = _report(
        "combined",
        {
            "classification": "cross_group_gradient_interference",
            "gates": {"pipeline_passed": True},
            "arm_telemetry": {
                name: {**telemetry, "host": host}
                for name, host in (
                    ("nested-subset", "john1"),
                    ("capacity-scaling", "john2"),
                    ("gradient-conflict", "john3"),
                    ("error-anatomy", "john4"),
                )
            },
            "duplicate_training_fraction": 0.0,
            "full_cohort_digest_blake3": "c" * 64,
        },
    )
    markdown = render_markdown(
        combined_report=combined,
        nested_report=nested,
        capacity_report=capacity,
        gradient_report=gradient,
        error_report=error,
        source_identity={
            "hosts": ["john1", "john2", "john3", "john4"],
            "files": 108,
            "bundle_sha256": "a" * 64,
        },
    )
    assert "Classification: `cross_group_gradient_interference`" in markdown
    assert "80.00%" in markdown
    assert "Gradient-conflict mitigation" not in markdown
    assert "gradient-conflict mitigation" in markdown


def test_validate_replays_and_summarize_campaign(tmp_path: Path) -> None:
    comparison_paths = []
    arms = (
        "nested-subset",
        "capacity-scaling",
        "gradient-conflict",
        "error-anatomy",
    )
    for index, arm in enumerate(arms):
        path = tmp_path / f"{arm}.json"
        path.write_text(
            json.dumps(
                {
                    "arm": arm,
                    "origin_host": f"john{index + 1}",
                    "replay_host": f"john{4 - index}",
                    "origin_scientific_blake3": str(index) * 64,
                    "replay_scientific_blake3": str(index) * 64,
                    "scientific_payload_identical": True,
                }
            )
        )
        comparison_paths.append(path)
    replay = validate_replay_comparisons(comparison_paths)
    assert replay["all_identical"]
    assert set(replay["reports"]) == set(arms)

    event_paths = []
    for index in range(8):
        path = tmp_path / f"event-{index}.jsonl"
        name = f"arm-{index % 4}"
        if index >= 4:
            name += "-replay"
        started = 100.0 + index
        finished = started + 10.0
        path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "event": "started",
                            "name": name,
                            "host": f"john{index % 4 + 1}",
                            "started_unix_seconds": started,
                        }
                    ),
                    json.dumps(
                        {
                            "event": "finished",
                            "name": name,
                            "host": f"john{index % 4 + 1}",
                            "ended_unix_seconds": finished,
                            "elapsed_seconds": 10.0,
                            "return_code": 0,
                        }
                    ),
                ]
            )
            + "\n"
        )
        event_paths.append(path)
    campaign = summarize_campaign_events(event_paths)
    assert campaign["origin_makespan_seconds"] == 13.0
    assert campaign["end_to_end_makespan_seconds"] == 17.0
    assert campaign["total_job_seconds"] == 80.0
    assert campaign["confirmation_compute_fraction"] == 0.5
