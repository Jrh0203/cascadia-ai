from __future__ import annotations

from tools.conditional_tile_successor_forensics_report import (
    build_combined,
    validate_arm,
)


def _arm(name: str, host: str, classification: str) -> dict[str, object]:
    scientific = {
        "arm": name,
        "classification": classification,
        "test_split_opened": False,
        "gameplay_opened": False,
        "new_teacher_compute_used": False,
        "external_compute_used": False,
    }
    from cascadia_mlx.full_legal_hierarchical_factor_retrieval import (
        _scientific_blake3,
    )

    return {
        "experiment_id": "conditional-tile-successor-forensics-v1",
        "host": host,
        "scientific": scientific,
        "scientific_blake3": _scientific_blake3(scientific),
        "execution": {
            "process_swaps": 0,
            "peak_process_rss_bytes": 1024,
        },
    }


def _queue() -> dict[str, object]:
    tasks = []
    for index, task_id in enumerate(
        (
            "forensic-factor-selector",
            "forensic-sampling-mass",
            "forensic-score-scale",
        )
    ):
        tasks.append(
            {
                "id": task_id,
                "status": "completed",
                "attempts": [
                    {
                        "claimed_unix_ms": 1000 + index,
                        "ended_unix_ms": 2000 + index,
                        "outcome": "completed",
                    }
                ],
            }
        )
    return {"tasks": tasks}


def test_validate_arm_accepts_closed_valid_payload() -> None:
    assert (
        validate_arm(
            _arm("sampling-mass", "john3", "uniform_query_sampling_not_explanatory"),
            expected_arm="sampling-mass",
            expected_host="john3",
        )
        == []
    )


def test_combined_selects_optimizer_and_normalized_action_selector() -> None:
    combined = build_combined(
        factor_selector=_arm(
            "factor-selector-ceiling",
            "john1",
            "complete_action_selector_required",
        ),
        sampling_mass=_arm(
            "sampling-mass",
            "john3",
            "uniform_query_sampling_not_explanatory",
        ),
        score_scale=_arm(
            "score-scale",
            "john4",
            "cross_stage_score_scale_mismatch",
        ),
        queue=_queue(),
    )
    scientific = combined["scientific"]
    assert scientific["pipeline_passed"]
    assert (
        scientific["mechanical_successors"]["if_adr0118_insufficient"]
        == "optimizer_schedule_treatment"
    )
    assert (
        scientific["mechanical_successors"]["if_adr0118_sufficient"]
        == "normalized_complete_action_selector"
    )
