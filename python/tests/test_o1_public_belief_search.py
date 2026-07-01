from __future__ import annotations

import json
from pathlib import Path

import pytest
from cascadia_mlx.o1_public_belief_search import (
    EXPECTED_PANEL_GROUPS,
    EXPERIMENT_ID,
    PublicBeliefSearchError,
    canonical_blake3,
    freeze_high_regret_panel,
    frozen_protocol,
)


def test_frozen_protocol_has_exact_640_trajectory_budget() -> None:
    protocol = frozen_protocol()
    budget = sum(
        active * samples
        for active, samples in zip(
            [64, 32, 16, 8],
            protocol["stage_additional_samples"],
            strict=True,
        )
    )
    assert budget == protocol["trajectories_per_group"] == 640
    assert protocol["leaf_model"] == "qualified-legacy-v4opp-exact-mlx-v1"
    assert (
        protocol["root_chance_policy"]
        == "condition-on-frozen-complete-turn-staged-prelude-context"
    )
    assert (
        protocol["hidden_order_policy"]
        == "sort-and-redeterminize-after-frozen-root-before-opponent-rotation"
    )


def test_freeze_panel_selects_exact_registered_high_regret_rows(tmp_path: Path) -> None:
    records = []
    for row in range(240):
        records.append(
            {
                "row": row,
                "group_id": 10_000 + row,
                "game_index": 61_003 + row // 80,
                "turn": row % 80,
                "r4800_scorable": True,
                "top1_retained_r4800_regret": (
                    0.75 if row < EXPECTED_PANEL_GROUPS else 0.25
                ),
            }
        )
    report = {
        "experiment_id": "o1-high-regret-draft-ranking-integration-v1",
        "arm": "z0-zero-intent",
        "claims": {"offline_validation_complete": True},
        "information_boundary": {"sealed_test_opened": False},
        "metrics": {"group_records": records},
    }
    source = tmp_path / "control.json"
    source.write_text(json.dumps(report))
    output = tmp_path / "panel.json"
    panel = freeze_high_regret_panel(control_report=source, output=output)
    assert panel["experiment_id"] == EXPERIMENT_ID
    assert len(panel["groups"]) == EXPECTED_PANEL_GROUPS
    assert panel["panel_id"] == canonical_blake3(
        {key: value for key, value in panel.items() if key != "panel_id"}
    )
    assert json.loads(output.read_text()) == panel


def test_freeze_panel_fails_closed_on_group_count_drift(tmp_path: Path) -> None:
    report = {
        "experiment_id": "o1-high-regret-draft-ranking-integration-v1",
        "arm": "z0-zero-intent",
        "claims": {"offline_validation_complete": True},
        "information_boundary": {"sealed_test_opened": False},
        "metrics": {
            "group_records": [
                {
                    "row": 0,
                    "group_id": 1,
                    "game_index": 1,
                    "turn": 0,
                    "r4800_scorable": True,
                    "top1_retained_r4800_regret": 1.0,
                }
            ]
        },
    }
    source = tmp_path / "control.json"
    source.write_text(json.dumps(report))
    with pytest.raises(PublicBeliefSearchError, match="expected 99"):
        freeze_high_regret_panel(
            control_report=source,
            output=tmp_path / "panel.json",
        )
