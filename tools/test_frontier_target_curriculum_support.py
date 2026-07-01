from __future__ import annotations

import json

from frontier_target_curriculum_support import audit_trajectory


def test_trajectory_selects_best_target_recall(tmp_path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    events = [
        {
            "epoch": 1,
            "validation": {
                "target_positive_recall": 0.25,
                "target_set_exact_fraction": 0.0,
                "top64_r4800_winner_recall": 0.75,
                "training_objective": 9.0,
            },
        },
        {
            "epoch": 2,
            "validation": {
                "target_positive_recall": 0.40,
                "target_set_exact_fraction": 0.05,
                "top64_r4800_winner_recall": 0.76,
                "training_objective": 8.0,
            },
        },
    ]
    metrics.write_text("\n".join(json.dumps(event) for event in events) + "\n")
    report = audit_trajectory(metrics)
    assert report["best_target_epoch"]["epoch"] == 2
    assert report["target_recall_range"] == {"min": 0.25, "max": 0.40}
    assert report["exact_target_sets_ever_recovered"]
