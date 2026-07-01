from __future__ import annotations

from pathlib import Path

from p1_relational_pointer_queue import EXPERIMENT_ID, campaign_spec, task_specs


def test_crossed_queue_uses_distinct_origins_and_replays() -> None:
    tasks = task_specs(
        bundle_relative=Path(
            "artifacts/experiments/p1-relational-hierarchical-pointer-foundation-v1/"
            f"bundles/{'a' * 64}"
        ),
        bundle_id="a" * 64,
    )
    by_id = {task["id"]: task for task in tasks}
    assert len(tasks) == 6
    assert by_id["p1ptr-v1-train-john2"]["compatible_hosts"] == ["john2"]
    assert by_id["p1ptr-v1-validation-john4"]["compatible_hosts"] == ["john4"]
    assert by_id["p1ptr-v1-train-replay-john4"]["compatible_hosts"] == ["john4"]
    assert by_id["p1ptr-v1-validation-replay-john2"]["compatible_hosts"] == ["john2"]
    assert by_id["p1ptr-v1-train-replay-john4"]["dependencies"] == [
        "p1ptr-v1-train-john2",
        "p1ptr-v1-validation-john4",
    ]
    assert by_id["p1ptr-v1-validation-replay-john2"]["dependencies"] == [
        "p1ptr-v1-train-john2",
        "p1ptr-v1-validation-john4",
    ]


def test_queue_commands_use_frozen_bundle_and_host_aliases() -> None:
    bundle_id = "b" * 64
    tasks = task_specs(
        bundle_relative=Path("artifacts/frozen") / bundle_id,
        bundle_id=bundle_id,
    )
    origins = tasks[:4]
    for task in origins:
        command = task["command"]
        assert bundle_id in command
        assert "--host" in command
        host = task["compatible_hosts"][0]
        assert command[command.index("--host") + 1] == host
        assert "tools/p1_relational_pointer_foundation.py" in command


def test_classification_waits_for_checksum_collection() -> None:
    tasks = task_specs(
        bundle_relative=Path("artifacts/frozen") / ("c" * 64),
        bundle_id="c" * 64,
    )
    assert tasks[-1]["id"] == "p1ptr-v1-classify"
    assert tasks[-1]["dependencies"] == ["p1ptr-v1-collect"]
    assert tasks[-1]["decision_terminal"] is True


def test_campaign_envelope_binds_bundle_and_task_count() -> None:
    tasks = task_specs(
        bundle_relative=Path("artifacts/frozen") / ("d" * 64),
        bundle_id="d" * 64,
    )
    envelope = campaign_spec(tasks, bundle_id="d" * 64)
    assert envelope["experiment_id"] == EXPERIMENT_ID
    assert envelope["bundle_id"] == "d" * 64
    assert envelope["task_count"] == 6
    assert envelope["tasks"] == tasks
