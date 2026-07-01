import json
from pathlib import Path

import blake3
import cluster_research_queue as queue
import pytest
import relational_substrate_mlx_post_campaign as campaign


def test_frozen_protocol_identity_matches_adr_0161() -> None:
    assert campaign.EXPERIMENT_ID == "relational-substrate-mlx-tournament-v1"
    assert campaign.PROTOCOL_ID == "r5-s3-s5-matched-mlx-v1"
    assert campaign.ADR_ID == "0161"


def _file_blake3(path: Path) -> str:
    return blake3.blake3(path.read_bytes()).hexdigest()


def _completed_control(repository: Path) -> None:
    run = repository / campaign.CONTROL_RUN
    checkpoint_name = "step-000003000-epoch-0000-batch-003000"
    checkpoint = run / "checkpoints" / checkpoint_name
    checkpoint.mkdir(parents=True)
    (checkpoint / "checkpoint.json").write_text('{"step":3000}\n')
    (checkpoint / "model.safetensors").write_bytes(b"exact-c0-model")
    (run / "latest.json").write_text(
        json.dumps({"checkpoint": checkpoint_name})
    )
    identity = {
        "experiment_id": campaign.EXPERIMENT_ID,
        "control": "exact-r2",
        "global_step": campaign.TRAINING_STEPS,
    }
    report = {
        "schema_version": 1,
        "experiment_id": campaign.EXPERIMENT_ID,
        "protocol_id": campaign.PROTOCOL_ID,
        "adr": campaign.ADR_ID,
        "mode": "production",
        "arm": campaign.CONTROL_ARM,
        "host": "john1",
        "optimization": {"global_step": campaign.TRAINING_STEPS},
        "scientific_identity": identity,
        "report_id": campaign.canonical_blake3(identity),
        "checkpoint": {
            "path": str(checkpoint),
            "manifest_blake3": _file_blake3(checkpoint / "checkpoint.json"),
            "model_blake3": _file_blake3(checkpoint / "model.safetensors"),
        },
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (run / "final-report.json").write_text(encoded)
    output = repository / campaign.CONTROL_REPORT
    output.parent.mkdir(parents=True)
    output.write_text(encoded)


def test_post_training_graph_pins_three_host_paired_replays(
    tmp_path: Path,
) -> None:
    _completed_control(tmp_path)
    tasks, control = campaign.build_task_specs(tmp_path)
    by_id = {task["id"]: task for task in tasks}
    assert len(tasks) == 6
    assert len(by_id) == len(tasks)
    assert control["checkpoint_name"].startswith("step-000003000")
    assert by_id["relmlx-c0-run-fanout"]["compatible_hosts"] == ["john1"]
    replay_ids = set()
    for replay in campaign.REPLAYS:
        replay_ids.add(replay.task_id)
        task = by_id[replay.task_id]
        assert task["compatible_hosts"] == [replay.host]
        assert task["dependencies"] == ["relmlx-c0-run-fanout"]
        assert task["resources"]["uses_mlx"] is True
        command = task["command"]
        assert command[command.index("--treatment-arm") + 1] == replay.treatment_arm
        assert (
            command[command.index("--control-report") + 1]
            == campaign._remote(
                replay.host,
                campaign.CONTROL_RUN / "final-report.json",
            )
        )
    assert (
        set(by_id["relmlx-c0-replay-collect"]["dependencies"]) == replay_ids
    )
    collect_command = by_id["relmlx-c0-replay-collect"]["command"]
    assert collect_command[:3] == [
        "/usr/bin/env",
        "-C",
        str(campaign.REMOTE_ROOTS["john1"]),
    ]
    assert "tools/cluster_artifact_collect.py" in collect_command
    assert by_id["relmlx-classify"]["dependencies"] == [
        "relmlx-c0-replay-collect"
    ]
    assert by_id["relmlx-classify"]["decision_terminal"] is True


def test_checkpoint_tampering_is_rejected(tmp_path: Path) -> None:
    _completed_control(tmp_path)
    latest = json.loads(
        (tmp_path / campaign.CONTROL_RUN / "latest.json").read_text()
    )
    model = (
        tmp_path
        / campaign.CONTROL_RUN
        / "checkpoints"
        / latest["checkpoint"]
        / "model.safetensors"
    )
    model.write_bytes(b"tampered")
    with pytest.raises(campaign.PostCampaignError, match="checkpoint bytes"):
        campaign.validate_completed_control(tmp_path)


def test_generated_tasks_satisfy_live_queue_schema(tmp_path: Path) -> None:
    _completed_control(tmp_path)
    payload = campaign.build_queue_spec(tmp_path)
    state = queue.empty_queue("test-relational-post-campaign", now_ms=100)
    queue.add_tasks(state, payload["tasks"], now_ms=200)
    queue.validate_queue(state)
    assert payload["task_count"] == 6
    assert len(state["tasks"]) == 6


def test_top_level_and_run_reports_must_match(tmp_path: Path) -> None:
    _completed_control(tmp_path)
    output = tmp_path / campaign.CONTROL_REPORT
    report = json.loads(output.read_text())
    report["host"] = "john9"
    output.write_text(json.dumps(report))
    with pytest.raises(campaign.PostCampaignError, match="output report differ"):
        campaign.validate_completed_control(tmp_path)
