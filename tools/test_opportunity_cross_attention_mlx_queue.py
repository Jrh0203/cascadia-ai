from __future__ import annotations

from pathlib import Path

from opportunity_cross_attention_mlx_queue import (
    ARM_HOSTS,
    ARMS,
    CAMPAIGN_ROOT,
    HOST_PREREQUISITES,
    REMOTE_ROOTS,
    TASK_PREFIX,
    build_task_specs,
    launch_root,
)


def test_queue_allocates_one_unique_arm_per_host(monkeypatch, tmp_path) -> None:
    repository = tmp_path / "repo"
    bundle = repository / "artifacts" / "bundles" / ("a" * 64)
    (bundle / "bin").mkdir(parents=True)
    (bundle / "bin" / "relational-substrate-r6-replay").write_bytes(b"x")
    monkeypatch.setattr(
        "opportunity_cross_attention_mlx_queue.validate_bundle",
        lambda path: {
            "bundle_id": path.name,
            "identity": {
                "experiment_id": "opportunity-cross-attention-mlx-tournament-v1"
            },
        },
    )

    tasks = build_task_specs(
        repository,
        bundle,
        control_identity={"report_id": "b" * 64},
    )
    by_id = {task["id"]: task for task in tasks}

    assert len(tasks) == 20
    for arm in ARMS:
        host = ARM_HOSTS[arm]
        slug = arm.replace("-", "_")
        run = by_id[f"{TASK_PREFIX}-run-{slug}"]
        assert run["compatible_hosts"] == [host]
        assert f"{TASK_PREFIX}-preflight-{host}" in run["dependencies"]
        smoke = by_id[f"{TASK_PREFIX}-smoke-{host}"]
        assert HOST_PREREQUISITES[host] in smoke["dependencies"]
    assert by_id[f"{TASK_PREFIX}-classify"]["decision_terminal"] is True


def test_queue_production_runs_share_all_preflight_dependencies(
    monkeypatch,
    tmp_path,
) -> None:
    repository = tmp_path / "repo"
    bundle = repository / "bundle" / ("c" * 64)
    (bundle / "bin").mkdir(parents=True)
    (bundle / "bin" / "relational-substrate-r6-replay").write_bytes(b"x")
    monkeypatch.setattr(
        "opportunity_cross_attention_mlx_queue.validate_bundle",
        lambda path: {
            "bundle_id": path.name,
            "identity": {
                "experiment_id": "opportunity-cross-attention-mlx-tournament-v1"
            },
        },
    )
    tasks = build_task_specs(
        repository,
        bundle,
        control_identity={"report_id": "d" * 64},
    )
    preflights = {
        f"{TASK_PREFIX}-preflight-{host}"
        for host in ARM_HOSTS.values()
    }
    runs = [
        task
        for task in tasks
        if task["id"].startswith(f"{TASK_PREFIX}-run-")
    ]

    assert len(runs) == 4
    assert all(preflights <= set(run["dependencies"]) for run in runs)


def test_john1_artifact_commands_use_absolute_paths(
    monkeypatch,
    tmp_path,
) -> None:
    repository = tmp_path / "repo"
    bundle = repository / "artifacts" / "bundles" / ("e" * 64)
    (bundle / "bin").mkdir(parents=True)
    (bundle / "bin" / "relational-substrate-r6-replay").write_bytes(b"x")
    monkeypatch.setattr(
        "opportunity_cross_attention_mlx_queue.validate_bundle",
        lambda path: {
            "bundle_id": path.name,
            "identity": {
                "experiment_id": "opportunity-cross-attention-mlx-tournament-v1"
            },
        },
    )
    tasks = build_task_specs(
        repository,
        bundle,
        control_identity={"report_id": "f" * 64},
    )
    by_title = {task["title"]: task for task in tasks}
    john1_root = str(REMOTE_ROOTS["john1"])

    inspected = (
        by_title["Fan out immutable ADR 0166 bundle"],
        by_title["Collect four common-arm smoke runs"],
        by_title["Fan out ADR 0166 launch controls"],
        by_title["Collect four ADR 0166 production arms"],
    )
    for task in inspected:
        command = task["command"]
        for flag in ("--source", "--local-root", "--output"):
            for index, value in enumerate(command):
                if value == flag:
                    assert command[index + 1].rstrip("/").startswith(john1_root)
        for index, value in enumerate(command):
            if value == "--artifact":
                assert command[index + 2].startswith(john1_root)


def test_relaunch_namespace_isolates_every_generated_artifact(
    monkeypatch,
    tmp_path,
) -> None:
    repository = tmp_path / "repo"
    bundle = repository / "artifacts" / "bundles" / ("1" * 64)
    (bundle / "bin").mkdir(parents=True)
    (bundle / "bin" / "relational-substrate-r6-replay").write_bytes(b"x")
    monkeypatch.setattr(
        "opportunity_cross_attention_mlx_queue.validate_bundle",
        lambda path: {
            "bundle_id": path.name,
            "identity": {
                "experiment_id": "opportunity-cross-attention-mlx-tournament-v1"
            },
        },
    )
    task_prefix = "oppquery-v3"
    artifact_root = launch_root("adr0172-relaunch-v1")

    tasks = build_task_specs(
        repository,
        bundle,
        control_identity={"report_id": "2" * 64},
        task_prefix=task_prefix,
        campaign_root=artifact_root,
    )

    assert len(tasks) == 20
    assert all(task["id"].startswith(f"{task_prefix}-") for task in tasks)
    assert all(
        Path(task["artifact_path"]).is_relative_to(artifact_root)
        for task in tasks
    )
    assert artifact_root == (
        CAMPAIGN_ROOT / "launches" / "adr0172-relaunch-v1"
    )
    run = next(
        task
        for task in tasks
        if task["id"] == f"{task_prefix}-run-t1_supply_query"
    )
    run_dir = run["command"][run["command"].index("--run-dir") + 1]
    assert str(artifact_root) in run_dir
