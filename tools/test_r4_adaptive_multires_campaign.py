from pathlib import Path

import r4_adaptive_multires_campaign as campaign


def test_r4_graph_assigns_unique_corpus_parts_to_all_hosts() -> None:
    specs = campaign.build_task_specs(
        Path("artifacts/experiments/r4/bundle/example")
    )
    by_id = {spec["id"]: spec for spec in specs}
    assert len(specs) == 11
    assert len(by_id) == len(specs)

    preflight_ids = {
        f"{campaign.TASK_PREFIX}-preflight-{host}" for host in campaign.HOSTS
    }
    for shard_index, host in enumerate(campaign.HOSTS):
        task = by_id[f"{campaign.TASK_PREFIX}-census-{host}"]
        assert task["compatible_hosts"] == [host]
        assert set(task["dependencies"]) == preflight_ids
        command = task["command"]
        assert f"--shard-index" in command
        assert command[command.index("--shard-index") + 1] == str(shard_index)
        assert (
            f"r0-spatial-position-corpus-v1-source-frozen-train-part-{shard_index}"
            in " ".join(command)
        )
        assert (
            f"r0-spatial-position-corpus-v1-source-frozen-validation-part-{shard_index}"
            in " ".join(command)
        )


def test_r4_graph_closes_with_collection_parity_and_aggregate() -> None:
    specs = campaign.build_task_specs(
        Path("artifacts/experiments/r4/bundle/example")
    )
    by_id = {spec["id"]: spec for spec in specs}
    collect = by_id[f"{campaign.TASK_PREFIX}-collect"]
    assert set(collect["dependencies"]) == {
        f"{campaign.TASK_PREFIX}-census-{host}" for host in campaign.HOSTS
    }
    assert collect["command"].count("--artifact") == 8
    for argument in collect["command"]:
        if "collected/" in argument or argument.endswith("collection.json"):
            assert argument.startswith("/Users/johnherrick/cascadia/")

    parity = by_id[f"{campaign.TASK_PREFIX}-adversarial-parity"]
    assert parity["dependencies"] == [f"{campaign.TASK_PREFIX}-collect"]
    assert parity["command"].count("--report") == 4
    assert "--require-pass" in parity["command"]

    aggregate = by_id[f"{campaign.TASK_PREFIX}-aggregate"]
    assert aggregate["decision_terminal"] is True
    assert aggregate["dependencies"] == [
        f"{campaign.TASK_PREFIX}-adversarial-parity"
    ]
    assert aggregate["command"].count("--report") == 4
    assert "--order-proof-output" in aggregate["command"]


def test_r4_postprocess_recovery_reuses_completed_censuses() -> None:
    specs = campaign.build_postprocess_recovery_specs(
        Path("artifacts/experiments/r4/bundle/example"),
        "-pathfix1",
    )
    assert [spec["id"] for spec in specs] == [
        "r4am-collect-pathfix1",
        "r4am-adversarial-parity-pathfix1",
        "r4am-aggregate-pathfix1",
    ]
    assert set(specs[0]["dependencies"]) == {
        f"r4am-census-{host}" for host in campaign.HOSTS
    }
    assert specs[1]["dependencies"] == ["r4am-collect-pathfix1"]
    assert specs[2]["dependencies"] == [
        "r4am-adversarial-parity-pathfix1"
    ]
