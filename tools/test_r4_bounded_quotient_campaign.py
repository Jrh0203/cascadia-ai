from pathlib import Path

import r4_bounded_quotient_campaign as campaign


def _specs() -> list[dict]:
    return campaign.build_task_specs(
        Path("artifacts/experiments/r4-bounded/bundle/example")
    )


def test_graph_runs_cross_host_parity_before_production() -> None:
    specs = _specs()
    by_id = {spec["id"]: spec for spec in specs}
    assert len(specs) == 12
    assert len(by_id) == len(specs)

    preflights = {
        f"{campaign.TASK_PREFIX}-preflight-{host}" for host in campaign.HOSTS
    }
    collect = by_id[f"{campaign.TASK_PREFIX}-collect-preflight"]
    assert set(collect["dependencies"]) == preflights
    assert collect["command"].count("--artifact") == 4

    parity_id = f"{campaign.TASK_PREFIX}-adversarial-parity"
    parity = by_id[parity_id]
    assert parity["dependencies"] == [
        f"{campaign.TASK_PREFIX}-collect-preflight"
    ]
    assert parity["command"].count("--report") == 4
    assert "--require-pass" in parity["command"]

    for host in campaign.HOSTS:
        census = by_id[f"{campaign.TASK_PREFIX}-census-{host}"]
        assert census["dependencies"] == [parity_id]


def test_four_hosts_run_four_distinct_full_corpus_arms() -> None:
    specs = _specs()
    by_id = {spec["id"]: spec for spec in specs}
    seen_arms = set()
    for host in campaign.HOSTS:
        census = by_id[f"{campaign.TASK_PREFIX}-census-{host}"]
        command = census["command"]
        arm = command[command.index("--arm") + 1]
        seen_arms.add(arm)
        assert arm == campaign.HOST_ARMS[host]
        assert census["compatible_hosts"] == [host]
        assert command.count("--dataset-root") == 8
        joined = " ".join(command)
        for part in range(4):
            assert f"frozen-train-part-{part}" in joined
            assert f"frozen-validation-part-{part}" in joined
        assert "--require-frozen" in command
    assert seen_arms == set(campaign.HOST_ARMS.values())


def test_graph_collects_unique_arms_and_closes_with_order_proof() -> None:
    specs = _specs()
    by_id = {spec["id"]: spec for spec in specs}
    collect = by_id[f"{campaign.TASK_PREFIX}-collect-arms"]
    assert set(collect["dependencies"]) == {
        f"{campaign.TASK_PREFIX}-census-{host}" for host in campaign.HOSTS
    }
    assert collect["command"].count("--artifact") == 4

    aggregate = by_id[f"{campaign.TASK_PREFIX}-aggregate"]
    assert aggregate["decision_terminal"] is True
    assert aggregate["dependencies"] == [
        f"{campaign.TASK_PREFIX}-collect-arms"
    ]
    assert aggregate["command"].count("--report") == 4
    assert "--adversarial-parity-report" in aggregate["command"]
    assert "--order-proof-output" in aggregate["command"]

    for task in specs:
        for argument in task["command"]:
            if (
                "collected/" in argument
                or "reports/" in argument
                or "arms/" in argument
                or argument.endswith("aggregate-forward.json")
                or argument.endswith("aggregate-reverse.json")
                or argument.endswith("order-proof.json")
            ):
                path = argument.split(":", 1)[-1]
                assert path.startswith(
                    tuple(str(root) for root in campaign.REMOTE_ROOTS.values())
                )
