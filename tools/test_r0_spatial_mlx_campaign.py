from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import r0_spatial_mlx_campaign as campaign
from cluster_research_queue import (
    add_task,
    claim_next,
    empty_queue,
    finish_task,
    load_queue,
)


def _write_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    monkeypatch.setattr(campaign, "Dataset", lambda *args, **kwargs: object())
    roots = []
    source = "7" * 64
    cards = {wildlife: "A" for wildlife in ("bear", "elk", "salmon", "hawk", "fox")}
    for part in campaign.dataset_parts(tmp_path / "corpus"):
        part.root.mkdir(parents=True)
        manifest = {
            "schema_version": 1,
            "dataset_id": (f"{campaign.STRATEGY_ID}-{part.split}-{part.first_game_index}"),
            "feature_schema": campaign.FEATURE_SCHEMA,
            "target_schema": campaign.TARGET_SCHEMA,
            "record_size": 864,
            "game": {
                "player_count": 4,
                "mode": "Standard",
                "scoring_cards": cards,
                "habitat_bonuses": False,
            },
            "split": part.split,
            "strategy": campaign.STRATEGY_ID,
            "first_game_index": part.first_game_index,
            "requested_games": part.games,
            "completed_games": part.games,
            "total_records": part.records,
            "provenance": {"v2_source_blake3": source},
            "shards": [
                {
                    "file": "shard-00000.csd",
                    "first_game_index": part.first_game_index,
                    "game_count": part.games,
                    "record_count": part.records,
                    "byte_count": 0,
                    "blake3": "8" * 64,
                }
            ],
        }
        (part.root / "dataset.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        roots.append(part.root)
    return roots


def _lock(path: Path) -> dict:
    identity = {
        "feature_schema": campaign.FEATURE_SCHEMA,
        "target_schema": campaign.TARGET_SCHEMA,
        "total_records": 60_000,
        "train_records": 50_000,
        "validation_records": 10_000,
        "source_v2_blake3": "1" * 64,
        "corpus_blake3": "2" * 64,
        "datasets": [{"order": index} for index in range(8)],
    }
    value = {
        "schema_version": 1,
        "contract_id": campaign.CORPUS_LOCK_CONTRACT,
        "lock_id": campaign.canonical_blake3(identity),
        "identity": identity,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return value


def test_corpus_lock_is_deterministic_and_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = _write_corpus(tmp_path, monkeypatch)
    first = campaign.freeze_corpus(roots, dataset_root=tmp_path / "corpus")
    second = campaign.freeze_corpus(roots, dataset_root=tmp_path / "corpus")
    assert first == second
    assert first["identity"]["train_records"] == 50_000
    assert first["identity"]["validation_records"] == 10_000
    assert [entry["order"] for entry in first["identity"]["datasets"]] == list(range(8))
    assert campaign.canonical_blake3(first["identity"]) == first["lock_id"]


def test_corpus_lock_rejects_ruleset_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = _write_corpus(tmp_path, monkeypatch)
    manifest_path = roots[0] / "dataset.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["game"]["habitat_bonuses"] = True
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(campaign.CampaignError, match="frozen R0 contract"):
        campaign.freeze_corpus(roots, dataset_root=tmp_path / "corpus")


def test_authorization_pins_bundle_protocol_source_corpus_and_binary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    lock_path = tmp_path / "control/corpus-lock.json"
    lock = _lock(lock_path)
    bundle_manifest = {
        "bundle_id": "3" * 64,
        "identity": {"binaries": [{"name": campaign.EXPORTER_BINARY, "blake3": "4" * 64}]},
    }
    monkeypatch.setattr(
        campaign,
        "validate_bundle_for_campaign",
        lambda path: bundle_manifest,
    )
    monkeypatch.setattr(
        campaign,
        "source_provenance",
        lambda path: {"v2_source_blake3": "5" * 64},
    )
    monkeypatch.setattr(campaign, "file_blake3", lambda path: "4" * 64)
    authorization = campaign.create_authorization(
        bundle=bundle,
        corpus_lock=lock_path,
        approved_by="parent",
        approved_unix_ms=99,
    )
    identity = authorization["identity"]
    assert identity["bundle_id"] == "3" * 64
    assert identity["corpus_lock_id"] == lock["lock_id"]
    assert identity["mlx_source_blake3"] == "5" * 64
    assert identity["exporter_executable_blake3"] == "4" * 64
    assert identity["authorized_arms"] == list(campaign.ARM_ORDER)
    assert campaign.canonical_blake3(identity) == authorization["authorization_id"]


def _task_specs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[dict], Path]:
    repository = tmp_path / "repo"
    bundle = repository / "artifacts/experiment/bundles/bundle-id"
    control = repository / "artifacts/experiment/control"
    queue = repository / "artifacts/cluster/research-queue-v1.json"
    experiment = repository / "artifacts/experiment"
    bundle.mkdir(parents=True)
    control.mkdir(parents=True)
    lock = control / "corpus-lock.json"
    authorization = control / "authorization.json"
    lock.touch()
    authorization.touch()
    monkeypatch.setattr(
        campaign,
        "validate_bundle_for_campaign",
        lambda path: {"bundle_id": "a" * 64},
    )
    monkeypatch.setattr(
        campaign,
        "validate_authorization",
        lambda *args, **kwargs: {"authorization_id": "b" * 64},
    )
    specs = campaign.build_task_specs(
        repository=repository,
        bundle=bundle,
        corpus_lock=lock,
        authorization=authorization,
        queue=queue,
        experiment_root=experiment,
        dataset_root=Path("artifacts/datasets/corpus"),
    )
    return specs, queue


def test_queue_graph_pins_four_primary_arms_and_backfills_historical441(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs, _queue = _task_specs(tmp_path, monkeypatch)
    by_id = {spec["id"]: spec for spec in specs}
    assert len(specs) == 15
    for arm, host in campaign.PRIMARY_ARMS:
        task = by_id[f"r0mlx-arm-{campaign._slug(arm)}"]
        assert task["compatible_hosts"] == [host]
        assert task["priority"] == 10
        assert task["resources"]["uses_mlx"] is True
    historical = by_id["r0mlx-arm-historical441"]
    assert historical["compatible_hosts"] == list(campaign.HOSTS)
    assert historical["priority"] == 20
    assert set(historical["dependencies"]) == {f"r0mlx-preflight-{host}" for host in campaign.HOSTS}
    assert historical["command"][:2] == ["/bin/zsh", "-lc"]
    assert "$R0_ROOT" in historical["command"][2]
    assert "/.venv/bin/python -B " in historical["command"][2]
    assert sum(spec["id"].startswith("r0mlx-preflight-") for spec in specs) == 4
    assert by_id["r0mlx-classification-order-proof"]["decision_terminal"] is True
    for spec in specs:
        command = spec["command"]
        if command[:2] == ["/bin/zsh", "-lc"]:
            continue
        python_indexes = [
            index for index, value in enumerate(command) if value.endswith("/.venv/bin/python")
        ]
        for index in python_indexes:
            assert command[index + 1] == "-B"


def test_portable_historical_command_is_valid_zsh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs, _queue = _task_specs(tmp_path, monkeypatch)
    command = next(spec["command"][2] for spec in specs if spec["id"] == "r0mlx-arm-historical441")
    completed = subprocess.run(
        ["/bin/zsh", "-n", "-c", command],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_queue_install_is_atomic_and_rejects_duplicate_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs, queue = _task_specs(tmp_path, monkeypatch)
    queue.parent.mkdir(parents=True, exist_ok=True)
    queue.write_text(json.dumps(empty_queue("test-campaign", now_ms=1)))
    campaign.install_task_specs(queue, specs)
    state = load_queue(queue)
    assert len(state["tasks"]) == 15
    with pytest.raises(campaign.CampaignError, match="already contains"):
        campaign.install_task_specs(queue, specs)


def test_scheduler_claims_primaries_before_backfilling_historical441(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs, _queue = _task_specs(tmp_path, monkeypatch)
    state = empty_queue("test-campaign", now_ms=1)
    for spec in specs:
        add_task(state, spec, now_ms=2)

    def complete_next(host: str, expected_id: str, now_ms: int) -> None:
        task = claim_next(state, host=host, lease_seconds=100, now_ms=now_ms)
        assert task is not None
        assert task["id"] == expected_id
        finish_task(
            state,
            task_id=task["id"],
            host=host,
            token=task["claim"]["token"],
            outcome="completed",
            artifact=task["artifact_path"],
            now_ms=now_ms + 1,
        )

    complete_next("john1", "r0mlx-bundle-fanout", 10)
    complete_next("john1", "r0mlx-control-fanout", 20)
    for index, host in enumerate(campaign.HOSTS):
        complete_next(host, f"r0mlx-preflight-{host}", 30 + index * 2)

    primary_claims = {}
    for index, (arm, host) in enumerate(campaign.PRIMARY_ARMS):
        task = claim_next(state, host=host, lease_seconds=100, now_ms=50 + index)
        assert task is not None
        assert task["id"] == f"r0mlx-arm-{campaign._slug(arm)}"
        primary_claims[host] = task

    released_host = "john3"
    released = primary_claims[released_host]
    finish_task(
        state,
        task_id=released["id"],
        host=released_host,
        token=released["claim"]["token"],
        outcome="completed",
        artifact=released["artifact_path"],
        now_ms=60,
    )
    backfill = claim_next(state, host=released_host, lease_seconds=100, now_ms=61)
    assert backfill is not None
    assert backfill["id"] == "r0mlx-arm-historical441"


def test_classification_order_proof_rejects_any_byte_drift(tmp_path: Path) -> None:
    forward = tmp_path / "forward.json"
    reverse = tmp_path / "reverse.json"
    forward.write_bytes(b'{"same":true}\n')
    reverse.write_bytes(b'{"same":true}\n')
    assert campaign.compare_classifications(forward, reverse)["byte_identical"] is True
    reverse.write_bytes(b'{"same":false}\n')
    with pytest.raises(campaign.CampaignError, match="differ"):
        campaign.compare_classifications(forward, reverse)
