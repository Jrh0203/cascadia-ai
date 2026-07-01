from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cascadia_mlx.r2_map_campaign_controller as controller
import pytest
from cascadia_mlx.r2_map_campaign_controller import (
    MAX_ATTEMPTS,
    CampaignControllerError,
    ControllerPaths,
    advance_campaign,
    build_phase_packets,
    import_benchmark_feed,
    import_receipt,
    initialize_controller,
    make_synthetic_receipt,
    phase_barrier,
    phase_templates,
    reconcile,
    recover_current_phase,
    run_isolated_dry_run,
    validate_receipt,
    validate_work_packet,
)
from cascadia_mlx.r2_map_contracts import (
    ALLOWED_HOSTS,
    PHASE_HOST_INTENTS,
    Phase,
    content_sha256,
    read_state,
)
from cascadia_mlx.r2_map_dashboard_status import DashboardStatusInputs, build_dashboard_status

ledger = controller.experiment_ledger
queue = controller.research_queue


def _commands() -> dict[str, list[str]]:
    return {
        template.operation: ["/usr/bin/true"]
        for phase in Phase
        for template in phase_templates(phase)
    }


def _initialize(tmp_path: Path) -> ControllerPaths:
    paths = ControllerPaths.under(tmp_path / "campaign")
    initialize_controller(paths, now_ms=1)
    return paths


def test_initialize_publishes_complete_versioned_packet_and_receipt_schemas(
    tmp_path: Path,
) -> None:
    paths = _initialize(tmp_path)
    schemas = {
        "work-packet": controller.WORK_PACKET_JSON_SCHEMA,
        "work-receipt": controller.WORK_RECEIPT_JSON_SCHEMA,
    }
    for name, schema in schemas.items():
        path = paths.root / f"control/contracts/r2-map-{name}-v2.schema.json"
        assert json.loads(path.read_text()) == schema
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])


def _advance(paths: ControllerPaths, **values: object) -> dict[str, object]:
    return advance_campaign(
        paths,
        commands=_commands(),
        artifact_root="reports/artifacts",
        reason="unit transition",
        now="1970-01-01T00:00:01.000Z",
        now_ms=10,
        synthetic=True,
        **values,
    )


def _complete_queue_task(paths: ControllerPaths, packet: dict[str, object], now_ms: int) -> None:
    with queue.locked_queue(paths.queue) as state:
        claimed = queue.claim_next(
            state,
            host=str(packet["host"]),
            lease_seconds=30,
            now_ms=now_ms,
        )
        assert claimed is not None
        assert claimed["id"] == packet["task_id"]
        queue.finish_task(
            state,
            task_id=str(packet["task_id"]),
            host=str(packet["host"]),
            token=claimed["claim"]["token"],
            outcome="completed",
            artifact=claimed["artifact_path"],
            now_ms=now_ms + 1,
        )


def test_every_named_task_has_a_typed_host_fixed_packet() -> None:
    kinds = {template.kind for phase in Phase for template in phase_templates(phase)}
    assert kinds == {
        "generate",
        "train",
        "longitudinal-benchmark",
        "candidate-gate",
        "aggregate",
    }
    for phase in Phase:
        for template in phase_templates(phase):
            assert template.host in ALLOWED_HOSTS
            assert template.host != "john4"


def test_queue_artifacts_are_remote_uris_not_local_paths(tmp_path: Path) -> None:
    paths = _initialize(tmp_path)
    _advance(paths)
    packet = validate_work_packet(json.loads(next(paths.packets.glob("*.json")).read_text()))
    task = controller.queue_task_for_packet(packet, created_unix_ms=1)
    assert task["artifact_path"].startswith("r2map+ssh://john2/")
    assert "/Volumes/John_1" not in task["artifact_path"]


def test_training_phase_reserves_john1_for_mlx_and_john2_john3_for_benchmark() -> None:
    templates = phase_templates(Phase.TRAINING_AND_BENCHMARKING)
    by_host = {host: [] for host in ALLOWED_HOSTS}
    for template in templates:
        by_host[template.host].append(template.kind)
    assert by_host["john1"] == ["train", "aggregate"]
    assert by_host["john2"] == ["longitudinal-benchmark"]
    assert by_host["john3"] == ["longitudinal-benchmark"]
    assert all(template.kind != "generate" for template in templates)
    assert PHASE_HOST_INTENTS[Phase.TRAINING_AND_BENCHMARKING] == {
        "john1": "train",
        "john2": "benchmark",
        "john3": "benchmark",
    }


def test_bootstrap_seed_leases_are_disjoint_and_total_exactly_100000(
    tmp_path: Path,
) -> None:
    paths = _initialize(tmp_path)
    _advance(paths)
    state = read_state(paths.state)
    packets = build_phase_packets(
        state,
        commands=_commands(),
        artifact_root="reports/artifacts",
        synthetic=True,
    )
    leases = [packet["seed_lease"] for packet in packets if packet["task_kind"] == "generate"]
    assert sum(lease["count"] for lease in leases) == 100_000
    covered = [
        set(range(lease["first_index"], lease["first_index"] + lease["count"])) for lease in leases
    ]
    assert not covered[0] & covered[1]
    assert not covered[0] & covered[2]
    assert not covered[1] & covered[2]


def test_packet_hash_tamper_and_john4_are_rejected(tmp_path: Path) -> None:
    paths = _initialize(tmp_path)
    _advance(paths)
    packet_path = next(paths.packets.glob("*.json"))
    packet = json.loads(packet_path.read_text())
    packet["command"] = ["/usr/bin/false"]
    with pytest.raises(CampaignControllerError, match="hash differs"):
        validate_work_packet(packet)
    packet = json.loads(packet_path.read_text())
    packet["command"] = ["ssh", "john4"]
    packet["packet_sha256"] = content_sha256(packet, hash_field="packet_sha256")
    with pytest.raises(CampaignControllerError, match="john4"):
        validate_work_packet(packet)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("storage", {"host": "john1", "root": "/tmp", "transport": "local"}, "storage"),
        ("artifact_root", "/Volumes/John_1/new", "remote-relative"),
        ("artifact_root", "/Users/johnherrick/new", "remote-relative"),
        ("artifact_root", "../escape", "remote-relative"),
    ],
)
def test_rehashed_packet_cannot_redirect_authoritative_storage(
    tmp_path: Path,
    field: str,
    replacement: object,
    message: str,
) -> None:
    paths = _initialize(tmp_path)
    _advance(paths)
    packet = json.loads(next(paths.packets.glob("*.json")).read_text())
    packet[field] = replacement
    packet["packet_sha256"] = content_sha256(packet, hash_field="packet_sha256")
    with pytest.raises(CampaignControllerError, match=message):
        validate_work_packet(packet)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("operation", "round-train", "operation is not registered"),
        ("task_kind", "candidate-gate", "registered phase template"),
        ("host", "john2", "registered phase template"),
    ],
)
def test_rehashed_packet_cannot_escape_its_phase_template(
    tmp_path: Path,
    field: str,
    replacement: str,
    message: str,
) -> None:
    paths = _initialize(tmp_path)
    _advance(paths)
    packet = json.loads(next(paths.packets.glob("*john1.json")).read_text())
    packet[field] = replacement
    packet["packet_sha256"] = content_sha256(packet, hash_field="packet_sha256")
    with pytest.raises(CampaignControllerError, match=message):
        validate_work_packet(packet)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("purpose", "candidate-gate"),
        ("first_index", 99),
        ("count", 1),
        ("stride", 2),
        ("round_index", 7),
    ],
)
def test_rehashed_seed_lease_cannot_escape_its_phase_domain(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    paths = _initialize(tmp_path)
    _advance(paths)
    packet = json.loads(next(paths.packets.glob("*john1.json")).read_text())
    packet["seed_lease"][field] = replacement
    packet["seed_lease"]["lease_sha256"] = content_sha256(
        packet["seed_lease"], hash_field="lease_sha256"
    )
    packet["packet_sha256"] = content_sha256(packet, hash_field="packet_sha256")
    with pytest.raises(CampaignControllerError, match="seed lease differs"):
        validate_work_packet(packet)


def test_receipt_rejects_seed_gap_swap_growth_and_failed_gate(tmp_path: Path) -> None:
    paths = _initialize(tmp_path)
    _advance(paths)
    packet = validate_work_packet(json.loads(next(paths.packets.glob("*john2.json")).read_text()))
    receipt = make_synthetic_receipt(packet, completed_unix_ms=20)

    redirected = json.loads(json.dumps(receipt))
    redirected["storage"]["host"] = "john1"
    redirected["receipt_sha256"] = content_sha256(
        redirected, hash_field="receipt_sha256"
    )
    with pytest.raises(CampaignControllerError, match="storage"):
        validate_receipt(redirected, packet=packet)

    escaped = json.loads(json.dumps(receipt))
    escaped["artifacts"][0]["path"] = "../john1-output"
    escaped["receipt_sha256"] = content_sha256(escaped, hash_field="receipt_sha256")
    with pytest.raises(CampaignControllerError, match="remote-relative"):
        validate_receipt(escaped, packet=packet)

    missing_receipt = json.loads(json.dumps(receipt))
    missing_receipt["artifacts"][0]["storage_receipt_relative"] = (
        "reports/not-a-worker-receipt.json"
    )
    missing_receipt["receipt_sha256"] = content_sha256(
        missing_receipt, hash_field="receipt_sha256"
    )
    with pytest.raises(CampaignControllerError, match="control/receipts"):
        validate_receipt(missing_receipt, packet=packet)

    invalid_size = json.loads(json.dumps(receipt))
    invalid_size["artifacts"][0]["bytes"] = True
    invalid_size["receipt_sha256"] = content_sha256(
        invalid_size, hash_field="receipt_sha256"
    )
    with pytest.raises(CampaignControllerError, match="artifact identity"):
        validate_receipt(invalid_size, packet=packet)

    gap = json.loads(json.dumps(receipt))
    gap["used_seed_prefix"]["last_index"] += 2
    gap["receipt_sha256"] = content_sha256(gap, hash_field="receipt_sha256")
    with pytest.raises(CampaignControllerError, match="contiguous"):
        validate_receipt(gap, packet=packet)

    swapped = json.loads(json.dumps(receipt))
    swapped["metrics"]["system_swap_delta_bytes"] = 1
    swapped["receipt_sha256"] = content_sha256(swapped, hash_field="receipt_sha256")
    with pytest.raises(CampaignControllerError, match="zero-swap"):
        validate_receipt(swapped, packet=packet)

    process_swapped = json.loads(json.dumps(receipt))
    process_swapped["metrics"]["process_swaps"] = 1
    process_swapped["receipt_sha256"] = content_sha256(process_swapped, hash_field="receipt_sha256")
    with pytest.raises(CampaignControllerError, match="zero-swap"):
        validate_receipt(process_swapped, packet=packet)

    failed = json.loads(json.dumps(receipt))
    failed["gates"]["identity"] = False
    failed["receipt_sha256"] = content_sha256(failed, hash_field="receipt_sha256")
    with pytest.raises(CampaignControllerError, match="failed scientific"):
        validate_receipt(failed, packet=packet)


def test_receipt_import_is_host_contained_idempotent_and_barrier_strict(
    tmp_path: Path,
) -> None:
    paths = _initialize(tmp_path)
    _advance(paths)
    packet = validate_work_packet(json.loads(next(paths.packets.glob("*john1.json")).read_text()))
    _complete_queue_task(paths, packet, 20)
    receipt = make_synthetic_receipt(packet, completed_unix_ms=21)
    receipt = controller._install_synthetic_storage_evidence(paths, packet, receipt)
    incoming = paths.incoming / "john1" / f"{packet['task_id']}.json"
    incoming.parent.mkdir(parents=True)
    incoming.write_text(json.dumps(receipt))
    first = import_receipt(paths, source=incoming)
    second = import_receipt(paths, source=incoming)
    assert first == second
    with pytest.raises(CampaignControllerError, match="not complete"):
        phase_barrier(paths)

    wrong_host = paths.incoming / "john2" / f"{packet['task_id']}.json"
    wrong_host.parent.mkdir(parents=True)
    wrong_host.write_text(json.dumps(receipt))
    with pytest.raises(CampaignControllerError, match="directory differs"):
        import_receipt(paths, source=wrong_host)


@pytest.mark.parametrize(
    "tamper",
    [
        "locator",
        "artifact-path",
        "artifact-size",
        "artifact-sha256",
        "storage-receipt-document",
        "storage-receipt-mode",
        "artifact-bytes",
    ],
)
def test_receipt_import_resolves_persisted_storage_evidence_and_rejects_tampering(
    tmp_path: Path,
    tamper: str,
) -> None:
    paths = _initialize(tmp_path)
    _advance(paths)
    packet = validate_work_packet(
        json.loads(next(paths.packets.glob("*john1.json")).read_text())
    )
    _complete_queue_task(paths, packet, 20)
    receipt = make_synthetic_receipt(packet, completed_unix_ms=21)
    receipt = controller._install_synthetic_storage_evidence(paths, packet, receipt)
    artifact = receipt["artifacts"][0]
    storage_path = paths.root / artifact["storage_receipt_relative"]
    artifact_path = paths.root / artifact["path"]

    if tamper == "locator":
        artifact["storage_receipt_relative"] = "control/receipts/req-missing.json"
    elif tamper == "artifact-path":
        artifact["path"] = f"{packet['artifact_root']}/different.artifact"
    elif tamper == "artifact-size":
        artifact["bytes"] += 1
    elif tamper == "artifact-sha256":
        artifact["sha256"] = "f" * 64
    elif tamper == "storage-receipt-document":
        stored = json.loads(storage_path.read_bytes())
        stored["result"]["size"] += 1
        stored["receipt_sha256"] = content_sha256(
            stored, hash_field="receipt_sha256"
        )
        storage_path.chmod(0o600)
        storage_path.write_bytes(controller.canonical_json_bytes(stored))
        storage_path.chmod(0o400)
    elif tamper == "storage-receipt-mode":
        storage_path.chmod(0o600)
    elif tamper == "artifact-bytes":
        artifact_path.chmod(0o600)
        artifact_path.write_bytes(artifact_path.read_bytes() + b"tamper")
        artifact_path.chmod(0o400)
    else:  # pragma: no cover - exhaustive parameter guard
        raise AssertionError(tamper)

    receipt["receipt_sha256"] = content_sha256(receipt, hash_field="receipt_sha256")
    incoming = paths.incoming / "john1" / f"{packet['task_id']}.json"
    incoming.parent.mkdir(parents=True, exist_ok=True)
    incoming.write_text(json.dumps(receipt))
    with pytest.raises(
        CampaignControllerError,
        match=r"artifact|publication|receipt|resolved|mutable|identity",
    ):
        import_receipt(paths, source=incoming)


def _install_transaction_receipt_fixture(
    paths: ControllerPaths,
    *,
    tamper_manifest: bool,
) -> tuple[dict[str, object], Path]:
    packet = validate_work_packet(
        json.loads(next(paths.packets.glob("*john1.json")).read_text())
    )
    _complete_queue_task(paths, packet, 20)
    target = f"{packet['artifact_root']}/tx-unit"
    artifact_relative = f"{target}/bundle.bin"
    payload = b"immutable transaction artifact"
    artifact_sha256 = hashlib.sha256(payload).hexdigest()
    manifest: dict[str, object] = {
        "schema_version": 1,
        "schema_id": controller.REMOTE_TRANSACTION_SCHEMA,
        "transaction_id": "tx-unit",
        "target_relative": target,
        "objects": [
            {
                "relative": "bundle.bin",
                "sha256": artifact_sha256,
                "size": len(payload),
            }
        ],
    }
    manifest["manifest_sha256"] = content_sha256(
        manifest, hash_field="manifest_sha256"
    )
    artifact_path = paths.root / artifact_relative
    manifest_path = paths.root / target / ".r2-map-transaction.json"
    controller._write_immutable_bytes(artifact_path, payload)
    controller._write_immutable_bytes(
        manifest_path, controller.canonical_json_bytes(manifest)
    )
    (paths.root / target).chmod(0o500)

    locator = "control/receipts/req-transaction-unit.json"
    storage_receipt: dict[str, object] = {
        "schema_version": 1,
        "schema_id": controller.REMOTE_RECEIPT_SCHEMA,
        "request_id": "req-transaction-unit",
        "command_sha256": hashlib.sha256(b"transaction command").hexdigest(),
        "operation": "transaction-commit",
        "status": "ok",
        "host": "john2",
        "host_identity_sha256": hashlib.sha256(b"john2 identity").hexdigest(),
        "root": str(controller.CAMPAIGN_ROOT),
        "completed_unix_ms": 21,
        "result": {
            "transaction_id": "tx-unit",
            "target_relative": target,
            "manifest_sha256": manifest["manifest_sha256"],
            "object_count": 1,
            "committed": True,
            "payload_size": 0,
            "payload_sha256": controller.EMPTY_SHA256,
        },
    }
    storage_receipt["receipt_sha256"] = content_sha256(
        storage_receipt, hash_field="receipt_sha256"
    )
    controller._write_immutable_bytes(
        paths.root / locator,
        controller.canonical_json_bytes(storage_receipt),
    )

    receipt = make_synthetic_receipt(packet, completed_unix_ms=21)
    receipt["artifacts"] = [
        {
            "label": receipt["artifacts"][0]["label"],
            "path": artifact_relative,
            "bytes": len(payload),
            "sha256": artifact_sha256,
            "storage_receipt_relative": locator,
            "storage_receipt_sha256": storage_receipt["receipt_sha256"],
        }
    ]
    receipt["receipt_sha256"] = content_sha256(receipt, hash_field="receipt_sha256")

    if tamper_manifest:
        changed = json.loads(manifest_path.read_bytes())
        changed["objects"][0]["sha256"] = "f" * 64
        changed["manifest_sha256"] = content_sha256(
            changed, hash_field="manifest_sha256"
        )
        (paths.root / target).chmod(0o700)
        manifest_path.chmod(0o600)
        manifest_path.write_bytes(controller.canonical_json_bytes(changed))
        manifest_path.chmod(0o400)
        (paths.root / target).chmod(0o500)

    incoming = paths.incoming / "john1" / f"{packet['task_id']}.json"
    incoming.parent.mkdir(parents=True, exist_ok=True)
    incoming.write_text(json.dumps(receipt))
    return packet, incoming


def test_transaction_receipt_resolves_manifest_and_artifact(tmp_path: Path) -> None:
    paths = _initialize(tmp_path)
    _advance(paths)
    packet, incoming = _install_transaction_receipt_fixture(
        paths, tamper_manifest=False
    )
    imported = import_receipt(paths, source=incoming)
    assert imported["task_id"] == packet["task_id"]


def test_transaction_receipt_rejects_manifest_tampering(tmp_path: Path) -> None:
    paths = _initialize(tmp_path)
    _advance(paths)
    _, incoming = _install_transaction_receipt_fixture(paths, tamper_manifest=True)
    with pytest.raises(CampaignControllerError, match="transaction provenance"):
        import_receipt(paths, source=incoming)


def _benchmark_feed() -> dict[str, object]:
    return {
        "id": "r2-map-expert-iteration-v1-longitudinal-r0",
        "title": "R2-MAP longitudinal focal benchmark",
        "hypothesis": "The frozen incumbent remains measurable while training is independent.",
        "summary": "Completed the fixed open panel.",
        "status": "completed",
        "outcome": "passed",
        "verdict": None,
        "plan_section": "W5",
        "started_unix_ms": 0,
        "completed_unix_ms": 0,
        "updated_unix_ms": 0,
        "hosts": ["john2", "john3"],
        "tags": ["r2-map", "longitudinal"],
        "task_ids": [],
        "metrics": [{"label": "Games", "value": "100", "tone": "neutral"}],
        "criteria": [{"label": "Complete fixed coverage", "passed": True, "observed": "100"}],
        "notes": ["Deterministic projection with zero placeholder times."],
        "artifacts": [
            {
                "label": "R2-MAP plan",
                "path": "docs/v2/R2_MAP_EXPERT_ITERATION_RESEARCH_PLAN.md",
            }
        ],
    }


def _install_benchmark_feed_fixture(
    paths: ControllerPaths,
) -> tuple[Path, dict[str, object], dict[str, object]]:
    packets = list(controller._phase_packets_on_disk(paths, read_state(paths.state)))
    non_aggregate = sorted(
        (packet for packet in packets if packet["task_kind"] != "aggregate"),
        key=lambda packet: (packet["task_kind"] != "train", packet["task_id"]),
    )
    aggregate = next(packet for packet in packets if packet["task_kind"] == "aggregate")
    for index, packet in enumerate([*non_aggregate, aggregate]):
        _complete_queue_task(paths, packet, 100 + index * 2)
    aggregate_path = paths.packets / f"{aggregate['task_id']}.json"
    packet = json.loads(aggregate_path.read_text())
    feed_path = paths.root / packet["artifact_root"] / "ledger-experiment.json"
    feed_path.parent.mkdir(parents=True, exist_ok=True)
    feed_path.write_text(json.dumps(_benchmark_feed(), sort_keys=True, indent=2) + "\n")
    receipt = make_synthetic_receipt(packet, completed_unix_ms=1234)
    receipt["artifacts"] = [
        {
            "label": "benchmark-ledger-feed",
            "path": feed_path.relative_to(paths.root).as_posix(),
            "bytes": feed_path.stat().st_size,
            "sha256": hashlib.sha256(feed_path.read_bytes()).hexdigest(),
            "storage_receipt_relative": "control/receipts/req-benchmark-feed.json",
            "storage_receipt_sha256": "e" * 64,
        }
    ]
    receipt["receipt_sha256"] = content_sha256(receipt, hash_field="receipt_sha256")
    receipt = controller._install_synthetic_storage_evidence(paths, packet, receipt)
    receipt_path = paths.receipts / f"{packet['task_id']}.json"
    controller._write_immutable_json(receipt_path, receipt)
    return feed_path, packet, receipt


def _reach_training_and_benchmarking(paths: ControllerPaths) -> None:
    while Phase(read_state(paths.state)["phase"]) is not Phase.TRAINING_AND_BENCHMARKING:
        phase = Phase(read_state(paths.state)["phase"])
        if phase_templates(phase):
            controller.complete_synthetic_phase(paths, now_ms=90)
        _advance(paths)


def test_benchmark_feed_import_is_receipt_bound_cas_safe_and_idempotent(
    tmp_path: Path,
) -> None:
    paths = _initialize(tmp_path)
    _advance(paths)
    _reach_training_and_benchmarking(paths)
    feed_path, packet, receipt = _install_benchmark_feed_fixture(paths)
    state_sha256 = read_state(paths.state)["state_sha256"]

    with pytest.raises(CampaignControllerError, match="state CAS differs"):
        import_benchmark_feed(
            paths,
            feed_path=feed_path,
            aggregate_task_id=str(packet["task_id"]),
            expected_state_sha256="f" * 64,
        )

    first = import_benchmark_feed(
        paths,
        feed_path=feed_path,
        aggregate_task_id=str(packet["task_id"]),
        expected_state_sha256=state_sha256,
    )
    second = import_benchmark_feed(
        paths,
        feed_path=feed_path,
        aggregate_task_id=str(packet["task_id"]),
        expected_state_sha256=state_sha256,
    )
    assert first == second
    assert first["completion_unix_ms"] == receipt["completed_unix_ms"] == 1234
    assert feed_path.read_text() == json.dumps(_benchmark_feed(), sort_keys=True, indent=2) + "\n"
    imported = next(
        experiment
        for experiment in ledger.read_ledger(paths.ledger)["experiments"]
        if experiment["id"] == _benchmark_feed()["id"]
    )
    assert imported["started_unix_ms"] == 1234
    assert imported["completed_unix_ms"] == 1234
    assert imported["updated_unix_ms"] == 1234
    assert first["feed_sha256"] in imported["notes"][-1]
    assert receipt["receipt_sha256"] in imported["notes"][-1]


def test_benchmark_feed_tampering_or_unbound_artifact_is_rejected(tmp_path: Path) -> None:
    paths = _initialize(tmp_path)
    _advance(paths)
    _reach_training_and_benchmarking(paths)
    feed_path, packet, receipt = _install_benchmark_feed_fixture(paths)
    state_sha256 = read_state(paths.state)["state_sha256"]
    feed_path.chmod(0o600)
    feed_path.write_text(feed_path.read_text() + " ")
    with pytest.raises(
        CampaignControllerError,
        match=(
            r"not uniquely bound|cannot be resolved|canonical remote-relative path|"
            r"receipt artifact is mutable, oversized, or has unsafe metadata"
        ),
    ):
        import_benchmark_feed(
            paths,
            feed_path=feed_path,
            aggregate_task_id=str(packet["task_id"]),
            expected_state_sha256=state_sha256,
        )

    feed_path.write_text(json.dumps(_benchmark_feed(), sort_keys=True, indent=2) + "\n")
    feed_path.chmod(0o400)
    receipt["artifacts"][0]["path"] = str(paths.root / "different.json")
    receipt["receipt_sha256"] = content_sha256(receipt, hash_field="receipt_sha256")
    imported_receipt_path = paths.receipts / f"{packet['task_id']}.json"
    imported_receipt_path.chmod(0o600)
    imported_receipt_path.write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n"
    )
    with pytest.raises(
        CampaignControllerError,
        match=r"not uniquely bound|cannot be resolved|canonical remote-relative path",
    ):
        import_benchmark_feed(
            paths,
            feed_path=feed_path,
            aggregate_task_id=str(packet["task_id"]),
            expected_state_sha256=state_sha256,
        )


def test_reconciliation_repairs_ledger_but_rejects_queue_identity_drift(
    tmp_path: Path,
) -> None:
    paths = _initialize(tmp_path)
    _advance(paths)
    value = ledger.read_ledger(paths.ledger)
    value["experiments"] = []
    ledger.write_ledger(paths.ledger, value)
    reconciliation = reconcile(paths, now_ms=20)
    assert reconciliation["ledger_experiment_id"] == "r2-map-expert-iteration-v1"
    assert len(ledger.read_ledger(paths.ledger)["experiments"]) == 1

    with queue.locked_queue(paths.queue) as state:
        state["tasks"][-1]["command"] = ["/usr/bin/false"]
    with pytest.raises(CampaignControllerError, match="drifted"):
        reconcile(paths, now_ms=21)


def test_retry_ceiling_is_same_host_and_stops_campaign(tmp_path: Path) -> None:
    paths = _initialize(tmp_path)
    _advance(paths)
    packet = validate_work_packet(json.loads(next(paths.packets.glob("*john1.json")).read_text()))
    for attempt in range(MAX_ATTEMPTS):
        with queue.locked_queue(paths.queue) as state:
            claimed = queue.claim_next(
                state,
                host="john1",
                lease_seconds=30,
                now_ms=30 + attempt * 2,
            )
            assert claimed is not None
            assert claimed["id"] == packet["task_id"]
            assert claimed["compatible_hosts"] == ["john1"]
            queue.finish_task(
                state,
                task_id=claimed["id"],
                host="john1",
                token=claimed["claim"]["token"],
                outcome="failed",
                retry=True,
                now_ms=31 + attempt * 2,
            )
    with pytest.raises(CampaignControllerError, match="retry ceiling"):
        reconcile(paths, now_ms=40)
    assert paths.stop.exists()


def test_three_consecutive_rejections_trigger_durable_stop(tmp_path: Path) -> None:
    paths = _initialize(tmp_path)
    previous = read_state(paths.state)["state_sha256"]
    for sequence, classification in enumerate(("reject", "inconclusive", "reject")):
        following = f"{sequence + 1:064x}"
        controller._append_history(
            paths,
            {
                "schema_version": 1,
                "from_state_sha256": previous,
                "to_state_sha256": following,
                "from_phase": Phase.PAIRED_CANDIDATE_GATE,
                "to_phase": Phase.CANDIDATE_REJECTED,
                "classification": classification,
                "at_unix_ms": sequence,
            },
        )
        previous = following
    result = controller.apply_stop_rules(paths, now_ms=99)
    assert result == {"consecutive_rejections": 3, "stopped": True}
    stop = json.loads(paths.stop.read_text())
    assert stop["reason"] == "three consecutive candidates were rejected or inconclusive"


def test_crash_after_state_cas_recovers_packets_queue_history_and_ledger(
    tmp_path: Path,
) -> None:
    paths = _initialize(tmp_path)

    def crash(stage: str) -> None:
        if stage == "after-state-cas":
            raise RuntimeError("injected crash")

    with pytest.raises(RuntimeError, match="injected"):
        _advance(paths, fault_injector=crash)
    assert read_state(paths.state)["phase"] == Phase.BOOTSTRAP_GENERATING
    assert not tuple(paths.packets.glob("*.json"))

    recovered = recover_current_phase(
        paths,
        commands=_commands(),
        artifact_root="reports/artifacts",
        now_ms=50,
        synthetic=True,
    )
    assert recovered["packet_count"] == 4
    assert len(queue.load_queue(paths.queue)["tasks"]) == 4
    assert len(ledger.read_ledger(paths.ledger)["experiments"]) == 1


def test_full_isolated_dry_run_covers_reject_and_promote_shapes(tmp_path: Path) -> None:
    paths = ControllerPaths.under(tmp_path / "w6")
    report = run_isolated_dry_run(paths, now_ms=100)
    assert report["transition_count"] == 21
    assert report["history_count"] == 21
    assert report["final_phase"] == Phase.INCUMBENT_PROMOTED
    assert report["final_promotion_index"] == 1
    assert report["final_round_index"] == 1
    assert report["queue_task_count"] == report["completed_queue_tasks"] == 30
    assert report["work_receipt_count"] == 30
    assert report["storage_receipt_count"] == 30
    assert report["receipt_count"] == 60
    assert report["stop_file_present"] is False
    history = paths.history.read_text()
    assert '"classification":"reject"' in history
    assert history.count('"classification":"promote"') == 2
    status = build_dashboard_status(
        DashboardStatusInputs(
            campaign_state=read_state(paths.state),
            host_receipts=json.loads((paths.dashboard_inputs / "host-receipts.json").read_text()),
            training_progress=json.loads(
                (paths.dashboard_inputs / "training-progress.json").read_text()
            ),
            benchmark_aggregate=json.loads(
                (paths.dashboard_inputs / "benchmark-aggregate.json").read_text()
            ),
        ),
        updated_unix_ms=1000,
    )
    assert status["phase"] == Phase.INCUMBENT_PROMOTED


def test_controller_does_not_mutate_global_john4_intent(tmp_path: Path) -> None:
    paths = _initialize(tmp_path)
    before = queue.load_queue(paths.queue)["hosts"]["john4"]
    reconcile(paths, now_ms=5)
    after = queue.load_queue(paths.queue)["hosts"]["john4"]
    assert after == before
