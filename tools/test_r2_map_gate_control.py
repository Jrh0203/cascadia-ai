from __future__ import annotations

import json
from pathlib import Path

import pytest
import r2_map_gate_control as subject


def _manifest(path: Path) -> Path:
    identity = {
        "maximum_width_panel_sha256": "01" * 32,
        "replay_pinecone_panel_sha256": "02" * 32,
        "source_bundle_sha256": "03" * 32,
        "serving_protocol_schema_sha256": "04" * 32,
        "market_action_schema_blake3": "05" * 32,
        "request_schema_blake3": "06" * 32,
        "response_schema_blake3": "07" * 32,
        "protocol_fixture_canonical_blake3": "08" * 32,
        "protocol_fixture_file_blake3": "09" * 32,
        "model_schema_sha256": "0a" * 32,
        "open_reference_seed_domain_id": "r2-map-open-reference-performance-100-v1",
    }
    path.write_text(
        json.dumps(
            {
                "schema_id": "cascadia.r2-map.reference-panel-manifest.v1.1",
                "campaign_id": subject.CAMPAIGN_ID,
                "manifest_sha256": "0b" * 32,
                "implementation_identity": identity,
            }
        )
    )
    return path


def test_registry_is_private_committed_and_repeat_verifiable(tmp_path: Path) -> None:
    counter = 0

    def entropy(size: int) -> bytes:
        nonlocal counter
        counter += 1
        return counter.to_bytes(size, "big")

    private = tmp_path / "private"
    public = tmp_path / "registry.json"
    registry = subject.create_registry(private, public, entropy=entropy)
    assert subject.verify_registry(private, public) == registry
    assert "seeds" not in public.read_text()
    assert len(registry["domains"]) == 3
    assert len(json.loads(next(private.glob("*20-v2.json")).read_text())["seeds"]) == 20


def test_prepare_bootstrap_gate_assigns_cross_arch_focal_with_all_greedy_field(
    tmp_path: Path,
) -> None:
    counter = 0

    def entropy(size: int) -> bytes:
        nonlocal counter
        counter += 1
        return counter.to_bytes(size, "big")

    private = tmp_path / "private"
    subject.create_registry(private, tmp_path / "registry.json", entropy=entropy)
    freeze = tmp_path / "freeze-receipt.json"
    freeze.write_text(json.dumps({"checkpoint_id": "checkpoint-75"}))
    weights = tmp_path / "exact.bin"
    weights.write_bytes(b"qualified-exact")
    image = "registry/r2@sha256:" + "a" * 64
    contract, field = subject.prepare_bootstrap_gate(
        stage="smoke",
        checkpoint_id="checkpoint-75",
        private_domain=private / "r2-map-strength-blinded-smoke-20-v2.json",
        output_directory=tmp_path / "gate",
        reference_manifest=_manifest(tmp_path / "manifest.json"),
        image_digest=image,
        candidate_freeze_receipt=freeze,
        exact_weights=weights,
    )
    assert contract["stage"] == "strength-blinded-smoke"
    assert contract["candidate_checkpoint_id"] == "checkpoint-75"
    assert contract["control_checkpoint_id"] == subject.QUALIFIED_EXACT_NNUE_CHECKPOINT_ID
    assert contract["inference_settings_id"] == subject.CROSS_ARCH_INFERENCE_SETTINGS_ID
    assert contract["execution_binding"] == {
        "image_digest": image,
        "candidate_freeze_receipt_sha256": subject._sha256_file(freeze),
        "exact_weights_sha256": subject._sha256_file(weights),
        "opponent_field_sha256": subject._sha256(subject._encoded_json(field)),
    }
    assert len(field["assignments"]) == 20
    for index, assignment in enumerate(field["assignments"]):
        assert "executor_shard" not in assignment
        assert assignment["focal_seat"] == index % 4
        assert len(assignment["opponents"]) == 3
        assert {value["checkpoint_id"] for value in assignment["opponents"]} == {"greedy-v1"}


def test_tampered_private_domain_is_rejected(tmp_path: Path) -> None:
    private = tmp_path / "private"
    public = tmp_path / "registry.json"
    subject.create_registry(private, public)
    path = private / "r2-map-strength-blinded-smoke-20-v2.json"
    value = json.loads(path.read_text())
    value["seeds"][0] = "ff" * 32
    path.write_text(json.dumps(value))
    with pytest.raises(subject.GateControlError, match="commitment differs"):
        subject.verify_registry(private, public)


def test_development_materialization_requires_passing_blinded_smoke(tmp_path: Path) -> None:
    private = tmp_path / "private"
    subject.create_registry(private, tmp_path / "registry.json")
    checkpoint = "checkpoint-7235"
    image = "registry/r2@sha256:" + "a" * 64
    freeze = tmp_path / "freeze-receipt.json"
    freeze.write_text(json.dumps({"checkpoint_id": checkpoint}))
    weights = tmp_path / "exact.bin"
    weights.write_bytes(b"qualified-exact")
    reference = _manifest(tmp_path / "manifest.json")
    smoke = tmp_path / "smoke"
    subject.prepare_bootstrap_gate(
        stage="smoke",
        checkpoint_id=checkpoint,
        private_domain=private / "r2-map-strength-blinded-smoke-20-v2.json",
        output_directory=smoke,
        reference_manifest=reference,
        image_digest=image,
        candidate_freeze_receipt=freeze,
        exact_weights=weights,
    )
    work_items = [
        {
            "work_item_id": f"pair-{index:04}",
            "pairs": 1,
            "physical_games": 2,
            "all_clean_shutdowns": True,
            "all_pinecone_conservation_checks_passed": True,
            "peak_rss_bytes": 1024,
            "maximum_swap_delta_bytes": 0,
        }
        for index in range(20)
    ]
    report = {
        "schema_id": subject.FOCAL_REPORT_SCHEMA,
        "contract_sha256": subject._sha256_file(smoke / "contract.json"),
        "work_items": work_items,
        "result": {
            "kind": "strength-blinded-smoke",
            "statistics": {
                "schema_version": 1,
                "protocol_id": "r2-map-focal-paired-v1",
                "stage": "strength-blinded-smoke",
                "strength_outputs_blinded": True,
                "pairs": 20,
                "physical_games": 40,
                "wall_seconds": 10.0,
                "games_per_second": 4.0,
                "peak_rss_bytes": 1024,
                "maximum_swap_delta_bytes": 0,
                "all_clean_shutdowns": True,
                "all_pinecone_conservation_checks_passed": True,
            },
        },
    }
    (smoke / "reports").mkdir()
    (smoke / "reports/focal-benchmark.json").write_text(json.dumps(report))
    scheduler = {
        "schema_id": subject.SCHEDULER_REPORT_SCHEMA,
        "stage": "smoke",
        "image_digest": image,
        "work_items": [{"item_id": f"pair-{index:04}"} for index in range(20)],
        "retry_count": 0,
    }
    scheduler["report_sha256"] = subject._digest(scheduler)
    (smoke / "reports/scheduler-provenance.json").write_text(json.dumps(scheduler))
    with pytest.raises(subject.GateControlError, match="requires admitted"):
        subject.prepare_bootstrap_gate(
            stage="development",
            checkpoint_id=checkpoint,
            private_domain=private / "r2-map-fixed-development-gate-250-v2.json",
            output_directory=tmp_path / "development-without-admission",
            reference_manifest=reference,
            image_digest=image,
            candidate_freeze_receipt=freeze,
            exact_weights=weights,
        )
    contract, _field = subject.prepare_bootstrap_gate(
        stage="development",
        checkpoint_id=checkpoint,
        private_domain=private / "r2-map-fixed-development-gate-250-v2.json",
        output_directory=tmp_path / "development",
        reference_manifest=reference,
        image_digest=image,
        candidate_freeze_receipt=freeze,
        exact_weights=weights,
        smoke_campaign_directory=smoke,
    )
    admission = tmp_path / "development/smoke-admission-receipt.json"
    assert admission.is_file()
    assert contract["execution_binding"]["smoke_admission_receipt_sha256"] == (
        subject._sha256_file(admission)
    )
    report["result"]["statistics"]["maximum_swap_delta_bytes"] = 1
    (smoke / "reports/focal-benchmark.json").write_text(json.dumps(report))
    with pytest.raises(subject.GateControlError, match="did not pass admission"):
        subject.prepare_bootstrap_gate(
            stage="development",
            checkpoint_id=checkpoint,
            private_domain=private / "r2-map-fixed-development-gate-250-v2.json",
            output_directory=tmp_path / "development-invalid-smoke",
            reference_manifest=reference,
            image_digest=image,
            candidate_freeze_receipt=freeze,
            exact_weights=weights,
            smoke_campaign_directory=smoke,
        )
