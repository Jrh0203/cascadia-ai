from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from cascadia_mlx.r2_map_train import R2MapTrainerConfig


def _binding() -> dict:
    return {
        "report_relative": "reports/w2-w3/sweep/packing-sweep.json",
        "report_sha256": "1" * 64,
        "report_object_sha256": "2" * 64,
        "report_object_token_sha256": "3" * 64,
        "publication_receipt_relative": "control/receipts/req-report.json",
        "publication_receipt_object_sha256": "4" * 64,
        "publication_receipt_sha256": "5" * 64,
        "local_write_attestation_relative": ("reports/w2-w3/sweep/local-write-attestation.json"),
        "local_write_attestation_object_sha256": "6" * 64,
        "local_write_attestation_object_token_sha256": "7" * 64,
        "local_write_attestation_sha256": "7" * 64,
        "local_write_attestation_publication_receipt_relative": (
            "control/receipts/req-john1-attestation-77777777777777777777777777777777.json"
        ),
        "local_write_attestation_publication_receipt_object_sha256": "8" * 64,
        "local_write_attestation_publication_receipt_object_token_sha256": "9" * 64,
        "local_write_attestation_publication_receipt_sha256": "0" * 64,
        "bootstrap_phase_barrier_identity_sha256": "a" * 64,
        "bootstrap_phase_barrier_sha256": "b" * 64,
        "bootstrap_phase_barrier_publication_receipt_sha256": "c" * 64,
        "bootstrap_controller_state_sha256": "d" * 64,
        "bootstrap_generation_manifest_payload_sha256": "e" * 64,
        "bootstrap_generation_manifest_identity_sha256": "f" * 64,
        "bootstrap_generation_manifest_publication_receipt_sha256": "0" * 64,
        "selected_group_batch_size": 16,
        "maximum_candidates_per_batch": 16_384,
        "schedule_steps": 120,
        "epochs": 12,
    }


def _config() -> R2MapTrainerConfig:
    return R2MapTrainerConfig(
        run_dir=Path("/private/var/empty/r2-map-test"),
        run_id="packing-binding-test",
        branch_id="main",
        source_blake3="8" * 64,
        dataset_blake3="9" * 64,
        adapter_protocol_id="r2-map-compact-on-demand-v1",
        group_batch_size=16,
        maximum_candidates_per_batch=16_384,
        packing_report_binding=_binding(),
        schedule_steps=120,
        warmup_steps=10,
    )


def test_packing_report_is_part_of_trainer_checkpoint_identity() -> None:
    config = _config()
    config.validate()
    assert config.identity_dict()["packing_report_binding"] == _binding()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("selected_group_batch_size", 32),
        ("maximum_candidates_per_batch", 8_192),
        ("schedule_steps", 119),
        ("epochs", 11),
        ("report_sha256", "not-a-digest"),
        (
            "local_write_attestation_publication_receipt_relative",
            "control/receipts/req-caller-chosen.json",
        ),
    ),
)
def test_packing_report_binding_rejects_manual_training_drift(field: str, value: object) -> None:
    config = _config()
    binding = _binding()
    binding[field] = value
    with pytest.raises(ValueError, match="packing"):
        replace(config, packing_report_binding=binding).validate()
