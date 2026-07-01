from __future__ import annotations

from pathlib import Path

import relational_substrate_mlx_control_replay as replay_tool
from cascadia_mlx.relational_substrate_mlx_cache import (
    open_data_verification_id,
)


def _control_report(checkpoint: Path) -> dict:
    report = {
        "schema_version": 1,
        "experiment_id": replay_tool.EXPERIMENT_ID,
        "protocol_id": replay_tool.PROTOCOL_ID,
        "adr": replay_tool.ADR_ID,
        "mode": "production",
        "arm": replay_tool.CONTROL_ARM,
        "host": "john1",
        "optimization": {"global_step": 3000},
        "r3_cache_id": "a" * 64,
        "relational_cache_id": "b" * 64,
        "s1_cache_id": "c" * 64,
        "checkpoint": {
            "manifest_blake3": replay_tool._checksum(
                checkpoint / "checkpoint.json"
            ),
            "model_blake3": replay_tool._checksum(
                checkpoint / "model.safetensors"
            ),
        },
    }
    report["scientific_identity"] = {
        key: value
        for key, value in report.items()
        if key != "schema_version"
    }
    report["report_id"] = replay_tool._canonical_blake3(
        report["scientific_identity"]
    )
    return report


def _authorization(r6_binary: Path) -> dict:
    open_data = {"open": True, "relational_cache_id": "b" * 64}
    identity = {
        "open_data_verification": open_data,
        "open_data_verification_id": open_data_verification_id(open_data),
        "r6_binary_blake3": replay_tool._checksum(r6_binary),
    }
    return {
        "schema_version": 1,
        "experiment_id": replay_tool.EXPERIMENT_ID,
        "protocol_id": replay_tool.PROTOCOL_ID,
        "adr": replay_tool.ADR_ID,
        "approved": True,
        "authorization_id": replay_tool._canonical_blake3(identity),
        "identity": identity,
    }


def _performance(
    *,
    checkpoint_model_blake3: str,
    open_data_verification_id_value: str,
) -> dict:
    return {
        "measurement": {
            "isolated_process": True,
            "checkpoint_model_blake3": checkpoint_model_blake3,
            "open_data_verification_id": (
                open_data_verification_id_value
            ),
            "verification_source": "cluster-preflight",
            "worker_runtime": {"host": "john2"},
            "request_id": "d" * 64,
            "result_id": "e" * 64,
        },
        "fixed_chunk": {"action_scores_per_second": 30_000.0},
        "combined_with_r6": {
            "groups": 240,
            "actions": 860_203,
            "r6_exact_parity_pass": True,
        },
        "r6_apply_undo": {
            "exact_parity_pass": True,
            "apply_failures": 0,
            "undo_failures": 0,
        },
        "memory": {},
    }


def test_build_replay_report_binds_checkpoint_r6_and_host(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "checkpoint.json").write_text("{}")
    (checkpoint / "model.safetensors").write_bytes(b"model")
    r6_binary = tmp_path / "r6"
    r6_binary.write_bytes(b"r6")
    control = _control_report(checkpoint)
    authorization = _authorization(r6_binary)

    replay = replay_tool.build_replay_report(
        control_report=control,
        authorization=authorization,
        performance=_performance(
            checkpoint_model_blake3=control["checkpoint"][
                "model_blake3"
            ],
            open_data_verification_id_value=authorization["identity"][
                "open_data_verification_id"
            ],
        ),
        checkpoint=checkpoint,
        r6_binary=r6_binary,
        treatment_arm="q1-r5-quotient-local",
        replay_host="john2",
    )

    assert replay["host"] == "john2"
    assert replay["control_report_id"] == control["report_id"]
    assert (
        replay["scientific_identity"]["assertions"][
            "r6_apply_undo_exact"
        ]
        is True
    )
