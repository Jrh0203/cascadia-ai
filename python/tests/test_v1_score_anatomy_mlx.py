from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest
from cascadia_mlx.r2_sparse_mlx_model import (
    R2SparseMlxModelConfig,
    R2SparseValueModel,
)
from cascadia_mlx.v1_score_anatomy_mlx import (
    ARMS,
    COMPONENT_NAMES,
    DENSE_HEX_BUDGET_REFERENCE,
    ROLE_ARMS,
    V1ScoreAnatomyError,
    V1ScoreAnatomyProtocol,
    classify_reports,
    loss_for_arm,
    parameter_layout_blake3,
    parameter_tensor_blake3,
    scalar_total_loss,
)


def _batch() -> SimpleNamespace:
    return SimpleNamespace(
        targets=mx.ones((2, 11), dtype=mx.float32),
    )


class _TotalOnlyModel:
    def __call__(self, batch: object) -> mx.array:
        return mx.ones((2, 11), dtype=mx.float32) * 0.1


def _report(role: str, *, mae: float, correlation: float, log_loss: float) -> dict:
    arm = ROLE_ARMS[role]
    final_hash = "a" * 64 if arm == "scalar-total-control" else "b" * 64
    probe_hash = "c" * 64 if arm == "scalar-total-control" else "d" * 64
    metrics = {
        "samples": 100,
        "component_metrics": [
            {
                "name": name,
                "mae": 1.0,
                "rmse": 1.2,
                "bias": 0.0,
                "correlation": 0.5,
            }
            for name in COMPONENT_NAMES
        ],
        "total": {
            "mae": mae,
            "rmse": mae + 0.5,
            "correlation": correlation,
        },
        "within_round_pairwise": {
            "accuracy": 0.6,
            "log_loss": log_loss,
        },
        "prediction_probe_blake3": probe_hash,
    }
    report = {
        "experiment_id": "v1-score-anatomy-matched-r2-v1",
        "protocol_id": "v1-score-anatomy-matched-r2-v1",
        "role": role,
        "arm": arm,
        "authorization": {
            "authorization_id": "1" * 64,
            "bundle_id": "2" * 64,
        },
        "cache": {
            "cache_id": "3" * 64,
            "corpus_lock_id": "4" * 64,
            "target_blake3": "5" * 64,
            "padded_sparse_objects_per_board": 92,
        },
        "model": {
            "parameter_count": 100,
            "parameter_layout_blake3": "6" * 64,
            "initial_parameter_tensor_blake3": "7" * 64,
            "final_parameter_tensor_blake3": final_hash,
        },
        "metrics": {"validation": metrics},
        "integrity": {"all_metrics_finite": True},
        "scientific_identity": {
            "arm": arm,
            "metrics": metrics,
            "final": final_hash,
            "optimization": {
                "global_step": 3_000,
                "training_examples": 96_000,
                "checkpoint_manifest_blake3": (
                    "8" * 64 if role.endswith("primary") else "9" * 64
                ),
            },
        },
    }
    return report


def test_protocol_uses_exact_sparse_state_below_121_reference() -> None:
    protocol = V1ScoreAnatomyProtocol()
    protocol.validate()
    assert protocol.max_sparse_objects_per_board == 92
    assert protocol.max_sparse_objects_per_board < DENSE_HEX_BUDGET_REFERENCE
    assert len(protocol.component_names) == 11


def test_arms_share_byte_identical_model_graph_and_initialization() -> None:
    fingerprints = []
    for _arm in ARMS:
        mx.random.seed(2026061801)
        model = R2SparseValueModel(
            R2SparseMlxModelConfig(architecture="perceiver-fixed-latents")
        )
        fingerprints.append(
            (
                parameter_layout_blake3(model),
                parameter_tensor_blake3(model),
            )
        )
    assert fingerprints[0] == fingerprints[1]


def test_scalar_and_component_losses_are_finite() -> None:
    scalar = scalar_total_loss(_TotalOnlyModel(), _batch())
    mx.eval(scalar)
    assert float(scalar.item()) >= 0.0
    assert loss_for_arm("scalar-total-control") is scalar_total_loss
    assert callable(loss_for_arm("component-anatomy"))
    with pytest.raises(V1ScoreAnatomyError, match="unknown"):
        loss_for_arm("not-an-arm")


def test_classifier_promotes_only_with_replay_and_all_gates(tmp_path: Path) -> None:
    reports = {
        "scalar-primary": _report(
            "scalar-primary",
            mae=2.5,
            correlation=0.20,
            log_loss=0.70,
        ),
        "scalar-replay": _report(
            "scalar-replay",
            mae=2.5,
            correlation=0.20,
            log_loss=0.70,
        ),
        "anatomy-primary": _report(
            "anatomy-primary",
            mae=2.45,
            correlation=0.25,
            log_loss=0.68,
        ),
        "anatomy-replay": _report(
            "anatomy-replay",
            mae=2.45,
            correlation=0.25,
            log_loss=0.68,
        ),
    }
    paths = {}
    for role, report in reports.items():
        path = tmp_path / f"{role}.json"
        path.write_text(json.dumps(report))
        paths[role] = path
    result = classify_reports(
        scalar_primary=paths["scalar-primary"],
        anatomy_primary=paths["anatomy-primary"],
        scalar_replay=paths["scalar-replay"],
        anatomy_replay=paths["anatomy-replay"],
    )
    assert result["scientific"]["promoted"] is True
    assert result["scientific"]["integrity_pass"] is True
    assert all(
        parity["scientific_identity_exact"]
        for parity in result["scientific"]["replay_parity"].values()
    )

    reports["anatomy-replay"]["model"][
        "final_parameter_tensor_blake3"
    ] = "e" * 64
    paths["anatomy-replay"].write_text(
        json.dumps(reports["anatomy-replay"])
    )
    result = classify_reports(
        scalar_primary=paths["scalar-primary"],
        anatomy_primary=paths["anatomy-primary"],
        scalar_replay=paths["scalar-replay"],
        anatomy_replay=paths["anatomy-replay"],
    )
    assert result["scientific"]["promoted"] is False
    assert result["scientific"]["integrity_pass"] is False
