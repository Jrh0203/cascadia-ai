from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import pytest
from cascadia_mlx.opportunity_cross_attention_mlx_model import (
    OpportunityCrossAttentionRanker,
)
from opportunity_cross_attention_mlx_smoke_compare import (
    EXPECTED_HOSTS,
    PASS,
    SMOKE_ARM,
    SMOKE_STEPS,
    SmokeParityError,
    _canonical_blake3,
    _checksum,
    _parse_host_paths,
    _report_scientific_identity,
    compare_smoke,
)


def _report(host: str, checkpoint: Path) -> dict:
    warm_start = {
        "warm_start_id": "a" * 64,
        "base_parameter_tensor_blake3": "b" * 64,
    }
    report = {
        "schema_version": 1,
        "experiment_id": "opportunity-cross-attention-mlx-tournament-v1",
        "protocol_id": "exact-r2-opportunity-query-factorial-v1",
        "adr": "0166",
        "mode": "bounded-smoke",
        "arm": SMOKE_ARM,
        "host": host,
        "data_arm": "c0-exact-r2",
        "r3_cache_id": "c" * 64,
        "relational_cache_id": "d" * 64,
        "s1_cache_id": "e" * 64,
        "r6_binary": {"path": "/tmp/r6", "blake3": "f" * 64},
        "protocol": {"protocol_id": "exact-r2-opportunity-query-factorial-v1"},
        "warm_start": warm_start,
        "zero_init_prediction_parity": {
            "exact_array_equal": True,
            "scores_blake3": "1" * 64,
        },
        "model": {
            "total_parameter_count": 735_426,
            "initial_all_parameter_tensor_blake3": "2" * 64,
            "initial_adapter_parameter_tensor_blake3": "3" * 64,
            "final_base_parameter_tensor_blake3": "b" * 64,
        },
        "optimization": {
            "global_step": SMOKE_STEPS,
            "loss_trace": [
                {
                    "step": step,
                    "batch_blake3": f"{step:064x}",
                    "candidates": 128,
                    "loss": 10.0 - step,
                }
                for step in range(1, SMOKE_STEPS + 1)
            ],
        },
        "checkpoint": {
            "path": str(checkpoint),
            "model_blake3": _checksum(checkpoint),
        },
        "metrics": {
            "prediction_panel": {
                "action_hashes": [f"{index:064x}" for index in range(4)],
                "scores": [4.0, 3.0, 2.0, 1.0],
                "standard_errors": [1.0, 1.1, 1.2, 1.3],
            }
        },
        "paired_panel": None,
        "paired_panel_id": None,
        "performance": {
            "combined_with_r6": {"r6_exact_parity_pass": True}
        },
        "runtime": {"host": host},
        "source": {"v2_source_blake3": "4" * 64},
        "controls": None,
        "information_boundary": {"sealed_test_opened": False},
        "claims": {
            "bounded_smoke_complete": True,
            "offline_comparison_complete": False,
            "base_parameters_frozen": True,
        },
    }
    report["scientific_identity"] = _report_scientific_identity(report)
    report["report_id"] = _canonical_blake3(report["scientific_identity"])
    return report


def test_four_host_smoke_is_order_invariant(tmp_path: Path) -> None:
    mx.random.seed(83)
    model = OpportunityCrossAttentionRanker()
    report_paths = []
    checkpoint_paths = {}
    for host in EXPECTED_HOSTS:
        checkpoint = tmp_path / f"{host}.safetensors"
        model.save_weights(str(checkpoint))
        report_path = tmp_path / f"{host}.json"
        report_path.write_text(
            json.dumps(_report(host, checkpoint), indent=2, sort_keys=True)
            + "\n"
        )
        report_paths.append(report_path)
        checkpoint_paths[host] = checkpoint

    forward = compare_smoke(report_paths, checkpoint_paths)
    reverse = compare_smoke(list(reversed(report_paths)), checkpoint_paths)

    assert forward["classification"] == PASS
    assert forward == reverse
    assert forward["scientific_identity"]["checks"] == {
        host: True for host in EXPECTED_HOSTS
    }


def test_report_identity_includes_optional_paired_panel_fields(
    tmp_path: Path,
) -> None:
    mx.random.seed(84)
    model = OpportunityCrossAttentionRanker()
    checkpoint = tmp_path / "john1.safetensors"
    model.save_weights(str(checkpoint))
    report = _report("john1", checkpoint)

    assert "paired_panel" in report["scientific_identity"]
    assert "paired_panel_id" in report["scientific_identity"]
    assert report["scientific_identity"]["paired_panel"] is None
    assert report["scientific_identity"]["paired_panel_id"] is None


def test_checkpoint_cli_requires_every_host() -> None:
    with pytest.raises(SmokeParityError, match="omit"):
        _parse_host_paths(["john1=/tmp/one"])
    with pytest.raises(SmokeParityError, match="unique"):
        _parse_host_paths(
            [
                "john1=/tmp/one",
                "john1=/tmp/two",
                "john2=/tmp/two",
                "john3=/tmp/three",
                "john4=/tmp/four",
            ]
        )
