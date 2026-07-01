from __future__ import annotations

import json

import mlx.core as mx
import pytest
from cascadia_mlx.opportunity_cross_attention_mlx_model import ARMS
from cascadia_mlx.opportunity_cross_attention_mlx_protocol import (
    ARM_HOSTS,
    OpportunityCrossAttentionTrainingProtocol,
    normalize_host,
)
from cascadia_mlx.opportunity_cross_attention_mlx_train import (
    _load_batch_trace,
    cross_arm_initialization,
)
from cascadia_mlx.relational_substrate_mlx_model import (
    RelationalSubstrateRanker,
)


def test_protocol_binds_one_arm_to_each_cluster_host() -> None:
    assert OpportunityCrossAttentionTrainingProtocol().to_dict()[
        "trainable_scope"
    ] == "opportunity-adapters-only"
    assert tuple(ARM_HOSTS) == ARMS
    assert set(ARM_HOSTS.values()) == {"john1", "john2", "john3", "john4"}
    assert normalize_host("Johns-Mac-mini.local") == "john1"


def test_cross_arm_initialization_is_identical_after_warm_start(
    tmp_path,
) -> None:
    mx.random.seed(73)
    base = RelationalSubstrateRanker()
    checkpoint = tmp_path / "c0-checkpoint"
    checkpoint.mkdir()
    base.save_weights(str(checkpoint / "model.safetensors"))

    proof = cross_arm_initialization(
        base,
        warm_start_checkpoint=checkpoint,
    )

    assert proof["total_parameter_count"] == 735_426
    assert proof["adapter_parameter_count"] == 104_256
    assert len(set(proof["cross_arm_total_parameter_counts"].values())) == 1
    assert (
        len(
            set(
                proof[
                    "cross_arm_initial_adapter_parameter_tensor_blake3"
                ].values()
            )
        )
        == 1
    )
    assert set(proof["cross_arm_base_parameter_tensor_blake3"].values()) == {
        proof["base_parameter_tensor_blake3"]
    }


def test_resume_trace_requires_exact_contiguous_steps(tmp_path) -> None:
    trace = tmp_path / "batch-trace.jsonl"
    events = [
        {
            "schema_version": 1,
            "step": step,
            "batch_blake3": f"{step:064x}",
            "loss": 1.0 / step,
            "candidates": 128,
            "elapsed_seconds": 0.25,
        }
        for step in range(1, 4)
    ]
    trace.write_text(
        "".join(json.dumps(event) + "\n" for event in events)
    )
    assert _load_batch_trace(trace, expected_steps=3) == events

    events[-1]["step"] = 4
    trace.write_text(
        "".join(json.dumps(event) + "\n" for event in events)
    )
    with pytest.raises(ValueError, match="resume checkpoint"):
        _load_batch_trace(trace, expected_steps=3)
