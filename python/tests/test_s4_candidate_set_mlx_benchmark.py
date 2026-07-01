from __future__ import annotations

import pytest
from cascadia_mlx.s4_candidate_set_mlx_benchmark import (
    ADR_ID,
    BENCHMARK_SCHEMA_VERSION,
    EXPERIMENT_ID,
    PROTOCOL_ID,
    S4ServingBenchmarkError,
    _canonical_blake3,
    _validate_request,
    _validate_result,
)
from cascadia_mlx.s4_candidate_set_mlx_model import S4_ARMS


def _request() -> dict[str, object]:
    open_data = {
        "cache_id": "1" * 64,
        "s1_cache_id": "2" * 64,
    }
    identity = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "arm": S4_ARMS[0],
        "train_dataset": "/tmp/train",
        "validation_dataset": "/tmp/validation",
        "cache": "/tmp/r3",
        "s1_cache": "/tmp/s1",
        "context_cache": "/tmp/context",
        "context_cache_id": "3" * 64,
        "run_dir": "/tmp/run",
        "checkpoint": {
            "path": "/tmp/run/checkpoints/step-0000000010",
            "manifest_blake3": "4" * 64,
            "model_blake3": "5" * 64,
            "global_step": 10,
        },
        "open_data_verification_id": _canonical_blake3(open_data),
        "open_data_verification": open_data,
        "verification_source": "in-process-full",
        "require_complete_open_corpus": False,
        "candidate_chunk": 256,
        "warmup_iterations": 3,
        "steady_iterations": 3,
        "decision_rows": [0, 1],
    }
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "request_id": _canonical_blake3(identity),
        "scientific_identity": identity,
    }


def test_s4_benchmark_request_is_content_addressed() -> None:
    request = _request()

    assert _validate_request(request)["arm"] == S4_ARMS[0]

    request["scientific_identity"]["steady_iterations"] = 4
    with pytest.raises(S4ServingBenchmarkError, match="malformed"):
        _validate_request(request)


def test_s4_benchmark_result_binds_context_cache() -> None:
    request = _request()
    identity = {
        "request_id": request["request_id"],
        "arm": S4_ARMS[0],
        "checkpoint": request["scientific_identity"]["checkpoint"],
        "context_cache_id": "3" * 64,
        "open_data_verification_id": request["scientific_identity"][
            "open_data_verification_id"
        ],
        "runtime": {"machine": "arm64", "default_device": "gpu"},
        "performance": {"fixed_chunk": {}, "complete_decisions": {}, "memory": {}},
    }
    result = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "result_id": _canonical_blake3(identity),
        "scientific_identity": identity,
    }

    _validate_result(result, request)

    result["scientific_identity"]["context_cache_id"] = "6" * 64
    with pytest.raises(S4ServingBenchmarkError, match="malformed"):
        _validate_result(result, request)
