from __future__ import annotations

import hashlib
import json

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from cascadia_v3_mlx.contracts import BASE_FEATURE_ROWS, V3MlxConfig
from cascadia_v3_mlx.dataset import SparseWidths, synthetic_batch
from cascadia_v3_mlx.export import quantize_model, quantized_forward_from_accumulators
from cascadia_v3_mlx.model import (
    ACCUMULATOR_HEADROOM_COEFFICIENT,
    ACCUMULATOR_HEADROOM_LIMIT,
    CsrBatch,
    CsrRows,
    SparseBatch,
    V3Nnue,
    accumulator_headroom_penalty,
    v3_loss,
)
from cascadia_v3_mlx.train import _canonical, _validate_phase2_authorization
from cascadia_v3_mlx.verify import _accumulate, _numpy_batch


def _config() -> V3MlxConfig:
    return V3MlxConfig()


def _csr_rows(
    indices: mx.array,
    counts: mx.array,
    mask: mx.array,
) -> CsrRows:
    indices_np = np.asarray(indices)
    counts_np = np.asarray(counts)
    mask_np = np.asarray(mask)
    lengths = mask_np.sum(axis=1, dtype=np.int32)
    offsets = np.concatenate((np.zeros(1, dtype=np.int32), np.cumsum(lengths, dtype=np.int32)))
    packed_indices = indices_np[mask_np].astype(np.int32)
    gradient_positions = np.argsort(packed_indices, kind="stable").astype(np.int32)
    gradient_features, starts = np.unique(
        packed_indices[gradient_positions], return_index=True
    )
    return CsrRows(
        offsets=mx.array(offsets),
        indices=mx.array(packed_indices),
        counts=mx.array(counts_np[mask_np].astype(np.float32)),
        row_indices=mx.array(np.repeat(np.arange(len(lengths), dtype=np.int32), lengths)),
        gradient_positions=mx.array(gradient_positions),
        gradient_features=mx.array(gradient_features.astype(np.int32)),
        gradient_offsets=mx.array(
            np.concatenate(
                (starts.astype(np.int32), np.array([len(packed_indices)], dtype=np.int32))
            )
        ),
    )


def _as_csr(batch: SparseBatch) -> CsrBatch:
    return CsrBatch(
        own_base=_csr_rows(
            batch.own_base_indices,
            batch.own_base_counts,
            batch.own_base_mask,
        ),
        field_base=_csr_rows(
            batch.field_base_indices,
            batch.field_base_counts,
            batch.field_base_mask,
        ),
        own_opportunities=_csr_rows(
            batch.own_opportunity_indices,
            batch.own_opportunity_counts,
            batch.own_opportunity_mask,
        ),
        field_opportunities=_csr_rows(
            batch.field_opportunity_indices,
            batch.field_opportunity_counts,
            batch.field_opportunity_mask,
        ),
        own_opportunity_factors=_csr_rows(
            batch.own_opportunity_factor_indices,
            batch.own_opportunity_factor_counts,
            batch.own_opportunity_factor_mask,
        ),
        field_opportunity_factors=_csr_rows(
            batch.field_opportunity_factor_indices,
            batch.field_opportunity_factor_counts,
            batch.field_opportunity_factor_mask,
        ),
        phase_buckets=batch.phase_buckets,
        targets=batch.targets,
        confidence_weights=batch.confidence_weights,
    )


def test_v3_forward_and_qat_loss_are_finite() -> None:
    config = _config()
    mx.random.seed(1)
    model = V3Nnue(config)
    batch = synthetic_batch(config, 2, 2, SparseWidths(3, 4, 5, 6))
    values = model(batch)
    loss = v3_loss(model, batch)
    mx.eval(values, loss)
    assert values.shape == (2,)
    assert np.isfinite(np.asarray(values)).all()
    assert np.isfinite(float(loss.item()))
    assert model.base_embedding.weight.shape == (BASE_FEATURE_ROWS, 1_024)


def test_accumulator_headroom_penalty_preserves_safe_values_and_punishes_excess() -> None:
    safe = mx.array(
        [[-ACCUMULATOR_HEADROOM_LIMIT, ACCUMULATOR_HEADROOM_LIMIT]],
        dtype=mx.float32,
    )
    unsafe = mx.array(
        [[-(ACCUMULATOR_HEADROOM_LIMIT + 2.0), ACCUMULATOR_HEADROOM_LIMIT + 4.0]],
        dtype=mx.float32,
    )
    safe_penalty = accumulator_headroom_penalty(safe, safe)
    unsafe_penalty = accumulator_headroom_penalty(unsafe, unsafe)
    mx.eval(safe_penalty, unsafe_penalty)
    assert float(safe_penalty.item()) == 0.0
    np.testing.assert_allclose(
        float(unsafe_penalty.item()),
        ACCUMULATOR_HEADROOM_COEFFICIENT * 16.0,
        rtol=1e-6,
    )


def test_numpy_integer_reference_has_exact_output_shape() -> None:
    config = _config()
    mx.random.seed(3)
    model = V3Nnue(config)
    batch = synthetic_batch(config, 3, 4, SparseWidths(3, 4, 5, 6))
    parameters = quantize_model(model)
    values = _numpy_batch(batch)
    own, field, direct = _accumulate(parameters, values)
    output = quantized_forward_from_accumulators(
        parameters,
        own,
        field,
        values["phase_buckets"],
        direct,
    )
    assert output.shape == (3,)
    assert output.dtype == np.int32


def test_native_csr_forward_and_custom_gradient_match_contract() -> None:
    config = _config()
    mx.random.seed(5)
    model = V3Nnue(config)
    padded = synthetic_batch(config, 2, 6, SparseWidths(3, 4, 5, 6))
    csr = _as_csr(padded)
    reference = model(padded)
    optimized = model.call_csr(csr)
    loss, gradients = nn.value_and_grad(model, v3_loss)(model, csr)
    mx.eval(reference, optimized, loss, gradients)
    np.testing.assert_allclose(np.asarray(optimized), np.asarray(reference), rtol=1e-5, atol=1e-5)
    assert np.isfinite(float(loss.item()))


def test_scientific_training_requires_checksum_bound_exact_phase(tmp_path) -> None:
    readiness = tmp_path / "readiness.json"
    readiness_value = {
        "schema_id": "cascadia-v3-part1-readiness-v1",
        "campaign_id": "cascadia-v3-radius7-stockfish-nnue-v1",
        "status": "green",
    }
    readiness_hash = hashlib.sha256(_canonical(readiness_value)).hexdigest()
    readiness_value["readiness_sha256"] = readiness_hash
    readiness.write_text(json.dumps(readiness_value))
    state = {
        "schema_id": "cascadia-v3-campaign-state-v1",
        "campaign_id": "cascadia-v3-radius7-stockfish-nnue-v1",
        "part": 2,
        "phase": "cycle-03-training",
        "phase2_authorized": True,
        "approved_readiness_sha256": readiness_hash,
        "readiness_path": str(readiness),
    }
    state["state_sha256"] = hashlib.sha256(_canonical(state)).hexdigest()
    path = tmp_path / "state.json"
    path.write_text(json.dumps(state))
    assert _validate_phase2_authorization(path, 3)["phase"] == "cycle-03-training"
    with np.testing.assert_raises_regex(ValueError, "exact authorized campaign phase"):
        _validate_phase2_authorization(path, 4)
