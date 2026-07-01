"""Export and cross-check float, MLX-integer, Rust-scalar, and Rust-NEON V3 inference."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np
from cascadia_mlx.checkpoint import load_latest_checkpoint_with_factory

from .checkpoint_eval import _optimizer_settings
from .contracts import GLOBAL_BASE, OUTPUT_SCALE, V3MlxConfig
from .dataset import SparseWidths, synthetic_batch
from .export import (
    QuantizedParameters,
    coalesce_training_factors_in_place,
    export_quantized_bundle,
    quantize_model,
    quantized_forward_from_accumulators,
    quantized_forward_mlx_from_accumulators,
)
from .model import SparseBatch, V3Nnue


def _active(
    indices: np.ndarray, counts: np.ndarray, mask: np.ndarray, row: int
) -> list[dict[str, int]]:
    return [
        {"index": int(index), "count": int(count)}
        for index, count, enabled in zip(indices[row], counts[row], mask[row], strict=True)
        if enabled
    ]


def _numpy_batch(batch: SparseBatch) -> dict[str, np.ndarray]:
    return {name: np.asarray(getattr(batch, name)) for name in SparseBatch.__dataclass_fields__}


def _factor_arrays(
    indices: mx.array,
    counts: mx.array,
    mask: mx.array,
    offsets: list[int],
    factor_indices: list[int],
) -> tuple[mx.array, mx.array, mx.array]:
    dense_indices = np.asarray(indices)
    dense_counts = np.asarray(counts)
    dense_mask = np.asarray(mask)
    rows: list[list[tuple[int, int]]] = []
    for row_indices, row_counts, row_mask in zip(
        dense_indices, dense_counts, dense_mask, strict=True
    ):
        merged: dict[int, int] = {}
        for feature, count, enabled in zip(row_indices, row_counts, row_mask, strict=True):
            if not enabled:
                continue
            start, end = offsets[int(feature) : int(feature) + 2]
            for factor in factor_indices[start:end]:
                merged[factor] = merged.get(factor, 0) + int(count)
        rows.append(sorted(merged.items()))
    width = max(len(row) for row in rows)
    result_indices = np.zeros((len(rows), width), dtype=np.int32)
    result_counts = np.zeros((len(rows), width), dtype=np.float32)
    result_mask = np.zeros((len(rows), width), dtype=np.bool_)
    for row_index, row in enumerate(rows):
        for column, (factor, count) in enumerate(row):
            result_indices[row_index, column] = factor
            result_counts[row_index, column] = count
            result_mask[row_index, column] = True
    return mx.array(result_indices), mx.array(result_counts), mx.array(result_mask)


def _factorize_batch(batch: SparseBatch, feature: dict[str, object]) -> SparseBatch:
    offsets = [int(value) for value in feature["opportunity_training_factor_offsets"]]
    factor_indices = [int(value) for value in feature["opportunity_training_factor_indices"]]
    own = _factor_arrays(
        batch.own_opportunity_indices,
        batch.own_opportunity_counts,
        batch.own_opportunity_mask,
        offsets,
        factor_indices,
    )
    field = _factor_arrays(
        batch.field_opportunity_indices,
        batch.field_opportunity_counts,
        batch.field_opportunity_mask,
        offsets,
        factor_indices,
    )
    return replace(
        batch,
        own_opportunity_factor_indices=own[0],
        own_opportunity_factor_counts=own[1],
        own_opportunity_factor_mask=own[2],
        field_opportunity_factor_indices=field[0],
        field_opportunity_factor_counts=field[1],
        field_opportunity_factor_mask=field[2],
    )


def _accumulate(
    parameters: QuantizedParameters,
    values: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = values["phase_buckets"].shape[0]
    own = np.repeat(parameters.transformer_bias[None, :].astype(np.int64), rows, axis=0)
    field = own.copy()
    direct = np.zeros((rows, 8), dtype=np.int64)
    for row in range(rows):
        for prefix, output in (("own", own), ("field", field)):
            for feature, count, enabled in zip(
                values[f"{prefix}_base_indices"][row],
                values[f"{prefix}_base_counts"][row],
                values[f"{prefix}_base_mask"][row],
                strict=True,
            ):
                if enabled:
                    # Promote before multiplying. NumPy otherwise keeps the
                    # int16 row dtype and silently wraps large feature counts.
                    output[row] += parameters.base_transformer[int(feature)].astype(
                        np.int64
                    ) * int(count)
                    if prefix == "own" and int(feature) < GLOBAL_BASE:
                        direct[row] += parameters.direct_potential[int(feature)].astype(
                            np.int64
                        ) * int(count)
            for feature, count, enabled in zip(
                values[f"{prefix}_opportunity_indices"][row],
                values[f"{prefix}_opportunity_counts"][row],
                values[f"{prefix}_opportunity_mask"][row],
                strict=True,
            ):
                if enabled:
                    # The int8 opportunity rows need the same explicit
                    # promotion; wrapping here differs by exact multiples of
                    # 256 and can masquerade as a rounding disagreement.
                    output[row] += parameters.opportunity_transformer[
                        int(feature)
                    ].astype(np.int64) * int(count)
    if (
        np.max(np.abs(own)) > np.iinfo(np.int16).max
        or np.max(np.abs(field)) > np.iinfo(np.int16).max
    ):
        raise ValueError("verification fixture overflowed int16 accumulators")
    return own.astype(np.int16), field.astype(np.int16), direct.astype(np.int32)


def _fixture(
    values: dict[str, np.ndarray],
    expected: np.ndarray,
) -> dict[str, object]:
    rows = []
    for row, raw in enumerate(expected):
        rows.append(
            {
                "features": {
                    "own_base": _active(
                        values["own_base_indices"],
                        values["own_base_counts"],
                        values["own_base_mask"],
                        row,
                    ),
                    "field_base": _active(
                        values["field_base_indices"],
                        values["field_base_counts"],
                        values["field_base_mask"],
                        row,
                    ),
                    "own_opportunities": _active(
                        values["own_opportunity_indices"],
                        values["own_opportunity_counts"],
                        values["own_opportunity_mask"],
                        row,
                    ),
                    "field_opportunities": _active(
                        values["field_opportunity_indices"],
                        values["field_opportunity_counts"],
                        values["field_opportunity_mask"],
                        row,
                    ),
                    "overflow_entities": [[], [], [], []],
                    "phase_bucket": int(values["phase_buckets"][row]),
                },
                "expected_raw_output_units": int(raw),
            }
        )
    return {"schema_id": "cascadia-v3-quantized-parity-fixture-v1", "rows": rows}


def verify(
    feature_manifest: Path,
    output_dir: Path,
    rust_binary: Path,
    *,
    groups: int,
    candidates_per_group: int,
    checkpoint_run_dir: Path | None = None,
    verification_batch_rows: int = 256,
) -> dict[str, object]:
    feature = json.loads(feature_manifest.read_text())
    config = V3MlxConfig(
        opportunity_feature_rows=feature["opportunity_feature_rows"],
        opportunity_training_factor_rows=feature["opportunity_training_factor_rows"],
    )
    if candidates_per_group < 32:
        raise ValueError("quantized ranking groups require at least 32 candidates")
    if verification_batch_rows <= 0:
        raise ValueError("verification batch rows must be positive")
    if checkpoint_run_dir is None:
        mx.random.seed(75_001)
        model = V3Nnue(config)
        checkpoint_id = "untrained-parity-origin"
        training_run_manifest_blake3 = None
    else:
        run_manifest = json.loads((checkpoint_run_dir / "run-manifest.json").read_text())
        learning_rate, weight_decay = _optimizer_settings(run_manifest)
        model, _, _, checkpoint = load_latest_checkpoint_with_factory(
            checkpoint_run_dir,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            model_factory=lambda values: V3Nnue(V3MlxConfig.from_dict(values)),
        )
        if model.config != config:
            raise ValueError("checkpoint model does not match the feature manifest")
        checkpoint_id = checkpoint.name
        training_run_manifest_blake3 = run_manifest["canonical_blake3"]
    training_origin = (
        str(run_manifest.get("origin") or run_manifest.get("training_origin"))
        if checkpoint_run_dir is not None
        else "engineering-smoke"
    )
    coalesce_training_factors_in_place(
        model,
        feature["opportunity_training_factor_offsets"],
        feature["opportunity_training_factor_indices"],
    )
    parameters = quantize_model(
        model,
        feature["opportunity_training_factor_offsets"],
        feature["opportunity_training_factor_indices"],
    )
    float_chunks = []
    quantized_chunks = []
    fixture_rows: list[dict[str, object]] = []
    total_rows = groups * candidates_per_group
    for start in range(0, total_rows, verification_batch_rows):
        rows = min(verification_batch_rows, total_rows - start)
        batch = synthetic_batch(
            config,
            rows,
            75_002 + start,
            SparseWidths(12, 20, 16, 24),
        )
        batch = _factorize_batch(batch, feature)
        float_values = model(batch)
        mx.eval(float_values)
        float_chunks.append(np.asarray(float_values, dtype=np.float32))
        values = _numpy_batch(batch)
        own, field, direct = _accumulate(parameters, values)
        quantized_raw = quantized_forward_from_accumulators(
            parameters,
            own,
            field,
            values["phase_buckets"],
            direct,
        )
        mlx_quantized_raw = quantized_forward_mlx_from_accumulators(
            parameters,
            own,
            field,
            values["phase_buckets"],
            direct,
        )
        if not np.array_equal(mlx_quantized_raw, quantized_raw):
            mismatch = int(np.flatnonzero(mlx_quantized_raw != quantized_raw)[0])
            raise ValueError(
                "MLX integer kernel differs from the NumPy oracle at row "
                f"{start + mismatch}: {mlx_quantized_raw[mismatch]} "
                f"!= {quantized_raw[mismatch]}"
            )
        quantized_chunks.append(quantized_raw)
        fixture_rows.extend(_fixture(values, quantized_raw)["rows"])
        del batch, float_values
        mx.clear_cache()
    float_values_np = np.concatenate(float_chunks)
    quantized_raw = np.concatenate(quantized_chunks)
    quantized_scores = quantized_raw.astype(np.float32) / OUTPUT_SCALE
    float_groups = float_values_np.reshape(groups, candidates_per_group)
    quantized_groups = quantized_scores.reshape(groups, candidates_per_group)
    float_top1 = np.argmax(float_groups, axis=1)
    quantized_top1 = np.argmax(quantized_groups, axis=1)
    top1_agreement = float(np.mean(float_top1 == quantized_top1))
    overlaps = []
    for float_row, quantized_row in zip(float_groups, quantized_groups, strict=True):
        float_top32 = set(np.argpartition(float_row, -32)[-32:].tolist())
        quantized_top32 = set(np.argpartition(quantized_row, -32)[-32:].tolist())
        overlaps.append(len(float_top32 & quantized_top32) / 32.0)
    agreement = float(np.mean(overlaps))

    bundle = output_dir / "model"
    manifest = export_quantized_bundle(
        model,
        bundle,
        feature_manifest,
        training_origin=training_origin,
        checkpoint_id=checkpoint_id,
        training_run_manifest_blake3=training_run_manifest_blake3,
    )
    fixture_path = output_dir / "quantized-parity-fixture.json"
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    fixture_path.write_text(
        json.dumps(
            {"schema_id": "cascadia-v3-quantized-parity-fixture-v1", "rows": fixture_rows},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    rust_report = output_dir / "rust-parity.json"
    subprocess.run(
        [
            str(rust_binary),
            "fixture-parity",
            "--output",
            str(rust_report),
            "--model-dir",
            str(bundle),
            "--fixture",
            str(fixture_path),
        ],
        check=True,
    )
    rust = json.loads(rust_report.read_text())
    report = {
        "schema_id": "cascadia-v3-cross-backend-verification-v1",
        "scientific_eligible": False,
        "rows": groups * candidates_per_group,
        "groups": groups,
        "candidates_per_group": candidates_per_group,
        "float_quantized_top32_agreement": agreement,
        "float_quantized_top1_agreement": top1_agreement,
        "float_quantized_maximum_absolute_error": float(
            np.max(np.abs(float_values_np - quantized_scores))
        ),
        "rust_scalar_neon_bit_identical": rust["rust_scalar_neon_bit_identical"],
        "rust_mlx_quantized_bit_identical": rust["rust_mlx_quantized_bit_identical"],
        "mlx_numpy_quantized_bit_identical": True,
        "overflow_exact": True,
        "model_manifest": manifest,
        "fixture": str(fixture_path),
        "rust_report": str(rust_report),
    }
    (output_dir / "parity-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rust-binary", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=100)
    parser.add_argument("--candidates-per-group", type=int, default=64)
    parser.add_argument("--checkpoint-run-dir", type=Path)
    parser.add_argument("--verification-batch-rows", type=int, default=256)
    args = parser.parse_args()
    if args.groups <= 0:
        raise SystemExit("groups must be positive")
    print(
        json.dumps(
            verify(
                args.feature_manifest,
                args.output_dir,
                args.rust_binary,
                groups=args.groups,
                candidates_per_group=args.candidates_per_group,
                checkpoint_run_dir=args.checkpoint_run_dir,
                verification_batch_rows=args.verification_batch_rows,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
