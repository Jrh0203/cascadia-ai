"""Convert, verify, and benchmark the qualified sparse NNUE on MLX."""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import time
import uuid
from collections import Counter
from pathlib import Path

import blake3
import mlx.core as mx
import numpy as np

from cascadia_mlx.legacy_nnue import (
    CORRECTED_NNUE_BASE_END,
    CORRECTED_NNUE_MAGIC,
    CORRECTED_NNUE_OPPONENT_END,
    CORRECTED_NNUE_OPPONENT_START,
    CORRECTED_NNUE_OVERFLOW_END,
    CORRECTED_NNUE_TERRAIN_START,
    LEGACY_NNUE_FEATURES,
    LegacyNnueError,
    LegacyNnueWeights,
    LegacyRustExactSparseNnue,
    LegacySparseNnue,
    checksum_file,
    convert_corrected_nnue,
    convert_legacy_nnue,
    load_legacy_nnue_manifest,
    pack_sparse_csr,
    pack_sparse_features,
    parse_legacy_nnue,
    reference_forward,
    remap_historical_features_to_corrected,
)
from cascadia_mlx.legacy_nnue_serve import (
    FRAME_HEADER,
    MESSAGE_PREDICT_SPARSE_NNUE,
    MESSAGE_SHUTDOWN,
    MESSAGE_SPARSE_NNUE_PREDICTION,
    PROTOCOL_MAGIC,
    PROTOCOL_VERSION,
    ROW_LENGTH,
    serve_legacy_nnue,
)

PARITY_FIXTURE_SCHEMA = 1
PARITY_FIXTURE_FEATURE_SCHEMA = "legacy-mid-v4opp-sparse-u16-v1"
PARITY_FIXTURE_GAME_INDEX = 92_000
PARITY_FIXTURE_RECORDS = 80


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _synthetic_features(seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    rows = [
        [],
        [0],
        [LEGACY_NNUE_FEATURES - 1],
        list(range(0, LEGACY_NNUE_FEATURES, 37)),
    ]
    for _ in range(256):
        count = int(rng.integers(1, 257))
        rows.append(rng.choice(LEGACY_NNUE_FEATURES, size=count, replace=False).tolist())
    return rows


def _error_metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    errors = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return {
        "maximum_absolute_error": float(np.max(errors)),
        "p99_absolute_error": float(np.percentile(errors, 99)),
        "mean_absolute_error": float(np.mean(errors)),
    }


def _array_blake3(array: np.ndarray) -> str:
    return blake3.blake3(np.asarray(array).tobytes(order="C")).hexdigest()


def _bit_identical(left: np.ndarray, right: np.ndarray) -> bool:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    return left.shape == right.shape and left.tobytes(order="C") == right.tobytes(order="C")


def _non_first_layer_tensors(weights: LegacyNnueWeights) -> dict[str, np.ndarray]:
    names = (
        "b1",
        "w2",
        "b2",
        "w3",
        "b3",
        "w3_policy",
        "b3_policy",
        "w3_wildlife",
        "b3_wildlife",
        "w3_habitat",
        "b3_habitat",
        "w3_heads",
        "b3_heads",
        "w3_var",
        "b3_var",
    )
    return {name: tensor for name in names if (tensor := getattr(weights, name)) is not None}


def _predict(model: LegacySparseNnue, feature_sets: list[list[int]]) -> np.ndarray:
    indices, mask = pack_sparse_features(feature_sets)
    values = model(indices, mask)
    mx.eval(values)
    return np.asarray(values, dtype=np.float32)


def _load_and_validate_fixture(path: Path) -> dict[str, object]:
    try:
        fixture = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise LegacyNnueError(f"cannot read NNUE parity fixture: {error}") from error
    expected_header = {
        "schema_version": PARITY_FIXTURE_SCHEMA,
        "feature_schema": PARITY_FIXTURE_FEATURE_SCHEMA,
        "split": "train",
        "first_game_index": PARITY_FIXTURE_GAME_INDEX,
        "games": 1,
        "feature_count": LEGACY_NNUE_FEATURES,
        "hidden1": 512,
        "hidden2": 64,
    }
    for key, expected in expected_header.items():
        if fixture.get(key) != expected:
            raise LegacyNnueError(
                f"NNUE parity fixture field {key} is {fixture.get(key)!r}, expected {expected!r}"
            )
    records = fixture.get("records")
    if not isinstance(records, list) or len(records) != PARITY_FIXTURE_RECORDS:
        raise LegacyNnueError(f"NNUE parity fixture must contain {PARITY_FIXTURE_RECORDS} records")

    duplicate_records = 0
    duplicate_occurrences = 0
    maximum_multiplicity = 1
    for decision_index, record in enumerate(records):
        if (
            not isinstance(record, dict)
            or record.get("game_index") != PARITY_FIXTURE_GAME_INDEX
            or record.get("decision_index") != decision_index
            or not isinstance(record.get("active_seat"), int)
            or not 0 <= record["active_seat"] < 4
            or not isinstance(record.get("rust_value"), (int, float))
            or not math.isfinite(record["rust_value"])
        ):
            raise LegacyNnueError(
                f"NNUE parity fixture record {decision_index} has invalid identity or value"
            )
        features = record.get("features")
        if not isinstance(features, list) or not features:
            raise LegacyNnueError(
                f"NNUE parity fixture record {decision_index} has no sparse features"
            )
        if any(
            not isinstance(index, int) or index < 0 or index >= LEGACY_NNUE_FEATURES
            for index in features
        ):
            raise LegacyNnueError(
                f"NNUE parity fixture record {decision_index} has an invalid feature index"
            )
        counts = Counter(features)
        record_maximum = max(counts.values())
        maximum_multiplicity = max(maximum_multiplicity, record_maximum)
        if len(counts) != len(features):
            duplicate_records += 1
            duplicate_occurrences += len(features) - len(counts)

    duplicate_header = {
        "records_with_duplicate_features": duplicate_records,
        "duplicate_feature_occurrences": duplicate_occurrences,
        "maximum_feature_multiplicity": maximum_multiplicity,
    }
    for key, expected in duplicate_header.items():
        if fixture.get(key) != expected:
            raise LegacyNnueError(f"NNUE parity fixture duplicate metadata {key} is inconsistent")

    provenance = fixture.get("provenance")
    if not isinstance(provenance, dict):
        raise LegacyNnueError("NNUE parity fixture has no provenance")
    if provenance.get("weights_blake3") != (
        "9e1d568693274fc537ac4f6d6f729abb1ee8da8330a78d1f78a1f62b733de400"
    ):
        raise LegacyNnueError("NNUE parity fixture weights provenance is invalid")
    if provenance.get("legacy_environment") != [
        ["MCE_LMR", "1"],
        ["MCE_DIVERSE_PREFILTER", "1"],
    ]:
        raise LegacyNnueError("NNUE parity fixture legacy environment is invalid")
    source = provenance.get("source")
    if (
        not isinstance(source, dict)
        or not isinstance(source.get("v2_source_blake3"), str)
        or len(source["v2_source_blake3"]) != 64
        or not isinstance(provenance.get("executable_blake3"), str)
        or len(provenance["executable_blake3"]) != 64
    ):
        raise LegacyNnueError("NNUE parity fixture source provenance is invalid")
    return fixture


def run_parity(
    source: Path,
    model_dir: Path,
    fixture_path: Path,
    output: Path,
    synthetic_seed: int,
) -> dict[str, object]:
    weights = parse_legacy_nnue(source)
    manifest = load_legacy_nnue_manifest(model_dir)
    model = LegacySparseNnue.load(model_dir)
    synthetic = _synthetic_features(synthetic_seed)
    synthetic_expected = np.asarray(
        [reference_forward(weights, features) for features in synthetic],
        dtype=np.float32,
    )
    synthetic_actual = _predict(model, synthetic)

    fixture = _load_and_validate_fixture(fixture_path)
    records = fixture["records"]
    real_features = [[int(index) for index in record["features"]] for record in records]
    real_expected = np.asarray([record["rust_value"] for record in records], dtype=np.float32)
    real_actual = _predict(model, real_features)
    repeated = _predict(model, real_features)

    synthetic_metrics = _error_metrics(synthetic_actual, synthetic_expected)
    real_metrics = _error_metrics(real_actual, real_expected)
    gates = {
        "artifact_integrity": True,
        "fixture_contract": True,
        "fixture_records": len(records) == PARITY_FIXTURE_RECORDS,
        "synthetic_maximum_absolute_error": (synthetic_metrics["maximum_absolute_error"] <= 1e-3),
        "real_maximum_absolute_error": real_metrics["maximum_absolute_error"] <= 1e-3,
        "real_p99_absolute_error": real_metrics["p99_absolute_error"] <= 5e-4,
        "real_mean_absolute_error": real_metrics["mean_absolute_error"] <= 1e-4,
        "deterministic_repeat": bool(np.array_equal(real_actual, repeated)),
        "finite": bool(np.all(np.isfinite(synthetic_actual)) and np.all(np.isfinite(real_actual))),
    }
    report = {
        "schema_version": 1,
        "device": str(mx.default_device()),
        "synthetic_seed": synthetic_seed,
        "synthetic_records": len(synthetic),
        "real_records": len(records),
        "inputs": {
            "source": {
                "path": str(source.resolve()),
                "bytes": source.stat().st_size,
                "blake3": checksum_file(source),
            },
            "model_manifest": {
                "path": str((model_dir / "model.json").resolve()),
                "blake3": checksum_file(model_dir / "model.json"),
            },
            "model_safetensors": manifest["files"]["model.safetensors"],
            "fixture": {
                "path": str(fixture_path.resolve()),
                "bytes": fixture_path.stat().st_size,
                "blake3": checksum_file(fixture_path),
                "v2_source_blake3": fixture["provenance"]["source"]["v2_source_blake3"],
                "executable_blake3": fixture["provenance"]["executable_blake3"],
                "records_with_duplicate_features": fixture["records_with_duplicate_features"],
                "duplicate_feature_occurrences": fixture["duplicate_feature_occurrences"],
                "maximum_feature_multiplicity": fixture["maximum_feature_multiplicity"],
            },
        },
        "synthetic": synthetic_metrics,
        "real": real_metrics,
        "gates": gates,
        "passed": all(gates.values()),
    }
    _write_json_atomic(output, report)
    return report


def run_corrected_parity(
    historical_source: Path,
    corrected_source: Path,
    fixture_path: Path,
    output: Path,
    model_dir: Path,
) -> dict[str, object]:
    """Prove Rust migration rows and Rust-order MLX predictions are exact."""

    historical = parse_legacy_nnue(historical_source)
    corrected = parse_legacy_nnue(corrected_source)
    if historical.container_magic != b"NNUE":
        raise LegacyNnueError("corrected parity control must use the historical NNUE container")
    if historical.feature_count != LEGACY_NNUE_FEATURES:
        raise LegacyNnueError("corrected parity control must be the 11,231-row champion layout")
    if corrected.container_magic != CORRECTED_NNUE_MAGIC or not corrected.is_corrected:
        raise LegacyNnueError("corrected parity treatment must use the NNUC corrected container")
    if corrected.version != historical.version:
        raise LegacyNnueError("historical and corrected checkpoint head versions differ")

    base_source = historical.w1[:CORRECTED_NNUE_BASE_END]
    base_treatment = corrected.w1[:CORRECTED_NNUE_BASE_END]
    opponent_source = historical.w1[10_862:LEGACY_NNUE_FEATURES]
    opponent_treatment = corrected.w1[CORRECTED_NNUE_OPPONENT_START:CORRECTED_NNUE_OPPONENT_END]
    corrected_tail = corrected.w1[CORRECTED_NNUE_TERRAIN_START:CORRECTED_NNUE_OVERFLOW_END]
    downstream_control = _non_first_layer_tensors(historical)
    downstream_treatment = _non_first_layer_tensors(corrected)
    downstream_names_match = downstream_control.keys() == downstream_treatment.keys()
    downstream_byte_parity = downstream_names_match and all(
        _bit_identical(downstream_control[name], downstream_treatment[name])
        for name in downstream_control
    )

    fixture = _load_and_validate_fixture(fixture_path)
    records = fixture["records"]
    historical_features = [[int(index) for index in record["features"]] for record in records]
    discarded_activations = sum(
        1
        for features in historical_features
        for index in features
        if CORRECTED_NNUE_BASE_END <= index < 10_862
    )
    corrected_features = [
        remap_historical_features_to_corrected(features) for features in historical_features
    ]
    rust_fixture_values = np.asarray(
        [record["rust_value"] for record in records],
        dtype=np.float32,
    )
    historical_reference = np.asarray(
        [reference_forward(historical, features) for features in historical_features],
        dtype=np.float32,
    )
    corrected_reference = np.asarray(
        [reference_forward(corrected, features) for features in corrected_features],
        dtype=np.float32,
    )
    exact_model = LegacyRustExactSparseNnue(corrected.tensors())
    offsets, indices = pack_sparse_csr(corrected_features)
    corrected_mlx = exact_model(offsets, indices)
    mx.eval(corrected_mlx)
    corrected_mlx_values = np.asarray(corrected_mlx, dtype=np.float32)
    artifact_manifest = None
    artifact_values = None
    if model_dir is not None:
        artifact_manifest = load_legacy_nnue_manifest(model_dir)
        if artifact_manifest["source"]["blake3"] != checksum_file(corrected_source):
            raise LegacyNnueError("corrected MLX artifact source does not match the checkpoint")
        artifact_model = LegacyRustExactSparseNnue.load(model_dir)
        artifact_predictions = artifact_model(offsets, indices)
        mx.eval(artifact_predictions)
        artifact_values = np.asarray(artifact_predictions, dtype=np.float32)

    tail_bits = corrected_tail.view(np.uint32)
    gates = {
        "historical_container": historical.container_magic == b"NNUE",
        "corrected_container": corrected.is_corrected,
        "head_version_match": corrected.version == historical.version,
        "base_rows_byte_identical": _bit_identical(base_source, base_treatment),
        "opponent_rows_byte_identical": _bit_identical(
            opponent_source,
            opponent_treatment,
        ),
        "corrected_tail_all_signed_zero": bool(np.all((tail_bits & 0x7FFF_FFFF) == 0)),
        "non_first_layer_tensors_byte_identical": downstream_byte_parity,
        "fixture_has_no_discarded_row_activations": discarded_activations == 0,
        "historical_reference_matches_rust_fixture_bits": _bit_identical(
            historical_reference,
            rust_fixture_values,
        ),
        "corrected_reference_matches_historical_bits": _bit_identical(
            corrected_reference,
            historical_reference,
        ),
        "corrected_mlx_matches_corrected_reference_bits": _bit_identical(
            corrected_mlx_values,
            corrected_reference,
        ),
        "corrected_mlx_matches_rust_fixture_bits": _bit_identical(
            corrected_mlx_values,
            rust_fixture_values,
        ),
        "corrected_artifact_integrity": artifact_manifest is not None,
        "corrected_artifact_matches_source_mlx_bits": (
            artifact_values is not None and _bit_identical(artifact_values, corrected_mlx_values)
        ),
        "corrected_artifact_matches_rust_fixture_bits": (
            artifact_values is not None and _bit_identical(artifact_values, rust_fixture_values)
        ),
        "finite": bool(np.all(np.isfinite(corrected_mlx_values))),
    }
    report = {
        "schema_version": 1,
        "experiment_id": "corrected-mid-tail-v1-mlx-foundation",
        "schema_id": corrected.schema_id,
        "device": str(mx.default_device()),
        "records": len(records),
        "discarded_row_activations": discarded_activations,
        "inputs": {
            "historical_checkpoint": {
                "path": str(historical_source.resolve()),
                "bytes": historical_source.stat().st_size,
                "blake3": checksum_file(historical_source),
            },
            "corrected_checkpoint": {
                "path": str(corrected_source.resolve()),
                "bytes": corrected_source.stat().st_size,
                "blake3": checksum_file(corrected_source),
                "head_version": corrected.version,
            },
            "rust_fixture": {
                "path": str(fixture_path.resolve()),
                "bytes": fixture_path.stat().st_size,
                "blake3": checksum_file(fixture_path),
                "records": len(records),
            },
            "mlx_artifact": (
                {
                    "path": str(model_dir.resolve()),
                    "manifest_blake3": checksum_file(model_dir / "model.json"),
                    "safetensors": artifact_manifest["files"]["model.safetensors"],
                }
                if model_dir is not None and artifact_manifest is not None
                else None
            ),
        },
        "first_layer": {
            "base_rows": {
                "count": CORRECTED_NNUE_BASE_END,
                "source_blake3": _array_blake3(base_source),
                "corrected_blake3": _array_blake3(base_treatment),
            },
            "opponent_rows": {
                "count": CORRECTED_NNUE_OPPONENT_END - CORRECTED_NNUE_OPPONENT_START,
                "source_blake3": _array_blake3(opponent_source),
                "corrected_blake3": _array_blake3(opponent_treatment),
            },
            "zero_initialized_tail_rows": len(corrected_tail),
            "zero_initialized_tail_blake3": _array_blake3(corrected_tail),
        },
        "predictions": {
            "historical_reference_vs_rust_fixture": _error_metrics(
                historical_reference,
                rust_fixture_values,
            ),
            "corrected_reference_vs_historical_reference": _error_metrics(
                corrected_reference,
                historical_reference,
            ),
            "corrected_mlx_vs_rust_fixture": _error_metrics(
                corrected_mlx_values,
                rust_fixture_values,
            ),
            "rust_fixture_blake3": _array_blake3(rust_fixture_values),
            "corrected_mlx_blake3": _array_blake3(corrected_mlx_values),
            "corrected_artifact_blake3": (
                _array_blake3(artifact_values) if artifact_values is not None else None
            ),
        },
        "gates": gates,
        "passed": all(gates.values()),
    }
    _write_json_atomic(output, report)
    return report


def run_benchmark(
    model_dir: Path,
    output: Path,
    seed: int,
    iterations: int,
) -> dict[str, object]:
    manifest = load_legacy_nnue_manifest(model_dir)
    model = LegacySparseNnue.load(model_dir)
    rng = np.random.default_rng(seed)
    results: dict[str, object] = {}
    for batch_size in (1, 32, 256):
        features = [
            rng.choice(
                LEGACY_NNUE_FEATURES,
                size=int(rng.integers(80, 181)),
                replace=False,
            ).tolist()
            for _ in range(batch_size)
        ]
        indices, mask = pack_sparse_features(features)
        for _ in range(5):
            warm = model(indices, mask)
            mx.eval(warm)
        latencies = []
        for _ in range(iterations):
            started = time.perf_counter()
            values = model(indices, mask)
            mx.eval(values)
            latencies.append(time.perf_counter() - started)
        durations = np.asarray(latencies, dtype=np.float64)
        results[str(batch_size)] = {
            "iterations": iterations,
            "p50_milliseconds": float(np.percentile(durations, 50) * 1000.0),
            "p90_milliseconds": float(np.percentile(durations, 90) * 1000.0),
            "p99_milliseconds": float(np.percentile(durations, 99) * 1000.0),
            "evaluations_per_second": float(batch_size / np.median(durations)),
        }
    batch32 = results["32"]
    report = {
        "schema_version": 1,
        "device": str(mx.default_device()),
        "seed": seed,
        "model": {
            "manifest_blake3": checksum_file(model_dir / "model.json"),
            "safetensors": manifest["files"]["model.safetensors"],
        },
        "batches": results,
        "gates": {"batch32_evaluations_per_second": (batch32["evaluations_per_second"] >= 2000.0)},
    }
    report["passed"] = all(report["gates"].values())
    _write_json_atomic(output, report)
    return report


def run_service_direct_parity(
    model_dir: Path,
    fixture_path: Path,
    output: Path,
) -> dict[str, object]:
    model = LegacySparseNnue.load(model_dir)
    fixture = _load_and_validate_fixture(fixture_path)
    records = fixture["records"]
    feature_sets = [[int(index) for index in record["features"]] for record in records]
    direct = _predict(model, feature_sets)

    request_id = 1
    request = bytearray(
        FRAME_HEADER.pack(
            PROTOCOL_MAGIC,
            PROTOCOL_VERSION,
            MESSAGE_PREDICT_SPARSE_NNUE,
            request_id,
            len(feature_sets),
        )
    )
    for features in feature_sets:
        request.extend(ROW_LENGTH.pack(len(features)))
        request.extend(np.asarray(features, dtype="<u2").tobytes())
    request.extend(
        FRAME_HEADER.pack(
            PROTOCOL_MAGIC,
            PROTOCOL_VERSION,
            MESSAGE_SHUTDOWN,
            request_id + 1,
            0,
        )
    )
    output_stream = io.BytesIO()
    serve_legacy_nnue(model, io.BytesIO(request), output_stream)
    response = output_stream.getvalue()
    header = FRAME_HEADER.unpack(response[: FRAME_HEADER.size])
    expected_header = (
        PROTOCOL_MAGIC,
        PROTOCOL_VERSION,
        MESSAGE_SPARSE_NNUE_PREDICTION,
        request_id,
        len(feature_sets),
    )
    if header != expected_header:
        raise LegacyNnueError(f"sparse service returned header {header!r}")
    service = np.frombuffer(response[FRAME_HEADER.size :], dtype="<f4")
    if service.shape != direct.shape:
        raise LegacyNnueError("sparse service returned the wrong prediction width")
    metrics = _error_metrics(service, direct)
    gates = {
        "fixture_records": len(feature_sets) == PARITY_FIXTURE_RECORDS,
        "finite": bool(np.all(np.isfinite(service))),
        "maximum_absolute_error": metrics["maximum_absolute_error"] <= 1e-6,
        "bit_identical": bool(np.array_equal(service, direct)),
    }
    report = {
        "schema_version": 1,
        "device": str(mx.default_device()),
        "records": len(feature_sets),
        "service_vs_direct_mlx": metrics,
        "gates": gates,
        "passed": all(gates.values()),
        "inputs": {
            "model_manifest_blake3": checksum_file(model_dir / "model.json"),
            "model_safetensors_blake3": checksum_file(model_dir / "model.safetensors"),
            "fixture_blake3": checksum_file(fixture_path),
        },
    }
    _write_json_atomic(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert = subparsers.add_parser("convert")
    convert.add_argument("--source", type=Path, required=True)
    convert.add_argument("--output", type=Path, required=True)

    convert_corrected = subparsers.add_parser("convert-corrected")
    convert_corrected.add_argument("--source", type=Path, required=True)
    convert_corrected.add_argument("--output", type=Path, required=True)

    parity = subparsers.add_parser("parity")
    parity.add_argument("--source", type=Path, required=True)
    parity.add_argument("--model-dir", type=Path, required=True)
    parity.add_argument("--fixture", type=Path, required=True)
    parity.add_argument("--output", type=Path, required=True)
    parity.add_argument("--synthetic-seed", type=int, default=20260619)

    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--model-dir", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--seed", type=int, default=20260619)
    benchmark.add_argument("--iterations", type=int, default=200)

    service_parity = subparsers.add_parser("service-parity")
    service_parity.add_argument("--model-dir", type=Path, required=True)
    service_parity.add_argument("--fixture", type=Path, required=True)
    service_parity.add_argument("--output", type=Path, required=True)

    corrected_parity = subparsers.add_parser("corrected-parity")
    corrected_parity.add_argument("--historical-source", type=Path, required=True)
    corrected_parity.add_argument("--corrected-source", type=Path, required=True)
    corrected_parity.add_argument("--model-dir", type=Path, required=True)
    corrected_parity.add_argument("--fixture", type=Path, required=True)
    corrected_parity.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    try:
        if args.command == "convert":
            report = convert_legacy_nnue(args.source, args.output)
        elif args.command == "convert-corrected":
            report = convert_corrected_nnue(args.source, args.output)
        elif args.command == "parity":
            report = run_parity(
                args.source,
                args.model_dir,
                args.fixture,
                args.output,
                args.synthetic_seed,
            )
        elif args.command == "benchmark":
            report = run_benchmark(
                args.model_dir,
                args.output,
                args.seed,
                args.iterations,
            )
        elif args.command == "corrected-parity":
            report = run_corrected_parity(
                args.historical_source,
                args.corrected_source,
                args.fixture,
                args.output,
                args.model_dir,
            )
        else:
            report = run_service_direct_parity(
                args.model_dir,
                args.fixture,
                args.output,
            )
    except LegacyNnueError as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
