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

import mlx.core as mx
import numpy as np

from cascadia_mlx.legacy_nnue import (
    LEGACY_NNUE_FEATURES,
    LegacyNnueError,
    LegacySparseNnue,
    checksum_file,
    convert_legacy_nnue,
    load_legacy_nnue_manifest,
    pack_sparse_features,
    parse_legacy_nnue,
    reference_forward,
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

    args = parser.parse_args()
    try:
        if args.command == "convert":
            report = convert_legacy_nnue(args.source, args.output)
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
