"""Long-lived sparse inference service for the qualified legacy NNUE."""

from __future__ import annotations

import argparse
import json
import mmap
import os
import struct
import sys
import time
from itertools import pairwise
from pathlib import Path
from typing import BinaryIO

import mlx.core as mx
import numpy as np

from cascadia_mlx.legacy_nnue import (
    LEGACY_NNUE_FEATURES,
    LEGACY_NNUE_HIDDEN2,
    LegacyRustExactSparseNnue,
    LegacySparseNnue,
    pack_sparse_features,
)
from cascadia_mlx.serve import (
    FRAME_HEADER,
    MAX_BATCH,
    MESSAGE_ERROR,
    MESSAGE_SHUTDOWN,
    PROTOCOL_MAGIC,
    PROTOCOL_VERSION,
    ProtocolError,
    _read_exact,
    _read_exact_or_eof,
)

MESSAGE_PREDICT_SPARSE_NNUE = 5
MESSAGE_SPARSE_NNUE_PREDICTION = 0x8005
MESSAGE_PREDICT_SPARSE_NNUE_CSR_EXACT = 6
MESSAGE_SPARSE_NNUE_CSR_EXACT_PREDICTION = 0x8006
MESSAGE_PREDICT_SPARSE_NNUE_CSR_EXACT_HIDDEN = 7
MESSAGE_SPARSE_NNUE_CSR_EXACT_HIDDEN_PREDICTION = 0x8007
MESSAGE_PREDICT_SPARSE_NNUE_CSR_EXACT_SHARED = 8
MESSAGE_SPARSE_NNUE_CSR_EXACT_SHARED_PREDICTION = 0x8008
MAX_SPARSE_FEATURES_PER_ROW = 4_096
ROW_LENGTH = struct.Struct("<H")
TOTAL_FEATURES = struct.Struct("<I")
SHARED_MAGIC = b"CSHM"
SHARED_VERSION = 1
SHARED_HEADER = struct.Struct("<4sIII")
REQUEST_SIZE_BUCKET_UPPER_BOUNDS = (32, 64, 128, 256, 512, 1_024, None)


def _align_to_four(value: int) -> int:
    return (value + 3) & ~3


def _stage_timings_enabled() -> bool:
    value = os.environ.get("CASCADIA_MLX_STAGE_TIMINGS")
    return value is not None and value != "" and value != "0"


def _activation_diagnostics_enabled() -> bool:
    value = os.environ.get("CASCADIA_MLX_ACTIVATION_DIAGNOSTICS")
    return value is not None and value != "" and value != "0"


def _request_size_bucket_index(rows: int) -> int:
    for index, upper_bound in enumerate(REQUEST_SIZE_BUCKET_UPPER_BOUNDS):
        if upper_bound is None or rows <= upper_bound:
            return index
    raise AssertionError("the final request-size bucket must be unbounded")


def _activation_report(
    model: LegacyRustExactSparseNnue,
    offsets: np.ndarray,
    features: np.ndarray,
) -> dict[str, int | float]:
    h1, h2, _ = model.all_hidden_and_output(mx.array(offsets), mx.array(features))
    counts = mx.stack(
        [
            mx.sum(h1 > 0.0),
            mx.array(h1.size, dtype=mx.uint32),
            mx.sum(h2 > 0.0),
            mx.array(h2.size, dtype=mx.uint32),
        ]
    )
    mx.eval(counts)
    positive_h1, total_h1, positive_h2, total_h2 = (int(value) for value in np.asarray(counts))
    return {
        "largest_batch_h1_positive": positive_h1,
        "largest_batch_h1_total": total_h1,
        "largest_batch_h1_density": positive_h1 / total_h1 if total_h1 else 0.0,
        "largest_batch_h2_positive": positive_h2,
        "largest_batch_h2_total": total_h2,
        "largest_batch_h2_density": positive_h2 / total_h2 if total_h2 else 0.0,
    }


def _prefix_sharing_report(
    offsets: np.ndarray,
    features: np.ndarray,
) -> dict[str, int | float]:
    rows = [
        tuple(int(value) for value in features[offsets[row] : offsets[row + 1]])
        for row in range(len(offsets) - 1)
    ]
    physical_features = sum(len(row) for row in rows)
    thresholds = (8, 16, 32, 64, 96, 128)

    def adjacent_savings(
        ordered_rows: list[tuple[int, ...]],
    ) -> tuple[int, dict[int, int]]:
        trie_edges = len(ordered_rows[0]) if ordered_rows else 0
        savings = {threshold: 0 for threshold in thresholds}
        for previous, current in pairwise(ordered_rows):
            common = 0
            for left, right in zip(previous, current, strict=False):
                if left != right:
                    break
                common += 1
            trie_edges += len(current) - common
            for threshold in savings:
                if common >= threshold:
                    savings[threshold] += threshold
        return trie_edges, savings

    contiguous_edges, contiguous_savings = adjacent_savings(rows)
    sorted_rows = sorted(rows)
    trie_edges, savings = adjacent_savings(sorted_rows)
    report: dict[str, int | float] = {
        "largest_batch_rows": len(rows),
        "largest_batch_features": physical_features,
        "largest_batch_trie_edges": trie_edges,
        "largest_batch_trie_reduction": (
            1.0 - trie_edges / physical_features if physical_features else 0.0
        ),
        "largest_batch_contiguous_trie_edges": contiguous_edges,
        "largest_batch_contiguous_trie_reduction": (
            1.0 - contiguous_edges / physical_features if physical_features else 0.0
        ),
    }
    for threshold, saved in savings.items():
        report[f"largest_batch_prefix_{threshold}_saved_features"] = saved
        report[f"largest_batch_prefix_{threshold}_reduction"] = (
            saved / physical_features if physical_features else 0.0
        )
        contiguous_saved = contiguous_savings[threshold]
        report[f"largest_batch_contiguous_prefix_{threshold}_saved_features"] = contiguous_saved
        report[f"largest_batch_contiguous_prefix_{threshold}_reduction"] = (
            contiguous_saved / physical_features if physical_features else 0.0
        )
    return report


def _emit_stage_timings(
    stats: dict[str, int],
    request_size_buckets: list[dict[str, int]],
    largest_batch: tuple[np.ndarray, np.ndarray] | None,
    exact_model: LegacyRustExactSparseNnue | None,
) -> None:
    if stats["requests"] == 0:
        return

    def milliseconds(nanoseconds: int) -> float:
        return nanoseconds / 1_000_000.0

    report = {
        "event": "legacy_nnue_service_stage_timings",
        "requests": stats["requests"],
        "rows": stats["rows"],
        "features": stats["features"],
        "payload_read_ms": milliseconds(stats["payload_read_ns"]),
        "decode_validate_ms": milliseconds(stats["decode_validate_ns"]),
        "graph_build_ms": milliseconds(stats["graph_build_ns"]),
        "mlx_eval_ms": milliseconds(stats["mlx_eval_ns"]),
        "materialize_validate_ms": milliseconds(stats["materialize_validate_ns"]),
        "response_write_ms": milliseconds(stats["response_write_ns"]),
        "request_total_ms": milliseconds(stats["request_total_ns"]),
        "request_size_buckets": [
            {
                "maximum_rows": upper_bound,
                "requests": bucket["requests"],
                "rows": bucket["rows"],
                "features": bucket["features"],
                "mlx_eval_ms": milliseconds(bucket["mlx_eval_ns"]),
            }
            for upper_bound, bucket in zip(
                REQUEST_SIZE_BUCKET_UPPER_BOUNDS,
                request_size_buckets,
                strict=True,
            )
        ],
    }
    report["accounted_ms"] = sum(
        report[key]
        for key in (
            "payload_read_ms",
            "decode_validate_ms",
            "graph_build_ms",
            "mlx_eval_ms",
            "materialize_validate_ms",
            "response_write_ms",
        )
    )
    if largest_batch is not None:
        report.update(_prefix_sharing_report(*largest_batch))
        if exact_model is not None and _activation_diagnostics_enabled():
            report.update(_activation_report(exact_model, *largest_batch))
    print(json.dumps(report, sort_keys=True), file=sys.stderr)


def serve_legacy_nnue(
    model: LegacySparseNnue,
    input_stream: BinaryIO,
    output_stream: BinaryIO,
    exact_model: LegacyRustExactSparseNnue | None = None,
    shared_buffer: mmap.mmap | bytearray | memoryview | None = None,
) -> None:
    """Serve ordered variable-length sparse batches until shutdown or EOF."""
    timings_enabled = _stage_timings_enabled()
    stats = {
        "requests": 0,
        "rows": 0,
        "features": 0,
        "payload_read_ns": 0,
        "decode_validate_ns": 0,
        "graph_build_ns": 0,
        "mlx_eval_ns": 0,
        "materialize_validate_ns": 0,
        "response_write_ns": 0,
        "request_total_ns": 0,
    }
    request_size_buckets = [
        {"requests": 0, "rows": 0, "features": 0, "mlx_eval_ns": 0}
        for _ in REQUEST_SIZE_BUCKET_UPPER_BOUNDS
    ]
    largest_batch: tuple[np.ndarray, np.ndarray] | None = None
    try:
        while True:
            header = _read_exact_or_eof(input_stream, FRAME_HEADER.size)
            if header is None:
                return
            magic, version, message_type, request_id, count = FRAME_HEADER.unpack(header)
            request_started = time.perf_counter_ns() if timings_enabled else 0
            try:
                if magic != PROTOCOL_MAGIC or version != PROTOCOL_VERSION:
                    raise ProtocolError("incompatible protocol header")
                if message_type == MESSAGE_SHUTDOWN:
                    if count:
                        raise ProtocolError("shutdown frame cannot contain records")
                    return
                if message_type not in (
                    MESSAGE_PREDICT_SPARSE_NNUE,
                    MESSAGE_PREDICT_SPARSE_NNUE_CSR_EXACT,
                    MESSAGE_PREDICT_SPARSE_NNUE_CSR_EXACT_HIDDEN,
                    MESSAGE_PREDICT_SPARSE_NNUE_CSR_EXACT_SHARED,
                ):
                    raise ProtocolError(f"unsupported message type {message_type}")
                if count == 0 or count > MAX_BATCH:
                    raise ProtocolError(f"invalid prediction batch size {count}")

                payload_started = time.perf_counter_ns() if timings_enabled else 0
                shared_response_offset: int | None = None
                if message_type == MESSAGE_PREDICT_SPARSE_NNUE:
                    feature_sets = []
                    total_features = 0
                    for row in range(count):
                        feature_count = ROW_LENGTH.unpack(
                            _read_exact(input_stream, ROW_LENGTH.size)
                        )[0]
                        if feature_count > MAX_SPARSE_FEATURES_PER_ROW:
                            raise ProtocolError(
                                f"sparse row {row} has {feature_count} features, "
                                f"maximum is {MAX_SPARSE_FEATURES_PER_ROW}"
                            )
                        payload = _read_exact(input_stream, feature_count * 2)
                        features = np.frombuffer(payload, dtype="<u2").astype(np.int32).tolist()
                        if any(index >= LEGACY_NNUE_FEATURES for index in features):
                            raise ProtocolError(f"sparse row {row} contains an out-of-range index")
                        total_features += feature_count
                        feature_sets.append(features)
                    if timings_enabled:
                        stats["payload_read_ns"] += time.perf_counter_ns() - payload_started
                    decode_started = time.perf_counter_ns() if timings_enabled else 0
                    indices, mask = pack_sparse_features(feature_sets)
                    if timings_enabled:
                        stats["decode_validate_ns"] += time.perf_counter_ns() - decode_started
                    graph_started = time.perf_counter_ns() if timings_enabled else 0
                    predictions = model(indices, mask)
                    response_type = MESSAGE_SPARSE_NNUE_PREDICTION
                else:
                    if exact_model is None:
                        raise ProtocolError("exact sparse NNUE operation is unavailable")
                    offsets_bytes = (count + 1) * 4
                    if message_type == MESSAGE_PREDICT_SPARSE_NNUE_CSR_EXACT_SHARED:
                        if shared_buffer is None:
                            raise ProtocolError("shared sparse NNUE transport is unavailable")
                        shared = memoryview(shared_buffer)
                        if len(shared) < SHARED_HEADER.size:
                            raise ProtocolError("shared sparse NNUE mapping is truncated")
                        (
                            shared_magic,
                            shared_version,
                            shared_request_id,
                            total_features,
                        ) = SHARED_HEADER.unpack_from(shared)
                        if (
                            shared_magic != SHARED_MAGIC
                            or shared_version != SHARED_VERSION
                            or shared_request_id != request_id
                        ):
                            raise ProtocolError("shared sparse NNUE request header did not match")
                        features_start = SHARED_HEADER.size + offsets_bytes
                        shared_response_offset = _align_to_four(features_start + total_features * 2)
                        required = shared_response_offset + count * 4
                        if required > len(shared):
                            raise ProtocolError(
                                f"shared sparse NNUE request requires {required} bytes, "
                                f"mapping has {len(shared)}"
                            )
                        offsets_source = shared
                        offsets_offset = SHARED_HEADER.size
                        features_source = shared
                        features_offset = features_start
                    else:
                        total_features = TOTAL_FEATURES.unpack(
                            _read_exact(input_stream, TOTAL_FEATURES.size)
                        )[0]
                        csr_payload = _read_exact(
                            input_stream,
                            offsets_bytes + total_features * 2,
                        )
                        offsets_source = csr_payload
                        offsets_offset = 0
                        features_source = csr_payload
                        features_offset = offsets_bytes
                    if total_features > count * MAX_SPARSE_FEATURES_PER_ROW:
                        raise ProtocolError(
                            f"exact sparse batch has {total_features} features, "
                            f"maximum is {count * MAX_SPARSE_FEATURES_PER_ROW}"
                        )
                    if timings_enabled:
                        stats["payload_read_ns"] += time.perf_counter_ns() - payload_started
                    decode_started = time.perf_counter_ns() if timings_enabled else 0
                    offsets = np.frombuffer(
                        offsets_source,
                        dtype="<u4",
                        count=count + 1,
                        offset=offsets_offset,
                    )
                    if offsets[0] != 0 or offsets[-1] != total_features:
                        raise ProtocolError("exact sparse offsets do not span the payload")
                    if np.any(offsets[1:] < offsets[:-1]):
                        raise ProtocolError("exact sparse offsets are not monotonic")
                    widths = offsets[1:] - offsets[:-1]
                    if np.any(widths > MAX_SPARSE_FEATURES_PER_ROW):
                        raise ProtocolError("exact sparse offsets contain an invalid row width")
                    features = np.frombuffer(
                        features_source,
                        dtype="<u2",
                        count=total_features,
                        offset=features_offset,
                    )
                    if np.any(features >= LEGACY_NNUE_FEATURES):
                        raise ProtocolError("exact sparse batch contains an out-of-range index")
                    exact_offsets = mx.array(offsets)
                    exact_features = mx.array(features)
                    if timings_enabled and (
                        largest_batch is None or count > len(largest_batch[0]) - 1
                    ):
                        largest_batch = (offsets.copy(), features.copy())
                    if timings_enabled:
                        stats["decode_validate_ns"] += time.perf_counter_ns() - decode_started
                    graph_started = time.perf_counter_ns() if timings_enabled else 0
                    if message_type == MESSAGE_PREDICT_SPARSE_NNUE_CSR_EXACT:
                        predictions = exact_model(exact_offsets, exact_features)
                        response_type = MESSAGE_SPARSE_NNUE_CSR_EXACT_PREDICTION
                    elif message_type == MESSAGE_PREDICT_SPARSE_NNUE_CSR_EXACT_HIDDEN:
                        hidden, values = exact_model.hidden_and_output(
                            exact_offsets,
                            exact_features,
                        )
                        predictions = mx.concatenate([hidden, values[:, None]], axis=1)
                        response_type = MESSAGE_SPARSE_NNUE_CSR_EXACT_HIDDEN_PREDICTION
                    else:
                        predictions = exact_model(exact_offsets, exact_features)
                        response_type = MESSAGE_SPARSE_NNUE_CSR_EXACT_SHARED_PREDICTION
                if timings_enabled:
                    stats["graph_build_ns"] += time.perf_counter_ns() - graph_started
                eval_started = time.perf_counter_ns() if timings_enabled else 0
                mx.eval(predictions)
                if timings_enabled:
                    evaluation_ns = time.perf_counter_ns() - eval_started
                    stats["mlx_eval_ns"] += evaluation_ns
                materialize_started = time.perf_counter_ns() if timings_enabled else 0
                values = np.asarray(predictions, dtype=np.float32)
                expected_shape = (
                    (count, LEGACY_NNUE_HIDDEN2 + 1)
                    if message_type == MESSAGE_PREDICT_SPARSE_NNUE_CSR_EXACT_HIDDEN
                    else (count,)
                )
                if values.shape != expected_shape or not np.all(np.isfinite(values)):
                    raise ProtocolError("sparse NNUE returned an invalid prediction tensor")
                if timings_enabled:
                    stats["materialize_validate_ns"] += time.perf_counter_ns() - materialize_started
                response_started = time.perf_counter_ns() if timings_enabled else 0
                value_bytes = values.astype("<f4", copy=False).tobytes(order="C")
                if shared_response_offset is not None:
                    shared = memoryview(shared_buffer)
                    shared[shared_response_offset : shared_response_offset + len(value_bytes)] = (
                        value_bytes
                    )
                output_stream.write(
                    FRAME_HEADER.pack(
                        PROTOCOL_MAGIC,
                        PROTOCOL_VERSION,
                        response_type,
                        request_id,
                        count,
                    )
                )
                if shared_response_offset is None:
                    output_stream.write(value_bytes)
                output_stream.flush()
                if timings_enabled:
                    now = time.perf_counter_ns()
                    stats["response_write_ns"] += now - response_started
                    stats["request_total_ns"] += now - request_started
                    stats["requests"] += 1
                    stats["rows"] += count
                    stats["features"] += total_features
                    bucket = request_size_buckets[_request_size_bucket_index(count)]
                    bucket["requests"] += 1
                    bucket["rows"] += count
                    bucket["features"] += total_features
                    bucket["mlx_eval_ns"] += evaluation_ns
            except Exception as error:
                message = str(error).encode()
                output_stream.write(
                    FRAME_HEADER.pack(
                        PROTOCOL_MAGIC,
                        PROTOCOL_VERSION,
                        MESSAGE_ERROR,
                        request_id,
                        len(message),
                    )
                )
                output_stream.write(message)
                output_stream.flush()
                if isinstance(error, (EOFError, ProtocolError)):
                    return
    finally:
        if timings_enabled:
            _emit_stage_timings(
                stats,
                request_size_buckets,
                largest_batch,
                exact_model,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--shared-memory", type=Path)
    args = parser.parse_args()
    model = LegacySparseNnue.load(args.model_dir)
    exact_model = LegacyRustExactSparseNnue(model.tensors)
    print(
        f"cascadia-mlx legacy NNUE serving {args.model_dir.name} on {mx.default_device()}",
        file=sys.stderr,
    )
    shared_file = args.shared_memory.open("r+b") if args.shared_memory is not None else None
    shared_buffer = (
        mmap.mmap(shared_file.fileno(), 0, access=mmap.ACCESS_WRITE)
        if shared_file is not None
        else None
    )
    try:
        serve_legacy_nnue(
            model,
            sys.stdin.buffer,
            sys.stdout.buffer,
            exact_model,
            shared_buffer,
        )
    finally:
        if shared_buffer is not None:
            shared_buffer.close()
        if shared_file is not None:
            shared_file.close()


if __name__ == "__main__":
    main()
