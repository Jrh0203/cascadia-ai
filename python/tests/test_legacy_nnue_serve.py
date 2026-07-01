from __future__ import annotations

import io

import mlx.core as mx
import numpy as np
from cascadia_mlx.legacy_nnue_serve import (
    FRAME_HEADER,
    MESSAGE_ERROR,
    MESSAGE_PREDICT_SPARSE_NNUE,
    MESSAGE_PREDICT_SPARSE_NNUE_CSR_EXACT,
    MESSAGE_PREDICT_SPARSE_NNUE_CSR_EXACT_HIDDEN,
    MESSAGE_PREDICT_SPARSE_NNUE_CSR_EXACT_SHARED,
    MESSAGE_SHUTDOWN,
    MESSAGE_SPARSE_NNUE_CSR_EXACT_HIDDEN_PREDICTION,
    MESSAGE_SPARSE_NNUE_CSR_EXACT_PREDICTION,
    MESSAGE_SPARSE_NNUE_CSR_EXACT_SHARED_PREDICTION,
    MESSAGE_SPARSE_NNUE_PREDICTION,
    PROTOCOL_MAGIC,
    PROTOCOL_VERSION,
    ROW_LENGTH,
    SHARED_HEADER,
    SHARED_MAGIC,
    SHARED_VERSION,
    TOTAL_FEATURES,
    _activation_report,
    _request_size_bucket_index,
    serve_legacy_nnue,
)


class _SumSparseModel:
    def __call__(self, indices: mx.array, mask: mx.array) -> mx.array:
        return mx.sum(indices.astype(mx.float32) * mask.astype(mx.float32), axis=1)


class _SumCsrModel:
    def __call__(self, offsets: mx.array, indices: mx.array) -> mx.array:
        assert offsets.dtype == mx.uint32
        assert indices.dtype == mx.uint16
        host_offsets = np.asarray(offsets)
        host_indices = np.asarray(indices)
        return mx.array(
            [
                float(host_indices[host_offsets[row] : host_offsets[row + 1]].sum())
                for row in range(len(host_offsets) - 1)
            ],
            dtype=mx.float32,
        )

    def hidden_and_output(
        self,
        offsets: mx.array,
        indices: mx.array,
    ) -> tuple[mx.array, mx.array]:
        values = self(offsets, indices)
        hidden = mx.broadcast_to(values[:, None], (values.shape[0], 64))
        return hidden, values

class _ActivationCsrModel:
    def all_hidden_and_output(
        self,
        offsets: mx.array,
        indices: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        del offsets, indices
        h1 = mx.array([[1.0, 0.0, -1.0, 2.0]], dtype=mx.float32)
        h2 = mx.array([[0.0, 3.0]], dtype=mx.float32)
        return h1, h2, mx.array([4.0], dtype=mx.float32)


def test_activation_report_counts_positive_hidden_values() -> None:
    report = _activation_report(
        _ActivationCsrModel(),
        np.asarray([0, 1], dtype=np.uint32),
        np.asarray([7], dtype=np.uint16),
    )

    assert report == {
        "largest_batch_h1_positive": 2,
        "largest_batch_h1_total": 4,
        "largest_batch_h1_density": 0.5,
        "largest_batch_h2_positive": 1,
        "largest_batch_h2_total": 2,
        "largest_batch_h2_density": 0.5,
    }


def test_request_size_buckets_cover_boundaries() -> None:
    assert [_request_size_bucket_index(rows) for rows in (1, 32, 33, 64)] == [0, 0, 1, 1]
    assert [_request_size_bucket_index(rows) for rows in (65, 128, 129, 256)] == [2, 2, 3, 3]
    assert [_request_size_bucket_index(rows) for rows in (257, 512, 513, 1_024)] == [4, 4, 5, 5]
    assert _request_size_bucket_index(1_025) == 6


def _request(request_id: int, rows: list[list[int]]) -> bytes:
    payload = bytearray(
        FRAME_HEADER.pack(
            PROTOCOL_MAGIC,
            PROTOCOL_VERSION,
            MESSAGE_PREDICT_SPARSE_NNUE,
            request_id,
            len(rows),
        )
    )
    for row in rows:
        payload.extend(ROW_LENGTH.pack(len(row)))
        payload.extend(np.asarray(row, dtype="<u2").tobytes())
    return bytes(payload)


def _exact_request(
    request_id: int,
    rows: list[list[int]],
    *,
    message_type: int = MESSAGE_PREDICT_SPARSE_NNUE_CSR_EXACT,
) -> bytes:
    offsets = [0]
    features = []
    for row in rows:
        features.extend(row)
        offsets.append(len(features))
    return (
        FRAME_HEADER.pack(
            PROTOCOL_MAGIC,
            PROTOCOL_VERSION,
            message_type,
            request_id,
            len(rows),
        )
        + TOTAL_FEATURES.pack(len(features))
        + np.asarray(offsets, dtype="<u4").tobytes()
        + np.asarray(features, dtype="<u2").tobytes()
    )


def test_sparse_service_preserves_empty_rows_order_and_duplicates() -> None:
    request_id = 31
    request = _request(request_id, [[], [1, 1, 2], [0, 11_230]]) + FRAME_HEADER.pack(
        PROTOCOL_MAGIC,
        PROTOCOL_VERSION,
        MESSAGE_SHUTDOWN,
        request_id + 1,
        0,
    )
    output = io.BytesIO()
    serve_legacy_nnue(_SumSparseModel(), io.BytesIO(request), output)
    response = output.getvalue()

    assert FRAME_HEADER.unpack(response[: FRAME_HEADER.size]) == (
        PROTOCOL_MAGIC,
        PROTOCOL_VERSION,
        MESSAGE_SPARSE_NNUE_PREDICTION,
        request_id,
        3,
    )
    assert np.array_equal(
        np.frombuffer(response[FRAME_HEADER.size :], dtype="<f4"),
        np.asarray([0.0, 4.0, 11_230.0], dtype=np.float32),
    )


def test_sparse_service_rejects_out_of_range_feature_with_typed_error() -> None:
    output = io.BytesIO()
    serve_legacy_nnue(
        _SumSparseModel(),
        io.BytesIO(_request(9, [[11_231]])),
        output,
    )
    response = output.getvalue()
    header = FRAME_HEADER.unpack(response[: FRAME_HEADER.size])
    message = response[FRAME_HEADER.size :].decode()

    assert header[:4] == (
        PROTOCOL_MAGIC,
        PROTOCOL_VERSION,
        MESSAGE_ERROR,
        9,
    )
    assert header[4] == len(message)
    assert "out-of-range" in message


def test_sparse_service_rejects_truncated_row() -> None:
    request = (
        FRAME_HEADER.pack(
            PROTOCOL_MAGIC,
            PROTOCOL_VERSION,
            MESSAGE_PREDICT_SPARSE_NNUE,
            17,
            1,
        )
        + ROW_LENGTH.pack(2)
        + np.asarray([3], dtype="<u2").tobytes()
    )
    output = io.BytesIO()
    serve_legacy_nnue(_SumSparseModel(), io.BytesIO(request), output)
    response = output.getvalue()
    header = FRAME_HEADER.unpack(response[: FRAME_HEADER.size])

    assert header[2] == MESSAGE_ERROR
    assert b"ended inside a frame" in response[FRAME_HEADER.size :]


def test_exact_sparse_service_preserves_empty_rows_order_and_duplicates() -> None:
    request_id = 47
    request = _exact_request(request_id, [[], [1, 1, 2], [0, 11_230]]) + FRAME_HEADER.pack(
        PROTOCOL_MAGIC,
        PROTOCOL_VERSION,
        MESSAGE_SHUTDOWN,
        request_id + 1,
        0,
    )
    output = io.BytesIO()
    serve_legacy_nnue(
        _SumSparseModel(),
        io.BytesIO(request),
        output,
        exact_model=_SumCsrModel(),
    )
    response = output.getvalue()

    assert FRAME_HEADER.unpack(response[: FRAME_HEADER.size]) == (
        PROTOCOL_MAGIC,
        PROTOCOL_VERSION,
        MESSAGE_SPARSE_NNUE_CSR_EXACT_PREDICTION,
        request_id,
        3,
    )
    assert np.array_equal(
        np.frombuffer(response[FRAME_HEADER.size :], dtype="<f4"),
        np.asarray([0.0, 4.0, 11_230.0], dtype=np.float32),
    )


def test_exact_hidden_service_returns_hidden_then_bit_identical_value() -> None:
    request_id = 53
    rows = [[], [1, 1, 2], [0, 11_230]]
    request = _exact_request(
        request_id,
        rows,
        message_type=MESSAGE_PREDICT_SPARSE_NNUE_CSR_EXACT_HIDDEN,
    ) + FRAME_HEADER.pack(
        PROTOCOL_MAGIC,
        PROTOCOL_VERSION,
        MESSAGE_SHUTDOWN,
        request_id + 1,
        0,
    )
    output = io.BytesIO()
    serve_legacy_nnue(
        _SumSparseModel(),
        io.BytesIO(request),
        output,
        exact_model=_SumCsrModel(),
    )
    response = output.getvalue()

    assert FRAME_HEADER.unpack(response[: FRAME_HEADER.size]) == (
        PROTOCOL_MAGIC,
        PROTOCOL_VERSION,
        MESSAGE_SPARSE_NNUE_CSR_EXACT_HIDDEN_PREDICTION,
        request_id,
        3,
    )
    values = np.frombuffer(response[FRAME_HEADER.size :], dtype="<f4").reshape(3, 65)
    expected = np.asarray([0.0, 4.0, 11_230.0], dtype=np.float32)
    assert np.array_equal(values[:, :64], np.broadcast_to(expected[:, None], (3, 64)))
    assert np.array_equal(values[:, 64], expected)


def test_exact_shared_service_reads_and_writes_only_the_mapping_payload() -> None:
    request_id = 59
    rows = [[], [1, 1, 2], [0, 11_230]]
    offsets = np.asarray([0, 0, 3, 5], dtype="<u4")
    features = np.asarray([1, 1, 2, 0, 11_230], dtype="<u2")
    features_start = SHARED_HEADER.size + offsets.nbytes
    response_offset = (features_start + features.nbytes + 3) & ~3
    shared = bytearray(response_offset + len(rows) * 4)
    SHARED_HEADER.pack_into(
        shared,
        0,
        SHARED_MAGIC,
        SHARED_VERSION,
        request_id,
        len(features),
    )
    shared[SHARED_HEADER.size : features_start] = offsets.tobytes()
    shared[features_start : features_start + features.nbytes] = features.tobytes()
    request = FRAME_HEADER.pack(
        PROTOCOL_MAGIC,
        PROTOCOL_VERSION,
        MESSAGE_PREDICT_SPARSE_NNUE_CSR_EXACT_SHARED,
        request_id,
        len(rows),
    ) + FRAME_HEADER.pack(
        PROTOCOL_MAGIC,
        PROTOCOL_VERSION,
        MESSAGE_SHUTDOWN,
        request_id + 1,
        0,
    )
    output = io.BytesIO()

    serve_legacy_nnue(
        _SumSparseModel(),
        io.BytesIO(request),
        output,
        exact_model=_SumCsrModel(),
        shared_buffer=shared,
    )

    assert output.getvalue() == FRAME_HEADER.pack(
        PROTOCOL_MAGIC,
        PROTOCOL_VERSION,
        MESSAGE_SPARSE_NNUE_CSR_EXACT_SHARED_PREDICTION,
        request_id,
        len(rows),
    )
    assert np.array_equal(
        np.frombuffer(shared, dtype="<f4", count=len(rows), offset=response_offset),
        np.asarray([0.0, 4.0, 11_230.0], dtype=np.float32),
    )


def test_exact_sparse_service_rejects_non_monotonic_offsets() -> None:
    request = (
        FRAME_HEADER.pack(
            PROTOCOL_MAGIC,
            PROTOCOL_VERSION,
            MESSAGE_PREDICT_SPARSE_NNUE_CSR_EXACT,
            61,
            2,
        )
        + TOTAL_FEATURES.pack(1)
        + np.asarray([0, 1, 0], dtype="<u4").tobytes()
        + np.asarray([3], dtype="<u2").tobytes()
    )
    output = io.BytesIO()
    serve_legacy_nnue(
        _SumSparseModel(),
        io.BytesIO(request),
        output,
        exact_model=_SumCsrModel(),
    )
    response = output.getvalue()

    assert FRAME_HEADER.unpack(response[: FRAME_HEADER.size])[2] == MESSAGE_ERROR
    assert b"do not span" in response[FRAME_HEADER.size :]
