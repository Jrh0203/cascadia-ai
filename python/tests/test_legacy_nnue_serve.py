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
    MESSAGE_SHUTDOWN,
    MESSAGE_SPARSE_NNUE_CSR_EXACT_HIDDEN_PREDICTION,
    MESSAGE_SPARSE_NNUE_CSR_EXACT_PREDICTION,
    MESSAGE_SPARSE_NNUE_PREDICTION,
    PROTOCOL_MAGIC,
    PROTOCOL_VERSION,
    ROW_LENGTH,
    TOTAL_FEATURES,
    serve_legacy_nnue,
)


class _SumSparseModel:
    def __call__(self, indices: mx.array, mask: mx.array) -> mx.array:
        return mx.sum(indices.astype(mx.float32) * mask.astype(mx.float32), axis=1)


class _SumCsrModel:
    def __call__(self, offsets: mx.array, indices: mx.array) -> mx.array:
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
