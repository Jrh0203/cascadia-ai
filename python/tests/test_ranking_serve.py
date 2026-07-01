from __future__ import annotations

import io
import struct

import mlx.core as mx
from cascadia_mlx.dataset import RECORD_SIZE
from cascadia_mlx.ranking_serve import MESSAGE_RANKING_PREDICTION, serve_ranking
from cascadia_mlx.serve import (
    FRAME_HEADER,
    MESSAGE_PREDICT,
    PROTOCOL_MAGIC,
    PROTOCOL_VERSION,
)


class ConstantRanker:
    def __call__(self, *_inputs: mx.array) -> mx.array:
        return mx.array([[2.5], [3.5]])


def test_ranking_service_returns_one_scalar_per_record() -> None:
    request = FRAME_HEADER.pack(
        PROTOCOL_MAGIC,
        PROTOCOL_VERSION,
        MESSAGE_PREDICT,
        9,
        2,
    ) + bytes(RECORD_SIZE * 2)
    output = io.BytesIO()
    serve_ranking(ConstantRanker(), io.BytesIO(request), output)
    payload = output.getvalue()
    header = FRAME_HEADER.unpack(payload[: FRAME_HEADER.size])

    assert header == (PROTOCOL_MAGIC, PROTOCOL_VERSION, MESSAGE_RANKING_PREDICTION, 9, 2)
    assert struct.unpack("<2f", payload[FRAME_HEADER.size :]) == (2.5, 3.5)
