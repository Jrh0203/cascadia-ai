from __future__ import annotations

import io
import struct

import mlx.core as mx
from cascadia_mlx.action_ranking_dataset import ACTION_POSITION_RECORD_SIZE
from cascadia_mlx.action_ranking_serve import (
    MESSAGE_ACTION_RANKING_PREDICTION,
    MESSAGE_PREDICT_ACTION_RANKING,
    serve_action_ranking,
)
from cascadia_mlx.serve import FRAME_HEADER, PROTOCOL_MAGIC, PROTOCOL_VERSION


class ConstantActionRanker:
    def __call__(self, *_inputs: mx.array) -> mx.array:
        return mx.array([[2.5], [3.5]])


def test_action_ranking_service_returns_one_scalar_per_action_record() -> None:
    request = FRAME_HEADER.pack(
        PROTOCOL_MAGIC,
        PROTOCOL_VERSION,
        MESSAGE_PREDICT_ACTION_RANKING,
        9,
        2,
    ) + bytes(ACTION_POSITION_RECORD_SIZE * 2)
    output = io.BytesIO()
    serve_action_ranking(ConstantActionRanker(), io.BytesIO(request), output)
    payload = output.getvalue()
    header = FRAME_HEADER.unpack(payload[: FRAME_HEADER.size])

    assert header == (
        PROTOCOL_MAGIC,
        PROTOCOL_VERSION,
        MESSAGE_ACTION_RANKING_PREDICTION,
        9,
        2,
    )
    assert struct.unpack("<2f", payload[FRAME_HEADER.size :]) == (2.5, 3.5)
