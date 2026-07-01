from __future__ import annotations

import io
import struct

import mlx.core as mx
from cascadia_mlx.action_ranking_dataset import ACTION_POSITION_RECORD_SIZE
from cascadia_mlx.action_ranking_serve import (
    MESSAGE_ACTION_RANKING_PREDICTION,
    MESSAGE_PREDICT_ACTION_RANKING,
)
from cascadia_mlx.public_beam_set_serve import serve_public_beam_set
from cascadia_mlx.serve import FRAME_HEADER, PROTOCOL_MAGIC, PROTOCOL_VERSION


class ConstantPublicBeamSet:
    def __call__(self, *_inputs: mx.array) -> mx.array:
        return mx.array([[91.25, 92.5]])


def test_public_beam_set_service_treats_request_as_one_candidate_group() -> None:
    request = FRAME_HEADER.pack(
        PROTOCOL_MAGIC,
        PROTOCOL_VERSION,
        MESSAGE_PREDICT_ACTION_RANKING,
        12,
        2,
    ) + bytes(ACTION_POSITION_RECORD_SIZE * 2)
    output = io.BytesIO()
    serve_public_beam_set(ConstantPublicBeamSet(), io.BytesIO(request), output)
    payload = output.getvalue()
    header = FRAME_HEADER.unpack(payload[: FRAME_HEADER.size])

    assert header == (
        PROTOCOL_MAGIC,
        PROTOCOL_VERSION,
        MESSAGE_ACTION_RANKING_PREDICTION,
        12,
        2,
    )
    assert struct.unpack("<2f", payload[FRAME_HEADER.size :]) == (91.25, 92.5)
