from __future__ import annotations

import io
import struct

import mlx.core as mx
from cascadia_mlx.dataset import RECORD_SIZE
from cascadia_mlx.imitation_dataset import PROPOSAL_ACTION_FEATURE_SIZE
from cascadia_mlx.imitation_serve import (
    MESSAGE_IMITATION_PREDICTION,
    MESSAGE_PREDICT_IMITATION,
    serve_imitation,
)
from cascadia_mlx.serve import FRAME_HEADER, PROTOCOL_MAGIC, PROTOCOL_VERSION


class ConstantImitationRanker:
    def __call__(self, *_inputs: mx.array) -> mx.array:
        return mx.array([[2.5, 3.5]])


def test_imitation_service_scores_actions_against_one_shared_state() -> None:
    request = (
        FRAME_HEADER.pack(
            PROTOCOL_MAGIC,
            PROTOCOL_VERSION,
            MESSAGE_PREDICT_IMITATION,
            9,
            2,
        )
        + bytes(RECORD_SIZE)
        + bytes(PROPOSAL_ACTION_FEATURE_SIZE * 2)
    )
    output = io.BytesIO()
    serve_imitation(ConstantImitationRanker(), io.BytesIO(request), output)
    payload = output.getvalue()
    header = FRAME_HEADER.unpack(payload[: FRAME_HEADER.size])

    assert header == (
        PROTOCOL_MAGIC,
        PROTOCOL_VERSION,
        MESSAGE_IMITATION_PREDICTION,
        9,
        2,
    )
    assert struct.unpack("<2f", payload[FRAME_HEADER.size :]) == (2.5, 3.5)
