from __future__ import annotations

import io
import struct
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from cascadia_mlx.checkpoint import TrainerState, save_checkpoint
from cascadia_mlx.dataset import RECORD_SIZE
from cascadia_mlx.model import EntitySetValueModel, ModelConfig
from cascadia_mlx.serve import (
    FRAME_HEADER,
    MESSAGE_PREDICT,
    MESSAGE_PREDICTION,
    MESSAGE_SHUTDOWN,
    PROTOCOL_MAGIC,
    PROTOCOL_VERSION,
    serve,
)


def test_service_returns_one_decomposed_prediction(tmp_path: Path) -> None:
    mx.random.seed(3)
    model = EntitySetValueModel(
        ModelConfig(hidden_dim=32, attention_heads=4, board_blocks=0, market_blocks=0)
    )
    optimizer = optim.AdamW(1e-3)
    save_checkpoint(tmp_path, model, optimizer, TrainerState())

    request_id = 42
    request = (
        FRAME_HEADER.pack(
            PROTOCOL_MAGIC,
            PROTOCOL_VERSION,
            MESSAGE_PREDICT,
            request_id,
            1,
        )
        + bytes(RECORD_SIZE)
        + FRAME_HEADER.pack(
            PROTOCOL_MAGIC,
            PROTOCOL_VERSION,
            MESSAGE_SHUTDOWN,
            request_id + 1,
            0,
        )
    )
    output = io.BytesIO()
    serve(model, io.BytesIO(request), output)
    response = output.getvalue()
    header = FRAME_HEADER.unpack(response[: FRAME_HEADER.size])
    predictions = np.frombuffer(response[FRAME_HEADER.size :], dtype="<f4")

    assert header == (
        PROTOCOL_MAGIC,
        PROTOCOL_VERSION,
        MESSAGE_PREDICTION,
        request_id,
        1,
    )
    assert predictions.shape == (11,)
    assert np.isfinite(predictions).all()


def test_frame_header_has_cross_language_width() -> None:
    assert FRAME_HEADER.size == 16
    assert struct.calcsize("<4sHHII") == 16
