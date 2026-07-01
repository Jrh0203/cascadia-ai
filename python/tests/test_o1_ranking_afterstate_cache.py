from __future__ import annotations

import numpy as np
from cascadia_mlx.o1_ranking_afterstate_cache import model_input_blake3
from cascadia_mlx.opponent_intent_dataset import OPPONENT_INTENT_RECORD_DTYPE


def test_model_input_hash_excludes_targets_and_game_identity() -> None:
    record = np.zeros((), dtype=OPPONENT_INTENT_RECORD_DTYPE)
    record["position"]["turn"] = 17
    record["position"]["active_seat"] = 2
    record["position"]["market_entities"][0, 0] = 3
    record["history_count"] = 1
    record["history"][0]["valid"] = 1
    record["history"][0]["age"] = 0
    baseline = model_input_blake3(record)

    hidden = record.copy()
    hidden["position"]["game_index"] = 123_456
    hidden["position"]["targets"] = 255

    assert model_input_blake3(hidden) == baseline


def test_model_input_hash_binds_public_state_and_history() -> None:
    record = np.zeros((), dtype=OPPONENT_INTENT_RECORD_DTYPE)
    record["history_count"] = 1
    record["history"][0]["valid"] = 1
    baseline = model_input_blake3(record)

    changed_market = record.copy()
    changed_market["position"]["market_entities"][0, 0] = 2
    changed_history = record.copy()
    changed_history["history"][0]["action"]["tile_slot"] = 1

    assert model_input_blake3(changed_market) != baseline
    assert model_input_blake3(changed_history) != baseline
