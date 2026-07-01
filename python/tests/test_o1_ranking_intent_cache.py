from __future__ import annotations

import mlx.core as mx
import numpy as np
from cascadia_mlx.o1_ranking_intent_cache import (
    INTENT_FEATURE_DIM,
    REFILL_PROPOSALS,
    deterministic_weighted_draw,
    prediction_to_intent_vector,
    public_refill_proposals,
    stratified_derangement,
)
from cascadia_mlx.opponent_intent_dataset import OPPONENT_INTENT_RECORD_DTYPE
from cascadia_mlx.opponent_intent_model import OpponentIntentPrediction


def test_deterministic_weighted_draw_is_stable_and_respects_zero_mass() -> None:
    counts = np.asarray([0, 4, 0, 9], dtype=np.int64)
    identity = {
        "split": "train",
        "group_id": 41,
        "action_hash": bytes(range(32)),
        "proposal_index": 3,
        "kind": "tile",
    }

    first = deterministic_weighted_draw(counts, **identity)
    second = deterministic_weighted_draw(counts, **identity)

    assert first == second
    assert first in {1, 3}


def test_public_refill_proposals_fill_only_candidate_depletions() -> None:
    record = np.zeros((), dtype=OPPONENT_INTENT_RECORD_DTYPE)
    record["history_count"] = 1
    action = record["history"][0]["action"]
    action["tile_slot"] = 1
    action["wildlife_slot"] = 3
    market = record["position"]["market_entities"]
    market[:] = np.arange(market.size, dtype=np.uint8).reshape(market.shape)
    market[1, 0] = 255
    market[3, 3] = 255
    original = record.copy()
    catalog = np.asarray(
        [[index % 5, 255, 1 << (index % 5), index % 2] for index in range(75)],
        dtype=np.uint8,
    )

    proposals, archetypes, wildlife = public_refill_proposals(
        record,
        split="validation",
        group_id=90,
        action_hash=bytes(range(32)),
        tile_counts=np.ones(75, dtype=np.int64),
        wildlife_counts=np.asarray([1, 2, 3, 4, 5], dtype=np.int64),
        catalog_entities=catalog,
    )

    assert proposals.shape == (REFILL_PROPOSALS,)
    assert np.array_equal(record, original)
    assert np.all(archetypes < 75)
    assert np.all(wildlife < 5)
    for proposal, archetype, wildlife_type in zip(
        proposals,
        archetypes,
        wildlife,
        strict=True,
    ):
        changed = proposal["position"]["market_entities"]
        expected = original["position"]["market_entities"].copy()
        expected[1, [0, 1, 2, 4]] = catalog[archetype]
        expected[1, 5:] = 0
        expected[3, 3] = wildlife_type
        expected[3, 5:] = 0
        assert np.array_equal(changed, expected)
        assert np.array_equal(proposal["history"], original["history"])
        assert np.array_equal(
            proposal["position"]["board_entities"],
            original["position"]["board_entities"],
        )


def test_prediction_vector_has_frozen_width_and_probability_segments() -> None:
    batch = 2
    prediction = OpponentIntentPrediction(
        disposition_logits=mx.zeros((batch, 4, 4)),
        pair_survival_logits=mx.zeros((batch, 4, 2)),
        final_slot_logits=mx.zeros((batch, 4, 4)),
        tile_slot_logits=mx.zeros((batch, 3, 4)),
        wildlife_slot_logits=mx.zeros((batch, 3, 4)),
        draft_kind_logits=mx.zeros((batch, 3, 2)),
        drafted_wildlife_logits=mx.zeros((batch, 3, 5)),
        replace_three_logits=mx.zeros((batch, 3, 2)),
    )

    values = np.asarray(prediction_to_intent_vector(prediction))

    assert values.shape == (batch, INTENT_FEATURE_DIM)
    assert np.allclose(values[:, 0:16], 0.25)
    assert np.allclose(values[:, 16:20], 0.5)
    assert np.allclose(values[:, 20:60], 0.25)
    assert np.allclose(values[:, 60:63], 0.5)
    assert np.allclose(values[:, 63:78], 0.2)
    assert np.allclose(values[:, 78:81], 0.5)


def test_stratified_derangement_merges_singleton_phases_without_fixed_points() -> None:
    group_ids = np.arange(8, dtype=np.uint64)
    hashes = np.asarray(
        [
            list(bytes([index]) * 32)
            for index in range(len(group_ids))
        ],
        dtype=np.uint8,
    )
    raw_strata = np.asarray(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
            [2, 0, 0],
            [3, 0, 0],
            [3, 0, 0],
            [3, 0, 0],
        ],
        dtype=np.uint8,
    )

    donors, effective = stratified_derangement(
        split="train",
        group_ids=group_ids,
        action_hashes=hashes,
        raw_strata=raw_strata,
    )

    assert sorted(donors.tolist()) == list(range(len(group_ids)))
    assert np.all(donors != np.arange(len(group_ids)))
    assert np.array_equal(effective[donors], effective)
    assert effective[0, 0] == 1
    for stratum in np.unique(effective, axis=0):
        assert np.sum(np.all(effective == stratum, axis=1)) >= 2
