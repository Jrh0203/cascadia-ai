from __future__ import annotations

import json
import struct
from functools import cache
from itertools import product
from pathlib import Path

import blake3
import numpy as np
import pytest
from cascadia_mlx.r2_map_market_decision import (
    MARKET_ACTION_SCHEMA_BLAKE3,
    MARKET_DECISION_ACTION_SIZE,
    MARKET_DECISION_FEATURE_DIM,
    MarketDecisionAction,
    MarketDecisionActionKind,
    MarketDecisionContractError,
    MarketDecisionKind,
    _all_refill_branches_stabilize_oracle,
    _refill_is_universally_stabilizing,
    canonical_market_actions,
    decode_market_decision_action_bytes,
    market_decision_action_id,
    market_replacement_is_universally_feasible,
    validate_canonical_market_action_order,
)
from cascadia_mlx.r2_map_serve import (
    MARKET_REQUEST_SCHEMA,
    MARKET_REQUEST_SCHEMA_BLAKE3,
    MARKET_REQUEST_TENSOR_DTYPES,
    MARKET_RESPONSE_SCHEMA,
    MARKET_RESPONSE_SCHEMA_BLAKE3,
    MARKET_RESPONSE_TENSOR_DTYPES,
    ordered_action_ids_blake3,
)


def _rows(actions: list[MarketDecisionAction]) -> np.ndarray:
    return np.stack([np.frombuffer(action.to_bytes(), dtype=np.uint8) for action in actions])


def _free(*, replace: bool = True) -> np.ndarray:
    actions = [
        MarketDecisionAction(
            MarketDecisionKind.FREE_THREE_OF_A_KIND, MarketDecisionActionKind.KEEP
        )
    ]
    if replace:
        actions.append(
            MarketDecisionAction(
                MarketDecisionKind.FREE_THREE_OF_A_KIND,
                MarketDecisionActionKind.REPLACE,
            )
        )
    return _rows(actions)


def _paid(masks: list[int]) -> np.ndarray:
    return _rows(
        [MarketDecisionAction(MarketDecisionKind.PAID_WIPES, MarketDecisionActionKind.STOP)]
        + [
            MarketDecisionAction(
                MarketDecisionKind.PAID_WIPES,
                MarketDecisionActionKind.PAID_WIPE,
                mask,
            )
            for mask in masks
        ]
    )


def test_canonical_bytes_and_features_are_frozen() -> None:
    action = MarketDecisionAction(
        MarketDecisionKind.PAID_WIPES,
        MarketDecisionActionKind.PAID_WIPE,
        0b1011,
    )
    encoded = action.to_bytes()
    assert len(encoded) == MARKET_DECISION_ACTION_SIZE == 8
    assert encoded == struct.pack("<BBBBI", 1, 1, 3, 0b1011, 0)
    assert MarketDecisionAction.from_bytes(encoded) == action
    features = action.features()
    assert features.dtype == np.float32
    assert features.shape == (MARKET_DECISION_FEATURE_DIM,)
    np.testing.assert_array_equal(features[:6], [0, 1, 0, 0, 0, 1])
    np.testing.assert_array_equal(features[6:10], [1, 1, 0, 1])
    np.testing.assert_array_equal(features[10:14], [0, 0, 1, 0])
    np.testing.assert_array_equal(features[14:], [1, 0])
    np.testing.assert_array_equal(
        decode_market_decision_action_bytes(_rows([action]))[0], features
    )


@pytest.mark.parametrize(
    "encoded",
    [
        b"short",
        struct.pack("<BBBBI", 2, 1, 3, 1, 0),
        struct.pack("<BBBBI", 1, 1, 3, 1, 9),
        struct.pack("<BBBBI", 1, 9, 3, 1, 0),
        struct.pack("<BBBBI", 1, 1, 3, 0, 0),
        struct.pack("<BBBBI", 1, 0, 0, 1, 0),
    ],
)
def test_noncanonical_or_illegal_action_bytes_fail_closed(encoded: bytes) -> None:
    with pytest.raises(MarketDecisionContractError):
        MarketDecisionAction.from_bytes(encoded)


def test_exact_public_legal_screens_cover_normal_and_exhaustion_states() -> None:
    assert len(
        validate_canonical_market_action_order(
            _free(),
            decision_kind=MarketDecisionKind.FREE_THREE_OF_A_KIND,
            public_nature_tokens=0,
            public_wildlife_bag_total=10,
            public_wildlife_bag_counts=(2, 2, 2, 2, 2),
            public_market_wildlife=(0, 0, 0, 1),
        )
    ) == 2
    assert len(
        validate_canonical_market_action_order(
            _free(replace=False),
            decision_kind=MarketDecisionKind.FREE_THREE_OF_A_KIND,
            public_nature_tokens=5,
            public_wildlife_bag_total=3,
            public_wildlife_bag_counts=(0, 3, 0, 0, 0),
            public_market_wildlife=(0, 0, 0, 1),
        )
    ) == 1
    assert len(
        validate_canonical_market_action_order(
            _paid(list(range(1, 16))),
            decision_kind=MarketDecisionKind.PAID_WIPES,
            public_nature_tokens=1,
            public_wildlife_bag_total=10,
            public_wildlife_bag_counts=(2, 2, 2, 2, 2),
            public_market_wildlife=(0, 1, 2, 3),
        )
    ) == 16
    for tokens, bag, counts in ((0, 10, (2, 2, 2, 2, 2)), (1, 0, (0, 0, 0, 0, 0))):
        assert len(
            validate_canonical_market_action_order(
                _paid([]),
                decision_kind=MarketDecisionKind.PAID_WIPES,
                public_nature_tokens=tokens,
                public_wildlife_bag_total=bag,
                public_wildlife_bag_counts=counts,
                public_market_wildlife=(0, 1, 2, 3),
            )
        ) == 1


@pytest.mark.parametrize(
    "rows",
    [
        _paid([1, 3]),
        _paid(list(range(1, 15))),
        _paid([2, 1, *range(3, 16)]),
        np.concatenate([_paid([1]), _paid([1])], axis=0),
        np.concatenate([_paid([]), _free(replace=False)], axis=0),
    ],
)
def test_partial_duplicate_reordered_or_mixed_paid_screens_fail_closed(
    rows: np.ndarray,
) -> None:
    with pytest.raises(MarketDecisionContractError):
        validate_canonical_market_action_order(
            rows,
            decision_kind=MarketDecisionKind.PAID_WIPES,
            public_nature_tokens=1,
            public_wildlife_bag_total=10,
            public_wildlife_bag_counts=(2, 2, 2, 2, 2),
            public_market_wildlife=(0, 1, 2, 3),
        )


def test_public_universal_legality_excludes_any_hidden_exhaustion_branch() -> None:
    assert not market_replacement_is_universally_feasible(
        public_market_wildlife=(0, 0, 0, 1),
        public_wildlife_bag_counts=(0, 3, 0, 0, 0),
        slot_mask=0b0111,
    )
    assert market_replacement_is_universally_feasible(
        public_market_wildlife=(0, 0, 0, 1),
        public_wildlife_bag_counts=(1, 2, 0, 0, 0),
        slot_mask=0b0111,
    )
    actions = canonical_market_actions(
        decision_kind=MarketDecisionKind.PAID_WIPES,
        public_nature_tokens=1,
        public_wildlife_bag_total=4,
        public_wildlife_bag_counts=(4, 0, 0, 0, 0),
        public_market_wildlife=(0, 1, 2, 3),
    )
    masks = [
        action.slot_mask
        for action in actions
        if action.action_kind is MarketDecisionActionKind.PAID_WIPE
    ]
    assert 0b1110 not in masks
    assert masks == sorted(masks)


def _brute_universal(
    market: tuple[int, int, int, int],
    bag: tuple[int, int, int, int, int],
    mask: int,
) -> bool:
    remaining = tuple(
        sum(
            1
            for slot, species_at_slot in enumerate(market)
            if mask & (1 << slot) == 0 and species_at_slot == species
        )
        for species in range(5)
    )

    @cache
    def resolve(
        visible: tuple[int, int, int, int, int],
        counts: tuple[int, int, int, int, int],
        needed: int,
    ) -> bool:
        sequences = {
            sequence
            for sequence in product(range(5), repeat=needed)
            if all(sequence.count(species) <= counts[species] for species in range(5))
        }
        if not sequences:
            return False
        for sequence in sequences:
            drawn = tuple(sequence.count(species) for species in range(5))
            filled = tuple(
                left + right for left, right in zip(visible, drawn, strict=True)
            )
            if max(filled) < 4:
                continue
            next_counts = tuple(
                left - right for left, right in zip(counts, drawn, strict=True)
            )
            if not resolve((0, 0, 0, 0, 0), next_counts, 4):
                return False
        return True

    return resolve(remaining, bag, mask.bit_count())


def test_dynamic_program_matches_independent_ordered_hidden_bag_enumeration() -> None:
    cases = (
        ((0, 0, 0, 1), (0, 3, 0, 0, 0)),
        ((0, 0, 0, 1), (1, 2, 0, 0, 0)),
        ((0, 1, 2, 3), (4, 0, 0, 0, 0)),
        ((0, 1, 2, 3), (4, 4, 1, 0, 0)),
        ((0, 1, 2, 3), (2, 2, 2, 2, 2)),
    )
    for market, bag in cases:
        for mask in range(1, 16):
            expected = _brute_universal(market, bag, mask)
            actual = market_replacement_is_universally_feasible(
                public_market_wildlife=market,
                public_wildlife_bag_counts=bag,
                slot_mask=mask,
            )
            assert actual is expected, (market, bag, mask)


def test_closed_form_exhaustively_matches_recursive_multiset_oracle() -> None:
    retained_states = tuple(
        retained
        for retained in product(range(4), repeat=5)
        if sum(retained) < 4
    )
    bag_states = tuple(product(range(5), repeat=5))
    _all_refill_branches_stabilize_oracle.cache_clear()
    for bag in bag_states:
        for retained in retained_states:
            needed = 4 - sum(retained)
            expected = _all_refill_branches_stabilize_oracle(
                retained,
                bag,
                needed,
            )
            actual = _refill_is_universally_stabilizing(retained, bag)
            assert actual is expected, (retained, bag)
    _all_refill_branches_stabilize_oracle.cache_clear()


def test_every_deletion_duplication_and_reordering_of_full_screen_fails() -> None:
    full = _paid(list(range(1, 16)))
    metadata = {
        "decision_kind": MarketDecisionKind.PAID_WIPES,
        "public_nature_tokens": 1,
        "public_wildlife_bag_total": 10,
        "public_wildlife_bag_counts": (2, 2, 2, 2, 2),
        "public_market_wildlife": (0, 1, 2, 3),
    }
    for index in range(len(full)):
        with pytest.raises(MarketDecisionContractError):
            validate_canonical_market_action_order(np.delete(full, index, axis=0), **metadata)
        duplicated = np.insert(full, index, full[index], axis=0)
        with pytest.raises(MarketDecisionContractError):
            validate_canonical_market_action_order(duplicated, **metadata)
    for index in range(len(full) - 1):
        reordered = full.copy()
        reordered[[index, index + 1]] = reordered[[index + 1, index]]
        with pytest.raises(MarketDecisionContractError):
            validate_canonical_market_action_order(reordered, **metadata)


def test_cross_language_market_protocol_fixture_is_canonical_and_hash_bound() -> None:
    path = Path("tests/fixtures/r2_map/public-market-decision-protocol-v3.json")
    fixture = json.loads(path.read_text())
    fixture_hash = fixture.pop("fixture_blake3")
    assert fixture_hash == blake3.blake3(
        json.dumps(fixture, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert fixture["request_schema"] == MARKET_REQUEST_SCHEMA
    assert fixture["response_schema"] == MARKET_RESPONSE_SCHEMA
    assert fixture["request_schema_blake3"] == MARKET_REQUEST_SCHEMA_BLAKE3
    assert fixture["response_schema_blake3"] == MARKET_RESPONSE_SCHEMA_BLAKE3
    assert fixture["action_schema_blake3"] == MARKET_ACTION_SCHEMA_BLAKE3
    assert fixture["request_tensor_dtypes"] == MARKET_REQUEST_TENSOR_DTYPES
    assert fixture["response_tensor_dtypes"] == MARKET_RESPONSE_TENSOR_DTYPES
    for case in fixture["cases"]:
        rows = [bytes.fromhex(value) for value in case["action_bytes_hex"]]
        ids = [market_decision_action_id(case["decision_id"], row) for row in rows]
        validate_canonical_market_action_order(
            np.asarray([list(row) for row in rows], dtype=np.uint8),
            decision_kind=MarketDecisionKind(case["decision_kind"]),
            public_nature_tokens=case["public_nature_tokens"],
            public_wildlife_bag_total=case["public_wildlife_bag_total"],
            public_wildlife_bag_counts=case["public_wildlife_bag_counts"],
            public_market_wildlife=case["public_market_wildlife"],
        )
        assert ids == case["action_ids"]
        assert ordered_action_ids_blake3(ids) == case["ordered_action_ids_blake3"]
