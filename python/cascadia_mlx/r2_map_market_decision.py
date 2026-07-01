"""Frozen public market-decision action contract for R2-MAP v1.1.

The bytes describe only the decision that is available before a stochastic
market refill.  They never contain the realized refill, bag order, seed, game
identity, policy identity, or any other future/hidden value.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from functools import cache
from itertools import product
from typing import Final

import blake3
import numpy as np

MARKET_DECISION_ACTION_SCHEMA_VERSION: Final = 1
MARKET_DECISION_ACTION_SIZE: Final = 8
MARKET_DECISION_FEATURE_DIM: Final = 16
_ACTION = struct.Struct("<BBBBI")
MARKET_ACTION_SCHEMA_BLAKE3: Final = blake3.blake3(
    json.dumps(
        {
            "schema": "r2-map-public-market-action-v1",
            "layout": "<BBBBI",
            "bytes": MARKET_DECISION_ACTION_SIZE,
            "features": MARKET_DECISION_FEATURE_DIM,
            "decision_kinds": {"free-three-of-a-kind": 0, "paid-wipes": 1},
            "action_kinds": {"keep": 0, "replace": 1, "stop": 2, "paid-wipe": 3},
            "paid_wipe_order": "stop-then-ascending-nonempty-four-slot-mask",
            "future_refill_fields": 0,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()


class MarketDecisionContractError(ValueError):
    """A public market-decision action or ordered legal set is invalid."""


class MarketDecisionKind(IntEnum):
    FREE_THREE_OF_A_KIND = 0
    PAID_WIPES = 1


class MarketDecisionActionKind(IntEnum):
    KEEP = 0
    REPLACE = 1
    STOP = 2
    PAID_WIPE = 3


@dataclass(frozen=True, slots=True)
class MarketDecisionAction:
    decision_kind: MarketDecisionKind
    action_kind: MarketDecisionActionKind
    slot_mask: int = 0

    def validate(self) -> None:
        if not isinstance(self.decision_kind, MarketDecisionKind) or not isinstance(
            self.action_kind, MarketDecisionActionKind
        ):
            raise MarketDecisionContractError("market decision enum identity differs")
        if (
            not isinstance(self.slot_mask, int)
            or isinstance(self.slot_mask, bool)
            or not 0 <= self.slot_mask <= 0x0F
        ):
            raise MarketDecisionContractError("market decision slot mask is outside four slots")
        if self.decision_kind is MarketDecisionKind.FREE_THREE_OF_A_KIND:
            if self.action_kind not in {
                MarketDecisionActionKind.KEEP,
                MarketDecisionActionKind.REPLACE,
            } or self.slot_mask != 0:
                raise MarketDecisionContractError("free replacement action semantics differ")
        elif self.action_kind is MarketDecisionActionKind.STOP:
            if self.slot_mask != 0:
                raise MarketDecisionContractError("paid-wipe Stop must have an empty mask")
        elif self.action_kind is MarketDecisionActionKind.PAID_WIPE:
            if self.slot_mask == 0:
                raise MarketDecisionContractError("paid wipe must select a nonempty subset")
        else:
            raise MarketDecisionContractError("paid-wipe decision action semantics differ")

    def to_bytes(self) -> bytes:
        self.validate()
        return _ACTION.pack(
            MARKET_DECISION_ACTION_SCHEMA_VERSION,
            int(self.decision_kind),
            int(self.action_kind),
            self.slot_mask,
            0,
        )

    @classmethod
    def from_bytes(cls, value: bytes | bytearray | memoryview) -> MarketDecisionAction:
        raw = bytes(value)
        if len(raw) != MARKET_DECISION_ACTION_SIZE:
            raise MarketDecisionContractError("market decision action byte width differs")
        version, decision, action, mask, reserved = _ACTION.unpack(raw)
        if version != MARKET_DECISION_ACTION_SCHEMA_VERSION or reserved != 0:
            raise MarketDecisionContractError(
                "market decision action schema or reserved bytes differ"
            )
        try:
            result = cls(MarketDecisionKind(decision), MarketDecisionActionKind(action), mask)
        except ValueError as error:
            raise MarketDecisionContractError("market decision action enum is unknown") from error
        result.validate()
        if result.to_bytes() != raw:
            raise MarketDecisionContractError("market decision action is not canonically encoded")
        return result

    def features(self) -> np.ndarray:
        """Return the frozen 16-d public semantic feature row."""
        self.validate()
        result = np.zeros(MARKET_DECISION_FEATURE_DIM, dtype=np.float32)
        result[int(self.decision_kind)] = 1.0
        result[2 + int(self.action_kind)] = 1.0
        for slot in range(4):
            result[6 + slot] = float(bool(self.slot_mask & (1 << slot)))
        cardinality = self.slot_mask.bit_count()
        if cardinality:
            result[10 + cardinality - 1] = 1.0
        paid = self.action_kind is MarketDecisionActionKind.PAID_WIPE
        result[14] = float(paid)
        result[15] = float(
            self.action_kind
            in {MarketDecisionActionKind.KEEP, MarketDecisionActionKind.STOP}
        )
        return result


def decode_market_decision_action_bytes(rows: np.ndarray) -> np.ndarray:
    """Decode canonical ``[..., 8]`` uint8 rows to frozen float32 features."""
    values = np.asarray(rows)
    if values.dtype != np.uint8 or values.ndim < 1 or values.shape[-1] != 8:
        raise MarketDecisionContractError("market decision action rows must be uint8 [..., 8]")
    flat = values.reshape(-1, MARKET_DECISION_ACTION_SIZE)
    if not len(flat):
        raise MarketDecisionContractError("market decision action rows cannot be empty")
    decoded = np.stack(
        [MarketDecisionAction.from_bytes(row.tobytes()).features() for row in flat], axis=0
    )
    return decoded.reshape(*values.shape[:-1], MARKET_DECISION_FEATURE_DIM)


def market_decision_action_id(decision_id: str, action_bytes: bytes) -> str:
    """Bind semantic action bytes to one immutable public decision identity."""
    if (
        not isinstance(decision_id, str)
        or len(decision_id) != 64
        or any(character not in "0123456789abcdef" for character in decision_id)
    ):
        raise MarketDecisionContractError("market decision identity must be a BLAKE3 digest")
    canonical = MarketDecisionAction.from_bytes(action_bytes).to_bytes()
    return blake3.blake3(
        b"r2-map-market-action-identity-v1" + bytes.fromhex(decision_id) + canonical
    ).hexdigest()


def validate_canonical_market_action_order(
    rows: np.ndarray,
    *,
    decision_kind: MarketDecisionKind,
    public_nature_tokens: int,
    public_wildlife_bag_total: int,
    public_wildlife_bag_counts: Sequence[int],
    public_market_wildlife: Sequence[int],
) -> tuple[MarketDecisionAction, ...]:
    """Reject anything but the exact public-universal legal action screen."""
    values = np.asarray(rows)
    if values.dtype != np.uint8 or values.ndim != 2 or values.shape[1] != 8 or not len(values):
        raise MarketDecisionContractError("market decision legal screen shape differs")
    actions = tuple(MarketDecisionAction.from_bytes(row.tobytes()) for row in values)
    if len({action.to_bytes() for action in actions}) != len(actions):
        raise MarketDecisionContractError("market decision legal screen repeats an action")
    kind = actions[0].decision_kind
    if any(action.decision_kind is not kind for action in actions):
        raise MarketDecisionContractError("market decision legal screen mixes decision stages")
    if kind is not decision_kind:
        raise MarketDecisionContractError("market decision kind metadata disagrees with actions")
    expected = canonical_market_actions(
        decision_kind=decision_kind,
        public_nature_tokens=public_nature_tokens,
        public_wildlife_bag_total=public_wildlife_bag_total,
        public_wildlife_bag_counts=public_wildlife_bag_counts,
        public_market_wildlife=public_market_wildlife,
    )
    if actions != expected:
        raise MarketDecisionContractError(
            "market decision screen differs from the complete public-universal legal set"
        )
    return actions


def canonical_market_actions(
    *,
    decision_kind: MarketDecisionKind,
    public_nature_tokens: int,
    public_wildlife_bag_total: int,
    public_wildlife_bag_counts: Sequence[int],
    public_market_wildlife: Sequence[int],
) -> tuple[MarketDecisionAction, ...]:
    """Derive the complete information-set legal screen from public counts only.

    Optional market replacements are legal only when every hidden bag order
    consistent with the public per-species counts reaches a stable market.
    This is a legality intersection across hidden states, not action pruning.
    ADR 0018/0078's rejection-conditioned teacher rollouts are deliberately
    not imported into direct gameplay.
    """
    bag, market = _validate_public_market_metadata(
        decision_kind=decision_kind,
        public_nature_tokens=public_nature_tokens,
        public_wildlife_bag_total=public_wildlife_bag_total,
        public_wildlife_bag_counts=public_wildlife_bag_counts,
        public_market_wildlife=public_market_wildlife,
    )
    if decision_kind is MarketDecisionKind.FREE_THREE_OF_A_KIND:
        actions = [
            MarketDecisionAction(decision_kind, MarketDecisionActionKind.KEEP)
        ]
        counts = tuple(market.count(species) for species in range(5))
        repeated = next((species for species, count in enumerate(counts) if count == 3), None)
        if repeated is None or sorted(count for count in counts if count) != [1, 3]:
            raise MarketDecisionContractError(
                "free replacement decision requires an exact public three-of-a-kind"
            )
        mask = sum(
            1 << slot for slot, species in enumerate(market) if species == repeated
        )
        if market_replacement_is_universally_feasible(
            public_market_wildlife=market,
            public_wildlife_bag_counts=bag,
            slot_mask=mask,
        ):
            actions.append(
                MarketDecisionAction(decision_kind, MarketDecisionActionKind.REPLACE)
            )
        return tuple(actions)

    if len(set(market)) == 1:
        raise MarketDecisionContractError(
            "paid-wipe decision cannot begin from an unresolved four-of-a-kind"
        )
    actions = [MarketDecisionAction(decision_kind, MarketDecisionActionKind.STOP)]
    if public_nature_tokens:
        actions.extend(
            MarketDecisionAction(
                decision_kind,
                MarketDecisionActionKind.PAID_WIPE,
                mask,
            )
            for mask in range(1, 16)
            if market_replacement_is_universally_feasible(
                public_market_wildlife=market,
                public_wildlife_bag_counts=bag,
                slot_mask=mask,
            )
        )
    return tuple(actions)


def market_replacement_is_universally_feasible(
    *,
    public_market_wildlife: Sequence[int],
    public_wildlife_bag_counts: Sequence[int],
    slot_mask: int,
) -> bool:
    """Return whether every public-consistent refill branch stabilizes."""
    market = _validate_species_vector(
        public_market_wildlife,
        length=4,
        upper=4,
        label="public market wildlife",
    )
    bag = _validate_species_vector(
        public_wildlife_bag_counts,
        length=5,
        upper=20,
        label="public wildlife bag counts",
    )
    if (
        not isinstance(slot_mask, int)
        or isinstance(slot_mask, bool)
        or not 1 <= slot_mask <= 15
    ):
        raise MarketDecisionContractError("market replacement mask is invalid")
    remaining = tuple(
        sum(
            1
            for slot, value in enumerate(market)
            if slot_mask & (1 << slot) == 0 and value == species
        )
        for species in range(5)
    )
    return _refill_is_universally_stabilizing(remaining, bag)


def _refill_is_universally_stabilizing(
    retained_market: tuple[int, ...],
    bag_counts: tuple[int, ...],
) -> bool:
    """Exact O(5) public-universal refill theorem used by live inference."""
    retained_total = sum(retained_market)
    if not 0 <= retained_total < 4:
        return False
    needed = 4 - retained_total
    bag_total = sum(bag_counts)
    if bag_total < needed:
        return False
    retained_species = tuple(
        species for species, count in enumerate(retained_market) if count
    )
    if len(retained_species) >= 2:
        return True
    if len(retained_species) == 1:
        species = retained_species[0]
        if bag_counts[species] < needed:
            return True
        remaining = list(bag_counts)
        remaining[species] -= needed
        return _empty_refill_is_universally_stabilizing(tuple(remaining))
    return _empty_refill_is_universally_stabilizing(bag_counts)


def _empty_refill_is_universally_stabilizing(bag_counts: tuple[int, ...]) -> bool:
    """Decide whether every hidden ordering eventually yields a mixed cohort."""
    total = sum(bag_counts)
    return total >= 4 and sum(count // 4 for count in bag_counts) < total // 4


@cache
def _all_refill_branches_stabilize_oracle(
    market_counts: tuple[int, ...],
    bag_counts: tuple[int, ...],
    needed: int,
) -> bool:
    """Independent recursive multiset oracle retained exclusively for tests."""
    if sum(market_counts) + needed != 4 or sum(bag_counts) < needed:
        return False
    draws = tuple(_draw_multisets_oracle(bag_counts, needed))
    if not draws:
        return False
    for draw in draws:
        filled = tuple(left + right for left, right in zip(market_counts, draw, strict=True))
        if max(filled) < 4:
            continue
        remaining = tuple(left - right for left, right in zip(bag_counts, draw, strict=True))
        if not _all_refill_branches_stabilize_oracle(
            (0, 0, 0, 0, 0), remaining, 4
        ):
            return False
    return True


def _draw_multisets_oracle(
    bag_counts: tuple[int, ...], needed: int
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        draw
        for draw in product(*(range(min(count, needed) + 1) for count in bag_counts))
        if sum(draw) == needed
    )


def _validate_public_market_metadata(
    *,
    decision_kind: MarketDecisionKind,
    public_nature_tokens: int,
    public_wildlife_bag_total: int,
    public_wildlife_bag_counts: Sequence[int],
    public_market_wildlife: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if not isinstance(decision_kind, MarketDecisionKind):
        raise MarketDecisionContractError("market decision kind metadata differs")
    if (
        not isinstance(public_nature_tokens, int)
        or isinstance(public_nature_tokens, bool)
        or not 0 <= public_nature_tokens <= 0xFF
        or not isinstance(public_wildlife_bag_total, int)
        or isinstance(public_wildlife_bag_total, bool)
        or not 0 <= public_wildlife_bag_total <= 100
    ):
        raise MarketDecisionContractError("public market resource count is invalid")
    bag = _validate_species_vector(
        public_wildlife_bag_counts,
        length=5,
        upper=20,
        label="public wildlife bag counts",
    )
    market = _validate_species_vector(
        public_market_wildlife,
        length=4,
        upper=4,
        label="public market wildlife",
    )
    if sum(bag) != public_wildlife_bag_total:
        raise MarketDecisionContractError("public wildlife bag total differs from species counts")
    if any(bag[species] + market.count(species) > 20 for species in range(5)):
        raise MarketDecisionContractError("public wildlife counts violate species conservation")
    return bag, market


def _validate_species_vector(
    values: Sequence[int], *, length: int, upper: int, label: str
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise MarketDecisionContractError(f"{label} is invalid")
    result = tuple(values)
    if len(result) != length or any(
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 0 <= value <= upper
        for value in result
    ):
        raise MarketDecisionContractError(f"{label} is invalid")
    return result
