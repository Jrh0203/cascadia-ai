#!/usr/bin/env python3
"""Render or verify the frozen R2-MAP public market protocol fixture."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import blake3

REPOSITORY = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPOSITORY / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from cascadia_mlx.r2_map_market_decision import (  # noqa: E402
    MARKET_ACTION_SCHEMA_BLAKE3,
    MarketDecisionKind,
    canonical_market_actions,
    market_decision_action_id,
)
from cascadia_mlx.r2_map_serve import (  # noqa: E402
    MARKET_REQUEST_SCHEMA,
    MARKET_REQUEST_SCHEMA_BLAKE3,
    MARKET_REQUEST_TENSOR_DTYPES,
    MARKET_RESPONSE_SCHEMA,
    MARKET_RESPONSE_SCHEMA_BLAKE3,
    MARKET_RESPONSE_TENSOR_DTYPES,
    ordered_action_ids_blake3,
)

DEFAULT_FIXTURE = REPOSITORY / "tests/fixtures/r2_map/public-market-decision-protocol-v3.json"


def _case(
    name: str,
    decision_id: str,
    decision_kind: MarketDecisionKind,
    *,
    public_nature_tokens: int,
    public_wildlife_bag_counts: list[int],
    public_market_wildlife: list[int],
) -> dict[str, Any]:
    total = sum(public_wildlife_bag_counts)
    actions = canonical_market_actions(
        decision_kind=decision_kind,
        public_nature_tokens=public_nature_tokens,
        public_wildlife_bag_total=total,
        public_wildlife_bag_counts=public_wildlife_bag_counts,
        public_market_wildlife=public_market_wildlife,
    )
    rows = [action.to_bytes() for action in actions]
    action_ids = [market_decision_action_id(decision_id, row) for row in rows]
    return {
        "name": name,
        "decision_id": decision_id,
        "decision_kind": int(decision_kind),
        "public_nature_tokens": public_nature_tokens,
        "public_wildlife_bag_total": total,
        "public_wildlife_bag_counts": public_wildlife_bag_counts,
        "public_market_wildlife": public_market_wildlife,
        "action_bytes_hex": [row.hex() for row in rows],
        "action_ids": action_ids,
        "ordered_action_ids_blake3": ordered_action_ids_blake3(action_ids),
    }


def fixture() -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": 3,
        "request_schema": MARKET_REQUEST_SCHEMA,
        "response_schema": MARKET_RESPONSE_SCHEMA,
        "request_schema_blake3": MARKET_REQUEST_SCHEMA_BLAKE3,
        "response_schema_blake3": MARKET_RESPONSE_SCHEMA_BLAKE3,
        "action_schema_blake3": MARKET_ACTION_SCHEMA_BLAKE3,
        "request_tensor_dtypes": MARKET_REQUEST_TENSOR_DTYPES,
        "response_tensor_dtypes": MARKET_RESPONSE_TENSOR_DTYPES,
        "legality": "public-universal-hidden-order-intersection-v1",
        "cases": [
            _case(
                "free-safe",
                "11" * 32,
                MarketDecisionKind.FREE_THREE_OF_A_KIND,
                public_nature_tokens=0,
                public_wildlife_bag_counts=[2, 2, 2, 2, 2],
                public_market_wildlife=[0, 0, 0, 1],
            ),
            _case(
                "free-hidden-exhaustion",
                "22" * 32,
                MarketDecisionKind.FREE_THREE_OF_A_KIND,
                public_nature_tokens=0,
                public_wildlife_bag_counts=[0, 3, 0, 0, 0],
                public_market_wildlife=[0, 0, 0, 1],
            ),
            _case(
                "paid-all-subsets",
                "33" * 32,
                MarketDecisionKind.PAID_WIPES,
                public_nature_tokens=2,
                public_wildlife_bag_counts=[2, 2, 2, 2, 2],
                public_market_wildlife=[0, 1, 2, 3],
            ),
            _case(
                "paid-hidden-exhaustion",
                "44" * 32,
                MarketDecisionKind.PAID_WIPES,
                public_nature_tokens=1,
                public_wildlife_bag_counts=[4, 0, 0, 0, 0],
                public_market_wildlife=[0, 1, 2, 3],
            ),
            _case(
                "paid-no-token",
                "55" * 32,
                MarketDecisionKind.PAID_WIPES,
                public_nature_tokens=0,
                public_wildlife_bag_counts=[2, 2, 2, 2, 2],
                public_market_wildlife=[0, 1, 2, 3],
            ),
        ],
    }
    value["fixture_blake3"] = blake3.blake3(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path)
    arguments = parser.parse_args()
    value = fixture()
    encoded = json.dumps(value, sort_keys=True, indent=2) + "\n"
    if arguments.check is not None:
        if arguments.check.read_text() != encoded:
            raise SystemExit("R2-MAP market protocol fixture differs from canonical rendering")
        return 0
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
