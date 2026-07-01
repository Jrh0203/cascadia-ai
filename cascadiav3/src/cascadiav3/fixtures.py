"""Deterministic tiny fixtures for pre-GPU contract validation."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from .hex import RADIUS6_CELL_COUNT, coord_ref
from .schema import SCHEMA_ID, attach_checksum, checksum


def tiny_actions() -> list[dict[str, Any]]:
    return [
        {
            "action_id": "a-draft0-place-east",
            "active_seat": 0,
            "cleanup_choice": "none",
            "nature_spend": 0,
            "draft_slot": 0,
            "tile_ref": "market_tile_0",
            "wildlife_ref": "market_wildlife_0",
            "target_coord_ref": coord_ref(1, 0),
            "rotation": 2,
            "wildlife_coord_ref": coord_ref(1, 0),
            "factor_labels": {
                "draft_slot": 0,
                "tile_coord_cell_index": coord_ref(1, 0)["cell_index"],
                "wildlife_coord_cell_index": coord_ref(1, 0)["cell_index"],
            },
        },
        {
            "action_id": "a-draft1-place-overflow",
            "active_seat": 0,
            "cleanup_choice": "none",
            "nature_spend": 1,
            "draft_slot": 1,
            "tile_ref": "market_tile_1",
            "wildlife_ref": "market_wildlife_1",
            "target_coord_ref": coord_ref(7, 0, owner_seat=0, placement_id=1001),
            "rotation": 5,
            "wildlife_coord_ref": coord_ref(7, 0, owner_seat=0, placement_id=1001),
            "factor_labels": {
                "draft_slot": 1,
                "tile_coord_overflow_id": 1001,
                "wildlife_coord_overflow_id": 1001,
            },
        },
    ]


def tiny_actions_three() -> list[dict[str, Any]]:
    actions = tiny_actions()
    actions.append(
        {
            "action_id": "a-draft2-place-northwest",
            "active_seat": 0,
            "cleanup_choice": "none",
            "nature_spend": 0,
            "draft_slot": 2,
            "tile_ref": "market_tile_2",
            "wildlife_ref": "market_wildlife_2",
            "target_coord_ref": coord_ref(-1, 1),
            "rotation": 1,
            "wildlife_coord_ref": coord_ref(-1, 1),
            "factor_labels": {
                "draft_slot": 2,
                "tile_coord_cell_index": coord_ref(-1, 1)["cell_index"],
                "wildlife_coord_cell_index": coord_ref(-1, 1)["cell_index"],
            },
        }
    )
    return actions


def tiny_search_root_record() -> dict[str, Any]:
    record = {
        "schema_id": SCHEMA_ID,
        "state_hash": "tiny-state-0001",
        "active_seat": 0,
        "legal_actions": tiny_actions(),
        "priors": [0.55, 0.45],
        "visits": [7, 5],
        "per_action_Q": [91.25, 90.75],
        "selected_action": "a-draft0-place-east",
        "chance_samples": [
            {
                "sample_id": "refill-000",
                "market_slot": 0,
                "tile_draw": "tile_demo_a",
                "wildlife_draw": "elk",
                "seed": 12345,
            }
        ],
        "final_score_vector": [92, 88, 81, 77],
        "score_decomposition": {
            "0": {"wildlife": 55, "habitat": 31, "nature_tokens": 6, "total": 92},
            "1": {"wildlife": 52, "habitat": 30, "nature_tokens": 6, "total": 88},
            "2": {"wildlife": 49, "habitat": 28, "nature_tokens": 4, "total": 81},
            "3": {"wildlife": 46, "habitat": 27, "nature_tokens": 4, "total": 77},
        },
        "rank_vector": [1, 2, 3, 4],
    }
    return attach_checksum(record)


def tiny_search_root_record_three_actions() -> dict[str, Any]:
    record = {
        "schema_id": SCHEMA_ID,
        "state_hash": "tiny-state-0002",
        "active_seat": 0,
        "legal_actions": tiny_actions_three(),
        "priors": [0.40, 0.35, 0.25],
        "visits": [8, 7, 5],
        "per_action_Q": [93.0, 91.5, 89.75],
        "selected_action": "a-draft0-place-east",
        "chance_samples": [
            {
                "sample_id": "refill-001",
                "market_slot": 1,
                "tile_draw": "tile_demo_b",
                "wildlife_draw": "hawk",
                "seed": 23456,
            }
        ],
        "final_score_vector": [94, 89, 82, 79],
        "score_decomposition": {
            "0": {"wildlife": 57, "habitat": 31, "nature_tokens": 6, "total": 94},
            "1": {"wildlife": 53, "habitat": 30, "nature_tokens": 6, "total": 89},
            "2": {"wildlife": 50, "habitat": 28, "nature_tokens": 4, "total": 82},
            "3": {"wildlife": 47, "habitat": 28, "nature_tokens": 4, "total": 79},
        },
        "rank_vector": [1, 2, 3, 4],
    }
    return attach_checksum(record)


def tiny_replay_records() -> list[dict[str, Any]]:
    return [tiny_search_root_record(), tiny_search_root_record_three_actions()]


def tiny_replay_manifest(root_record: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_id": SCHEMA_ID,
        "source_generator": "cascadiav3.fixtures.tiny_search_root_record",
        "seed_domain": "fixed-demo-seed",
        "record_count": 1,
        "checksum": checksum(root_record),
        "scientific_eligibility": "dry_run",
        "created_at_utc": datetime(2026, 6, 29, tzinfo=UTC).isoformat(),
        "format": "jsonl",
        "notes": "CPU-only pre-GPU fixture; not training evidence.",
    }


def radius6_census(coords: list[dict[str, int]]) -> dict[str, Any]:
    total = len(coords)
    in_radius = 0
    overflow_examples: list[dict[str, int]] = []

    for coord in coords:
        q = coord["q"]
        r = coord["r"]
        s = -q - r
        member = max(abs(q), abs(r), abs(s)) <= 6
        if member:
            in_radius += 1
        elif len(overflow_examples) < 5:
            overflow_examples.append({"q": q, "r": r, "s": s})

    return {
        "canonical_radius": 6,
        "canonical_cell_count": RADIUS6_CELL_COUNT,
        "total_coordinates": total,
        "in_radius6": in_radius,
        "overflow": total - in_radius,
        "overflow_examples": overflow_examples,
    }
