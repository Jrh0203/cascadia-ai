"""Torch-ready feature extraction from Cascadia v3 search-root fixtures."""

from __future__ import annotations

from typing import Any

SCORE_CATEGORIES = (
    "bear",
    "elk",
    "salmon",
    "hawk",
    "fox",
    "habitat",
    "nature_tokens",
)

STATE_FEATURE_DIM = 6
ACTION_FEATURE_DIM = 16


def coord_features(coord: dict[str, Any]) -> list[float]:
    return [
        float(coord["q"]),
        float(coord["r"]),
        float(coord["s"]),
        1.0 if coord["kind"] == "canonical" else 0.0,
        1.0 if coord["kind"] == "overflow" else 0.0,
        float(coord.get("cell_index", -1) if coord.get("cell_index") is not None else -1),
    ]


def action_features(root: dict[str, Any]) -> list[list[float]]:
    rows = []
    for action in root["legal_actions"]:
        row = [
            float(action["active_seat"]),
            float(action["nature_spend"]),
            float(action["draft_slot"]),
            float(action["rotation"]),
        ]
        row.extend(coord_features(action["target_coord_ref"]))
        row.extend(coord_features(action["wildlife_coord_ref"]))
        rows.append(row)
    return rows


def state_features(root: dict[str, Any]) -> list[float]:
    return [
        float(root["active_seat"]),
        float(len(root["legal_actions"])),
        float(sum(root["visits"])),
        float(max(root["visits"])),
        float(sum(root["priors"])),
        float(sum(root["final_score_vector"]) / len(root["final_score_vector"])),
    ]


def target_score_decomposition(root: dict[str, Any]) -> list[list[float]]:
    by_category: list[list[float]] = []
    for category in SCORE_CATEGORIES:
        values = []
        for seat in range(4):
            parts = root["score_decomposition"][str(seat)]
            if category == "habitat":
                values.append(float(parts["habitat"]))
            elif category == "nature_tokens":
                values.append(float(parts["nature_tokens"]))
            else:
                # The tiny fixture has total wildlife only, not per-species true labels.
                values.append(float(parts["wildlife"]) / 5.0)
        by_category.append(values)
    return by_category
