"""Standard-library mock model smoke for CascadiaFormer-S tensor contracts.

This is intentionally not a neural network. It verifies the shape contract that
the future GPU implementation must satisfy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MockCascadiaFormerSConfig:
    layers: int = 8
    d_model: int = 384
    heads: int = 8
    seats: int = 4
    score_categories: tuple[str, ...] = (
        "bear",
        "elk",
        "salmon",
        "hawk",
        "fox",
        "habitat",
        "nature_tokens",
    )


def mock_forward(
    *,
    state_tokens: list[dict[str, Any]],
    action_tokens: list[dict[str, Any]],
    cgab_edges: list[dict[str, Any]],
    config: MockCascadiaFormerSConfig | None = None,
) -> dict[str, Any]:
    cfg = config or MockCascadiaFormerSConfig()
    action_count = len(action_tokens)
    if action_count == 0:
        raise ValueError("mock_forward requires at least one legal action")

    return {
        "model": "CascadiaFormer-Zero-S-mock",
        "layers": cfg.layers,
        "d_model": cfg.d_model,
        "heads": cfg.heads,
        "state_token_count": len(state_tokens),
        "action_token_count": action_count,
        "cgab_edge_count": len(cgab_edges),
        "legal_action_logits": [0.0 for _ in action_tokens],
        "value_vector": [0.0 for _ in range(cfg.seats)],
        "rank_logits": [[0.0 for _ in range(cfg.seats)] for _ in range(cfg.seats)],
        "score_decomposition": {
            category: [0.0 for _ in range(cfg.seats)] for category in cfg.score_categories
        },
    }


def validate_mock_output(output: dict[str, Any], *, action_count: int, seats: int = 4) -> None:
    if len(output["legal_action_logits"]) != action_count:
        raise AssertionError("legal_action_logits length does not match action count")
    if len(output["value_vector"]) != seats:
        raise AssertionError("value_vector must have one value per seat")
    if len(output["rank_logits"]) != seats:
        raise AssertionError("rank_logits outer length must equal seats")
    if any(len(row) != seats for row in output["rank_logits"]):
        raise AssertionError("rank_logits inner length must equal seats")
    for category, values in output["score_decomposition"].items():
        if len(values) != seats:
            raise AssertionError(f"score_decomposition {category} must have one value per seat")
