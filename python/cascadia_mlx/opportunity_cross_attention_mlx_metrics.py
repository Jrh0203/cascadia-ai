"""Quality and serving evidence for opportunity query-conditioning arms."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import numpy as np

from cascadia_mlx.complete_decision_mlx_benchmark import (
    CompleteDecisionBatchAdapter,
    benchmark_complete_decisions,
)
from cascadia_mlx.relational_substrate_mlx_metrics import (
    CANDIDATE_CHUNK,
    evaluate_relational_substrate,
)

PARENT_INPUT_COUNT = 12


def evaluate_opportunity_cross_attention(
    model: object,
    dataset: object,
    *,
    rows: np.ndarray | None = None,
    candidate_chunk: int = CANDIDATE_CHUNK,
    prediction_panel_size: int = 64,
) -> dict[str, Any]:
    """Evaluate every arm on the same exact-R2 factual surface."""
    return evaluate_relational_substrate(
        model,
        dataset,
        arm="c0-exact-r2",
        rows=rows,
        candidate_chunk=candidate_chunk,
        prediction_panel_size=prediction_panel_size,
    )


def benchmark_opportunity_cross_attention(
    model: object,
    dataset: object,
    *,
    row: int = 0,
    candidate_chunk: int = CANDIDATE_CHUNK,
    warmup_iterations: int = 5,
    steady_iterations: int = 30,
    decision_rows: np.ndarray | None = None,
) -> dict[str, Any]:
    """Benchmark candidate-query adapters with their exact memories present."""
    return benchmark_complete_decisions(
        model,
        dataset,
        arm="c0-exact-r2",
        adapter=CompleteDecisionBatchAdapter(
            inputs=_batch_inputs,
            parent_input_count=PARENT_INPUT_COUNT,
            parent_batch=_parent_batch,
            model_batch=_model_batch,
        ),
        row=row,
        candidate_chunk=candidate_chunk,
        warmup_iterations=warmup_iterations,
        steady_iterations=steady_iterations,
        decision_rows=decision_rows,
    )


def _batch_inputs(batch: object) -> tuple[mx.array, ...]:
    base = batch.base
    parent = batch.parent
    return (
        parent.r2_token_features,
        parent.r2_token_types,
        parent.r2_token_mask,
        parent.relational_values,
        parent.relational_classes,
        parent.relational_mask,
        parent.market_features,
        parent.market_mask,
        parent.player_features,
        parent.player_mask,
        parent.global_features,
        parent.transform_ids,
        batch.candidate_token_features,
        batch.candidate_token_mask,
        base.action_features,
        base.prior_features,
        base.staged_market_entities,
        base.staged_market_mask,
        base.candidate_mask,
        base.screen_value,
        batch.supply_vector,
        batch.staged_supply_vector,
        batch.selected_archetype,
        batch.frontier_features,
        batch.derivative_features,
        batch.supply_tokens,
        batch.supply_mask,
    )


def _parent_batch(values: tuple[mx.array, ...]) -> SimpleNamespace:
    return SimpleNamespace(
        r2_token_features=values[0],
        r2_token_types=values[1],
        r2_token_mask=values[2],
        relational_values=values[3],
        relational_classes=values[4],
        relational_mask=values[5],
        market_features=values[6],
        market_mask=values[7],
        player_features=values[8],
        player_mask=values[9],
        global_features=values[10],
        transform_ids=values[11],
    )


def _model_batch(values: tuple[mx.array, ...]) -> SimpleNamespace:
    base = SimpleNamespace(
        action_features=values[14],
        prior_features=values[15],
        staged_market_entities=values[16],
        staged_market_mask=values[17],
        candidate_mask=values[18],
        screen_value=values[19],
    )
    return SimpleNamespace(
        parent=_parent_batch(values[:PARENT_INPUT_COUNT]),
        base=base,
        candidate_token_features=values[12],
        candidate_token_mask=values[13],
        supply_vector=values[20],
        staged_supply_vector=values[21],
        selected_archetype=values[22],
        frontier_features=values[23],
        derivative_features=values[24],
        supply_tokens=values[25],
        supply_mask=values[26],
    )


__all__ = [
    "benchmark_opportunity_cross_attention",
    "evaluate_opportunity_cross_attention",
]
