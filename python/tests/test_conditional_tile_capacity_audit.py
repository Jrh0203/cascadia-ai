from __future__ import annotations

import mlx.core as mx
from cascadia_mlx.conditional_tile_capacity_audit import (
    AttentionTileRanker,
    classify_capacity_audit,
)
from cascadia_mlx.full_legal_hierarchical_factor_retrieval import (
    PARENT_STATE_DIM,
    STAGE_CONTEXT_DIMS,
    STAGE_ITEM_DIMS,
)


def _report(recall: float, exact: float) -> dict[str, object]:
    return {
        "best": {
            "target_factor_recall": recall,
            "exact_query_fraction": exact,
        }
    }


def test_attention_ranker_preserves_masked_shape() -> None:
    model = AttentionTileRanker()
    mask = mx.array([[True, True, False]])
    output = model(
        mx.zeros((1, PARENT_STATE_DIM)),
        mx.zeros((1, STAGE_CONTEXT_DIMS["tile"])),
        mx.zeros((1, 3, STAGE_ITEM_DIMS["tile"])),
        mask,
    )
    mx.eval(output)
    assert output.shape == (1, 3)
    assert float(output[0, 2].item()) == -1e9


def test_capacity_classification_prioritizes_local_fit() -> None:
    assert (
        classify_capacity_audit(
            _report(0.99, 0.94),
            _report(1.0, 1.0),
            _report(1.0, 1.0),
        )
        == "local_baseline_fit_insufficient"
    )


def test_capacity_classification_accepts_medium_baseline_fit() -> None:
    assert (
        classify_capacity_audit(
            _report(1.0, 1.0),
            _report(0.99, 0.95),
            _report(1.0, 1.0),
        )
        == "full_data_scale_or_optimization_insufficient"
    )


def test_capacity_classification_selects_relational_attention() -> None:
    assert (
        classify_capacity_audit(
            _report(1.0, 1.0),
            _report(0.90, 0.70),
            _report(0.99, 0.95),
        )
        == "query_relational_representation_insufficient"
    )


def test_capacity_classification_retains_unresolved_shared_fit() -> None:
    assert (
        classify_capacity_audit(
            _report(1.0, 1.0),
            _report(0.94, 0.82),
            _report(0.96, 0.88),
        )
        == "shared_capacity_or_optimization_insufficient"
    )
