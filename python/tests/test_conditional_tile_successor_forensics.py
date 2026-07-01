from __future__ import annotations

import numpy as np
from cascadia_mlx.conditional_tile_successor_forensics import (
    _rank_selector_scores,
    _width_bucket,
    classify_sampling_mass,
    classify_score_scale,
    selector_gates,
)


def _sampling_split(
    *,
    train_like: bool,
) -> dict[str, object]:
    weak = {
        "queries": 20,
        "target_factors": 400,
        "target_hits": 200,
        "target_misses": 200,
        "exact_queries": 0,
        "query_share": 0.20,
        "target_share": 0.35 if train_like else 0.32,
        "miss_share": 0.40 if train_like else 0.36,
        "target_mass_to_query_share": 1.75 if train_like else 1.60,
        "miss_mass_to_query_share": 2.00 if train_like else 1.80,
    }
    empty = {
        "queries": 0,
        "target_factors": 0,
        "target_hits": 0,
        "target_misses": 0,
        "exact_queries": 0,
        "query_share": 0.0,
        "target_share": 0.0,
        "miss_share": 0.0,
        "target_mass_to_query_share": 0.0,
        "miss_mass_to_query_share": 0.0,
    }
    return {
        "width": {
            "within_budget": empty,
            "width_33_64": empty,
            "width_65_96": weak,
            "width_97_128": empty,
            "width_129_plus": empty,
        }
    }


def test_width_buckets_are_frozen_at_tile_boundaries() -> None:
    assert _width_bucket(32) == "within_budget"
    assert _width_bucket(33) == "width_33_64"
    assert _width_bucket(64) == "width_33_64"
    assert _width_bucket(65) == "width_65_96"
    assert _width_bucket(97) == "width_97_128"
    assert _width_bucket(129) == "width_129_plus"


def test_sampling_mass_requires_replicated_train_validation_stratum() -> None:
    classification, matching = classify_sampling_mass(
        _sampling_split(train_like=True),
        _sampling_split(train_like=False),
    )
    assert classification == "target_mass_sampling_mismatch"
    assert matching == ["width_65_96"]


def test_score_scale_requires_both_dispersion_metrics_on_both_splits() -> None:
    def split(multiplier: float) -> dict[str, object]:
        return {
            "draft": {
                "query_standard_deviation": {"median": 1.0},
                "query_range": {"median": 2.0},
            },
            "tile": {
                "query_standard_deviation": {"median": 5.0 * multiplier},
                "query_range": {"median": 10.0 * multiplier},
            },
            "wildlife": {
                "query_standard_deviation": {"median": 1.1},
                "query_range": {"median": 2.2},
            },
        }

    classification, ratios = classify_score_scale(split(1.0), split(1.0))
    assert classification == "cross_stage_score_scale_mismatch"
    assert ratios["train_query_standard_deviation_median_ratio"] >= 4.0


def test_rank_selector_scores_are_query_local_and_monotone() -> None:
    scores = _rank_selector_scores(
        ranks=np.asarray([2.0, 0.0, 1.0, 3.0], dtype=np.float32),
        rank_mask=np.asarray([True, True, True, True]),
        offsets=np.asarray([0, 3, 4], dtype=np.int32),
    )
    percentile = scores["rank_percentile_sum"]
    assert percentile[1] > percentile[2] > percentile[0]
    assert percentile[3] == 1.0


def test_selector_gates_cover_overall_phase_and_subsets() -> None:
    strong = {
        "target_positive_recall": 0.99,
        "r4800_winner_retention": 0.99,
        "mean_retained_r4800_regret": 0.01,
    }
    report = {
        "overall": strong,
        "phase": {
            "early": strong,
            "middle": strong,
            "late": strong,
        },
        "subsets": {
            "nature_token_available": strong,
            "independent_draft_winner": strong,
        },
    }
    gates = selector_gates(report)
    assert len(gates) == 13
    assert all(gates.values())
