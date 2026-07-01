from __future__ import annotations

import numpy as np
import pytest
from cascadia_mlx.r3_action_edit_mlx_cache import (
    R3_LOCAL_PATCH_TOKEN,
    R3_TOKEN_FEATURES,
)
from cascadia_mlx.r3_action_edit_mlx_forensics import (
    _local_token_counts,
    aggregate_records,
    summarize_records,
)


def _record(
    *,
    winner_rank: int,
    covered: bool,
    phase: str,
    candidate_count: int,
    global_tokens: int,
) -> dict:
    return {
        "winner_rank": winner_rank,
        "best_confidence_set_rank": winner_rank if covered else winner_rank + 100,
        "candidate_count": candidate_count,
        "winner_recalled_top64": winner_rank < 64,
        "confidence_set_covered_top64": covered,
        "top64_retained_r4800_regret": float(winner_rank) / 100.0,
        "winner_token_count": 70,
        "winner_global_token_count": global_tokens,
        "phase": phase,
        "low_supply": phase == "late",
        "independent_draft_winner": winner_rank % 2 == 0,
    }


def test_summarize_records_reports_quality_and_shape() -> None:
    records = [
        _record(
            winner_rank=2,
            covered=True,
            phase="early",
            candidate_count=500,
            global_tokens=12,
        ),
        _record(
            winner_rank=80,
            covered=False,
            phase="late",
            candidate_count=5000,
            global_tokens=60,
        ),
    ]

    summary = summarize_records(records)
    assert summary["groups"] == 2
    assert summary["top64_winner_recall"] == 0.5
    assert summary["top64_confidence_set_coverage_95"] == 0.5
    assert summary["mean_top64_retained_r4800_regret"] == pytest.approx(0.41)
    assert summary["candidate_count"]["maximum"] == 5000


def test_aggregate_records_uses_fixed_interpretable_strata() -> None:
    records = [
        _record(
            winner_rank=1,
            covered=True,
            phase="early",
            candidate_count=256,
            global_tokens=8,
        ),
        _record(
            winner_rank=65,
            covered=False,
            phase="late",
            candidate_count=5000,
            global_tokens=64,
        ),
    ]

    aggregate = aggregate_records(records)
    assert aggregate["strata"]["phase"]["early"]["groups"] == 1
    assert aggregate["strata"]["phase"]["late"]["groups"] == 1
    assert aggregate["strata"]["action_width"]["0001-0512"]["groups"] == 1
    assert aggregate["strata"]["action_width"]["4097-plus"]["groups"] == 1
    assert aggregate["strata"]["winner_global_token_count"]["00-16"]["groups"] == 1
    assert aggregate["strata"]["winner_global_token_count"]["49-plus"]["groups"] == 1


def test_local_token_counts_maps_one_based_codec_to_zero_based_one_hot() -> None:
    features = np.zeros((2, 8, R3_TOKEN_FEATURES), dtype=np.float32)
    mask = np.zeros((2, 8), dtype=np.bool_)
    mask[0, :7] = True
    mask[1, :8] = True
    features[0, :7, R3_LOCAL_PATCH_TOKEN - 1] = 1.0
    features[1, :3, R3_LOCAL_PATCH_TOKEN - 1] = 1.0
    features[1, 3:8, 0] = 1.0

    assert _local_token_counts(features, mask).tolist() == [7, 3]
