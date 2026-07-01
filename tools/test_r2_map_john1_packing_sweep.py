from __future__ import annotations

import pytest
from r2_map_john1_packing_sweep import (
    _quantile_index,
    _representative_indices,
    _select_group_batch_size,
    _synthetic_maximum_width_batch,
    _wall_projections,
    parser,
)


def test_representative_indices_cover_selected_median_and_maximum_widths() -> None:
    widths = [1, 7, 1, 3, 1, 15, 1, 1]
    assert _quantile_index([1, 3, 5], widths, 0.5) == 1
    assert _representative_indices(
        widths,
        group_batch_size=4,
        maximum_candidates_per_batch=16,
        imitation_quantile=None,
    ) == [0, 2, 4, 6]
    assert _representative_indices(
        widths,
        group_batch_size=4,
        maximum_candidates_per_batch=16,
        imitation_quantile=0.5,
    ) == [1, 0]
    assert _representative_indices(
        widths,
        group_batch_size=4,
        maximum_candidates_per_batch=16,
        imitation_quantile=1.0,
    ) == [5]
    assert _representative_indices(
        widths,
        group_batch_size=4,
        maximum_candidates_per_batch=16,
        imitation_quantile=None,
        imitation_width=7,
    ) == [1, 0]
    with pytest.raises(ValueError, match="one width rule"):
        _representative_indices(
            widths,
            group_batch_size=4,
            maximum_candidates_per_batch=16,
            imitation_quantile=0.5,
            imitation_width=7,
        )
    with pytest.raises(RuntimeError, match="exact group cap"):
        _representative_indices(
            [1, 7, 1, 3, 1, 15],
            group_batch_size=4,
            maximum_candidates_per_batch=16,
            imitation_quantile=None,
        )


def test_qualifying_cli_exposes_only_transaction_authorities_not_redundant_paths() -> None:
    option_strings = {
        option
        for action in parser()._actions
        for option in action.option_strings
    }
    assert {
        "--source-transaction-manifest-relative",
        "--source-transaction-commit-receipt-relative",
        "--dataset-transaction-manifest-relative",
    }.issubset(option_strings)
    assert option_strings.isdisjoint(
        {
            "--compact-index-relative",
            "--shard-root-relative",
            "--exporter-relative",
            "--source-manifest-relative",
            "--reference-manifest-relative",
            "--dataset-transaction-commit-receipt-relative",
        }
    )


def test_wall_projection_uses_exact_steps_compute_and_remote_window_work() -> None:
    plan = {
        "group_batch_size": 64,
        "epoch_plans": [
            {
                "steps": 3,
                "draft_groups": 100,
                "padded_draft_candidates": 300,
            }
        ],
    }
    measurements = [
        {
            "label": "g64-production",
            "timed_steps": 3,
            "step_durations_ns": [
                2_000_000_000,
                3_000_000_000,
                4_000_000_000,
            ],
            "remote_window_durations_ns": [1_000_000_000],
            "remote_window_duration_ns_per_step": [1_000_000_000, 0, 0],
        }
    ]
    representatives = [
        {
            "label": f"g64-{suffix}",
            "step_durations_ns": [1_000_000_000] * 3,
        }
        for suffix in ("selected", "imitation-p50", "imitation-max")
    ]
    projected = _wall_projections(
        [plan],
        measurements,
        representatives,
        remote_source_durations_ns=[1_000_000_000, 1_000_000_000],
    )[0]
    assert projected["steps_per_epoch"] == [3]
    assert projected["central_12_epoch_wall_seconds"] == pytest.approx(11.0)
    assert projected["conservative_12_epoch_wall_seconds"] == pytest.approx(14.0)


def test_selector_uses_passing_minimum_conservative_wall_then_lower_cap() -> None:
    plans = [
        {
            "group_batch_size": group,
            "maximum_candidate_width": 15,
            "totals": {"steps": 120},
        }
        for group in (16, 32, 64)
    ]
    measurements = []
    for group in (16, 32, 64):
        for suffix in ("selected", "imitation-p50", "imitation-max", "production"):
            measurements.append(
                {
                    "label": f"g{group}-{suffix}",
                    "timed_steps": 5,
                    "resource_receipt": {
                        "maximum_rss_bytes": 1,
                        "process_swaps": 0,
                        "system_swap_delta_bytes": 0,
                    },
                    "mlx_memory": {"cache_bytes": 1},
                    "training_counters": {"padded_draft_candidates": 50},
                }
            )
    projections = [
        {
            "group_batch_size": 16,
            "conservative_12_epoch_wall_seconds": 20.0,
        },
        {
            "group_batch_size": 32,
            "conservative_12_epoch_wall_seconds": 10.0,
        },
        {
            "group_batch_size": 64,
            "conservative_12_epoch_wall_seconds": 10.0,
        },
    ]
    selected = _select_group_batch_size(
        plans,
        measurements,
        projections,
        maximum_candidates_per_batch=16_384,
    )
    assert selected["selected_group_batch_size"] == 32


def test_synthetic_maximum_width_batch_is_policy_identifiable_and_fully_masked() -> None:
    batch = _synthetic_maximum_width_batch(7, "a" * 64)
    assert batch.validate() == (1, 7)
    assert batch.bootstrap_policy_mask.tolist() == [True]
    assert batch.selected_action_index.tolist() == [6]
    assert batch.score_target_mask.tolist() == [[False, False, False, False, False, False, True]]
    assert batch.opponent_valid_mask.tolist() == [[False, False, False]]
    assert batch.market_disposition_mask.tolist() == [[False, False, False, False]]
