from __future__ import annotations

import pytest
from cascadia_mlx.r2_map_dataset import (
    R2MapDatasetError,
    _accumulate_packing_statistics,
    _empty_packing_statistics,
)
from cascadia_mlx.r2_map_pipe_dataset import R2MapPackedPipeDatasetAdapter


def _game(index: int, widths: list[int] | None = None) -> dict[str, object]:
    return {
        "global_game_index": index,
        "game_id": f"{index:064x}",
        "candidate_widths": widths or list(range(1, 81)),
        "split": "train",
        "source_file_name": "source.r2sh",
    }


def test_focal_width_projection_uses_game_index_mod_four() -> None:
    game = _game(7)
    assert R2MapPackedPipeDatasetAdapter._focal_widths(game) == tuple(range(4, 81, 4))
    game["global_game_index"] = 4
    assert R2MapPackedPipeDatasetAdapter._focal_widths(game) == tuple(range(1, 78, 4))


def test_focal_width_projection_rejects_partial_or_invalid_games() -> None:
    with pytest.raises(R2MapDatasetError, match="80 indexed widths"):
        R2MapPackedPipeDatasetAdapter._focal_widths(_game(0, [1] * 20))
    widths = [1] * 80
    widths[3] = 0
    with pytest.raises(R2MapDatasetError, match="invalid width"):
        R2MapPackedPipeDatasetAdapter._focal_widths(_game(3, widths))


def test_candidate_cost_packing_isolates_oversize_screens_without_pruning() -> None:
    statistics = _empty_packing_statistics(0)
    widths = [1, 1, 441, 1, 12_972, 1, 1]
    _accumulate_packing_statistics(
        statistics,
        widths,
        group_batch_size=64,
        maximum_candidates_per_batch=4_096,
    )
    assert statistics == {
        "epoch": 0,
        "steps": 3,
        "draft_groups": len(widths),
        "selected_only_groups": 5,
        "draft_policy_targets": 2,
        "draft_candidates": sum(widths),
        "padded_draft_candidates": 1_764 + 12_972 + 2,
        "maximum_batch_groups": 4,
        "minimum_batch_groups": 1,
    }
