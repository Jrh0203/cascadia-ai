from __future__ import annotations

from collections.abc import Iterator
from typing import ClassVar

import numpy as np
from frontier_hierarchical_tile_collision_audit import audit_cache


class _Cache:
    split = "train"
    manifest: ClassVar[dict[str, object]] = {
        "payload_blake3": "cache",
        "queries": {"tile": 2},
        "items": {"tile": 4},
    }

    def iter_shards(self) -> Iterator[dict[str, np.ndarray]]:
        yield {
            "group_state": np.asarray([[1.0], [1.0]], dtype=np.float32),
            "tile_query_group": np.asarray([0, 1], dtype=np.int32),
            "tile_query_context": np.asarray(
                [[2.0], [2.0]],
                dtype=np.float32,
            ),
            "tile_query_offsets": np.asarray([0, 2, 4], dtype=np.int64),
            "tile_item_features": np.asarray(
                [[3.0], [4.0], [3.0], [5.0]],
                dtype=np.float32,
            ),
            "tile_item_target": np.asarray(
                [True, False, False, True],
                dtype=np.bool_,
            ),
            "tile_item_rank": np.asarray(
                [1.0, 2.0, 3.0, 4.0],
                dtype=np.float32,
            ),
            "tile_item_rank_mask": np.ones(4, dtype=np.bool_),
        }


def test_audit_cache_exactly_verifies_conflicting_fingerprints() -> None:
    report = audit_cache(_Cache())
    assert report["all_queries_covered"]
    assert report["all_items_covered"]
    assert report["unique_model_input_fingerprints"] == 3
    assert report["repeated_fingerprint_occurrences"] == 1
    assert report["candidate_conflicting_fingerprints"] == 1
    assert report["exact_target_conflicting_representations"] == 1
    assert report["exact_rank_conflicting_representations"] == 1
    assert report["target_positive_occurrences_in_conflicts"] == 1
    assert report["occurrences_in_target_conflicts"] == 2
    assert report["target_positive_conflict_fraction"] == 0.5
    assert report["exact_target_collision_material_at_1pct"]
