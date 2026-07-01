from __future__ import annotations

import copy

import numpy as np
import pytest
from cascadia_mlx.graded_oracle_dataset import (
    _CANDIDATE_DTYPE,
    GradedOracleGroupHeader,
)
from cascadia_mlx.s4_candidate_relation_census import (
    K_VALUES,
    aggregate_records,
    analyze_candidate_group,
    candidate_relation_keys,
    merge_reports,
    stable_screen_order,
)


def _candidates(count: int = 4) -> np.ndarray:
    candidates = np.zeros(count, dtype=_CANDIDATE_DTYPE)
    candidates["screen_rank"] = np.arange(count, dtype=np.uint16) + 1
    candidates["action_hash"][:, 0] = np.arange(count, dtype=np.uint8) + 1
    candidates["action"]["draft_kind"] = [0, 0, 1, 1][:count]
    candidates["action"]["tile_slot"] = [0, 0, 1, 1][:count]
    candidates["action"]["wildlife_slot"] = [0, 0, 1, 1][:count]
    candidates["action"]["tile_id"] = [10, 10, 20, 20][:count]
    candidates["action"]["tile_q"] = [0, 0, 1, 2][:count]
    candidates["action"]["tile_r"] = [0, 0, 1, 2][:count]
    candidates["action"]["rotation"] = [0, 0, 1, 2][:count]
    candidates["action"]["wildlife_present"] = 1
    candidates["action"]["wildlife_q"] = [1, 2, 1, 2][:count]
    candidates["action"]["wildlife_r"] = [0, 0, 1, 2][:count]
    candidates["r4800"]["mean"] = [10.0, 9.5, 9.0, 8.0][:count]
    candidates["r4800"]["stddev"] = 1.0
    candidates["r4800"]["samples"] = 4800
    return candidates


def _header(group_id: int, turn: int = 30) -> GradedOracleGroupHeader:
    return GradedOracleGroupHeader(
        group_id=group_id,
        public_state_hash=np.zeros(32, dtype=np.uint8),
        candidate_count=4,
        selected_index=0,
        champion_index=0,
        turn=turn,
        selected_draft_kind=0,
    )


def test_relation_keys_preserve_exact_draft_and_equivalent_afterstate() -> None:
    candidates = _candidates()
    afterstates = np.zeros((4, 32), dtype=np.uint8)
    afterstates[2:, 0] = [2, 3]
    keys = candidate_relation_keys(candidates["action"], afterstates)

    assert keys["same_draft"][0][0] == keys["same_draft"][0][1]
    assert keys["same_sibling_plan"][0][0] == keys["same_sibling_plan"][0][1]
    assert keys["same_frontier"][0][0] == keys["same_frontier"][0][1]
    assert keys["same_wildlife_destination"][0][0] != keys["same_wildlife_destination"][0][1]
    assert keys["equivalent_afterstate"][0][0] == keys["equivalent_afterstate"][0][1]


def test_relation_keys_ignore_nonsemantic_struct_padding() -> None:
    candidates = _candidates(2)
    actions = np.zeros(2, dtype=candidates["action"].dtype)
    for field in actions.dtype.names or ():
        actions[field] = candidates["action"][field]
    action_bytes = actions.view(np.uint8).reshape(2, -1)
    action_bytes[0, 39:42] = [1, 2, 3]
    action_bytes[1, 39:42] = [4, 5, 6]
    afterstates = np.zeros((2, 32), dtype=np.uint8)

    keys = candidate_relation_keys(actions, afterstates)

    assert keys["same_draft"][0][0] == keys["same_draft"][0][1]
    assert keys["same_sibling_plan"][0][0] == keys["same_sibling_plan"][0][1]


def test_group_analysis_reports_relation_graph_and_oracle_retention() -> None:
    candidates = _candidates()
    candidates["screen_rank"] = [2, 1, 3, 4]
    afterstates = np.zeros((4, 32), dtype=np.uint8)
    afterstates[2:, 0] = [2, 3]

    record = analyze_candidate_group(
        split="validation",
        row=7,
        header=_header(44),
        candidates=candidates,
        afterstate_hashes=afterstates,
        selected_index=0,
    )

    assert stable_screen_order(candidates).tolist() == [1, 0, 2, 3]
    assert record["selected_screen_rank_zero_based"] == 1
    for k in K_VALUES:
        context = record["contexts"][str(k)]
        assert context["winner_retained"] is True
        assert context["confidence_set_covered"] is True
        assert context["relations"]["same_draft"]["anchor_candidates_with_sibling"] == 4
        assert context["relations"]["equivalent_afterstate"]["anchor_pair_edges"] == 1
        assert context["graph"]["largest_component"] >= 2

    aggregate = aggregate_records([record])
    assert aggregate["validation"]["contexts"]["128"]["winner_retention"] == 1.0
    assert (
        aggregate["validation"]["contexts"]["128"]["relations"]["same_draft"][
            "winner_linked_to_anchor_fraction"
        ]
        == 1.0
    )


def _synthetic_shard(modulus: int, remainder: int) -> dict[str, object]:
    records = []
    for split, total in (("train", 560), ("validation", 240)):
        for row in range(total):
            if row % modulus != remainder:
                continue
            candidates = _candidates()
            records.append(
                analyze_candidate_group(
                    split=split,
                    row=row,
                    header=_header((0 if split == "train" else 10_000) + row),
                    candidates=candidates,
                    afterstate_hashes=np.zeros((4, 32), dtype=np.uint8),
                    selected_index=0,
                )
            )
    identity = {
        "schema_version": 1,
        "experiment_id": "s4-candidate-relation-foundation-v1",
        "open_data_verification_id": "a" * 64,
        "cache_id": "b" * 64,
        "row_shard": {"modulus": modulus, "remainder": remainder},
    }
    return {
        **identity,
        "scientific_identity": identity,
        "records": records,
        "report_id": f"{remainder:064x}",
    }


def test_merge_is_order_invariant_and_rejects_overlap() -> None:
    reports = [_synthetic_shard(3, remainder) for remainder in range(3)]
    forward = merge_reports(reports)
    reverse = merge_reports(reversed(reports))

    assert forward == reverse
    assert forward["train_groups"] == 560
    assert forward["validation_groups"] == 240
    assert forward["aggregate"]["validation"]["contexts"]["256"]["confidence_set_coverage"] == 1.0

    duplicated = copy.deepcopy(reports)
    duplicated[2]["scientific_identity"]["row_shard"]["remainder"] = 1
    with pytest.raises(ValueError, match="remainders"):
        merge_reports(duplicated)
