import json

import pytest

from tools.derive_hex_bipartite_edge_bounds import (
    CAP,
    collect_shards,
    combine_components,
    connected_component_maximum,
)


def test_component_dp_supports_disconnected_cross_edge_graphs() -> None:
    component = [[0] * (CAP + 1) for _ in range(CAP + 1)]
    component[1][1] = 1
    component[1][2] = 2
    component[2][1] = 2
    component[2][2] = 3

    combined = combine_components(component)

    assert combined[1][2] == 2
    assert combined[2][2] == 3
    assert combined[4][4] == 6


def test_one_by_one_component_has_one_edge() -> None:
    proof = connected_component_maximum(1, 1, seconds=2, workers=1)
    assert proof["status"] == "OPTIMAL"
    assert proof["maximum"] == 1
    assert proof["best_bound"] == 1


def test_one_by_two_component_qualifies_the_left_vertex() -> None:
    proof = connected_component_maximum(
        1,
        2,
        seconds=2,
        workers=1,
        metric="qualified_left",
    )
    assert proof["status"] == "OPTIMAL"
    assert proof["maximum"] == 1
    assert proof["best_bound"] == 1


def test_collector_requires_exact_disjoint_symmetric_coverage(tmp_path) -> None:
    proofs = []
    for left in range(1, CAP + 1):
        for right in range(1, CAP + 1):
            value = min(left * right, left + right)
            proofs.append(
                {
                    "left": left,
                    "right": right,
                    "metric": "edges",
                    "status": "OPTIMAL",
                    "maximum": value,
                    "best_bound": value,
                }
            )
    shard = tmp_path / "shard.json"
    shard.write_text(
        json.dumps(
            {
                "schema": "hex-bipartite-bound-shard-v1",
                "metric": "edges",
                "proofs": proofs,
            }
        )
    )
    payload = collect_shards([shard])
    assert payload["proof_complete"]
    assert len(payload["proofs"]) == CAP * CAP

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(shard.read_text())
    with pytest.raises(ValueError, match="duplicate component pair"):
        collect_shards([shard, duplicate])
