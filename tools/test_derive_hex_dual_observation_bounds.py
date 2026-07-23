import json

import pytest

from tools.derive_hex_bipartite_edge_bounds import CAP
from tools.derive_hex_dual_observation_bounds import (
    collect_shards,
    combine_components,
    connected_component_maximum,
)


def test_one_fox_can_observe_two_singleton_target_classes() -> None:
    proof = connected_component_maximum(1, 1, 1, seconds=2, workers=1)
    assert proof["status"] == "OPTIMAL"
    assert proof["maximum"] == 1
    assert proof["best_bound"] == 1


def test_impossible_connected_component_is_an_exact_infeasibility() -> None:
    proof = connected_component_maximum(1, 6, 6, seconds=2, workers=1)
    assert proof["status"] == "INFEASIBLE"
    assert proof["maximum"] is None


def test_three_dimensional_component_dp_combines_disconnected_support() -> None:
    component = [
        [[0] * (CAP + 1) for _ in range(CAP + 1)]
        for _ in range(CAP + 1)
    ]
    component[1][1][1] = 1
    combined = combine_components(component)
    assert combined[1][1][1] == 1
    assert combined[2][2][2] == 2
    assert combined[6][6][6] == 6


def test_collector_requires_complete_exact_symmetric_table(tmp_path) -> None:
    proofs = []
    for foxes in range(1, CAP + 1):
        for first in range(1, CAP + 1):
            for second in range(1, CAP + 1):
                value = min(foxes, first * second)
                proofs.append(
                    {
                        "foxes": foxes,
                        "first_targets": first,
                        "second_targets": second,
                        "status": "OPTIMAL",
                        "maximum": value,
                        "best_bound": value,
                    }
                )
    shard = tmp_path / "shard.json"
    shard.write_text(
        json.dumps(
            {
                "schema": "hex-dual-observation-bound-shard-v1",
                "proofs": proofs,
            }
        )
    )
    payload = collect_shards([shard])
    assert payload["proof_complete"]
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(shard.read_text())
    with pytest.raises(ValueError, match="duplicate component counts"):
        collect_shards([shard, duplicate])


def test_collector_accepts_exactly_infeasible_component_types(tmp_path) -> None:
    proofs = []
    for foxes in range(1, CAP + 1):
        for first in range(1, CAP + 1):
            for second in range(1, CAP + 1):
                infeasible = (foxes, first, second) == (1, 6, 6)
                value = None if infeasible else min(foxes, first * second)
                proofs.append(
                    {
                        "foxes": foxes,
                        "first_targets": first,
                        "second_targets": second,
                        "status": "INFEASIBLE" if infeasible else "OPTIMAL",
                        "maximum": value,
                        "best_bound": value,
                    }
                )
    shard = tmp_path / "infeasible.json"
    shard.write_text(
        json.dumps(
            {
                "schema": "hex-dual-observation-bound-shard-v1",
                "proofs": proofs,
            }
        )
    )

    payload = collect_shards([shard])

    assert payload["proof_complete"]
    assert payload["connected_component_maximum"][1][6][6] == 0
    assert payload["global_maximum"][1][6][6] == 1
