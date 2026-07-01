from __future__ import annotations

from types import SimpleNamespace

from cascadia_mlx.graded_oracle_frontier_anchor_evaluate import (
    _dataset_identity,
    _validate_run_kind,
)


def test_run_kind_uses_the_top_level_manifest_contract() -> None:
    _validate_run_kind({"kind": "graded-oracle-frontier-anchored-ranking"})
    try:
        _validate_run_kind(
            {
                "training": {
                    "kind": "graded-oracle-frontier-anchored-ranking"
                }
            }
        )
    except ValueError:
        pass
    else:
        raise AssertionError("nested run kind must not satisfy the contract")


def test_dataset_identity_omits_host_specific_paths() -> None:
    dataset = SimpleNamespace(
        manifest={
            "dataset_id": "validation-id",
            "completed_games": 3,
            "seeds": [1, 2, 3],
        },
        split="validation",
        group_count=240,
        candidate_count=860203,
        root="/host-specific/path",
    )
    assert _dataset_identity(dataset, "manifest-hash") == {
        "dataset_id": "validation-id",
        "split": "validation",
        "games": 3,
        "seeds": [1, 2, 3],
        "groups": 240,
        "candidates": 860203,
        "manifest_blake3": "manifest-hash",
    }
