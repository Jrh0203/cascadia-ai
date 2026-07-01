from __future__ import annotations

from types import SimpleNamespace

from cascadia_mlx.graded_oracle_local_geometry_evaluate import (
    _dataset_identity,
    _validate_run_kind,
    local_geometry_validation_gates,
)


def _metrics() -> dict[str, object]:
    return {
        "top64_r4800_winner_recall": 0.99,
        "mean_top64_retained_r4800_regret": 0.10,
        "all_groups_scored_once": True,
        "all_candidates_scored_once": True,
        "all_scores_finite": True,
        "phase": {
            name: {
                "top64_r4800_winner_recall": 0.98,
                "mean_top64_retained_r4800_regret": 0.10,
            }
            for name in ("early", "middle", "late")
        },
        "subsets": {
            name: {
                "groups": 20,
                "top64_r4800_winner_recall": 0.95,
                "mean_top64_retained_r4800_regret": 0.20,
            }
            for name in ("nature_token_available", "independent_draft_winner")
        },
    }


def _confidence() -> dict[str, object]:
    top64 = {
        "confidence_set_coverage_95": 0.99,
        "distinguishable_winner_recall": 0.98,
    }
    return {
        "overall": {"ranking": {"model": {"top64": top64}}},
        "phases": {
            name: {
                "ranking": {
                    "model": {
                        "top64": {"confidence_set_coverage_95": 0.98}
                    }
                }
            }
            for name in ("early", "middle", "late")
        },
        "integrity": {
            "all_groups_seen_once": True,
            "all_candidates_seen_once": True,
            "all_model_scores_finite": True,
            "test_split_opened": False,
        },
    }


def test_local_geometry_gates_apply_exact_frozen_thresholds() -> None:
    gates = local_geometry_validation_gates(_metrics(), _confidence())
    assert all(gates.values())


def test_local_geometry_exact_recall_remains_strictly_greater_than_0_98() -> None:
    metrics = _metrics()
    metrics["top64_r4800_winner_recall"] = 0.98
    gates = local_geometry_validation_gates(metrics, _confidence())
    assert not gates["top64_r4800_winner_recall_strictly_greater_than_0_98"]


def test_local_geometry_run_kind_uses_top_level_manifest_contract() -> None:
    _validate_run_kind({"kind": "graded-oracle-local-geometry-ranking"})
    try:
        _validate_run_kind(
            {"training": {"kind": "graded-oracle-local-geometry-ranking"}}
        )
    except ValueError:
        pass
    else:
        raise AssertionError("nested run kind must not satisfy the manifest contract")


def test_scientific_dataset_identity_is_host_path_independent() -> None:
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
