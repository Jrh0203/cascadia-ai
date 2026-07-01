from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from cascadia_mlx.graded_oracle_frontier_anchor import (
    GRADED_SOURCE_CHAMPION_FRONTIER,
)
from cascadia_mlx.oracle_proposal_selector import (
    EXPERIMENT_ID,
    _scientific_blake3,
    _write_json_atomic,
    classify_reports,
    combine,
    filter_factor_batch,
    oracle_proposal_mask,
    proposal_payload_blake3,
)


def _hierarchy_arrays() -> dict[str, np.ndarray]:
    return {
        "group_action_offsets": np.asarray([0, 4], dtype=np.int64),
        "action_source_flags": np.asarray(
            [GRADED_SOURCE_CHAMPION_FRONTIER, 0, 0, 0],
            dtype=np.int32,
        ),
        "phase": np.asarray([1], dtype=np.int8),
        "nature_tokens": np.asarray([2], dtype=np.int8),
        "selected_index": np.asarray([1], dtype=np.int32),
        "action_draft_kind": np.asarray([0, 1, 0, 0], dtype=np.int8),
        "action_hash": np.arange(4 * 32, dtype=np.uint8).reshape(4, 32),
        "draft_action_item": np.asarray([0, 1, 2, 3], dtype=np.int32),
        "draft_item_target": np.asarray([False, True, True, True]),
        "tile_action_item": np.asarray([0, 1, 2, 3], dtype=np.int32),
        "tile_item_target": np.asarray([False, True, True, False]),
        "wildlife_action_item": np.asarray([0, 1, 2, 3], dtype=np.int32),
        "wildlife_item_target": np.asarray([False, True, False, True]),
    }


def test_oracle_proposal_keeps_frontier_and_all_factor_targets() -> None:
    mask = oracle_proposal_mask(_hierarchy_arrays(), 0)
    assert mask.tolist() == [True, True, False, False]


def test_json_outputs_normalize_numpy_scalars_and_arrays(tmp_path: Path) -> None:
    value = {
        "passed": np.bool_(True),
        "count": np.int64(7),
        "score": np.float32(0.5),
        "values": np.asarray([1, 2], dtype=np.int32),
    }
    assert len(_scientific_blake3(value)) == 64
    path = tmp_path / "report.json"
    _write_json_atomic(path, value)
    assert json.loads(path.read_text()) == {
        "passed": True,
        "count": 7,
        "score": 0.5,
        "values": [1, 2],
    }


def test_filter_factor_batch_remaps_selected_index() -> None:
    arrays = _hierarchy_arrays()
    metadata = {
        "group_offsets": np.asarray([0, 4], dtype=np.int64),
        "target": np.asarray([False, True, False, False]),
        "source_flags": arrays["action_source_flags"],
        "screen_rank": np.arange(4, dtype=np.int32),
        "action_hash": arrays["action_hash"],
        "selected_index": np.asarray([1], dtype=np.int32),
        "r4800_mean": np.arange(4, dtype=np.float32),
        "r4800_mask": np.ones(4, dtype=np.bool_),
    }
    factors = np.arange(4 * 2 * 3, dtype=np.float32).reshape(4, 2, 3)
    filtered, result, counts = filter_factor_batch(
        factors,
        metadata,
        iter(
            [
                {
                    "action_hash": arrays["action_hash"],
                    "proposal": oracle_proposal_mask(arrays, 0),
                    "phase": 1,
                    "nature_token_available": True,
                    "independent_draft_winner": True,
                }
            ]
        ),
    )
    assert filtered.shape == (2, 2, 3)
    assert result["group_offsets"].tolist() == [0, 2]
    assert result["selected_index"].tolist() == [1]
    assert result["target"].tolist() == [False, True]
    assert result["phase"].tolist() == [1]
    assert result["nature_token_available"].tolist() == [True]
    assert result["independent_draft_winner"].tolist() == [True]
    assert counts == {
        "groups": 1,
        "original_actions": 4,
        "retained_actions": 2,
        "selected_outside_proposal": 0,
    }


def _report(
    *,
    train_recall: float,
    train_exact: float,
    validation_recall: float,
    pipeline: bool = True,
    phase_winner_recall: float = 1.0,
) -> dict[str, object]:
    metrics = {
        "all_groups_scored_once": pipeline,
        "all_candidates_scored_once": pipeline,
        "all_scores_finite": pipeline,
        "target_positive_recall": train_recall,
        "target_set_exact_fraction": train_exact,
        "top64_r4800_winner_recall": 1.0,
        "mean_top64_retained_r4800_regret": 0.0,
    }
    validation = dict(metrics)
    validation["target_positive_recall"] = validation_recall
    validation["phase"] = {
        name: {
            **metrics,
            "groups": 80,
            "top64_r4800_winner_recall": phase_winner_recall,
        }
        for name in ("early", "middle", "late")
    }
    validation["subsets"] = {
        name: {
            **metrics,
            "groups": 20,
        }
        for name in (
            "nature_token_available",
            "independent_draft_winner",
        )
    }
    return {
        "experiment_id": EXPERIMENT_ID,
        "train": metrics,
        "validation": validation,
        "finite_training": pipeline,
        "checkpoint_selection_uses_validation": False,
        "validation_evaluations": 1,
        "execution": {
            "peak_process_rss_bytes": 1,
            "process_swaps": 0,
        },
        "test_split_opened": False,
        "gameplay_opened": False,
        "new_teacher_compute_used": False,
        "external_compute_used": False,
    }


def test_classification_selects_smallest_feasible_architecture() -> None:
    reports = {
        "wide-concat": _report(
            train_recall=0.96,
            train_exact=0.51,
            validation_recall=0.91,
        ),
        "screen-relative": _report(
            train_recall=0.99,
            train_exact=0.90,
            validation_recall=0.95,
        ),
        "factor-attention": _report(
            train_recall=0.99,
            train_exact=0.90,
            validation_recall=0.95,
        ),
        "pairwise-gated": _report(
            train_recall=0.99,
            train_exact=0.90,
            validation_recall=0.95,
        ),
    }
    result = classify_reports(reports)
    assert result["classification"] == "oracle_proposal_selector_feasible"
    assert result["selected_kind"] == "wide-concat"


def test_pipeline_failure_precedes_feasibility() -> None:
    reports = {
        kind: _report(
            train_recall=1.0,
            train_exact=1.0,
            validation_recall=1.0,
            pipeline=kind != "factor-attention",
        )
        for kind in (
            "wide-concat",
            "screen-relative",
            "factor-attention",
            "pairwise-gated",
        )
    }
    result = classify_reports(reports)
    assert result["classification"] == "oracle_proposal_selector_pipeline_invalid"
    assert result["selected_kind"] is None


def test_phase_guardrail_is_part_of_selector_feasibility() -> None:
    reports = {
        kind: _report(
            train_recall=1.0,
            train_exact=1.0,
            validation_recall=1.0,
            phase_winner_recall=0.96,
        )
        for kind in (
            "wide-concat",
            "screen-relative",
            "factor-attention",
            "pairwise-gated",
        )
    }
    result = classify_reports(reports)
    assert result["classification"] == "oracle_proposal_selector_representation_insufficient"
    assert result["selected_kind"] is None


def _cache_manifest(split: str) -> dict[str, object]:
    manifest: dict[str, object] = {
        "cache_schema": "graded-oracle-oracle-proposal-factors-v1",
        "experiment_id": EXPERIMENT_ID,
        "split": split,
        "source_factor_payload_blake3": f"{split}-factor",
        "source_hierarchy_payload_blake3": f"{split}-hierarchy",
        "dataset_manifest_blake3": f"{split}-dataset",
        "factor_names": ["factor"],
        "factor_count": 1,
        "factor_dim": 1,
        "slice_metadata_fields": [
            "phase",
            "nature_token_available",
            "independent_draft_winner",
        ],
        "groups": 1,
        "original_candidates": 1,
        "candidates": 1,
        "selected_outside_proposal": 0,
        "batches": [],
    }
    manifest["payload_blake3"] = proposal_payload_blake3(manifest)
    return manifest


def _write_combine_fixture(
    artifact_root: Path,
    *,
    source_identity_matches: bool,
) -> None:
    train_cache = _cache_manifest("train")
    validation_cache = _cache_manifest("validation")
    for split, manifest in (
        ("train", train_cache),
        ("validation", validation_cache),
    ):
        path = artifact_root / "cache" / split
        path.mkdir(parents=True)
        (path / "cache.json").write_text(json.dumps(manifest))
    reports = artifact_root / "reports"
    reports.mkdir(parents=True)
    (reports / "source-identity-john1.json").write_text(
        json.dumps(
            {
                "identity_kind": "complete-mlx-runtime-source-v1",
                "bundle_sha256": "same",
            }
        )
    )
    (reports / "source-identity-john4.json").write_text(
        json.dumps(
            {
                "identity_kind": "complete-mlx-runtime-source-v1",
                "bundle_sha256": ("same" if source_identity_matches else "different"),
            }
        )
    )
    for kind in (
        "wide-concat",
        "screen-relative",
        "factor-attention",
        "pairwise-gated",
    ):
        report = _report(
            train_recall=1.0,
            train_exact=1.0,
            validation_recall=1.0,
        )
        report["train_cache_payload_blake3"] = train_cache["payload_blake3"]
        report["validation_cache_payload_blake3"] = validation_cache["payload_blake3"]
        run = artifact_root / "runs" / kind
        run.mkdir(parents=True)
        (run / "report.json").write_text(json.dumps(report))


def test_combine_invalidates_cross_host_source_drift(tmp_path: Path) -> None:
    _write_combine_fixture(tmp_path, source_identity_matches=False)
    result = combine(
        artifact_root=tmp_path,
        markdown_path=tmp_path / "result.md",
    )
    assert result["scientific"]["classification"] == "oracle_proposal_selector_pipeline_invalid"
    assert not result["scientific"]["campaign_pipeline"]["passed"]
