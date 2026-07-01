from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

_MODULE_PATH = Path(__file__).with_name("feature_schema_activation_census_report.py")
_SPEC = importlib.util.spec_from_file_location(
    "feature_schema_activation_census_report",
    _MODULE_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
reporter = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = reporter
_SPEC.loader.exec_module(reporter)


def _phase_seat() -> dict:
    return {
        phase: {
            **{
                seat: {
                    "rows": 1,
                    "active_rows": 1,
                    "nonzero_values": 1,
                    "values": 1,
                }
                for seat in reporter.SEATS
            },
            "unknown": {
                "rows": 0,
                "active_rows": 0,
                "nonzero_values": 0,
                "values": 0,
            },
        }
        for phase in reporter.PHASES
    }


def _block(block_id: str, schema: str, rows: int = 1) -> dict:
    return {
        "block_id": block_id,
        "schema": schema,
        "implementation_status": "implemented",
        "width": 1,
        "census": {
            "rows": rows,
            "dead_channel_count": 0,
            "constant_channel_count": 0,
            "rare_channel_count": 0,
            "collision_status": "no_channel_alias_detected",
            "phase_seat": _phase_seat(),
        },
    }


def _fixture() -> tuple[dict, dict]:
    blocks = [
        _block("v2.board.coordinates", "compact-entity-v2"),
        *[
            _block(f"modern.{index}", "compact-entity-v2")
            for index in range(reporter.EXPECTED_MODERN_BLOCKS - 1)
        ],
        _block(
            "legacy.mid_tail_historical_adjacency_prefix",
            "legacy-mid-v4opp-11231",
            reporter.EXPECTED_LEGACY_ROWS,
        ),
        _block(
            "legacy.cell_core",
            "legacy-mid-v4opp-11231",
            reporter.EXPECTED_LEGACY_ROWS,
        ),
        *[
            _block(
                f"legacy.{index}",
                "legacy-mid-v4opp-11231",
                reporter.EXPECTED_LEGACY_ROWS,
            )
            for index in range(reporter.EXPECTED_LEGACY_BLOCKS - 2)
        ],
        {
            "block_id": "future.one",
            "schema": "future-v0",
            "implementation_status": "unimplemented",
            "width": 1,
            "census": {"rows": 0, "status": ["unimplemented_unmeasurable"]},
        },
    ]
    blocks[1]["census"]["collision_status"] = "unknown"
    evidence = [
        {
            "evidence_id": f"evidence:{index}",
            "kind": "other",
            "split": "train",
            "rows_scanned": 0,
        }
        for index in range(reporter.EXPECTED_EVIDENCE - 7)
    ]
    evidence.extend(
        [
            {
                "evidence_id": "graded:train",
                "kind": "graded_dataset_shard",
                "split": "train",
                "rows_scanned": 2_135_111,
            },
            {
                "evidence_id": "graded:validation",
                "kind": "graded_dataset_shard",
                "split": "validation",
                "rows_scanned": 860_203,
            },
            {
                "evidence_id": "factor:train",
                "kind": "candidate_factor_cache_batch",
                "split": "train",
                "rows_scanned": 2_135_111,
            },
            {
                "evidence_id": "factor:validation",
                "kind": "candidate_factor_cache_batch",
                "split": "validation",
                "rows_scanned": 860_203,
            },
            {
                "evidence_id": "hierarchical:train",
                "kind": "hierarchical_factor_cache_shard",
                "split": "train",
                "rows_scanned": 3_839_415,
            },
            {
                "evidence_id": "hierarchical:validation",
                "kind": "hierarchical_factor_cache_shard",
                "split": "validation",
                "rows_scanned": 1_557_653,
            },
            {
                "evidence_id": "legacy:train",
                "kind": "legacy_sparse_feature_shard",
                "split": "train",
                "rows_scanned": reporter.EXPECTED_LEGACY_ROWS,
            },
        ]
    )
    scientific = {
        "experiment_id": reporter.EXPERIMENT_ID,
        "manifest_scientific_blake3": reporter.EXPECTED_MANIFEST_SCIENTIFIC_BLAKE3,
        "config": {"merged_shards": [[0, 4], [1, 4], [2, 4], [3, 4]]},
        "evidence": evidence,
        "blocks": blocks,
        "closed_domains": {
            "test_split_opened": False,
            "gameplay_opened": False,
            "new_teacher_compute_used": False,
            "external_compute_used": False,
            "hidden_teacher_values_used_as_features": False,
        },
    }
    result = {
        "scientific": scientific,
        "scientific_blake3": reporter.scientific_blake3(scientific),
    }
    manifest_payload = {
        "experiment_id": reporter.EXPERIMENT_ID,
        "schemas": [],
    }
    manifest = {
        **manifest_payload,
        "scientific_blake3": reporter.scientific_blake3(manifest_payload),
    }
    return result, manifest


def test_complete_classification_passes_all_gates(monkeypatch) -> None:
    result, manifest = _fixture()
    monkeypatch.setattr(
        reporter,
        "EXPECTED_MANIFEST_SCIENTIFIC_BLAKE3",
        manifest["scientific_blake3"],
    )
    result["scientific"]["manifest_scientific_blake3"] = manifest["scientific_blake3"]
    result["scientific_blake3"] = reporter.scientific_blake3(result["scientific"])
    classified = reporter.classify(result, copy.deepcopy(result), manifest)
    assert classified["scientific"]["classification"] == reporter.COMPLETE
    assert classified["scientific"]["complete"] is True
    assert all(
        gate["passed"] for gate in classified["scientific"]["gates"].values()
    )


def test_missing_legacy_measurement_fails_closed(monkeypatch) -> None:
    result, manifest = _fixture()
    monkeypatch.setattr(
        reporter,
        "EXPECTED_MANIFEST_SCIENTIFIC_BLAKE3",
        manifest["scientific_blake3"],
    )
    result["scientific"]["manifest_scientific_blake3"] = manifest["scientific_blake3"]
    result["scientific"]["blocks"][-2]["census"]["rows"] = 0
    result["scientific_blake3"] = reporter.scientific_blake3(result["scientific"])
    classified = reporter.classify(result, copy.deepcopy(result), manifest)
    assert classified["scientific"]["classification"] == reporter.INCOMPLETE
    assert classified["scientific"]["gates"]["all_implemented_blocks_measured"][
        "passed"
    ] is False


def test_merge_order_scientific_drift_fails_closed(monkeypatch) -> None:
    result, manifest = _fixture()
    monkeypatch.setattr(
        reporter,
        "EXPECTED_MANIFEST_SCIENTIFIC_BLAKE3",
        manifest["scientific_blake3"],
    )
    result["scientific"]["manifest_scientific_blake3"] = manifest["scientific_blake3"]
    result["scientific_blake3"] = reporter.scientific_blake3(result["scientific"])
    reverse = copy.deepcopy(result)
    reverse["scientific"]["config"]["unexpected"] = True
    reverse["scientific_blake3"] = reporter.scientific_blake3(reverse["scientific"])
    classified = reporter.classify(result, reverse, manifest)
    assert classified["scientific"]["classification"] == reporter.INCOMPLETE
    assert classified["scientific"]["gates"]["merge_order_determinism"]["passed"] is False
