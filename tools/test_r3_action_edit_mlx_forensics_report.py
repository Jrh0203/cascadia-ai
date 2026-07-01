from __future__ import annotations

import json
from pathlib import Path

import pytest
from cascadia_mlx.r3_action_edit_mlx_cache import (
    ADR_ID,
    ARMS,
    EXPERIMENT_ID,
    PROTOCOL_ID,
)
from cascadia_mlx.r3_action_edit_mlx_forensics import (
    ATLAS_KIND,
    ATLAS_SCHEMA_VERSION,
)
from r3_action_edit_mlx_forensics_report import (
    _canonical_blake3,
    _checksum,
    _validate_atlas,
    compare_atlas_records,
)


def _record(
    group_id: int,
    *,
    confidence: bool,
    recall: bool,
    rank: int,
    global_tokens: int,
) -> dict:
    return {
        "group_id": group_id,
        "game_index": 0,
        "turn": group_id,
        "personal_turn": group_id // 4,
        "phase": "early",
        "low_supply": False,
        "independent_draft_winner": False,
        "candidate_count": 100,
        "winner_index": 0,
        "winner_r4800": 90.0,
        "winner_rank": rank,
        "winner_recalled_top64": recall,
        "confidence_set_covered_top64": confidence,
        "best_confidence_set_rank": rank,
        "top64_retained_r4800_regret": rank / 100.0,
        "winner_token_count": global_tokens + 7,
        "winner_local_token_count": 7,
        "winner_global_token_count": global_tokens,
    }


def test_compare_atlas_records_counts_radius_direction() -> None:
    records = {
        ARMS[0]: {
            1: _record(1, confidence=True, recall=True, rank=10, global_tokens=60),
            2: _record(2, confidence=False, recall=False, rank=80, global_tokens=60),
        },
        ARMS[1]: {
            1: _record(1, confidence=False, recall=False, rank=90, global_tokens=55),
            2: _record(2, confidence=False, recall=False, rank=90, global_tokens=55),
        },
        ARMS[2]: {
            1: _record(1, confidence=True, recall=True, rank=20, global_tokens=55),
            2: _record(2, confidence=False, recall=False, rank=80, global_tokens=55),
        },
        ARMS[3]: {
            1: _record(1, confidence=True, recall=True, rank=5, global_tokens=55),
            2: _record(2, confidence=True, recall=True, rank=10, global_tokens=55),
        },
    }

    report = compare_atlas_records(records)
    assert report["groups"] == 2
    assert report["treatment_patterns"]["confidence_coverage_t1_t2_t3"] == {
        "001": 1,
        "011": 1,
    }
    assert report["mechanism_counts"][
        "smaller_radius1_passes_while_radius3_fails"
    ] == 2
    assert report["mechanism_counts"][
        "larger_radius3_passes_while_radius1_fails"
    ] == 0


def test_compare_atlas_records_rejects_public_group_drift() -> None:
    records = {
        arm: {1: _record(1, confidence=True, recall=True, rank=1, global_tokens=50)}
        for arm in ARMS
    }
    records[ARMS[2]][1]["candidate_count"] = 101

    with pytest.raises(ValueError, match="public group facts differ"):
        compare_atlas_records(records)


def test_atlas_accepts_signed_64_bit_group_identity(
    tmp_path: Path,
) -> None:
    details = tmp_path / "details.jsonl"
    record = {
        "schema_version": ATLAS_SCHEMA_VERSION,
        "group_id": -5_482_088_856_184_735_585,
        "candidate_count": 7,
    }
    details.write_text(json.dumps(record) + "\n")
    report = {
        "schema_version": ATLAS_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "atlas_kind": ATLAS_KIND,
        "arm": ARMS[0],
        "classifier_eligible": False,
        "validation_groups": 1,
        "validation_candidates": 7,
        "scientific_identity": {
            "arm": ARMS[0],
            "details_blake3": _checksum(details),
        },
    }
    report["report_id"] = _canonical_blake3(report)
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))

    _, records = _validate_atlas(report_path, details)
    assert list(records) == [record["group_id"]]
