from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[2] / "tools" / "final_strength_benchmark.py"
SPEC = importlib.util.spec_from_file_location("final_strength_benchmark", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def score(total: int) -> dict[str, object]:
    return {
        "habitat": [6, 6, 6, 6, 6],
        "wildlife": [12, 12, 12, 12, total - 82],
        "nature_tokens": 4,
        "habitat_bonus": [0, 0, 0, 0, 0],
        "base_total": total,
        "total": total,
    }


def game_report(seed: int, baseline: list[int], treatment: list[int]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "smoke-passed",
        "seed_domain": "final",
        "rollouts": 600,
        "clean_shutdown": True,
        "gates": {"smoke_passed": True},
        "comparison": {
            "protocol_id": MODULE.PROTOCOL_ID,
            "baseline_id": MODULE.BASELINE_ID,
            "treatment_id": MODULE.TREATMENT_ID,
            "games": 1,
            "first_seed": seed,
        },
        "game_records": [
            {
                "seed": seed,
                "game_seed": [seed % 256] * 32,
                "baseline_scores": [score(value) for value in baseline],
                "treatment_scores": [score(value) for value in treatment],
                "baseline_decision_seconds": [0.01] * 80,
                "treatment_decision_seconds": [1.5] * 80,
                "baseline_elapsed_seconds": 1.0,
                "treatment_elapsed_seconds": 120.0,
            }
        ],
    }


def write_shard(path: Path, reports: list[dict[str, object]]) -> None:
    games = path / "games"
    games.mkdir(parents=True)
    seeds = []
    for report in reports:
        seed = int(report["game_records"][0]["seed"])  # type: ignore[index]
        seeds.append(seed)
        report_path = games / f"{seed}.json"
        report_path.write_text(json.dumps(report))
        metadata = {
            "seed": seed,
            "host": f"host-{seed % 2}",
            "source_revision": "abc123",
            "fingerprints": {"binary_sha256": "frozen"},
            "report_sha256": MODULE.sha256_file(report_path),
        }
        (games / f"{seed}.meta.json").write_text(json.dumps(metadata))
    (path / "shard.json").write_text(
        json.dumps(
            {
                "complete": True,
                "experiment_id": MODULE.EXPERIMENT_ID,
                "protocol_id": MODULE.PROTOCOL_ID,
                "rollouts": 600,
                "fingerprints": {"binary_sha256": "frozen"},
                "completed_seeds": seeds,
            }
        )
    )


def test_validate_game_report_requires_raw_four_seat_record() -> None:
    report = game_report(10, [90, 91, 92, 93], [95, 96, 97, 98])
    MODULE.validate_game_report(report, 10, 600)
    report["game_records"][0]["treatment_scores"].pop()  # type: ignore[index]
    with pytest.raises(ValueError, match="four seat"):
        MODULE.validate_game_report(report, 10, 600)


def test_aggregate_computes_game_block_statistics_and_full_coverage(tmp_path: Path) -> None:
    shard_a = tmp_path / "a"
    shard_b = tmp_path / "b"
    write_shard(
        shard_a,
        [game_report(100, [90, 90, 90, 90], [98, 98, 98, 98])],
    )
    write_shard(
        shard_b,
        [game_report(101, [92, 92, 92, 92], [102, 102, 102, 102])],
    )
    output = tmp_path / "report.json"
    markdown = tmp_path / "report.md"
    MODULE.aggregate([shard_a, shard_b], 100, 2, output, markdown, None)
    report = json.loads(output.read_text())
    assert report["treatment"]["mean_score"] == 100.0
    assert report["paired_baseline"]["mean_score"] == 91.0
    assert report["paired_delta"]["mean"] == 9.0
    assert report["target"]["reached"] is True
    assert report["integrity"]["complete_seed_suite"] is True
    assert report["host_game_counts"] == {"host-0": 1, "host-1": 1}
    assert "Mean base score: **100.000**" in markdown.read_text()


def test_aggregate_rejects_missing_seed(tmp_path: Path) -> None:
    shard = tmp_path / "shard"
    write_shard(shard, [game_report(100, [90] * 4, [95] * 4)])
    with pytest.raises(ValueError, match="coverage mismatch"):
        MODULE.aggregate(
            [shard],
            100,
            2,
            tmp_path / "report.json",
            tmp_path / "report.md",
            None,
        )
