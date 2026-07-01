from __future__ import annotations

import importlib.util
import json
import subprocess
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
    fingerprints = {
        "binary_sha256": "frozen-binary",
        "model_manifest_sha256": "frozen-manifest",
        "model_safetensors_sha256": "frozen-model",
        "weights_sha256": "frozen-weights",
    }
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
            "fingerprints": fingerprints,
            "completed_at": f"2026-06-14T00:00:{seed % 60:02d}+00:00",
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
                "fingerprints": fingerprints,
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


def test_macos_sleep_guard_wakes_and_holds_until_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_calls: list[list[str]] = []
    popen_calls: list[list[str]] = []

    class FakeGuard:
        terminated = False
        waited = False

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout: int) -> int:
            assert timeout == 5
            self.waited = True
            return 0

        def kill(self) -> None:
            raise AssertionError("healthy guard must not be killed")

    guard = FakeGuard()

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        run_calls.append(command)
        assert kwargs["check"] is True
        return subprocess.CompletedProcess(command, 0)

    def fake_popen(command: list[str], **kwargs: object) -> FakeGuard:
        popen_calls.append(command)
        return guard

    monkeypatch.setattr(MODULE.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(MODULE.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(MODULE.subprocess, "run", fake_run)
    monkeypatch.setattr(MODULE.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(MODULE.os, "getpid", lambda: 1234)

    with MODULE.macos_sleep_guard() as mode:
        assert mode == "macos-caffeinate-ims-v1"
        assert guard.terminated is False

    assert run_calls == [["/usr/bin/caffeinate", "-u", "-t", "5"]]
    assert popen_calls == [["/usr/bin/caffeinate", "-ims", "-w", "1234"]]
    assert guard.terminated is True
    assert guard.waited is True


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
    assert "| Bear | 12.000 | 12.000 | +0.000 |" in markdown.read_text()
    assert "| P90 decision | 1500.0 ms | 10.0 ms |" in markdown.read_text()


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
