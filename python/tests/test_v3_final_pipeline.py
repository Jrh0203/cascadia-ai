from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest


def _module() -> Any:
    path = Path(__file__).resolve().parents[2] / "tools/v3_final_pipeline.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("v3_final_pipeline_tested", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def _cycle(root: Path, cycle: int) -> None:
    directory = root / f"phase2/cycles/cycle-{cycle:02d}"
    _write(
        directory / "corpus.json",
        {
            "passed": True,
            "cycle": cycle,
            "games": 10_000,
            "training_entries": 200_000,
            "canonical_sha256": "a" * 64,
        },
    )
    _write(
        directory / "teacher-label-corpus.json",
        {
            "passed": True,
            "cycle": cycle,
            "roots": 2_500,
            "candidate_estimates": 70_000,
            "rollouts": 1_500_000,
            "canonical_sha256": "b" * 64,
        },
    )
    candidates = [
        {
            "label": f"cycle-{cycle:02d}-origin-{origin}",
            "eligible": True,
            "quantized_validation_loss": 0.01 + origin / 100,
            "open_game_mean": 85 + origin,
            "training_report": {
                "examples_seen": 1_200_000,
                "elapsed_seconds": 100,
                "latest_loss": {"loss": 0.001},
            },
        }
        for origin in (1, 2)
    ]
    _write(
        directory / "training/selection.json",
        {
            "passed": True,
            "cycle": cycle,
            "selected": candidates[1]["label"],
            "candidates": candidates,
        },
    )
    _write(
        directory / "training/candidate.json",
        {
            "passed": True,
            "selected_origin": candidates[1]["label"],
            "model_manifest_sha256": "c" * 64,
            "weights_sha256": "d" * 64,
            "parity_report_sha256": "e" * 64,
        },
    )
    for origin in (1, 2):
        _write(
            directory / f"training/origin-{origin}/loss.json",
            {"samples": [{"step": 1, "loss": 0.01}, {"step": 2, "loss": 0.001}]},
        )
    _write(
        directory / "promotion/report.json",
        {
            "passed": True,
            "cycle": cycle,
            "verdict": "retain-incumbent-inconclusive",
            "pairs_per_tier": 500,
            "tiers": {
                tier: {"mean_delta": 0.1}
                for tier in ("direct", "k32-r64", "k32-r600", "equal-wall-time")
            },
        },
    )
    _write(
        directory / "promotion/champion.json",
        {"passed": True, "cycle": cycle, "promoted": False},
    )


def test_campaign_history_requires_and_preserves_all_ten_cycles(tmp_path: Path) -> None:
    module = _module()
    module.ROOT = tmp_path
    for cycle in range(1, 11):
        _cycle(tmp_path, cycle)
    history = module._campaign_history()
    assert [item["cycle"] for item in history] == list(range(1, 11))
    assert all(len(item["loss_curves"]) == 2 for item in history)
    assert history[-1]["promotion"]["pairs_per_tier"] == 500


def test_campaign_history_rejects_missing_cycle(tmp_path: Path) -> None:
    module = _module()
    module.ROOT = tmp_path
    for cycle in range(1, 10):
        _cycle(tmp_path, cycle)
    with pytest.raises(module.FinalPipelineError, match="cycle 10 is missing"):
        module._campaign_history()


def test_swap_parser_uses_binary_units() -> None:
    module = _module()
    assert module._parse_swap_used("total = 10M used = 8.25M free = 1.75M") == round(
        8.25 * 1024**2
    )


def test_all_v3_aggregate_excludes_out_of_domain_elapsed_time(tmp_path: Path) -> None:
    module = _module()
    module.FINAL = tmp_path / "final"
    accepted = module.FINAL / "all-v3-accepted/request"
    seats = [{"decision_seconds": 1.0} for _ in range(4)]
    _write(
        accepted / "in-domain/all-v3.json",
        {
            "elapsed_seconds": 5.0,
            "records": [
                {"game_index": 0, "seats": seats},
                {"game_index": 1, "seats": seats},
            ],
        },
    )
    _write(
        accepted / "extension/all-v3.json",
        {
            "elapsed_seconds": 7.0,
            "records": [
                {"game_index": 1_000, "seats": seats},
                {"game_index": 1_001, "seats": seats},
            ],
        },
    )
    output = module._aggregate_all_v3(2)
    assert json.loads(output.read_text())["resource_metrics"] == {
        "decision_seconds": 8.0,
        "worker_elapsed_seconds": 5.0,
    }
