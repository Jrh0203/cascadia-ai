#!/usr/bin/env python3
"""Evaluate ADR 0078's frozen checkpoint on ADR 0079's sealed test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
from cascadia_mlx.checkpoint import load_checkpoint_pointer_with_factory
from cascadia_mlx.counterfactual_advantage_dataset import (
    CounterfactualAdvantageDataset,
)
from cascadia_mlx.counterfactual_advantage_evaluate import evaluate_validation_run
from cascadia_mlx.counterfactual_advantage_model import (
    CounterfactualAdvantageModelConfig,
    CounterfactualAdvantageRanker,
)
from cascadia_mlx.counterfactual_advantage_train import (
    evaluate_counterfactual_advantage,
)
from cascadia_mlx.run_manifest import source_provenance

EXPERIMENT = "r12-counterfactual-advantage-set-ranker-v1-sealed-test-20260613"
PARENT_EXPERIMENT = "r12-counterfactual-advantage-set-ranker-v1-20260613"
EXPECTED_FIRST_GAME_INDEX = 71_000
EXPECTED_GAMES = 32
MIN_OBJECTIVE_IMPROVEMENT = 0.10
MAX_CENTERED_MAE = 0.75
MIN_CENTERED_MAE_IMPROVEMENT = 0.10
MIN_CENTERED_CORRELATION = 0.55
MIN_TOP_VALUE_RECALL = 0.50
MIN_RECALL_GAIN_OVER_H6 = 0.05
MAX_MEAN_REGRET = 0.40
MIN_REGRET_REDUCTION_OVER_H6 = 0.05


def evaluate_test_run(
    run_dir: str | Path,
    dataset_root: str | Path,
    validation_report_path: str | Path,
    authorization_path: str | Path,
    *,
    group_batch_size: int = 32,
) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    dataset_root = Path(dataset_root).resolve()
    validation_report_path = Path(validation_report_path).resolve()
    authorization_path = Path(authorization_path).resolve()
    validation_report = _read_json(validation_report_path, "validation report")
    authorization = _read_json(authorization_path, "test authorization")
    run = _read_json(run_dir / "run.json", "run manifest")
    best = _read_json(run_dir / "best.json", "best pointer")

    if validation_report.get("experiment") != PARENT_EXPERIMENT:
        raise ValueError("test authorization references the wrong parent experiment")
    if not validation_report.get("passed") or validation_report.get("failed_gates"):
        raise ValueError("sealed test access denied because ADR 0078 validation failed")
    if validation_report.get("domain") != "validation":
        raise ValueError("sealed test access requires an ADR 0078 validation report")
    if validation_report.get("test_domain_opened") is not False:
        raise ValueError("ADR 0078 validation report already opened the test domain")
    if run.get("kind") != "r12-counterfactual-advantage-set-ranking":
        raise ValueError("run is not the frozen ADR 0078 set ranker")

    dataset = CounterfactualAdvantageDataset(dataset_root)
    _validate_test_manifest(dataset.manifest)
    if dataset.manifest["teacher"] != run["datasets"]["teacher"]:
        raise ValueError("test teacher differs from the frozen training teacher")

    training = run["training"]
    model, _optimizer, _state, checkpoint = load_checkpoint_pointer_with_factory(
        run_dir,
        pointer="best",
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        model_factory=lambda values: CounterfactualAdvantageRanker(
            CounterfactualAdvantageModelConfig.from_dict(values)
        ),
    )
    checkpoint_manifest = checkpoint / "checkpoint.json"
    checkpoint_blake3 = _blake3(checkpoint_manifest)
    if checkpoint.name != best.get("checkpoint"):
        raise ValueError("best pointer changed before sealed-test evaluation")

    initial_model = CounterfactualAdvantageRanker(model.config)
    initial_metrics = evaluate_counterfactual_advantage(
        initial_model,
        dataset,
        group_batch_size,
    )
    metrics = evaluate_counterfactual_advantage(model, dataset, group_batch_size)

    replay = evaluate_validation_run(
        run_dir,
        validation_report["dataset"],
        group_batch_size=group_batch_size,
    )
    validation_replay_exact = replay == validation_report
    validation_sha256 = _sha256(validation_report_path)
    gates = evaluate_gates(
        metrics=metrics,
        initial_metrics=initial_metrics,
        validation_report=validation_report,
        validation_replay_exact=validation_replay_exact,
        checkpoint_name=checkpoint.name,
        checkpoint_blake3=checkpoint_blake3,
        authorization=authorization,
        validation_report_sha256=validation_sha256,
        test_created_unix_seconds=int(dataset.manifest["created_unix_seconds"]),
        source_matches=(
            source_provenance(Path(__file__).resolve().parents[1])["v2_source_blake3"]
            == run["source"]["v2_source_blake3"]
        ),
    )
    failed = [name for name, passed in gates.items() if not passed]
    return {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "parent_experiment": PARENT_EXPERIMENT,
        "domain": "test",
        "run_dir": str(run_dir),
        "dataset": str(dataset_root),
        "dataset_id": dataset.manifest["dataset_id"],
        "dataset_manifest_blake3": _blake3(dataset_root / "dataset.json"),
        "validation_report": str(validation_report_path),
        "validation_report_sha256": validation_sha256,
        "authorization": str(authorization_path),
        "authorization_sha256": _sha256(authorization_path),
        "checkpoint": checkpoint.name,
        "checkpoint_manifest_blake3": checkpoint_blake3,
        "evaluator_blake3": _blake3(Path(__file__)),
        "device": str(mx.default_device()),
        "initial_test": initial_metrics,
        "test": metrics,
        "validation_replay_exact": validation_replay_exact,
        "gates": gates,
        "failed_gates": failed,
        "passed": not failed,
        "test_domain_opened": True,
        "gameplay_domain_opened": False,
    }


def evaluate_gates(
    *,
    metrics: dict[str, Any],
    initial_metrics: dict[str, Any],
    validation_report: dict[str, Any],
    validation_replay_exact: bool,
    checkpoint_name: str,
    checkpoint_blake3: str,
    authorization: dict[str, Any],
    validation_report_sha256: str,
    test_created_unix_seconds: int,
    source_matches: bool,
) -> dict[str, bool]:
    h6 = metrics["h6_selected_baseline"]
    absent = authorization.get("test_absent_on_nodes", {})
    return {
        "validation_passed_before_test_collection": (
            authorization.get("experiment") == EXPERIMENT
            and authorization.get("parent_experiment") == PARENT_EXPERIMENT
            and authorization.get("validation_passed") is True
            and authorization.get("validation_report_sha256") == validation_report_sha256
            and bool(absent)
            and all(value is True for value in absent.values())
            and int(authorization.get("authorized_at_unix_seconds", 0)) <= test_created_unix_seconds
        ),
        "validation_checkpoint_is_unchanged": (
            checkpoint_name == validation_report.get("checkpoint")
            and checkpoint_blake3 == validation_report.get("checkpoint_manifest_blake3")
        ),
        "validation_replay_is_bit_exact": validation_replay_exact,
        "mlx_gpu_device": str(mx.default_device()) == "Device(gpu, 0)",
        "source_matches_training_run": source_matches,
        "test_objective_improves_at_least_10_percent": (
            metrics["decision_objective"]
            <= (1.0 - MIN_OBJECTIVE_IMPROVEMENT) * initial_metrics["decision_objective"]
        ),
        "centered_mae_at_most_0_75": (metrics["centered_mean_absolute_error"] <= MAX_CENTERED_MAE),
        "centered_mae_improves_at_least_10_percent": (
            metrics["centered_mean_absolute_error"]
            <= (1.0 - MIN_CENTERED_MAE_IMPROVEMENT)
            * initial_metrics["centered_mean_absolute_error"]
        ),
        "centered_correlation_at_least_0_55": (
            metrics["centered_advantage_correlation"] >= MIN_CENTERED_CORRELATION
        ),
        "top_value_recall_at_least_0_50": (metrics["top_value_recall"] >= MIN_TOP_VALUE_RECALL),
        "top_value_recall_at_least_h6_plus_0_05": (
            metrics["top_value_recall"] >= h6["top_value_recall"] + MIN_RECALL_GAIN_OVER_H6
        ),
        "mean_regret_at_most_0_40": (metrics["mean_top_action_regret"] <= MAX_MEAN_REGRET),
        "mean_regret_at_least_0_05_below_h6": (
            metrics["mean_top_action_regret"]
            <= h6["mean_top_action_regret"] - MIN_REGRET_REDUCTION_OVER_H6
        ),
    }


def _validate_test_manifest(manifest: dict[str, Any]) -> None:
    expected = {
        "split": "test",
        "first_game_index": EXPECTED_FIRST_GAME_INDEX,
        "requested_games": EXPECTED_GAMES,
        "completed_games": EXPECTED_GAMES,
        "maximum_candidates": 4,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"test manifest {key}={manifest.get(key)!r}, expected {value!r}")
    teacher = manifest.get("teacher", {})
    teacher_expected = {
        "groups_per_game": 16,
        "samples_per_candidate": 12,
        "candidate_count": 4,
        "candidate_selection": "selected-high-median-low-v1",
        "stabilization_conditioning": "reject-unstable-market-trajectories-v1",
    }
    for key, value in teacher_expected.items():
        if teacher.get(key) != value:
            raise ValueError(f"test teacher {key}={teacher.get(key)!r}, expected {value!r}")


def _markdown(report: dict[str, Any]) -> str:
    test = report["test"]
    initial = report["initial_test"]
    h6 = test["h6_selected_baseline"]
    rows = [
        ("Test objective", test["decision_objective"], initial["decision_objective"]),
        (
            "Centered MAE",
            test["centered_mean_absolute_error"],
            initial["centered_mean_absolute_error"],
        ),
        (
            "Centered correlation",
            test["centered_advantage_correlation"],
            initial["centered_advantage_correlation"],
        ),
        ("Top-value recall", test["top_value_recall"], h6["top_value_recall"]),
        (
            "Mean top-action regret",
            test["mean_top_action_regret"],
            h6["mean_top_action_regret"],
        ),
    ]
    lines = [
        "# ADR 0079 R12 Set-Ranker Sealed Test",
        "",
        f"Status: **{'passed' if report['passed'] else 'rejected'} on sealed test**.",
        "",
        f"Checkpoint: `{report['checkpoint']}`",
        "",
        "| Metric | Model | Frozen comparison |",
        "|---|---:|---:|",
    ]
    lines.extend(f"| {name} | {value:.6f} | {baseline:.6f} |" for name, value, baseline in rows)
    lines.extend(["", "## Gates", ""])
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'}: `{name}`" for name, passed in report["gates"].items()
    )
    lines.extend(["", "Gameplay and promotion remain closed.", ""])
    return "\n".join(lines)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def _blake3(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--group-batch-size", type=int, default=32)
    args = parser.parse_args()
    report = evaluate_test_run(
        args.run_dir,
        args.dataset,
        args.validation_report,
        args.authorization,
        group_batch_size=args.group_batch_size,
    )
    _write_json_atomic(args.output, report)
    if args.markdown_output is not None:
        _write_text_atomic(args.markdown_output, _markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
