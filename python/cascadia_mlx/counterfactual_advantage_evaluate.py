"""Apply ADR 0078 validation gates to the selected R12 checkpoint."""

from __future__ import annotations

import argparse
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
from cascadia_mlx.counterfactual_advantage_model import (
    CounterfactualAdvantageModelConfig,
    CounterfactualAdvantageRanker,
)
from cascadia_mlx.counterfactual_advantage_train import (
    evaluate_counterfactual_advantage,
)
from cascadia_mlx.run_manifest import source_provenance

MIN_OBJECTIVE_IMPROVEMENT = 0.10
MAX_CENTERED_MAE = 0.75
MIN_CENTERED_MAE_IMPROVEMENT = 0.10
MIN_CENTERED_CORRELATION = 0.55
MIN_TOP_VALUE_RECALL = 0.50
MIN_RECALL_GAIN_OVER_H6 = 0.05
MAX_MEAN_REGRET = 0.40
MIN_REGRET_REDUCTION_OVER_H6 = 0.05


def evaluate_validation_run(
    run_dir: str | Path,
    dataset_root: str | Path,
    *,
    group_batch_size: int = 32,
) -> dict[str, Any]:
    """Load the integrity-checked best checkpoint and apply every frozen gate."""
    run_dir = Path(run_dir).resolve()
    dataset_root = Path(dataset_root).resolve()
    try:
        run = json.loads((run_dir / "run.json").read_text())
        best = json.loads((run_dir / "best.json").read_text())
        initial = json.loads((run_dir / "initial-validation.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read counterfactual-advantage run: {error}") from error
    if run.get("kind") != "r12-counterfactual-advantage-set-ranking":
        raise ValueError("run is not an ADR 0078 counterfactual-advantage ranker")
    dataset = CounterfactualAdvantageDataset(dataset_root)
    if dataset.split != "validation":
        raise ValueError("ADR 0078 evaluator requires the validation split")
    expected_manifest = run["datasets"]["validation_manifest_blake3"]
    if _checksum(dataset_root / "dataset.json") != expected_manifest:
        raise ValueError("validation dataset does not match the frozen run manifest")
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
    if checkpoint.name != best.get("checkpoint"):
        raise ValueError("best pointer changed while loading the selected checkpoint")

    metrics = evaluate_counterfactual_advantage(model, dataset, group_batch_size)
    initial_metrics = initial["validation"]
    h6 = metrics["h6_selected_baseline"]
    gates = {
        "mlx_gpu_device": str(mx.default_device()) == "Device(gpu, 0)",
        "validation_objective_improves_at_least_10_percent": (
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
        "selected_metrics_match_best_pointer": _selected_metrics_match(metrics, best),
        "source_matches_training_run": (
            source_provenance(Path(__file__).resolve().parents[2])["v2_source_blake3"]
            == run["source"]["v2_source_blake3"]
        ),
    }
    failed = [name for name, passed in gates.items() if not passed]
    return {
        "schema_version": 1,
        "experiment": "r12-counterfactual-advantage-set-ranker-v1-20260613",
        "domain": "validation",
        "run_dir": str(run_dir),
        "dataset": str(dataset_root),
        "dataset_id": dataset.manifest["dataset_id"],
        "dataset_manifest_blake3": expected_manifest,
        "checkpoint": checkpoint.name,
        "checkpoint_manifest_blake3": _checksum(checkpoint / "checkpoint.json"),
        "device": str(mx.default_device()),
        "initial_validation": initial_metrics,
        "validation": metrics,
        "gates": gates,
        "failed_gates": failed,
        "passed": not failed,
        "test_domain_opened": False,
        "gameplay_domain_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--group-batch-size", type=int, default=32)
    args = parser.parse_args()
    report = evaluate_validation_run(
        args.run_dir,
        args.dataset,
        group_batch_size=args.group_batch_size,
    )
    _write_json_atomic(args.output, report)
    if args.markdown_output is not None:
        _write_text_atomic(args.markdown_output, _markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))


def _selected_metrics_match(metrics: dict[str, Any], best: dict[str, Any]) -> bool:
    selected = best.get("validation", {})
    keys = (
        "decision_objective",
        "centered_mean_absolute_error",
        "centered_advantage_correlation",
        "top_value_recall",
        "mean_top_action_regret",
    )
    return all(abs(float(metrics[key]) - float(selected[key])) <= 1e-9 for key in keys)


def _markdown(report: dict[str, Any]) -> str:
    validation = report["validation"]
    initial = report["initial_validation"]
    h6 = validation["h6_selected_baseline"]
    rows = [
        ("Validation objective", validation["decision_objective"], initial["decision_objective"]),
        (
            "Centered MAE",
            validation["centered_mean_absolute_error"],
            initial["centered_mean_absolute_error"],
        ),
        (
            "Centered correlation",
            validation["centered_advantage_correlation"],
            initial["centered_advantage_correlation"],
        ),
        ("Top-value recall", validation["top_value_recall"], h6["top_value_recall"]),
        (
            "Mean top-action regret",
            validation["mean_top_action_regret"],
            h6["mean_top_action_regret"],
        ),
    ]
    lines = [
        "# ADR 0078 R12 Counterfactual-Advantage Set Ranker",
        "",
        f"Status: **{'passed' if report['passed'] else 'rejected'} on validation**.",
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
    lines.extend(
        [
            "",
            "Test and gameplay domains remain closed.",
            "",
        ]
    )
    return "\n".join(lines)


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


def _checksum(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
