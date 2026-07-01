"""Open-split evaluation for the ADR 0091 target-only curriculum pilot."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import time
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx

from cascadia_mlx.graded_oracle_dataset import GradedOracleDataset
from cascadia_mlx.graded_oracle_frontier_anchor import evaluate_frontier_anchored
from cascadia_mlx.graded_oracle_frontier_target_curriculum_train import (
    EXPERIMENT_ID,
)
from cascadia_mlx.graded_oracle_model import (
    GradedOracleModelConfig,
    GradedOracleRanker,
)


def frontier_open_pilot_gates(report: dict[str, Any]) -> dict[str, bool]:
    """Apply the shared open-split proposer-pilot thresholds."""
    train = report["train"]
    validation = report["validation"]
    gates = {
        "train_target_positive_recall_at_least_0_60": (
            float(train["target_positive_recall"]) >= 0.60
        ),
        "train_target_set_exact_fraction_at_least_0_05": (
            float(train["target_set_exact_fraction"]) >= 0.05
        ),
        "validation_target_positive_recall_at_least_0_50": (
            float(validation["target_positive_recall"]) >= 0.50
        ),
        "validation_target_set_exact_fraction_at_least_0_01": (
            float(validation["target_set_exact_fraction"]) >= 0.01
        ),
        "validation_exact_winner_recall_at_least_0_75": (
            float(validation["top64_r4800_winner_recall"]) >= 0.75
        ),
        "validation_confidence_coverage_at_least_0_90": (
            float(validation["top64_confidence_set_coverage_95"]) >= 0.90
        ),
        "validation_retained_regret_below_0_15": (
            float(validation["mean_top64_retained_r4800_regret"]) < 0.15
        ),
        "all_train_groups_scored_once": bool(train["all_groups_scored_once"]),
        "all_train_candidates_scored_once": bool(
            train["all_candidates_scored_once"]
        ),
        "all_validation_groups_scored_once": bool(
            validation["all_groups_scored_once"]
        ),
        "all_validation_candidates_scored_once": bool(
            validation["all_candidates_scored_once"]
        ),
        "all_scores_finite": bool(train["all_scores_finite"])
        and bool(validation["all_scores_finite"]),
        "sealed_test_unopened": not bool(report["test_split_opened"]),
    }
    gates["pilot_passed"] = all(gates.values())
    return gates


target_curriculum_gates = frontier_open_pilot_gates


def evaluate_frontier_open_pilot(
    *,
    run_dir: Path,
    train_dataset: Path,
    validation_dataset: Path,
    expected_kind: str,
    experiment_id: str,
    output_name: str = "open-evaluation.json",
) -> dict[str, Any]:
    """Evaluate one selected proposer checkpoint on both open splits."""
    started = time.perf_counter()
    run = json.loads((run_dir / "run.json").read_text())
    if run.get("kind") != expected_kind:
        raise ValueError("frontier pilot run kind does not match")
    best = json.loads((run_dir / "best.json").read_text())
    checkpoint = run_dir / "checkpoints" / str(best["checkpoint"])
    manifest = json.loads((checkpoint / "checkpoint.json").read_text())
    model_path = checkpoint / "model.safetensors"
    metadata = manifest["files"]["model.safetensors"]
    if (
        model_path.stat().st_size != int(metadata["bytes"])
        or _checksum(model_path) != metadata["blake3"]
    ):
        raise ValueError("selected frontier-pilot model failed integrity validation")
    train = GradedOracleDataset(train_dataset)
    validation = GradedOracleDataset(validation_dataset)
    if train.split != "train" or validation.split != "validation":
        raise ValueError("frontier-pilot evaluation accepts only open splits")
    if _checksum(train.root / "dataset.json") != run["datasets"]["train_manifest_blake3"]:
        raise ValueError("frontier-pilot train dataset identity drifted")
    if (
        _checksum(validation.root / "dataset.json")
        != run["datasets"]["validation_manifest_blake3"]
    ):
        raise ValueError("frontier-pilot validation dataset identity drifted")
    model = GradedOracleRanker(
        GradedOracleModelConfig.from_dict(manifest["model_config"])
    )
    model.load_weights(str(model_path))
    mx.eval(model.parameters())
    train_metrics = evaluate_frontier_anchored(model, train, group_batch_size=64)
    validation_metrics = evaluate_frontier_anchored(
        model,
        validation,
        group_batch_size=64,
    )
    scientific = {
        "checkpoint": checkpoint.name,
        "checkpoint_manifest_blake3": _checksum(checkpoint / "checkpoint.json"),
        "model_blake3": _checksum(model_path),
        "train_dataset_id": train.manifest["dataset_id"],
        "train_manifest_blake3": _checksum(train.root / "dataset.json"),
        "validation_dataset_id": validation.manifest["dataset_id"],
        "validation_manifest_blake3": _checksum(
            validation.root / "dataset.json"
        ),
        "train": train_metrics,
        "validation": validation_metrics,
        "test_split_opened": False,
    }
    gates = frontier_open_pilot_gates(scientific)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss = int(usage.ru_maxrss)
    if platform.system() != "Darwin":
        peak_rss *= 1024
    report = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "evaluation_kind": "open-train-validation",
        "scientific": scientific,
        "scientific_blake3": _canonical_digest(scientific),
        "gates": gates,
        "passed": gates["pilot_passed"],
        "execution": {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_process_rss_bytes": peak_rss,
            "process_swaps": int(getattr(usage, "ru_nswap", 0)),
        },
    }
    _write_json_atomic(run_dir / output_name, report)
    return report


def evaluate_target_curriculum(
    *,
    run_dir: Path,
    train_dataset: Path,
    validation_dataset: Path,
) -> dict[str, Any]:
    """Evaluate the selected ADR 0091 checkpoint on both open splits."""
    return evaluate_frontier_open_pilot(
        run_dir=run_dir,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        expected_kind="graded-oracle-frontier-target-curriculum",
        experiment_id=EXPERIMENT_ID,
    )


def _checksum(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(value: object) -> str:
    return blake3.blake3(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_target_curriculum(
        run_dir=args.run_dir,
        train_dataset=args.train_dataset,
        validation_dataset=args.validation_dataset,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
