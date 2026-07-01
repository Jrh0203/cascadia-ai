"""Selected-checkpoint open evaluation for ADR 0100."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import socket
import time
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx

from cascadia_mlx.graded_oracle_frontier_anchor import (
    _system_swap_used_bytes,
    benchmark_frontier_anchored,
)
from cascadia_mlx.graded_oracle_frontier_expected_rank import (
    EXPECTED_RANK_STUDENT_TEMPERATURE,
    EXPECTED_RANK_TARGET_SCALE,
    EXPERIMENT_ID,
    ExpectedRankDataset,
    evaluate_frontier_expected_rank,
)
from cascadia_mlx.graded_oracle_model import (
    GradedOracleModelConfig,
    GradedOracleRanker,
)

EXPECTED_RUN_KIND = "graded-oracle-frontier-expected-rank"


def evaluate_selected_expected_rank(
    *,
    run_dir: Path,
    train_dataset: Path,
    validation_dataset: Path,
    train_cache: Path,
    validation_cache: Path,
    experiment_id: str = EXPERIMENT_ID,
    target_scale: float = EXPECTED_RANK_TARGET_SCALE,
    student_temperature: float = EXPECTED_RANK_STUDENT_TEMPERATURE,
    expected_run_kind: str = EXPECTED_RUN_KIND,
) -> dict[str, Any]:
    """Evaluate and benchmark the preregistered best checkpoint."""
    started = time.perf_counter()
    swap_before = _system_swap_used_bytes()
    run = json.loads((run_dir / "run.json").read_text())
    if run.get("kind") != expected_run_kind:
        raise ValueError("expected-rank run kind does not match")
    if run["training"]["seed"] != 2026061626:
        raise ValueError("expected-rank run seed drifted")
    best = json.loads((run_dir / "best.json").read_text())
    checkpoint = run_dir / "checkpoints" / str(best["checkpoint"])
    manifest = json.loads((checkpoint / "checkpoint.json").read_text())
    model_path = checkpoint / "model.safetensors"
    metadata = manifest["files"]["model.safetensors"]
    if (
        model_path.stat().st_size != int(metadata["bytes"])
        or _checksum(model_path) != metadata["blake3"]
    ):
        raise ValueError("selected expected-rank model failed integrity")

    train = ExpectedRankDataset(
        train_dataset,
        train_cache,
        experiment_id=experiment_id,
        target_scale=target_scale,
        student_temperature=student_temperature,
    )
    validation = ExpectedRankDataset(
        validation_dataset,
        validation_cache,
        experiment_id=experiment_id,
        target_scale=target_scale,
        student_temperature=student_temperature,
    )
    if train.split != "train" or validation.split != "validation":
        raise ValueError("expected-rank evaluation accepts only open splits")
    if (
        _checksum(train.root / "dataset.json")
        != run["datasets"]["train_manifest_blake3"]
        or _checksum(validation.root / "dataset.json")
        != run["datasets"]["validation_manifest_blake3"]
    ):
        raise ValueError("expected-rank evaluation dataset identity drifted")

    model = GradedOracleRanker(
        GradedOracleModelConfig.from_dict(manifest["model_config"])
    )
    model.load_weights(str(model_path))
    model.eval()
    mx.eval(model.parameters())
    train_metrics = evaluate_frontier_expected_rank(model, train, 64)
    validation_metrics = evaluate_frontier_expected_rank(model, validation, 64)
    performance = benchmark_frontier_anchored(model, validation.base)

    scientific = {
        "checkpoint": checkpoint.name,
        "checkpoint_manifest_blake3": _checksum(
            checkpoint / "checkpoint.json"
        ),
        "model_blake3": _checksum(model_path),
        "target_scale": target_scale,
        "student_temperature": student_temperature,
        "train_dataset_id": train.manifest["dataset_id"],
        "train_manifest_blake3": _checksum(train.root / "dataset.json"),
        "train_cache_identity": train.cache.manifest[
            "ordered_group_action_identity_blake3"
        ],
        "validation_dataset_id": validation.manifest["dataset_id"],
        "validation_manifest_blake3": _checksum(
            validation.root / "dataset.json"
        ),
        "validation_cache_identity": validation.cache.manifest[
            "ordered_group_action_identity_blake3"
        ],
        "train": train_metrics,
        "validation": validation_metrics,
        "performance": performance,
        "test_split_opened": False,
        "gameplay_opened": False,
        "new_teacher_compute_used": False,
        "external_compute_used": False,
    }
    swap_after = _system_swap_used_bytes()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss = int(usage.ru_maxrss)
    if platform.system() != "Darwin":
        peak_rss *= 1024
    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "scientific": scientific,
        "telemetry": {
            "host": socket.gethostname(),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_process_rss_bytes": peak_rss,
            "process_swaps": int(getattr(usage, "ru_nswap", 0)),
            "system_swap_before_bytes": swap_before,
            "system_swap_after_bytes": swap_after,
            "system_swap_delta_bytes": (
                None
                if swap_before is None or swap_after is None
                else swap_after - swap_before
            ),
        },
    }


def _checksum(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_selected_expected_rank(
        run_dir=args.run_dir,
        train_dataset=args.train_dataset,
        validation_dataset=args.validation_dataset,
        train_cache=args.train_cache,
        validation_cache=args.validation_cache,
    )
    _write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
