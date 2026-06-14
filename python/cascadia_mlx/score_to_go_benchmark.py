"""Benchmark a verified score-to-go checkpoint on real dataset records."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import numpy as np

from cascadia_mlx.checkpoint import load_checkpoint_pointer_with_factory
from cascadia_mlx.score_to_go_dataset import (
    ScoreToGoDataset,
    decode_score_to_go_records,
)
from cascadia_mlx.score_to_go_model import ScoreToGoModelConfig, ScoreToGoValueModel


def benchmark_score_to_go(
    *,
    run_dir: Path,
    dataset_path: Path,
    output: Path,
    batch_size: int = 256,
    warmup_iterations: int = 10,
    iterations: int = 100,
    maximum_p90_milliseconds_per_item: float = 25.0,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations cannot be negative")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if maximum_p90_milliseconds_per_item <= 0:
        raise ValueError("maximum latency must be positive")

    dataset = ScoreToGoDataset(dataset_path)
    records = []
    remaining = batch_size
    for shard in dataset.shards:
        shard_records = np.asarray(shard.records())
        take = min(remaining, len(shard_records))
        records.append(shard_records[:take].copy())
        remaining -= take
        if remaining == 0:
            break
    if remaining:
        raise ValueError(f"dataset has only {batch_size - remaining} records; need {batch_size}")
    batch = decode_score_to_go_records(np.concatenate(records))

    model, _, _, checkpoint = load_checkpoint_pointer_with_factory(
        run_dir,
        pointer="best",
        learning_rate=3e-4,
        weight_decay=1e-4,
        model_factory=lambda values: ScoreToGoValueModel(ScoreToGoModelConfig.from_dict(values)),
    )
    model.eval()

    def predict() -> mx.array:
        return model.predict_components(
            batch.board_entities,
            batch.board_mask,
            batch.market_entities,
            batch.market_mask,
            batch.global_features,
        )

    for _ in range(warmup_iterations):
        mx.eval(predict())

    latencies = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        started = time.perf_counter()
        mx.eval(predict())
        latencies[index] = time.perf_counter() - started

    p50_ms = float(np.percentile(latencies, 50) * 1000.0)
    p90_ms = float(np.percentile(latencies, 90) * 1000.0)
    p99_ms = float(np.percentile(latencies, 99) * 1000.0)
    p90_per_item = p90_ms / batch_size
    report = {
        "schema_version": 1,
        "device": str(mx.default_device()),
        "run_dir": str(run_dir.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_manifest_blake3": _checksum(checkpoint / "checkpoint.json"),
        "dataset": str(dataset_path.resolve()),
        "dataset_manifest_blake3": _checksum(dataset_path / "dataset.json"),
        "batch_size": batch_size,
        "warmup_iterations": warmup_iterations,
        "iterations": iterations,
        "latency": {
            "p50_milliseconds": p50_ms,
            "p90_milliseconds": p90_ms,
            "p99_milliseconds": p99_ms,
            "p90_milliseconds_per_item": p90_per_item,
            "p50_positions_per_second": float(batch_size / np.median(latencies)),
        },
        "gates": {
            "apple_gpu": str(mx.default_device()) == "Device(gpu, 0)",
            "batch_size_256": batch_size == 256,
            "p90_milliseconds_per_item_at_most_25": (
                p90_per_item <= maximum_p90_milliseconds_per_item
            ),
        },
    }
    report["passed"] = all(report["gates"].values())
    _write_json_atomic(output, report)
    return report


def _checksum(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--maximum-p90-milliseconds-per-item", type=float, default=25.0)
    args = parser.parse_args()
    report = benchmark_score_to_go(
        run_dir=args.run_dir,
        dataset_path=args.dataset,
        output=args.output,
        batch_size=args.batch_size,
        warmup_iterations=args.warmup_iterations,
        iterations=args.iterations,
        maximum_p90_milliseconds_per_item=args.maximum_p90_milliseconds_per_item,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
