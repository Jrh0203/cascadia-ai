#!/usr/bin/env python3
"""Benchmark the frozen parent on John2/John3 while John1 trains a V3 cycle."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from cascadia_cluster import ContainerInput, Resources
from v3_model_stage import _digest, bundle_environment, model_identity, stage_model
from v3_phase2_pipeline import (
    PipelineError,
    _client,
    _monitor,
    _validate_fabric,
    _validate_image,
    _write_atomic,
)

V1 = Path(
    "/Users/johnherrick/cascadia-bench/v3-nnue/phase2/inputs/v1/qualified-v1.bin"
)


def build_jobs(store: Any, model: Path, *, games_per_worker: int) -> list[ContainerInput]:
    """Build two 10-CPU jobs, which the 9/10/10 fabric can place only on John2/3."""
    if games_per_worker <= 0:
        raise PipelineError("training benchmark games must be positive")
    identity = model_identity(model)
    staged = stage_model(store, model, "parent")
    v1 = store.stage_file(V1, target="/inputs/v1")
    return [
        ContainerInput(
            key=f"parent-benchmark-{index}",
            args=(
                "v3-campaign-worker",
                "collect",
                "--output",
                f"/outputs/parent-benchmark-{index}.v3g",
                "--games",
                str(games_per_worker),
                "--first-game-index",
                str(3_000_000_000 + index * games_per_worker),
                "--component",
                "engineering-expert-smoke",
                "--v1-weights",
                v1.mounted_path,
                "--v3-model-dir",
                staged.materialized_directory,
            ),
            environment={
                "RAYON_NUM_THREADS": "10",
                "CASCADIA_MODEL_BUNDLES_JSON": bundle_environment([staged]),
            },
            inputs=(v1, *staged.references),
            application_metadata={
                "campaign": "cascadia-v3",
                "stage": "training-parent-benchmark",
                "worker_index": str(index),
                "games": str(games_per_worker),
                "parent_model_id": identity,
                "placement_constraint": "cpu-10-excludes-john1",
                "scientific_training_eligible": "false",
            },
        )
        for index in (0, 1)
    ]


def _validate(item_directory: Path, job: ContainerInput) -> dict[str, int]:
    shards = sorted(item_directory.glob("*.v3g"))
    receipts = sorted(item_directory.glob("*.receipt.json"))
    if len(shards) != 1 or len(receipts) != 1:
        raise PipelineError(f"parent benchmark artifacts are incomplete for {job.key}")
    shard = shards[0]
    receipt = json.loads(receipts[0].read_text())
    games = int(job.application_metadata["games"])
    parent = job.application_metadata["parent_model_id"]
    policy = receipt.get("policy_seat_games", {})
    benchmark = receipt.get("focal_benchmark", {})
    if (
        receipt.get("schema_id") != "cascadia-v3-collection-shard-receipt-v1"
        or receipt.get("scientific_eligible") is not False
        or receipt.get("component") != "engineering-expert-smoke"
        or receipt.get("games") != games
        or receipt.get("records") != games
        or receipt.get("newest_model_seats_per_expert_game") != 1
        or receipt.get("bytes") != shard.stat().st_size
        or receipt.get("blake3") != _digest(shard)
        or policy.get(parent) != games
        or sum(int(count) for count in policy.values()) != games * 4
        or benchmark.get("count") != games
        or sum(int(count) for count in benchmark.get("score_histogram", {}).values())
        != games
    ):
        raise PipelineError(f"parent benchmark receipt is invalid for {job.key}")
    totals = {
        "games": games,
        "bytes": shard.stat().st_size,
        "score_sum": int(benchmark["score_sum"]),
        "nature_tokens_sum": int(benchmark["nature_tokens_sum"]),
        "pinecones_sum": int(benchmark["pinecones_sum"]),
    }
    for name, value in benchmark["wildlife_sums"].items():
        totals[f"wildlife_{name}"] = int(value)
    for name, value in benchmark["terrain_sums"].items():
        totals[f"terrain_{name}"] = int(value)
    for score, count in benchmark["score_histogram"].items():
        totals[f"score_{int(score):03d}"] = int(count)
    return totals


def _quantile(histogram: dict[int, int], fraction: float) -> int:
    count = sum(histogram.values())
    if count <= 0 or not 0.0 <= fraction <= 1.0:
        raise PipelineError("benchmark quantile input is invalid")
    target = max(0, math.ceil(fraction * count) - 1)
    observed = 0
    for score, frequency in sorted(histogram.items()):
        observed += frequency
        if observed > target:
            return score
    raise PipelineError("benchmark histogram does not cover its declared count")


def _benchmark_summary(totals: dict[str, int]) -> dict[str, Any]:
    games = int(totals["games"])
    histogram = {
        int(name.removeprefix("score_")): int(count)
        for name, count in totals.items()
        if name.startswith("score_") and name != "score_sum"
    }
    if games <= 0 or sum(histogram.values()) != games:
        raise PipelineError("combined training benchmark histogram is incomplete")
    return {
        "games": games,
        "mean": totals["score_sum"] / games,
        "p10": _quantile(histogram, 0.10),
        "p50": _quantile(histogram, 0.50),
        "p90": _quantile(histogram, 0.90),
        "score_histogram": {str(score): count for score, count in sorted(histogram.items())},
        "wildlife_means": {
            name.removeprefix("wildlife_"): value / games
            for name, value in totals.items()
            if name.startswith("wildlife_")
        },
        "terrain_means": {
            name.removeprefix("terrain_"): value / games
            for name, value in totals.items()
            if name.startswith("terrain_")
        },
        "nature_tokens_mean": totals["nature_tokens_sum"] / games,
        "pinecones_mean": totals["pinecones_sum"] / games,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_image(args.image)
    client = _client(args.state_directory, args.artifact_directory)
    _validate_fabric(client.api.nodes())
    store = client.object_store
    assert store is not None
    jobs = build_jobs(store, args.parent_model, games_per_worker=args.games_per_worker)
    completion = _monitor(
        client=client,
        image=args.image,
        jobs=jobs,
        resources=Resources(cpu=10, memory_gib=4.0, disk_gib=2),
        request_id=args.request_id,
        experiment_id=f"cascadia-v3-cycle-{args.cycle:02d}-training-parent-benchmark",
        artifact_directory=args.artifact_directory,
        progress=args.progress,
        timeout_seconds=2 * 60 * 60,
        validate=_validate,
    )
    completion.update(
        {
            "schema_id": "cascadia-v3-training-parent-benchmark-v1",
            "cycle": args.cycle,
            "parent_model_dir": str(args.parent_model.resolve()),
            "parent_model_id": model_identity(args.parent_model),
            "john1_excluded_by_cpu_request": True,
            "scientific_training_eligible": False,
            "benchmark": _benchmark_summary(completion["totals"]),
        }
    )
    _write_atomic(args.completion, completion)
    return completion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--parent-model", type=Path, required=True)
    parser.add_argument("--games-per-worker", type=int, default=500)
    parser.add_argument("--state-directory", type=Path, required=True)
    parser.add_argument("--artifact-directory", type=Path, required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--completion", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.cycle <= 10:
        raise SystemExit("cycle is outside 1..=10")
    try:
        result = run(args)
    except (PipelineError, OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
