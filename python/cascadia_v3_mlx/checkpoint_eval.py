"""Evaluate and export a checksum-bound Cascadia V3 MLX checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import numpy as np
from cascadia_mlx.checkpoint import load_latest_checkpoint_with_factory

from .contracts import V3MlxConfig
from .export import export_quantized_bundle
from .model import V3Nnue
from .stream import RustBatchStream


class CheckpointEvaluationError(ValueError):
    """The checkpoint or evaluation domain is invalid."""


def _checksum(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _optimizer_settings(manifest: dict[str, Any]) -> tuple[float, float]:
    learning_rate = manifest.get("base_learning_rate")
    weight_decay = manifest.get("weight_decay")
    if learning_rate is None:
        optimizer = manifest.get("optimizer", {})
        learning_rate = optimizer.get("learning_rate")
        weight_decay = optimizer.get("weight_decay")
    try:
        learning_rate = float(learning_rate)
        weight_decay = float(weight_decay)
    except (TypeError, ValueError) as error:
        raise CheckpointEvaluationError("run manifest has no valid optimizer settings") from error
    if learning_rate <= 0 or weight_decay < 0:
        raise CheckpointEvaluationError("run manifest optimizer settings are invalid")
    return learning_rate, weight_decay


def load_checkpoint(run_dir: Path) -> tuple[V3Nnue, Path, dict[str, Any]]:
    manifest_path = run_dir / "run-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CheckpointEvaluationError("checkpoint run manifest is unreadable") from error
    learning_rate, weight_decay = _optimizer_settings(manifest)
    model, _, _, checkpoint = load_latest_checkpoint_with_factory(
        run_dir,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        model_factory=lambda values: V3Nnue(V3MlxConfig.from_dict(values)),
    )
    return model, checkpoint, manifest


def evaluate(
    *,
    model: V3Nnue,
    stream_binary: Path,
    datasets: list[Path],
    campaign_state: Path,
    batch_size: int,
    max_examples: int | None,
    cycle: int | None,
) -> dict[str, object]:
    stream = RustBatchStream(
        stream_binary,
        datasets,
        model.config,
        batch_size=batch_size,
        epochs=1,
        allow_scientific_data=True,
        campaign_state=campaign_state,
        cycle=cycle,
        teacher_lambda=1.0,
        max_examples=max_examples,
        uniform_phase=False,
        expansion_threads=8,
    )
    rows = 0
    weighted_power_sum = 0.0
    absolute_sum = 0.0
    squared_sum = 0.0
    confidence_sum = 0.0
    phase_rows = [0] * 8
    started = time.perf_counter()
    try:
        for batch in stream:
            predictions = model.call_csr(batch)
            residual = predictions - batch.targets
            weighted = batch.confidence_weights * mx.power(mx.abs(residual) / 100.0, 2.4)
            values = np.asarray(residual, dtype=np.float64)
            confidence = np.asarray(batch.confidence_weights, dtype=np.float64)
            phases = np.asarray(batch.phase_buckets, dtype=np.int64)
            weighted_power_sum += float(mx.sum(weighted).item())
            absolute_sum += float(np.abs(values).sum())
            squared_sum += float(np.square(values).sum())
            confidence_sum += float(confidence.sum())
            rows += values.size
            for phase in range(8):
                phase_rows[phase] += int(np.count_nonzero(phases == phase))
    finally:
        stream.close()
    elapsed = time.perf_counter() - started
    if rows == 0 or not all(
        math.isfinite(value)
        for value in (weighted_power_sum, absolute_sum, squared_sum, confidence_sum)
    ):
        raise CheckpointEvaluationError("validation produced no finite rows")
    return {
        "schema_id": "cascadia-v3-quantized-validation-v1",
        "passed": True,
        "examples": rows,
        "quantized_power_loss": weighted_power_sum / rows,
        "mae_score_points": absolute_sum / rows,
        "rmse_score_points": math.sqrt(squared_sum / rows),
        "mean_confidence_weight": confidence_sum / rows,
        "phase_rows": phase_rows,
        "elapsed_seconds": elapsed,
        "examples_per_second": rows / max(elapsed, 1e-9),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    required = [
        args.run_dir / "run-manifest.json",
        args.run_dir / "latest.json",
        args.feature_manifest,
        args.stream_binary,
        args.campaign_state,
        *args.dataset,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise CheckpointEvaluationError(f"evaluation inputs are missing: {missing}")
    model, checkpoint, manifest = load_checkpoint(args.run_dir)
    metrics = evaluate(
        model=model,
        stream_binary=args.stream_binary,
        datasets=args.dataset,
        campaign_state=args.campaign_state,
        batch_size=args.batch_size,
        max_examples=args.max_examples,
        cycle=args.cycle,
    )
    model_manifest = None
    if args.export_dir is not None:
        model_manifest = export_quantized_bundle(
            model,
            args.export_dir,
            args.feature_manifest,
            training_origin=args.training_origin,
            checkpoint_id=checkpoint.name,
            training_run_manifest_blake3=manifest.get("canonical_blake3"),
        )
    game_report = None
    if args.games:
        if args.export_dir is None or args.game_binary is None:
            raise CheckpointEvaluationError("open games require export-dir and game-binary")
        game_report = args.output.with_name(f"{args.output.stem}-open-games.json")
        subprocess.run(
            [
                str(args.game_binary),
                "direct-games",
                "--output",
                str(game_report),
                "--model-dir",
                str(args.export_dir),
                "--games",
                str(args.games),
                "--first-seed",
                str(args.first_seed),
            ],
            check=True,
        )
    result = {
        "schema_id": "cascadia-v3-checkpoint-evaluation-v1",
        "passed": True,
        "training_origin": args.training_origin,
        "run_dir": str(args.run_dir.resolve()),
        "run_manifest_blake3": manifest.get("canonical_blake3"),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_manifest_blake3": _checksum(checkpoint / "checkpoint.json"),
        "validation": metrics,
        "export": model_manifest,
        "open_games": str(game_report.resolve()) if game_report is not None else None,
    }
    _write_atomic(args.output, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--stream-binary", type=Path, required=True)
    parser.add_argument("--campaign-state", type=Path, required=True)
    parser.add_argument("--cycle", type=int)
    parser.add_argument("--dataset", type=Path, action="append", default=[])
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--training-origin", required=True)
    parser.add_argument("--export-dir", type=Path)
    parser.add_argument("--game-binary", type=Path)
    parser.add_argument("--games", type=int, default=0)
    parser.add_argument("--first-seed", type=int, default=1_700_000)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if not args.dataset or args.batch_size <= 0:
        raise SystemExit("validation requires datasets and a positive batch size")
    if args.max_examples is not None and args.max_examples <= 0:
        raise SystemExit("max-examples must be positive")
    if args.games < 0 or args.first_seed < 0:
        raise SystemExit("game count and seed must be nonnegative")
    if args.cycle is not None and not 1 <= args.cycle <= 10:
        raise SystemExit("cycle is outside 1..=10")
    try:
        result = run(args)
    except (
        CheckpointEvaluationError,
        OSError,
        ValueError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
    ) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
