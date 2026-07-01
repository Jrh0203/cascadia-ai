"""Two-pass MLX training profiler for the Cascadia V3 readiness gate."""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import subprocess
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from cascadia_mlx.checkpoint import load_latest_checkpoint_with_factory

from .contracts import V3MlxConfig
from .dataset import SparseWidths, synthetic_batch
from .model import CsrBatch, V3Nnue, v3_loss
from .stream import RustBatchStream


def _physical_memory() -> int:
    return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())


def _swap_used() -> int:
    text = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True)
    match = re.search(r"used\s*=\s*([0-9.]+)([KMG])", text)
    if match:
        number = float(match.group(1))
        return int(number * {"K": 1024, "M": 1024**2, "G": 1024**3}[match.group(2)])
    raise ValueError(f"cannot parse vm.swapusage: {text}")


def profile(
    feature_manifest: Path,
    output: Path,
    *,
    pass_id: int,
    measured_examples: int,
    microbatch_size: int,
    datasets: list[Path] | None = None,
    stream_binary: Path | None = None,
    checkpoint_run_dir: Path | None = None,
) -> dict[str, object]:
    if pass_id not in (1, 2):
        raise ValueError("profile pass must be 1 or 2")
    if measured_examples <= 0 or microbatch_size <= 0:
        raise ValueError("profile sizes must be positive")
    feature = json.loads(feature_manifest.read_text())
    config = V3MlxConfig(
        opportunity_feature_rows=feature["opportunity_feature_rows"],
        opportunity_training_factor_rows=feature["opportunity_training_factor_rows"],
    )
    padding = 2 if pass_id == 1 else 1
    widths = SparseWidths(48 * padding, 96 * padding, 96 * padding, 192 * padding)
    mx.clear_cache()
    mx.reset_peak_memory()
    initialization_started = time.perf_counter()
    if checkpoint_run_dir is None:
        model = V3Nnue(config)
        optimizer = optim.AdamW(learning_rate=5e-4, weight_decay=1e-6)
        checkpoint_identity = None
    else:
        run_manifest = json.loads((checkpoint_run_dir / "run-manifest.json").read_text())
        optimizer_config = run_manifest["optimizer"]
        model, optimizer, _, checkpoint = load_latest_checkpoint_with_factory(
            checkpoint_run_dir,
            learning_rate=float(optimizer_config["learning_rate"]),
            weight_decay=float(optimizer_config["weight_decay"]),
            model_factory=lambda values: V3Nnue(V3MlxConfig.from_dict(values)),
        )
        if model.config != config:
            raise ValueError("profile checkpoint differs from the feature manifest")
        checkpoint_identity = checkpoint.name
    mx.eval(model.parameters())
    initialization_seconds = time.perf_counter() - initialization_started
    loss_and_grad = nn.value_and_grad(model, v3_loss)
    swap_before = _swap_used()
    rows = []
    total_examples = 0
    total_training_seconds = 0.0
    for logical_batch_size in (8_192, 16_384, 32_768):
        sample_count = (
            logical_batch_size
            if pass_id == 2 and datasets and stream_binary
            else min(measured_examples, logical_batch_size)
        )
        build_seconds = 0.0
        forward_seconds = 0.0
        backward_seconds = 0.0
        optimizer_seconds = 0.0
        losses = []
        if pass_id == 2 and datasets and stream_binary:
            started = time.perf_counter()
            stream = RustBatchStream(
                stream_binary,
                datasets,
                config,
                batch_size=logical_batch_size,
                epochs=1,
                allow_scientific_data=False,
                expansion_threads=8,
            )
            batch = next(stream)
            build_seconds += time.perf_counter() - started
            batches = [batch]
            sample_count = int(batch.targets.shape[0])
        else:
            batches = []
            for offset in range(0, sample_count, microbatch_size):
                started = time.perf_counter()
                batch = synthetic_batch(
                    config,
                    min(microbatch_size, sample_count - offset),
                    76_000 + logical_batch_size + offset,
                    widths,
                )
                build_seconds += time.perf_counter() - started
                batches.append(batch)
        for batch in batches:
            started = time.perf_counter()
            value = model.call_csr(batch) if isinstance(batch, CsrBatch) else model(batch)
            mx.eval(value)
            forward_seconds += time.perf_counter() - started

            started = time.perf_counter()
            loss, gradients = loss_and_grad(model, batch)
            mx.eval(loss, gradients)
            backward_seconds += time.perf_counter() - started

            started = time.perf_counter()
            optimizer.update(model, gradients)
            mx.eval(model.parameters(), optimizer.state)
            optimizer_seconds += time.perf_counter() - started
            losses.append(float(loss.item()))
        if pass_id == 2 and datasets and stream_binary:
            stream.close()
        backward_optimizer_seconds = backward_seconds + optimizer_seconds
        training_seconds = build_seconds + backward_optimizer_seconds
        total_examples += sample_count
        total_training_seconds += training_seconds
        rows.append(
            {
                "logical_batch_size": logical_batch_size,
                "measured_examples": sample_count,
                "microbatch_size": microbatch_size,
                "decode_and_batch_seconds": build_seconds,
                "forward_seconds": forward_seconds,
                "backward_seconds": backward_seconds,
                "optimizer_seconds": optimizer_seconds,
                "backward_optimizer_seconds": backward_optimizer_seconds,
                "training_examples_per_second": sample_count / max(training_seconds, 1e-9),
                "projected_logical_batch_seconds": logical_batch_size
                * training_seconds
                / sample_count,
                "mean_loss": sum(losses) / len(losses),
            }
        )
        # MLX deliberately retains compiled graphs and temporary buffers in a
        # process-wide cache. Keeping three campaign-sized shapes resident at
        # once makes the profiler itself create memory pressure that the
        # trainer never creates (the trainer uses one selected batch size).
        # Release each trial before constructing the next one so the measured
        # peak and swap delta describe the production lifecycle.
        del batches
        gc.collect()
        mx.clear_cache()
    mx.eval(model.parameters(), optimizer.state)
    gc.collect()
    mx.clear_cache()
    swap_after = _swap_used()
    examples_per_second = total_examples / total_training_seconds
    # Fixed campaign exposure projection used only as an engineering capacity
    # estimate. Part 2 remains sealed regardless of this value.
    projected_exposures = 120_000_000
    projected_part2_seconds = projected_exposures / max(examples_per_second, 1e-9)
    report = {
        "schema_id": "cascadia-v3-mlx-profile-v1",
        "profile_pass": pass_id,
        "optimization": (
            "correct-reference-padding" if pass_id == 1 else "bucketed-exact-width-prefetch"
        ),
        "scientific_eligible": False,
        "mlx_device": str(mx.default_device()),
        "initialization_seconds": initialization_seconds,
        "batch_profiles": rows,
        "examples": total_examples,
        "elapsed_training_seconds": total_training_seconds,
        "examples_per_second": examples_per_second,
        "projected_part2_seconds": projected_part2_seconds,
        "active_memory_bytes": int(mx.get_active_memory()),
        "peak_memory_bytes": int(mx.get_peak_memory()),
        "cache_memory_bytes": int(mx.get_cache_memory()),
        "physical_memory_bytes": _physical_memory(),
        "swap_before_bytes": swap_before,
        "swap_after_bytes": swap_after,
        "swap_delta_bytes": max(0, swap_after - swap_before),
        "process_id": os.getpid(),
        "checkpoint_identity": checkpoint_identity,
        "measurement_scope": {
            "decode_and_batch": "native decode, CSR construction, and host-to-MLX transfer",
            "forward": "sparse gathers, segmented accumulation, fake quantization, and dense graph",
            "backward": "loss graph, reverse sparse accumulation, and gradient synchronization",
            "optimizer": "AdamW update and parameter synchronization",
            "evaluation": "reported by the standalone forward measurement",
            "checkpoint": "measured by the interruption/resume smoke receipt",
            "gpu_utilization": (
                "MLX exposes device and idle wall gaps but not a stable utilization counter"
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pass-id", type=int, choices=(1, 2), required=True)
    parser.add_argument("--measured-examples", type=int, default=128)
    parser.add_argument("--microbatch-size", type=int, default=16)
    parser.add_argument("--dataset", type=Path, action="append", default=[])
    parser.add_argument("--batch-stream-binary", type=Path)
    parser.add_argument("--checkpoint-run-dir", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            profile(
                args.feature_manifest,
                args.output,
                pass_id=args.pass_id,
                measured_examples=args.measured_examples,
                microbatch_size=args.microbatch_size,
                datasets=args.dataset,
                stream_binary=args.batch_stream_binary,
                checkpoint_run_dir=args.checkpoint_run_dir,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
