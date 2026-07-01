#!/usr/bin/env python3
"""Maximum-width reconstruction audit for ADR 0097 candidate factors."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import socket
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from cascadia_mlx.graded_oracle_dataset import (
    GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
    GRADED_ORACLE_PACKED_ACTION_LIMIT,
    decode_graded_oracle_groups,
)
from cascadia_mlx.graded_oracle_factor_integration import (
    EXPERIMENT_ID,
    FACTOR_COUNT,
    FACTOR_DIM,
    MLX_CACHE_LIMIT_BYTES,
    PROBE_KINDS,
    balanced_factor_binary_loss,
    build_factor_probe,
    configure_mlx_memory,
    mlx_memory_snapshot,
)
from cascadia_mlx.graded_oracle_frontier_warm_start import (
    EXPECTED_WARM_START_CHECKPOINT,
    EXPECTED_WARM_START_MANIFEST_BLAKE3,
    EXPECTED_WARM_START_MODEL_BLAKE3,
    checksum,
    load_frontier_warm_start,
)
from cascadia_mlx.graded_oracle_model import (
    GRADED_ORACLE_RESIDUAL_RANGE,
    encode_graded_oracle_factor_batch,
    encode_graded_oracle_prepool_batch,
    predict_graded_oracle_batch,
)
from graded_oracle_max_width_smoke import (
    system_swap_used_bytes,
    widest_unsealed_group,
    write_json_atomic,
)
from mlx.utils import tree_flatten


def run_factor_reconstruction_audit(
    dataset_roots: list[Path],
    *,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    """Reconstruct the unchanged ranker from its seven exported factors."""
    dataset, shard_index, ref, identity = widest_unsealed_group(dataset_roots)
    if ref.candidate_count > GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS:
        raise ValueError("widest group exceeds the frozen singleton ceiling")
    batch = decode_graded_oracle_groups(
        dataset.shards[shard_index].bytes(),
        (ref,),
    )
    model = load_frontier_warm_start(checkpoint_dir)
    model.eval()

    allocator = configure_mlx_memory()
    swap_before = system_swap_used_bytes()
    started = time.perf_counter()
    prediction = predict_graded_oracle_batch(model, batch)
    factors = encode_graded_oracle_factor_batch(model, batch)
    prepool = encode_graded_oracle_prepool_batch(model, batch)
    groups, candidates = batch.screen_value.shape
    reconstructed_prepool = model.candidate_projection(
        factors.reshape(groups, candidates, -1)
    ) * batch.candidate_mask[..., None]
    embeddings = model.encode_output_from_prepool(
        reconstructed_prepool,
        batch.candidate_mask,
    )
    reconstructed_residuals = (
        GRADED_ORACLE_RESIDUAL_RANGE
        * mx.tanh(model.residual_head(embeddings).reshape(groups, candidates))
        * batch.candidate_mask
    )
    reconstructed_standard_errors = (
        nn.softplus(
            model.standard_error_head(embeddings).reshape(groups, candidates)
        )
        + 1e-4
    ) * batch.candidate_mask
    count = ref.candidate_count
    flat_factors = factors.reshape(-1, FACTOR_COUNT, FACTOR_DIM)[:count]
    offsets = (0, count)
    screen_rank = np.asarray(batch.screen_rank)[0, :count]
    action_hash = np.asarray(batch.action_hash)[0, :count]
    mx.eval(
        prediction.residuals,
        prediction.standard_errors,
        factors,
        prepool,
        reconstructed_prepool,
        embeddings,
        reconstructed_residuals,
        reconstructed_standard_errors,
    )
    reconstruction_memory = mlx_memory_snapshot()
    factor_values = np.asarray(factors)
    prepool_values = np.asarray(prepool)
    rebuilt_prepool_values = np.asarray(reconstructed_prepool)
    original_residuals = np.asarray(prediction.residuals)
    original_standard_errors = np.asarray(prediction.standard_errors)
    rebuilt_residuals = np.asarray(reconstructed_residuals)
    rebuilt_standard_errors = np.asarray(reconstructed_standard_errors)
    target = mx.array(np.arange(count) < min(64, count))
    eligible = mx.ones((count,), dtype=mx.bool_)
    probe_report = {}
    for kind in PROBE_KINDS:
        mx.clear_cache()
        mx.reset_peak_memory()
        probe = build_factor_probe(kind)
        loss_and_grad = nn.value_and_grad(
            probe,
            balanced_factor_binary_loss,
        )
        loss, gradients = loss_and_grad(
            probe,
            flat_factors,
            target,
            eligible,
            offsets,
            screen_rank,
            action_hash,
        )
        scores = probe(
            flat_factors,
            offsets,
            screen_rank,
            action_hash,
        )
        mx.eval(loss, gradients, scores)
        values = np.asarray(scores)
        memory_before_clear = mlx_memory_snapshot()
        mx.clear_cache()
        memory_after_clear = mlx_memory_snapshot()
        probe_report[kind] = {
            "scores": len(values),
            "all_finite": bool(np.all(np.isfinite(values))),
            "loss": float(loss.item()),
            "loss_finite": bool(np.isfinite(float(loss.item()))),
            "gradient_tensors": len(tree_flatten(gradients)),
            "mlx_memory_before_clear": memory_before_clear,
            "mlx_memory_after_clear": memory_after_clear,
        }
        del probe, gradients, loss, scores
        mx.clear_cache()
    elapsed_seconds = time.perf_counter() - started
    swap_after = system_swap_used_bytes()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss = int(usage.ru_maxrss)
    if platform.system() != "Darwin":
        peak_rss *= 1024

    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "audit": "maximum-width-candidate-factor-reconstruction",
        "host": socket.gethostname().split(".")[0],
        "device": str(mx.default_device()),
        "mlx_version": version("mlx"),
        "dataset": str(dataset.root.resolve()),
        "dataset_manifest_blake3": checksum(dataset.root / "dataset.json"),
        "split": dataset.split,
        **identity,
        "checkpoint": EXPECTED_WARM_START_CHECKPOINT,
        "checkpoint_manifest_blake3": EXPECTED_WARM_START_MANIFEST_BLAKE3,
        "model_blake3": EXPECTED_WARM_START_MODEL_BLAKE3,
        "candidate_count": count,
        "factor_count": int(factor_values.shape[-2]),
        "factor_dim": int(factor_values.shape[-1]),
        "factor_dtype": str(factor_values.dtype),
        "packed_action_target": GRADED_ORACLE_PACKED_ACTION_LIMIT,
        "maximum_group_actions": GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
        "singleton_overflow_exercised": (
            count > GRADED_ORACLE_PACKED_ACTION_LIMIT
        ),
        "all_factors_finite": bool(np.all(np.isfinite(factor_values))),
        "candidate_projection_bit_exact": bool(
            np.array_equal(prepool_values, rebuilt_prepool_values)
        ),
        "residual_head_bit_exact": bool(
            np.array_equal(original_residuals, rebuilt_residuals)
        ),
        "standard_error_head_bit_exact": bool(
            np.array_equal(
                original_standard_errors,
                rebuilt_standard_errors,
            )
        ),
        "probes": probe_report,
        "mlx_allocator": allocator,
        "reconstruction_mlx_memory": reconstruction_memory,
        "elapsed_seconds": elapsed_seconds,
        "candidate_vectors_per_second": count / max(elapsed_seconds, 1e-9),
        "peak_process_rss_bytes": peak_rss,
        "process_swaps": int(getattr(usage, "ru_nswap", 0)),
        "system_swap_before_bytes": swap_before,
        "system_swap_after_bytes": swap_after,
        "system_swap_delta_bytes": (
            None
            if swap_before is None or swap_after is None
            else swap_after - swap_before
        ),
        "test_split_opened": False,
    }
    report["passed"] = all(
        [
            report["singleton_overflow_exercised"],
            report["factor_count"] == FACTOR_COUNT,
            report["factor_dim"] == FACTOR_DIM,
            report["factor_dtype"] == "float32",
            report["all_factors_finite"],
            report["candidate_projection_bit_exact"],
            report["residual_head_bit_exact"],
            report["standard_error_head_bit_exact"],
            all(
                value["scores"] == count
                and value["all_finite"]
                and value["loss_finite"]
                and value["gradient_tensors"] > 0
                and value["mlx_memory_before_clear"][
                    "peak_active_memory_bytes"
                ]
                <= 6 * 1024**3
                and value["mlx_memory_before_clear"]["cache_memory_bytes"]
                <= MLX_CACHE_LIMIT_BYTES + 128 * 1024**2
                and value["mlx_memory_after_clear"]["cache_memory_bytes"] == 0
                for value in probe_report.values()
            ),
            report["peak_process_rss_bytes"] <= 6 * 1024**3,
            report["process_swaps"] == 0,
            report["system_swap_delta_bytes"] is not None,
            report["system_swap_delta_bytes"] <= 0,
        ]
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_factor_reconstruction_audit(
        args.dataset,
        checkpoint_dir=args.checkpoint,
    )
    write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
