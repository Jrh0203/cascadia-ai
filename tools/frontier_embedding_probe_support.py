#!/usr/bin/env python3
"""Independent maximum-width audit for ADR 0094 candidate embeddings."""

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
from cascadia_mlx.graded_oracle_embedding_probe import EXPERIMENT_ID
from cascadia_mlx.graded_oracle_frontier_warm_start import (
    EXPECTED_WARM_START_CHECKPOINT,
    EXPECTED_WARM_START_MANIFEST_BLAKE3,
    EXPECTED_WARM_START_MODEL_BLAKE3,
    checksum,
    load_frontier_warm_start,
)
from cascadia_mlx.graded_oracle_model import (
    GRADED_ORACLE_RESIDUAL_RANGE,
    encode_graded_oracle_batch,
    predict_graded_oracle_batch,
)
from graded_oracle_max_width_smoke import (
    system_swap_used_bytes,
    widest_unsealed_group,
    write_json_atomic,
)


def run_embedding_reconstruction_audit(
    dataset_roots: list[Path],
    *,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    """Prove the exported tensor exactly reconstructs both original heads."""
    dataset, shard_index, ref, identity = widest_unsealed_group(dataset_roots)
    if ref.candidate_count > GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS:
        raise ValueError("widest group exceeds the frozen singleton ceiling")
    batch = decode_graded_oracle_groups(
        dataset.shards[shard_index].bytes(),
        (ref,),
    )
    model = load_frontier_warm_start(checkpoint_dir)
    model.eval()

    swap_before = system_swap_used_bytes()
    started = time.perf_counter()
    prediction = predict_graded_oracle_batch(model, batch)
    embeddings = encode_graded_oracle_batch(model, batch)
    groups, candidates = batch.screen_value.shape
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
    mx.eval(
        prediction.residuals,
        prediction.standard_errors,
        embeddings,
        reconstructed_residuals,
        reconstructed_standard_errors,
    )
    elapsed_seconds = time.perf_counter() - started
    swap_after = system_swap_used_bytes()

    embedding_values = np.asarray(embeddings)
    candidate_mask = np.asarray(batch.candidate_mask)
    original_residuals = np.asarray(prediction.residuals)
    original_standard_errors = np.asarray(prediction.standard_errors)
    rebuilt_residuals = np.asarray(reconstructed_residuals)
    rebuilt_standard_errors = np.asarray(reconstructed_standard_errors)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss = int(usage.ru_maxrss)
    if platform.system() != "Darwin":
        peak_rss *= 1024

    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "audit": "maximum-width-pre-head-embedding-reconstruction",
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
        "candidate_count": ref.candidate_count,
        "embedding_dim": int(embedding_values.shape[-1]),
        "embedding_dtype": str(embedding_values.dtype),
        "packed_action_target": GRADED_ORACLE_PACKED_ACTION_LIMIT,
        "maximum_group_actions": GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
        "singleton_overflow_exercised": (
            ref.candidate_count > GRADED_ORACLE_PACKED_ACTION_LIMIT
        ),
        "all_embeddings_finite": bool(
            np.all(np.isfinite(embedding_values[candidate_mask]))
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
        "elapsed_seconds": elapsed_seconds,
        "candidate_embeddings_per_second": (
            ref.candidate_count / max(elapsed_seconds, 1e-9)
        ),
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
            report["embedding_dim"] == model.config.hidden_dim,
            report["embedding_dtype"] == "float32",
            report["all_embeddings_finite"],
            report["residual_head_bit_exact"],
            report["standard_error_head_bit_exact"],
            report["peak_process_rss_bytes"] <= 4 * 1024**3,
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
    report = run_embedding_reconstruction_audit(
        args.dataset,
        checkpoint_dir=args.checkpoint,
    )
    write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
