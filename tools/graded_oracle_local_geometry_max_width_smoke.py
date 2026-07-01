#!/usr/bin/env python3
"""Exercise the widest unsealed group with the ADR 0088 MLX treatment."""

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
import mlx.optimizers as optim
import numpy as np
from cascadia_mlx.graded_oracle_dataset import (
    GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
    GRADED_ORACLE_PACKED_ACTION_LIMIT,
    GRADED_ORACLE_PRIOR_DIM,
    GRADED_ORACLE_PRIOR_SCHEMA,
    decode_graded_oracle_groups,
)
from cascadia_mlx.graded_oracle_local_geometry_model import (
    LOCAL_GEOMETRY_RELATION_SCHEMA,
    LocalGeometryModelConfig,
    LocalGeometryRanker,
)
from cascadia_mlx.graded_oracle_model import (
    graded_oracle_loss,
    predict_graded_oracle_batch,
)
from graded_oracle_max_width_smoke import (
    checksum,
    system_swap_used_bytes,
    widest_unsealed_group,
    write_json_atomic,
)


def run_smoke(dataset_roots: list[Path], *, seed: int) -> dict[str, Any]:
    """Run one exact forward/backward update on the widest open group."""
    dataset, shard_index, ref, identity = widest_unsealed_group(dataset_roots)
    if ref.candidate_count > GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS:
        raise ValueError("widest group exceeds the frozen singleton ceiling")
    shard = dataset.shards[shard_index]
    batch = decode_graded_oracle_groups(shard.bytes(), (ref,))

    mx.random.seed(seed)
    model = LocalGeometryRanker(LocalGeometryModelConfig())
    optimizer = optim.AdamW(learning_rate=1e-4, weight_decay=1e-4)
    prediction = predict_graded_oracle_batch(model, batch)
    forward_started = time.perf_counter()
    mx.eval(prediction.scores, prediction.standard_errors)
    forward_seconds = time.perf_counter() - forward_started
    initial_scores = np.asarray(prediction.scores)
    screen = np.asarray(batch.screen_value)

    loss_and_grad = nn.value_and_grad(model, graded_oracle_loss)
    swap_before = system_swap_used_bytes()
    update_started = time.perf_counter()
    loss, gradients = loss_and_grad(model, batch)
    optimizer.update(model, gradients)
    mx.eval(model.parameters(), optimizer.state, loss)
    update_seconds = time.perf_counter() - update_started
    swap_after = system_swap_used_bytes()

    updated = predict_graded_oracle_batch(model, batch)
    mx.eval(updated.scores, updated.standard_errors)
    updated_scores = np.asarray(updated.scores)
    updated_standard_errors = np.asarray(updated.standard_errors)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss = int(usage.ru_maxrss)
    if platform.system() != "Darwin":
        peak_rss *= 1024

    report = {
        "schema_version": 1,
        "experiment_id": "complete-action-local-geometry-ranker-v1",
        "smoke": "maximum-width-local-geometry-forward-backward",
        "host": socket.gethostname(),
        "device": str(mx.default_device()),
        "mlx_version": version("mlx"),
        "model_config": model.config.to_dict(),
        "relation_schema": LOCAL_GEOMETRY_RELATION_SCHEMA,
        "prior_feature_schema": GRADED_ORACLE_PRIOR_SCHEMA,
        "prior_feature_count": GRADED_ORACLE_PRIOR_DIM,
        "teacher_provenance_used_as_model_input": False,
        "seed": seed,
        "dataset": str(dataset.root.resolve()),
        "dataset_manifest_blake3": checksum(dataset.root / "dataset.json"),
        "split": dataset.split,
        **identity,
        "candidate_count": ref.candidate_count,
        "packed_action_target": GRADED_ORACLE_PACKED_ACTION_LIMIT,
        "maximum_group_actions": GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
        "singleton_overflow_exercised": (
            ref.candidate_count > GRADED_ORACLE_PACKED_ACTION_LIMIT
        ),
        "initial_screen_bit_exact": bool(np.array_equal(initial_scores, screen)),
        "initial_scores_finite": bool(np.all(np.isfinite(initial_scores))),
        "updated_scores_finite": bool(np.all(np.isfinite(updated_scores))),
        "updated_standard_errors_positive": bool(
            np.all(updated_standard_errors[np.asarray(batch.candidate_mask)] > 0)
        ),
        "loss": float(loss.item()),
        "loss_finite": bool(np.isfinite(float(loss.item()))),
        "forward_seconds": forward_seconds,
        "optimizer_step_seconds": update_seconds,
        "action_scores_per_second": ref.candidate_count / max(forward_seconds, 1e-9),
        "peak_process_rss_bytes": peak_rss,
        "process_swaps": int(getattr(usage, "ru_nswap", 0)),
        "system_swap_before_bytes": swap_before,
        "system_swap_after_bytes": swap_after,
        "system_swap_delta_bytes": (
            None
            if swap_before is None or swap_after is None
            else swap_after - swap_before
        ),
    }
    report["passed"] = all(
        [
            report["singleton_overflow_exercised"],
            report["initial_screen_bit_exact"],
            report["initial_scores_finite"],
            report["updated_scores_finite"],
            report["updated_standard_errors_positive"],
            report["loss_finite"],
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
    parser.add_argument("--seed", type=int, default=2026061601)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_smoke(args.dataset, seed=args.seed)
    write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
