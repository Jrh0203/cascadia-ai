#!/usr/bin/env python3
"""Exercise the widest open group with frontier-anchored set supervision."""

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
from cascadia_mlx.graded_oracle_frontier_anchor import (
    FRONTIER_ANCHORED_WIDTH,
    build_frontier_anchored_target_mask,
    frontier_anchored_loss,
    frontier_anchored_retained_indices,
)
from cascadia_mlx.graded_oracle_model import (
    GradedOracleModelConfig,
    GradedOracleRanker,
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

    target = build_frontier_anchored_target_mask(
        r1200_mean=np.asarray(batch.r1200_mean),
        r1200_mask=np.asarray(batch.r1200_mask),
        source_flags=np.asarray(batch.source_flags),
        candidate_mask=np.asarray(batch.candidate_mask),
        action_hashes=np.asarray(batch.action_hash),
    )
    mx.random.seed(seed)
    model = GradedOracleRanker(GradedOracleModelConfig())
    optimizer = optim.AdamW(learning_rate=1e-4, weight_decay=1e-4)
    prediction = predict_graded_oracle_batch(model, batch)
    forward_started = time.perf_counter()
    mx.eval(prediction.scores)
    forward_seconds = time.perf_counter() - forward_started
    initial_scores = np.asarray(prediction.scores)
    screen = np.asarray(batch.screen_value)
    mask = np.asarray(batch.candidate_mask)[0]
    retained = frontier_anchored_retained_indices(
        scores=initial_scores[0][mask],
        source_flags=np.asarray(batch.source_flags)[0][mask],
        action_hashes=np.asarray(batch.action_hash)[0][mask],
    )

    loss_and_grad = nn.value_and_grad(model, frontier_anchored_loss)
    swap_before = system_swap_used_bytes()
    update_started = time.perf_counter()
    loss, gradients = loss_and_grad(model, batch)
    optimizer.update(model, gradients)
    mx.eval(model.parameters(), optimizer.state, loss)
    update_seconds = time.perf_counter() - update_started
    swap_after = system_swap_used_bytes()

    updated = predict_graded_oracle_batch(model, batch)
    mx.eval(updated.scores)
    updated_scores = np.asarray(updated.scores)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss = int(usage.ru_maxrss)
    if platform.system() != "Darwin":
        peak_rss *= 1024

    report = {
        "schema_version": 1,
        "experiment_id": "complete-action-frontier-anchored-set-ranker-v1",
        "smoke": "maximum-width-frontier-anchored-forward-backward",
        "host": socket.gethostname(),
        "device": str(mx.default_device()),
        "mlx_version": version("mlx"),
        "model_config": model.config.to_dict(),
        "prior_feature_schema": GRADED_ORACLE_PRIOR_SCHEMA,
        "prior_feature_count": GRADED_ORACLE_PRIOR_DIM,
        "teacher_provenance_used_as_model_input": False,
        "frontier_membership_used_only_by_selector_and_loss": True,
        "seed": seed,
        "dataset": str(dataset.root.resolve()),
        "dataset_manifest_blake3": checksum(dataset.root / "dataset.json"),
        "split": dataset.split,
        **identity,
        "candidate_count": ref.candidate_count,
        "proposal_width": FRONTIER_ANCHORED_WIDTH,
        "target_positive_count": int(np.sum(target)),
        "retained_count": len(retained),
        "packed_action_target": GRADED_ORACLE_PACKED_ACTION_LIMIT,
        "maximum_group_actions": GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
        "singleton_overflow_exercised": (
            ref.candidate_count > GRADED_ORACLE_PACKED_ACTION_LIMIT
        ),
        "initial_screen_bit_exact": bool(np.array_equal(initial_scores, screen)),
        "initial_scores_finite": bool(np.all(np.isfinite(initial_scores))),
        "updated_scores_finite": bool(np.all(np.isfinite(updated_scores))),
        "loss": float(loss.item()),
        "loss_finite": bool(np.isfinite(float(loss.item()))),
        "forward_seconds": forward_seconds,
        "optimizer_step_seconds": update_seconds,
        "action_scores_per_second": ref.candidate_count
        / max(forward_seconds, 1e-9),
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
            report["loss_finite"],
            report["retained_count"] == FRONTIER_ANCHORED_WIDTH,
            report["target_positive_count"] > 0,
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
