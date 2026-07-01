#!/usr/bin/env python3
"""Maximum-width forward/backward audit for ADR 0098 raw constructors."""

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
    decode_graded_oracle_groups,
)
from cascadia_mlx.graded_oracle_factor_integration import (
    MLX_CACHE_LIMIT_BYTES,
    configure_mlx_memory,
    mlx_memory_snapshot,
)
from cascadia_mlx.graded_oracle_frontier_anchor import (
    GRADED_SOURCE_CHAMPION_FRONTIER,
    build_frontier_anchored_target_mask,
)
from cascadia_mlx.graded_oracle_frontier_warm_start import checksum
from cascadia_mlx.graded_oracle_raw_factor_construction import (
    EXPERIMENT_ID,
    PROBE_KINDS,
    PROBE_SEEDS,
    batch_counts,
    build_raw_factor_probe,
    parameter_count,
    raw_factor_probe_loss,
    score_raw_factor_batch,
)
from graded_oracle_max_width_smoke import (
    system_swap_used_bytes,
    widest_unsealed_group,
    write_json_atomic,
)
from mlx.utils import tree_flatten


def run_maximum_width_audit(
    dataset_roots: list[Path],
    *,
    kind: str,
) -> dict[str, Any]:
    if kind not in PROBE_KINDS:
        raise ValueError("unsupported raw factor-construction probe")
    dataset, shard_index, ref, identity = widest_unsealed_group(dataset_roots)
    if ref.candidate_count > GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS:
        raise ValueError("widest group exceeds maximum_group_actions")
    batch = decode_graded_oracle_groups(
        dataset.shards[shard_index].bytes(),
        (ref,),
    )
    counts = batch_counts(batch)
    target_values = build_frontier_anchored_target_mask(
        r1200_mean=np.asarray(batch.r1200_mean),
        r1200_mask=np.asarray(batch.r1200_mask),
        source_flags=np.asarray(batch.source_flags),
        candidate_mask=np.asarray(batch.candidate_mask),
        action_hashes=batch.action_hash,
    )
    source_flags = np.asarray(batch.source_flags)
    eligible_values = np.asarray(batch.candidate_mask) & (
        (source_flags & GRADED_SOURCE_CHAMPION_FRONTIER) == 0
    )

    allocator = configure_mlx_memory()
    mx.random.seed(PROBE_SEEDS[kind])
    model = build_raw_factor_probe(kind)
    optimizer = optim.AdamW(learning_rate=3e-4, weight_decay=1e-4)
    loss_and_grad = nn.value_and_grad(model, raw_factor_probe_loss)
    swap_before = system_swap_used_bytes()
    started = time.perf_counter()

    initial_scores = score_raw_factor_batch(model, batch, counts)
    mx.eval(initial_scores)
    forward_seconds = time.perf_counter() - started
    initial_values = np.asarray(initial_scores)
    forward_memory = mlx_memory_snapshot()

    update_started = time.perf_counter()
    loss, gradients = loss_and_grad(
        model,
        batch.board_entities,
        batch.board_mask,
        batch.market_entities,
        batch.market_mask,
        batch.global_features,
        batch.public_supply,
        batch.action_features,
        batch.prior_features,
        batch.staged_market_entities,
        batch.staged_market_mask,
        batch.staged_public_supply,
        batch.candidate_mask,
        mx.array(target_values),
        mx.array(eligible_values),
        counts,
        np.asarray(batch.screen_rank),
        batch.action_hash,
    )
    optimizer.update(model, gradients)
    mx.eval(model.parameters(), optimizer.state, loss)
    update_seconds = time.perf_counter() - update_started
    update_memory = mlx_memory_snapshot()
    gradient_tensors = len(tree_flatten(gradients))
    del gradients

    updated_scores = score_raw_factor_batch(model, batch, counts)
    mx.eval(updated_scores)
    updated_values = np.asarray(updated_scores)
    memory_before_clear = mlx_memory_snapshot()
    mx.clear_cache()
    memory_after_clear = mlx_memory_snapshot()
    swap_after = system_swap_used_bytes()
    elapsed = time.perf_counter() - started

    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss = int(usage.ru_maxrss)
    if platform.system() != "Darwin":
        peak_rss *= 1024
    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "audit": "maximum-width-raw-factor-forward-backward",
        "kind": kind,
        "seed": PROBE_SEEDS[kind],
        "host": _canonical_host(),
        "device": str(mx.default_device()),
        "mlx_version": version("mlx"),
        "dataset": str(dataset.root.resolve()),
        "dataset_manifest_blake3": checksum(dataset.root / "dataset.json"),
        "split": dataset.split,
        **identity,
        "candidate_count": ref.candidate_count,
        "parameter_count": parameter_count(model),
        "packed_action_target": GRADED_ORACLE_PACKED_ACTION_LIMIT,
        "maximum_group_actions": GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
        "singleton_overflow_exercised": (ref.candidate_count > GRADED_ORACLE_PACKED_ACTION_LIMIT),
        "scores": int(initial_values.size),
        "initial_scores_finite": bool(np.all(np.isfinite(initial_values))),
        "updated_scores_finite": bool(np.all(np.isfinite(updated_values))),
        "loss": float(loss.item()),
        "loss_finite": bool(np.isfinite(float(loss.item()))),
        "gradient_tensors": gradient_tensors,
        "forward_seconds": forward_seconds,
        "optimizer_step_seconds": update_seconds,
        "elapsed_seconds": elapsed,
        "forward_candidates_per_second": (ref.candidate_count / max(forward_seconds, 1e-9)),
        "mlx_allocator": allocator,
        "forward_mlx_memory": forward_memory,
        "update_mlx_memory": update_memory,
        "mlx_memory_before_clear": memory_before_clear,
        "mlx_memory_after_clear": memory_after_clear,
        "peak_process_rss_bytes": peak_rss,
        "process_swaps": int(getattr(usage, "ru_nswap", 0)),
        "system_swap_before_bytes": swap_before,
        "system_swap_after_bytes": swap_after,
        "system_swap_delta_bytes": (
            None if swap_before is None or swap_after is None else swap_after - swap_before
        ),
        "test_split_opened": False,
        "gameplay_opened": False,
        "external_compute_used": False,
    }
    report["passed"] = all(
        [
            report["singleton_overflow_exercised"],
            report["scores"] == ref.candidate_count,
            report["initial_scores_finite"],
            report["updated_scores_finite"],
            report["loss_finite"],
            report["gradient_tensors"] > 0,
            report["update_mlx_memory"]["peak_active_memory_bytes"] <= 6 * 1024**3,
            report["mlx_memory_before_clear"]["cache_memory_bytes"]
            <= MLX_CACHE_LIMIT_BYTES + 128 * 1024**2,
            report["mlx_memory_after_clear"]["cache_memory_bytes"] == 0,
            report["peak_process_rss_bytes"] <= 6 * 1024**3,
            report["process_swaps"] == 0,
            report["system_swap_delta_bytes"] is not None,
            report["system_swap_delta_bytes"] <= 0,
            not report["test_split_opened"],
            not report["gameplay_opened"],
            not report["external_compute_used"],
        ]
    )
    return report


def _canonical_host() -> str:
    host = socket.gethostname().split(".")[0].lower()
    return "john1" if host == "johns-mac-mini" else host


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--kind", choices=PROBE_KINDS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_maximum_width_audit(args.dataset, kind=args.kind)
    write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
