#!/usr/bin/env python3
"""Exercise the widest unsealed graded-oracle group with the frozen MLX model."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import resource
import socket
import subprocess
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from cascadia_mlx.graded_oracle_dataset import (
    _GROUP_HEADER_DTYPE,
    GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
    GRADED_ORACLE_PACKED_ACTION_LIMIT,
    GRADED_ORACLE_PRIOR_DIM,
    GRADED_ORACLE_PRIOR_SCHEMA,
    GradedOracleDataset,
    GradedOracleGroupRef,
    decode_graded_oracle_groups,
)
from cascadia_mlx.graded_oracle_model import (
    GradedOracleModelConfig,
    GradedOracleRanker,
    graded_oracle_loss,
    predict_graded_oracle_batch,
)

_SWAP_USED_RE = re.compile(r"used = ([0-9.]+)([KMG])")


def checksum(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def system_swap_used_bytes() -> int | None:
    if platform.system() != "Darwin":
        return None
    output = subprocess.run(
        ["sysctl", "-n", "vm.swapusage"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    match = _SWAP_USED_RE.search(output)
    if match is None:
        raise ValueError("cannot parse macOS swap usage")
    scale = {"K": 1024, "M": 1024**2, "G": 1024**3}[match.group(2)]
    return int(float(match.group(1)) * scale)


def widest_unsealed_group(
    roots: list[Path],
) -> tuple[GradedOracleDataset, int, GradedOracleGroupRef, dict[str, int]]:
    selected: tuple[
        GradedOracleDataset,
        int,
        GradedOracleGroupRef,
        dict[str, int],
    ] | None = None
    for root in roots:
        dataset = GradedOracleDataset(root, verify_checksums=True)
        if dataset.split not in {"train", "validation"}:
            raise ValueError("maximum-width smoke accepts only unsealed splits")
        for shard_index, shard in enumerate(dataset.shards):
            raw = shard.bytes()
            for ref in shard.groups:
                header = np.frombuffer(
                    raw,
                    dtype=_GROUP_HEADER_DTYPE,
                    count=1,
                    offset=ref.header_offset,
                )[0]
                identity = {
                    "raw_seed": int(header["raw_seed"]),
                    "completed_turns": int(header["completed_turns"]),
                    "group_id": int(header["group_id"]),
                }
                candidate = (dataset, shard_index, ref, identity)
                if selected is None or ref.candidate_count > selected[2].candidate_count:
                    selected = candidate
    if selected is None:
        raise ValueError("no graded-oracle group found")
    return selected


def run_smoke(
    dataset_roots: list[Path],
    *,
    seed: int,
) -> dict[str, Any]:
    dataset, shard_index, ref, identity = widest_unsealed_group(dataset_roots)
    if ref.candidate_count > GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS:
        raise ValueError("widest group exceeds the frozen singleton ceiling")
    shard = dataset.shards[shard_index]
    batch = decode_graded_oracle_groups(shard.bytes(), (ref,))

    mx.random.seed(seed)
    model = GradedOracleRanker(GradedOracleModelConfig())
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
        "experiment_id": "complete-action-graded-oracle-ranker-v1",
        "smoke": "maximum-width-frozen-forward-backward",
        "host": socket.gethostname(),
        "device": str(mx.default_device()),
        "mlx_version": version("mlx"),
        "model_config": model.config.to_dict(),
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


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


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
