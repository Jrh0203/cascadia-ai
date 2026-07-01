#!/usr/bin/env python3
"""Measure one exhaustive 6,372-action R2-MAP reference-service request."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import resource
import subprocess
import time
from pathlib import Path
from typing import Any

import blake3
import numpy as np
from cascadia_mlx.r2_map_model import parameter_count
from cascadia_mlx.r2_map_protocol_fixture import MODEL_IDENTITY, ExactFixtureModel
from cascadia_mlx.r2_map_serve import (
    PROTOCOL_MAGIC,
    PROTOCOL_VERSION,
    REFERENCE_MAX_CANDIDATES_PER_GROUP,
    REQUEST_SCHEMA,
    REQUEST_SCHEMA_BLAKE3,
    R2MapCheckpointRegistry,
    R2MapRegistryEntry,
    _request_identity_blake3,
    ordered_action_ids_blake3,
    score_grouped_request,
)
from cascadia_mlx.r2_map_tensor_contract import (
    BOARD_SLOTS,
    BOARD_TOKEN_CAPACITY,
    GLOBAL_FEATURES,
    MARKET_FEATURES,
    PLAYER_FEATURES,
    TOKEN_FEATURES,
)

_SWAP_USED_RE = re.compile(r"used = ([0-9.]+)([KMG])")


def _swap_used_bytes() -> int | None:
    if platform.system() != "Darwin":
        return None
    value = subprocess.run(
        ["sysctl", "-n", "vm.swapusage"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    match = _SWAP_USED_RE.search(value)
    if match is None:
        raise ValueError("cannot parse macOS swap usage")
    return int(float(match.group(1)) * {"K": 1024, "M": 1024**2, "G": 1024**3}[match.group(2)])


def _logical_zeros(shape: tuple[int, ...], dtype: str) -> np.ndarray[Any, Any]:
    """Represent zero-padded wire tensors without pre-touching redundant pages."""
    return np.broadcast_to(np.zeros((1,) * len(shape), dtype=dtype), shape)


def _source_identity(repository: Path) -> dict[str, Any]:
    files = [
        "python/cascadia_mlx/r2_map_model.py",
        "python/cascadia_mlx/r2_map_serve.py",
        "tools/r2_map_max_width_service_smoke.py",
    ]
    identities = {
        name: blake3.blake3((repository / name).read_bytes()).hexdigest() for name in files
    }
    return {
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "source_files_blake3": identities,
        "source_bundle_blake3": blake3.blake3(
            json.dumps(identities, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def _output_blake3(outputs: dict[str, np.ndarray[Any, Any]]) -> str:
    digest = blake3.blake3()
    for name, value in outputs.items():
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def run_smoke(*, run_dir: Path | None, checkpoint: Path | None) -> dict[str, Any]:
    count = REFERENCE_MAX_CANDIDATES_PER_GROUP
    action_ids = [blake3.blake3(index.to_bytes(8, "little")).hexdigest() for index in range(count)]
    group = {
        "group_id": blake3.blake3(b"maximum-width-group").hexdigest(),
        "decision_id": blake3.blake3(b"maximum-width-decision").hexdigest(),
        "model": dict(MODEL_IDENTITY),
        "expected_legal_action_count": count,
        "action_ids": action_ids,
        "enumeration_indices": list(range(count)),
        "ordered_action_ids_blake3": ordered_action_ids_blake3(action_ids),
    }
    tensors: dict[str, np.ndarray[Any, Any]] = {
        "candidate_offsets": np.asarray([0, count], dtype="<i4"),
        "parent_token_features": _logical_zeros(
            (1, BOARD_SLOTS, BOARD_TOKEN_CAPACITY, TOKEN_FEATURES), "<f4"
        ),
        "parent_token_types": _logical_zeros(
            (1, BOARD_SLOTS, BOARD_TOKEN_CAPACITY), "<i4"
        ),
        "parent_token_mask": _logical_zeros(
            (1, BOARD_SLOTS, BOARD_TOKEN_CAPACITY), "|u1"
        ),
        "parent_market_features": _logical_zeros((1, 4, MARKET_FEATURES), "<f4"),
        "parent_market_mask": np.ones((1, 4), dtype="|u1"),
        "parent_player_features": _logical_zeros((1, BOARD_SLOTS, PLAYER_FEATURES), "<f4"),
        "parent_player_mask": np.ones((1, BOARD_SLOTS), dtype="|u1"),
        "parent_global_features": _logical_zeros((1, GLOBAL_FEATURES), "<f4"),
        "candidate_token_features": _logical_zeros(
            (count, BOARD_SLOTS, BOARD_TOKEN_CAPACITY, TOKEN_FEATURES), "<f4"
        ),
        "candidate_token_types": _logical_zeros(
            (count, BOARD_SLOTS, BOARD_TOKEN_CAPACITY), "<i4"
        ),
        "candidate_token_mask": _logical_zeros(
            (count, BOARD_SLOTS, BOARD_TOKEN_CAPACITY), "|u1"
        ),
        "candidate_market_features": _logical_zeros((count, 4, MARKET_FEATURES), "<f4"),
        "candidate_market_mask": np.ones((count, 4), dtype="|u1"),
        "candidate_player_features": _logical_zeros(
            (count, BOARD_SLOTS, PLAYER_FEATURES), "<f4"
        ),
        "candidate_player_mask": np.ones((count, BOARD_SLOTS), dtype="|u1"),
        "candidate_global_features": _logical_zeros((count, GLOBAL_FEATURES), "<f4"),
        "action_bytes": _logical_zeros((count, 128), "|u1"),
        "exact_afterstate_scores": np.arange(count, dtype="<f4"),
    }
    metadata = {
        "schema_version": 1,
        "schema_id": REQUEST_SCHEMA,
        "request_schema_blake3": REQUEST_SCHEMA_BLAKE3,
        "group_count": 1,
        "candidate_count": count,
        "groups": [group],
    }
    registry = R2MapCheckpointRegistry(capacity=1)
    if run_dir is None or checkpoint is None:
        entry = R2MapRegistryEntry(model=ExactFixtureModel(), **MODEL_IDENTITY)
        registry.register_model(entry)
        model_kind = "deterministic-protocol-fixture"
        model_parameters = None
    else:
        entry = registry.register_verified_checkpoint(
            run_dir=run_dir, checkpoint_path=checkpoint, pinned=True
        )
        model_kind = "verified-width-192-r2-map-checkpoint"
        model_parameters = parameter_count(entry.model)
    group["model"] = entry.identity()
    swap_before = _swap_used_bytes()
    started = time.perf_counter()
    response, outputs = score_grouped_request(registry, metadata, tensors)
    elapsed = time.perf_counter() - started
    swap_after = _swap_used_bytes()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss = int(usage.ru_maxrss)
    if platform.system() != "Darwin":
        peak_rss *= 1024
    scores = outputs["action_scores"]
    report = {
        "schema_version": 1,
        "schema_id": "r2-map-maximum-width-service-smoke-v1",
        "source_identity": _source_identity(Path(__file__).resolve().parents[1]),
        "protocol": {
            "magic": PROTOCOL_MAGIC.decode(),
            "version": PROTOCOL_VERSION,
            "request_schema": REQUEST_SCHEMA,
            "request_schema_blake3": REQUEST_SCHEMA_BLAKE3,
        },
        "model_kind": model_kind,
        "model_identity": entry.identity(),
        "model_config": None if model_parameters is None else entry.model.config.to_dict(),
        "model_parameter_count": model_parameters,
        "run_dir": None if run_dir is None else str(run_dir.resolve()),
        "checkpoint": None if checkpoint is None else str(checkpoint.resolve()),
        "request_identity_blake3": _request_identity_blake3(metadata),
        "group_id": group["group_id"],
        "decision_id": group["decision_id"],
        "ordered_action_ids_blake3": group["ordered_action_ids_blake3"],
        "output_tensors_blake3": _output_blake3(outputs),
        "candidate_count": count,
        "response_candidate_count": response["candidate_count"],
        "response_action_ids_preserve_order": response["groups"][0]["action_ids"] == action_ids,
        "all_scores_finite": bool(np.all(np.isfinite(scores))),
        "all_actions_scored_once": response["groups"][0]["diagnostics"]["actions_scored"] == count,
        "pruned_actions": response["diagnostics"]["pruned_actions"],
        "remote_inference": response["diagnostics"]["remote_inference"],
        "elapsed_seconds": elapsed,
        "actions_per_second": count / elapsed,
        "peak_process_rss_bytes": peak_rss,
        "process_swaps": int(getattr(usage, "ru_nswap", 0)),
        "system_swap_before_bytes": swap_before,
        "system_swap_after_bytes": swap_after,
        "system_swap_delta_bytes": (
            None if swap_before is None or swap_after is None else swap_after - swap_before
        ),
    }
    report["passed"] = all(
        [
            report["response_candidate_count"] == count,
            report["response_action_ids_preserve_order"],
            report["all_scores_finite"],
            report["all_actions_scored_once"],
            report["pruned_actions"] == 0,
            report["remote_inference"] is False,
            report["process_swaps"] == 0,
            report["system_swap_delta_bytes"] is not None,
            report["system_swap_delta_bytes"] <= 0,
        ]
    )
    registry.close()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if (args.run_dir is None) != (args.checkpoint is None):
        parser.error("--run-dir and --checkpoint must be supplied together")
    report = run_smoke(run_dir=args.run_dir, checkpoint=args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
