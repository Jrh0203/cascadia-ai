#!/usr/bin/env python3
"""Audit exact model-visible collisions in ADR 0115 tile retrieval."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import socket
import time
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any

import blake3
import numpy as np
from cascadia_mlx.full_legal_hierarchical_factor_retrieval import (
    EXPERIMENT_ID,
    HierarchicalFactorCache,
)

ANALYSIS_ID = "conditional-tile-exact-input-collision-audit-v1"


def _fingerprint(
    state_digest: bytes,
    context: np.ndarray,
    item: np.ndarray,
) -> bytes:
    hasher = blake3.blake3()
    hasher.update(state_digest)
    hasher.update(np.ascontiguousarray(context).tobytes())
    hasher.update(np.ascontiguousarray(item).tobytes())
    return hasher.digest()


def _packed_label(target: bool, rank_mask: bool, rank: float) -> int:
    rank_bits = int(np.asarray(np.float32(rank)).view(np.uint32).item())
    return int(target) | (int(rank_mask) << 1) | (rank_bits << 2)


def _raw_representation(
    state: np.ndarray,
    context: np.ndarray,
    item: np.ndarray,
) -> bytes:
    return b"".join(np.ascontiguousarray(value).tobytes() for value in (state, context, item))


def _collision_details(
    cache: HierarchicalFactorCache,
    candidate_fingerprints: set[bytes],
) -> dict[str, int]:
    if not candidate_fingerprints:
        return {
            "exact_duplicate_representations": 0,
            "exact_target_conflicting_representations": 0,
            "exact_rank_conflicting_representations": 0,
            "target_positive_occurrences_in_conflicts": 0,
            "occurrences_in_target_conflicts": 0,
        }
    observations: dict[bytes, dict[bytes, list[int]]] = defaultdict(lambda: defaultdict(list))
    for arrays in cache.iter_shards():
        offsets = arrays["tile_query_offsets"]
        groups = arrays["tile_query_group"]
        contexts = arrays["tile_query_context"]
        items = arrays["tile_item_features"]
        targets = arrays["tile_item_target"]
        ranks = arrays["tile_item_rank"]
        rank_masks = arrays["tile_item_rank_mask"]
        state_digests = [
            blake3.blake3(np.ascontiguousarray(state).tobytes()).digest()
            for state in arrays["group_state"]
        ]
        for query_index, (left, right) in enumerate(pairwise(offsets)):
            group_index = int(groups[query_index])
            state_digest = state_digests[group_index]
            context = contexts[query_index]
            for item_index in range(int(left), int(right)):
                fingerprint = _fingerprint(
                    state_digest,
                    context,
                    items[item_index],
                )
                if fingerprint not in candidate_fingerprints:
                    continue
                representation = _raw_representation(
                    arrays["group_state"][group_index],
                    context,
                    items[item_index],
                )
                observations[fingerprint][representation].append(
                    _packed_label(
                        bool(targets[item_index]),
                        bool(rank_masks[item_index]),
                        float(ranks[item_index]),
                    )
                )

    exact_duplicates = 0
    target_conflicts = 0
    rank_conflicts = 0
    positive_occurrences = 0
    target_conflict_occurrences = 0
    for representations in observations.values():
        for labels in representations.values():
            if len(labels) > 1:
                exact_duplicates += 1
            targets = {label & 1 for label in labels}
            ranks = {label >> 1 for label in labels}
            if len(targets) > 1:
                target_conflicts += 1
                positive_occurrences += sum(label & 1 for label in labels)
                target_conflict_occurrences += len(labels)
            if len(ranks) > 1:
                rank_conflicts += 1
    return {
        "exact_duplicate_representations": exact_duplicates,
        "exact_target_conflicting_representations": target_conflicts,
        "exact_rank_conflicting_representations": rank_conflicts,
        "target_positive_occurrences_in_conflicts": positive_occurrences,
        "occurrences_in_target_conflicts": target_conflict_occurrences,
    }


def audit_cache(cache: HierarchicalFactorCache) -> dict[str, Any]:
    """Measure contradictory labels for byte-identical tile model inputs."""
    first_label: dict[bytes, int] = {}
    candidate_conflicts: set[bytes] = set()
    queries = 0
    items_total = 0
    target_positives = 0
    for arrays in cache.iter_shards():
        offsets = arrays["tile_query_offsets"]
        groups = arrays["tile_query_group"]
        contexts = arrays["tile_query_context"]
        items = arrays["tile_item_features"]
        targets = arrays["tile_item_target"]
        ranks = arrays["tile_item_rank"]
        rank_masks = arrays["tile_item_rank_mask"]
        state_digests = [
            blake3.blake3(np.ascontiguousarray(state).tobytes()).digest()
            for state in arrays["group_state"]
        ]
        for query_index, (left, right) in enumerate(pairwise(offsets)):
            queries += 1
            group_index = int(groups[query_index])
            state_digest = state_digests[group_index]
            context = contexts[query_index]
            for item_index in range(int(left), int(right)):
                target = bool(targets[item_index])
                label = _packed_label(
                    target,
                    bool(rank_masks[item_index]),
                    float(ranks[item_index]),
                )
                fingerprint = _fingerprint(
                    state_digest,
                    context,
                    items[item_index],
                )
                previous = first_label.setdefault(fingerprint, label)
                if previous != label:
                    candidate_conflicts.add(fingerprint)
                items_total += 1
                target_positives += int(target)

    details = _collision_details(cache, candidate_conflicts)
    return {
        "split": cache.split,
        "cache_payload_blake3": cache.manifest["payload_blake3"],
        "queries": queries,
        "items": items_total,
        "target_positive_occurrences": target_positives,
        "unique_model_input_fingerprints": len(first_label),
        "repeated_fingerprint_occurrences": items_total - len(first_label),
        "candidate_conflicting_fingerprints": len(candidate_conflicts),
        **details,
        "target_positive_conflict_fraction": (
            details["target_positive_occurrences_in_conflicts"] / max(target_positives, 1)
        ),
        "exact_target_collision_material_at_1pct": (
            details["target_positive_occurrences_in_conflicts"] / max(target_positives, 1) >= 0.01
        ),
        "all_queries_covered": queries == int(cache.manifest["queries"]["tile"]),
        "all_items_covered": items_total == int(cache.manifest["items"]["tile"]),
    }


def _resource_usage() -> dict[str, int]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak = int(usage.ru_maxrss)
    if platform.system() != "Darwin":
        peak *= 1024
    return {
        "peak_rss_bytes": peak,
        "process_swaps": int(usage.ru_nswap),
    }


def run(cache_root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    cache = HierarchicalFactorCache(cache_root)
    scientific = audit_cache(cache)
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "analysis": ANALYSIS_ID,
        "host": socket.gethostname(),
        "scientific": scientific,
        "scientific_blake3": blake3.blake3(
            json.dumps(
                scientific,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "execution": {
            "elapsed_seconds": time.perf_counter() - started,
            **_resource_usage(),
        },
        "training_used": False,
        "gradients_used": False,
        "optimizer_updates_used": False,
        "test_split_opened": False,
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.cache)
    _write_json(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
