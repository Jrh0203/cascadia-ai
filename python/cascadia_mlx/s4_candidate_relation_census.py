"""Exact open-corpus census for linear-memory candidate-set context."""

from __future__ import annotations

import argparse
import json
import os
import socket
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import blake3
import numpy as np

from cascadia_mlx.graded_oracle_dataset import (
    GradedOracleDataset,
    GradedOracleGroupHeader,
    GradedOracleGroupRef,
    inspect_graded_oracle_candidate_records,
)
from cascadia_mlx.r3_action_edit_mlx_cache import (
    R3ActionEditMlxCache,
    open_data_verification_id,
    open_data_verification_identity,
)
from cascadia_mlx.s1_exact_supply_mlx_cache import S1ExactSupplyCache

SCHEMA_VERSION = 1
EXPERIMENT_ID = "s4-candidate-relation-foundation-v1"
K_VALUES = (64, 128, 256)
INDUCING_POINTS = (8, 16, 32)
NORMAL_95 = 1.959963984540054
UNCERTAINTY_FLOOR = 1.0
RELATIONS = (
    "same_draft",
    "same_frontier",
    "same_tile_pose",
    "same_wildlife_destination",
    "same_sibling_plan",
    "equivalent_afterstate",
)
_DRAFT_VARIANT_FIELDS = (
    "tile_q",
    "tile_r",
    "rotation",
    "wildlife_present",
    "wildlife_q",
    "wildlife_r",
    "immediate_score",
    "immediate_deltas",
)


def _canonical_blake3(value: Any) -> str:
    return blake3.blake3(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _checksum(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not an object: {path}")
    return value


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _normalize_host(host: str) -> str:
    lowered = host.lower()
    for known in ("john1", "john2", "john3", "john4"):
        if known in lowered:
            return known
    return host


def _void_rows(values: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values)
    if contiguous.ndim == 0:
        raise ValueError("fixed-width relation keys require a row dimension")
    bytes_by_row = contiguous.view(np.uint8).reshape(len(contiguous), -1)
    width = bytes_by_row.shape[1]
    return bytes_by_row.view(np.dtype((np.void, width))).reshape(-1)


def _canonical_structured_copy(values: np.ndarray) -> np.ndarray:
    """Copy named fields into zero-filled storage so padding is deterministic."""
    values = np.asarray(values)
    if values.dtype.names is None:
        raise ValueError("canonical structured copies require named fields")
    canonical = np.zeros(values.shape, dtype=values.dtype)
    for field in values.dtype.names:
        canonical[field] = values[field]
    return canonical


def candidate_relation_keys(
    actions: np.ndarray,
    afterstate_hashes: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return exact fixed-width relation keys and their validity masks."""
    actions = np.asarray(actions)
    afterstate_hashes = np.asarray(afterstate_hashes, dtype=np.uint8)
    if actions.ndim != 1:
        raise ValueError("candidate actions must be a rank-one structured array")
    if afterstate_hashes.shape != (len(actions), 32):
        raise ValueError("candidate afterstate hashes must be [candidate, 32]")

    draft = _canonical_structured_copy(actions)
    for field in _DRAFT_VARIANT_FIELDS:
        draft[field] = 0
    draft_keys = _void_rows(draft)

    frontier = np.empty((len(actions), 2), dtype=np.int8)
    frontier[:, 0] = actions["tile_q"]
    frontier[:, 1] = actions["tile_r"]
    frontier_keys = _void_rows(frontier)

    tile_pose = np.empty((len(actions), 3), dtype=np.int8)
    tile_pose[:, :2] = frontier
    tile_pose[:, 2] = actions["rotation"].astype(np.int8)
    tile_pose_keys = _void_rows(tile_pose)

    wildlife = np.empty((len(actions), 2), dtype=np.int8)
    wildlife[:, 0] = actions["wildlife_q"]
    wildlife[:, 1] = actions["wildlife_r"]
    wildlife_keys = _void_rows(wildlife)
    wildlife_valid = actions["wildlife_present"].astype(np.bool_)

    draft_bytes = np.ascontiguousarray(draft).view(np.uint8).reshape(len(actions), -1)
    pose_bytes = np.ascontiguousarray(tile_pose).view(np.uint8).reshape(len(actions), -1)
    sibling_keys = _void_rows(np.concatenate([draft_bytes, pose_bytes], axis=1))

    valid = np.ones(len(actions), dtype=np.bool_)
    return {
        "same_draft": (draft_keys, valid),
        "same_frontier": (frontier_keys, valid),
        "same_tile_pose": (tile_pose_keys, valid),
        "same_wildlife_destination": (wildlife_keys, wildlife_valid),
        "same_sibling_plan": (sibling_keys, valid),
        "equivalent_afterstate": (_void_rows(afterstate_hashes), valid),
    }


def stable_screen_order(candidates: np.ndarray) -> np.ndarray:
    """Order by observable screen rank and break ties by exact action hash."""
    candidates = np.asarray(candidates)
    keys = np.empty(
        len(candidates),
        dtype=[("screen_rank", "<u2"), ("action_hash", "V32")],
    )
    keys["screen_rank"] = candidates["screen_rank"]
    keys["action_hash"] = _void_rows(candidates["action_hash"])
    return np.argsort(keys, order=("screen_rank", "action_hash"), kind="stable")


def confidence_set(
    means: np.ndarray,
    stddev: np.ndarray,
    samples: np.ndarray,
    mask: np.ndarray,
    winner: int,
) -> np.ndarray:
    """Match the frozen ADR 0150 pairwise 95 percent confidence set."""
    means = np.asarray(means, dtype=np.float64)
    stddev = np.asarray(stddev, dtype=np.float64)
    samples = np.asarray(samples, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.bool_)
    if winner < 0 or winner >= len(means) or not mask[winner]:
        raise ValueError("winner must have an R4800 label")
    standard_error = np.sqrt(np.square(stddev) / np.maximum(samples, 1.0) + UNCERTAINTY_FLOOR**2)
    pairwise = np.sqrt(np.square(standard_error[winner]) + np.square(standard_error))
    result = np.zeros(len(means), dtype=np.bool_)
    result[mask] = means[winner] - means[mask] <= NORMAL_95 * pairwise[mask]
    return result


def retained_regret(
    retained: np.ndarray,
    teacher: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Return loss from keeping only a candidate subset under R4800 labels."""
    retained = np.asarray(retained, dtype=np.int64)
    teacher = np.asarray(teacher, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.bool_)
    labeled = teacher[mask]
    if not len(labeled):
        raise ValueError("candidate group has no R4800 labels")
    retained_labeled = retained[mask[retained]]
    if not len(retained_labeled):
        return float(np.max(labeled) - np.min(labeled))
    return float(np.max(labeled) - np.max(teacher[retained_labeled]))


def _relation_to_anchor(
    keys: np.ndarray,
    valid: np.ndarray,
    anchors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    count = len(keys)
    anchor_mask = np.zeros(count, dtype=np.bool_)
    anchor_mask[anchors] = True
    valid_anchors = anchors[valid[anchors]]
    linked = np.zeros(count, dtype=np.bool_)
    anchor_siblings = np.zeros(count, dtype=np.bool_)
    if not len(valid_anchors):
        return linked, anchor_siblings, 0, 0

    unique, counts = np.unique(keys[valid_anchors], return_counts=True)
    query_indices = np.flatnonzero(valid)
    insertion = np.searchsorted(unique, keys[query_indices])
    in_range = insertion < len(unique)
    matched = np.zeros(len(query_indices), dtype=np.bool_)
    matched[in_range] = unique[insertion[in_range]] == keys[query_indices[in_range]]
    degrees = np.zeros(len(query_indices), dtype=np.int64)
    degrees[matched] = counts[insertion[matched]]
    degrees -= anchor_mask[query_indices]
    linked[query_indices] = degrees > 0

    anchor_insertion = np.searchsorted(unique, keys[valid_anchors])
    anchor_siblings[valid_anchors] = counts[anchor_insertion] > 1
    edges = int(np.sum(counts * (counts - 1) // 2))
    return linked, anchor_siblings, len(unique), edges


def _union_graph(
    relation_keys: dict[str, tuple[np.ndarray, np.ndarray]],
    anchors: np.ndarray,
) -> dict[str, int | float]:
    width = len(anchors)
    parent = np.arange(width, dtype=np.int32)
    degree = np.zeros(width, dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    relation_edges = 0
    for keys, valid in relation_keys.values():
        valid_positions = np.flatnonzero(valid[anchors])
        if not len(valid_positions):
            continue
        selected_keys = keys[anchors[valid_positions]]
        _, inverse, counts = np.unique(
            selected_keys,
            return_inverse=True,
            return_counts=True,
        )
        relation_edges += int(np.sum(counts * (counts - 1) // 2))
        first: dict[int, int] = {}
        for local_position, relation_id in zip(
            valid_positions,
            inverse,
            strict=True,
        ):
            relation = int(relation_id)
            position = int(local_position)
            degree[position] += int(counts[relation]) - 1
            if relation in first:
                union(first[relation], position)
            else:
                first[relation] = position

    component_sizes: dict[int, int] = defaultdict(int)
    for index in range(width):
        component_sizes[find(index)] += 1
    possible_edges = width * (width - 1) // 2
    return {
        "components": len(component_sizes),
        "largest_component": max(component_sizes.values(), default=0),
        "isolated_candidates": int(np.sum(degree == 0)),
        "relation_edges_with_overlap": relation_edges,
        "relation_edge_density_with_overlap": (
            relation_edges / possible_edges if possible_edges else 0.0
        ),
    }


def analyze_candidate_group(
    *,
    split: str,
    row: int,
    header: GradedOracleGroupHeader,
    candidates: np.ndarray,
    afterstate_hashes: np.ndarray,
    selected_index: int,
) -> dict[str, Any]:
    """Measure exact target retention and candidate-relation topology."""
    if split not in {"train", "validation"}:
        raise ValueError("candidate relation census accepts train or validation")
    if not len(candidates) or selected_index < 0 or selected_index >= len(candidates):
        raise ValueError("candidate relation census group is malformed")
    if int(candidates["r4800"]["samples"][selected_index]) <= 0:
        raise ValueError("candidate relation census winner lacks R4800")

    order = stable_screen_order(candidates)
    inverse_order = np.empty(len(order), dtype=np.int64)
    inverse_order[order] = np.arange(len(order), dtype=np.int64)
    r4800 = candidates["r4800"]
    r4800_mask = r4800["samples"] > 0
    confidence = confidence_set(
        r4800["mean"],
        r4800["stddev"],
        r4800["samples"],
        r4800_mask,
        selected_index,
    )
    relation_keys = candidate_relation_keys(candidates["action"], afterstate_hashes)
    contexts: dict[str, Any] = {}
    for requested_k in K_VALUES:
        anchors = order[: min(requested_k, len(order))]
        anchor_mask = np.zeros(len(candidates), dtype=np.bool_)
        anchor_mask[anchors] = True
        relation_reports: dict[str, Any] = {}
        union_linked = np.zeros(len(candidates), dtype=np.bool_)
        union_anchor_siblings = np.zeros(len(candidates), dtype=np.bool_)
        for name in RELATIONS:
            keys, valid = relation_keys[name]
            linked, anchor_siblings, unique_keys, edges = _relation_to_anchor(
                keys,
                valid,
                anchors,
            )
            union_linked |= linked
            union_anchor_siblings |= anchor_siblings
            relation_reports[name] = {
                "unique_anchor_keys": unique_keys,
                "anchor_candidates_with_sibling": int(np.sum(anchor_siblings[anchors])),
                "anchor_pair_edges": edges,
                "all_candidates_linked_to_anchor": int(np.sum(linked)),
                "winner_linked_to_anchor": bool(linked[selected_index]),
                "confidence_candidates_linked_to_anchor": int(np.sum(linked & confidence)),
            }
        contexts[str(requested_k)] = {
            "requested_k": requested_k,
            "anchor_count": len(anchors),
            "winner_retained": bool(anchor_mask[selected_index]),
            "winner_screen_rank_zero_based": int(inverse_order[selected_index]),
            "confidence_set_covered": bool(np.any(anchor_mask & confidence)),
            "confidence_set_retained": int(np.sum(anchor_mask & confidence)),
            "confidence_set_size": int(np.sum(confidence)),
            "r4800_retained": int(np.sum(r4800_mask[anchors])),
            "r4800_total": int(np.sum(r4800_mask)),
            "retained_r4800_regret": retained_regret(
                anchors,
                r4800["mean"],
                r4800_mask,
            ),
            "union_all_candidates_linked_to_anchor": int(np.sum(union_linked)),
            "union_anchor_candidates_with_sibling": int(np.sum(union_anchor_siblings[anchors])),
            "union_winner_linked_to_anchor": bool(union_linked[selected_index]),
            "union_confidence_candidates_linked_to_anchor": int(np.sum(union_linked & confidence)),
            "graph": _union_graph(relation_keys, anchors),
            "attention_cost": {
                "dense_pair_scores": len(anchors) ** 2,
                "inducing_pair_scores": {
                    str(points): 2 * len(anchors) * points for points in INDUCING_POINTS
                },
            },
            "relations": relation_reports,
        }
    turn = int(header.turn)
    return {
        "split": split,
        "row": row,
        "group_id": int(header.group_id),
        "turn": turn,
        "phase": "early" if turn < 27 else "middle" if turn < 54 else "late",
        "candidate_count": len(candidates),
        "selected_index": selected_index,
        "selected_screen_rank_zero_based": int(inverse_order[selected_index]),
        "contexts": contexts,
    }


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "maximum": 0.0}
    return {
        "mean": float(np.mean(array)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p99": float(np.quantile(array, 0.99)),
        "maximum": float(np.max(array)),
    }


def _aggregate_context(records: Sequence[dict[str, Any]], requested_k: int) -> dict[str, Any]:
    contexts = [record["contexts"][str(requested_k)] for record in records]
    groups = len(contexts)
    candidate_total = sum(int(record["candidate_count"]) for record in records)
    anchor_total = sum(int(context["anchor_count"]) for context in contexts)
    relations: dict[str, Any] = {}
    for name in RELATIONS:
        reports = [context["relations"][name] for context in contexts]
        relations[name] = {
            "groups_with_anchor_collision": sum(
                report["anchor_candidates_with_sibling"] > 0 for report in reports
            ),
            "anchor_candidates_with_sibling": sum(
                report["anchor_candidates_with_sibling"] for report in reports
            ),
            "anchor_candidates_with_sibling_fraction": (
                sum(report["anchor_candidates_with_sibling"] for report in reports) / anchor_total
                if anchor_total
                else 0.0
            ),
            "anchor_pair_edges": sum(report["anchor_pair_edges"] for report in reports),
            "all_candidates_linked_to_anchor": sum(
                report["all_candidates_linked_to_anchor"] for report in reports
            ),
            "all_candidates_linked_to_anchor_fraction": (
                sum(report["all_candidates_linked_to_anchor"] for report in reports)
                / candidate_total
                if candidate_total
                else 0.0
            ),
            "winner_linked_to_anchor_fraction": (
                sum(report["winner_linked_to_anchor"] for report in reports) / groups
                if groups
                else 0.0
            ),
        }
    return {
        "groups": groups,
        "candidates": candidate_total,
        "anchors": anchor_total,
        "winner_retention": (
            sum(context["winner_retained"] for context in contexts) / groups if groups else 0.0
        ),
        "confidence_set_coverage": (
            sum(context["confidence_set_covered"] for context in contexts) / groups
            if groups
            else 0.0
        ),
        "mean_retained_r4800_regret": (
            float(np.mean([context["retained_r4800_regret"] for context in contexts]))
            if groups
            else 0.0
        ),
        "union_all_candidates_linked_to_anchor_fraction": (
            sum(context["union_all_candidates_linked_to_anchor"] for context in contexts)
            / candidate_total
            if candidate_total
            else 0.0
        ),
        "union_anchor_candidates_with_sibling_fraction": (
            sum(context["union_anchor_candidates_with_sibling"] for context in contexts)
            / anchor_total
            if anchor_total
            else 0.0
        ),
        "union_winner_linked_to_anchor_fraction": (
            sum(context["union_winner_linked_to_anchor"] for context in contexts) / groups
            if groups
            else 0.0
        ),
        "graph": {
            "components": _quantiles([context["graph"]["components"] for context in contexts]),
            "largest_component": _quantiles(
                [context["graph"]["largest_component"] for context in contexts]
            ),
            "isolated_candidates": _quantiles(
                [context["graph"]["isolated_candidates"] for context in contexts]
            ),
            "relation_edges_with_overlap": sum(
                context["graph"]["relation_edges_with_overlap"] for context in contexts
            ),
        },
        "attention_cost": {
            "dense_pair_scores": sum(
                context["attention_cost"]["dense_pair_scores"] for context in contexts
            ),
            "inducing_pair_scores": {
                str(points): sum(
                    context["attention_cost"]["inducing_pair_scores"][str(points)]
                    for context in contexts
                )
                for points in INDUCING_POINTS
            },
        },
        "relations": relations,
    }


def aggregate_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate exact shard records, including fixed width and phase strata."""
    ordered = sorted(records, key=lambda record: (record["split"], record["row"]))
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in ordered:
        by_split[str(record["split"])].append(record)
    result: dict[str, Any] = {}
    for split, split_records in sorted(by_split.items()):
        strata: dict[str, dict[str, list[dict[str, Any]]]] = {
            "phase": defaultdict(list),
            "action_width": defaultdict(list),
        }
        for record in split_records:
            strata["phase"][str(record["phase"])].append(record)
            count = int(record["candidate_count"])
            bucket = (
                "0001-0512"
                if count <= 512
                else "0513-2048"
                if count <= 2048
                else "2049-4096"
                if count <= 4096
                else "4097-plus"
            )
            strata["action_width"][bucket].append(record)
        result[split] = {
            "groups": len(split_records),
            "candidates": sum(record["candidate_count"] for record in split_records),
            "candidate_count": _quantiles([record["candidate_count"] for record in split_records]),
            "selected_screen_rank_zero_based": _quantiles(
                [record["selected_screen_rank_zero_based"] for record in split_records]
            ),
            "contexts": {str(k): _aggregate_context(split_records, k) for k in K_VALUES},
            "strata": {
                dimension: {
                    key: {str(k): _aggregate_context(values, k) for k in K_VALUES}
                    for key, values in sorted(groups.items())
                }
                for dimension, groups in strata.items()
            },
        }
    return result


def _selected_rows(total: int, modulus: int, remainder: int) -> list[int]:
    if modulus <= 0 or remainder < 0 or remainder >= modulus:
        raise ValueError("row shard modulus/remainder is invalid")
    return [row for row in range(total) if row % modulus == remainder]


def _dataset_locations(
    dataset: GradedOracleDataset,
) -> dict[int, tuple[np.memmap, GradedOracleGroupRef, GradedOracleGroupHeader]]:
    locations: dict[
        int,
        tuple[np.memmap, GradedOracleGroupRef, GradedOracleGroupHeader],
    ] = {}
    for raw, ref, header in dataset.raw_group_headers():
        if header.group_id in locations:
            raise ValueError(f"graded-oracle group ID is duplicated: {header.group_id}")
        locations[header.group_id] = (raw, ref, header)
    return locations


def build_shard_records(
    *,
    cache: R3ActionEditMlxCache,
    train_dataset: GradedOracleDataset,
    validation_dataset: GradedOracleDataset,
    modulus: int,
    remainder: int,
) -> list[dict[str, Any]]:
    """Build one disjoint modulo shard over both open splits."""
    records: list[dict[str, Any]] = []
    datasets = {"train": train_dataset, "validation": validation_dataset}
    for split in ("train", "validation"):
        dataset = datasets[split]
        if dataset.split != split:
            raise ValueError(f"{split} dataset manifest names the wrong split")
        source = cache.splits[split]
        locations = _dataset_locations(dataset)
        tensors = source.tensors
        offsets = np.asarray(tensors["candidate_offsets"], dtype=np.uint64)
        group_ids = np.asarray(tensors["group_ids"], dtype=np.uint64)
        for row in _selected_rows(source.groups, modulus, remainder):
            group_id = int(group_ids[row])
            if group_id not in locations:
                raise ValueError(f"{split} dataset is missing group {group_id}")
            raw, ref, header = locations[group_id]
            full_candidates = inspect_graded_oracle_candidate_records(raw, ref)
            start = int(offsets[row])
            end = int(offsets[row + 1])
            source_indices = np.asarray(
                tensors["source_candidate_indices"][start:end],
                dtype=np.int64,
            )
            candidates = full_candidates[source_indices]
            expected_hashes = np.asarray(
                tensors["action_hashes"][start:end],
                dtype=np.uint8,
            )
            if not np.array_equal(candidates["action_hash"], expected_hashes):
                raise ValueError(f"{split} action identity drifted at row {row}")
            selected_source = int(tensors["selected_source_indices"][row])
            selected_matches = np.flatnonzero(source_indices == selected_source)
            if len(selected_matches) != 1:
                raise ValueError(f"{split} selected action is absent at row {row}")
            records.append(
                analyze_candidate_group(
                    split=split,
                    row=row,
                    header=header,
                    candidates=candidates,
                    afterstate_hashes=np.asarray(
                        tensors["control_after_hashes"][start:end],
                        dtype=np.uint8,
                    ),
                    selected_index=int(selected_matches[0]),
                )
            )
    return records


def run_shard(
    *,
    train_dataset_root: Path,
    validation_dataset_root: Path,
    cache_root: Path,
    s1_cache_root: Path,
    authorization_path: Path,
    modulus: int,
    remainder: int,
) -> dict[str, Any]:
    """Validate the frozen open-data proof and execute one census shard."""
    authorization = _read_json(authorization_path, "ADR 0150 authorization")
    identity = authorization.get("identity")
    if authorization.get("approved") is not True or not isinstance(identity, dict):
        raise ValueError("ADR 0150 authorization is invalid")
    expected_open_data = identity.get("open_data_verification")
    if not isinstance(expected_open_data, dict):
        raise ValueError("ADR 0150 open-data proof is absent")
    proof_id = open_data_verification_id(expected_open_data)
    if proof_id != identity.get("open_data_verification_id"):
        raise ValueError("ADR 0150 open-data proof digest differs")

    cache = R3ActionEditMlxCache(
        cache_root,
        verify_checksums=False,
        verify_semantics=False,
        require_complete=True,
    )
    s1_cache = S1ExactSupplyCache(
        s1_cache_root,
        verify_checksums=False,
        verify_semantics=False,
        require_complete=True,
    )
    observed_open_data = open_data_verification_identity(
        cache=cache,
        s1_cache=s1_cache,
        train_dataset=train_dataset_root,
        validation_dataset=validation_dataset_root,
    )
    if observed_open_data != expected_open_data:
        raise ValueError("S4 census open-data identity differs from authorization")
    train_dataset = GradedOracleDataset(
        train_dataset_root,
        verify_checksums=False,
    )
    validation_dataset = GradedOracleDataset(
        validation_dataset_root,
        verify_checksums=False,
    )
    records = build_shard_records(
        cache=cache,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        modulus=modulus,
        remainder=remainder,
    )
    records.sort(key=lambda record: (record["split"], record["row"]))
    scientific_identity = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "open_data_verification_id": proof_id,
        "cache_id": cache.cache_id,
        "row_shard": {"modulus": modulus, "remainder": remainder},
        "rows": {
            split: [record["row"] for record in records if record["split"] == split]
            for split in ("train", "validation")
        },
        "group_ids_blake3": _canonical_blake3(
            [[record["split"], record["row"], record["group_id"]] for record in records]
        ),
    }
    report = {
        **scientific_identity,
        "scientific_identity": scientific_identity,
        "operational": {
            "host": _normalize_host(socket.gethostname().split(".")[0]),
        },
        "records": records,
        "aggregate": aggregate_records(records),
    }
    report["report_id"] = _canonical_blake3(
        {
            "scientific_identity": scientific_identity,
            "records": records,
            "aggregate": report["aggregate"],
        }
    )
    return report


def merge_reports(reports: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Merge disjoint modulo shards and require complete two-split coverage."""
    values = list(reports)
    if not values:
        raise ValueError("S4 relation census merge requires shard reports")
    proof_ids = {report.get("open_data_verification_id") for report in values}
    cache_ids = {report.get("cache_id") for report in values}
    moduli = {
        report.get("scientific_identity", {}).get("row_shard", {}).get("modulus")
        for report in values
    }
    if len(proof_ids) != 1 or len(cache_ids) != 1 or len(moduli) != 1:
        raise ValueError("S4 relation census shard identities differ")
    modulus = next(iter(moduli))
    if not isinstance(modulus, int) or modulus <= 0 or len(values) != modulus:
        raise ValueError("S4 relation census requires exactly one report per remainder")
    remainders = {
        report.get("scientific_identity", {}).get("row_shard", {}).get("remainder")
        for report in values
    }
    if remainders != set(range(modulus)):
        raise ValueError("S4 relation census shard remainders are incomplete")

    records: list[dict[str, Any]] = []
    for report in values:
        if report.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("S4 relation census shard schema differs")
        shard_records = report.get("records")
        if not isinstance(shard_records, list):
            raise ValueError("S4 relation census shard records are absent")
        records.extend(shard_records)
    records.sort(key=lambda record: (record["split"], record["row"]))
    keys = [(record["split"], record["row"]) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("S4 relation census shard rows overlap")
    expected = [
        *(("train", row) for row in range(560)),
        *(("validation", row) for row in range(240)),
    ]
    if keys != expected:
        raise ValueError("S4 relation census merge does not cover the full open corpus")

    scientific_identity = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "open_data_verification_id": next(iter(proof_ids)),
        "cache_id": next(iter(cache_ids)),
        "shards": modulus,
        "train_groups": 560,
        "validation_groups": 240,
        "records_blake3": _canonical_blake3(records),
    }
    aggregate = aggregate_records(records)
    merged = {
        **scientific_identity,
        "scientific_identity": scientific_identity,
        "source_report_ids": sorted(str(report["report_id"]) for report in values),
        "aggregate": aggregate,
    }
    merged["report_id"] = _canonical_blake3(merged)
    return merged


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    shard = subparsers.add_parser("shard")
    shard.add_argument("--train-dataset", type=Path, required=True)
    shard.add_argument("--validation-dataset", type=Path, required=True)
    shard.add_argument("--cache", type=Path, required=True)
    shard.add_argument("--s1-cache", type=Path, required=True)
    shard.add_argument("--authorization", type=Path, required=True)
    shard.add_argument("--row-modulus", type=int, required=True)
    shard.add_argument("--row-remainder", type=int, required=True)
    shard.add_argument("--output", type=Path, required=True)
    merge = subparsers.add_parser("merge")
    merge.add_argument("--report", type=Path, action="append", required=True)
    merge.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "shard":
        report = run_shard(
            train_dataset_root=args.train_dataset,
            validation_dataset_root=args.validation_dataset,
            cache_root=args.cache,
            s1_cache_root=args.s1_cache,
            authorization_path=args.authorization,
            modulus=args.row_modulus,
            remainder=args.row_remainder,
        )
    else:
        report = merge_reports(_read_json(path, "S4 relation census shard") for path in args.report)
    _write_json_atomic(args.output, report)
    print(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "report_id": report["report_id"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
