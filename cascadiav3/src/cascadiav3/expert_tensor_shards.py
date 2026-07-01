"""Packed expert tensor shards for CascadiaFormer training."""

from __future__ import annotations

import argparse
import hashlib
import json
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SHARD_VERSION = "cascadiav3.expert_tensor_shard.v1"
TOKEN_FEATURE_DIM = 41
ACTION_FEATURE_DIM = 61


def _scalar_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _max_int(values: Any) -> int:
    return int(values.max(initial=0)) if getattr(values, "size", 0) else 0


@dataclass(frozen=True)
class ExpertTensorSummary:
    path: str
    version: str
    record_count: int
    total_token_count: int
    total_action_count: int
    total_relation_edge_count: int
    token_feature_dim: int
    action_feature_dim: int
    max_token_count: int
    max_action_count: int
    max_relation_edge_count: int
    output_bytes: int
    output_sha256: str
    relation_tail_present: bool
    relation_tail_shape: list[int] | None
    relation_tail_dtype: str | None
    relation_tail_token_capacity: int | None
    relation_tail_action_capacity: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "version": self.version,
            "record_count": self.record_count,
            "total_token_count": self.total_token_count,
            "total_action_count": self.total_action_count,
            "total_relation_edge_count": self.total_relation_edge_count,
            "token_feature_dim": self.token_feature_dim,
            "action_feature_dim": self.action_feature_dim,
            "max_token_count": self.max_token_count,
            "max_action_count": self.max_action_count,
            "max_relation_edge_count": self.max_relation_edge_count,
            "output_bytes": self.output_bytes,
            "output_sha256": self.output_sha256,
            "bytes_per_record": self.output_bytes / max(1, self.record_count),
            "relation_tail_present": self.relation_tail_present,
            "relation_tail_shape": self.relation_tail_shape,
            "relation_tail_dtype": self.relation_tail_dtype,
            "relation_tail_token_capacity": self.relation_tail_token_capacity,
            "relation_tail_action_capacity": self.relation_tail_action_capacity,
        }


class ExpertTensorShard:
    def __init__(self, path: Path) -> None:
        import numpy as np

        self.path = path
        self._npz = np.load(path, allow_pickle=False)
        self.version = _scalar_string(self._npz["version"].item())
        if self.version != SHARD_VERSION:
            raise ValueError(f"unsupported expert tensor shard version {self.version!r}")
        self.metadata = json.loads(_scalar_string(self._npz["metadata_json"].item()))
        self.tokens = self._npz["tokens"]
        self.actions = self._npz["actions"]
        self.token_offsets = self._npz["token_offsets"]
        self.action_offsets = self._npz["action_offsets"]
        self.relation_edges = self._npz["relation_edges"]
        self.relation_offsets = self._npz["relation_offsets"]
        self.selected_action_index = self._npz["selected_action_index"]
        self.target_q = self._npz["target_q"]
        self.target_score_to_go = self._npz["target_score_to_go"]
        self.q_valid = self._npz["q_valid"]
        self.priors = self._npz["priors"]
        self.visits = self._npz["visits"]
        self.q_variance = self._npz["q_variance"]
        self.q_count = self._npz["q_count"]
        self.truncated_count = self._npz["truncated_count"]
        self.exact_afterstate_score_active = self._npz["exact_afterstate_score_active"]
        self.final_score_vector = self._npz["final_score_vector"]
        self.rank_vector = self._npz["rank_vector"]
        self.score_decomposition = self._npz["score_decomposition"]
        self.relation_tail = self._npz["relation_tail"] if "relation_tail" in self._npz.files else None
        self._validate_shapes()

    def _validate_shapes(self) -> None:
        if self.tokens.ndim != 2 or self.tokens.shape[1] != TOKEN_FEATURE_DIM:
            raise ValueError(f"token feature shape mismatch: {self.tokens.shape}")
        if self.actions.ndim != 2 or self.actions.shape[1] != ACTION_FEATURE_DIM:
            raise ValueError(f"action feature shape mismatch: {self.actions.shape}")
        if self.relation_edges.ndim != 2 or self.relation_edges.shape[1] != 3:
            raise ValueError(f"relation edge shape mismatch: {self.relation_edges.shape}")
        record_count = int(self.selected_action_index.shape[0])
        if self.token_offsets.shape[0] != record_count + 1:
            raise ValueError("token_offsets length mismatch")
        if self.action_offsets.shape[0] != record_count + 1:
            raise ValueError("action_offsets length mismatch")
        if self.relation_offsets.shape[0] != record_count + 1:
            raise ValueError("relation_offsets length mismatch")
        total_actions = int(self.actions.shape[0])
        for name in (
            "target_q",
            "target_score_to_go",
            "q_valid",
            "priors",
            "visits",
            "q_variance",
            "q_count",
            "truncated_count",
            "exact_afterstate_score_active",
        ):
            if getattr(self, name).shape[0] != total_actions:
                raise ValueError(f"{name} length mismatch")
        if self.final_score_vector.shape != (record_count, 4):
            raise ValueError("final_score_vector shape mismatch")
        if self.rank_vector.shape != (record_count, 4):
            raise ValueError("rank_vector shape mismatch")
        if self.score_decomposition.shape != (record_count, 3, 4):
            raise ValueError("score_decomposition shape mismatch")
        if self.relation_tail is not None:
            if self.relation_tail.ndim != 3:
                raise ValueError(f"relation_tail shape mismatch: {self.relation_tail.shape}")
            if self.relation_tail.shape[0] != record_count:
                raise ValueError("relation_tail record count mismatch")
            if self.relation_tail.dtype.name not in {"uint8", "int16", "int32", "int64"}:
                raise ValueError(f"relation_tail dtype must be integer, got {self.relation_tail.dtype}")
            if int(self.relation_tail.max(initial=0)) >= 256:
                raise ValueError("relation_tail relation ids must fit uint8 semantics")

    def __len__(self) -> int:
        return int(self.selected_action_index.shape[0])

    def example(self, index: int) -> dict[str, Any]:
        token_start = int(self.token_offsets[index])
        token_end = int(self.token_offsets[index + 1])
        action_start = int(self.action_offsets[index])
        action_end = int(self.action_offsets[index + 1])
        relation_start = int(self.relation_offsets[index])
        relation_end = int(self.relation_offsets[index + 1])
        example = {
            "tokens": self.tokens[token_start:token_end],
            "actions": self.actions[action_start:action_end],
            "relation_edges": self.relation_edges[relation_start:relation_end],
            "selected_action_index": int(self.selected_action_index[index]),
            "target_q": self.target_q[action_start:action_end],
            "target_score_to_go": self.target_score_to_go[action_start:action_end],
            "q_valid": self.q_valid[action_start:action_end],
            "priors": self.priors[action_start:action_end],
            "visits": self.visits[action_start:action_end],
            "q_variance": self.q_variance[action_start:action_end],
            "q_count": self.q_count[action_start:action_end],
            "truncated_count": self.truncated_count[action_start:action_end],
            "exact_afterstate_score_active": self.exact_afterstate_score_active[action_start:action_end],
            "final_score_vector": self.final_score_vector[index],
            "rank_vector": self.rank_vector[index],
            "score_decomposition": self.score_decomposition[index],
        }
        if self.relation_tail is not None:
            example["relation_tail"] = self.relation_tail[index]
        return example

    def close(self) -> None:
        self._npz.close()


def _string_array(value: str):  # type: ignore[no-untyped-def]
    import numpy as np

    return np.array(value, dtype=np.str_)


def _retained_action_indices(  # type: ignore[no-untyped-def]
    target_q,
    q_valid,
    selected_action_index: int,
    top_k: int,
    filter_mode: str = "top-q-with-selected",
    priors=None,
    greedy_prefix_k: int | None = None,
):
    import numpy as np

    action_count = int(target_q.shape[0])
    if action_count <= 0:
        raise ValueError("cannot filter an expert root with no actions")
    if selected_action_index < 0 or selected_action_index >= action_count:
        raise ValueError(f"selected action index {selected_action_index} outside action_count={action_count}")
    if top_k <= 0:
        raise ValueError("--top-k must be positive")
    if action_count <= top_k:
        return np.arange(action_count, dtype=np.int64)
    if filter_mode == "greedy-prefix-strict":
        return np.arange(top_k, dtype=np.int64)
    if filter_mode == "greedy-prefix-with-selected":
        keep = list(range(min(top_k, action_count)))
        if selected_action_index not in keep:
            keep.append(int(selected_action_index))
        return np.asarray(sorted(keep), dtype=np.int64)
    if filter_mode == "greedy-prefix-plus-prior-with-selected":
        if priors is None:
            raise ValueError("greedy-prefix-plus-prior-with-selected requires priors")
        greedy_count = top_k // 2 if greedy_prefix_k is None else int(greedy_prefix_k)
        if greedy_count < 0:
            raise ValueError("--greedy-prefix-k must be >= 0")
        keep = list(range(min(greedy_count, action_count)))
        seen = set(keep)
        if selected_action_index not in seen:
            keep.append(int(selected_action_index))
            seen.add(int(selected_action_index))
        prior_scores = np.asarray(priors, dtype=np.float32)
        if prior_scores.shape[0] != action_count:
            raise ValueError("priors length must match action_count")
        ranked_by_prior = np.argsort(-prior_scores, kind="stable")
        for index in ranked_by_prior:
            action_index = int(index)
            if action_index in seen:
                continue
            keep.append(action_index)
            seen.add(action_index)
            if len(keep) >= top_k:
                break
        if len(keep) < top_k:
            for action_index in range(action_count):
                if action_index in seen:
                    continue
                keep.append(action_index)
                seen.add(action_index)
                if len(keep) >= top_k:
                    break
        return np.asarray(sorted(keep), dtype=np.int64)
    if filter_mode != "top-q-with-selected":
        raise ValueError(f"unsupported expert tensor filter mode {filter_mode!r}")

    valid = q_valid.astype(bool, copy=False)
    scores = np.where(valid, target_q, -np.inf)
    # Stable descending sort: if rollout labels tie, preserve original legal-action order.
    ranked = np.argsort(-scores, kind="stable")
    keep = [int(selected_action_index)]
    seen = {int(selected_action_index)}
    for index in ranked:
        action_index = int(index)
        if action_index in seen:
            continue
        keep.append(action_index)
        seen.add(action_index)
        if len(keep) >= top_k:
            break
    if len(keep) < top_k:
        for action_index in range(action_count):
            if action_index in seen:
                continue
            keep.append(action_index)
            seen.add(action_index)
            if len(keep) >= top_k:
                break
    return np.asarray(sorted(keep), dtype=np.int64)


def _remap_relation_edges_for_actions(  # type: ignore[no-untyped-def]
    edges,
    *,
    token_count: int,
    action_keep_map,
):
    import numpy as np

    if edges.size == 0:
        return edges.copy()
    remapped = edges.copy()
    keep = np.ones((edges.shape[0],), dtype=bool)
    action_count = int(action_keep_map.shape[0])
    for column in (0, 1):
        positions = edges[:, column]
        action_edge_rows = np.flatnonzero(positions >= token_count)
        if action_edge_rows.size == 0:
            continue
        old_actions = positions[action_edge_rows].astype(np.int64, copy=False) - token_count
        valid_old = (old_actions >= 0) & (old_actions < action_count)
        mapped = np.full((action_edge_rows.shape[0],), -1, dtype=np.int64)
        mapped[valid_old] = action_keep_map[old_actions[valid_old]]
        valid_new = mapped >= 0
        keep[action_edge_rows[~valid_new]] = False
        remapped[action_edge_rows[valid_new], column] = token_count + mapped[valid_new]
    return remapped[keep]


def _save_expert_tensor_shard(  # type: ignore[no-untyped-def]
    *,
    out_path: Path,
    metadata: dict[str, Any],
    tokens,
    actions,
    token_offsets,
    action_offsets,
    relation_edges,
    relation_offsets,
    selected_action_index,
    target_q,
    target_score_to_go,
    q_valid,
    priors,
    visits,
    q_variance,
    q_count,
    truncated_count,
    exact_afterstate_score_active,
    final_score_vector,
    rank_vector,
    score_decomposition,
    relation_tail=None,
) -> None:
    import numpy as np

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f"{out_path.name}.tmp")
    arrays = {
        "version": _string_array(SHARD_VERSION),
        "metadata_json": _string_array(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
        "tokens": tokens,
        "actions": actions,
        "token_offsets": token_offsets,
        "action_offsets": action_offsets,
        "relation_edges": relation_edges,
        "relation_offsets": relation_offsets,
        "selected_action_index": selected_action_index,
        "target_q": target_q,
        "target_score_to_go": target_score_to_go,
        "q_valid": q_valid,
        "priors": priors,
        "visits": visits,
        "q_variance": q_variance,
        "q_count": q_count,
        "truncated_count": truncated_count,
        "exact_afterstate_score_active": exact_afterstate_score_active,
        "final_score_vector": final_score_vector,
        "rank_vector": rank_vector,
        "score_decomposition": score_decomposition,
    }
    if relation_tail is not None:
        arrays["relation_tail"] = relation_tail
    with tmp_path.open("wb") as handle:
        np.savez(handle, **arrays)
    tmp_path.replace(out_path)


def _materialized_relation_tail(  # type: ignore[no-untyped-def]
    shard: ExpertTensorShard,
    *,
    token_capacity: int,
    action_capacity: int,
):
    import numpy as np

    record_count = len(shard)
    seq_capacity = token_capacity + action_capacity
    relation_tail = np.zeros((record_count, action_capacity, seq_capacity), dtype=np.uint8)
    token_counts = shard.token_offsets[1:] - shard.token_offsets[:-1]
    action_counts = shard.action_offsets[1:] - shard.action_offsets[:-1]
    for index in range(record_count):
        token_count = int(token_counts[index])
        action_count = int(action_counts[index])
        relation_start = int(shard.relation_offsets[index])
        relation_end = int(shard.relation_offsets[index + 1])
        edges = shard.relation_edges[relation_start:relation_end]
        if edges.size == 0:
            continue
        sources = edges[:, 0].astype(np.int64, copy=False)
        targets = edges[:, 1].astype(np.int64, copy=False)
        relation_ids = edges[:, 2].astype(np.int64, copy=False)
        source_is_action = sources >= token_count
        if not source_is_action.any():
            continue
        rows = sources[source_is_action] - token_count
        targets = targets[source_is_action]
        relation_ids = relation_ids[source_is_action]
        target_is_action = targets >= token_count
        cols = targets.copy()
        cols[target_is_action] = token_capacity + (targets[target_is_action] - token_count)
        valid = (
            (rows >= 0)
            & (rows < action_count)
            & (rows < action_capacity)
            & (cols >= 0)
            & (cols < seq_capacity)
            & (relation_ids >= 0)
            & (relation_ids < 256)
        )
        if valid.any():
            relation_tail[index, rows[valid], cols[valid]] = relation_ids[valid].astype(np.uint8, copy=False)
    return relation_tail


def filter_expert_tensor_shard(
    in_path: Path,
    out_path: Path,
    *,
    top_k: int,
    filter_mode: str = "top-q-with-selected",
    greedy_prefix_k: int | None = None,
    report_path: Path | None = None,
) -> dict[str, Any]:
    import numpy as np

    shard = ExpertTensorShard(in_path)
    try:
        record_count = len(shard)
        if record_count <= 0:
            raise ValueError("input expert tensor shard is empty")

        action_chunks = []
        target_q_chunks = []
        target_score_to_go_chunks = []
        q_valid_chunks = []
        priors_chunks = []
        visits_chunks = []
        q_variance_chunks = []
        q_count_chunks = []
        truncated_count_chunks = []
        exact_afterstate_score_active_chunks = []
        relation_chunks = []
        action_offsets = [0]
        relation_offsets = [0]
        selected = np.empty((record_count,), dtype=np.int16)
        selected_action_dropped_count = 0

        original_action_counts = shard.action_offsets[1:] - shard.action_offsets[:-1]
        original_relation_counts = shard.relation_offsets[1:] - shard.relation_offsets[:-1]

        for index in range(record_count):
            token_start = int(shard.token_offsets[index])
            token_end = int(shard.token_offsets[index + 1])
            token_count = token_end - token_start
            action_start = int(shard.action_offsets[index])
            action_end = int(shard.action_offsets[index + 1])
            relation_start = int(shard.relation_offsets[index])
            relation_end = int(shard.relation_offsets[index + 1])
            action_count = action_end - action_start
            selected_old = int(shard.selected_action_index[index])
            keep_local = _retained_action_indices(
                shard.target_q[action_start:action_end],
                shard.q_valid[action_start:action_end],
                selected_old,
                top_k,
                filter_mode=filter_mode,
                priors=shard.priors[action_start:action_end],
                greedy_prefix_k=greedy_prefix_k,
            )
            keep_global = action_start + keep_local
            action_keep_map = np.full((action_count,), -1, dtype=np.int64)
            action_keep_map[keep_local] = np.arange(keep_local.shape[0], dtype=np.int64)
            selected_new = int(action_keep_map[selected_old])
            if selected_new < 0:
                if filter_mode == "greedy-prefix-strict":
                    selected_action_dropped_count += 1
                    selected_new = 0
                else:
                    raise ValueError(f"filtered root {index} dropped the selected action")
            selected[index] = selected_new

            action_chunks.append(shard.actions[keep_global])
            target_q_chunks.append(shard.target_q[keep_global])
            target_score_to_go_chunks.append(shard.target_score_to_go[keep_global])
            q_valid_chunks.append(shard.q_valid[keep_global])
            priors_chunks.append(shard.priors[keep_global])
            visits_chunks.append(shard.visits[keep_global])
            q_variance_chunks.append(shard.q_variance[keep_global])
            q_count_chunks.append(shard.q_count[keep_global])
            truncated_count_chunks.append(shard.truncated_count[keep_global])
            exact_afterstate_score_active_chunks.append(shard.exact_afterstate_score_active[keep_global])

            remapped_edges = _remap_relation_edges_for_actions(
                shard.relation_edges[relation_start:relation_end],
                token_count=token_count,
                action_keep_map=action_keep_map,
            )
            relation_chunks.append(remapped_edges)
            action_offsets.append(action_offsets[-1] + int(keep_local.shape[0]))
            relation_offsets.append(relation_offsets[-1] + int(remapped_edges.shape[0]))

        actions = np.concatenate(action_chunks, axis=0)
        relation_edges = np.concatenate(relation_chunks, axis=0)
        metadata = dict(shard.metadata)
        metadata["filter"] = {
            "kind": filter_mode,
            "source_path": str(in_path),
            "source_sha256": _sha256(in_path),
            "top_k": int(top_k),
            "greedy_prefix_k": greedy_prefix_k,
            "original_record_count": record_count,
            "original_total_action_count": int(shard.actions.shape[0]),
            "original_max_action_count": int(original_action_counts.max(initial=0)),
            "original_total_relation_edge_count": int(shard.relation_edges.shape[0]),
            "original_max_relation_edge_count": int(original_relation_counts.max(initial=0)),
            "selected_action_dropped_count": selected_action_dropped_count,
            "strict_drop_replacement_index": 0 if filter_mode == "greedy-prefix-strict" else None,
        }
        metadata["record_count"] = record_count
        metadata["total_action_count"] = int(actions.shape[0])
        metadata["total_relation_edge_count"] = int(relation_edges.shape[0])
        metadata["max_action_count"] = _max_int(np.diff(np.asarray(action_offsets, dtype=np.int64)))
        metadata["max_relation_edge_count"] = _max_int(np.diff(np.asarray(relation_offsets, dtype=np.int64)))

        _save_expert_tensor_shard(
            out_path=out_path,
            metadata=metadata,
            tokens=shard.tokens,
            actions=actions,
            token_offsets=shard.token_offsets,
            action_offsets=np.asarray(action_offsets, dtype=np.int64),
            relation_edges=relation_edges,
            relation_offsets=np.asarray(relation_offsets, dtype=np.int64),
            selected_action_index=selected,
            target_q=np.concatenate(target_q_chunks, axis=0),
            target_score_to_go=np.concatenate(target_score_to_go_chunks, axis=0),
            q_valid=np.concatenate(q_valid_chunks, axis=0),
            priors=np.concatenate(priors_chunks, axis=0),
            visits=np.concatenate(visits_chunks, axis=0),
            q_variance=np.concatenate(q_variance_chunks, axis=0),
            q_count=np.concatenate(q_count_chunks, axis=0),
            truncated_count=np.concatenate(truncated_count_chunks, axis=0),
            exact_afterstate_score_active=np.concatenate(exact_afterstate_score_active_chunks, axis=0),
            final_score_vector=shard.final_score_vector,
            rank_vector=shard.rank_vector,
            score_decomposition=shard.score_decomposition,
        )
    finally:
        shard.close()

    summary = summarize_expert_tensor_shard(out_path).to_dict()
    summary["status"] = "pass"
    summary["filter"] = metadata["filter"]
    if filter_mode != "greedy-prefix-with-selected" and summary["max_action_count"] > top_k:
        raise ValueError(f"filtered shard max_action_count={summary['max_action_count']} exceeds top_k={top_k}")
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def materialize_relation_tail_shard(
    in_path: Path,
    out_path: Path,
    *,
    report_path: Path | None = None,
) -> dict[str, Any]:
    import numpy as np

    shard = ExpertTensorShard(in_path)
    try:
        record_count = len(shard)
        if record_count <= 0:
            raise ValueError("input expert tensor shard is empty")
        token_counts = shard.token_offsets[1:] - shard.token_offsets[:-1]
        action_counts = shard.action_offsets[1:] - shard.action_offsets[:-1]
        token_capacity = int(token_counts.max(initial=0))
        action_capacity = int(action_counts.max(initial=0))
        if token_capacity <= 0 or action_capacity <= 0:
            raise ValueError("cannot materialize relation tail for empty token/action capacity")
        relation_tail = _materialized_relation_tail(
            shard,
            token_capacity=token_capacity,
            action_capacity=action_capacity,
        )
        metadata = dict(shard.metadata)
        metadata["relation_tail"] = {
            "kind": "action_rows_fixed_capacity",
            "source_path": str(in_path),
            "source_sha256": _sha256(in_path),
            "dtype": "uint8",
            "token_capacity": token_capacity,
            "action_capacity": action_capacity,
            "seq_capacity": token_capacity + action_capacity,
            "record_count": record_count,
            "semantics": (
                "relation_tail[record, action, column] stores relation ids for action-row "
                "queries; columns are token-capacity slots followed by action-capacity slots"
            ),
        }
        _save_expert_tensor_shard(
            out_path=out_path,
            metadata=metadata,
            tokens=shard.tokens,
            actions=shard.actions,
            token_offsets=shard.token_offsets,
            action_offsets=shard.action_offsets,
            relation_edges=shard.relation_edges,
            relation_offsets=shard.relation_offsets,
            selected_action_index=shard.selected_action_index,
            target_q=shard.target_q,
            target_score_to_go=shard.target_score_to_go,
            q_valid=shard.q_valid,
            priors=shard.priors,
            visits=shard.visits,
            q_variance=shard.q_variance,
            q_count=shard.q_count,
            truncated_count=shard.truncated_count,
            exact_afterstate_score_active=shard.exact_afterstate_score_active,
            final_score_vector=shard.final_score_vector,
            rank_vector=shard.rank_vector,
            score_decomposition=shard.score_decomposition,
            relation_tail=relation_tail,
        )
    finally:
        shard.close()

    summary = summarize_expert_tensor_shard(out_path).to_dict()
    summary["status"] = "pass"
    summary["relation_tail"] = metadata["relation_tail"]
    if not summary["relation_tail_present"]:
        raise ValueError("materialized shard did not contain relation_tail")
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


class ExpertTensorCorpus:
    def __init__(self, paths: list[Path]) -> None:
        if not paths:
            raise ValueError("ExpertTensorCorpus requires at least one shard")
        self.paths = paths
        self.shards = [ExpertTensorShard(path) for path in paths]
        self.cumulative: list[int] = []
        total = 0
        for shard in self.shards:
            total += len(shard)
            self.cumulative.append(total)
        if total <= 0:
            raise ValueError("expert tensor corpus is empty")

    def __len__(self) -> int:
        return self.cumulative[-1]

    def source_lengths(self) -> list[int]:
        return [len(shard) for shard in self.shards]

    def example(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = bisect_right(self.cumulative, index)
        previous = 0 if shard_index == 0 else self.cumulative[shard_index - 1]
        return self.shards[shard_index].example(index - previous)

    def examples(self, indices: list[int]) -> list[dict[str, Any]]:
        return [self.example(index) for index in indices]

    def close(self) -> None:
        for shard in self.shards:
            shard.close()


def collate_expert_tensor_examples(examples: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    if not examples:
        raise ValueError("collate_expert_tensor_examples requires at least one example")
    batch_size = len(examples)
    token_counts = [int(example["tokens"].shape[0]) for example in examples]
    action_counts = [int(example["actions"].shape[0]) for example in examples]
    token_dim = int(examples[0]["tokens"].shape[1])
    action_dim = int(examples[0]["actions"].shape[1])
    has_relation_tail = all(example.get("relation_tail") is not None for example in examples)
    if has_relation_tail:
        tail_shapes = [tuple(example["relation_tail"].shape) for example in examples]
        tail_action_capacities = [int(shape[0]) for shape in tail_shapes]
        tail_token_capacities = [int(shape[1] - shape[0]) for shape in tail_shapes]
        if any(token_capacity < 0 for token_capacity in tail_token_capacities):
            raise ValueError(f"invalid relation_tail shapes: {tail_shapes}")
        max_actions = max(tail_action_capacities)
        max_tokens = max(tail_token_capacities)
        if max_tokens < max(token_counts):
            raise ValueError("relation_tail token capacity is smaller than a batch token count")
        if max_actions < max(action_counts):
            raise ValueError("relation_tail action capacity is smaller than a batch action count")
    else:
        max_tokens = max(token_counts)
        max_actions = max(action_counts)
    seq_len = max_tokens + max_actions
    tokens = torch.zeros((batch_size, max_tokens, token_dim), dtype=torch.float32)
    token_mask = torch.zeros((batch_size, max_tokens), dtype=torch.bool)
    actions = torch.zeros((batch_size, max_actions, action_dim), dtype=torch.float32)
    action_mask = torch.zeros((batch_size, max_actions), dtype=torch.bool)
    relation_ids = None if has_relation_tail else torch.zeros((batch_size, seq_len, seq_len), dtype=torch.long)
    relation_tail = torch.zeros((batch_size, max_actions, seq_len), dtype=torch.uint8) if has_relation_tail else None
    target_q = torch.zeros((batch_size, max_actions), dtype=torch.float32)
    target_score_to_go = torch.zeros((batch_size, max_actions), dtype=torch.float32)
    q_valid = torch.zeros((batch_size, max_actions), dtype=torch.bool)
    visits = torch.zeros((batch_size, max_actions), dtype=torch.float32)
    q_variance = torch.zeros((batch_size, max_actions), dtype=torch.float32)
    q_count = torch.ones((batch_size, max_actions), dtype=torch.float32)
    truncated_count = torch.zeros((batch_size, max_actions), dtype=torch.float32)
    exact_afterstate = torch.zeros((batch_size, max_actions), dtype=torch.float32)
    selected = torch.zeros((batch_size,), dtype=torch.long)
    greedy_selected = torch.zeros((batch_size,), dtype=torch.long)
    target_value = torch.zeros((batch_size, 4), dtype=torch.float32)
    target_rank = torch.zeros((batch_size, 4), dtype=torch.long)
    target_score = torch.zeros((batch_size, 3, 4), dtype=torch.float32)

    for batch_index, example in enumerate(examples):
        token_count = token_counts[batch_index]
        action_count = action_counts[batch_index]
        tokens[batch_index, :token_count] = torch.as_tensor(example["tokens"], dtype=torch.float32)
        token_mask[batch_index, :token_count] = True
        actions[batch_index, :action_count] = torch.as_tensor(example["actions"], dtype=torch.float32)
        action_mask[batch_index, :action_count] = True
        target_q[batch_index, :action_count] = torch.as_tensor(example["target_q"], dtype=torch.float32)
        target_score_to_go[batch_index, :action_count] = torch.as_tensor(
            example["target_score_to_go"],
            dtype=torch.float32,
        )
        q_valid[batch_index, :action_count] = torch.as_tensor(example["q_valid"], dtype=torch.bool)
        visits[batch_index, :action_count] = torch.as_tensor(example["visits"], dtype=torch.float32)
        q_variance[batch_index, :action_count] = torch.as_tensor(example["q_variance"], dtype=torch.float32)
        q_count[batch_index, :action_count] = torch.as_tensor(example["q_count"], dtype=torch.float32)
        truncated_count[batch_index, :action_count] = torch.as_tensor(example["truncated_count"], dtype=torch.float32)
        exact_afterstate[batch_index, :action_count] = torch.as_tensor(
            example["exact_afterstate_score_active"],
            dtype=torch.float32,
        )
        selected[batch_index] = int(example["selected_action_index"])
        target_value[batch_index] = torch.as_tensor(example["final_score_vector"], dtype=torch.float32)
        target_rank[batch_index] = torch.as_tensor(example["rank_vector"], dtype=torch.long) - 1
        target_score[batch_index] = torch.as_tensor(example["score_decomposition"], dtype=torch.float32)
        if relation_tail is not None:
            tail = torch.as_tensor(example["relation_tail"], dtype=torch.uint8)
            tail_action_capacity = int(tail.shape[0])
            tail_token_capacity = int(tail.shape[1] - tail_action_capacity)
            relation_tail[batch_index, :tail_action_capacity, :tail_token_capacity] = tail[
                :,
                :tail_token_capacity,
            ]
            relation_tail[
                batch_index,
                :tail_action_capacity,
                max_tokens : max_tokens + tail_action_capacity,
            ] = tail[:, tail_token_capacity : tail_token_capacity + tail_action_capacity]
        else:
            assert relation_ids is not None
            for source, target, relation_id in example["relation_edges"]:
                source = int(source)
                target = int(target)
                if source >= token_count:
                    source = max_tokens + (source - token_count)
                if target >= token_count:
                    target = max_tokens + (target - token_count)
                if 0 <= source < seq_len and 0 <= target < seq_len:
                    relation_ids[batch_index, source, target] = int(relation_id)

    batch = {
        "tokens": tokens,
        "token_mask": token_mask,
        "actions": actions,
        "action_mask": action_mask,
        "combined_seq_len": seq_len,
        "target_q": target_q,
        "q_valid": q_valid,
        "target_score_to_go": target_score_to_go,
        "visits": visits,
        "target_q_variance": q_variance,
        "target_q_count": q_count,
        "target_truncated_count": truncated_count,
        "exact_afterstate_score_active": exact_afterstate,
        "selected_action_index": selected,
        "greedy_action_index": greedy_selected,
        "target_value": target_value,
        "target_rank": target_rank,
        "target_score": target_score,
        "schema_ids": [SHARD_VERSION] * batch_size,
        "state_hashes": ["packed-expert-tensor"] * batch_size,
        "token_counts": token_counts,
        "action_counts": action_counts,
    }
    if relation_tail is not None:
        batch["relation_tail"] = relation_tail
    else:
        batch["relation_ids"] = relation_ids
    return batch


def summarize_expert_tensor_shard(path: Path) -> ExpertTensorSummary:
    shard = ExpertTensorShard(path)
    try:
        action_counts = shard.action_offsets[1:] - shard.action_offsets[:-1]
        token_counts = shard.token_offsets[1:] - shard.token_offsets[:-1]
        relation_counts = shard.relation_offsets[1:] - shard.relation_offsets[:-1]
        relation_tail_present = shard.relation_tail is not None
        relation_tail_meta = shard.metadata.get("relation_tail", {}) if relation_tail_present else {}
        return ExpertTensorSummary(
            path=str(path),
            version=shard.version,
            record_count=len(shard),
            total_token_count=int(shard.tokens.shape[0]),
            total_action_count=int(shard.actions.shape[0]),
            total_relation_edge_count=int(shard.relation_edges.shape[0]),
            token_feature_dim=int(shard.tokens.shape[1]),
            action_feature_dim=int(shard.actions.shape[1]),
            max_token_count=int(token_counts.max(initial=0)),
            max_action_count=int(action_counts.max(initial=0)),
            max_relation_edge_count=int(relation_counts.max(initial=0)),
            output_bytes=path.stat().st_size,
            output_sha256=_sha256(path),
            relation_tail_present=relation_tail_present,
            relation_tail_shape=list(shard.relation_tail.shape) if shard.relation_tail is not None else None,
            relation_tail_dtype=str(shard.relation_tail.dtype) if shard.relation_tail is not None else None,
            relation_tail_token_capacity=(
                int(relation_tail_meta.get("token_capacity"))
                if relation_tail_meta.get("token_capacity") is not None
                else None
            ),
            relation_tail_action_capacity=(
                int(relation_tail_meta.get("action_capacity"))
                if relation_tail_meta.get("action_capacity") is not None
                else None
            ),
        )
    finally:
        shard.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summarize-shard")
    parser.add_argument("--filter-shard")
    parser.add_argument("--materialize-relation-tail")
    parser.add_argument("--out")
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument(
        "--filter-mode",
        choices=[
            "top-q-with-selected",
            "greedy-prefix-with-selected",
            "greedy-prefix-strict",
            "greedy-prefix-plus-prior-with-selected",
        ],
        default="top-q-with-selected",
    )
    parser.add_argument(
        "--greedy-prefix-k",
        type=int,
        default=None,
        help="Greedy prefix size for greedy-prefix-plus-prior-with-selected; defaults to top_k // 2",
    )
    parser.add_argument("--report")
    args = parser.parse_args()
    if args.filter_shard:
        if not args.out:
            parser.error("--filter-shard requires --out")
        summary = filter_expert_tensor_shard(
            Path(args.filter_shard),
            Path(args.out),
            top_k=args.top_k,
            filter_mode=args.filter_mode,
            greedy_prefix_k=args.greedy_prefix_k,
            report_path=Path(args.report) if args.report else None,
        )
    elif args.materialize_relation_tail:
        if not args.out:
            parser.error("--materialize-relation-tail requires --out")
        summary = materialize_relation_tail_shard(
            Path(args.materialize_relation_tail),
            Path(args.out),
            report_path=Path(args.report) if args.report else None,
        )
    elif args.summarize_shard:
        summary = summarize_expert_tensor_shard(Path(args.summarize_shard)).to_dict()
        summary["status"] = "pass"
        if args.report:
            Path(args.report).parent.mkdir(parents=True, exist_ok=True)
            Path(args.report).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        parser.error("one of --summarize-shard, --filter-shard, or --materialize-relation-tail is required")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
