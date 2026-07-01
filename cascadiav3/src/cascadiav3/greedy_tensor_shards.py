"""Compact tensor shards for greedy behavior-cloning corpora.

JSONL remains the audit/debug format for simulator exports. This module turns
bounded JSONL shards into the numeric arrays consumed by the greedy policy
pretraining model so large runs do not pay the long-term storage and parse cost
of raw simulator records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .torch_public_token_merit import PUBLIC_TOKEN_FEATURE_DIM, public_token_features
from .torch_semantic_relation_bias_merit import (
    SEMANTIC_PUBLIC_TOKEN_ACTION_FEATURE_DIM,
    semantic_public_token_action_features,
)

SHARD_VERSION = "greedy_policy_tensor_shard_v1"
DEFAULT_SOURCE = "greedy_policy_no_search_corpus"
SUPPORTED_DTYPES = {"float16", "float32"}


@dataclass(frozen=True)
class TensorShardSummary:
    path: str
    version: str
    dtype: str
    record_count: int
    total_token_count: int
    total_action_count: int
    token_feature_dim: int
    action_feature_dim: int
    max_token_count: int
    max_action_count: int
    output_bytes: int
    output_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "version": self.version,
            "dtype": self.dtype,
            "record_count": self.record_count,
            "total_token_count": self.total_token_count,
            "total_action_count": self.total_action_count,
            "token_feature_dim": self.token_feature_dim,
            "action_feature_dim": self.action_feature_dim,
            "max_token_count": self.max_token_count,
            "max_action_count": self.max_action_count,
            "output_bytes": self.output_bytes,
            "output_sha256": self.output_sha256,
            "bytes_per_record": self.output_bytes / max(1, self.record_count),
        }


def parse_paths(raw: str) -> list[Path]:
    paths = [Path(part.strip()) for part in raw.split(",") if part.strip()]
    if not paths:
        raise ValueError("at least one input path is required")
    return paths


def iter_jsonl_records(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        if str(path) == "-":
            for line in sys.stdin:
                stripped = line.strip()
                if stripped:
                    yield json.loads(stripped)
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    yield json.loads(stripped)


def selected_action_index(record: dict[str, Any]) -> int:
    selected = record["selected_action"]
    action_ids = [action["action_id"] for action in record["legal_actions"]]
    try:
        return action_ids.index(selected)
    except ValueError as exc:
        raise ValueError(f"selected action missing from legal actions for {record['state_hash']}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _string_array(value: str):  # type: ignore[no-untyped-def]
    import numpy as np

    return np.array(value, dtype=np.str_)


def _scalar_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _dtype_for(name: str):  # type: ignore[no-untyped-def]
    import numpy as np

    if name == "float16":
        return np.float16
    if name == "float32":
        return np.float32
    raise ValueError(f"unsupported dtype {name!r}; expected one of {sorted(SUPPORTED_DTYPES)}")


def _record_source(record: dict[str, Any]) -> str | None:
    return record.get("metadata", {}).get("source")


def _first_pass(
    paths: list[Path],
    *,
    require_source: str,
    max_records: int | None,
) -> dict[str, Any]:
    record_count = 0
    total_token_count = 0
    total_action_count = 0
    max_token_count = 0
    max_action_count = 0
    first_state_hash = None
    last_state_hash = None

    for record in iter_jsonl_records(paths):
        if require_source and _record_source(record) != require_source:
            continue
        token_count = int(record["public_tokens"]["token_count"])
        action_count = len(record["legal_actions"])
        if action_count <= 0:
            raise ValueError(f"record {record.get('state_hash')} has no legal actions")
        selected = selected_action_index(record)
        if selected < 0 or selected >= action_count:
            raise ValueError(f"selected action index out of range for {record.get('state_hash')}")
        if first_state_hash is None:
            first_state_hash = record.get("state_hash")
        last_state_hash = record.get("state_hash")
        record_count += 1
        total_token_count += token_count
        total_action_count += action_count
        max_token_count = max(max_token_count, token_count)
        max_action_count = max(max_action_count, action_count)
        if max_records is not None and record_count >= max_records:
            break

    if record_count == 0:
        raise ValueError("no greedy policy records matched the requested source")
    return {
        "record_count": record_count,
        "total_token_count": total_token_count,
        "total_action_count": total_action_count,
        "max_token_count": max_token_count,
        "max_action_count": max_action_count,
        "first_state_hash": first_state_hash,
        "last_state_hash": last_state_hash,
    }


def _metadata(
    *,
    jsonl_paths: list[Path],
    dtype_name: str,
    require_source: str,
    record_count: int,
    total_token_count: int,
    total_action_count: int,
    max_token_count: int,
    max_action_count: int,
    first_state_hash: str | None,
    last_state_hash: str | None,
) -> dict[str, Any]:
    return {
        "version": SHARD_VERSION,
        "source": require_source,
        "source_paths": [str(path) for path in jsonl_paths],
        "format": "npz",
        "dtype": dtype_name,
        "record_count": record_count,
        "total_token_count": total_token_count,
        "total_action_count": total_action_count,
        "token_feature_dim": PUBLIC_TOKEN_FEATURE_DIM,
        "action_feature_dim": SEMANTIC_PUBLIC_TOKEN_ACTION_FEATURE_DIM,
        "max_token_count": max_token_count,
        "max_action_count": max_action_count,
        "first_state_hash": first_state_hash,
        "last_state_hash": last_state_hash,
        "canonical_model_inputs": [
            "public_token_features",
            "semantic_public_token_action_features",
            "selected_action_index",
        ],
        "omitted_by_design": [
            "raw_legal_action_json",
            "state_hashes",
            "action_ids",
            "relations",
            "per_action_Q",
            "score_decompositions",
        ],
    }


def _save_npz(
    out_path: Path,
    *,
    metadata: dict[str, Any],
    tokens: Any,
    actions: Any,
    token_offsets: Any,
    action_offsets: Any,
    selected_action_indices: Any,
) -> None:
    import numpy as np

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            version=_string_array(SHARD_VERSION),
            metadata_json=_string_array(json.dumps(metadata, sort_keys=True)),
            tokens=tokens,
            actions=actions,
            token_offsets=token_offsets,
            action_offsets=action_offsets,
            selected_action_index=selected_action_indices,
        )


def _summary_with_io(
    out_path: Path,
    *,
    started: float,
    jsonl_paths: list[Path],
    input_bytes: int | None,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    summary = summarize_tensor_shard(out_path).to_dict()
    elapsed = time.perf_counter() - started
    record_count = int(summary["record_count"])
    summary.update(
        {
            "status": "pass",
            "source_paths": [str(path) for path in jsonl_paths],
            "input_jsonl_bytes": input_bytes,
            "input_jsonl_bytes_per_record": (input_bytes / max(1, record_count)) if input_bytes is not None else None,
            "output_to_input_ratio": (summary["output_bytes"] / max(1, input_bytes)) if input_bytes else None,
            "conversion_seconds": elapsed,
            "records_per_second": record_count / max(elapsed, 1.0e-9),
            "metadata": metadata,
        }
    )
    return summary


def _write_tensor_shard_streaming(
    jsonl_paths: list[Path],
    out_path: Path,
    *,
    dtype_name: str,
    require_source: str,
    max_records: int | None,
) -> dict[str, Any]:
    import numpy as np

    dtype = _dtype_for(dtype_name)
    started = time.perf_counter()
    token_chunks = []
    action_chunks = []
    token_offsets = [0]
    action_offsets = [0]
    selected_action_indices = []
    max_token_count = 0
    max_action_count = 0
    first_state_hash = None
    last_state_hash = None

    for record in iter_jsonl_records(jsonl_paths):
        if require_source and _record_source(record) != require_source:
            continue
        token_rows = np.asarray(public_token_features(record), dtype=dtype)
        action_rows = np.asarray(semantic_public_token_action_features(record), dtype=dtype)
        if token_rows.shape[1] != PUBLIC_TOKEN_FEATURE_DIM:
            raise ValueError(f"token feature dimension mismatch: {token_rows.shape}")
        if action_rows.shape[1] != SEMANTIC_PUBLIC_TOKEN_ACTION_FEATURE_DIM:
            raise ValueError(f"action feature dimension mismatch: {action_rows.shape}")
        if first_state_hash is None:
            first_state_hash = record.get("state_hash")
        last_state_hash = record.get("state_hash")
        token_chunks.append(token_rows)
        action_chunks.append(action_rows)
        token_offsets.append(token_offsets[-1] + int(token_rows.shape[0]))
        action_offsets.append(action_offsets[-1] + int(action_rows.shape[0]))
        selected_action_indices.append(selected_action_index(record))
        max_token_count = max(max_token_count, int(token_rows.shape[0]))
        max_action_count = max(max_action_count, int(action_rows.shape[0]))
        if max_records is not None and len(selected_action_indices) >= max_records:
            break

    record_count = len(selected_action_indices)
    if record_count == 0:
        raise ValueError("no greedy policy records matched the requested source")
    tokens = np.concatenate(token_chunks, axis=0)
    actions = np.concatenate(action_chunks, axis=0)
    token_offsets_np = np.asarray(token_offsets, dtype=np.int64)
    action_offsets_np = np.asarray(action_offsets, dtype=np.int64)
    selected_np = np.asarray(selected_action_indices, dtype=np.int16)
    metadata = _metadata(
        jsonl_paths=jsonl_paths,
        dtype_name=dtype_name,
        require_source=require_source,
        record_count=record_count,
        total_token_count=int(tokens.shape[0]),
        total_action_count=int(actions.shape[0]),
        max_token_count=max_token_count,
        max_action_count=max_action_count,
        first_state_hash=first_state_hash,
        last_state_hash=last_state_hash,
    )
    _save_npz(
        out_path,
        metadata=metadata,
        tokens=tokens,
        actions=actions,
        token_offsets=token_offsets_np,
        action_offsets=action_offsets_np,
        selected_action_indices=selected_np,
    )
    input_bytes = None
    if all(str(path) != "-" and path.exists() for path in jsonl_paths):
        input_bytes = sum(path.stat().st_size for path in jsonl_paths)
    return _summary_with_io(
        out_path,
        started=started,
        jsonl_paths=jsonl_paths,
        input_bytes=input_bytes,
        metadata=metadata,
    )


def write_tensor_shard_from_jsonl(
    jsonl_paths: list[Path],
    out_path: Path,
    *,
    dtype_name: str = "float16",
    require_source: str = DEFAULT_SOURCE,
    max_records: int | None = None,
) -> dict[str, Any]:
    import numpy as np

    if any(str(path) == "-" for path in jsonl_paths):
        return _write_tensor_shard_streaming(
            jsonl_paths,
            out_path,
            dtype_name=dtype_name,
            require_source=require_source,
            max_records=max_records,
        )

    dtype = _dtype_for(dtype_name)
    started = time.perf_counter()
    stats = _first_pass(jsonl_paths, require_source=require_source, max_records=max_records)
    record_count = int(stats["record_count"])
    total_token_count = int(stats["total_token_count"])
    total_action_count = int(stats["total_action_count"])

    tokens = np.empty((total_token_count, PUBLIC_TOKEN_FEATURE_DIM), dtype=dtype)
    actions = np.empty((total_action_count, SEMANTIC_PUBLIC_TOKEN_ACTION_FEATURE_DIM), dtype=dtype)
    token_offsets = np.zeros(record_count + 1, dtype=np.int64)
    action_offsets = np.zeros(record_count + 1, dtype=np.int64)
    selected_action_indices = np.zeros(record_count, dtype=np.int16)

    record_index = 0
    token_cursor = 0
    action_cursor = 0
    for record in iter_jsonl_records(jsonl_paths):
        if require_source and _record_source(record) != require_source:
            continue
        token_rows = np.asarray(public_token_features(record), dtype=np.float32)
        action_rows = np.asarray(semantic_public_token_action_features(record), dtype=np.float32)
        if token_rows.shape[1] != PUBLIC_TOKEN_FEATURE_DIM:
            raise ValueError(f"token feature dimension mismatch: {token_rows.shape}")
        if action_rows.shape[1] != SEMANTIC_PUBLIC_TOKEN_ACTION_FEATURE_DIM:
            raise ValueError(f"action feature dimension mismatch: {action_rows.shape}")
        token_end = token_cursor + token_rows.shape[0]
        action_end = action_cursor + action_rows.shape[0]
        tokens[token_cursor:token_end] = token_rows.astype(dtype, copy=False)
        actions[action_cursor:action_end] = action_rows.astype(dtype, copy=False)
        token_offsets[record_index] = token_cursor
        action_offsets[record_index] = action_cursor
        selected_action_indices[record_index] = selected_action_index(record)
        record_index += 1
        token_cursor = token_end
        action_cursor = action_end
        if max_records is not None and record_index >= max_records:
            break

    token_offsets[record_count] = token_cursor
    action_offsets[record_count] = action_cursor
    if record_index != record_count or token_cursor != total_token_count or action_cursor != total_action_count:
        raise RuntimeError("tensor shard fill counts diverged from first pass")

    metadata = _metadata(
        jsonl_paths=jsonl_paths,
        dtype_name=dtype_name,
        require_source=require_source,
        record_count=record_count,
        total_token_count=total_token_count,
        total_action_count=total_action_count,
        max_token_count=int(stats["max_token_count"]),
        max_action_count=int(stats["max_action_count"]),
        first_state_hash=stats["first_state_hash"],
        last_state_hash=stats["last_state_hash"],
    )
    _save_npz(
        out_path,
        metadata=metadata,
        tokens=tokens,
        actions=actions,
        token_offsets=token_offsets,
        action_offsets=action_offsets,
        selected_action_indices=selected_action_indices,
    )
    input_bytes = sum(path.stat().st_size for path in jsonl_paths if path.exists())
    return _summary_with_io(
        out_path,
        started=started,
        jsonl_paths=jsonl_paths,
        input_bytes=input_bytes,
        metadata=metadata,
    )


def load_tensor_shard_arrays(path: Path) -> dict[str, Any]:
    import numpy as np

    with np.load(path, allow_pickle=False) as shard:
        version = _scalar_string(shard["version"].item())
        if version != SHARD_VERSION:
            raise ValueError(f"unsupported tensor shard version {version!r}")
        metadata = json.loads(_scalar_string(shard["metadata_json"].item()))
        return {
            "metadata": metadata,
            "tokens": shard["tokens"],
            "actions": shard["actions"],
            "token_offsets": shard["token_offsets"],
            "action_offsets": shard["action_offsets"],
            "selected_action_index": shard["selected_action_index"],
        }


def tensor_shard_record_count(path: Path) -> int:
    import numpy as np

    with np.load(path, allow_pickle=False) as shard:
        version = _scalar_string(shard["version"].item())
        if version != SHARD_VERSION:
            raise ValueError(f"unsupported tensor shard version {version!r}")
        return int(shard["selected_action_index"].shape[0])


def summarize_tensor_shard(path: Path) -> TensorShardSummary:
    import numpy as np

    with np.load(path, allow_pickle=False) as shard:
        version = _scalar_string(shard["version"].item())
        metadata = json.loads(_scalar_string(shard["metadata_json"].item()))
        tokens = shard["tokens"]
        actions = shard["actions"]
        token_offsets = shard["token_offsets"]
        action_offsets = shard["action_offsets"]
        selected = shard["selected_action_index"]
        if version != SHARD_VERSION:
            raise ValueError(f"unsupported tensor shard version {version!r}")
        if tokens.shape[1] != PUBLIC_TOKEN_FEATURE_DIM:
            raise ValueError(f"token feature dimension mismatch: {tokens.shape}")
        if actions.shape[1] != SEMANTIC_PUBLIC_TOKEN_ACTION_FEATURE_DIM:
            raise ValueError(f"action feature dimension mismatch: {actions.shape}")
        if token_offsets.shape[0] != selected.shape[0] + 1:
            raise ValueError("token offsets length does not match selected-action count")
        if action_offsets.shape[0] != selected.shape[0] + 1:
            raise ValueError("action offsets length does not match selected-action count")
        action_counts = action_offsets[1:] - action_offsets[:-1]
        if action_counts.size and bool((selected.astype(np.int64) >= action_counts).any()):
            raise ValueError("selected action index exceeds max action count")
        dtype = str(metadata.get("dtype", tokens.dtype.name))
        return TensorShardSummary(
            path=str(path),
            version=version,
            dtype=dtype,
            record_count=int(selected.shape[0]),
            total_token_count=int(tokens.shape[0]),
            total_action_count=int(actions.shape[0]),
            token_feature_dim=int(tokens.shape[1]),
            action_feature_dim=int(actions.shape[1]),
            max_token_count=int((token_offsets[1:] - token_offsets[:-1]).max(initial=0)),
            max_action_count=int(action_counts.max(initial=0)),
            output_bytes=path.stat().st_size,
            output_sha256=_sha256(path),
        )


def validate_tensor_shard_against_jsonl(
    jsonl_paths: list[Path],
    shard_path: Path,
    *,
    require_source: str = DEFAULT_SOURCE,
    max_records: int | None = None,
) -> dict[str, Any]:
    import numpy as np

    shard = load_tensor_shard_arrays(shard_path)
    tokens = shard["tokens"]
    actions = shard["actions"]
    token_offsets = shard["token_offsets"]
    action_offsets = shard["action_offsets"]
    selected = shard["selected_action_index"]
    dtype = tokens.dtype

    expected_tokens = []
    expected_actions = []
    expected_token_offsets = [0]
    expected_action_offsets = [0]
    expected_selected = []

    for record in iter_jsonl_records(jsonl_paths):
        if require_source and _record_source(record) != require_source:
            continue
        token_rows = np.asarray(public_token_features(record), dtype=dtype)
        action_rows = np.asarray(semantic_public_token_action_features(record), dtype=dtype)
        expected_tokens.append(token_rows)
        expected_actions.append(action_rows)
        expected_token_offsets.append(expected_token_offsets[-1] + int(token_rows.shape[0]))
        expected_action_offsets.append(expected_action_offsets[-1] + int(action_rows.shape[0]))
        expected_selected.append(selected_action_index(record))
        if max_records is not None and len(expected_selected) >= max_records:
            break

    if not expected_selected:
        raise ValueError("no greedy policy records matched the requested source")

    expected_tokens_np = np.concatenate(expected_tokens, axis=0)
    expected_actions_np = np.concatenate(expected_actions, axis=0)
    expected_token_offsets_np = np.asarray(expected_token_offsets, dtype=np.int64)
    expected_action_offsets_np = np.asarray(expected_action_offsets, dtype=np.int64)
    expected_selected_np = np.asarray(expected_selected, dtype=np.int16)

    def max_abs(left, right) -> float:  # type: ignore[no-untyped-def]
        if left.shape != right.shape:
            return float("inf")
        if left.size == 0:
            return 0.0
        return float(np.max(np.abs(left.astype(np.float32) - right.astype(np.float32))))

    report = {
        "status": "pass",
        "jsonl_paths": [str(path) for path in jsonl_paths],
        "shard_path": str(shard_path),
        "record_count": int(expected_selected_np.shape[0]),
        "token_shape": list(tokens.shape),
        "action_shape": list(actions.shape),
        "expected_token_shape": list(expected_tokens_np.shape),
        "expected_action_shape": list(expected_actions_np.shape),
        "token_offsets_match": bool(np.array_equal(token_offsets, expected_token_offsets_np)),
        "action_offsets_match": bool(np.array_equal(action_offsets, expected_action_offsets_np)),
        "selected_action_index_match": bool(np.array_equal(selected, expected_selected_np)),
        "tokens_match": bool(np.array_equal(tokens, expected_tokens_np)),
        "actions_match": bool(np.array_equal(actions, expected_actions_np)),
        "tokens_max_abs_diff": max_abs(tokens, expected_tokens_np),
        "actions_max_abs_diff": max_abs(actions, expected_actions_np),
        "metadata": shard["metadata"],
    }
    checks = [
        report["token_offsets_match"],
        report["action_offsets_match"],
        report["selected_action_index_match"],
        report["tokens_match"],
        report["actions_match"],
    ]
    if not all(checks):
        report["status"] = "fail"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", help="Comma-separated greedy policy JSONL path(s), or '-' for stdin")
    parser.add_argument("--out", help="Output .npz tensor shard path")
    parser.add_argument("--dtype", choices=sorted(SUPPORTED_DTYPES), default="float16")
    parser.add_argument("--require-source", default=DEFAULT_SOURCE)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--report", help="Optional JSON report path")
    parser.add_argument("--validate-shard", help="Validate an existing .npz shard against --jsonl")
    parser.add_argument("--summarize-shard", help="Summarize an existing .npz shard")
    args = parser.parse_args()

    if args.summarize_shard:
        summary = summarize_tensor_shard(Path(args.summarize_shard)).to_dict()
        summary["status"] = "pass"
    elif args.validate_shard:
        if not args.jsonl:
            raise ValueError("--validate-shard requires --jsonl")
        summary = validate_tensor_shard_against_jsonl(
            parse_paths(args.jsonl),
            Path(args.validate_shard),
            require_source=args.require_source,
            max_records=args.max_records,
        )
    else:
        if not args.jsonl or not args.out:
            raise ValueError("tensor shard creation requires --jsonl and --out")
        summary = write_tensor_shard_from_jsonl(
            parse_paths(args.jsonl),
            Path(args.out),
            dtype_name=args.dtype,
            require_source=args.require_source,
            max_records=args.max_records,
        )
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
