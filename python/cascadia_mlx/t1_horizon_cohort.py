"""Strict exact-R2 development cohort for the T1 search-horizon experiment."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import numpy as np

from cascadia_mlx.o1_ranking_cohort import (
    COHORT_WIDTH,
    O1RankingCohortCache,
    cohort_row_blake3,
    stable_score_ranking,
)
from cascadia_mlx.r3_action_edit_mlx_cache import (
    CONTROL_ARM,
    CONTROL_MATERIALIZATION_PREVERIFIED_VECTORIZED,
    R3ActionEditMlxCache,
    open_data_verification_id,
    open_data_verification_identity,
)
from cascadia_mlx.r3_action_edit_mlx_model import (
    R3ActionEditModelConfig,
    R3ActionEditRanker,
)
from cascadia_mlx.s1_exact_supply_mlx_cache import S1ExactSupplyCache

EXPERIMENT_ID = "t1-search-horizon-decomposition-v1"
PROTOCOL_ID = "t1-strict-train-top64-cohort-v1"
COHORT_SCHEMA = "t1-strict-exact-r2-top64-cohort-v1"
SCHEMA_VERSION = 1
CANDIDATE_CHUNK = 256

_DTYPES = {
    "|u1": np.dtype("u1"),
    "<u2": np.dtype("<u2"),
    "<u8": np.dtype("<u8"),
    "<f4": np.dtype("<f4"),
}
_TENSORS = {
    "group_ids": ("<u8", lambda groups: (groups,)),
    "game_indices": ("<u8", lambda groups: (groups,)),
    "turns": ("|u1", lambda groups: (groups,)),
    "current_players": ("|u1", lambda groups: (groups,)),
    "candidate_positions": ("<u2", lambda groups: (groups, COHORT_WIDTH)),
    "source_candidate_indices": ("<u2", lambda groups: (groups, COHORT_WIDTH)),
    "base_ranks": ("<u2", lambda groups: (groups, COHORT_WIDTH)),
    "base_scores": ("<f4", lambda groups: (groups, COHORT_WIDTH)),
    "action_hashes": ("|u1", lambda groups: (groups, COHORT_WIDTH, 32)),
    "direct_cohort_indices": ("<u2", lambda groups: (groups,)),
    "rescored_flags": ("|u1", lambda groups: (groups,)),
    "cohort_hashes": ("|u1", lambda groups: (groups, 32)),
}


class T1HorizonCohortError(ValueError):
    """The strict T1 cohort is invalid or cannot be reproduced."""


def is_strict_top64(ranks: np.ndarray) -> bool:
    """Return whether one row contains each zero-based rank from 0 through 63."""
    values = np.asarray(ranks, dtype=np.int64)
    return values.shape == (COHORT_WIDTH,) and np.array_equal(
        np.sort(values),
        np.arange(COHORT_WIDTH, dtype=np.int64),
    )


def strict_top64_positions(
    scores: np.ndarray,
    action_hashes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return source-ordered strict top-64 positions and their model ranks."""
    ranking = stable_score_ranking(scores, action_hashes)
    if len(ranking) < COHORT_WIDTH:
        raise T1HorizonCohortError("strict T1 cohort requires at least 64 candidates")
    selected = ranking[:COHORT_WIDTH]
    inverse = np.empty(len(ranking), dtype=np.int64)
    inverse[ranking] = np.arange(len(ranking), dtype=np.int64)
    ordered = np.sort(selected)
    ranks = inverse[ordered]
    if not is_strict_top64(ranks):
        raise AssertionError("strict top-64 selection lost a model rank")
    return ordered, ranks


class T1HorizonCohort:
    """Checksum-verified strict training cohort."""

    def __init__(
        self,
        root: str | Path,
        *,
        verify_checksums: bool = True,
        require_complete: bool = True,
    ):
        self.root = Path(root)
        self.manifest = _read_json(self.root / "cohort.json", "T1 cohort manifest")
        identity = self.manifest.get("scientific_identity")
        groups = int(self.manifest.get("groups", -1))
        if (
            self.manifest.get("schema_version") != SCHEMA_VERSION
            or self.manifest.get("cohort_schema") != COHORT_SCHEMA
            or self.manifest.get("experiment_id") != EXPERIMENT_ID
            or self.manifest.get("protocol_id") != PROTOCOL_ID
            or not isinstance(identity, dict)
            or canonical_blake3(identity) != self.manifest.get("cohort_id")
            or self.root.name != self.manifest.get("cohort_id")
            or groups <= 0
            or not isinstance(self.manifest.get("complete_train_corpus"), bool)
            or (
                require_complete
                and self.manifest.get("complete_train_corpus") is not True
            )
        ):
            raise T1HorizonCohortError("unsupported T1 cohort envelope")
        files = self.manifest.get("files")
        if not isinstance(files, dict) or set(files) != set(_TENSORS):
            raise T1HorizonCohortError("T1 cohort tensor set drifted")
        self.tensors: dict[str, np.memmap] = {}
        for name, (dtype_code, shape_factory) in _TENSORS.items():
            shape = shape_factory(groups)
            spec = files[name]
            path = self.root / str(spec.get("file"))
            dtype = _DTYPES[dtype_code]
            expected_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
            if (
                spec.get("dtype") != dtype_code
                or spec.get("shape") != list(shape)
                or spec.get("bytes") != expected_bytes
                or path.parent != self.root
                or not path.is_file()
                or path.stat().st_size != expected_bytes
                or (verify_checksums and checksum(path) != spec.get("blake3"))
            ):
                raise T1HorizonCohortError(f"T1 cohort tensor failed integrity: {name}")
            self.tensors[name] = np.memmap(path, mode="r", dtype=dtype, shape=shape)
        self._verify_semantics()

    @property
    def cohort_id(self) -> str:
        return str(self.manifest["cohort_id"])

    def _verify_semantics(self) -> None:
        groups = int(self.manifest["groups"])
        group_ids = np.asarray(self.tensors["group_ids"], dtype=np.uint64)
        positions = np.asarray(self.tensors["candidate_positions"], dtype=np.int64)
        sources = np.asarray(
            self.tensors["source_candidate_indices"],
            dtype=np.int64,
        )
        ranks = np.asarray(self.tensors["base_ranks"], dtype=np.int64)
        scores = np.asarray(self.tensors["base_scores"], dtype=np.float32)
        hashes = np.asarray(self.tensors["action_hashes"], dtype=np.uint8)
        direct = np.asarray(self.tensors["direct_cohort_indices"], dtype=np.int64)
        cohort_hashes = np.asarray(self.tensors["cohort_hashes"], dtype=np.uint8)
        if (
            len(np.unique(group_ids)) != groups
            or np.any(np.diff(positions, axis=1) <= 0)
            or np.any(np.diff(sources, axis=1) <= 0)
            or not np.isfinite(scores).all()
        ):
            raise T1HorizonCohortError("T1 cohort ordering or scores are invalid")
        for row in range(groups):
            if not is_strict_top64(ranks[row]):
                raise T1HorizonCohortError(f"T1 row {row} is not strict top 64")
            rank_zero = np.flatnonzero(ranks[row] == 0)
            if len(rank_zero) != 1 or direct[row] != rank_zero[0]:
                raise T1HorizonCohortError(f"T1 direct baseline drifted at row {row}")
            expected_hash = cohort_row_blake3(
                group_id=int(group_ids[row]),
                candidate_positions=positions[row],
                source_candidate_indices=sources[row],
                base_ranks=ranks[row],
                base_scores=scores[row],
                action_hashes=hashes[row],
            )
            if expected_hash != bytes(cohort_hashes[row]):
                raise T1HorizonCohortError(f"T1 cohort hash drifted at row {row}")


def build_t1_horizon_cohort(
    *,
    train_dataset: Path,
    validation_dataset: Path,
    source_cohort_root: Path,
    r3_cache_root: Path,
    s1_cache_root: Path,
    warm_start_checkpoint: Path,
    r3_authorization: Path,
    output_root: Path,
    receipt: Path,
    maximum_groups: int | None = None,
) -> dict[str, Any]:
    """Build the exact strict top-64 train cohort, rescoring only repaired rows."""
    if maximum_groups is not None and maximum_groups <= 0:
        raise T1HorizonCohortError("maximum_groups must be positive")
    production = maximum_groups is None
    mx.set_default_device(mx.gpu)
    source = O1RankingCohortCache(
        source_cohort_root,
        verify_checksums=True,
        require_complete=True,
    )
    source_train = source.split("train")
    r3 = R3ActionEditMlxCache(
        r3_cache_root,
        verify_checksums=not production,
        verify_semantics=not production,
        require_complete=True,
    )
    s1 = S1ExactSupplyCache(
        s1_cache_root,
        verify_checksums=not production,
        verify_semantics=not production,
        require_complete=True,
    )
    open_data = open_data_verification_identity(
        cache=r3,
        s1_cache=s1,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
    )
    verification_id = open_data_verification_id(open_data)
    authorization = _read_json(r3_authorization, "R3 authorization")
    if (
        authorization.get("approved") is not True
        or authorization.get("identity", {}).get("open_data_verification") != open_data
        or authorization.get("identity", {}).get("open_data_verification_id")
        != verification_id
    ):
        raise T1HorizonCohortError("R3 authorization does not bind the open data")
    dataset = r3.bind_dataset(
        train_dataset,
        s1_cache=s1,
        verify_dataset_checksums=not production,
        preverified_open_data_proof_id=verification_id,
    )
    if (
        source_train.groups != dataset.group_count
        or source_train.tensors["group_ids"].tobytes()
        != np.asarray(dataset.source.tensors["group_ids"], dtype="<u8").tobytes()
    ):
        raise T1HorizonCohortError("source cohort and train dataset groups differ")
    model, warm_start = _load_warm_start(warm_start_checkpoint)
    groups = dataset.group_count if maximum_groups is None else min(
        dataset.group_count,
        maximum_groups,
    )
    arrays = _empty_arrays(groups)
    reused_rows = 0
    rescored_rows = 0
    source_candidates = 0
    started = time.perf_counter()

    output_root.mkdir(parents=True, exist_ok=True)
    temporary = output_root / f".tmp-t1-horizon-cohort-{os.getpid()}-{time.time_ns()}"
    temporary.mkdir()
    try:
        for row in range(groups):
            source_ranks = np.asarray(
                source_train.tensors["base_ranks"][row],
                dtype=np.int64,
            )
            if is_strict_top64(source_ranks):
                positions = np.asarray(
                    source_train.tensors["candidate_positions"][row],
                    dtype=np.int64,
                )
                selected_sources = np.asarray(
                    source_train.tensors["source_candidate_indices"][row],
                    dtype=np.int64,
                )
                ranks = source_ranks
                selected_scores = np.asarray(
                    source_train.tensors["base_scores"][row],
                    dtype=np.float32,
                )
                selected_hashes = np.asarray(
                    source_train.tensors["action_hashes"][row],
                    dtype=np.uint8,
                )
                reused_rows += 1
            else:
                (
                    positions,
                    selected_sources,
                    ranks,
                    selected_scores,
                    selected_hashes,
                    candidate_count,
                ) = _rescore_strict_row(dataset, model, row)
                source_candidates += candidate_count
                rescored_rows += 1
            _store_row(
                arrays=arrays,
                source=source_train,
                row=row,
                positions=positions,
                selected_sources=selected_sources,
                ranks=ranks,
                selected_scores=selected_scores,
                selected_hashes=selected_hashes,
                rescored=not is_strict_top64(source_ranks),
            )
            if (row + 1) % 20 == 0 or row + 1 == groups:
                print(
                    json.dumps(
                        {
                            "event": "t1-horizon-cohort-progress",
                            "groups": row + 1,
                            "reused_rows": reused_rows,
                            "rescored_rows": rescored_rows,
                            "rescored_source_candidates": source_candidates,
                            "elapsed_seconds": time.perf_counter() - started,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                mx.clear_cache()

        files = {
            name: _write_tensor(temporary, name, values)
            for name, values in arrays.items()
        }
        complete = groups == dataset.group_count
        scientific_identity = {
            "schema_version": SCHEMA_VERSION,
            "cohort_schema": COHORT_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "protocol_id": PROTOCOL_ID,
            "complete_train_corpus": complete,
            "dataset_id": dataset.base.manifest["dataset_id"],
            "dataset_manifest_blake3": checksum(train_dataset / "dataset.json"),
            "source_cohort_id": source.cache_id,
            "source_cohort_manifest_blake3": checksum(source_cohort_root / "cache.json"),
            "r3_cache_id": r3.cache_id,
            "s1_cache_id": s1.cache_id,
            "open_data_verification_id": verification_id,
            "r3_authorization_blake3": checksum(r3_authorization),
            "warm_start": warm_start,
            "selection": {
                "base_arm": CONTROL_ARM,
                "cohort_width": COHORT_WIDTH,
                "split": "train",
                "policy": "strict-top64-without-label-insertion",
                "stable_tie_break": "ascending-canonical-action-blake3",
                "reuse_rule": "copy-only-when-source-base-ranks-are-exactly-0-through-63",
                "repair_rule": "rescore-full-source-group-with-frozen-exact-r2",
            },
            "groups": groups,
            "reused_rows": reused_rows,
            "rescored_rows": rescored_rows,
            "rescored_source_candidates": source_candidates,
            "files": files,
            "claim_boundary": {
                "open_train_only": True,
                "validation_opened": False,
                "sealed_test_opened": False,
                "gameplay_run": False,
            },
        }
        cohort_id = canonical_blake3(scientific_identity)
        manifest = {
            **scientific_identity,
            "cohort_id": cohort_id,
            "scientific_identity": scientific_identity,
        }
        _write_json_atomic(temporary / "cohort.json", manifest)
        final_root = output_root / cohort_id
        if final_root.exists():
            if _read_json(final_root / "cohort.json", "existing T1 cohort") != manifest:
                raise T1HorizonCohortError(
                    f"T1 cohort content-address collision at {final_root}"
                )
            shutil.rmtree(temporary)
        else:
            os.replace(temporary, final_root)
        T1HorizonCohort(
            final_root,
            verify_checksums=True,
            require_complete=production,
        )
        result = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "cohort_id": cohort_id,
            "cohort_root": str(final_root.resolve()),
            "groups": groups,
            "reused_rows": reused_rows,
            "rescored_rows": rescored_rows,
            "complete_train_corpus": complete,
        }
        _write_json_atomic(receipt, result)
        return result
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _empty_arrays(groups: int) -> dict[str, np.ndarray]:
    return {
        name: np.empty(shape_factory(groups), dtype=_DTYPES[dtype_code])
        for name, (dtype_code, shape_factory) in _TENSORS.items()
    }


def _rescore_strict_row(
    dataset: Any,
    model: R3ActionEditRanker,
    row: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    batch = dataset.batch(
        [row],
        arm=CONTROL_ARM,
        transform_ids=[0],
        verify_control_hashes=False,
        control_materialization=CONTROL_MATERIALIZATION_PREVERIFIED_VECTORIZED,
    )
    mask = np.asarray(batch.base.candidate_mask, dtype=np.bool_)[0]
    count = int(mask.sum())
    scores = _score_batch(model, batch, count)
    hashes = np.asarray(batch.base.action_hash, dtype=np.uint8)[0, :count]
    positions, ranks = strict_top64_positions(scores, hashes)
    sources = np.asarray(batch.source_candidate_indices, dtype=np.int64)[0, :count]
    return (
        positions,
        sources[positions],
        ranks,
        np.asarray(scores[positions], dtype=np.float32),
        hashes[positions],
        count,
    )


def _store_row(
    *,
    arrays: dict[str, np.ndarray],
    source: Any,
    row: int,
    positions: np.ndarray,
    selected_sources: np.ndarray,
    ranks: np.ndarray,
    selected_scores: np.ndarray,
    selected_hashes: np.ndarray,
    rescored: bool,
) -> None:
    if not is_strict_top64(ranks):
        raise T1HorizonCohortError(f"row {row} is not strict top 64")
    direct = np.flatnonzero(np.asarray(ranks) == 0)
    if len(direct) != 1:
        raise T1HorizonCohortError(f"row {row} has no unique direct baseline")
    group_id = int(source.tensors["group_ids"][row])
    arrays["group_ids"][row] = group_id
    arrays["game_indices"][row] = source.tensors["game_indices"][row]
    arrays["turns"][row] = source.tensors["turns"][row]
    arrays["current_players"][row] = source.tensors["current_players"][row]
    arrays["candidate_positions"][row] = positions
    arrays["source_candidate_indices"][row] = selected_sources
    arrays["base_ranks"][row] = ranks
    arrays["base_scores"][row] = selected_scores
    arrays["action_hashes"][row] = selected_hashes
    arrays["direct_cohort_indices"][row] = direct[0]
    arrays["rescored_flags"][row] = int(rescored)
    arrays["cohort_hashes"][row] = np.frombuffer(
        cohort_row_blake3(
            group_id=group_id,
            candidate_positions=positions,
            source_candidate_indices=selected_sources,
            base_ranks=ranks,
            base_scores=selected_scores,
            action_hashes=selected_hashes,
        ),
        dtype=np.uint8,
    )


def _score_batch(
    model: R3ActionEditRanker,
    batch: Any,
    count: int,
) -> np.ndarray:
    model.eval()
    parent = model.encode_parent(batch)
    mx.eval(parent)
    chunks = []
    for start in range(0, count, CANDIDATE_CHUNK):
        prediction = model.predict(
            batch,
            candidate_slice=slice(start, min(start + CANDIDATE_CHUNK, count)),
            parent_state=parent,
        )
        mx.eval(prediction.scores)
        chunks.append(np.asarray(prediction.scores, dtype=np.float32)[0])
    scores = np.concatenate(chunks)
    if scores.shape != (count,) or not np.isfinite(scores).all():
        raise T1HorizonCohortError("frozen exact-R2 scoring emitted invalid scores")
    return scores


def _load_warm_start(
    checkpoint: Path,
) -> tuple[R3ActionEditRanker, dict[str, Any]]:
    manifest = _read_json(checkpoint / "checkpoint.json", "exact-R2 checkpoint")
    files = manifest.get("files")
    if not isinstance(files, dict) or "model.safetensors" not in files:
        raise T1HorizonCohortError("exact-R2 checkpoint manifest is incomplete")
    for name, expected in files.items():
        path = checkpoint / name
        if (
            not isinstance(expected, dict)
            or not path.is_file()
            or path.stat().st_size != expected.get("bytes")
            or checksum(path) != expected.get("blake3")
        ):
            raise T1HorizonCohortError(f"exact-R2 checkpoint failed integrity: {name}")
    config = R3ActionEditModelConfig.from_dict(manifest["model_config"])
    if config.arm != CONTROL_ARM:
        raise T1HorizonCohortError("T1 warm start must use the exact-R2 control")
    model = R3ActionEditRanker(config)
    model.load_weights(str(checkpoint / "model.safetensors"))
    mx.eval(model.parameters())
    model.eval()
    return model, {
        "checkpoint_id": manifest["checkpoint_id"],
        "checkpoint_manifest_blake3": checksum(checkpoint / "checkpoint.json"),
        "model_blake3": checksum(checkpoint / "model.safetensors"),
        "model_config": config.to_dict(),
    }


def _write_tensor(root: Path, name: str, values: np.ndarray) -> dict[str, Any]:
    path = root / f"{name.replace('_', '-')}.bin"
    array = np.ascontiguousarray(values)
    with path.open("wb") as handle:
        handle.write(array.tobytes(order="C"))
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "file": path.name,
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "bytes": path.stat().st_size,
        "blake3": checksum(path),
    }


def checksum(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def canonical_blake3(value: object) -> str:
    return blake3.blake3(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise T1HorizonCohortError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise T1HorizonCohortError(f"{label} must be an object")
    return value


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--source-cohort-root", type=Path, required=True)
    parser.add_argument("--r3-cache-root", type=Path, required=True)
    parser.add_argument("--s1-cache-root", type=Path, required=True)
    parser.add_argument("--warm-start-checkpoint", type=Path, required=True)
    parser.add_argument("--r3-authorization", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--maximum-groups", type=int)
    args = parser.parse_args()
    result = build_t1_horizon_cohort(
        train_dataset=args.train_dataset,
        validation_dataset=args.validation_dataset,
        source_cohort_root=args.source_cohort_root,
        r3_cache_root=args.r3_cache_root,
        s1_cache_root=args.s1_cache_root,
        warm_start_checkpoint=args.warm_start_checkpoint,
        r3_authorization=args.r3_authorization,
        output_root=args.output_root,
        receipt=args.receipt,
        maximum_groups=args.maximum_groups,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
