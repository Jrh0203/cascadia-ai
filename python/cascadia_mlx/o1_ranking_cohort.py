"""Immutable exact-R2 top-64 cohorts for the O1 ranking integration."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import numpy as np

from cascadia_mlx.r3_action_edit_mlx_cache import (
    CONTROL_ARM,
    CONTROL_MATERIALIZATION_PREVERIFIED_VECTORIZED,
    R3ActionEditMlxCache,
    R3ActionEditMlxDataset,
    open_data_verification_id,
    open_data_verification_identity,
)
from cascadia_mlx.r3_action_edit_mlx_model import (
    R3ActionEditModelConfig,
    R3ActionEditRanker,
)
from cascadia_mlx.s1_exact_supply_mlx_cache import S1ExactSupplyCache

EXPERIMENT_ID = "o1-high-regret-draft-ranking-integration-v1"
PROTOCOL_ID = "o1-intent-conditioned-exact-r2-reranker-v1"
ADR_ID = "0188"
COHORT_SCHEMA_VERSION = 1
COHORT_SCHEMA = "o1-ranking-exact-r2-top64-cohort-v1"
COHORT_WIDTH = 64
CANDIDATE_CHUNK = 256

_COHORT_HASH_DOMAIN = b"cascadia-v2-o1-ranking-cohort-row-v1"
_DTYPES = {
    "|u1": np.dtype("u1"),
    "<i2": np.dtype("<i2"),
    "<u2": np.dtype("<u2"),
    "<u8": np.dtype("<u8"),
    "<f4": np.dtype("<f4"),
}
_SPLIT_TENSORS = {
    "group_ids": ("<u8", lambda groups: (groups,)),
    "game_indices": ("<u8", lambda groups: (groups,)),
    "turns": ("|u1", lambda groups: (groups,)),
    "current_players": ("|u1", lambda groups: (groups,)),
    "candidate_positions": ("<u2", lambda groups: (groups, COHORT_WIDTH)),
    "source_candidate_indices": ("<u2", lambda groups: (groups, COHORT_WIDTH)),
    "base_ranks": ("<u2", lambda groups: (groups, COHORT_WIDTH)),
    "base_scores": ("<f4", lambda groups: (groups, COHORT_WIDTH)),
    "action_hashes": ("|u1", lambda groups: (groups, COHORT_WIDTH, 32)),
    "selected_source_indices": ("<u2", lambda groups: (groups,)),
    "champion_source_indices": ("<u2", lambda groups: (groups,)),
    "selected_cohort_indices": ("<i2", lambda groups: (groups,)),
    "cohort_hashes": ("|u1", lambda groups: (groups, 32)),
}


class O1RankingCohortError(ValueError):
    """The O1 ranking cohort is incompatible or not reproducible."""


@dataclass(frozen=True)
class O1RankingCohortSplit:
    """Memory-mapped immutable cohort tensors for one role."""

    name: str
    groups: int
    source_candidates: int
    tensors: dict[str, np.memmap]
    group_rows: dict[int, int]

    def positions(self, rows: np.ndarray) -> tuple[np.ndarray, ...]:
        values = np.asarray(self.tensors["candidate_positions"][rows], dtype=np.int64)
        return tuple(row.copy() for row in values)


class O1RankingCohortCache:
    """Checksum-verified train/validation top-64 selection."""

    def __init__(
        self,
        root: str | Path,
        *,
        verify_checksums: bool = True,
        require_complete: bool = True,
    ):
        self.root = Path(root)
        self.manifest = _read_json(self.root / "cache.json", "O1 ranking cohort manifest")
        self._validate_envelope(require_complete=require_complete)
        self.splits = {
            split: self._load_split(split, verify_checksums=verify_checksums)
            for split in self.manifest["splits"]
        }

    @property
    def cache_id(self) -> str:
        return str(self.manifest["cache_id"])

    def split(self, name: str) -> O1RankingCohortSplit:
        try:
            return self.splits[name]
        except KeyError as error:
            raise O1RankingCohortError(f"cohort split is absent: {name}") from error

    def _validate_envelope(self, *, require_complete: bool) -> None:
        manifest = self.manifest
        identity = manifest.get("scientific_identity")
        if (
            manifest.get("schema_version") != COHORT_SCHEMA_VERSION
            or manifest.get("cache_schema") != COHORT_SCHEMA
            or manifest.get("experiment_id") != EXPERIMENT_ID
            or manifest.get("protocol_id") != PROTOCOL_ID
            or manifest.get("adr") != ADR_ID
            or not isinstance(identity, dict)
            or _canonical_blake3(identity) != manifest.get("cache_id")
            or self.root.name != manifest.get("cache_id")
        ):
            raise O1RankingCohortError("unsupported or invalid O1 ranking cohort envelope")
        if require_complete and manifest.get("complete_open_corpus") is not True:
            raise O1RankingCohortError("production O1 ranking cohort must cover all open groups")
        if manifest.get("selection") != {
            "base_arm": CONTROL_ARM,
            "candidate_chunk": CANDIDATE_CHUNK,
            "cohort_width": COHORT_WIDTH,
            "geometry": "canonical",
            "stable_tie_break": "ascending-canonical-action-blake3",
            "train": "top64-if-selected-present-else-top63-plus-selected",
            "validation": "strict-top64",
        }:
            raise O1RankingCohortError("O1 ranking cohort selection contract drifted")
        splits = manifest.get("splits")
        if not isinstance(splits, dict) or not splits or set(splits) - {"train", "validation"}:
            raise O1RankingCohortError("O1 ranking cohort split set is malformed")
        if identity.get("splits") != splits:
            raise O1RankingCohortError("O1 ranking scientific identity does not bind its splits")

    def _load_split(self, split: str, *, verify_checksums: bool) -> O1RankingCohortSplit:
        raw = self.manifest["splits"][split]
        if not isinstance(raw, dict):
            raise O1RankingCohortError(f"{split} cohort manifest is malformed")
        groups = int(raw.get("groups", -1))
        source_candidates = int(raw.get("source_candidates", -1))
        if groups <= 0 or source_candidates < groups * COHORT_WIDTH:
            raise O1RankingCohortError(f"{split} cohort counts are invalid")
        files = raw.get("files")
        if not isinstance(files, dict) or set(files) != set(_SPLIT_TENSORS):
            raise O1RankingCohortError(f"{split} cohort tensor set drifted")
        tensors: dict[str, np.memmap] = {}
        for name, (dtype_code, shape_factory) in _SPLIT_TENSORS.items():
            shape = shape_factory(groups)
            spec = files[name]
            if (
                not isinstance(spec, dict)
                or spec.get("dtype") != dtype_code
                or spec.get("shape") != list(shape)
            ):
                raise O1RankingCohortError(f"{split} cohort tensor shape drifted: {name}")
            path = self.root / str(spec.get("file"))
            dtype = _DTYPES[dtype_code]
            expected_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
            if (
                path.parent != self.root
                or not path.is_file()
                or path.stat().st_size != expected_bytes
                or spec.get("bytes") != expected_bytes
                or (
                    verify_checksums
                    and _checksum(path) != spec.get("blake3")
                )
            ):
                raise O1RankingCohortError(f"{split} cohort tensor failed integrity: {name}")
            tensors[name] = np.memmap(path, mode="r", dtype=dtype, shape=shape)
        self._verify_split_semantics(split, tensors)
        group_ids = np.asarray(tensors["group_ids"], dtype=np.uint64)
        return O1RankingCohortSplit(
            name=split,
            groups=groups,
            source_candidates=source_candidates,
            tensors=tensors,
            group_rows={int(group_id): row for row, group_id in enumerate(group_ids)},
        )

    def _verify_split_semantics(
        self,
        split: str,
        tensors: dict[str, np.memmap],
    ) -> None:
        group_ids = np.asarray(tensors["group_ids"], dtype=np.uint64)
        positions = np.asarray(tensors["candidate_positions"], dtype=np.int64)
        sources = np.asarray(tensors["source_candidate_indices"], dtype=np.int64)
        ranks = np.asarray(tensors["base_ranks"], dtype=np.int64)
        scores = np.asarray(tensors["base_scores"], dtype=np.float32)
        hashes = np.asarray(tensors["action_hashes"], dtype=np.uint8)
        selected_sources = np.asarray(tensors["selected_source_indices"], dtype=np.int64)
        selected_cohort = np.asarray(tensors["selected_cohort_indices"], dtype=np.int64)
        cohort_hashes = np.asarray(tensors["cohort_hashes"], dtype=np.uint8)
        if (
            len(np.unique(group_ids)) != len(group_ids)
            or np.any(np.diff(positions, axis=1) <= 0)
            or np.any(np.diff(sources, axis=1) <= 0)
            or not np.isfinite(scores).all()
            or np.any(ranks < 0)
        ):
            raise O1RankingCohortError(f"{split} cohort ordering or scores are invalid")
        for row, group_id in enumerate(group_ids):
            selected_matches = np.flatnonzero(sources[row] == selected_sources[row])
            expected_selected = int(selected_matches[0]) if len(selected_matches) else -1
            if selected_cohort[row] != expected_selected:
                raise O1RankingCohortError(f"{split} selected-action mapping drifted at row {row}")
            if split == "train" and expected_selected < 0:
                raise O1RankingCohortError(f"train cohort omitted selected action at row {row}")
            if split == "validation" and set(int(value) for value in ranks[row]) != set(
                range(COHORT_WIDTH)
            ):
                raise O1RankingCohortError(f"validation cohort is not strict top64 at row {row}")
            observed = cohort_row_blake3(
                group_id=int(group_id),
                candidate_positions=positions[row],
                source_candidate_indices=sources[row],
                base_ranks=ranks[row],
                base_scores=scores[row],
                action_hashes=hashes[row],
            )
            if observed != bytes(cohort_hashes[row]):
                raise O1RankingCohortError(f"{split} cohort hash drifted at row {row}")


def stable_score_ranking(scores: np.ndarray, action_hashes: np.ndarray) -> np.ndarray:
    """Rank descending float scores with ascending 32-byte action hashes."""
    values = np.asarray(scores, dtype=np.float32)
    hashes = np.asarray(action_hashes, dtype=np.uint8)
    if (
        values.ndim != 1
        or hashes.shape != (len(values), 32)
        or not len(values)
        or not np.isfinite(values).all()
    ):
        raise ValueError("stable score ranking requires finite scores and aligned hashes")
    return np.asarray(
        sorted(
            range(len(values)),
            key=lambda index: (-float(values[index]), bytes(hashes[index])),
        ),
        dtype=np.int64,
    )


def select_cohort_positions(
    scores: np.ndarray,
    action_hashes: np.ndarray,
    *,
    split: str,
    selected_position: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return source-ordered positions and their zero-based base ranks."""
    ranking = stable_score_ranking(scores, action_hashes)
    if len(ranking) < COHORT_WIDTH:
        raise ValueError("O1 ranking cohort requires at least 64 candidates")
    if split == "train":
        if selected_position < 0 or selected_position >= len(ranking):
            raise ValueError("train selected position is out of range")
        selected = ranking[:COHORT_WIDTH].copy()
        if selected_position not in selected:
            selected = np.concatenate([ranking[: COHORT_WIDTH - 1], [selected_position]])
    elif split == "validation":
        selected = ranking[:COHORT_WIDTH].copy()
    else:
        raise ValueError("open O1 cohort split must be train or validation")
    if len(np.unique(selected)) != COHORT_WIDTH:
        raise AssertionError("O1 cohort selection duplicated a candidate")
    inverse = np.empty(len(ranking), dtype=np.int64)
    inverse[ranking] = np.arange(len(ranking), dtype=np.int64)
    ordered = np.sort(selected)
    return ordered, inverse[ordered]


def cohort_row_blake3(
    *,
    group_id: int,
    candidate_positions: np.ndarray,
    source_candidate_indices: np.ndarray,
    base_ranks: np.ndarray,
    base_scores: np.ndarray,
    action_hashes: np.ndarray,
) -> bytes:
    """Hash one complete cohort row in its stored source order."""
    positions = np.asarray(candidate_positions, dtype="<u2")
    sources = np.asarray(source_candidate_indices, dtype="<u2")
    ranks = np.asarray(base_ranks, dtype="<u2")
    scores = np.asarray(base_scores, dtype="<f4")
    hashes = np.asarray(action_hashes, dtype=np.uint8)
    if (
        positions.shape != (COHORT_WIDTH,)
        or sources.shape != (COHORT_WIDTH,)
        or ranks.shape != (COHORT_WIDTH,)
        or scores.shape != (COHORT_WIDTH,)
        or hashes.shape != (COHORT_WIDTH, 32)
    ):
        raise ValueError("O1 cohort row has the wrong shape")
    digest = blake3.blake3()
    digest.update(_COHORT_HASH_DOMAIN)
    digest.update(int(group_id).to_bytes(8, "little"))
    digest.update(positions.tobytes())
    digest.update(sources.tobytes())
    digest.update(ranks.tobytes())
    digest.update(scores.tobytes())
    digest.update(hashes.tobytes(order="C"))
    return digest.digest()


def build_o1_ranking_cohort(
    *,
    train_dataset: Path,
    validation_dataset: Path,
    r3_cache_root: Path,
    s1_cache_root: Path,
    warm_start_checkpoint: Path,
    authorization_path: Path,
    output_root: Path,
    receipt_path: Path,
    maximum_groups_per_split: int | None = None,
) -> dict[str, Any]:
    """Score the open corpus and materialize the frozen top-64 cohort."""
    if maximum_groups_per_split is not None and maximum_groups_per_split <= 0:
        raise ValueError("maximum_groups_per_split must be positive")
    production = maximum_groups_per_split is None
    mx.set_default_device(mx.gpu)
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
    authorization = _read_json(authorization_path, "R3 authorization")
    authorization_identity = authorization.get("identity")
    if (
        authorization.get("approved") is not True
        or not isinstance(authorization_identity, dict)
        or authorization_identity.get("open_data_verification") != open_data
        or authorization_identity.get("open_data_verification_id") != verification_id
    ):
        raise O1RankingCohortError("R3 authorization does not verify the requested open data")
    datasets = {
        "train": r3.bind_dataset(
            train_dataset,
            s1_cache=s1,
            verify_dataset_checksums=not production,
            preverified_open_data_proof_id=verification_id,
        ),
        "validation": r3.bind_dataset(
            validation_dataset,
            s1_cache=s1,
            verify_dataset_checksums=not production,
            preverified_open_data_proof_id=verification_id,
        ),
    }
    model, warm_start = _load_warm_start(warm_start_checkpoint)

    output_root.mkdir(parents=True, exist_ok=True)
    temporary = output_root / f".tmp-o1-ranking-cohort-{os.getpid()}-{time.time_ns()}"
    temporary.mkdir()
    try:
        split_manifests = {
            split: _build_split(
                split=split,
                dataset=dataset,
                model=model,
                root=temporary,
                maximum_groups=maximum_groups_per_split,
            )
            for split, dataset in datasets.items()
        }
        complete_open_corpus = production and all(
            split_manifests[name]["groups"] == datasets[name].group_count
            for name in datasets
        )
        selection = {
            "base_arm": CONTROL_ARM,
            "candidate_chunk": CANDIDATE_CHUNK,
            "cohort_width": COHORT_WIDTH,
            "geometry": "canonical",
            "stable_tie_break": "ascending-canonical-action-blake3",
            "train": "top64-if-selected-present-else-top63-plus-selected",
            "validation": "strict-top64",
        }
        hidden_information = {
            "open_train_used": True,
            "open_validation_used": True,
            "sealed_test_opened": False,
            "gameplay_run": False,
            "hidden_order_exported": False,
            "future_refill_exported": False,
        }
        scientific_identity = {
            "schema_version": COHORT_SCHEMA_VERSION,
            "cache_schema": COHORT_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "protocol_id": PROTOCOL_ID,
            "adr": ADR_ID,
            "complete_open_corpus": complete_open_corpus,
            "selection": selection,
            "open_data_verification": open_data,
            "open_data_verification_id": verification_id,
            "warm_start": warm_start,
            "hidden_information": hidden_information,
            "splits": split_manifests,
        }
        cache_id = _canonical_blake3(scientific_identity)
        manifest = {
            **scientific_identity,
            "cache_id": cache_id,
            "scientific_identity": scientific_identity,
        }
        _write_json_atomic(temporary / "cache.json", manifest)
        final_root = output_root / cache_id
        if final_root.exists():
            existing = _read_json(final_root / "cache.json", "existing O1 ranking cohort")
            if existing != manifest:
                raise O1RankingCohortError(
                    f"O1 ranking cohort content-address collision at {final_root}"
                )
            shutil.rmtree(temporary)
        else:
            os.replace(temporary, final_root)
        receipt = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "cache_id": cache_id,
            "cache_root": str(final_root.resolve()),
            "complete_open_corpus": complete_open_corpus,
            "train_groups": split_manifests["train"]["groups"],
            "validation_groups": split_manifests["validation"]["groups"],
            "open_data_verification_id": verification_id,
            "warm_start_model_blake3": warm_start["model_blake3"],
        }
        _write_json_atomic(receipt_path, receipt)
        O1RankingCohortCache(
            final_root,
            verify_checksums=True,
            require_complete=production,
        )
        return receipt
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _build_split(
    *,
    split: str,
    dataset: R3ActionEditMlxDataset,
    model: R3ActionEditRanker,
    root: Path,
    maximum_groups: int | None,
) -> dict[str, Any]:
    groups = dataset.group_count if maximum_groups is None else min(
        dataset.group_count,
        maximum_groups,
    )
    arrays = {
        "group_ids": np.empty(groups, dtype="<u8"),
        "game_indices": np.empty(groups, dtype="<u8"),
        "turns": np.empty(groups, dtype="u1"),
        "current_players": np.empty(groups, dtype="u1"),
        "candidate_positions": np.empty((groups, COHORT_WIDTH), dtype="<u2"),
        "source_candidate_indices": np.empty((groups, COHORT_WIDTH), dtype="<u2"),
        "base_ranks": np.empty((groups, COHORT_WIDTH), dtype="<u2"),
        "base_scores": np.empty((groups, COHORT_WIDTH), dtype="<f4"),
        "action_hashes": np.empty((groups, COHORT_WIDTH, 32), dtype="u1"),
        "selected_source_indices": np.empty(groups, dtype="<u2"),
        "champion_source_indices": np.empty(groups, dtype="<u2"),
        "selected_cohort_indices": np.empty(groups, dtype="<i2"),
        "cohort_hashes": np.empty((groups, 32), dtype="u1"),
    }
    source_candidates = 0
    started = time.perf_counter()
    for row in range(groups):
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
        selected_position = int(np.asarray(batch.base.selected_index)[0])
        positions, ranks = select_cohort_positions(
            scores,
            hashes,
            split=split,
            selected_position=selected_position,
        )
        sources = np.asarray(batch.source_candidate_indices, dtype=np.int64)[0, :count]
        selected_sources = sources[positions]
        selected_source = int(sources[selected_position])
        champion_position = int(np.asarray(batch.base.champion_index)[0])
        champion_source = int(sources[champion_position])
        selected_matches = np.flatnonzero(selected_sources == selected_source)
        selected_cohort = int(selected_matches[0]) if len(selected_matches) else -1
        selected_scores = np.asarray(scores[positions], dtype="<f4")
        selected_hashes = hashes[positions]
        group_id = int(
            np.asarray(dataset.source.tensors["group_ids"], dtype=np.uint64)[row]
        )

        arrays["group_ids"][row] = group_id
        arrays["game_indices"][row] = int(np.asarray(batch.base.game_index)[0])
        arrays["turns"][row] = int(np.asarray(batch.base.turn)[0])
        arrays["current_players"][row] = int(np.asarray(batch.base.current_player)[0])
        arrays["candidate_positions"][row] = positions
        arrays["source_candidate_indices"][row] = selected_sources
        arrays["base_ranks"][row] = ranks
        arrays["base_scores"][row] = selected_scores
        arrays["action_hashes"][row] = selected_hashes
        arrays["selected_source_indices"][row] = selected_source
        arrays["champion_source_indices"][row] = champion_source
        arrays["selected_cohort_indices"][row] = selected_cohort
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
        source_candidates += count
        if (row + 1) % 10 == 0 or row + 1 == groups:
            print(
                json.dumps(
                    {
                        "event": "o1-ranking-cohort-progress",
                        "split": split,
                        "groups": row + 1,
                        "source_candidates": source_candidates,
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if (row + 1) % 20 == 0:
            mx.clear_cache()

    files = {
        name: _write_tensor(root, f"{split}-{name.replace('_', '-')}.bin", values)
        for name, values in arrays.items()
    }
    return {
        "split": split,
        "groups": groups,
        "source_candidates": source_candidates,
        "retained_candidates": groups * COHORT_WIDTH,
        "complete_open_split": groups == dataset.group_count,
        "dataset_id": dataset.base.manifest["dataset_id"],
        "r3_cache_id": dataset.cache.cache_id,
        "s1_cache_id": dataset.s1_cache.cache_id,
        "files": files,
    }


def _score_batch(
    model: R3ActionEditRanker,
    batch: object,
    count: int,
) -> np.ndarray:
    model.eval()
    parent = model.encode_parent(batch)
    mx.eval(parent)
    chunks: list[np.ndarray] = []
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
        raise O1RankingCohortError("frozen exact-R2 scoring emitted invalid scores")
    return scores


def _load_warm_start(
    checkpoint: Path,
) -> tuple[R3ActionEditRanker, dict[str, Any]]:
    manifest = _read_json(checkpoint / "checkpoint.json", "exact-R2 checkpoint")
    files = manifest.get("files")
    if not isinstance(files, dict) or "model.safetensors" not in files:
        raise O1RankingCohortError("exact-R2 checkpoint file manifest is incomplete")
    for name, expected in files.items():
        path = checkpoint / name
        if (
            not isinstance(expected, dict)
            or not path.is_file()
            or path.stat().st_size != expected.get("bytes")
            or _checksum(path) != expected.get("blake3")
        ):
            raise O1RankingCohortError(f"exact-R2 checkpoint failed integrity: {name}")
    config = R3ActionEditModelConfig.from_dict(manifest["model_config"])
    if config.arm != CONTROL_ARM:
        raise O1RankingCohortError("O1 warm start must be the exact-R2 control arm")
    model = R3ActionEditRanker(config)
    model.load_weights(str(checkpoint / "model.safetensors"))
    mx.eval(model.parameters())
    model.eval()
    return model, {
        "checkpoint_id": manifest["checkpoint_id"],
        "checkpoint_manifest_blake3": _checksum(checkpoint / "checkpoint.json"),
        "model_blake3": _checksum(checkpoint / "model.safetensors"),
        "model_config": config.to_dict(),
    }


def _write_tensor(root: Path, file_name: str, values: np.ndarray) -> dict[str, Any]:
    path = root / file_name
    array = np.ascontiguousarray(values)
    with path.open("wb") as handle:
        handle.write(array.tobytes(order="C"))
        handle.flush()
        os.fsync(handle.fileno())
    dtype = array.dtype.str
    if dtype not in _DTYPES:
        raise O1RankingCohortError(f"unsupported cohort tensor dtype: {dtype}")
    return {
        "file": file_name,
        "dtype": dtype,
        "shape": list(array.shape),
        "bytes": path.stat().st_size,
        "blake3": _checksum(path),
    }


def _canonical_blake3(value: object) -> str:
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
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise O1RankingCohortError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise O1RankingCohortError(f"{label} must be a JSON object")
    return value


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or verify the frozen O1 exact-R2 top-64 cohort"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--train-dataset", type=Path, required=True)
    build.add_argument("--validation-dataset", type=Path, required=True)
    build.add_argument("--r3-cache", type=Path, required=True)
    build.add_argument("--s1-cache", type=Path, required=True)
    build.add_argument("--warm-start-checkpoint", type=Path, required=True)
    build.add_argument("--authorization", type=Path, required=True)
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--receipt", type=Path, required=True)
    build.add_argument("--maximum-groups-per-split", type=int)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--cache", type=Path, required=True)
    verify.add_argument("--allow-partial", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "build":
        result = build_o1_ranking_cohort(
            train_dataset=args.train_dataset,
            validation_dataset=args.validation_dataset,
            r3_cache_root=args.r3_cache,
            s1_cache_root=args.s1_cache,
            warm_start_checkpoint=args.warm_start_checkpoint,
            authorization_path=args.authorization,
            output_root=args.output_root,
            receipt_path=args.receipt,
            maximum_groups_per_split=args.maximum_groups_per_split,
        )
    else:
        cache = O1RankingCohortCache(
            args.cache,
            verify_checksums=True,
            require_complete=not args.allow_partial,
        )
        result = {
            "schema_version": 1,
            "cache_id": cache.cache_id,
            "cache_root": str(cache.root.resolve()),
            "splits": {
                name: {
                    "groups": split.groups,
                    "source_candidates": split.source_candidates,
                }
                for name, split in cache.splits.items()
            },
        }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
