"""Fail-closed loader for authoritative ADR 0188 candidate afterstates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import blake3
import numpy as np

from cascadia_mlx.o1_ranking_cohort import (
    ADR_ID,
    COHORT_WIDTH,
    EXPERIMENT_ID,
    PROTOCOL_ID,
    O1RankingCohortCache,
)
from cascadia_mlx.opponent_intent_dataset import OPPONENT_INTENT_RECORD_DTYPE

AFTERSTATE_SCHEMA_VERSION = 1
AFTERSTATE_SCHEMA = "o1-ranking-public-afterstate-cache-v1"
_MODEL_INPUT_HASH_DOMAIN = b"cascadia-v2-o1-ranking-model-input-v1"


class O1RankingAfterstateError(ValueError):
    """The authoritative afterstate cache is absent, corrupt, or misaligned."""


@dataclass(frozen=True)
class O1RankingAfterstateSplit:
    """Aligned candidate records and provenance tensors for one split."""

    name: str
    groups: int
    group_ids: np.memmap
    source_candidate_indices: np.memmap
    action_hashes: np.memmap
    records: np.memmap
    model_input_hashes: np.memmap


class O1RankingAfterstateCache:
    """Content-addressed target-free O1 records produced by Rust replay."""

    def __init__(
        self,
        root: str | Path,
        *,
        cohort: O1RankingCohortCache,
        verify_checksums: bool = True,
        verify_model_inputs: bool = True,
        require_complete: bool = True,
    ):
        self.root = Path(root)
        self.cohort = cohort
        self.manifest = _read_json(
            self.root / "cache.json",
            "O1 ranking afterstate manifest",
        )
        self._validate_envelope(require_complete=require_complete)
        self.splits = {
            name: self._load_split(
                name,
                verify_checksums=verify_checksums,
                verify_model_inputs=verify_model_inputs,
            )
            for name in self.manifest["splits"]
        }

    @property
    def cache_id(self) -> str:
        return str(self.manifest["cache_id"])

    def split(self, name: str) -> O1RankingAfterstateSplit:
        try:
            return self.splits[name]
        except KeyError as error:
            raise O1RankingAfterstateError(f"afterstate split is absent: {name}") from error

    def _validate_envelope(self, *, require_complete: bool) -> None:
        manifest = self.manifest
        identity = manifest.get("scientific_identity")
        if (
            manifest.get("schema_version") != AFTERSTATE_SCHEMA_VERSION
            or manifest.get("cache_schema") != AFTERSTATE_SCHEMA
            or manifest.get("experiment_id") != EXPERIMENT_ID
            or manifest.get("protocol_id") != PROTOCOL_ID
            or manifest.get("adr") != ADR_ID
            or manifest.get("cohort_id") != self.cohort.cache_id
            or manifest.get("cohort_manifest_blake3")
            != _checksum(self.cohort.root / "cache.json")
            or not isinstance(identity, dict)
            or _canonical_blake3(identity) != manifest.get("cache_id")
            or self.root.name != manifest.get("cache_id")
        ):
            raise O1RankingAfterstateError("invalid O1 ranking afterstate envelope")
        if require_complete and manifest.get("complete_open_corpus") is not True:
            raise O1RankingAfterstateError(
                "production O1 ranking afterstates must cover all open groups"
            )
        hidden = manifest.get("hidden_information")
        if hidden != {
            "public_candidate_afterstate_only": True,
            "champion_history_only": True,
            "target_fields_zeroed": True,
            "policy_identity_exported": False,
            "hidden_post_draft_refill_exported": False,
            "hidden_stack_order_exported": False,
            "hidden_bag_order_exported": False,
            "sealed_test_opened": False,
            "gameplay_run": False,
        }:
            raise O1RankingAfterstateError("afterstate hidden-information boundary drifted")
        if set(manifest.get("splits", {})) != set(self.cohort.splits):
            raise O1RankingAfterstateError("afterstate and cohort split sets differ")

    def _load_split(
        self,
        split: str,
        *,
        verify_checksums: bool,
        verify_model_inputs: bool,
    ) -> O1RankingAfterstateSplit:
        raw = self.manifest["splits"][split]
        cohort = self.cohort.split(split)
        groups = int(raw.get("groups", -1))
        if (
            groups != cohort.groups
            or raw.get("cohort_groups") != cohort.groups
            or raw.get("candidates") != cohort.groups * COHORT_WIDTH
            or raw.get("dataset_id")
            != self.cohort.manifest["splits"][split]["dataset_id"]
        ):
            raise O1RankingAfterstateError(f"{split} afterstate counts or dataset drifted")
        expected = {
            "group_ids": ("<u8", (groups,), np.dtype("<u8")),
            "source_candidate_indices": (
                "<u2",
                (groups, COHORT_WIDTH),
                np.dtype("<u2"),
            ),
            "action_hashes": (
                "|u1",
                (groups, COHORT_WIDTH, 32),
                np.dtype("u1"),
            ),
            "records": (
                "|u1",
                (groups, COHORT_WIDTH, OPPONENT_INTENT_RECORD_DTYPE.itemsize),
                OPPONENT_INTENT_RECORD_DTYPE,
            ),
            "model_input_hashes": (
                "|u1",
                (groups, COHORT_WIDTH, 32),
                np.dtype("u1"),
            ),
        }
        files = raw.get("files")
        if not isinstance(files, dict) or set(files) != set(expected):
            raise O1RankingAfterstateError(f"{split} afterstate tensor set drifted")
        loaded: dict[str, np.memmap] = {}
        for name, (dtype_code, manifest_shape, memmap_dtype) in expected.items():
            spec = files[name]
            if (
                not isinstance(spec, dict)
                or spec.get("dtype") != dtype_code
                or spec.get("shape") != list(manifest_shape)
            ):
                raise O1RankingAfterstateError(f"{split} afterstate shape drifted: {name}")
            path = self.root / str(spec.get("file"))
            expected_bytes = (
                groups * COHORT_WIDTH * OPPONENT_INTENT_RECORD_DTYPE.itemsize
                if name == "records"
                else int(np.prod(manifest_shape, dtype=np.int64))
                * np.dtype(memmap_dtype).itemsize
            )
            if (
                path.parent != self.root
                or not path.is_file()
                or path.stat().st_size != expected_bytes
                or spec.get("bytes") != expected_bytes
                or (verify_checksums and _checksum(path) != spec.get("blake3"))
            ):
                raise O1RankingAfterstateError(
                    f"{split} afterstate tensor failed integrity: {name}"
                )
            shape = (
                (groups, COHORT_WIDTH)
                if name == "records"
                else manifest_shape
            )
            loaded[name] = np.memmap(
                path,
                mode="r",
                dtype=memmap_dtype,
                shape=shape,
            )
        if (
            not np.array_equal(loaded["group_ids"], cohort.tensors["group_ids"])
            or not np.array_equal(
                loaded["source_candidate_indices"],
                cohort.tensors["source_candidate_indices"],
            )
            or not np.array_equal(
                loaded["action_hashes"],
                cohort.tensors["action_hashes"],
            )
        ):
            raise O1RankingAfterstateError(f"{split} afterstates do not align with cohort")
        self._verify_records(
            split,
            loaded["records"],
            loaded["model_input_hashes"],
            verify_model_inputs=verify_model_inputs,
        )
        return O1RankingAfterstateSplit(
            name=split,
            groups=groups,
            group_ids=loaded["group_ids"],
            source_candidate_indices=loaded["source_candidate_indices"],
            action_hashes=loaded["action_hashes"],
            records=loaded["records"],
            model_input_hashes=loaded["model_input_hashes"],
        )

    def _verify_records(
        self,
        split: str,
        records: np.memmap,
        expected_hashes: np.memmap,
        *,
        verify_model_inputs: bool,
    ) -> None:
        cohort = self.cohort.split(split)
        flat = records.reshape(-1)
        expected_turns = np.repeat(
            np.asarray(cohort.tensors["turns"], dtype=np.uint8),
            COHORT_WIDTH,
        )
        expected_seats = np.repeat(
            np.asarray(cohort.tensors["current_players"], dtype=np.uint8),
            COHORT_WIDTH,
        )
        history_count = flat["history_count"].astype(np.int64)
        history = flat["history"]
        if (
            np.any(flat["focal_turn"] != expected_turns)
            or np.any(flat["focal_seat"] != expected_seats)
            or np.any(flat["position"]["turn"] != expected_turns + 1)
            or np.any(flat["position"]["active_seat"] != expected_seats)
            or np.any(flat["seat_policy_codes"] != 255)
            or np.any(flat["opponent_targets"] != np.zeros((), flat["opponent_targets"].dtype))
            or np.any(flat["survival_targets"] != np.zeros((), flat["survival_targets"].dtype))
            or np.any(flat["final_scores"] != 0)
            or np.any((history_count < 1) | (history_count > 12))
        ):
            raise O1RankingAfterstateError(f"{split} afterstate record contract drifted")
        for index, count in enumerate(history_count):
            active = history[index, :count]
            padding = history[index, count:]
            if (
                np.any(active["valid"] != 1)
                or not np.array_equal(
                    active["age"],
                    np.arange(count - 1, -1, -1, dtype=np.uint8),
                )
                or active[-1]["relative_seat"] != 0
                or np.any(padding["valid"] != 0)
            ):
                raise O1RankingAfterstateError(
                    f"{split} afterstate history drifted at candidate {index}"
                )
        if verify_model_inputs:
            observed = np.empty((len(flat), 32), dtype=np.uint8)
            for index, record in enumerate(flat):
                observed[index] = np.frombuffer(
                    model_input_blake3(record),
                    dtype=np.uint8,
                )
            if not np.array_equal(observed.reshape(expected_hashes.shape), expected_hashes):
                raise O1RankingAfterstateError(
                    f"{split} afterstate model-input hashes drifted"
                )


def model_input_blake3(record: np.void) -> bytes:
    """Reproduce the Rust model-input identity from one structured O1 record."""
    position = np.asarray(record["position"]).copy()
    position["game_index"] = 0
    position["targets"] = 0
    digest = blake3.blake3()
    digest.update(_MODEL_INPUT_HASH_DOMAIN)
    digest.update(position.tobytes())
    digest.update(bytes([int(record["history_count"])]))
    digest.update(np.asarray(record["history"]).tobytes())
    return digest.digest()


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
        raise O1RankingAfterstateError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise O1RankingAfterstateError(f"{label} must be a JSON object")
    return value
