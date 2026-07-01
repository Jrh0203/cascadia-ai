"""Public-refill-marginalized O1 probability features for ADR 0188."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import numpy as np

from cascadia_mlx.o1_ranking_afterstate_cache import (
    O1RankingAfterstateCache,
)
from cascadia_mlx.o1_ranking_cohort import (
    ADR_ID,
    COHORT_WIDTH,
    EXPERIMENT_ID,
    PROTOCOL_ID,
    O1RankingCohortCache,
)
from cascadia_mlx.opponent_intent_dataset import (
    OPPONENT_INTENT_RECORD_DTYPE,
    decode_opponent_intent_inputs,
)
from cascadia_mlx.opponent_intent_experiment import load_final_model
from cascadia_mlx.opponent_intent_model import (
    OpponentIntentPrediction,
    OpponentIntentSurvivalModel,
)
from cascadia_mlx.s1_exact_supply_mlx_cache import (
    ARCHETYPE_COUNT,
    S1ExactSupplyCache,
)

INTENT_CACHE_SCHEMA_VERSION = 1
INTENT_CACHE_SCHEMA = "o1-ranking-public-refill-intent-cache-v1"
INTENT_FEATURE_DIM = 81
REFILL_PROPOSALS = 8
INFERENCE_CANDIDATE_BATCH = 128
REFILL_SEED_DOMAIN = b"cascadia-v2-o1-ranking-public-refill-v1"
SHUFFLE_SEED_DOMAIN = b"cascadia-v2-o1-ranking-stratified-shuffle-v1"
FEATURE_HASH_DOMAIN = b"cascadia-v2-o1-ranking-intent-feature-v1"

ARMS = (
    "z0-zero-intent",
    "b1-a0-public-state",
    "p2-a2-history-auxiliary",
    "s3-a2-stratified-shuffle",
)
ZERO_ARM, PUBLIC_STATE_ARM, HISTORY_ARM, SHUFFLE_ARM = ARMS

_TERRAIN_INDEX = {
    "Mountain": 0,
    "Forest": 1,
    "Prairie": 2,
    "Wetland": 3,
    "River": 4,
}
_DTYPES = {
    "|u1": np.dtype("u1"),
    "<u2": np.dtype("<u2"),
    "<u4": np.dtype("<u4"),
    "<u8": np.dtype("<u8"),
    "<f4": np.dtype("<f4"),
}
_TENSOR_CONTRACT = {
    "group_ids": ("<u8", lambda groups: (groups,)),
    "source_candidate_indices": ("<u2", lambda groups: (groups, COHORT_WIDTH)),
    "action_hashes": ("|u1", lambda groups: (groups, COHORT_WIDTH, 32)),
    "proposal_archetypes": (
        "|u1",
        lambda groups: (groups, COHORT_WIDTH, REFILL_PROPOSALS),
    ),
    "proposal_wildlife": (
        "|u1",
        lambda groups: (groups, COHORT_WIDTH, REFILL_PROPOSALS),
    ),
    "a0_features": (
        "<f4",
        lambda groups: (groups, COHORT_WIDTH, INTENT_FEATURE_DIM),
    ),
    "a2_features": (
        "<f4",
        lambda groups: (groups, COHORT_WIDTH, INTENT_FEATURE_DIM),
    ),
    "shuffle_strata": ("|u1", lambda groups: (groups, COHORT_WIDTH, 3)),
    "shuffle_source_indices": ("<u4", lambda groups: (groups, COHORT_WIDTH)),
    "candidate_feature_hashes": ("|u1", lambda groups: (groups, COHORT_WIDTH, 32)),
}


class O1RankingIntentCacheError(ValueError):
    """The public-refill O1 feature cache is invalid or misaligned."""


@dataclass(frozen=True)
class O1RankingIntentSplit:
    """Memory-mapped intent surfaces for one split."""

    name: str
    groups: int
    tensors: dict[str, np.memmap]


class O1RankingIntentCache:
    """Aligned A0, A2, zero, and shuffled O1 candidate features."""

    def __init__(
        self,
        root: str | Path,
        *,
        cohort: O1RankingCohortCache,
        afterstates: O1RankingAfterstateCache,
        verify_checksums: bool = True,
        verify_semantics: bool = True,
        require_complete: bool = True,
    ):
        self.root = Path(root)
        self.cohort = cohort
        self.afterstates = afterstates
        self.manifest = _read_json(self.root / "cache.json", "O1 ranking intent manifest")
        self._validate_envelope(require_complete=require_complete)
        self.splits = {
            split: self._load_split(
                split,
                verify_checksums=verify_checksums,
                verify_semantics=verify_semantics,
            )
            for split in self.manifest["splits"]
        }

    @property
    def cache_id(self) -> str:
        return str(self.manifest["cache_id"])

    def split(self, name: str) -> O1RankingIntentSplit:
        try:
            return self.splits[name]
        except KeyError as error:
            raise O1RankingIntentCacheError(f"intent split is absent: {name}") from error

    def arm_features(
        self,
        split: str,
        arm: str,
        rows: np.ndarray,
    ) -> np.ndarray:
        """Materialize one arm's exact `[groups, 64, 81]` feature tensor."""
        if arm not in ARMS:
            raise ValueError(f"unknown O1 ranking arm: {arm}")
        source = self.split(split)
        selected = np.asarray(rows, dtype=np.int64)
        if (
            selected.ndim != 1
            or not len(selected)
            or np.any(selected < 0)
            or np.any(selected >= source.groups)
        ):
            raise IndexError("O1 intent rows must be nonempty and in range")
        if arm == ZERO_ARM:
            return np.zeros(
                (len(selected), COHORT_WIDTH, INTENT_FEATURE_DIM),
                dtype=np.float32,
            )
        if arm == PUBLIC_STATE_ARM:
            return np.asarray(source.tensors["a0_features"][selected]).copy()
        if arm == HISTORY_ARM:
            return np.asarray(source.tensors["a2_features"][selected]).copy()
        flat = np.asarray(source.tensors["a2_features"]).reshape(
            source.groups * COHORT_WIDTH,
            INTENT_FEATURE_DIM,
        )
        donors = np.asarray(
            source.tensors["shuffle_source_indices"][selected],
            dtype=np.int64,
        )
        return flat[donors]

    def _validate_envelope(self, *, require_complete: bool) -> None:
        manifest = self.manifest
        identity = manifest.get("scientific_identity")
        if (
            manifest.get("schema_version") != INTENT_CACHE_SCHEMA_VERSION
            or manifest.get("cache_schema") != INTENT_CACHE_SCHEMA
            or manifest.get("experiment_id") != EXPERIMENT_ID
            or manifest.get("protocol_id") != PROTOCOL_ID
            or manifest.get("adr") != ADR_ID
            or manifest.get("cohort_id") != self.cohort.cache_id
            or manifest.get("afterstate_cache_id") != self.afterstates.cache_id
            or not isinstance(identity, dict)
            or _canonical_blake3(identity) != manifest.get("cache_id")
            or self.root.name != manifest.get("cache_id")
        ):
            raise O1RankingIntentCacheError("invalid O1 ranking intent-cache envelope")
        if require_complete and manifest.get("complete_open_corpus") is not True:
            raise O1RankingIntentCacheError("production intent cache must cover open corpus")
        if manifest.get("sampler") != {
            "domain": REFILL_SEED_DOMAIN.decode(),
            "proposals": REFILL_PROPOSALS,
            "tile_distribution": "exact-public-archetype-counts-divided-by-unseen",
            "wildlife_distribution": "candidate-staged-public-bag-counts",
            "without_replacement_within_kind": True,
            "fills_candidate_depletions_only": True,
        }:
            raise O1RankingIntentCacheError("public refill sampler contract drifted")
        if manifest.get("feature_order") != _feature_order():
            raise O1RankingIntentCacheError("O1 intent feature order drifted")
        if set(manifest.get("splits", {})) != set(self.cohort.splits):
            raise O1RankingIntentCacheError("intent and cohort split sets differ")

    def _load_split(
        self,
        split: str,
        *,
        verify_checksums: bool,
        verify_semantics: bool,
    ) -> O1RankingIntentSplit:
        raw = self.manifest["splits"][split]
        cohort = self.cohort.split(split)
        groups = int(raw.get("groups", -1))
        if (
            groups <= 0
            or groups > cohort.groups
            or raw.get("candidates") != groups * COHORT_WIDTH
            or raw.get("complete_open_split") != (groups == cohort.groups)
            or raw.get("cohort_split_blake3")
            != _canonical_blake3(self.cohort.manifest["splits"][split])
        ):
            raise O1RankingIntentCacheError(f"{split} intent split identity drifted")
        files = raw.get("files")
        if not isinstance(files, dict) or set(files) != set(_TENSOR_CONTRACT):
            raise O1RankingIntentCacheError(f"{split} intent tensor set drifted")
        tensors: dict[str, np.memmap] = {}
        for name, (dtype_code, shape_factory) in _TENSOR_CONTRACT.items():
            shape = shape_factory(groups)
            spec = files[name]
            if (
                not isinstance(spec, dict)
                or spec.get("dtype") != dtype_code
                or spec.get("shape") != list(shape)
            ):
                raise O1RankingIntentCacheError(f"{split} intent shape drifted: {name}")
            path = self.root / str(spec.get("file"))
            dtype = _DTYPES[dtype_code]
            expected_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
            if (
                path.parent != self.root
                or not path.is_file()
                or path.stat().st_size != expected_bytes
                or spec.get("bytes") != expected_bytes
                or (verify_checksums and _checksum(path) != spec.get("blake3"))
            ):
                raise O1RankingIntentCacheError(
                    f"{split} intent tensor failed integrity: {name}"
                )
            tensors[name] = np.memmap(path, mode="r", dtype=dtype, shape=shape)
        if (
            not np.array_equal(
                tensors["group_ids"],
                cohort.tensors["group_ids"][:groups],
            )
            or not np.array_equal(
                tensors["source_candidate_indices"],
                cohort.tensors["source_candidate_indices"][:groups],
            )
            or not np.array_equal(
                tensors["action_hashes"],
                cohort.tensors["action_hashes"][:groups],
            )
        ):
            raise O1RankingIntentCacheError(f"{split} intent cache does not align with cohort")
        if verify_semantics:
            _verify_split_semantics(split, tensors)
        return O1RankingIntentSplit(name=split, groups=groups, tensors=tensors)


def deterministic_weighted_draw(
    counts: np.ndarray,
    *,
    split: str,
    group_id: int,
    action_hash: bytes,
    proposal_index: int,
    kind: str,
    draw_index: int = 0,
) -> int:
    """Draw from nonnegative integer counts with a stable BLAKE3 uniform."""
    weights = np.asarray(counts, dtype=np.int64)
    total = int(weights.sum())
    if (
        weights.ndim != 1
        or not len(weights)
        or np.any(weights < 0)
        or total <= 0
        or len(action_hash) != 32
        or proposal_index < 0
        or draw_index < 0
    ):
        raise ValueError("deterministic refill draw received invalid counts or identity")
    digest = blake3.blake3()
    digest.update(REFILL_SEED_DOMAIN)
    digest.update(split.encode())
    digest.update(int(group_id).to_bytes(8, "little"))
    digest.update(action_hash)
    digest.update(int(proposal_index).to_bytes(4, "little"))
    digest.update(kind.encode())
    digest.update(int(draw_index).to_bytes(4, "little"))
    uniform = int.from_bytes(digest.digest(length=8), "little")
    target = (uniform * total) >> 64
    cumulative = np.cumsum(weights, dtype=np.int64)
    return int(np.searchsorted(cumulative, target, side="right"))


def public_refill_proposals(
    record: np.void,
    *,
    split: str,
    group_id: int,
    action_hash: bytes,
    tile_counts: np.ndarray,
    wildlife_counts: np.ndarray,
    catalog_entities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fill only the candidate's missing tile and wildlife components."""
    history_count = int(record["history_count"])
    if history_count <= 0:
        raise O1RankingIntentCacheError("candidate afterstate has no age-zero action")
    action = record["history"][history_count - 1]["action"]
    tile_slot = int(action["tile_slot"])
    wildlife_slot = int(action["wildlife_slot"])
    if (
        record["position"]["market_entities"][tile_slot, 0] != 255
        or record["position"]["market_entities"][wildlife_slot, 3] != 255
    ):
        raise O1RankingIntentCacheError("candidate afterstate depletion disagrees with action")
    expanded = np.repeat(
        np.asarray(record, dtype=OPPONENT_INTENT_RECORD_DTYPE)[None],
        REFILL_PROPOSALS,
        axis=0,
    )
    archetypes = np.empty(REFILL_PROPOSALS, dtype=np.uint8)
    wildlife = np.empty(REFILL_PROPOSALS, dtype=np.uint8)
    for proposal in range(REFILL_PROPOSALS):
        archetype = deterministic_weighted_draw(
            tile_counts,
            split=split,
            group_id=group_id,
            action_hash=action_hash,
            proposal_index=proposal,
            kind="tile",
        )
        wildlife_type = deterministic_weighted_draw(
            wildlife_counts,
            split=split,
            group_id=group_id,
            action_hash=action_hash,
            proposal_index=proposal,
            kind="wildlife",
        )
        market = expanded[proposal]["position"]["market_entities"]
        market[tile_slot, [0, 1, 2, 4]] = catalog_entities[archetype]
        market[tile_slot, 5:] = 0
        market[wildlife_slot, 3] = wildlife_type
        market[wildlife_slot, 5:] = 0
        archetypes[proposal] = archetype
        wildlife[proposal] = wildlife_type
    return expanded, archetypes, wildlife


def prediction_to_intent_vector(
    prediction: OpponentIntentPrediction,
) -> mx.array:
    """Flatten probabilities in the frozen 81-value ADR 0188 order."""
    probabilities = (
        mx.softmax(prediction.disposition_logits, axis=-1),
        mx.softmax(prediction.pair_survival_logits, axis=-1)[..., 1],
        mx.softmax(prediction.final_slot_logits, axis=-1),
        mx.softmax(prediction.tile_slot_logits, axis=-1),
        mx.softmax(prediction.wildlife_slot_logits, axis=-1),
        mx.softmax(prediction.draft_kind_logits, axis=-1)[..., 1],
        mx.softmax(prediction.drafted_wildlife_logits, axis=-1),
        mx.softmax(prediction.replace_three_logits, axis=-1)[..., 1],
    )
    batch = prediction.disposition_logits.shape[0]
    vector = mx.concatenate(
        [value.reshape(batch, -1) for value in probabilities],
        axis=-1,
    )
    if vector.shape != (batch, INTENT_FEATURE_DIM):
        raise AssertionError("O1 intent vector width drifted")
    return vector


def stratified_derangement(
    *,
    split: str,
    group_ids: np.ndarray,
    action_hashes: np.ndarray,
    raw_strata: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Preserve effective strata and marginals while destroying alignment."""
    groups = np.asarray(group_ids, dtype=np.uint64)
    hashes = np.asarray(action_hashes, dtype=np.uint8)
    strata = np.asarray(raw_strata, dtype=np.uint8).copy()
    if (
        groups.ndim != 1
        or hashes.shape != (len(groups), 32)
        or strata.shape != (len(groups), 3)
    ):
        raise ValueError("stratified shuffle inputs do not align")
    for draft_kind in np.unique(strata[:, 1]):
        for depletion in np.unique(strata[strata[:, 1] == draft_kind, 2]):
            category = np.flatnonzero(
                (strata[:, 1] == draft_kind) & (strata[:, 2] == depletion)
            )
            if len(category) < 2:
                raise O1RankingIntentCacheError(
                    "shuffle draft/depletion category cannot be deranged"
                )
            phase_members = {
                phase: category[strata[category, 0] == phase]
                for phase in np.unique(strata[category, 0])
            }
            stable_phases = [
                int(phase)
                for phase, members in phase_members.items()
                if len(members) >= 2
            ]
            if not stable_phases:
                target_phase = int(np.min(strata[category, 0]))
                strata[category, 0] = target_phase
                continue
            for phase, members in phase_members.items():
                if len(members) != 1:
                    continue
                target_phase = min(
                    stable_phases,
                    key=lambda candidate: (abs(candidate - int(phase)), candidate),
                )
                strata[members, 0] = target_phase

    donors = np.empty(len(groups), dtype=np.uint32)
    grouped: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index, stratum in enumerate(strata):
        grouped[tuple(int(value) for value in stratum)].append(index)
    for stratum, members in grouped.items():
        if len(members) < 2:
            raise O1RankingIntentCacheError(
                f"effective shuffle stratum remains singleton: {stratum}"
            )
        ordered = sorted(
            members,
            key=lambda index: _shuffle_key(
                split,
                int(groups[index]),
                bytes(hashes[index]),
            ),
        )
        for position, target in enumerate(ordered):
            donors[target] = ordered[(position + 1) % len(ordered)]
    if (
        np.any(donors == np.arange(len(donors), dtype=np.uint32))
        or len(np.unique(donors)) != len(donors)
        or not np.array_equal(strata[donors], strata)
    ):
        raise O1RankingIntentCacheError("stratified shuffle is not a valid derangement")
    return donors, strata


def build_o1_ranking_intent_cache(
    *,
    cohort_root: Path,
    afterstate_root: Path,
    s1_cache_root: Path,
    a0_report_path: Path,
    a0_model_path: Path,
    a2_report_path: Path,
    a2_model_path: Path,
    output_root: Path,
    receipt_path: Path,
    maximum_groups_per_split: int | None = None,
) -> dict[str, Any]:
    """Generate immutable public-refill proposal and O1 probability tensors."""
    if maximum_groups_per_split is not None and maximum_groups_per_split <= 0:
        raise ValueError("maximum_groups_per_split must be positive")
    production = maximum_groups_per_split is None
    mx.set_default_device(mx.gpu)
    cohort = O1RankingCohortCache(
        cohort_root,
        verify_checksums=True,
        require_complete=True,
    )
    afterstates = O1RankingAfterstateCache(
        afterstate_root,
        cohort=cohort,
        verify_checksums=True,
        verify_model_inputs=True,
        require_complete=True,
    )
    s1 = S1ExactSupplyCache(
        s1_cache_root,
        verify_checksums=True,
        verify_semantics=True,
        require_complete=True,
    )
    expected_s1 = cohort.manifest["open_data_verification"]["s1_cache_id"]
    if s1.cache_id != expected_s1:
        raise O1RankingIntentCacheError("S1 cache differs from the frozen cohort")
    a0_report = _read_json(a0_report_path, "A0 final report")
    a2_report = _read_json(a2_report_path, "A2 final report")
    if (
        a0_report.get("arm") != "a0-public-state"
        or a2_report.get("arm") != "a2-next-draft-auxiliary"
    ):
        raise O1RankingIntentCacheError("O1 reports do not identify the frozen A0 and A2 arms")
    a0 = load_final_model(a0_report, a0_model_path)
    a2 = load_final_model(a2_report, a2_model_path)
    model_identity = {
        "a0": _model_identity(a0_report, a0_report_path, a0_model_path),
        "a2": _model_identity(a2_report, a2_report_path, a2_model_path),
    }
    catalog_entities = _catalog_market_entities(s1.manifest["catalog"])

    output_root.mkdir(parents=True, exist_ok=True)
    temporary = output_root / f".tmp-o1-ranking-intent-{os.getpid()}-{time.time_ns()}"
    temporary.mkdir()
    try:
        split_manifests = {
            split: _build_split(
                split=split,
                cohort=cohort,
                afterstates=afterstates,
                s1=s1,
                a0=a0,
                a2=a2,
                catalog_entities=catalog_entities,
                root=temporary,
                maximum_groups=maximum_groups_per_split,
            )
            for split in cohort.splits
        }
        complete_open_corpus = production and all(
            split_manifests[name]["groups"] == cohort.splits[name].groups
            for name in cohort.splits
        )
        sampler = {
            "domain": REFILL_SEED_DOMAIN.decode(),
            "proposals": REFILL_PROPOSALS,
            "tile_distribution": "exact-public-archetype-counts-divided-by-unseen",
            "wildlife_distribution": "candidate-staged-public-bag-counts",
            "without_replacement_within_kind": True,
            "fills_candidate_depletions_only": True,
        }
        hidden_information = {
            "public_afterstates_only": True,
            "public_supply_only": True,
            "champion_history_only": True,
            "realized_post_draft_refill_read": False,
            "hidden_stack_order_read": False,
            "hidden_bag_order_read": False,
            "policy_identity_read": False,
            "sealed_test_opened": False,
            "gameplay_run": False,
        }
        implementation = {
            path.name: _checksum(path)
            for path in (
                Path(__file__),
                Path(__file__).with_name("o1_ranking_afterstate_cache.py"),
                Path(__file__).with_name("opponent_intent_dataset.py"),
                Path(__file__).with_name("opponent_intent_model.py"),
            )
        }
        scientific_identity = {
            "schema_version": INTENT_CACHE_SCHEMA_VERSION,
            "cache_schema": INTENT_CACHE_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "protocol_id": PROTOCOL_ID,
            "adr": ADR_ID,
            "complete_open_corpus": complete_open_corpus,
            "cohort_id": cohort.cache_id,
            "cohort_manifest_blake3": _checksum(cohort.root / "cache.json"),
            "afterstate_cache_id": afterstates.cache_id,
            "afterstate_manifest_blake3": _checksum(afterstates.root / "cache.json"),
            "s1_cache_id": s1.cache_id,
            "s1_manifest_blake3": _checksum(s1.manifest_path),
            "models": model_identity,
            "sampler": sampler,
            "feature_order": _feature_order(),
            "shuffle": {
                "domain": SHUFFLE_SEED_DOMAIN.decode(),
                "strata": [
                    "phase-quartile-with-singletons-merged-adjacently",
                    "candidate-draft-kind",
                    "single-slot-versus-split-slot-depletion",
                ],
                "fixed_points": 0,
            },
            "implementation": implementation,
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
            existing = _read_json(final_root / "cache.json", "existing intent cache")
            if existing != manifest:
                raise O1RankingIntentCacheError(
                    f"intent-cache content-address collision at {final_root}"
                )
            shutil.rmtree(temporary)
        else:
            os.replace(temporary, final_root)
        receipt = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "cache_id": cache_id,
            "cache_root": str(final_root.resolve()),
            "cohort_id": cohort.cache_id,
            "afterstate_cache_id": afterstates.cache_id,
            "complete_open_corpus": complete_open_corpus,
            "train_groups": split_manifests["train"]["groups"],
            "validation_groups": split_manifests["validation"]["groups"],
            "candidates": sum(
                int(split_manifest["candidates"])
                for split_manifest in split_manifests.values()
            ),
        }
        _write_json_atomic(receipt_path, receipt)
        O1RankingIntentCache(
            final_root,
            cohort=cohort,
            afterstates=afterstates,
            verify_checksums=True,
            verify_semantics=True,
            require_complete=production,
        )
        return receipt
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _build_split(
    *,
    split: str,
    cohort: O1RankingCohortCache,
    afterstates: O1RankingAfterstateCache,
    s1: S1ExactSupplyCache,
    a0: OpponentIntentSurvivalModel,
    a2: OpponentIntentSurvivalModel,
    catalog_entities: np.ndarray,
    root: Path,
    maximum_groups: int | None,
) -> dict[str, Any]:
    cohort_split = cohort.split(split)
    afterstate_split = afterstates.split(split)
    s1_split = s1.splits[split]
    groups = cohort_split.groups if maximum_groups is None else min(
        cohort_split.groups,
        maximum_groups,
    )
    candidate_count = groups * COHORT_WIDTH
    group_ids = np.asarray(cohort_split.tensors["group_ids"][:groups]).copy()
    sources = np.asarray(
        cohort_split.tensors["source_candidate_indices"][:groups]
    ).copy()
    action_hashes = np.asarray(cohort_split.tensors["action_hashes"][:groups]).copy()
    records = np.asarray(afterstate_split.records[:groups])
    proposal_archetypes = np.empty(
        (groups, COHORT_WIDTH, REFILL_PROPOSALS),
        dtype=np.uint8,
    )
    proposal_wildlife = np.empty_like(proposal_archetypes)
    a0_features = np.empty(
        (groups, COHORT_WIDTH, INTENT_FEATURE_DIM),
        dtype=np.float32,
    )
    a2_features = np.empty_like(a0_features)
    raw_strata = np.empty((candidate_count, 3), dtype=np.uint8)
    repeated_group_ids = np.repeat(group_ids, COHORT_WIDTH)
    flat_hashes = action_hashes.reshape(candidate_count, 32)
    flat_records = records.reshape(candidate_count)
    flat_sources = sources.reshape(candidate_count)
    started = time.perf_counter()

    for start in range(0, candidate_count, INFERENCE_CANDIDATE_BATCH):
        end = min(start + INFERENCE_CANDIDATE_BATCH, candidate_count)
        proposal_records = np.empty(
            ((end - start) * REFILL_PROPOSALS,),
            dtype=OPPONENT_INTENT_RECORD_DTYPE,
        )
        for local, flat_index in enumerate(range(start, end)):
            group_row, candidate = divmod(flat_index, COHORT_WIDTH)
            group_id = int(group_ids[group_row])
            s1_row = s1_split.group_rows[group_id]
            tile_counts = np.asarray(
                s1_split.tensors["exact_supply_values"][s1_row, 5:80],
                dtype=np.int64,
            )
            offset = int(s1_split.tensors["candidate_offsets"][s1_row])
            wildlife_counts = np.asarray(
                s1_split.tensors["staged_wildlife_counts"][
                    offset + int(flat_sources[flat_index])
                ],
                dtype=np.int64,
            )
            expanded, archetypes, wildlife = public_refill_proposals(
                flat_records[flat_index],
                split=split,
                group_id=group_id,
                action_hash=bytes(flat_hashes[flat_index]),
                tile_counts=tile_counts,
                wildlife_counts=wildlife_counts,
                catalog_entities=catalog_entities,
            )
            proposal_records[
                local * REFILL_PROPOSALS : (local + 1) * REFILL_PROPOSALS
            ] = expanded
            proposal_archetypes[group_row, candidate] = archetypes
            proposal_wildlife[group_row, candidate] = wildlife
            history_count = int(flat_records[flat_index]["history_count"])
            action = flat_records[flat_index]["history"][history_count - 1]["action"]
            raw_strata[flat_index] = [
                min(int(flat_records[flat_index]["focal_turn"]) // 20, 3),
                int(action["draft_kind"]),
                int(action["tile_slot"] != action["wildlife_slot"]),
            ]
        inputs = decode_opponent_intent_inputs(proposal_records)
        vectors = []
        for model in (a0, a2):
            model.eval()
            vector = prediction_to_intent_vector(model(inputs))
            mx.eval(vector)
            values = np.asarray(vector, dtype=np.float32).reshape(
                end - start,
                REFILL_PROPOSALS,
                INTENT_FEATURE_DIM,
            )
            vectors.append(values.mean(axis=1, dtype=np.float32))
        a0_features.reshape(candidate_count, INTENT_FEATURE_DIM)[start:end] = vectors[0]
        a2_features.reshape(candidate_count, INTENT_FEATURE_DIM)[start:end] = vectors[1]
        if end % (INFERENCE_CANDIDATE_BATCH * 20) == 0 or end == candidate_count:
            print(
                json.dumps(
                    {
                        "event": "o1-ranking-intent-progress",
                        "split": split,
                        "candidates": end,
                        "total_candidates": candidate_count,
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            mx.clear_cache()

    donors, effective_strata = stratified_derangement(
        split=split,
        group_ids=repeated_group_ids,
        action_hashes=flat_hashes,
        raw_strata=raw_strata,
    )
    feature_hashes = np.empty((candidate_count, 32), dtype=np.uint8)
    flat_archetypes = proposal_archetypes.reshape(candidate_count, REFILL_PROPOSALS)
    flat_wildlife = proposal_wildlife.reshape(candidate_count, REFILL_PROPOSALS)
    flat_a0 = a0_features.reshape(candidate_count, INTENT_FEATURE_DIM)
    flat_a2 = a2_features.reshape(candidate_count, INTENT_FEATURE_DIM)
    for index in range(candidate_count):
        digest = blake3.blake3()
        digest.update(FEATURE_HASH_DOMAIN)
        digest.update(split.encode())
        digest.update(int(repeated_group_ids[index]).to_bytes(8, "little"))
        digest.update(bytes(flat_hashes[index]))
        digest.update(flat_archetypes[index].tobytes())
        digest.update(flat_wildlife[index].tobytes())
        digest.update(flat_a0[index].astype("<f4", copy=False).tobytes())
        digest.update(flat_a2[index].astype("<f4", copy=False).tobytes())
        digest.update(effective_strata[index].tobytes())
        digest.update(int(donors[index]).to_bytes(4, "little"))
        feature_hashes[index] = np.frombuffer(digest.digest(), dtype=np.uint8)

    arrays = {
        "group_ids": group_ids.astype("<u8", copy=False),
        "source_candidate_indices": sources.astype("<u2", copy=False),
        "action_hashes": action_hashes.astype("u1", copy=False),
        "proposal_archetypes": proposal_archetypes,
        "proposal_wildlife": proposal_wildlife,
        "a0_features": a0_features.astype("<f4", copy=False),
        "a2_features": a2_features.astype("<f4", copy=False),
        "shuffle_strata": effective_strata.reshape(groups, COHORT_WIDTH, 3),
        "shuffle_source_indices": donors.reshape(groups, COHORT_WIDTH).astype(
            "<u4",
            copy=False,
        ),
        "candidate_feature_hashes": feature_hashes.reshape(groups, COHORT_WIDTH, 32),
    }
    files = {
        name: _write_tensor(root, f"{split}-{name.replace('_', '-')}.bin", values)
        for name, values in arrays.items()
    }
    return {
        "split": split,
        "groups": groups,
        "candidates": candidate_count,
        "complete_open_split": groups == cohort_split.groups,
        "cohort_split_blake3": _canonical_blake3(cohort.manifest["splits"][split]),
        "fixed_points": int(
            np.sum(donors == np.arange(candidate_count, dtype=np.uint32))
        ),
        "files": files,
    }


def _verify_split_semantics(
    split: str,
    tensors: dict[str, np.memmap],
) -> None:
    a0 = np.asarray(tensors["a0_features"], dtype=np.float32)
    a2 = np.asarray(tensors["a2_features"], dtype=np.float32)
    for name, values in (("a0", a0), ("a2", a2)):
        if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
            raise O1RankingIntentCacheError(f"{split} {name} features are invalid")
        _verify_probability_segments(values, f"{split} {name}")
    donors = np.asarray(tensors["shuffle_source_indices"], dtype=np.int64).reshape(-1)
    strata = np.asarray(tensors["shuffle_strata"], dtype=np.uint8).reshape(-1, 3)
    count = len(donors)
    if (
        np.any(donors < 0)
        or np.any(donors >= count)
        or np.any(donors == np.arange(count))
        or len(np.unique(donors)) != count
        or not np.array_equal(strata[donors], strata)
    ):
        raise O1RankingIntentCacheError(f"{split} shuffled A2 mapping is invalid")
    archetypes = np.asarray(tensors["proposal_archetypes"], dtype=np.uint8)
    wildlife = np.asarray(tensors["proposal_wildlife"], dtype=np.uint8)
    if np.any(archetypes >= ARCHETYPE_COUNT) or np.any(wildlife >= 5):
        raise O1RankingIntentCacheError(f"{split} refill proposals are out of range")
    expected_hashes = np.asarray(tensors["candidate_feature_hashes"], dtype=np.uint8).reshape(
        count,
        32,
    )
    group_ids = np.repeat(
        np.asarray(tensors["group_ids"], dtype=np.uint64),
        COHORT_WIDTH,
    )
    action_hashes = np.asarray(tensors["action_hashes"], dtype=np.uint8).reshape(count, 32)
    flat_archetypes = archetypes.reshape(count, REFILL_PROPOSALS)
    flat_wildlife = wildlife.reshape(count, REFILL_PROPOSALS)
    flat_a0 = a0.reshape(count, INTENT_FEATURE_DIM)
    flat_a2 = a2.reshape(count, INTENT_FEATURE_DIM)
    observed = np.empty_like(expected_hashes)
    for index in range(count):
        digest = blake3.blake3()
        digest.update(FEATURE_HASH_DOMAIN)
        digest.update(split.encode())
        digest.update(int(group_ids[index]).to_bytes(8, "little"))
        digest.update(bytes(action_hashes[index]))
        digest.update(flat_archetypes[index].tobytes())
        digest.update(flat_wildlife[index].tobytes())
        digest.update(flat_a0[index].astype("<f4", copy=False).tobytes())
        digest.update(flat_a2[index].astype("<f4", copy=False).tobytes())
        digest.update(strata[index].tobytes())
        digest.update(int(donors[index]).to_bytes(4, "little"))
        observed[index] = np.frombuffer(digest.digest(), dtype=np.uint8)
    if not np.array_equal(observed, expected_hashes):
        raise O1RankingIntentCacheError(f"{split} candidate feature hashes drifted")


def _verify_probability_segments(values: np.ndarray, label: str) -> None:
    flat = values.reshape(-1, INTENT_FEATURE_DIM)
    distributions = (
        flat[:, 0:16].reshape(-1, 4),
        flat[:, 20:36].reshape(-1, 4),
        flat[:, 36:48].reshape(-1, 4),
        flat[:, 48:60].reshape(-1, 4),
        flat[:, 63:78].reshape(-1, 5),
    )
    for distribution in distributions:
        if not np.allclose(distribution.sum(axis=-1), 1.0, atol=2e-5):
            raise O1RankingIntentCacheError(f"{label} probability segment does not sum to one")


def _catalog_market_entities(catalog: list[dict[str, Any]]) -> np.ndarray:
    if len(catalog) != ARCHETYPE_COUNT:
        raise O1RankingIntentCacheError("S1 semantic catalog width drifted")
    entities = np.empty((ARCHETYPE_COUNT, 4), dtype=np.uint8)
    for expected_id, entry in enumerate(catalog):
        archetype = entry.get("archetype")
        if entry.get("id") != expected_id or not isinstance(archetype, dict):
            raise O1RankingIntentCacheError("S1 catalog IDs are not canonical")
        primary = archetype.get("primary_terrain")
        secondary = archetype.get("secondary_terrain")
        wildlife = int(archetype.get("wildlife", -1))
        if (
            primary not in _TERRAIN_INDEX
            or (secondary is not None and secondary not in _TERRAIN_INDEX)
            or wildlife < 0
            or wildlife > 0b1_1111
        ):
            raise O1RankingIntentCacheError("S1 catalog archetype is malformed")
        entities[expected_id] = [
            _TERRAIN_INDEX[primary],
            255 if secondary is None else _TERRAIN_INDEX[secondary],
            wildlife,
            int(bool(archetype.get("keystone"))),
        ]
    return entities


def _model_identity(
    report: dict[str, Any],
    report_path: Path,
    model_path: Path,
) -> dict[str, Any]:
    return {
        "arm": report["arm"],
        "report_id": report["report_id"],
        "report_blake3": _checksum(report_path),
        "model_blake3": _checksum(model_path),
        "parameter_tensor_blake3": report["model"]["final_parameter_tensor_blake3"],
        "config": report["model"]["config"],
    }


def _feature_order() -> list[dict[str, Any]]:
    return [
        {"name": "disposition", "shape": [4, 4], "width": 16},
        {"name": "pair_survival_positive", "shape": [4], "width": 4},
        {"name": "final_slot", "shape": [4, 4], "width": 16},
        {"name": "opponent_tile_slot", "shape": [3, 4], "width": 12},
        {"name": "opponent_wildlife_slot", "shape": [3, 4], "width": 12},
        {"name": "opponent_independent_draft", "shape": [3], "width": 3},
        {"name": "opponent_drafted_wildlife", "shape": [3, 5], "width": 15},
        {"name": "opponent_free_replacement", "shape": [3], "width": 3},
    ]


def _shuffle_key(split: str, group_id: int, action_hash: bytes) -> bytes:
    digest = blake3.blake3()
    digest.update(SHUFFLE_SEED_DOMAIN)
    digest.update(split.encode())
    digest.update(int(group_id).to_bytes(8, "little"))
    digest.update(action_hash)
    return digest.digest()


def _write_tensor(root: Path, file_name: str, values: np.ndarray) -> dict[str, Any]:
    path = root / file_name
    array = np.ascontiguousarray(values)
    with path.open("wb") as handle:
        handle.write(array.tobytes(order="C"))
        handle.flush()
        os.fsync(handle.fileno())
    dtype = array.dtype.str
    if dtype not in _DTYPES:
        raise O1RankingIntentCacheError(f"unsupported intent tensor dtype: {dtype}")
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
        raise O1RankingIntentCacheError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise O1RankingIntentCacheError(f"{label} must be a JSON object")
    return value


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or verify public-refill O1 candidate features"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--cohort", type=Path, required=True)
    build.add_argument("--afterstates", type=Path, required=True)
    build.add_argument("--s1-cache", type=Path, required=True)
    build.add_argument("--a0-report", type=Path, required=True)
    build.add_argument("--a0-model", type=Path, required=True)
    build.add_argument("--a2-report", type=Path, required=True)
    build.add_argument("--a2-model", type=Path, required=True)
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--receipt", type=Path, required=True)
    build.add_argument("--maximum-groups-per-split", type=int)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--cohort", type=Path, required=True)
    verify.add_argument("--afterstates", type=Path, required=True)
    verify.add_argument("--cache", type=Path, required=True)
    verify.add_argument("--allow-partial", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "build":
        result = build_o1_ranking_intent_cache(
            cohort_root=args.cohort,
            afterstate_root=args.afterstates,
            s1_cache_root=args.s1_cache,
            a0_report_path=args.a0_report,
            a0_model_path=args.a0_model,
            a2_report_path=args.a2_report,
            a2_model_path=args.a2_model,
            output_root=args.output_root,
            receipt_path=args.receipt,
            maximum_groups_per_split=args.maximum_groups_per_split,
        )
    else:
        cohort = O1RankingCohortCache(
            args.cohort,
            verify_checksums=True,
            require_complete=True,
        )
        afterstates = O1RankingAfterstateCache(
            args.afterstates,
            cohort=cohort,
            verify_checksums=True,
            verify_model_inputs=True,
            require_complete=True,
        )
        cache = O1RankingIntentCache(
            args.cache,
            cohort=cohort,
            afterstates=afterstates,
            verify_checksums=True,
            verify_semantics=True,
            require_complete=not args.allow_partial,
        )
        result = {
            "schema_version": 1,
            "cache_id": cache.cache_id,
            "cache_root": str(cache.root.resolve()),
            "splits": {
                split: {"groups": source.groups}
                for split, source in cache.splits.items()
            },
        }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
