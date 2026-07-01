"""Expected-rank supervision, caching, and evaluation for ADR 0100."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import numpy as np

from cascadia_mlx.frontier_supervision_identifiability import (
    SupervisionGroup,
    expected_normal_ranks,
    iter_supervision_groups,
    parallel_group_map,
    standard_error,
)
from cascadia_mlx.graded_oracle_dataset import (
    GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
    GRADED_ORACLE_PACKED_ACTION_LIMIT,
    GradedOracleBatch,
    GradedOracleDataset,
    rotate_graded_oracle_batch,
)
from cascadia_mlx.graded_oracle_frontier_anchor import (
    FRONTIER_ANCHORED_WIDTH,
    GRADED_SOURCE_CHAMPION_FRONTIER,
    frontier_anchored_retained_indices,
)
from cascadia_mlx.graded_oracle_identifiability import NORMAL_95
from cascadia_mlx.graded_oracle_model import (
    GradedOracleRanker,
    predict_graded_oracle_batch,
)

EXPERIMENT_ID = "complete-action-frontier-expected-rank-v1"
EXPECTED_RANK_CACHE_SCHEMA_VERSION = 1
EXPECTED_RANK_TARGET_SCALE = 64.0
EXPECTED_RANK_STUDENT_TEMPERATURE = 2.0
EXPECTED_RANK_CACHE_WORKERS = 8
_PHASE_NAMES = {0: "early", 1: "middle", 2: "late"}
_EXPECTED_DATASET_MANIFESTS = {
    "train": "7ed12c943d75a786ccd4ccbe11a6b0146aad4fe5ed40f0cbaf1d652f5ac0bb99",
    "validation": "302ceb7a57482b0fb5fb12963521be35aafc121a36f572e6b9f47def1b820a31",
}


@dataclass(frozen=True)
class ExpectedRankBatch:
    """A graded-oracle batch with immutable aligned expected-rank targets."""

    base: GradedOracleBatch
    expected_rank: mx.array
    expected_rank_mask: mx.array
    target_scale: float = EXPECTED_RANK_TARGET_SCALE
    student_temperature: float = EXPECTED_RANK_STUDENT_TEMPERATURE

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)


@dataclass
class _SliceAccumulator:
    groups: int = 0
    exact: int = 0
    confidence: int = 0
    distinguishable_groups: int = 0
    distinguishable_exact: int = 0
    regret: float = 0.0

    def add(self, observation: dict[str, bool | float]) -> None:
        self.groups += 1
        self.exact += int(observation["exact"])
        self.confidence += int(observation["confidence"])
        self.regret += float(observation["regret"])
        if bool(observation["distinguishable"]):
            self.distinguishable_groups += 1
            self.distinguishable_exact += int(observation["exact"])

    def report(self) -> dict[str, float | int | None]:
        denominator = max(self.groups, 1)
        return {
            "groups": self.groups,
            "top64_r4800_winner_recall": self.exact / denominator,
            "top64_confidence_set_coverage_95": self.confidence / denominator,
            "top64_distinguishable_winner_recall": (
                self.distinguishable_exact / self.distinguishable_groups
                if self.distinguishable_groups
                else None
            ),
            "distinguishable_groups": self.distinguishable_groups,
            "mean_top64_retained_r4800_regret": self.regret / denominator,
        }


@dataclass
class _EvaluationAccumulator:
    groups: int = 0
    candidates: int = 0
    nonfinite_scores: int = 0
    total_loss: float = 0.0
    target_slots: int = 0
    target_recalled: int = 0
    target_exact_groups: int = 0
    rank_correlations: list[float] = field(default_factory=list)
    target_entropies: list[float] = field(default_factory=list)
    frontier_counts: list[float] = field(default_factory=list)
    model: _SliceAccumulator = field(default_factory=_SliceAccumulator)
    screen: _SliceAccumulator = field(default_factory=_SliceAccumulator)
    target_ceiling: _SliceAccumulator = field(default_factory=_SliceAccumulator)
    phases: dict[str, _SliceAccumulator] = field(
        default_factory=lambda: {
            name: _SliceAccumulator() for name in _PHASE_NAMES.values()
        }
    )
    subsets: dict[str, _SliceAccumulator] = field(
        default_factory=lambda: {
            "nature_token_available": _SliceAccumulator(),
            "independent_draft_winner": _SliceAccumulator(),
        }
    )
    action_families: dict[str, _SliceAccumulator] = field(
        default_factory=lambda: {
            "paired": _SliceAccumulator(),
            "independent": _SliceAccumulator(),
            "same_slot_independent": _SliceAccumulator(),
            "free_refresh": _SliceAccumulator(),
            "paid_wipe": _SliceAccumulator(),
        }
    )
    widths: dict[str, _SliceAccumulator] = field(
        default_factory=lambda: {
            "at_most_2048": _SliceAccumulator(),
            "2049_to_4096": _SliceAccumulator(),
            "above_4096": _SliceAccumulator(),
        }
    )


class ExpectedRankTargetCache:
    """Memory-mapped target ranks aligned to one immutable dataset."""

    def __init__(
        self,
        root: str | Path,
        dataset: GradedOracleDataset,
        *,
        experiment_id: str = EXPERIMENT_ID,
        target_scale: float = EXPECTED_RANK_TARGET_SCALE,
        student_temperature: float = EXPECTED_RANK_STUDENT_TEMPERATURE,
    ):
        self.root = Path(root)
        try:
            self.manifest = json.loads((self.root / "manifest.json").read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read expected-rank cache: {error}") from error
        _validate_cache_manifest(
            self.root,
            self.manifest,
            dataset,
            experiment_id=experiment_id,
            target_scale=target_scale,
            student_temperature=student_temperature,
        )
        self.experiment_id = experiment_id
        self.target_scale = target_scale
        self.student_temperature = student_temperature
        self.group_ids = np.load(self.root / "group_ids.npy", mmap_mode="r")
        self.candidate_counts = np.load(
            self.root / "candidate_counts.npy",
            mmap_mode="r",
        )
        self.offsets = np.load(self.root / "offsets.npy", mmap_mode="r")
        self.expected_ranks = np.load(
            self.root / "expected_ranks.npy",
            mmap_mode="r",
        )
        self._group_index = {
            int(group_id): index for index, group_id in enumerate(self.group_ids)
        }
        if len(self._group_index) != len(self.group_ids):
            raise ValueError("expected-rank cache contains duplicate group IDs")

    def ranks_for_batch(self, batch: GradedOracleBatch) -> tuple[mx.array, mx.array]:
        """Return padded ranks and masks for one decoded dataset batch."""
        group_ids = np.asarray(batch.group_id)
        candidate_mask = np.asarray(batch.candidate_mask)
        ranks = np.full(candidate_mask.shape, np.nan, dtype=np.float32)
        for row, group_id in enumerate(group_ids):
            unsigned_group_id = int(group_id) & ((1 << 64) - 1)
            try:
                index = self._group_index[unsigned_group_id]
            except KeyError as error:
                raise ValueError("expected-rank cache is missing a group") from error
            count = int(np.sum(candidate_mask[row]))
            if count != int(self.candidate_counts[index]):
                raise ValueError("expected-rank cache candidate count drifted")
            start = int(self.offsets[index])
            stop = int(self.offsets[index + 1])
            ranks[row, :count] = self.expected_ranks[start:stop]
        mask = np.isfinite(ranks) & candidate_mask
        if np.any(np.sum(mask, axis=1) < 1):
            raise ValueError("expected-rank batch contains an empty target cohort")
        return mx.array(np.nan_to_num(ranks, nan=0.0)), mx.array(mask)


class ExpectedRankDataset:
    """A graded-oracle dataset paired with a verified target cache."""

    def __init__(
        self,
        root: str | Path,
        cache_root: str | Path,
        *,
        verify_checksums: bool = True,
        experiment_id: str = EXPERIMENT_ID,
        target_scale: float = EXPECTED_RANK_TARGET_SCALE,
        student_temperature: float = EXPECTED_RANK_STUDENT_TEMPERATURE,
    ):
        self.base = GradedOracleDataset(root, verify_checksums=verify_checksums)
        self.cache = ExpectedRankTargetCache(
            cache_root,
            self.base,
            experiment_id=experiment_id,
            target_scale=target_scale,
            student_temperature=student_temperature,
        )
        self.root = self.base.root
        self.manifest = self.base.manifest
        self.shards = self.base.shards
        self.experiment_id = experiment_id
        self.target_scale = target_scale
        self.student_temperature = student_temperature

    @property
    def split(self) -> str:
        return self.base.split

    @property
    def group_count(self) -> int:
        return self.base.group_count

    @property
    def candidate_count(self) -> int:
        return self.base.candidate_count

    def batches(self, *args: Any, **kwargs: Any) -> Iterator[ExpectedRankBatch]:
        for batch in self.base.batches(*args, **kwargs):
            ranks, mask = self.cache.ranks_for_batch(batch)
            yield ExpectedRankBatch(
                batch,
                ranks,
                mask,
                self.target_scale,
                self.student_temperature,
            )


def build_expected_rank_cache(
    dataset_root: str | Path,
    cache_root: str | Path,
    *,
    workers: int = EXPECTED_RANK_CACHE_WORKERS,
    overwrite: bool = False,
    experiment_id: str = EXPERIMENT_ID,
    target_scale: float = EXPECTED_RANK_TARGET_SCALE,
    student_temperature: float = EXPECTED_RANK_STUDENT_TEMPERATURE,
) -> dict[str, Any]:
    """Build one deterministic expected-rank sidecar cache atomically."""
    if target_scale <= 0.0:
        raise ValueError("expected-rank target scale must be positive")
    if student_temperature <= 0.0:
        raise ValueError("expected-rank student temperature must be positive")
    dataset = GradedOracleDataset(dataset_root)
    _validate_open_dataset(dataset)
    cache_root = Path(cache_root)
    if cache_root.exists():
        if not overwrite:
            return ExpectedRankTargetCache(
                cache_root,
                dataset,
                experiment_id=experiment_id,
                target_scale=target_scale,
                student_temperature=student_temperature,
            ).manifest
        shutil.rmtree(cache_root)
    cache_root.parent.mkdir(parents=True, exist_ok=True)

    group_ids: list[int] = []
    candidate_counts: list[int] = []
    ranks: list[np.ndarray] = []
    identity = blake3.blake3()
    for result in parallel_group_map(
        _expected_rank_cache_group,
        iter_supervision_groups(dataset),
        workers,
    ):
        group_ids.append(int(result["group_id"]))
        candidate_counts.append(int(result["candidate_count"]))
        ranks.append(result["expected_ranks"])
        identity.update(bytes.fromhex(str(result["identity_blake3"])))

    group_array = np.asarray(group_ids, dtype=np.uint64)
    count_array = np.asarray(candidate_counts, dtype=np.uint32)
    offsets = np.concatenate(
        [
            np.zeros(1, dtype=np.uint64),
            np.cumsum(count_array, dtype=np.uint64),
        ]
    )
    rank_array = np.concatenate(ranks).astype(np.float32, copy=False)
    if (
        len(group_array) != dataset.group_count
        or int(offsets[-1]) != dataset.candidate_count
        or len(rank_array) != dataset.candidate_count
    ):
        raise ValueError("expected-rank cache coverage does not match dataset")

    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{cache_root.name}.",
            dir=cache_root.parent,
        )
    )
    try:
        arrays = {
            "group_ids.npy": group_array,
            "candidate_counts.npy": count_array,
            "offsets.npy": offsets,
            "expected_ranks.npy": rank_array,
        }
        for name, values in arrays.items():
            np.save(temporary / name, values, allow_pickle=False)
        files = {
            name: _file_identity(temporary / name)
            for name in sorted(arrays)
        }
        finite = rank_array[np.isfinite(rank_array)]
        manifest = {
            "schema_version": EXPECTED_RANK_CACHE_SCHEMA_VERSION,
            "experiment_id": experiment_id,
            "dataset": {
                "path": str(dataset.root.resolve()),
                "dataset_id": dataset.manifest["dataset_id"],
                "split": dataset.split,
                "manifest_blake3": _checksum(dataset.root / "dataset.json"),
                "groups": dataset.group_count,
                "candidates": dataset.candidate_count,
                "shards": dataset.manifest["shards"],
            },
            "target": {
                "definition": "1+sum_j P(value_j>value_i)",
                "cohort": "r1200-labeled-nonfrontier",
                "standard_error": "stddev/sqrt(max(samples,1))",
                "target_mass": f"exp(-(expected_rank-1)/{target_scale:g})",
                "target_scale": target_scale,
                "student_temperature": student_temperature,
            },
            "coverage": {
                "groups": len(group_array),
                "candidates": len(rank_array),
                "eligible_candidates": len(finite),
                "minimum_expected_rank": float(np.min(finite)),
                "maximum_expected_rank": float(np.max(finite)),
            },
            "ordered_group_action_identity_blake3": identity.hexdigest(),
            "files": files,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        os.replace(temporary, cache_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return ExpectedRankTargetCache(
        cache_root,
        dataset,
        experiment_id=experiment_id,
        target_scale=target_scale,
        student_temperature=student_temperature,
    ).manifest


def compare_expected_rank_caches(
    left_root: str | Path,
    right_root: str | Path,
) -> dict[str, Any]:
    """Compare two independently generated caches without path-dependent fields."""
    left_root = Path(left_root)
    right_root = Path(right_root)
    left = json.loads((left_root / "manifest.json").read_text())
    right = json.loads((right_root / "manifest.json").read_text())
    left_scientific = _cache_scientific_payload(left)
    right_scientific = _cache_scientific_payload(right)
    files = sorted(left["files"])
    byte_identity = {
        name: (left_root / name).read_bytes() == (right_root / name).read_bytes()
        for name in files
    }
    return {
        "scientific_payload_identical": left_scientific == right_scientific,
        "file_bytes_identical": byte_identity,
        "all_file_bytes_identical": all(byte_identity.values()),
        "left": left_scientific,
        "right": right_scientific,
    }


def compare_expected_rank_array_payloads(
    left_root: str | Path,
    right_root: str | Path,
) -> dict[str, Any]:
    """Compare aligned rank arrays while allowing protocol metadata to differ."""
    left_root = Path(left_root)
    right_root = Path(right_root)
    names = (
        "group_ids.npy",
        "candidate_counts.npy",
        "offsets.npy",
        "expected_ranks.npy",
    )
    byte_identity = {
        name: (left_root / name).read_bytes() == (right_root / name).read_bytes()
        for name in names
    }
    left = json.loads((left_root / "manifest.json").read_text())
    right = json.loads((right_root / "manifest.json").read_text())
    return {
        "file_bytes_identical": byte_identity,
        "all_file_bytes_identical": all(byte_identity.values()),
        "ordered_group_action_identity_identical": (
            left["ordered_group_action_identity_blake3"]
            == right["ordered_group_action_identity_blake3"]
        ),
        "left_experiment_id": left["experiment_id"],
        "right_experiment_id": right["experiment_id"],
        "left_target_scale": float(left["target"]["target_scale"]),
        "right_target_scale": float(right["target"]["target_scale"]),
    }


def rotate_expected_rank_batch(
    batch: ExpectedRankBatch,
    rotations: int | np.ndarray,
) -> ExpectedRankBatch:
    """Rotate observables while preserving scalar target alignment."""
    return ExpectedRankBatch(
        rotate_graded_oracle_batch(batch.base, rotations),
        batch.expected_rank,
        batch.expected_rank_mask,
        batch.target_scale,
        batch.student_temperature,
    )


def randomly_rotate_expected_rank_batch(
    batch: ExpectedRankBatch,
    seed: int,
) -> ExpectedRankBatch:
    """Sample one exact uniform rotation per decision group."""
    rng = np.random.default_rng(seed)
    return rotate_expected_rank_batch(
        batch,
        rng.integers(0, 6, size=batch.action_features.shape[0]),
    )


def expected_rank_loss_from_scores(
    scores: mx.array,
    expected_rank: mx.array,
    target_mask: mx.array,
    eligible_mask: mx.array,
    *,
    target_scale: float = EXPECTED_RANK_TARGET_SCALE,
    student_temperature: float = EXPECTED_RANK_STUDENT_TEMPERATURE,
) -> mx.array:
    """Cross entropy from smooth expected-rank mass to deployed student scores."""
    if target_scale <= 0.0:
        raise ValueError("expected-rank target scale must be positive")
    if student_temperature <= 0.0:
        raise ValueError("expected-rank student temperature must be positive")
    target_logits = mx.where(
        target_mask,
        -(expected_rank - 1.0) / target_scale,
        -1e9,
    )
    target_probabilities = mx.softmax(target_logits, axis=-1)
    student_logits = mx.where(
        eligible_mask,
        scores / student_temperature,
        -1e9,
    )
    log_probabilities = student_logits - mx.logsumexp(
        student_logits,
        axis=-1,
        keepdims=True,
    )
    per_group = -mx.sum(
        mx.where(
            target_mask,
            target_probabilities * log_probabilities,
            0.0,
        ),
        axis=-1,
    )
    valid = mx.any(target_mask, axis=-1) & mx.any(eligible_mask, axis=-1)
    return _masked_mean(per_group, valid)


def frontier_expected_rank_loss(
    model: GradedOracleRanker,
    batch: ExpectedRankBatch,
) -> mx.array:
    """Apply the single frozen ADR 0100 training objective."""
    prediction = predict_graded_oracle_batch(model, batch)
    frontier = (
        batch.source_flags.astype(mx.int32) & GRADED_SOURCE_CHAMPION_FRONTIER
    ) != 0
    eligible = batch.candidate_mask & ~frontier
    return expected_rank_loss_from_scores(
        prediction.scores,
        batch.expected_rank,
        batch.expected_rank_mask,
        eligible,
        target_scale=float(
            getattr(batch, "target_scale", EXPECTED_RANK_TARGET_SCALE)
        ),
        student_temperature=float(
            getattr(
                batch,
                "student_temperature",
                EXPECTED_RANK_STUDENT_TEMPERATURE,
            )
        ),
    )


def build_expected_rank_target_mask(
    *,
    expected_rank: np.ndarray,
    expected_rank_mask: np.ndarray,
    source_flags: np.ndarray,
    candidate_mask: np.ndarray,
    action_hashes: np.ndarray,
    width: int = FRONTIER_ANCHORED_WIDTH,
) -> np.ndarray:
    """Build the deterministic top expected-rank nonfrontier width fill."""
    shape = candidate_mask.shape
    if (
        expected_rank.shape != shape
        or expected_rank_mask.shape != shape
        or source_flags.shape != shape
        or action_hashes.shape[:2] != shape
    ):
        raise ValueError("expected-rank target arrays have inconsistent shapes")
    target = np.zeros(shape, dtype=np.bool_)
    for row, mask in enumerate(candidate_mask):
        count = int(np.sum(mask))
        indices = np.arange(count, dtype=np.int32)
        frontier = indices[
            (source_flags[row, :count] & GRADED_SOURCE_CHAMPION_FRONTIER) != 0
        ]
        quota = min(width, count) - len(frontier)
        eligible = indices[expected_rank_mask[row, :count]]
        if quota < 0 or len(eligible) < quota:
            raise ValueError("expected-rank cohort cannot fill anchored width")
        ranking = np.asarray(
            sorted(
                (int(index) for index in eligible),
                key=lambda index: (
                    float(expected_rank[row, index]),
                    bytes(action_hashes[row, index]),
                ),
            ),
            dtype=np.int32,
        )
        target[row, ranking[:quota]] = True
    return target


def evaluate_frontier_expected_rank(
    model: GradedOracleRanker,
    dataset: ExpectedRankDataset,
    group_batch_size: int,
) -> dict[str, Any]:
    """Evaluate learned scores against expected-rank fit and R4800 quality."""
    model.eval()
    return _evaluate_expected_rank(dataset, group_batch_size, model=model)


def evaluate_expected_rank_screen_baseline(
    dataset: ExpectedRankDataset,
    group_batch_size: int = 64,
) -> dict[str, Any]:
    """Evaluate the unchanged anchored screen without treatment weights."""
    return _evaluate_expected_rank(dataset, group_batch_size, model=None)


def expected_rank_validation_gates(
    report: dict[str, Any],
    *,
    performance_by_host: dict[str, dict[str, Any]] | None = None,
    optimization_audit_passed: bool = True,
    replay_identical: bool = True,
    cache_identical: bool = True,
) -> dict[str, bool]:
    """Apply the complete frozen ADR 0100 pilot thresholds."""
    train = report["train"]
    validation = report["validation"]
    distinguishable = validation["top64_distinguishable_winner_recall"]
    gates = {
        "cache_identity_passed": cache_identical,
        "optimization_audit_passed": optimization_audit_passed,
        "selected_replay_identical": replay_identical,
        "train_expected_rank_target_recall_at_least_0_80": (
            float(train["expected_rank_target_positive_recall"]) >= 0.80
        ),
        "train_expected_rank_exact_sets_at_least_0_25": (
            float(train["expected_rank_target_set_exact_fraction"]) >= 0.25
        ),
        "validation_expected_rank_target_recall_at_least_0_50": (
            float(validation["expected_rank_target_positive_recall"]) >= 0.50
        ),
        "validation_expected_rank_exact_sets_at_least_0_01": (
            float(validation["expected_rank_target_set_exact_fraction"]) >= 0.01
        ),
        "validation_r4800_winner_recall_strictly_above_0_98": (
            float(validation["top64_r4800_winner_recall"]) > 0.98
        ),
        "validation_confidence_coverage_at_least_0_99": (
            float(validation["top64_confidence_set_coverage_95"]) >= 0.99
        ),
        "validation_distinguishable_recall_at_least_0_98": (
            distinguishable is not None and float(distinguishable) >= 0.98
        ),
        "validation_retained_regret_below_0_03": (
            float(validation["mean_top64_retained_r4800_regret"]) < 0.03
        ),
        "all_groups_and_candidates_scored_once": all(
            bool(metrics["all_groups_scored_once"])
            and bool(metrics["all_candidates_scored_once"])
            for metrics in (train, validation)
        ),
        "all_scores_finite": bool(train["all_scores_finite"])
        and bool(validation["all_scores_finite"]),
        "sealed_test_unopened": not bool(report["test_split_opened"]),
        "gameplay_unopened": not bool(report["gameplay_opened"]),
        "new_teacher_compute_unused": not bool(report["new_teacher_compute_used"]),
        "external_compute_unused": not bool(report["external_compute_used"]),
    }
    for phase, values in validation["phase"].items():
        gates[f"{phase}_winner_recall_at_least_0_98"] = (
            float(values["top64_r4800_winner_recall"]) >= 0.98
        )
        gates[f"{phase}_confidence_coverage_at_least_0_98"] = (
            float(values["top64_confidence_set_coverage_95"]) >= 0.98
        )
        gates[f"{phase}_retained_regret_below_0_03"] = (
            float(values["mean_top64_retained_r4800_regret"]) < 0.03
        )
    for subset in ("nature_token_available", "independent_draft_winner"):
        values = validation["subsets"][subset]
        if int(values["groups"]) >= 20:
            gates[f"{subset}_winner_recall_at_least_0_95"] = (
                float(values["top64_r4800_winner_recall"]) >= 0.95
            )
            gates[f"{subset}_retained_regret_below_0_25"] = (
                float(values["mean_top64_retained_r4800_regret"]) < 0.25
            )
    if performance_by_host is not None:
        for host, performance in performance_by_host.items():
            gates[f"{host}_performance_passed"] = bool(performance["passed"])
    gates["pilot_passed"] = all(gates.values())
    return gates


def classify_expected_rank_pilot(
    gates: dict[str, bool],
) -> str:
    """Return the frozen ADR 0100 classification."""
    pipeline_names = (
        "cache_identity_passed",
        "optimization_audit_passed",
        "selected_replay_identical",
        "all_groups_and_candidates_scored_once",
        "all_scores_finite",
        "sealed_test_unopened",
        "gameplay_unopened",
        "new_teacher_compute_unused",
        "external_compute_unused",
    )
    if not all(gates.get(name, False) for name in pipeline_names):
        return "expected_rank_pipeline_invalid"
    if gates.get("pilot_passed", False):
        return "expected_rank_model_sufficient"
    train_fit = (
        gates["train_expected_rank_target_recall_at_least_0_80"]
        and gates["train_expected_rank_exact_sets_at_least_0_25"]
    )
    return (
        "expected_rank_train_fit_only"
        if train_fit
        else "expected_rank_optimization_underfit"
    )


def _expected_rank_cache_group(group: SupervisionGroup) -> dict[str, Any]:
    count = group.candidate_count
    frontier = (
        group.source_flags & GRADED_SOURCE_CHAMPION_FRONTIER
    ) != 0
    eligible = (group.r1200_samples > 0) & ~frontier
    indices = np.flatnonzero(eligible)
    if len(indices) < 1:
        raise ValueError("expected-rank cache group has no eligible actions")
    errors = standard_error(
        group.r1200_stddev[indices],
        group.r1200_samples[indices],
    )
    values = expected_normal_ranks(group.r1200_mean[indices], errors)
    ranks = np.full(count, np.nan, dtype=np.float32)
    ranks[indices] = values.astype(np.float32)
    identity = blake3.blake3()
    identity.update(int(group.group_id).to_bytes(8, "little"))
    identity.update(int(count).to_bytes(4, "little"))
    identity.update(np.ascontiguousarray(group.action_hash).tobytes())
    identity.update(np.ascontiguousarray(group.source_flags).tobytes())
    identity.update(np.ascontiguousarray(group.r1200_samples).tobytes())
    identity.update(ranks.tobytes())
    return {
        "group_id": group.group_id,
        "candidate_count": count,
        "expected_ranks": ranks,
        "identity_blake3": identity.hexdigest(),
    }


def _evaluate_expected_rank(
    dataset: ExpectedRankDataset,
    group_batch_size: int,
    *,
    model: GradedOracleRanker | None,
) -> dict[str, Any]:
    accumulator = _EvaluationAccumulator()
    for batch in dataset.batches(
        group_batch_size,
        maximum_actions_per_batch=GRADED_ORACLE_PACKED_ACTION_LIMIT,
        maximum_group_actions=GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
    ):
        if model is None:
            scores = np.asarray(batch.screen_value)
            batch_loss = _numpy_expected_rank_loss(batch, scores)
        else:
            prediction = predict_graded_oracle_batch(model, batch)
            loss = frontier_expected_rank_loss(model, batch)
            mx.eval(prediction.scores, loss)
            scores = np.asarray(prediction.scores)
            batch_loss = float(loss.item())
        masks = np.asarray(batch.candidate_mask)
        screen = np.asarray(batch.screen_value)
        ranks = np.asarray(batch.expected_rank)
        rank_masks = np.asarray(batch.expected_rank_mask)
        source_flags = np.asarray(batch.source_flags)
        action_hashes = np.asarray(batch.action_hash)
        r4800_mean = np.asarray(batch.r4800_mean)
        r4800_stddev = np.asarray(batch.r4800_stddev)
        r4800_samples = np.asarray(batch.r4800_samples)
        r4800_mask = np.asarray(batch.r4800_mask)
        selected = np.asarray(batch.selected_index)
        phases = np.asarray(batch.phase)
        tokens = np.asarray(batch.active_nature_tokens)
        draft_kind = np.asarray(batch.draft_kind)
        same_slot = np.asarray(batch.same_slot_independent)
        free_refresh = np.asarray(batch.replace_three_of_a_kind)
        wipe_count = np.asarray(batch.wipe_count)
        targets = build_expected_rank_target_mask(
            expected_rank=ranks,
            expected_rank_mask=rank_masks,
            source_flags=source_flags,
            candidate_mask=masks,
            action_hashes=action_hashes,
        )
        accumulator.total_loss += batch_loss * len(scores)

        for row, mask in enumerate(masks):
            count = int(np.sum(mask))
            group_scores = scores[row, :count]
            group_screen = screen[row, :count]
            group_flags = source_flags[row, :count]
            group_hashes = action_hashes[row, :count]
            group_ranks = ranks[row, :count]
            group_rank_mask = rank_masks[row, :count]
            group_target = targets[row, :count]
            group_r4800 = r4800_mean[row, :count]
            group_r4800_stddev = r4800_stddev[row, :count]
            group_r4800_samples = r4800_samples[row, :count]
            group_r4800_mask = r4800_mask[row, :count]
            winner = int(selected[row])

            retained = frontier_anchored_retained_indices(
                scores=group_scores,
                source_flags=group_flags,
                action_hashes=group_hashes,
            )
            screen_retained = frontier_anchored_retained_indices(
                scores=group_screen,
                source_flags=group_flags,
                action_hashes=group_hashes,
            )
            frontier = np.flatnonzero(
                (group_flags & GRADED_SOURCE_CHAMPION_FRONTIER) != 0
            ).astype(np.int32)
            target_retained = np.concatenate(
                [frontier, np.flatnonzero(group_target).astype(np.int32)]
            )
            observation = _decision_observation(
                retained,
                winner,
                group_r4800,
                group_r4800_stddev,
                group_r4800_samples,
                group_r4800_mask,
                group_hashes,
            )
            screen_observation = _decision_observation(
                screen_retained,
                winner,
                group_r4800,
                group_r4800_stddev,
                group_r4800_samples,
                group_r4800_mask,
                group_hashes,
            )
            ceiling_observation = _decision_observation(
                target_retained,
                winner,
                group_r4800,
                group_r4800_stddev,
                group_r4800_samples,
                group_r4800_mask,
                group_hashes,
            )
            accumulator.model.add(observation)
            accumulator.screen.add(screen_observation)
            accumulator.target_ceiling.add(ceiling_observation)

            phase = _PHASE_NAMES.get(int(phases[row]))
            if phase is None:
                raise ValueError("expected-rank evaluation received invalid phase")
            accumulator.phases[phase].add(observation)
            if int(tokens[row]) > 0:
                accumulator.subsets["nature_token_available"].add(observation)
            if int(draft_kind[row, winner]) == 1:
                accumulator.subsets["independent_draft_winner"].add(observation)
                accumulator.action_families["independent"].add(observation)
            else:
                accumulator.action_families["paired"].add(observation)
            if int(same_slot[row, winner]) != 0:
                accumulator.action_families["same_slot_independent"].add(observation)
            if int(free_refresh[row, winner]) != 0:
                accumulator.action_families["free_refresh"].add(observation)
            if int(wipe_count[row, winner]) > 0:
                accumulator.action_families["paid_wipe"].add(observation)
            width_name = (
                "at_most_2048"
                if count <= 2048
                else "2049_to_4096"
                if count <= 4096
                else "above_4096"
            )
            accumulator.widths[width_name].add(observation)

            retained_nonfrontier = retained[
                (
                    group_flags[retained]
                    & GRADED_SOURCE_CHAMPION_FRONTIER
                )
                == 0
            ]
            recalled = int(np.sum(group_target[retained_nonfrontier]))
            target_count = int(np.sum(group_target))
            accumulator.target_slots += target_count
            accumulator.target_recalled += recalled
            accumulator.target_exact_groups += int(recalled == target_count)
            accumulator.rank_correlations.append(
                _rank_correlation(
                    group_scores[group_rank_mask],
                    -group_ranks[group_rank_mask],
                )
            )
            accumulator.target_entropies.append(
                _expected_rank_entropy(
                    group_ranks[group_rank_mask],
                    target_scale=dataset.target_scale,
                )
            )
            accumulator.frontier_counts.append(float(len(frontier)))
            accumulator.nonfinite_scores += int(
                np.sum(~np.isfinite(group_scores))
            )
            accumulator.groups += 1
            accumulator.candidates += count

    if accumulator.groups == 0:
        raise ValueError("expected-rank evaluation dataset is empty")
    model_report = accumulator.model.report()
    return {
        "groups": accumulator.groups,
        "candidates": accumulator.candidates,
        "expected_groups": dataset.group_count,
        "expected_candidates": dataset.candidate_count,
        "all_groups_scored_once": accumulator.groups == dataset.group_count,
        "all_candidates_scored_once": (
            accumulator.candidates == dataset.candidate_count
        ),
        "nonfinite_scores": accumulator.nonfinite_scores,
        "all_scores_finite": accumulator.nonfinite_scores == 0,
        "training_objective": accumulator.total_loss / accumulator.groups,
        "proposal_width": FRONTIER_ANCHORED_WIDTH,
        "expected_rank_target_positive_recall": (
            accumulator.target_recalled / max(accumulator.target_slots, 1)
        ),
        "expected_rank_target_positive_miss_rate": (
            1.0
            - accumulator.target_recalled / max(accumulator.target_slots, 1)
        ),
        "expected_rank_target_set_exact_fraction": (
            accumulator.target_exact_groups / accumulator.groups
        ),
        "expected_rank_score_correlation": _distribution(
            accumulator.rank_correlations
        ),
        "expected_rank_target_entropy_bits": _distribution(
            accumulator.target_entropies
        ),
        "frontier_count": _distribution(accumulator.frontier_counts),
        "top64_r4800_winner_recall": model_report[
            "top64_r4800_winner_recall"
        ],
        "top64_r4800_winner_miss_rate": (
            1.0 - float(model_report["top64_r4800_winner_recall"])
        ),
        "top64_confidence_set_coverage_95": model_report[
            "top64_confidence_set_coverage_95"
        ],
        "top64_distinguishable_winner_recall": model_report[
            "top64_distinguishable_winner_recall"
        ],
        "mean_top64_retained_r4800_regret": model_report[
            "mean_top64_retained_r4800_regret"
        ],
        "screen": accumulator.screen.report(),
        "target_ceiling": accumulator.target_ceiling.report(),
        "phase": {
            name: values.report() for name, values in accumulator.phases.items()
        },
        "subsets": {
            name: values.report() for name, values in accumulator.subsets.items()
        },
        "action_family": {
            name: values.report()
            for name, values in accumulator.action_families.items()
        },
        "group_width": {
            name: values.report() for name, values in accumulator.widths.items()
        },
    }


def _numpy_expected_rank_loss(
    batch: ExpectedRankBatch,
    scores: np.ndarray,
) -> float:
    ranks = np.asarray(batch.expected_rank)
    target_mask = np.asarray(batch.expected_rank_mask)
    flags = np.asarray(batch.source_flags)
    candidate_mask = np.asarray(batch.candidate_mask)
    frontier = (flags & GRADED_SOURCE_CHAMPION_FRONTIER) != 0
    eligible = candidate_mask & ~frontier
    total = 0.0
    groups = 0
    for row in range(len(scores)):
        target_scale = float(
            getattr(batch, "target_scale", EXPECTED_RANK_TARGET_SCALE)
        )
        student_temperature = float(
            getattr(
                batch,
                "student_temperature",
                EXPECTED_RANK_STUDENT_TEMPERATURE,
            )
        )
        target_logits = -(ranks[row, target_mask[row]] - 1.0) / (
            target_scale
        )
        target_probabilities = _softmax(target_logits)
        student_logits = (
            scores[row, eligible[row]] / student_temperature
        )
        student_log_probabilities = student_logits - _logsumexp(student_logits)
        eligible_indices = np.flatnonzero(eligible[row])
        target_indices = np.flatnonzero(target_mask[row])
        positions = {
            int(index): position
            for position, index in enumerate(eligible_indices)
        }
        total -= float(
            np.sum(
                target_probabilities
                * np.asarray(
                    [
                        student_log_probabilities[positions[int(index)]]
                        for index in target_indices
                    ]
                )
            )
        )
        groups += 1
    return total / max(groups, 1)


def _decision_observation(
    retained: np.ndarray,
    winner: int,
    r4800_mean: np.ndarray,
    r4800_stddev: np.ndarray,
    r4800_samples: np.ndarray,
    r4800_mask: np.ndarray,
    action_hashes: np.ndarray,
) -> dict[str, bool | float]:
    labeled = np.flatnonzero(r4800_mask).astype(np.int32)
    if len(labeled) < 2:
        raise ValueError("expected-rank evaluation requires two R4800 actions")
    ranking = np.asarray(
        sorted(
            (int(index) for index in labeled),
            key=lambda index: (
                -float(r4800_mean[index]),
                bytes(action_hashes[index]),
            ),
        ),
        dtype=np.int32,
    )
    if int(ranking[0]) != winner:
        raise ValueError("stored selected action is not stable R4800 winner")
    runner_up = int(ranking[1])
    errors = r4800_stddev / np.sqrt(np.maximum(r4800_samples, 1.0))
    distinguishable = r4800_mean[winner] - r4800_mean[runner_up] > (
        NORMAL_95 * np.hypot(errors[winner], errors[runner_up])
    )
    confidence = np.zeros(len(r4800_mean), dtype=np.bool_)
    confidence[labeled] = r4800_mean[winner] - r4800_mean[labeled] <= (
        NORMAL_95 * np.hypot(errors[winner], errors[labeled])
    )
    retained_labeled = retained[r4800_mask[retained]]
    regret = (
        float(r4800_mean[winner] - np.max(r4800_mean[retained_labeled]))
        if len(retained_labeled)
        else float(np.ptp(r4800_mean[labeled]))
    )
    return {
        "exact": bool(np.any(retained == winner)),
        "confidence": bool(np.any(confidence[retained])),
        "distinguishable": bool(distinguishable),
        "regret": regret,
    }


def _expected_rank_entropy(
    ranks: np.ndarray,
    *,
    target_scale: float = EXPECTED_RANK_TARGET_SCALE,
) -> float:
    probabilities = _softmax(
        -(np.asarray(ranks, dtype=np.float64) - 1.0) / target_scale
    )
    return float(
        -np.sum(
            np.where(
                probabilities > 0.0,
                probabilities * np.log2(probabilities),
                0.0,
            )
        )
    )


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2:
        return 0.0
    left_rank = np.argsort(np.argsort(left, kind="stable"), kind="stable")
    right_rank = np.argsort(np.argsort(right, kind="stable"), kind="stable")
    value = np.corrcoef(left_rank, right_rank)[0, 1]
    return 0.0 if not np.isfinite(value) else float(value)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials)


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + float(np.log(np.sum(np.exp(values - maximum))))


def _masked_mean(values: mx.array, mask: mx.array) -> mx.array:
    weights = mask.astype(values.dtype)
    return mx.sum(values * weights) / mx.maximum(mx.sum(weights), 1.0)


def _distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.all(np.isfinite(array)):
        raise ValueError("expected-rank distribution requires finite values")
    return {
        "count": len(array),
        "mean": float(np.mean(array)),
        "min": float(np.min(array)),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(np.max(array)),
    }


def _validate_open_dataset(dataset: GradedOracleDataset) -> None:
    if dataset.split not in _EXPECTED_DATASET_MANIFESTS:
        raise ValueError("expected-rank supervision accepts only open splits")
    if (
        _checksum(dataset.root / "dataset.json")
        != _EXPECTED_DATASET_MANIFESTS[dataset.split]
    ):
        raise ValueError("ADR 0100 dataset identity drifted")


def _validate_cache_manifest(
    root: Path,
    manifest: dict[str, Any],
    dataset: GradedOracleDataset,
    *,
    experiment_id: str = EXPERIMENT_ID,
    target_scale: float = EXPECTED_RANK_TARGET_SCALE,
    student_temperature: float = EXPECTED_RANK_STUDENT_TEMPERATURE,
) -> None:
    _validate_open_dataset(dataset)
    if (
        manifest.get("schema_version") != EXPECTED_RANK_CACHE_SCHEMA_VERSION
        or manifest.get("experiment_id") != experiment_id
    ):
        raise ValueError("unsupported expected-rank cache schema")
    identity = manifest.get("dataset", {})
    if (
        identity.get("dataset_id") != dataset.manifest["dataset_id"]
        or identity.get("split") != dataset.split
        or identity.get("manifest_blake3")
        != _checksum(dataset.root / "dataset.json")
        or int(identity.get("groups", -1)) != dataset.group_count
        or int(identity.get("candidates", -1)) != dataset.candidate_count
        or identity.get("shards") != dataset.manifest["shards"]
    ):
        raise ValueError("expected-rank cache dataset identity drifted")
    target = manifest.get("target", {})
    if (
        target.get("definition") != "1+sum_j P(value_j>value_i)"
        or target.get("cohort") != "r1200-labeled-nonfrontier"
        or float(target.get("target_scale", -1.0)) != target_scale
        or float(target.get("student_temperature", -1.0))
        != student_temperature
    ):
        raise ValueError("expected-rank cache target contract drifted")
    for name, metadata in manifest.get("files", {}).items():
        path = root / name
        if (
            not path.is_file()
            or path.stat().st_size != int(metadata["bytes"])
            or _checksum(path) != metadata["blake3"]
            or _sha256(path) != metadata["sha256"]
        ):
            raise ValueError(f"expected-rank cache file failed identity: {name}")


def _cache_scientific_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    dataset = dict(manifest["dataset"])
    dataset.pop("path", None)
    return {
        "schema_version": manifest["schema_version"],
        "experiment_id": manifest["experiment_id"],
        "dataset": dataset,
        "target": manifest["target"],
        "coverage": manifest["coverage"],
        "ordered_group_action_identity_blake3": manifest[
            "ordered_group_action_identity_blake3"
        ],
        "files": manifest["files"],
    }


def _file_identity(path: Path) -> dict[str, int | str]:
    return {
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "blake3": _checksum(path),
    }


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checksum(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
