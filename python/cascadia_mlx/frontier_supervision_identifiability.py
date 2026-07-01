"""Finite-teacher supervision audits for ADR 0099."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import socket
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import blake3
import numpy as np

from cascadia_mlx.graded_oracle_dataset import (
    _CANDIDATE_DTYPE,
    _GROUP_HEADER_DTYPE,
    GradedOracleDataset,
)
from cascadia_mlx.graded_oracle_frontier_anchor import (
    FRONTIER_ANCHORED_WIDTH,
    GRADED_SOURCE_CHAMPION_FRONTIER,
)
from cascadia_mlx.graded_oracle_identifiability import NORMAL_95

EXPERIMENT_ID = "complete-action-frontier-supervision-identifiability-v1"

BOUNDARY_SIGNAL = "boundary-signal"
CROSS_FIDELITY = "cross-fidelity"
TEACHER_RESAMPLING = "teacher-resampling"
EXPECTED_RANK_CEILING = "expected-rank-ceiling"
AUDIT_KINDS = (
    BOUNDARY_SIGNAL,
    CROSS_FIDELITY,
    TEACHER_RESAMPLING,
    EXPECTED_RANK_CEILING,
)

RESAMPLING_SEED = 2026061625
RESAMPLING_DRAWS = 512
AUDIT_WORKERS = 8
_PHASE_NAMES = {0: "early", 1: "middle", 2: "late"}


@dataclass(frozen=True)
class SupervisionGroup:
    """Minimal immutable teacher record needed by the ADR 0099 audits."""

    group_id: int
    phase: int
    selected_index: int
    source_flags: np.ndarray
    action_hash: np.ndarray
    r600_mean: np.ndarray
    r600_stddev: np.ndarray
    r600_samples: np.ndarray
    r1200_mean: np.ndarray
    r1200_stddev: np.ndarray
    r1200_samples: np.ndarray
    r4800_mean: np.ndarray
    r4800_stddev: np.ndarray
    r4800_samples: np.ndarray

    @property
    def candidate_count(self) -> int:
        return len(self.source_flags)


@dataclass
class CeilingAccumulator:
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
        if observation["distinguishable"]:
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


def iter_supervision_groups(dataset: GradedOracleDataset) -> Iterator[SupervisionGroup]:
    """Stream only target metadata from the grouped binary dataset."""
    if dataset.split not in {"train", "validation"}:
        raise ValueError("ADR 0099 accepts only open train or validation data")
    for shard in dataset.shards:
        raw = shard.bytes()
        for ref in shard.groups:
            header = np.frombuffer(
                raw,
                dtype=_GROUP_HEADER_DTYPE,
                count=1,
                offset=ref.header_offset,
            )[0]
            candidates = np.frombuffer(
                raw,
                dtype=_CANDIDATE_DTYPE,
                count=ref.candidate_count,
                offset=ref.candidate_offset,
            )
            yield SupervisionGroup(
                group_id=int(header["group_id"]),
                phase=int(header["phase"]),
                selected_index=int(header["selected_index"]),
                source_flags=np.asarray(candidates["source_flags"]),
                action_hash=np.asarray(candidates["action_hash"]),
                r600_mean=np.asarray(candidates["r600"]["mean"], dtype=np.float64),
                r600_stddev=np.asarray(
                    candidates["r600"]["stddev"],
                    dtype=np.float64,
                ),
                r600_samples=np.asarray(
                    candidates["r600"]["samples"],
                    dtype=np.float64,
                ),
                r1200_mean=np.asarray(
                    candidates["r1200"]["mean"],
                    dtype=np.float64,
                ),
                r1200_stddev=np.asarray(
                    candidates["r1200"]["stddev"],
                    dtype=np.float64,
                ),
                r1200_samples=np.asarray(
                    candidates["r1200"]["samples"],
                    dtype=np.float64,
                ),
                r4800_mean=np.asarray(
                    candidates["r4800"]["mean"],
                    dtype=np.float64,
                ),
                r4800_stddev=np.asarray(
                    candidates["r4800"]["stddev"],
                    dtype=np.float64,
                ),
                r4800_samples=np.asarray(
                    candidates["r4800"]["samples"],
                    dtype=np.float64,
                ),
            )


def standard_error(stddev: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Return the frozen raw finite-teacher standard error."""
    return stddev / np.sqrt(np.maximum(samples, 1.0))


def frontier_and_target(
    group: SupervisionGroup,
    tier: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build one stable frontier-anchored learned quota from a fidelity tier."""
    means = getattr(group, f"{tier}_mean")
    samples = getattr(group, f"{tier}_samples")
    indices = np.arange(group.candidate_count, dtype=np.int32)
    frontier_mask = (group.source_flags & GRADED_SOURCE_CHAMPION_FRONTIER) != 0
    frontier = stable_ranking(means, group.action_hash, indices[frontier_mask])
    quota = min(FRONTIER_ANCHORED_WIDTH, group.candidate_count) - len(frontier)
    eligible = indices[(samples > 0) & ~frontier_mask]
    if quota < 0 or len(eligible) < quota:
        raise ValueError(f"{tier} cohort cannot fill the anchored width")
    ranking = stable_ranking(means, group.action_hash, eligible)
    return frontier, ranking[:quota], ranking[quota:]


def stable_ranking(
    scores: np.ndarray,
    action_hashes: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """Sort descending by score and ascending by the canonical action hash."""
    return np.asarray(
        sorted(
            (int(index) for index in indices),
            key=lambda index: (-float(scores[index]), bytes(action_hashes[index])),
        ),
        dtype=np.int32,
    )


def boundary_group(group: SupervisionGroup) -> dict[str, Any]:
    """Measure one group's nominal cutoff significance."""
    _frontier, target, excluded = frontier_and_target(group, "r1200")
    if not len(target) or not len(excluded):
        raise ValueError("boundary audit requires both target and excluded rows")
    weakest = int(target[-1])
    strongest_excluded = int(excluded[0])
    errors = standard_error(group.r1200_stddev, group.r1200_samples)
    margin = float(group.r1200_mean[weakest] - group.r1200_mean[strongest_excluded])
    combined = float(np.hypot(errors[weakest], errors[strongest_excluded]))
    slot_margins = group.r1200_mean[target] - group.r1200_mean[strongest_excluded]
    slot_errors = np.hypot(errors[target], errors[strongest_excluded])
    return {
        "margin": margin,
        "z_score": margin / max(combined, 1e-12),
        "robust_group": margin > NORMAL_95 * combined,
        "robust_slots": int(np.sum(slot_margins > NORMAL_95 * slot_errors)),
        "target_slots": len(target),
        "candidates": group.candidate_count,
    }


def audit_boundary(
    dataset: GradedOracleDataset,
    *,
    workers: int = 1,
) -> dict[str, Any]:
    """Measure nominal R1200 cutoff and target-slot significance."""
    margins: list[float] = []
    z_scores: list[float] = []
    robust_groups = 0
    robust_slots = 0
    target_slots = 0
    groups = 0
    candidates = 0
    for result in parallel_group_map(
        boundary_group,
        iter_supervision_groups(dataset),
        workers,
    ):
        margins.append(float(result["margin"]))
        z_scores.append(float(result["z_score"]))
        robust_groups += int(result["robust_group"])
        robust_slots += int(result["robust_slots"])
        target_slots += int(result["target_slots"])
        groups += 1
        candidates += int(result["candidates"])
    report = {
        "groups": groups,
        "candidates": candidates,
        "target_slots": target_slots,
        "robust_target_slot_fraction_95": robust_slots / max(target_slots, 1),
        "robust_complete_set_fraction_95": robust_groups / max(groups, 1),
        "cutoff_margin": distribution(margins),
        "cutoff_z_score": distribution(z_scores),
    }
    report["gate_passed"] = (
        report["robust_target_slot_fraction_95"] >= 0.80
        and report["robust_complete_set_fraction_95"] >= 0.25
    )
    return report


def cross_fidelity_group(group: SupervisionGroup) -> dict[str, Any]:
    """Compare one group's independently reconstructed fidelity targets."""
    frontier1200, target1200, _excluded1200 = frontier_and_target(group, "r1200")
    indices = np.arange(group.candidate_count, dtype=np.int32)
    frontier_mask = (group.source_flags & GRADED_SOURCE_CHAMPION_FRONTIER) != 0
    frontier600 = stable_ranking(
        group.r600_mean,
        group.action_hash,
        indices[frontier_mask],
    )
    if set(map(int, frontier600)) != set(map(int, frontier1200)):
        raise ValueError("frontier membership drifted across fidelities")
    quota = len(target1200)
    eligible600 = indices[(group.r600_samples > 0) & ~frontier_mask]
    fillable = len(eligible600) >= quota
    target600 = (
        stable_ranking(group.r600_mean, group.action_hash, eligible600)[:quota]
        if fillable
        else np.empty(0, dtype=np.int32)
    )
    set600 = set(map(int, target600))
    set1200 = set(map(int, target1200))
    overlap = len(set600 & set1200) if fillable else 0
    common = np.flatnonzero(
        (group.r600_samples > 0)
        & (group.r1200_samples > 0)
        & ((group.source_flags & GRADED_SOURCE_CHAMPION_FRONTIER) == 0)
    )
    retained600 = set600 | set(map(int, frontier600))
    retained1200 = set1200 | set(map(int, frontier1200))
    return {
        "fillable": fillable,
        "overlap": overlap,
        "target_total": len(target1200) if fillable else 0,
        "exact": fillable and set600 == set1200,
        "jaccard": (overlap / max(len(set600 | set1200), 1) if fillable else None),
        "correlation": (
            rank_correlation(
                group.r600_mean[common],
                group.r1200_mean[common],
            )
            if len(common) >= 2
            else None
        ),
        "r600_winner": fillable and group.selected_index in retained600,
        "r1200_winner": group.selected_index in retained1200,
        "candidates": group.candidate_count,
    }


def audit_cross_fidelity(
    dataset: GradedOracleDataset,
    *,
    workers: int = 1,
) -> dict[str, Any]:
    """Compare independently reconstructed R600 and R1200 learned quotas."""
    recovered = 0
    total = 0
    exact = 0
    jaccards: list[float] = []
    correlations: list[float] = []
    r600_winner = 0
    r1200_winner = 0
    comparable_groups = 0
    groups = 0
    candidates = 0
    for result in parallel_group_map(
        cross_fidelity_group,
        iter_supervision_groups(dataset),
        workers,
    ):
        if result["fillable"]:
            comparable_groups += 1
            recovered += int(result["overlap"])
            total += int(result["target_total"])
            exact += int(result["exact"])
            jaccards.append(float(result["jaccard"]))
            r600_winner += int(result["r600_winner"])
        if result["correlation"] is not None:
            correlations.append(float(result["correlation"]))
        r1200_winner += int(result["r1200_winner"])
        groups += 1
        candidates += int(result["candidates"])
    report = {
        "groups": groups,
        "candidates": candidates,
        "comparable_groups": comparable_groups,
        "r600_cohort_coverage_fraction": comparable_groups / max(groups, 1),
        "r600_target_recall_of_r1200": recovered / max(total, 1),
        "r600_r1200_exact_set_fraction": exact / max(comparable_groups, 1),
        "r600_r1200_jaccard": distribution_or_empty(jaccards),
        "common_cohort_rank_correlation": distribution_or_empty(correlations),
        "r600_anchored_r4800_winner_recall": (r600_winner / max(comparable_groups, 1)),
        "r1200_anchored_r4800_winner_recall": r1200_winner / max(groups, 1),
    }
    report["gate_passed"] = (
        report["r600_cohort_coverage_fraction"] == 1.0
        and report["r600_target_recall_of_r1200"] >= 0.80
        and report["r600_r1200_exact_set_fraction"] >= 0.25
    )
    return report


def resampling_group(group: SupervisionGroup) -> dict[str, Any]:
    """Measure one group's finite-teacher target stability."""
    _frontier, target, excluded = frontier_and_target(group, "r1200")
    eligible = np.concatenate([target, excluded])
    quota = len(target)
    nominal = np.zeros(len(eligible), dtype=np.bool_)
    nominal[:quota] = True
    means = group.r1200_mean[eligible]
    errors = standard_error(
        group.r1200_stddev[eligible],
        group.r1200_samples[eligible],
    )
    hash_order = sorted(
        range(len(eligible)),
        key=lambda index: bytes(group.action_hash[int(eligible[index])]),
    )
    tie_offset = np.zeros(len(eligible), dtype=np.float64)
    tie_offset[hash_order] = np.linspace(
        1e-12,
        0.0,
        len(eligible),
        endpoint=False,
    )
    rng = np.random.default_rng(np.uint64(RESAMPLING_SEED) ^ np.uint64(group.group_id))
    inclusion = np.zeros(len(eligible), dtype=np.int64)
    recall_sum = 0.0
    jaccard_sum = 0.0
    exact_draws = 0
    for chunk_start in range(0, RESAMPLING_DRAWS, 64):
        chunk = min(64, RESAMPLING_DRAWS - chunk_start)
        draws = (
            means[None, :]
            + rng.standard_normal((chunk, len(eligible))) * errors[None, :]
            + tie_offset[None, :]
        )
        selected = np.argpartition(draws, -quota, axis=1)[:, -quota:]
        intersection = np.sum(nominal[selected], axis=1)
        recall_sum += float(np.sum(intersection / quota))
        jaccard_sum += float(np.sum(intersection / (2 * quota - intersection)))
        exact_draws += int(np.sum(intersection == quota))
        np.add.at(inclusion, selected.reshape(-1), 1)
    probabilities = inclusion / RESAMPLING_DRAWS
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    entropy = -clipped * np.log2(clipped) - (1.0 - clipped) * np.log2(1.0 - clipped)
    return {
        "recall_sum": recall_sum,
        "jaccard_sum": jaccard_sum,
        "exact_draws": exact_draws,
        "target_probabilities": probabilities[:quota].tolist(),
        "entropies": entropy.tolist(),
        "candidates": group.candidate_count,
    }


def audit_resampling(
    dataset: GradedOracleDataset,
    *,
    workers: int = 1,
) -> dict[str, Any]:
    """Resample finite R1200 means and rebuild the learned quota."""
    recall_sum = 0.0
    jaccard_sum = 0.0
    exact_draws = 0
    draw_groups = 0
    target_probabilities: list[float] = []
    entropies: list[float] = []
    groups = 0
    candidates = 0
    for result in parallel_group_map(
        resampling_group,
        iter_supervision_groups(dataset),
        workers,
    ):
        recall_sum += float(result["recall_sum"])
        jaccard_sum += float(result["jaccard_sum"])
        exact_draws += int(result["exact_draws"])
        draw_groups += RESAMPLING_DRAWS
        target_probabilities.extend(result["target_probabilities"])
        entropies.extend(result["entropies"])
        groups += 1
        candidates += int(result["candidates"])
    report = {
        "groups": groups,
        "candidates": candidates,
        "seed": RESAMPLING_SEED,
        "draws_per_group": RESAMPLING_DRAWS,
        "mean_nominal_target_recall": recall_sum / max(draw_groups, 1),
        "exact_set_reproduction_fraction": exact_draws / max(draw_groups, 1),
        "mean_jaccard": jaccard_sum / max(draw_groups, 1),
        "nominal_target_inclusion_probability": distribution(target_probabilities),
        "candidate_membership_entropy_bits": distribution(entropies),
    }
    report["gate_passed"] = (
        report["mean_nominal_target_recall"] >= 0.80
        and report["exact_set_reproduction_fraction"] >= 0.25
    )
    return report


def expected_rank_group(group: SupervisionGroup) -> dict[str, Any]:
    """Evaluate one group's uncertainty-aware expected-rank target."""
    frontier, nominal_target, excluded = frontier_and_target(group, "r1200")
    eligible = np.concatenate([nominal_target, excluded])
    quota = len(nominal_target)
    means = group.r1200_mean[eligible]
    errors = standard_error(
        group.r1200_stddev[eligible],
        group.r1200_samples[eligible],
    )
    ranks = expected_normal_ranks(means, errors)
    selected_local = stable_ranking(
        -ranks,
        group.action_hash[eligible],
        np.arange(len(eligible), dtype=np.int32),
    )[:quota]
    selected = eligible[selected_local]
    nominal_set = set(map(int, nominal_target))
    selected_set = set(map(int, selected))
    return {
        "overlap": len(nominal_set & selected_set),
        "target_total": quota,
        "exact_target": nominal_set == selected_set,
        "observation": r4800_observation(
            group,
            np.concatenate([frontier, selected]),
        ),
        "phase": group.phase,
        "candidates": group.candidate_count,
    }


def audit_expected_rank(
    dataset: GradedOracleDataset,
    *,
    workers: int = 1,
) -> dict[str, Any]:
    """Evaluate an uncertainty-aware expected-rank target ceiling."""
    overall = CeilingAccumulator()
    phases = {name: CeilingAccumulator() for name in _PHASE_NAMES.values()}
    target_recalled = 0
    target_total = 0
    exact_targets = 0
    groups = 0
    candidates = 0
    for result in parallel_group_map(
        expected_rank_group,
        iter_supervision_groups(dataset),
        workers,
    ):
        target_recalled += int(result["overlap"])
        target_total += int(result["target_total"])
        exact_targets += int(result["exact_target"])
        overall.add(result["observation"])
        phase_name = _PHASE_NAMES.get(int(result["phase"]))
        if phase_name is None:
            raise ValueError("expected-rank audit received an invalid phase")
        phases[phase_name].add(result["observation"])
        groups += 1
        candidates += int(result["candidates"])
    report = {
        "groups": groups,
        "candidates": candidates,
        "nominal_target_recall": target_recalled / max(target_total, 1),
        "nominal_target_exact_fraction": exact_targets / max(groups, 1),
        "overall": overall.report(),
        "phase": {name: accumulator.report() for name, accumulator in phases.items()},
    }
    report["gate_passed"] = expected_rank_ceiling_gate(report)
    return report


def expected_normal_ranks(means: np.ndarray, errors: np.ndarray) -> np.ndarray:
    """Compute 1 + the expected number of peers beating each action."""
    if means.ndim != 1 or errors.shape != means.shape or not len(means):
        raise ValueError("expected-rank arrays are invalid")
    result = np.empty(len(means), dtype=np.float64)
    for start in range(0, len(means), 128):
        stop = min(start + 128, len(means))
        denominator = np.hypot(
            errors[start:stop, None],
            errors[None, :],
        )
        differences = means[None, :] - means[start:stop, None]
        z_scores = np.divide(
            differences,
            denominator,
            out=np.zeros_like(differences),
            where=denominator > 0,
        )
        probabilities = normal_cdf(z_scores)
        probabilities = np.where(
            denominator > 0,
            probabilities,
            np.where(differences > 0, 1.0, np.where(differences < 0, 0.0, 0.5)),
        )
        result[start:stop] = np.sum(probabilities, axis=1) + 0.5
    return result


def normal_cdf(values: np.ndarray) -> np.ndarray:
    """Vectorized normal CDF using the A&S 7.1.26 approximation."""
    absolute = np.abs(np.asarray(values, dtype=np.float64))
    t = 1.0 / (1.0 + 0.2316419 * absolute)
    polynomial = t * (
        0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))
    )
    density = np.exp(-0.5 * absolute**2) / math.sqrt(2.0 * math.pi)
    positive = 1.0 - density * polynomial
    return np.where(values >= 0.0, positive, 1.0 - positive)


def r4800_observation(
    group: SupervisionGroup,
    retained: np.ndarray,
) -> dict[str, bool | float]:
    """Measure the unchanged R4800 width-64 ceiling outcome."""
    labeled = np.flatnonzero(group.r4800_samples > 0).astype(np.int32)
    if len(labeled) < 2:
        raise ValueError("R4800 ceiling requires at least two labeled actions")
    ranking = stable_ranking(group.r4800_mean, group.action_hash, labeled)
    winner = int(ranking[0])
    if winner != group.selected_index:
        raise ValueError("stored selected action is not the stable R4800 winner")
    runner_up = int(ranking[1])
    errors = standard_error(group.r4800_stddev, group.r4800_samples)
    distinguishable = group.r4800_mean[winner] - group.r4800_mean[runner_up] > NORMAL_95 * np.hypot(
        errors[winner], errors[runner_up]
    )
    confidence = np.zeros(group.candidate_count, dtype=np.bool_)
    confidence[labeled] = group.r4800_mean[winner] - group.r4800_mean[
        labeled
    ] <= NORMAL_95 * np.hypot(errors[winner], errors[labeled])
    retained_labeled = retained[group.r4800_samples[retained] > 0]
    regret = (
        float(group.r4800_mean[winner] - np.max(group.r4800_mean[retained_labeled]))
        if len(retained_labeled)
        else float(np.ptp(group.r4800_mean[labeled]))
    )
    return {
        "exact": bool(np.any(retained == winner)),
        "confidence": bool(np.any(confidence[retained])),
        "distinguishable": bool(distinguishable),
        "regret": regret,
    }


def expected_rank_ceiling_gate(report: dict[str, Any]) -> bool:
    """Apply the frozen ADR 0099 soft-ceiling thresholds."""
    overall = report["overall"]
    distinguishable = overall["top64_distinguishable_winner_recall"]
    return (
        overall["top64_r4800_winner_recall"] > 0.98
        and overall["top64_confidence_set_coverage_95"] >= 0.99
        and distinguishable is not None
        and distinguishable >= 0.98
        and overall["mean_top64_retained_r4800_regret"] < 0.03
        and all(
            values["top64_r4800_winner_recall"] >= 0.98
            and values["top64_confidence_set_coverage_95"] >= 0.98
            and values["mean_top64_retained_r4800_regret"] < 0.03
            for values in report["phase"].values()
        )
    )


def rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Return Pearson correlation of stable ordinal ranks."""
    if len(left) < 2 or right.shape != left.shape:
        raise ValueError("rank correlation requires paired arrays")
    left_rank = np.argsort(np.argsort(left, kind="stable"), kind="stable")
    right_rank = np.argsort(np.argsort(right, kind="stable"), kind="stable")
    left_centered = left_rank - np.mean(left_rank)
    right_centered = right_rank - np.mean(right_rank)
    denominator = float(np.sqrt(np.sum(left_centered**2) * np.sum(right_centered**2)))
    return float(np.sum(left_centered * right_centered) / denominator) if denominator > 0 else 0.0


def distribution(values: list[float]) -> dict[str, float | int]:
    """Return deterministic descriptive statistics."""
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.all(np.isfinite(array)):
        raise ValueError("distribution requires finite values")
    return {
        "count": len(array),
        "mean": float(np.mean(array)),
        "min": float(np.min(array)),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(np.max(array)),
    }


def distribution_or_empty(
    values: list[float],
) -> dict[str, float | int | None]:
    """Return descriptive statistics or an explicit empty distribution."""
    if values:
        return distribution(values)
    return {
        "count": 0,
        "mean": None,
        "min": None,
        "p10": None,
        "median": None,
        "p90": None,
        "max": None,
    }


def parallel_group_map(
    function: Any,
    groups: Iterator[SupervisionGroup],
    workers: int,
) -> Iterator[dict[str, Any]]:
    """Map groups in deterministic input order across bounded CPU workers."""
    if workers <= 0:
        raise ValueError("audit workers must be positive")
    if workers == 1:
        yield from map(function, groups)
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        yield from executor.map(function, groups, chunksize=1)


def run_audit(
    *,
    kind: str,
    train_dataset_root: Path,
    validation_dataset_root: Path,
    workers: int = AUDIT_WORKERS,
) -> dict[str, Any]:
    """Run one frozen origin or replay audit."""
    if kind not in AUDIT_KINDS:
        raise ValueError("unsupported supervision-identifiability audit")
    train = GradedOracleDataset(train_dataset_root)
    validation = GradedOracleDataset(validation_dataset_root)
    expected_manifests = {
        "train": "7ed12c943d75a786ccd4ccbe11a6b0146aad4fe5ed40f0cbaf1d652f5ac0bb99",
        "validation": "302ceb7a57482b0fb5fb12963521be35aafc121a36f572e6b9f47def1b820a31",
    }
    identities = {
        "train": checksum(train.root / "dataset.json"),
        "validation": checksum(validation.root / "dataset.json"),
    }
    if identities != expected_manifests:
        raise ValueError("ADR 0099 dataset identity drifted")
    function = {
        BOUNDARY_SIGNAL: audit_boundary,
        CROSS_FIDELITY: audit_cross_fidelity,
        TEACHER_RESAMPLING: audit_resampling,
        EXPECTED_RANK_CEILING: audit_expected_rank,
    }[kind]
    started = time.perf_counter()
    scientific = {
        "kind": kind,
        "train_dataset_manifest_blake3": identities["train"],
        "validation_dataset_manifest_blake3": identities["validation"],
        "workers": workers,
        "train": function(train, workers=workers),
        "validation": function(validation, workers=workers),
        "test_split_opened": False,
        "gameplay_opened": False,
        "new_teacher_compute_used": False,
        "external_compute_used": False,
    }
    elapsed = time.perf_counter() - started
    usage = resource_usage()
    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "host": canonical_host(),
        "scientific": scientific,
        "scientific_blake3": scientific_blake3(scientific),
        "execution": {
            "elapsed_seconds": elapsed,
            "workers": workers,
            "candidates_per_second": (train.candidate_count + validation.candidate_count)
            / max(elapsed, 1e-9),
            **usage,
        },
    }
    return report


def scientific_blake3(value: dict[str, Any]) -> str:
    return blake3.blake3(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def checksum(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def resource_usage() -> dict[str, int]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    peak_rss = max(int(usage.ru_maxrss), int(children.ru_maxrss))
    if platform.system() != "Darwin":
        peak_rss *= 1024
    return {
        "peak_process_rss_bytes": peak_rss,
        "process_swaps": int(getattr(usage, "ru_nswap", 0) + getattr(children, "ru_nswap", 0)),
    }


def canonical_host() -> str:
    host = socket.gethostname().split(".")[0].lower()
    return "john1" if host == "johns-mac-mini" else host


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=AUDIT_KINDS, required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument(
        "--workers",
        type=int,
        choices=[AUDIT_WORKERS],
        default=AUDIT_WORKERS,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_audit(
        kind=args.kind,
        train_dataset_root=args.train_dataset,
        validation_dataset_root=args.validation_dataset,
        workers=args.workers,
    )
    write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
