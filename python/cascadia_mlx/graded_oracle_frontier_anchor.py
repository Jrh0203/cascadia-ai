"""Frontier-anchored set supervision and evaluation for complete actions."""

from __future__ import annotations

import platform
import re
import resource
import subprocess
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import numpy as np

from cascadia_mlx.graded_oracle_dataset import (
    GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
    GRADED_ORACLE_PACKED_ACTION_LIMIT,
    GradedOracleDataset,
)
from cascadia_mlx.graded_oracle_identifiability import NORMAL_95
from cascadia_mlx.graded_oracle_model import (
    GradedOracleRanker,
    predict_graded_oracle_batch,
)

FRONTIER_ANCHORED_WIDTH = 64
GRADED_SOURCE_CHAMPION_FRONTIER = 1 << 1
_PHASE_NAMES = {0: "early", 1: "middle", 2: "late"}
_SWAP_USED_RE = re.compile(r"used = ([0-9.]+)([KMG])")


@dataclass
class _SliceAccumulator:
    groups: int = 0
    exact: int = 0
    confidence: int = 0
    distinguishable_groups: int = 0
    distinguishable_exact: int = 0
    regret: float = 0.0

    def add(
        self,
        *,
        exact: bool,
        confidence: bool,
        distinguishable: bool,
        regret: float,
    ) -> None:
        self.groups += 1
        self.exact += int(exact)
        self.confidence += int(confidence)
        self.regret += regret
        if distinguishable:
            self.distinguishable_groups += 1
            self.distinguishable_exact += int(exact)

    def report(self) -> dict[str, float | int | None]:
        groups = max(self.groups, 1)
        return {
            "groups": self.groups,
            "top64_r4800_winner_recall": self.exact / groups,
            "top64_confidence_set_coverage_95": self.confidence / groups,
            "top64_distinguishable_winner_recall": (
                self.distinguishable_exact / self.distinguishable_groups
                if self.distinguishable_groups
                else None
            ),
            "distinguishable_groups": self.distinguishable_groups,
            "mean_top64_retained_r4800_regret": self.regret / groups,
        }


@dataclass
class _EvaluationAccumulator:
    groups: int = 0
    candidates: int = 0
    nonfinite_scores: int = 0
    total_loss: float = 0.0
    frontier_counts: list[float] = field(default_factory=list)
    target_positive_slots: int = 0
    target_positive_recalled: int = 0
    target_set_exact_groups: int = 0
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


def build_frontier_anchored_target_mask(
    *,
    r1200_mean: np.ndarray,
    r1200_mask: np.ndarray,
    source_flags: np.ndarray,
    candidate_mask: np.ndarray,
    action_hashes: np.ndarray,
    width: int = FRONTIER_ANCHORED_WIDTH,
) -> np.ndarray:
    """Build the non-frontier R1200 set that fills a frontier-anchored top K."""
    shape = candidate_mask.shape
    if (
        r1200_mean.shape != shape
        or r1200_mask.shape != shape
        or source_flags.shape != shape
        or action_hashes.shape[:2] != shape
    ):
        raise ValueError("frontier-anchored target arrays have inconsistent shapes")
    target = np.zeros(shape, dtype=np.bool_)
    for group_index, mask in enumerate(candidate_mask):
        count = int(np.sum(mask))
        if count == 0:
            raise ValueError("frontier-anchored target received an empty group")
        indices = np.arange(count, dtype=np.int32)
        frontier = indices[
            (source_flags[group_index, :count] & GRADED_SOURCE_CHAMPION_FRONTIER)
            != 0
        ]
        if len(frontier) > min(width, count):
            raise ValueError("champion frontier exceeds the anchored proposal width")
        quota = min(width, count) - len(frontier)
        labeled_nonfrontier = indices[
            r1200_mask[group_index, :count]
            & (
                (
                    source_flags[group_index, :count]
                    & GRADED_SOURCE_CHAMPION_FRONTIER
                )
                == 0
            )
        ]
        if len(labeled_nonfrontier) < quota:
            raise ValueError("R1200 non-frontier cohort cannot fill the anchored width")
        ranking = _stable_subset_ranking(
            r1200_mean[group_index, :count],
            action_hashes[group_index, :count],
            labeled_nonfrontier,
        )
        target[group_index, ranking[:quota]] = True
    return target


def frontier_anchored_retained_indices(
    *,
    scores: np.ndarray,
    source_flags: np.ndarray,
    action_hashes: np.ndarray,
    width: int = FRONTIER_ANCHORED_WIDTH,
) -> np.ndarray:
    """Retain every champion-frontier action, then fill with scored actions."""
    count = len(scores)
    if (
        len(source_flags) != count
        or len(action_hashes) != count
        or count == 0
    ):
        raise ValueError("frontier-anchored ranking arrays have inconsistent lengths")
    indices = np.arange(count, dtype=np.int32)
    frontier_mask = (
        source_flags[:count] & GRADED_SOURCE_CHAMPION_FRONTIER
    ) != 0
    frontier = _stable_subset_ranking(scores, action_hashes, indices[frontier_mask])
    if len(frontier) > min(width, count):
        raise ValueError("champion frontier exceeds the anchored proposal width")
    nonfrontier = _stable_subset_ranking(
        scores,
        action_hashes,
        indices[~frontier_mask],
    )
    quota = min(width, count) - len(frontier)
    return np.concatenate([frontier, nonfrontier[:quota]])


def frontier_anchored_loss_components(
    model: GradedOracleRanker,
    batch: object,
) -> dict[str, mx.array]:
    """Compute the frozen set-valued proposer objective."""
    prediction = predict_graded_oracle_batch(model, batch)
    target = mx.array(_target_mask_from_batch(batch))
    frontier = (
        batch.source_flags.astype(mx.int32) & GRADED_SOURCE_CHAMPION_FRONTIER
    ) != 0
    eligible = batch.candidate_mask & ~frontier
    r1200_nonfrontier = batch.r1200_mask & eligible
    scored = batch.r600_mask | batch.r1200_mask | batch.r4800_mask
    screen_only_nonfrontier = eligible & ~scored
    return {
        "target_set_cross_entropy": _uniform_set_cross_entropy(
            prediction.scores,
            eligible,
            target,
        ),
        "r1200_listwise": _masked_soft_target_cross_entropy(
            prediction.scores,
            batch.r1200_mean,
            r1200_nonfrontier,
            temperature=2.0,
        ),
        "screen_only_regularization": _masked_mean(
            prediction.residuals**2,
            screen_only_nonfrontier,
        ),
    }


def frontier_anchored_loss(
    model: GradedOracleRanker,
    batch: object,
) -> mx.array:
    """Train set coverage first, retain graded R1200 order, and bound drift."""
    components = frontier_anchored_loss_components(model, batch)
    return (
        components["target_set_cross_entropy"]
        + 0.5 * components["r1200_listwise"]
        + 0.01 * components["screen_only_regularization"]
    )


def evaluate_frontier_anchored(
    model: GradedOracleRanker,
    dataset: GradedOracleDataset,
    group_batch_size: int,
) -> dict[str, Any]:
    """Evaluate the exact width-64 frontier union without opening sealed data."""
    model.eval()
    accumulator = _EvaluationAccumulator()
    for batch in dataset.batches(
        group_batch_size,
        maximum_actions_per_batch=GRADED_ORACLE_PACKED_ACTION_LIMIT,
        maximum_group_actions=GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
    ):
        prediction = predict_graded_oracle_batch(model, batch)
        loss = frontier_anchored_loss(model, batch)
        mx.eval(prediction.scores, loss)
        scores = np.asarray(prediction.scores)
        masks = np.asarray(batch.candidate_mask)
        screen = np.asarray(batch.screen_value)
        source_flags = np.asarray(batch.source_flags)
        action_hashes = np.asarray(batch.action_hash)
        r1200_mean = np.asarray(batch.r1200_mean)
        r1200_mask = np.asarray(batch.r1200_mask)
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
        targets = build_frontier_anchored_target_mask(
            r1200_mean=r1200_mean,
            r1200_mask=r1200_mask,
            source_flags=source_flags,
            candidate_mask=masks,
            action_hashes=action_hashes,
        )
        accumulator.total_loss += float(loss.item()) * len(scores)

        for group_index, mask in enumerate(masks):
            count = int(np.sum(mask))
            group_scores = scores[group_index, :count]
            group_screen = screen[group_index, :count]
            group_flags = source_flags[group_index, :count]
            group_hashes = action_hashes[group_index, :count]
            group_r4800 = r4800_mean[group_index, :count]
            group_r4800_stddev = r4800_stddev[group_index, :count]
            group_r4800_samples = r4800_samples[group_index, :count]
            group_r4800_mask = r4800_mask[group_index, :count]
            winner = int(selected[group_index])
            target = targets[group_index, :count]
            model_retained = frontier_anchored_retained_indices(
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
                [frontier, np.flatnonzero(target).astype(np.int32)]
            )
            observation = _decision_observation(
                retained=model_retained,
                winner=winner,
                r4800_mean=group_r4800,
                r4800_stddev=group_r4800_stddev,
                r4800_samples=group_r4800_samples,
                r4800_mask=group_r4800_mask,
                action_hashes=group_hashes,
            )
            screen_observation = _decision_observation(
                retained=screen_retained,
                winner=winner,
                r4800_mean=group_r4800,
                r4800_stddev=group_r4800_stddev,
                r4800_samples=group_r4800_samples,
                r4800_mask=group_r4800_mask,
                action_hashes=group_hashes,
            )
            ceiling_observation = _decision_observation(
                retained=target_retained,
                winner=winner,
                r4800_mean=group_r4800,
                r4800_stddev=group_r4800_stddev,
                r4800_samples=group_r4800_samples,
                r4800_mask=group_r4800_mask,
                action_hashes=group_hashes,
            )
            accumulator.model.add(**observation)
            accumulator.screen.add(**screen_observation)
            accumulator.target_ceiling.add(**ceiling_observation)
            phase = _PHASE_NAMES.get(int(phases[group_index]))
            if phase is None:
                raise ValueError("frontier-anchored group has an invalid phase")
            accumulator.phases[phase].add(**observation)
            if int(tokens[group_index]) > 0:
                accumulator.subsets["nature_token_available"].add(**observation)
            if int(draft_kind[group_index, winner]) == 1:
                accumulator.subsets["independent_draft_winner"].add(**observation)
                accumulator.action_families["independent"].add(**observation)
            else:
                accumulator.action_families["paired"].add(**observation)
            if int(same_slot[group_index, winner]) != 0:
                accumulator.action_families["same_slot_independent"].add(
                    **observation
                )
            if int(free_refresh[group_index, winner]) != 0:
                accumulator.action_families["free_refresh"].add(**observation)
            if int(wipe_count[group_index, winner]) > 0:
                accumulator.action_families["paid_wipe"].add(**observation)

            retained_nonfrontier = model_retained[
                (
                    group_flags[model_retained]
                    & GRADED_SOURCE_CHAMPION_FRONTIER
                )
                == 0
            ]
            recalled_targets = int(np.sum(target[retained_nonfrontier]))
            target_count = int(np.sum(target))
            accumulator.target_positive_slots += target_count
            accumulator.target_positive_recalled += recalled_targets
            accumulator.target_set_exact_groups += int(
                recalled_targets == target_count
            )
            accumulator.frontier_counts.append(float(len(frontier)))
            accumulator.nonfinite_scores += int(
                np.sum(~np.isfinite(group_scores))
            )
            accumulator.groups += 1
            accumulator.candidates += count

    if accumulator.groups == 0:
        raise ValueError("frontier-anchored evaluation dataset is empty")
    model_report = accumulator.model.report()
    return {
        "groups": accumulator.groups,
        "candidates": accumulator.candidates,
        "expected_groups": dataset.group_count,
        "expected_candidates": dataset.candidate_count,
        "all_groups_scored_once": accumulator.groups == dataset.group_count,
        "all_candidates_scored_once": accumulator.candidates == dataset.candidate_count,
        "nonfinite_scores": accumulator.nonfinite_scores,
        "all_scores_finite": accumulator.nonfinite_scores == 0,
        "training_objective": accumulator.total_loss / accumulator.groups,
        "proposal_width": FRONTIER_ANCHORED_WIDTH,
        "frontier_count": _distribution(accumulator.frontier_counts),
        "target_positive_recall": (
            accumulator.target_positive_recalled
            / max(accumulator.target_positive_slots, 1)
        ),
        "target_set_exact_fraction": (
            accumulator.target_set_exact_groups / accumulator.groups
        ),
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
    }


def evaluate_frontier_anchored_target_ceiling(
    dataset: GradedOracleDataset,
) -> dict[str, Any]:
    """Measure the deterministic frontier-plus-R1200 target upper bound."""
    overall = _SliceAccumulator()
    phases = {
        name: _SliceAccumulator() for name in _PHASE_NAMES.values()
    }
    subsets = {
        "nature_token_available": _SliceAccumulator(),
        "independent_draft_winner": _SliceAccumulator(),
    }
    frontier_counts: list[float] = []
    groups = 0
    candidates = 0
    for batch in dataset.batches(
        64,
        maximum_actions_per_batch=GRADED_ORACLE_PACKED_ACTION_LIMIT,
        maximum_group_actions=GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
    ):
        masks = np.asarray(batch.candidate_mask)
        source_flags = np.asarray(batch.source_flags)
        action_hashes = np.asarray(batch.action_hash)
        r1200_mean = np.asarray(batch.r1200_mean)
        r1200_mask = np.asarray(batch.r1200_mask)
        r4800_mean = np.asarray(batch.r4800_mean)
        r4800_stddev = np.asarray(batch.r4800_stddev)
        r4800_samples = np.asarray(batch.r4800_samples)
        r4800_mask = np.asarray(batch.r4800_mask)
        selected = np.asarray(batch.selected_index)
        phase_values = np.asarray(batch.phase)
        tokens = np.asarray(batch.active_nature_tokens)
        draft_kind = np.asarray(batch.draft_kind)
        targets = build_frontier_anchored_target_mask(
            r1200_mean=r1200_mean,
            r1200_mask=r1200_mask,
            source_flags=source_flags,
            candidate_mask=masks,
            action_hashes=action_hashes,
        )
        for group_index, mask in enumerate(masks):
            count = int(np.sum(mask))
            flags = source_flags[group_index, :count]
            frontier = np.flatnonzero(
                (flags & GRADED_SOURCE_CHAMPION_FRONTIER) != 0
            ).astype(np.int32)
            retained = np.concatenate(
                [
                    frontier,
                    np.flatnonzero(targets[group_index, :count]).astype(np.int32),
                ]
            )
            if len(retained) != min(FRONTIER_ANCHORED_WIDTH, count):
                raise ValueError("frontier-anchored target ceiling has wrong width")
            winner = int(selected[group_index])
            observation = _decision_observation(
                retained=retained,
                winner=winner,
                r4800_mean=r4800_mean[group_index, :count],
                r4800_stddev=r4800_stddev[group_index, :count],
                r4800_samples=r4800_samples[group_index, :count],
                r4800_mask=r4800_mask[group_index, :count],
                action_hashes=action_hashes[group_index, :count],
            )
            overall.add(**observation)
            phase = _PHASE_NAMES.get(int(phase_values[group_index]))
            if phase is None:
                raise ValueError("frontier-anchored group has an invalid phase")
            phases[phase].add(**observation)
            if int(tokens[group_index]) > 0:
                subsets["nature_token_available"].add(**observation)
            if int(draft_kind[group_index, winner]) == 1:
                subsets["independent_draft_winner"].add(**observation)
            frontier_counts.append(float(len(frontier)))
            groups += 1
            candidates += count
    if groups == 0:
        raise ValueError("frontier-anchored target ceiling received no groups")
    return {
        "schema_version": 1,
        "split": dataset.split,
        "dataset_id": dataset.manifest["dataset_id"],
        "groups": groups,
        "candidates": candidates,
        "expected_groups": dataset.group_count,
        "expected_candidates": dataset.candidate_count,
        "all_groups_seen_once": groups == dataset.group_count,
        "all_candidates_seen_once": candidates == dataset.candidate_count,
        "proposal_width": FRONTIER_ANCHORED_WIDTH,
        "frontier_count": _distribution(frontier_counts),
        "overall": overall.report(),
        "phase": {
            name: values.report() for name, values in phases.items()
        },
        "subsets": {
            name: values.report() for name, values in subsets.items()
        },
        "test_split_opened": dataset.split == "test",
    }


def frontier_anchored_target_ceiling_gates(
    report: dict[str, Any],
) -> dict[str, bool]:
    """Require enough target headroom before any proposer training starts."""
    overall = report["overall"]
    distinguishable = overall["top64_distinguishable_winner_recall"]
    gates = {
        "top64_exact_recall_strictly_greater_than_0_98": (
            float(overall["top64_r4800_winner_recall"]) > 0.98
        ),
        "top64_confidence_coverage_at_least_0_99": (
            float(overall["top64_confidence_set_coverage_95"]) >= 0.99
        ),
        "top64_distinguishable_recall_at_least_0_98": (
            distinguishable is not None and float(distinguishable) >= 0.98
        ),
        "mean_retained_regret_below_0_03": (
            float(overall["mean_top64_retained_r4800_regret"]) < 0.03
        ),
        "every_phase_exact_recall_at_least_0_98": all(
            float(values["top64_r4800_winner_recall"]) >= 0.98
            for values in report["phase"].values()
        ),
        "every_phase_confidence_coverage_at_least_0_98": all(
            float(values["top64_confidence_set_coverage_95"]) >= 0.98
            for values in report["phase"].values()
        ),
        "every_phase_retained_regret_below_0_03": all(
            float(values["mean_top64_retained_r4800_regret"]) < 0.03
            for values in report["phase"].values()
        ),
        "every_eligible_subset_exact_recall_at_least_0_95": all(
            int(values["groups"]) < 20
            or float(values["top64_r4800_winner_recall"]) >= 0.95
            for values in report["subsets"].values()
        ),
        "every_eligible_subset_retained_regret_below_0_25": all(
            int(values["groups"]) < 20
            or float(values["mean_top64_retained_r4800_regret"]) < 0.25
            for values in report["subsets"].values()
        ),
        "all_groups_seen_once": bool(report["all_groups_seen_once"]),
        "all_candidates_seen_once": bool(report["all_candidates_seen_once"]),
        "proposal_width_is_exactly_64": int(report["proposal_width"]) == 64,
        "sealed_test_unopened": not bool(report["test_split_opened"]),
    }
    gates["target_ceiling_passed"] = all(gates.values())
    return gates


def benchmark_frontier_anchored(
    model: GradedOracleRanker,
    dataset: GradedOracleDataset,
    *,
    maximum_groups: int | None = None,
    warmup_groups: int = 3,
) -> dict[str, Any]:
    """Measure complete model scoring plus deterministic anchored selection."""
    model.eval()
    batches = dataset.batches(
        1,
        maximum_actions_per_batch=GRADED_ORACLE_PACKED_ACTION_LIMIT,
        maximum_group_actions=GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
    )
    warmed: list[object] = []
    for batch in batches:
        prediction = predict_graded_oracle_batch(model, batch)
        mx.eval(prediction.scores)
        warmed.append(batch)
        if len(warmed) >= warmup_groups:
            break
    if not warmed:
        raise ValueError("frontier-anchored benchmark dataset is empty")

    swap_before = _system_swap_used_bytes()
    latencies: list[float] = []
    scored_actions = 0
    nonfinite_scores = 0
    measured = 0
    for batch in _chain(warmed, batches):
        if maximum_groups is not None and measured >= maximum_groups:
            break
        started = time.perf_counter()
        prediction = predict_graded_oracle_batch(model, batch)
        mx.eval(prediction.scores)
        mask = np.asarray(batch.candidate_mask)[0]
        scores = np.asarray(prediction.scores)[0][mask]
        source_flags = np.asarray(batch.source_flags)[0][mask]
        action_hashes = np.asarray(batch.action_hash)[0][mask]
        retained = frontier_anchored_retained_indices(
            scores=scores,
            source_flags=source_flags,
            action_hashes=action_hashes,
        )
        elapsed = time.perf_counter() - started
        if len(retained) != min(FRONTIER_ANCHORED_WIDTH, len(scores)):
            raise ValueError("frontier-anchored benchmark retained the wrong width")
        latencies.append(elapsed)
        scored_actions += len(scores)
        nonfinite_scores += int(np.sum(~np.isfinite(scores)))
        measured += 1

    swap_after = _system_swap_used_bytes()
    total_seconds = float(np.sum(latencies))
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss = int(usage.ru_maxrss)
    if platform.system() != "Darwin":
        peak_rss *= 1024
    process_swaps = int(getattr(usage, "ru_nswap", 0))
    swap_delta = (
        None
        if swap_before is None or swap_after is None
        else swap_after - swap_before
    )
    report = {
        "groups": measured,
        "actions": scored_actions,
        "elapsed_seconds": total_seconds,
        "action_scores_per_second": scored_actions / max(total_seconds, 1e-9),
        "mean_decision_milliseconds": 1000.0 * total_seconds / measured,
        "p99_decision_milliseconds": 1000.0 * float(np.quantile(latencies, 0.99)),
        "peak_process_rss_bytes": peak_rss,
        "process_swaps": process_swaps,
        "system_swap_before_bytes": swap_before,
        "system_swap_after_bytes": swap_after,
        "system_swap_delta_bytes": swap_delta,
        "nonfinite_scores": nonfinite_scores,
    }
    report["gates"] = {
        "action_scores_per_second_at_least_20000": (
            report["action_scores_per_second"] >= 20_000.0
        ),
        "p99_decision_milliseconds_at_most_250": (
            report["p99_decision_milliseconds"] <= 250.0
        ),
        "peak_process_rss_at_most_4_gib": peak_rss <= 4 * 1024**3,
        "process_swaps_zero": process_swaps == 0,
        "system_swap_not_consumed": swap_delta is not None and swap_delta <= 0,
        "all_scores_finite": nonfinite_scores == 0,
    }
    report["passed"] = all(report["gates"].values())
    return report


def frontier_anchored_validation_gates(
    metrics: dict[str, Any],
    performance_by_host: dict[str, dict[str, Any]] | None = None,
) -> dict[str, bool]:
    """Apply the frozen validation and integrity thresholds."""
    distinguishable = metrics["top64_distinguishable_winner_recall"]
    gates = {
        "top64_r4800_winner_recall_strictly_greater_than_0_98": (
            float(metrics["top64_r4800_winner_recall"]) > 0.98
        ),
        "top64_confidence_set_coverage_at_least_0_99": (
            float(metrics["top64_confidence_set_coverage_95"]) >= 0.99
        ),
        "top64_distinguishable_winner_recall_at_least_0_98": (
            distinguishable is not None and float(distinguishable) >= 0.98
        ),
        "mean_top64_retained_r4800_regret_strictly_less_than_0_15": (
            float(metrics["mean_top64_retained_r4800_regret"]) < 0.15
        ),
        "all_groups_scored_once": bool(metrics["all_groups_scored_once"]),
        "all_candidates_scored_once": bool(metrics["all_candidates_scored_once"]),
        "all_scores_finite": bool(metrics["all_scores_finite"]),
        "proposal_width_is_exactly_64": int(metrics["proposal_width"]) == 64,
    }
    for name, values in metrics["phase"].items():
        gates[f"{name}_top64_recall_at_least_0_97"] = (
            float(values["top64_r4800_winner_recall"]) >= 0.97
        )
        gates[f"{name}_confidence_set_coverage_at_least_0_98"] = (
            float(values["top64_confidence_set_coverage_95"]) >= 0.98
        )
        gates[f"{name}_retained_regret_strictly_less_than_0_20"] = (
            float(values["mean_top64_retained_r4800_regret"]) < 0.20
        )
    for name in ("nature_token_available", "independent_draft_winner"):
        values = metrics["subsets"][name]
        if int(values["groups"]) >= 20:
            gates[f"{name}_top64_recall_at_least_0_95"] = (
                float(values["top64_r4800_winner_recall"]) >= 0.95
            )
            gates[f"{name}_retained_regret_strictly_less_than_0_25"] = (
                float(values["mean_top64_retained_r4800_regret"]) < 0.25
            )
    if performance_by_host is not None:
        for host, performance in performance_by_host.items():
            gates[f"{host}_performance_passed"] = bool(performance["passed"])
    return gates


def _target_mask_from_batch(batch: object) -> np.ndarray:
    return build_frontier_anchored_target_mask(
        r1200_mean=np.asarray(batch.r1200_mean),
        r1200_mask=np.asarray(batch.r1200_mask),
        source_flags=np.asarray(batch.source_flags),
        candidate_mask=np.asarray(batch.candidate_mask),
        action_hashes=np.asarray(batch.action_hash),
    )


def _decision_observation(
    *,
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
        raise ValueError("frontier-anchored evaluation requires two R4800 actions")
    ranking = _stable_subset_ranking(r4800_mean, action_hashes, labeled)
    if int(ranking[0]) != winner:
        raise ValueError("selected action is not the stable R4800 winner")
    runner_up = int(ranking[1])
    standard_error = r4800_stddev / np.sqrt(np.maximum(r4800_samples, 1.0))
    distinguishable = float(r4800_mean[winner] - r4800_mean[runner_up]) > (
        NORMAL_95
        * float(np.hypot(standard_error[winner], standard_error[runner_up]))
    )
    confidence_set = np.zeros(len(r4800_mean), dtype=np.bool_)
    winner_se = standard_error[winner]
    differences = r4800_mean[winner] - r4800_mean[labeled]
    thresholds = NORMAL_95 * np.hypot(winner_se, standard_error[labeled])
    confidence_set[labeled] = differences <= thresholds
    retained_labeled = retained[r4800_mask[retained]]
    regret = (
        float(r4800_mean[winner] - np.max(r4800_mean[retained_labeled]))
        if len(retained_labeled)
        else float(np.ptp(r4800_mean[labeled]))
    )
    return {
        "exact": bool(np.any(retained == winner)),
        "confidence": bool(np.any(confidence_set[retained])),
        "distinguishable": distinguishable,
        "regret": regret,
    }


def _uniform_set_cross_entropy(
    scores: mx.array,
    eligible_mask: mx.array,
    target_mask: mx.array,
) -> mx.array:
    target_count = mx.sum(target_mask, axis=-1, keepdims=True)
    masked_scores = mx.where(eligible_mask, scores, -1e9)
    log_probabilities = masked_scores - mx.logsumexp(
        masked_scores,
        axis=-1,
        keepdims=True,
    )
    target_weights = target_mask.astype(scores.dtype) / target_count
    return mx.mean(-mx.sum(target_weights * log_probabilities, axis=-1))


def _masked_soft_target_cross_entropy(
    scores: mx.array,
    targets: mx.array,
    mask: mx.array,
    *,
    temperature: float,
) -> mx.array:
    masked_scores = mx.where(mask, scores / temperature, -1e9)
    masked_targets = mx.where(mask, targets / temperature, -1e9)
    target_probabilities = mx.softmax(masked_targets, axis=-1)
    log_probabilities = masked_scores - mx.logsumexp(
        masked_scores,
        axis=-1,
        keepdims=True,
    )
    per_group = -mx.sum(
        mx.where(mask, target_probabilities * log_probabilities, 0.0),
        axis=-1,
    )
    return _masked_mean(per_group, mx.any(mask, axis=-1))


def _masked_mean(values: mx.array, mask: mx.array) -> mx.array:
    weights = mask.astype(values.dtype)
    return mx.sum(values * weights) / mx.maximum(mx.sum(weights), 1.0)


def _stable_subset_ranking(
    scores: np.ndarray,
    action_hashes: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        sorted(
            (int(index) for index in indices),
            key=lambda index: (-float(scores[index]), bytes(action_hashes[index])),
        ),
        dtype=np.int32,
    )


def _distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(array),
        "mean": float(np.mean(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "p95": float(np.quantile(array, 0.95)),
    }


def _system_swap_used_bytes() -> int | None:
    if platform.system() != "Darwin":
        return None
    try:
        output = subprocess.run(
            ["sysctl", "-n", "vm.swapusage"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    match = _SWAP_USED_RE.search(output)
    if match is None:
        return None
    scale = {"K": 1024, "M": 1024**2, "G": 1024**3}[match.group(2)]
    return int(float(match.group(1)) * scale)


def _chain(
    first: Iterable[object],
    second: Iterable[object],
) -> Iterator[object]:
    yield from first
    yield from second
