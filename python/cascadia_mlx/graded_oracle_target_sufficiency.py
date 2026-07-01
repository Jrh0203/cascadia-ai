"""Audit whether the frozen R1200 cohort can support a robust top-64 proposer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import numpy as np

from cascadia_mlx.checkpoint import load_checkpoint_pointer_with_factory
from cascadia_mlx.graded_oracle_dataset import (
    GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
    GRADED_ORACLE_PACKED_ACTION_LIMIT,
    GradedOracleDataset,
)
from cascadia_mlx.graded_oracle_identifiability import (
    EXPECTED_CHECKPOINT,
    EXPECTED_MANIFEST_BLAKE3,
    EXPECTED_MODEL_BLAKE3,
    EXPECTED_SOURCE_V2_BLAKE3,
    NORMAL_95,
)
from cascadia_mlx.graded_oracle_model import (
    GradedOracleModelConfig,
    GradedOracleRanker,
    predict_graded_oracle_batch,
)

EXPERIMENT_ID = "complete-action-r1200-target-sufficiency-v1"
RANK_WIDTHS = (1, 8, 32, 64)
PHASE_NAMES = {0: "early", 1: "middle", 2: "late"}
HOST_ALIASES = {
    "Johns-Mac-mini": "john1",
    "john1": "john1",
    "john2": "john2",
    "john3": "john3",
}


@dataclass(frozen=True)
class RankingObservation:
    """One ranking's relation to the frozen R4800 target."""

    exact_winner_recalled: bool
    confidence_set_covered: bool
    retained_regret: float


@dataclass(frozen=True)
class DecisionObservation:
    """All frozen target-sufficiency measurements for one decision."""

    candidates: int
    r1200_candidates: int
    r4800_candidates: int
    r1200_confidence_set_size_95: int
    r4800_confidence_set_size_95: int
    confidence_sets_intersect: bool
    r1200_winner_in_r4800_confidence_set_95: bool
    r4800_winner_distinguishable_95: bool
    cohort_has_top64_capacity: bool
    phase: str
    nature_token_available: bool
    independent_draft_winner: bool
    raw_seed: int
    rankings: dict[str, dict[int, RankingObservation]]
    model_top64_label_counts: dict[str, int]


@dataclass
class RankingAccumulator:
    """Aggregate recall and regret for one ranker and width."""

    exact_winner_recalled: int = 0
    confidence_set_covered: int = 0
    retained_regret: float = 0.0
    distinguishable_groups: int = 0
    distinguishable_winner_recalled: int = 0

    def add(self, observation: RankingObservation, distinguishable: bool) -> None:
        self.exact_winner_recalled += int(observation.exact_winner_recalled)
        self.confidence_set_covered += int(observation.confidence_set_covered)
        self.retained_regret += observation.retained_regret
        if distinguishable:
            self.distinguishable_groups += 1
            self.distinguishable_winner_recalled += int(observation.exact_winner_recalled)

    def report(self, groups: int) -> dict[str, Any]:
        return {
            "exact_winner_recall": self.exact_winner_recalled / groups,
            "confidence_set_coverage_95": self.confidence_set_covered / groups,
            "mean_retained_r4800_regret": self.retained_regret / groups,
            "distinguishable_groups": self.distinguishable_groups,
            "distinguishable_winner_recall": (
                self.distinguishable_winner_recalled / self.distinguishable_groups
                if self.distinguishable_groups
                else None
            ),
        }


@dataclass
class AuditAccumulator:
    """Aggregate one complete split or slice."""

    groups: int = 0
    candidates: int = 0
    r1200_candidates: int = 0
    r4800_candidates: int = 0
    confidence_sets_intersect: int = 0
    r1200_winner_in_r4800_confidence_set_95: int = 0
    cohort_has_top64_capacity: int = 0
    r1200_confidence_set_size_95: list[float] = field(default_factory=list)
    r4800_confidence_set_size_95: list[float] = field(default_factory=list)
    rankings: dict[str, dict[int, RankingAccumulator]] = field(
        default_factory=lambda: {
            name: {width: RankingAccumulator() for width in RANK_WIDTHS}
            for name in ("model", "r1200_cohort_oracle", "screen")
        }
    )
    model_top64_label_counts: dict[str, int] = field(
        default_factory=lambda: {
            "r600": 0,
            "r1200": 0,
            "r4800": 0,
            "screen_only": 0,
        }
    )
    model_top64_slots: int = 0

    def add(self, observation: DecisionObservation) -> None:
        self.groups += 1
        self.candidates += observation.candidates
        self.r1200_candidates += observation.r1200_candidates
        self.r4800_candidates += observation.r4800_candidates
        self.confidence_sets_intersect += int(observation.confidence_sets_intersect)
        self.r1200_winner_in_r4800_confidence_set_95 += int(
            observation.r1200_winner_in_r4800_confidence_set_95
        )
        self.cohort_has_top64_capacity += int(observation.cohort_has_top64_capacity)
        self.r1200_confidence_set_size_95.append(float(observation.r1200_confidence_set_size_95))
        self.r4800_confidence_set_size_95.append(float(observation.r4800_confidence_set_size_95))
        for name, widths in observation.rankings.items():
            for width, ranking in widths.items():
                self.rankings[name][width].add(
                    ranking,
                    observation.r4800_winner_distinguishable_95,
                )
        for name, count in observation.model_top64_label_counts.items():
            self.model_top64_label_counts[name] += count
        self.model_top64_slots += min(64, observation.candidates)

    def report(self) -> dict[str, Any]:
        if self.groups == 0:
            raise ValueError("R1200 target-sufficiency audit received no groups")
        return {
            "groups": self.groups,
            "candidates": self.candidates,
            "r1200_candidates": self.r1200_candidates,
            "r4800_candidates": self.r4800_candidates,
            "mean_candidates": self.candidates / self.groups,
            "mean_r1200_candidates": self.r1200_candidates / self.groups,
            "mean_r4800_candidates": self.r4800_candidates / self.groups,
            "r1200_confidence_set_size_95": _distribution(self.r1200_confidence_set_size_95),
            "r4800_confidence_set_size_95": _distribution(self.r4800_confidence_set_size_95),
            "confidence_set_intersection_fraction": (self.confidence_sets_intersect / self.groups),
            "r1200_winner_in_r4800_confidence_set_95_fraction": (
                self.r1200_winner_in_r4800_confidence_set_95 / self.groups
            ),
            "cohort_top64_capacity_fraction": (self.cohort_has_top64_capacity / self.groups),
            "ranking": {
                name: {
                    f"top{width}": accumulator.report(self.groups)
                    for width, accumulator in widths.items()
                }
                for name, widths in self.rankings.items()
            },
            "model_top64_label_composition": {
                name: {
                    "count": count,
                    "fraction": count / max(self.model_top64_slots, 1),
                    "mean_per_group": count / self.groups,
                }
                for name, count in self.model_top64_label_counts.items()
            },
            "model_top64_slots": self.model_top64_slots,
        }


def audit_dataset(
    run_dir: str | Path,
    dataset_path: str | Path,
) -> dict[str, Any]:
    """Run ADR 0087 on one already-open split."""
    started = time.perf_counter()
    run_dir = Path(run_dir)
    dataset = GradedOracleDataset(dataset_path)
    if dataset.split not in EXPECTED_MANIFEST_BLAKE3:
        raise ValueError("ADR 0087 accepts only the train or validation split")
    dataset_manifest = _checksum(dataset.root / "dataset.json")
    if dataset_manifest != EXPECTED_MANIFEST_BLAKE3[dataset.split]:
        raise ValueError("ADR 0087 dataset manifest identity drifted")

    model, checkpoint, run = _load_frozen_model(run_dir, dataset, dataset_manifest)
    overall = AuditAccumulator()
    phases = {name: AuditAccumulator() for name in PHASE_NAMES.values()}
    subsets = {
        "nature_token_available": AuditAccumulator(),
        "independent_draft_winner": AuditAccumulator(),
    }
    games: dict[int, AuditAccumulator] = {}
    groups_seen = 0
    candidates_seen = 0
    nonfinite_model_scores = 0
    nonfinite_teacher_values = 0

    for batch in dataset.batches(
        64,
        maximum_actions_per_batch=GRADED_ORACLE_PACKED_ACTION_LIMIT,
        maximum_group_actions=GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
    ):
        prediction = predict_graded_oracle_batch(model, batch)
        mx.eval(prediction.scores)
        scores = np.asarray(prediction.scores)
        masks = np.asarray(batch.candidate_mask)
        screen = np.asarray(batch.screen_value)
        selected = np.asarray(batch.selected_index)
        hashes = np.asarray(batch.action_hash)
        r600_mean = np.asarray(batch.r600_mean)
        r600_mask = np.asarray(batch.r600_mask)
        r1200_mean = np.asarray(batch.r1200_mean)
        r1200_stddev = np.asarray(batch.r1200_stddev)
        r1200_samples = np.asarray(batch.r1200_samples)
        r1200_mask = np.asarray(batch.r1200_mask)
        r4800_mean = np.asarray(batch.r4800_mean)
        r4800_stddev = np.asarray(batch.r4800_stddev)
        r4800_samples = np.asarray(batch.r4800_samples)
        r4800_mask = np.asarray(batch.r4800_mask)
        phase_values = np.asarray(batch.phase)
        tokens = np.asarray(batch.active_nature_tokens)
        draft_kind = np.asarray(batch.draft_kind)
        raw_seed = np.asarray(batch.game_index)

        for group_index, mask in enumerate(masks):
            count = int(np.sum(mask))
            model_scores = scores[group_index, :count]
            r600_values = r600_mean[group_index, :count]
            r1200_values = r1200_mean[group_index, :count]
            r4800_values = r4800_mean[group_index, :count]
            r600_group_mask = r600_mask[group_index, :count]
            r1200_group_mask = r1200_mask[group_index, :count]
            r4800_group_mask = r4800_mask[group_index, :count]
            nonfinite_model_scores += int(np.sum(~np.isfinite(model_scores)))
            nonfinite_teacher_values += int(
                np.sum(~np.isfinite(r600_values[r600_group_mask]))
                + np.sum(~np.isfinite(r1200_values[r1200_group_mask]))
                + np.sum(~np.isfinite(r4800_values[r4800_group_mask]))
            )
            winner = int(selected[group_index])
            observation = analyze_decision(
                model_scores=model_scores,
                screen_scores=screen[group_index, :count],
                action_hashes=hashes[group_index, :count],
                selected_index=winner,
                r600_mask=r600_group_mask,
                r1200_mean=r1200_values,
                r1200_stddev=r1200_stddev[group_index, :count],
                r1200_samples=r1200_samples[group_index, :count],
                r1200_mask=r1200_group_mask,
                r4800_mean=r4800_values,
                r4800_stddev=r4800_stddev[group_index, :count],
                r4800_samples=r4800_samples[group_index, :count],
                r4800_mask=r4800_group_mask,
                phase=int(phase_values[group_index]),
                nature_token_available=int(tokens[group_index]) > 0,
                independent_draft_winner=(int(draft_kind[group_index, winner]) == 1),
                raw_seed=int(raw_seed[group_index]),
            )
            overall.add(observation)
            phases[observation.phase].add(observation)
            if observation.nature_token_available:
                subsets["nature_token_available"].add(observation)
            if observation.independent_draft_winner:
                subsets["independent_draft_winner"].add(observation)
            games.setdefault(observation.raw_seed, AuditAccumulator()).add(observation)
            groups_seen += 1
            candidates_seen += count

    overall_report = overall.report()
    phase_reports = {name: accumulator.report() for name, accumulator in phases.items()}
    subset_reports = {
        name: accumulator.report() for name, accumulator in subsets.items() if accumulator.groups
    }
    integrity = {
        "split_allowed": dataset.split in {"train", "validation"},
        "groups_seen": groups_seen,
        "expected_groups": dataset.group_count,
        "all_groups_seen_once": groups_seen == dataset.group_count,
        "candidates_seen": candidates_seen,
        "expected_candidates": dataset.candidate_count,
        "all_candidates_seen_once": candidates_seen == dataset.candidate_count,
        "nonfinite_model_scores": nonfinite_model_scores,
        "all_model_scores_finite": nonfinite_model_scores == 0,
        "nonfinite_teacher_values": nonfinite_teacher_values,
        "all_teacher_values_finite": nonfinite_teacher_values == 0,
        "checkpoint_identity_passed": checkpoint.name == EXPECTED_CHECKPOINT,
        "model_identity_passed": (
            _checksum(checkpoint / "model.safetensors") == EXPECTED_MODEL_BLAKE3
        ),
        "dataset_identity_passed": (dataset_manifest == EXPECTED_MANIFEST_BLAKE3[dataset.split]),
        "source_identity_passed": (run["source"]["v2_source_blake3"] == EXPECTED_SOURCE_V2_BLAKE3),
        "test_split_opened": False,
    }
    gates = interpretation_gates(overall_report, phase_reports, integrity)
    scientific = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "dataset": {
            "dataset_id": dataset.manifest["dataset_id"],
            "split": dataset.split,
            "games": dataset.manifest["completed_games"],
            "seeds": dataset.manifest["seeds"],
            "groups": dataset.group_count,
            "candidates": dataset.candidate_count,
            "manifest_blake3": dataset_manifest,
        },
        "model": {
            "checkpoint": checkpoint.name,
            "model_blake3": EXPECTED_MODEL_BLAKE3,
            "source_v2_blake3": EXPECTED_SOURCE_V2_BLAKE3,
            "prior_feature_schema": run["training"]["model"]["prior_feature_schema"],
        },
        "method": {
            "implementation_blake3": _checksum(Path(__file__)),
            "normal_95": NORMAL_95,
            "cohort_oracle": (
                "R1200-labeled actions by descending R1200 mean and action hash, "
                "then unlabeled actions by descending screen value and action hash"
            ),
            "caveat": (
                "The R1200 cohort oracle is an information upper bound, not a "
                "deployable player or score claim. Confidence sets use the same "
                "normal independence approximation as ADR 0086."
            ),
        },
        "overall": overall_report,
        "phases": phase_reports,
        "subsets": subset_reports,
        "games": {str(seed): accumulator.report() for seed, accumulator in sorted(games.items())},
        "integrity": integrity,
        "interpretation_gates": gates,
        "classification": classify_interpretation(gates),
    }
    host_name = socket.gethostname().split(".")[0]
    host = HOST_ALIASES.get(host_name, host_name)
    elapsed = time.perf_counter() - started
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "host": host,
        "scientific": scientific,
        "scientific_blake3": _canonical_digest(scientific),
        "execution": {
            "elapsed_seconds": elapsed,
            "groups_per_second": groups_seen / max(elapsed, 1e-9),
            "test_split_opened": False,
        },
    }


def analyze_decision(
    *,
    model_scores: np.ndarray,
    screen_scores: np.ndarray,
    action_hashes: np.ndarray,
    selected_index: int,
    r600_mask: np.ndarray,
    r1200_mean: np.ndarray,
    r1200_stddev: np.ndarray,
    r1200_samples: np.ndarray,
    r1200_mask: np.ndarray,
    r4800_mean: np.ndarray,
    r4800_stddev: np.ndarray,
    r4800_samples: np.ndarray,
    r4800_mask: np.ndarray,
    phase: int,
    nature_token_available: bool,
    independent_draft_winner: bool,
    raw_seed: int,
) -> DecisionObservation:
    """Measure the frozen cohort oracle and two deployed rankers."""
    count = len(model_scores)
    arrays = (
        screen_scores,
        action_hashes,
        r600_mask,
        r1200_mean,
        r1200_stddev,
        r1200_samples,
        r1200_mask,
        r4800_mean,
        r4800_stddev,
        r4800_samples,
        r4800_mask,
    )
    if any(len(value) != count for value in arrays):
        raise ValueError("ADR 0087 decision arrays have inconsistent lengths")
    if phase not in PHASE_NAMES:
        raise ValueError("ADR 0087 decision has invalid phase")
    if not np.all(np.isfinite(model_scores)) or not np.all(np.isfinite(screen_scores)):
        raise ValueError("ADR 0087 decision has non-finite ranking scores")

    r1200_labeled = np.flatnonzero(r1200_mask)
    r4800_labeled = np.flatnonzero(r4800_mask)
    if len(r1200_labeled) < min(2, count):
        raise ValueError("ADR 0087 requires at least two R1200-labeled actions")
    if len(r4800_labeled) < 2:
        raise ValueError("ADR 0087 requires at least two R4800-labeled actions")

    r1200_ranking = _stable_subset_ranking(
        r1200_mean,
        action_hashes,
        r1200_labeled,
    )
    r4800_ranking = _stable_subset_ranking(
        r4800_mean,
        action_hashes,
        r4800_labeled,
    )
    r1200_winner = int(r1200_ranking[0])
    r4800_winner = int(r4800_ranking[0])
    r4800_runner_up = int(r4800_ranking[1])
    if r4800_winner != selected_index:
        raise ValueError("ADR 0087 selected action is not the stable R4800 argmax")

    r1200_confidence = _confidence_set(
        r1200_winner,
        r1200_labeled,
        r1200_mean,
        r1200_stddev,
        r1200_samples,
        count,
    )
    r4800_confidence = _confidence_set(
        r4800_winner,
        r4800_labeled,
        r4800_mean,
        r4800_stddev,
        r4800_samples,
        count,
    )
    r4800_standard_error = r4800_stddev / np.sqrt(np.maximum(r4800_samples, 1.0))
    winner_margin = float(r4800_mean[r4800_winner] - r4800_mean[r4800_runner_up])
    winner_combined_se = float(
        np.hypot(
            r4800_standard_error[r4800_winner],
            r4800_standard_error[r4800_runner_up],
        )
    )
    distinguishable = winner_margin > NORMAL_95 * winner_combined_se

    all_indices = np.arange(count, dtype=np.int32)
    unlabeled = all_indices[~r1200_mask]
    cohort_oracle = np.concatenate(
        [
            r1200_ranking,
            _stable_subset_ranking(screen_scores, action_hashes, unlabeled),
        ]
    )
    rankings = {}
    ranked_indices = {
        "model": _stable_subset_ranking(
            model_scores,
            action_hashes,
            all_indices,
        ),
        "r1200_cohort_oracle": cohort_oracle,
        "screen": _stable_subset_ranking(
            screen_scores,
            action_hashes,
            all_indices,
        ),
    }
    for name, ranking in ranked_indices.items():
        widths = {}
        for width in RANK_WIDTHS:
            retained = ranking[: min(width, count)]
            retained_labeled = retained[r4800_mask[retained]]
            regret = (
                float(r4800_mean[r4800_winner] - np.max(r4800_mean[retained_labeled]))
                if len(retained_labeled)
                else float(np.max(r4800_mean[r4800_labeled]) - np.min(r4800_mean[r4800_labeled]))
            )
            widths[width] = RankingObservation(
                exact_winner_recalled=bool(np.any(retained == r4800_winner)),
                confidence_set_covered=bool(np.any(r4800_confidence[retained])),
                retained_regret=regret,
            )
        rankings[name] = widths

    model_top64 = ranked_indices["model"][: min(64, count)]
    scored_mask = r600_mask | r1200_mask | r4800_mask
    return DecisionObservation(
        candidates=count,
        r1200_candidates=len(r1200_labeled),
        r4800_candidates=len(r4800_labeled),
        r1200_confidence_set_size_95=int(np.sum(r1200_confidence)),
        r4800_confidence_set_size_95=int(np.sum(r4800_confidence)),
        confidence_sets_intersect=bool(np.any(r1200_confidence & r4800_confidence)),
        r1200_winner_in_r4800_confidence_set_95=bool(r4800_confidence[r1200_winner]),
        r4800_winner_distinguishable_95=distinguishable,
        cohort_has_top64_capacity=len(r1200_labeled) >= min(64, count),
        phase=PHASE_NAMES[phase],
        nature_token_available=nature_token_available,
        independent_draft_winner=independent_draft_winner,
        raw_seed=raw_seed,
        rankings=rankings,
        model_top64_label_counts={
            "r600": int(np.sum(r600_mask[model_top64])),
            "r1200": int(np.sum(r1200_mask[model_top64])),
            "r4800": int(np.sum(r4800_mask[model_top64])),
            "screen_only": int(np.sum(~scored_mask[model_top64])),
        },
    )


def interpretation_gates(
    overall: dict[str, Any],
    phases: dict[str, dict[str, Any]],
    integrity: dict[str, Any],
) -> dict[str, bool]:
    """Apply ADR 0087's frozen interpretation gates."""
    oracle = overall["ranking"]["r1200_cohort_oracle"]["top64"]
    integrity_passed = (
        all(
            bool(value)
            for key, value in integrity.items()
            if key
            not in {
                "groups_seen",
                "expected_groups",
                "candidates_seen",
                "expected_candidates",
                "nonfinite_model_scores",
                "nonfinite_teacher_values",
                "test_split_opened",
            }
        )
        and not integrity["test_split_opened"]
    )
    distinguishable_recall = oracle["distinguishable_winner_recall"]
    gates = {
        "oracle_top64_confidence_set_coverage_at_least_0_99": (
            oracle["confidence_set_coverage_95"] >= 0.99
        ),
        "oracle_top64_distinguishable_winner_recall_at_least_0_98": (
            distinguishable_recall is not None and distinguishable_recall >= 0.98
        ),
        "oracle_top64_exact_winner_recall_at_least_0_95": (oracle["exact_winner_recall"] >= 0.95),
        "oracle_top64_retained_regret_below_0_03": (oracle["mean_retained_r4800_regret"] < 0.03),
        "every_phase_oracle_top64_confidence_coverage_at_least_0_98": all(
            values["ranking"]["r1200_cohort_oracle"]["top64"]["confidence_set_coverage_95"] >= 0.98
            for values in phases.values()
        ),
        "confidence_set_intersection_fraction_at_least_0_95": (
            overall["confidence_set_intersection_fraction"] >= 0.95
        ),
        "every_group_has_top64_cohort_capacity": (overall["cohort_top64_capacity_fraction"] == 1.0),
        "integrity_passed": integrity_passed,
    }
    gates["target_sufficient_for_set_valued_proposer"] = all(gates.values())
    return gates


def classify_interpretation(gates: dict[str, bool]) -> str:
    """Return the single frozen ADR 0087 interpretation."""
    if gates["target_sufficient_for_set_valued_proposer"]:
        return "target_sufficient_for_set_valued_proposer"
    return "target_insufficient_for_set_valued_proposer"


def aggregate_reports(
    train_report_path: str | Path,
    validation_report_paths: Sequence[str | Path],
) -> dict[str, Any]:
    """Require one train audit and identical validation replay on all Macs."""
    train_path = Path(train_report_path)
    train = json.loads(train_path.read_text())
    if train["scientific"]["dataset"]["split"] != "train":
        raise ValueError("ADR 0087 aggregate requires one train report")
    validation_reports = [json.loads(Path(path).read_text()) for path in validation_report_paths]
    hosts = {str(report["host"]) for report in validation_reports}
    if hosts != {"john1", "john2", "john3"}:
        raise ValueError("ADR 0087 aggregate requires validation replays from all three Macs")
    digests = {report["scientific_blake3"] for report in validation_reports}
    if len(digests) != 1:
        raise ValueError("ADR 0087 validation scientific metrics differ across hosts")
    validation = validation_reports[0]["scientific"]
    if validation["dataset"]["split"] != "validation":
        raise ValueError("ADR 0087 aggregate received a non-validation replay")
    if validation["integrity"]["test_split_opened"]:
        raise ValueError("ADR 0087 aggregate detected sealed-test access")
    result = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "complete",
        "train": train["scientific"],
        "validation": validation,
        "cross_host_validation": {
            "hosts": sorted(hosts),
            "scientific_blake3": next(iter(digests)),
            "scientific_metrics_identical": True,
            "execution": {
                str(report["host"]): report["execution"] for report in validation_reports
            },
        },
        "sealed_test": {
            "opened": False,
            "authorized": False,
        },
        "classification": validation["classification"],
        "next_action": (
            "Train one fixed-architecture, confidence-set-aware K1024-to-K64 "
            "proposer under a new preregistration."
            if validation["classification"] == "target_sufficient_for_set_valued_proposer"
            else "Do not train the proposed cohort ranker; revise teacher "
            "allocation or observable representation."
        ),
        "input_sha256": {
            str(train_path): _sha256(train_path),
            **{str(Path(path)): _sha256(Path(path)) for path in validation_report_paths},
        },
    }
    result["passed"] = (
        result["cross_host_validation"]["scientific_metrics_identical"]
        and not result["sealed_test"]["opened"]
        and validation["integrity"]["all_groups_seen_once"]
        and validation["integrity"]["all_candidates_seen_once"]
    )
    return result


def render_markdown(report: dict[str, Any]) -> str:
    """Render the aggregate from the same machine-readable result."""
    train = report["train"]["overall"]
    validation = report["validation"]["overall"]
    oracle = validation["ranking"]["r1200_cohort_oracle"]["top64"]
    model = validation["ranking"]["model"]["top64"]
    composition = validation["model_top64_label_composition"]
    phase_rows = []
    for name, values in report["validation"]["phases"].items():
        phase_oracle = values["ranking"]["r1200_cohort_oracle"]["top64"]
        phase_rows.append(
            "| {} | {:.2%} | {:.2%} | {:.6f} |".format(
                name.title(),
                phase_oracle["exact_winner_recall"],
                phase_oracle["confidence_set_coverage_95"],
                phase_oracle["mean_retained_r4800_regret"],
            )
        )
    return (
        "\n".join(
            [
                "# Complete-Action R1200 Target Sufficiency V1",
                "",
                f"Status: **{report['classification'].replace('_', ' ')}**",
                "",
                "## Result",
                "",
                "| Metric | Train | Validation |",
                "|---|---:|---:|",
                "| R1200/R4800 95% set intersection | {:.2%} | {:.2%} |".format(
                    train["confidence_set_intersection_fraction"],
                    validation["confidence_set_intersection_fraction"],
                ),
                "| R1200 winner inside R4800 95% set | {:.2%} | {:.2%} |".format(
                    train["r1200_winner_in_r4800_confidence_set_95_fraction"],
                    validation["r1200_winner_in_r4800_confidence_set_95_fraction"],
                ),
                "| Mean R1200 cohort size | {:.2f} | {:.2f} |".format(
                    train["mean_r1200_candidates"],
                    validation["mean_r1200_candidates"],
                ),
                "",
                "## Validation Top 64",
                "",
                "| Ranker | Exact winner | 95% set coverage | Regret |",
                "|---|---:|---:|---:|",
                "| Selected MLX model | {:.2%} | {:.2%} | {:.6f} |".format(
                    model["exact_winner_recall"],
                    model["confidence_set_coverage_95"],
                    model["mean_retained_r4800_regret"],
                ),
                "| R1200 cohort oracle | {:.2%} | {:.2%} | {:.6f} |".format(
                    oracle["exact_winner_recall"],
                    oracle["confidence_set_coverage_95"],
                    oracle["mean_retained_r4800_regret"],
                ),
                "",
                "## Phase",
                "",
                "| Phase | Exact winner | 95% set coverage | Regret |",
                "|---|---:|---:|---:|",
                *phase_rows,
                "",
                "## Current Model Top-64 Composition",
                "",
                "- R1200-labeled: {:.2%}.".format(composition["r1200"]["fraction"]),
                "- R4800-labeled: {:.2%}.".format(composition["r4800"]["fraction"]),
                "- Screen-only: {:.2%}.".format(composition["screen_only"]["fraction"]),
                "",
                "## Interpretation",
                "",
                report["next_action"],
                "",
                "The train and validation audit opened no sealed-test group. Validation",
                "scientific metrics and complete rankings were identical on john1, john2,",
                "and john3.",
            ]
        )
        + "\n"
    )


def _load_frozen_model(
    run_dir: Path,
    dataset: GradedOracleDataset,
    dataset_manifest: str,
) -> tuple[GradedOracleRanker, Path, dict[str, Any]]:
    run = json.loads((run_dir / "run.json").read_text())
    best = json.loads((run_dir / "best.json").read_text())
    training = run["training"]
    if best["checkpoint"] != EXPECTED_CHECKPOINT:
        raise ValueError("ADR 0087 selected checkpoint identity drifted")
    if run["source"]["v2_source_blake3"] != EXPECTED_SOURCE_V2_BLAKE3:
        raise ValueError("ADR 0087 source identity drifted")
    expected_dataset = (
        run["datasets"]["train_manifest_blake3"]
        if dataset.split == "train"
        else run["datasets"]["validation_manifest_blake3"]
    )
    if expected_dataset != dataset_manifest:
        raise ValueError("ADR 0087 run and dataset manifests disagree")
    model, _optimizer, _state, checkpoint = load_checkpoint_pointer_with_factory(
        run_dir,
        pointer="best",
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        model_factory=lambda values: GradedOracleRanker(GradedOracleModelConfig.from_dict(values)),
    )
    if (
        checkpoint.name != EXPECTED_CHECKPOINT
        or _checksum(checkpoint / "model.safetensors") != EXPECTED_MODEL_BLAKE3
    ):
        raise ValueError("ADR 0087 model tensor identity drifted")
    model.eval()
    return model, checkpoint, run


def _confidence_set(
    winner: int,
    labeled: np.ndarray,
    means: np.ndarray,
    stddev: np.ndarray,
    samples: np.ndarray,
    count: int,
) -> np.ndarray:
    standard_error = stddev / np.sqrt(np.maximum(samples, 1.0))
    differences = means[winner] - means[labeled]
    pairwise_se = np.hypot(standard_error[winner], standard_error[labeled])
    confidence = np.zeros(count, dtype=bool)
    confidence[labeled] = differences <= NORMAL_95 * pairwise_se
    return confidence


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


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    quantiles = np.quantile(
        array,
        [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0],
    )
    return {
        "mean": float(np.mean(array)),
        "p10": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p90": float(quantiles[4]),
        "p95": float(quantiles[5]),
        "max": float(quantiles[6]),
    }


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return blake3.blake3(payload).hexdigest()


def _checksum(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--run-dir", type=Path, required=True)
    audit.add_argument("--dataset", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--train-report", type=Path, required=True)
    aggregate.add_argument(
        "--validation-report",
        type=Path,
        action="append",
        required=True,
    )
    aggregate.add_argument("--output-json", type=Path, required=True)
    aggregate.add_argument("--output-markdown", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "audit":
        report = audit_dataset(args.run_dir, args.dataset)
        _write_json_atomic(args.output, report)
    else:
        report = aggregate_reports(args.train_report, args.validation_report)
        _write_json_atomic(args.output_json, report)
        _write_text_atomic(args.output_markdown, render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
