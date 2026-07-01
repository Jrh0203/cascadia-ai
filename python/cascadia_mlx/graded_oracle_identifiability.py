"""Audit whether finite-sample R4800 argmaxes are identifiable learning targets."""

from __future__ import annotations

import argparse
import json
import math
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
from cascadia_mlx.graded_oracle_model import (
    GradedOracleModelConfig,
    GradedOracleRanker,
    predict_graded_oracle_batch,
)

EXPERIMENT_ID = "complete-action-r4800-identifiability-v1"
EXPECTED_CHECKPOINT = "step-000003592-epoch-0008-batch-000000"
EXPECTED_MODEL_BLAKE3 = "6d2a7bb57fd905e50636a20da012f40017cc3a59c1ebde06eff20f8f974940e8"
EXPECTED_SOURCE_V2_BLAKE3 = "4247caaf01dbb60fd158fea1e3f0caa08431ada0683347204aff359639af3bad"
EXPECTED_MANIFEST_BLAKE3 = {
    "train": "7ed12c943d75a786ccd4ccbe11a6b0146aad4fe5ed40f0cbaf1d652f5ac0bb99",
    "validation": "302ceb7a57482b0fb5fb12963521be35aafc121a36f572e6b9f47def1b820a31",
}
NORMAL_95 = 1.959963984540054
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
    exact_winner_recalled: bool
    confidence_set_covered: bool
    retained_regret: float


@dataclass(frozen=True)
class DecisionObservation:
    candidates: int
    r4800_candidates: int
    top_two_margin: float
    combined_standard_error: float
    margin_z_score: float
    winner_standard_error: float
    runner_up_standard_error: float
    distinguishable_95: bool
    separated_intervals_95: bool
    confidence_set_size_68: int
    confidence_set_size_95: int
    r1200_r4800_argmax_agree: bool
    r1200_winner_in_r4800_confidence_set_95: bool
    phase: str
    nature_token_available: bool
    independent_draft_winner: bool
    raw_seed: int
    rankings: dict[str, dict[int, RankingObservation]]


@dataclass
class RankingAccumulator:
    exact_winner_recalled: int = 0
    confidence_set_covered: int = 0
    retained_regret: float = 0.0
    distinguishable_groups: int = 0
    distinguishable_winner_recalled: int = 0
    exact_winner_misses: int = 0
    misses_retaining_confidence_equivalent: int = 0

    def add(self, observation: RankingObservation, distinguishable: bool) -> None:
        self.exact_winner_recalled += int(observation.exact_winner_recalled)
        self.confidence_set_covered += int(observation.confidence_set_covered)
        self.retained_regret += observation.retained_regret
        if distinguishable:
            self.distinguishable_groups += 1
            self.distinguishable_winner_recalled += int(observation.exact_winner_recalled)
        if not observation.exact_winner_recalled:
            self.exact_winner_misses += 1
            self.misses_retaining_confidence_equivalent += int(
                observation.confidence_set_covered
            )

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
            "exact_winner_misses": self.exact_winner_misses,
            "misses_retaining_confidence_equivalent_fraction": (
                self.misses_retaining_confidence_equivalent / self.exact_winner_misses
                if self.exact_winner_misses
                else 1.0
            ),
        }


@dataclass
class AuditAccumulator:
    groups: int = 0
    candidates: int = 0
    r4800_candidates: int = 0
    distinguishable_95: int = 0
    separated_intervals_95: int = 0
    r1200_r4800_argmax_agree: int = 0
    r1200_winner_in_r4800_confidence_set_95: int = 0
    top_two_margin: list[float] = field(default_factory=list)
    combined_standard_error: list[float] = field(default_factory=list)
    margin_z_score: list[float] = field(default_factory=list)
    winner_standard_error: list[float] = field(default_factory=list)
    runner_up_standard_error: list[float] = field(default_factory=list)
    confidence_set_size_68: list[float] = field(default_factory=list)
    confidence_set_size_95: list[float] = field(default_factory=list)
    ranking: dict[str, dict[int, RankingAccumulator]] = field(
        default_factory=lambda: {
            name: {width: RankingAccumulator() for width in RANK_WIDTHS}
            for name in ("model", "screen")
        }
    )

    def add(self, observation: DecisionObservation) -> None:
        self.groups += 1
        self.candidates += observation.candidates
        self.r4800_candidates += observation.r4800_candidates
        self.distinguishable_95 += int(observation.distinguishable_95)
        self.separated_intervals_95 += int(observation.separated_intervals_95)
        self.r1200_r4800_argmax_agree += int(
            observation.r1200_r4800_argmax_agree
        )
        self.r1200_winner_in_r4800_confidence_set_95 += int(
            observation.r1200_winner_in_r4800_confidence_set_95
        )
        self.top_two_margin.append(observation.top_two_margin)
        self.combined_standard_error.append(observation.combined_standard_error)
        self.margin_z_score.append(observation.margin_z_score)
        self.winner_standard_error.append(observation.winner_standard_error)
        self.runner_up_standard_error.append(observation.runner_up_standard_error)
        self.confidence_set_size_68.append(float(observation.confidence_set_size_68))
        self.confidence_set_size_95.append(float(observation.confidence_set_size_95))
        for name, widths in observation.rankings.items():
            for width, ranking_observation in widths.items():
                self.ranking[name][width].add(
                    ranking_observation,
                    observation.distinguishable_95,
                )

    def report(self) -> dict[str, Any]:
        if self.groups == 0:
            raise ValueError("R4800 identifiability audit received no groups")
        return {
            "groups": self.groups,
            "candidates": self.candidates,
            "r4800_candidates": self.r4800_candidates,
            "mean_candidates": self.candidates / self.groups,
            "mean_r4800_candidates": self.r4800_candidates / self.groups,
            "top_two_margin": _distribution(self.top_two_margin),
            "combined_standard_error": _distribution(self.combined_standard_error),
            "margin_z_score": _distribution(self.margin_z_score),
            "winner_standard_error": _distribution(self.winner_standard_error),
            "runner_up_standard_error": _distribution(self.runner_up_standard_error),
            "confidence_set_size_68": _distribution(self.confidence_set_size_68),
            "confidence_set_size_95": _distribution(self.confidence_set_size_95),
            "distinguishable_winner_95_fraction": self.distinguishable_95 / self.groups,
            "separated_confidence_intervals_95_fraction": (
                self.separated_intervals_95 / self.groups
            ),
            "r1200_r4800_argmax_agreement": (
                self.r1200_r4800_argmax_agree / self.groups
            ),
            "r1200_winner_in_r4800_confidence_set_95_fraction": (
                self.r1200_winner_in_r4800_confidence_set_95 / self.groups
            ),
            "ranking": {
                name: {
                    f"top{width}": accumulator.report(self.groups)
                    for width, accumulator in widths.items()
                }
                for name, widths in self.ranking.items()
            },
        }


def audit_dataset(
    run_dir: str | Path,
    dataset_path: str | Path,
) -> dict[str, Any]:
    """Run the frozen ADR 0086 audit on train or validation only."""
    started = time.perf_counter()
    run_dir = Path(run_dir)
    dataset = GradedOracleDataset(dataset_path)
    if dataset.split not in EXPECTED_MANIFEST_BLAKE3:
        raise ValueError("ADR 0086 accepts only the already-open train or validation split")
    dataset_manifest = _checksum(dataset.root / "dataset.json")
    if dataset_manifest != EXPECTED_MANIFEST_BLAKE3[dataset.split]:
        raise ValueError("ADR 0086 dataset manifest identity drifted")

    run = json.loads((run_dir / "run.json").read_text())
    best = json.loads((run_dir / "best.json").read_text())
    training = run["training"]
    if best["checkpoint"] != EXPECTED_CHECKPOINT:
        raise ValueError("ADR 0086 selected checkpoint identity drifted")
    if run["source"]["v2_source_blake3"] != EXPECTED_SOURCE_V2_BLAKE3:
        raise ValueError("ADR 0086 source identity drifted")
    expected_dataset = (
        run["datasets"]["train_manifest_blake3"]
        if dataset.split == "train"
        else run["datasets"]["validation_manifest_blake3"]
    )
    if expected_dataset != dataset_manifest:
        raise ValueError("ADR 0086 run and dataset manifests disagree")

    model, _optimizer, _state, checkpoint = load_checkpoint_pointer_with_factory(
        run_dir,
        pointer="best",
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        model_factory=lambda values: GradedOracleRanker(
            GradedOracleModelConfig.from_dict(values)
        ),
    )
    model_path = checkpoint / "model.safetensors"
    if checkpoint.name != EXPECTED_CHECKPOINT or _checksum(model_path) != EXPECTED_MODEL_BLAKE3:
        raise ValueError("ADR 0086 model tensor identity drifted")
    model.eval()

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
        r1200 = np.asarray(batch.r1200_mean)
        r1200_stddev = np.asarray(batch.r1200_stddev)
        r1200_samples = np.asarray(batch.r1200_samples)
        r1200_mask = np.asarray(batch.r1200_mask)
        r4800 = np.asarray(batch.r4800_mean)
        r4800_stddev = np.asarray(batch.r4800_stddev)
        r4800_samples = np.asarray(batch.r4800_samples)
        r4800_mask = np.asarray(batch.r4800_mask)
        phases_raw = np.asarray(batch.phase)
        tokens = np.asarray(batch.active_nature_tokens)
        draft_kind = np.asarray(batch.draft_kind)
        raw_seed = np.asarray(batch.game_index)

        for group_index, mask in enumerate(masks):
            count = int(np.sum(mask))
            model_scores = scores[group_index, :count]
            nonfinite_model_scores += int(np.sum(~np.isfinite(model_scores)))
            winner = int(selected[group_index])
            observation = analyze_decision(
                model_scores=model_scores,
                screen_scores=screen[group_index, :count],
                action_hashes=hashes[group_index, :count],
                selected_index=winner,
                r1200_mean=r1200[group_index, :count],
                r1200_stddev=r1200_stddev[group_index, :count],
                r1200_samples=r1200_samples[group_index, :count],
                r1200_mask=r1200_mask[group_index, :count],
                r4800_mean=r4800[group_index, :count],
                r4800_stddev=r4800_stddev[group_index, :count],
                r4800_samples=r4800_samples[group_index, :count],
                r4800_mask=r4800_mask[group_index, :count],
                phase=int(phases_raw[group_index]),
                nature_token_available=int(tokens[group_index]) > 0,
                independent_draft_winner=int(draft_kind[group_index, winner]) == 1,
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
        name: accumulator.report()
        for name, accumulator in subsets.items()
        if accumulator.groups
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
        "checkpoint_identity_passed": checkpoint.name == EXPECTED_CHECKPOINT,
        "model_identity_passed": _checksum(model_path) == EXPECTED_MODEL_BLAKE3,
        "dataset_identity_passed": dataset_manifest == EXPECTED_MANIFEST_BLAKE3[dataset.split],
        "source_identity_passed": (
            run["source"]["v2_source_blake3"] == EXPECTED_SOURCE_V2_BLAKE3
        ),
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
            "prior_feature_schema": training["model"]["prior_feature_schema"],
        },
        "method": {
            "implementation_blake3": _checksum(Path(__file__)),
            "normal_95": NORMAL_95,
            "standard_error": "stddev / sqrt(max(samples, 1))",
            "confidence_set": (
                "winner_mean - action_mean <= z * hypot(winner_se, action_se)"
            ),
            "r1200_comparison": "argmax over actions labeled by both R1200 and R4800",
            "ranking_widths": list(RANK_WIDTHS),
            "caveat": (
                "Normal diagnostics use an independence approximation; adaptive "
                "allocation, shared game structure, and common random numbers can "
                "violate it. Confidence sets are not posterior probabilities."
            ),
        },
        "overall": overall_report,
        "phases": phase_reports,
        "subsets": subset_reports,
        "games": {
            str(seed): accumulator.report()
            for seed, accumulator in sorted(games.items())
        },
        "integrity": integrity,
        "interpretation_gates": gates,
        "classification": classify_interpretation(gates),
    }
    host = HOST_ALIASES.get(socket.gethostname().split(".")[0], socket.gethostname().split(".")[0])
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "host": host,
        "scientific": scientific,
        "scientific_blake3": _canonical_digest(scientific),
        "execution": {
            "elapsed_seconds": time.perf_counter() - started,
            "groups_per_second": groups_seen / max(time.perf_counter() - started, 1e-9),
            "test_split_opened": False,
        },
    }


def analyze_decision(
    *,
    model_scores: np.ndarray,
    screen_scores: np.ndarray,
    action_hashes: np.ndarray,
    selected_index: int,
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
    count = len(model_scores)
    arrays = (
        screen_scores,
        action_hashes,
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
        raise ValueError("ADR 0086 decision arrays have inconsistent lengths")
    if phase not in PHASE_NAMES:
        raise ValueError("ADR 0086 decision has invalid phase")
    if not np.all(np.isfinite(model_scores)) or not np.all(np.isfinite(screen_scores)):
        raise ValueError("ADR 0086 decision has non-finite ranking scores")

    labeled = np.flatnonzero(r4800_mask)
    if len(labeled) < 2:
        raise ValueError("ADR 0086 requires at least two R4800-labeled actions")
    r4800_ranking = _stable_subset_ranking(r4800_mean, action_hashes, labeled)
    winner = int(r4800_ranking[0])
    runner_up = int(r4800_ranking[1])
    if winner != selected_index:
        raise ValueError("ADR 0086 selected action is not the stable R4800 argmax")
    standard_error = r4800_stddev / np.sqrt(np.maximum(r4800_samples, 1.0))
    margin = float(r4800_mean[winner] - r4800_mean[runner_up])
    combined = float(np.hypot(standard_error[winner], standard_error[runner_up]))
    z_score = margin / combined if combined > 0 else math.inf
    confidence_68 = np.zeros(count, dtype=bool)
    confidence_95 = np.zeros(count, dtype=bool)
    differences = r4800_mean[winner] - r4800_mean[labeled]
    pairwise_se = np.hypot(standard_error[winner], standard_error[labeled])
    confidence_68[labeled] = differences <= pairwise_se
    confidence_95[labeled] = differences <= NORMAL_95 * pairwise_se
    winner_lower = r4800_mean[winner] - NORMAL_95 * standard_error[winner]
    alternative_upper = np.max(
        r4800_mean[r4800_ranking[1:]]
        + NORMAL_95 * standard_error[r4800_ranking[1:]]
    )

    common = np.flatnonzero(r1200_mask & r4800_mask)
    if len(common) < 2:
        raise ValueError("ADR 0086 requires two common R1200/R4800 actions")
    r1200_winner = int(_stable_subset_ranking(r1200_mean, action_hashes, common)[0])
    rankings = {}
    for name, ranking_scores in (
        ("model", model_scores),
        ("screen", screen_scores),
    ):
        ranking = _stable_subset_ranking(
            ranking_scores,
            action_hashes,
            np.arange(count, dtype=np.int32),
        )
        widths = {}
        for width in RANK_WIDTHS:
            retained = ranking[: min(width, count)]
            retained_labeled = retained[r4800_mask[retained]]
            regret = (
                float(r4800_mean[winner] - np.max(r4800_mean[retained_labeled]))
                if len(retained_labeled)
                else float(np.max(r4800_mean[labeled]) - np.min(r4800_mean[labeled]))
            )
            widths[width] = RankingObservation(
                exact_winner_recalled=bool(np.any(retained == winner)),
                confidence_set_covered=bool(np.any(confidence_95[retained])),
                retained_regret=regret,
            )
        rankings[name] = widths

    return DecisionObservation(
        candidates=count,
        r4800_candidates=len(labeled),
        top_two_margin=margin,
        combined_standard_error=combined,
        margin_z_score=z_score,
        winner_standard_error=float(standard_error[winner]),
        runner_up_standard_error=float(standard_error[runner_up]),
        distinguishable_95=margin > NORMAL_95 * combined,
        separated_intervals_95=winner_lower > alternative_upper,
        confidence_set_size_68=int(np.sum(confidence_68)),
        confidence_set_size_95=int(np.sum(confidence_95)),
        r1200_r4800_argmax_agree=r1200_winner == winner,
        r1200_winner_in_r4800_confidence_set_95=bool(confidence_95[r1200_winner]),
        phase=PHASE_NAMES[phase],
        nature_token_available=nature_token_available,
        independent_draft_winner=independent_draft_winner,
        raw_seed=raw_seed,
        rankings=rankings,
    )


def interpretation_gates(
    overall: dict[str, Any],
    phases: dict[str, dict[str, Any]],
    integrity: dict[str, Any],
) -> dict[str, bool]:
    model_top64 = overall["ranking"]["model"]["top64"]
    integrity_passed = all(
        bool(value)
        for key, value in integrity.items()
        if key
        not in {
            "groups_seen",
            "expected_groups",
            "candidates_seen",
            "expected_candidates",
            "nonfinite_model_scores",
        }
        and key != "test_split_opened"
    ) and not integrity["test_split_opened"]
    gates = {
        "distinguishable_winner_fraction_at_most_0_50": (
            overall["distinguishable_winner_95_fraction"] <= 0.50
        ),
        "mean_confidence_set_size_95_at_least_4": (
            overall["confidence_set_size_95"]["mean"] >= 4.0
        ),
        "model_top64_confidence_set_coverage_at_least_0_98": (
            model_top64["confidence_set_coverage_95"] >= 0.98
        ),
        "model_top64_retained_regret_below_0_15": (
            model_top64["mean_retained_r4800_regret"] < 0.15
        ),
        "model_miss_equivalent_fraction_at_least_0_95": (
            model_top64["misses_retaining_confidence_equivalent_fraction"] >= 0.95
        ),
        "every_phase_model_top64_confidence_coverage_at_least_0_95": all(
            values["ranking"]["model"]["top64"]["confidence_set_coverage_95"] >= 0.95
            for values in phases.values()
        ),
        "integrity_passed": integrity_passed,
    }
    distinguishable_recall = model_top64["distinguishable_winner_recall"]
    gates["model_top64_distinguishable_winner_recall_at_least_0_90"] = (
        distinguishable_recall is not None and distinguishable_recall >= 0.90
    )
    gates["target_ambiguity_dominant"] = all(
        gates[name]
        for name in (
            "distinguishable_winner_fraction_at_most_0_50",
            "mean_confidence_set_size_95_at_least_4",
            "model_top64_confidence_set_coverage_at_least_0_98",
            "model_top64_retained_regret_below_0_15",
            "model_miss_equivalent_fraction_at_least_0_95",
            "every_phase_model_top64_confidence_coverage_at_least_0_95",
            "integrity_passed",
        )
    )
    gates["representation_or_optimization_material"] = (
        not gates["target_ambiguity_dominant"]
        and (
            not gates["model_top64_distinguishable_winner_recall_at_least_0_90"]
            or not gates["model_top64_confidence_set_coverage_at_least_0_98"]
        )
    )
    return gates


def classify_interpretation(gates: dict[str, bool]) -> str:
    if gates["target_ambiguity_dominant"]:
        return "target_ambiguity_dominant"
    if gates["representation_or_optimization_material"]:
        return "representation_or_optimization_material"
    return "inconclusive"


def aggregate_reports(
    train_report_path: str | Path,
    validation_report_paths: Sequence[str | Path],
) -> dict[str, Any]:
    train_path = Path(train_report_path)
    train = json.loads(train_path.read_text())
    if train["scientific"]["dataset"]["split"] != "train":
        raise ValueError("ADR 0086 aggregate requires one train report")
    validation_reports = [json.loads(Path(path).read_text()) for path in validation_report_paths]
    hosts = {str(report["host"]) for report in validation_reports}
    if hosts != {"john1", "john2", "john3"}:
        raise ValueError("ADR 0086 aggregate requires validation replays from all three Macs")
    digests = {report["scientific_blake3"] for report in validation_reports}
    if len(digests) != 1:
        raise ValueError("ADR 0086 validation scientific metrics differ across hosts")
    validation = validation_reports[0]["scientific"]
    if validation["dataset"]["split"] != "validation":
        raise ValueError("ADR 0086 aggregate received a non-validation replay")
    if validation["integrity"]["test_split_opened"]:
        raise ValueError("ADR 0086 aggregate detected sealed-test access")
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
                str(report["host"]): report["execution"]
                for report in validation_reports
            },
        },
        "sealed_test": {
            "opened": False,
            "authorized": False,
        },
        "classification": validation["classification"],
        "next_action": (
            "Revise the public-information continuation/oracle target; exact R4800 "
            "winner imitation is closed as the primary target."
            if validation["classification"] == "target_ambiguity_dominant"
            else "Revise observable representation or optimization before continuation work."
        ),
        "input_sha256": {
            str(train_path): _sha256(train_path),
            **{
                str(Path(path)): _sha256(Path(path))
                for path in validation_report_paths
            },
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
    train = report["train"]["overall"]
    validation = report["validation"]["overall"]
    model = validation["ranking"]["model"]["top64"]
    screen = validation["ranking"]["screen"]["top64"]
    phase_rows = []
    for name, values in report["validation"]["phases"].items():
        phase_rows.append(
            "| {} | {:.2%} | {:.2f} | {:.2%} | {:.4f} |".format(
                name.title(),
                values["distinguishable_winner_95_fraction"],
                values["confidence_set_size_95"]["mean"],
                values["ranking"]["model"]["top64"]["confidence_set_coverage_95"],
                values["ranking"]["model"]["top64"]["mean_retained_r4800_regret"],
            )
        )
    return "\n".join(
        [
            "# Complete-Action R4800 Identifiability V1",
            "",
            f"Status: **{report['classification'].replace('_', ' ')}**",
            "",
            "## Result",
            "",
            "| Metric | Train | Validation |",
            "|---|---:|---:|",
            "| R4800 winner distinguishable at 95% | {:.2%} | {:.2%} |".format(
                train["distinguishable_winner_95_fraction"],
                validation["distinguishable_winner_95_fraction"],
            ),
            "| Mean 95% confidence-set size | {:.2f} | {:.2f} |".format(
                train["confidence_set_size_95"]["mean"],
                validation["confidence_set_size_95"]["mean"],
            ),
            "| R1200/R4800 argmax agreement | {:.2%} | {:.2%} |".format(
                train["r1200_r4800_argmax_agreement"],
                validation["r1200_r4800_argmax_agreement"],
            ),
            "| R1200 winner inside R4800 95% set | {:.2%} | {:.2%} |".format(
                train["r1200_winner_in_r4800_confidence_set_95_fraction"],
                validation["r1200_winner_in_r4800_confidence_set_95_fraction"],
            ),
            "",
            "## Validation Ranking",
            "",
            "| Ranker | Exact winner recall | 95% set coverage | Retained regret |",
            "|---|---:|---:|---:|",
            "| Historical screen top 64 | {:.2%} | {:.2%} | {:.6f} |".format(
                screen["exact_winner_recall"],
                screen["confidence_set_coverage_95"],
                screen["mean_retained_r4800_regret"],
            ),
            "| Selected MLX top 64 | {:.2%} | {:.2%} | {:.6f} |".format(
                model["exact_winner_recall"],
                model["confidence_set_coverage_95"],
                model["mean_retained_r4800_regret"],
            ),
            "",
            "Selected-model exact-winner misses retaining a statistically equivalent",
            "action: {:.2%}.".format(
                model["misses_retaining_confidence_equivalent_fraction"]
            ),
            "",
            "## Phase",
            "",
            "| Phase | Distinguishable winner | Mean 95% set | Model set coverage | Regret |",
            "|---|---:|---:|---:|---:|",
            *phase_rows,
            "",
            "## Interpretation",
            "",
            report["next_action"],
            "",
            "The train and validation audit opened no sealed-test group. Validation",
            "scientific metrics and complete action rankings were identical on john1,",
            "john2, and john3.",
        ]
    ) + "\n"


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
    finite = array[np.isfinite(array)]
    if len(finite) == 0:
        return {
            "mean": math.inf,
            "p10": math.inf,
            "p25": math.inf,
            "p50": math.inf,
            "p75": math.inf,
            "p90": math.inf,
            "p95": math.inf,
            "max": math.inf,
        }
    quantiles = np.quantile(finite, [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0])
    return {
        "mean": float(np.mean(finite)),
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
    import hashlib

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
    aggregate.add_argument("--validation-report", type=Path, action="append", required=True)
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
