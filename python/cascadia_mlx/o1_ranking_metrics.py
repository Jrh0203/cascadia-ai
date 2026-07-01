"""Frozen validation metrics and paired inference for ADR 0188."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import blake3
import mlx.core as mx
import numpy as np

from cascadia_mlx.o1_ranking_cohort import COHORT_WIDTH
from cascadia_mlx.o1_ranking_dataset import O1RankingDataset
from cascadia_mlx.o1_ranking_model import O1IntentConditionedRanker
from cascadia_mlx.r3_action_edit_mlx_cache import LOW_SUPPLY_MAX_UNSEEN

BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 2026061720
EVALUATION_GROUP_BATCH = 8


@dataclass(frozen=True)
class O1RankingEvaluation:
    """JSON metrics plus deterministic raw prediction tensors."""

    metrics: dict[str, Any]
    scores: np.ndarray
    standard_errors: np.ndarray


@dataclass
class _Slice:
    groups: int = 0
    r4800_scorable_groups: int = 0
    regret: float = 0.0
    recall: int = 0
    pairwise_correct: float = 0.0
    pairwise_total: int = 0

    def add(
        self,
        *,
        regret: float | None,
        recall: bool | None,
        pairwise_correct: float,
        pairwise_total: int,
    ) -> None:
        self.groups += 1
        if regret is not None and recall is not None:
            self.r4800_scorable_groups += 1
            self.regret += regret
            self.recall += int(recall)
        self.pairwise_correct += pairwise_correct
        self.pairwise_total += pairwise_total

    def report(self) -> dict[str, float | int]:
        return {
            "groups": self.groups,
            "r4800_scorable_groups": self.r4800_scorable_groups,
            "mean_top1_retained_r4800_regret": (
                self.regret / self.r4800_scorable_groups
                if self.r4800_scorable_groups
                else 0.0
            ),
            "top1_retained_r4800_winner_recall": (
                self.recall / self.r4800_scorable_groups
                if self.r4800_scorable_groups
                else 0.0
            ),
            "r1200_pairwise_ordering_accuracy": (
                self.pairwise_correct / self.pairwise_total
                if self.pairwise_total
                else 0.0
            ),
            "r1200_pairwise_comparisons": self.pairwise_total,
        }


def evaluate_o1_ranking(
    model: O1IntentConditionedRanker,
    dataset: O1RankingDataset,
    *,
    group_batch: int = EVALUATION_GROUP_BATCH,
) -> O1RankingEvaluation:
    """Score every fixed-cohort action exactly once and retain paired evidence."""
    if dataset.split != "validation":
        raise ValueError("O1 primary evaluation requires the open validation split")
    if group_batch <= 0:
        raise ValueError("O1 evaluation group batch must be positive")
    model.eval()
    mx.clear_cache()
    mx.reset_peak_memory()
    scores = np.empty(
        (dataset.group_count, COHORT_WIDTH),
        dtype="<f4",
    )
    standard_errors = np.empty_like(scores)
    group_records: list[dict[str, Any]] = []
    slices = {
        "opening": _Slice(),
        "early_middle": _Slice(),
        "late_middle": _Slice(),
        "endgame": _Slice(),
        "low_supply": _Slice(),
        "nature_token_available": _Slice(),
        "independent_draft_winner": _Slice(),
    }
    all_slice = _Slice()
    candidate_count = 0
    started = time.perf_counter()
    batch_latencies: list[float] = []

    for start in range(0, dataset.group_count, group_batch):
        end = min(start + group_batch, dataset.group_count)
        rows = np.arange(start, end, dtype=np.int64)
        materialized_started = time.perf_counter()
        batch = dataset.batch(rows)
        prediction = model(batch)
        mx.eval(prediction.scores, prediction.standard_errors)
        batch_latencies.append(time.perf_counter() - materialized_started)
        batch_scores = np.asarray(prediction.scores, dtype=np.float32)
        batch_errors = np.asarray(prediction.standard_errors, dtype=np.float32)
        if (
            batch_scores.shape != (len(rows), COHORT_WIDTH)
            or batch_errors.shape != batch_scores.shape
            or not np.isfinite(batch_scores).all()
            or not np.isfinite(batch_errors).all()
            or np.any(batch_errors <= 0)
        ):
            raise ValueError("O1 validation emitted invalid prediction tensors")
        scores[start:end] = batch_scores
        standard_errors[start:end] = batch_errors
        hashes = np.asarray(batch.base.action_hash, dtype=np.uint8)
        r4800 = np.asarray(batch.base.r4800_mean, dtype=np.float32)
        r4800_mask = np.asarray(batch.base.r4800_mask, dtype=np.bool_)
        r1200 = np.asarray(batch.base.r1200_mean, dtype=np.float32)
        r1200_mask = np.asarray(batch.base.r1200_mask, dtype=np.bool_)
        turns = np.asarray(batch.base.turn, dtype=np.int64)
        game_indices = np.asarray(batch.base.game_index, dtype=np.uint64)
        nature_tokens = np.asarray(
            batch.base.active_nature_tokens,
            dtype=np.int64,
        )
        draft_kind = np.asarray(batch.base.draft_kind, dtype=np.int64)
        group_ids = np.asarray(
            dataset.source.tensors["group_ids"][rows],
            dtype=np.uint64,
        )
        for local, row in enumerate(rows):
            group = o1_group_metrics(
                scores=batch_scores[local],
                action_hashes=hashes[local],
                r4800_mean=r4800[local],
                r4800_mask=r4800_mask[local],
                r1200_mean=r1200[local],
                r1200_mask=r1200_mask[local],
            )
            turn = int(turns[local])
            phase = (
                "opening"
                if turn < 20
                else "early_middle"
                if turn < 40
                else "late_middle"
                if turn < 60
                else "endgame"
            )
            memberships = {
                phase,
                *(
                    ["low_supply"]
                    if 81 - turn <= LOW_SUPPLY_MAX_UNSEEN
                    else []
                ),
                *(
                    ["nature_token_available"]
                    if int(nature_tokens[local]) > 0
                    else []
                ),
                *(
                    ["independent_draft_winner"]
                    if group["r4800_scorable"]
                    and int(
                        draft_kind[
                            local,
                            group["retained_winner_index"],
                        ]
                    )
                    == 1
                    else []
                ),
            }
            all_slice.add(
                regret=group["top1_retained_r4800_regret"],
                recall=group["top1_retained_r4800_winner_recalled"],
                pairwise_correct=group["r1200_pairwise_correct"],
                pairwise_total=group["r1200_pairwise_total"],
            )
            for name in memberships:
                slices[name].add(
                    regret=group["top1_retained_r4800_regret"],
                    recall=group["top1_retained_r4800_winner_recalled"],
                    pairwise_correct=group["r1200_pairwise_correct"],
                    pairwise_total=group["r1200_pairwise_total"],
                )
            group_records.append(
                {
                    "row": int(row),
                    "group_id": int(group_ids[local]),
                    "game_index": int(game_indices[local]),
                    "turn": turn,
                    "active_nature_tokens": int(nature_tokens[local]),
                    **group,
                }
            )
        candidate_count += len(rows) * COHORT_WIDTH

    elapsed = time.perf_counter() - started
    digest = blake3.blake3()
    digest.update(b"cascadia-v2-o1-ranking-validation-predictions-v1")
    digest.update(scores.tobytes(order="C"))
    digest.update(standard_errors.tobytes(order="C"))
    prediction_hash = digest.hexdigest()
    latency = np.asarray(batch_latencies, dtype=np.float64)
    primary = all_slice.report()
    metrics: dict[str, Any] = {
        "groups": dataset.group_count,
        "candidates": candidate_count,
        "expected_groups": dataset.group_count,
        "expected_candidates": dataset.candidate_count,
        "all_groups_scored_once": len(group_records) == dataset.group_count,
        "all_candidates_scored_once": candidate_count == dataset.candidate_count,
        "all_scores_and_uncertainties_finite": True,
        **primary,
        "subsets": {
            name: value.report()
            for name, value in slices.items()
        },
        "group_records": group_records,
        "prediction_tensor": {
            "dtype": "<f4",
            "shape": [dataset.group_count, COHORT_WIDTH],
            "blake3": prediction_hash,
            "panel_rows": min(4, dataset.group_count),
            "score_panel": scores[:4].tolist(),
            "standard_error_panel": standard_errors[:4].tolist(),
        },
        "performance": {
            "wall_seconds": elapsed,
            "candidates_per_second": candidate_count / max(elapsed, 1e-12),
            "group_batch": group_batch,
            "batch_latency_mean_seconds": float(np.mean(latency)),
            "batch_latency_p50_seconds": float(np.quantile(latency, 0.50)),
            "batch_latency_p90_seconds": float(np.quantile(latency, 0.90)),
            "peak_active_memory_bytes": int(mx.get_peak_memory()),
        },
    }
    return O1RankingEvaluation(
        metrics=metrics,
        scores=scores,
        standard_errors=standard_errors,
    )


def o1_group_metrics(
    *,
    scores: np.ndarray,
    action_hashes: np.ndarray,
    r4800_mean: np.ndarray,
    r4800_mask: np.ndarray,
    r1200_mean: np.ndarray,
    r1200_mask: np.ndarray,
) -> dict[str, Any]:
    """Compute one fixed-cohort group's primary and secondary endpoints."""
    predicted = np.asarray(scores, dtype=np.float64)
    hashes = np.asarray(action_hashes, dtype=np.uint8)
    r4800 = np.asarray(r4800_mean, dtype=np.float64)
    mask4800 = np.asarray(r4800_mask, dtype=np.bool_)
    r1200 = np.asarray(r1200_mean, dtype=np.float64)
    mask1200 = np.asarray(r1200_mask, dtype=np.bool_)
    if (
        predicted.shape != (COHORT_WIDTH,)
        or hashes.shape != (COHORT_WIDTH, 32)
        or r4800.shape != predicted.shape
        or mask4800.shape != predicted.shape
        or r1200.shape != predicted.shape
        or mask1200.shape != predicted.shape
        or not np.isfinite(predicted).all()
    ):
        raise ValueError("O1 group metric inputs are malformed")
    ranking = stable_ranking(predicted, hashes)
    top1 = int(ranking[0])
    labeled = np.flatnonzero(mask4800)
    if len(labeled):
        retained_winner = int(
            sorted(
                labeled,
                key=lambda index: (
                    -float(r4800[index]),
                    bytes(hashes[index]),
                ),
            )[0]
        )
        best = float(r4800[retained_winner])
        regret = (
            best - float(r4800[top1])
            if mask4800[top1]
            else best - float(np.min(r4800[labeled]))
        )
        recall: bool | None = top1 == retained_winner
        winner_hash: str | None = bytes(hashes[retained_winner]).hex()
    else:
        retained_winner = -1
        regret = None
        recall = None
        winner_hash = None
    pairwise_correct, pairwise_total = pairwise_ordering_stats(
        predicted,
        r1200,
        mask1200,
    )
    return {
        "top1_index": top1,
        "top1_action_hash": bytes(hashes[top1]).hex(),
        "top1_r4800_labeled": bool(mask4800[top1]),
        "r4800_scorable": bool(len(labeled)),
        "retained_winner_index": retained_winner,
        "retained_winner_action_hash": winner_hash,
        "top1_retained_r4800_regret": regret,
        "top1_retained_r4800_winner_recalled": recall,
        "r1200_pairwise_correct": pairwise_correct,
        "r1200_pairwise_total": pairwise_total,
        "r1200_pairwise_accuracy": (
            pairwise_correct / pairwise_total if pairwise_total else 0.0
        ),
    }


def stable_ranking(scores: np.ndarray, action_hashes: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    hashes = np.asarray(action_hashes, dtype=np.uint8)
    if values.ndim != 1 or hashes.shape != (len(values), 32):
        raise ValueError("stable ranking inputs do not align")
    return np.asarray(
        sorted(
            range(len(values)),
            key=lambda index: (-float(values[index]), bytes(hashes[index])),
        ),
        dtype=np.int64,
    )


def pairwise_ordering_stats(
    scores: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, int]:
    """Return half-credit-for-predicted-ties ordering evidence."""
    predicted = np.asarray(scores, dtype=np.float64)
    teacher = np.asarray(targets, dtype=np.float64)
    valid = np.flatnonzero(np.asarray(mask, dtype=np.bool_))
    correct = 0.0
    total = 0
    for left_position, left in enumerate(valid):
        for right in valid[left_position + 1 :]:
            target_delta = teacher[left] - teacher[right]
            if target_delta == 0:
                continue
            prediction_delta = predicted[left] - predicted[right]
            total += 1
            if prediction_delta == 0:
                correct += 0.5
            elif math.copysign(1.0, prediction_delta) == math.copysign(
                1.0,
                target_delta,
            ):
                correct += 1.0
    return correct, total


def game_clustered_bootstrap(
    treatment_regret: np.ndarray,
    control_regret: np.ndarray,
    game_indices: np.ndarray,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    """Bootstrap treatment-minus-control regret over source-game clusters."""
    treatment = np.asarray(treatment_regret, dtype=np.float64)
    control = np.asarray(control_regret, dtype=np.float64)
    games = np.asarray(game_indices, dtype=np.uint64)
    if (
        treatment.ndim != 1
        or treatment.shape != control.shape
        or treatment.shape != games.shape
        or not len(treatment)
        or replicates <= 0
        or not np.isfinite(treatment).all()
        or not np.isfinite(control).all()
    ):
        raise ValueError("paired game bootstrap inputs are invalid")
    unique_games = np.unique(games)
    game_means = np.asarray(
        [
            np.mean((treatment - control)[games == game])
            for game in unique_games
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    sampled = rng.integers(
        0,
        len(unique_games),
        size=(replicates, len(unique_games)),
    )
    bootstrap = np.mean(game_means[sampled], axis=1)
    return {
        "replicates": replicates,
        "seed": seed,
        "games": len(unique_games),
        "mean_difference": float(np.mean(treatment - control)),
        "ci95_lower": float(np.quantile(bootstrap, 0.025)),
        "ci95_upper": float(np.quantile(bootstrap, 0.975)),
    }
