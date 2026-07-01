from __future__ import annotations

import numpy as np
from cascadia_mlx.o1_ranking_cohort import COHORT_WIDTH
from cascadia_mlx.o1_ranking_metrics import (
    game_clustered_bootstrap,
    o1_group_metrics,
    pairwise_ordering_stats,
    stable_ranking,
)


def _hashes() -> np.ndarray:
    return np.asarray(
        [list(index.to_bytes(2, "little") * 16) for index in range(COHORT_WIDTH)],
        dtype=np.uint8,
    )


def test_stable_ranking_uses_hash_for_score_ties() -> None:
    scores = np.zeros(COHORT_WIDTH, dtype=np.float32)
    hashes = _hashes()[::-1].copy()

    ranking = stable_ranking(scores, hashes)

    assert bytes(hashes[ranking[0]]) == min(bytes(value) for value in hashes)


def test_group_metrics_penalize_unlabeled_top1_by_labeled_range() -> None:
    scores = np.arange(COHORT_WIDTH, dtype=np.float32)
    r4800 = np.zeros(COHORT_WIDTH, dtype=np.float32)
    r4800[:3] = [10.0, 12.0, 11.0]
    r4800_mask = np.zeros(COHORT_WIDTH, dtype=np.bool_)
    r4800_mask[:3] = True
    r1200 = np.arange(COHORT_WIDTH, dtype=np.float32)

    metrics = o1_group_metrics(
        scores=scores,
        action_hashes=_hashes(),
        r4800_mean=r4800,
        r4800_mask=r4800_mask,
        r1200_mean=r1200,
        r1200_mask=np.ones(COHORT_WIDTH, dtype=np.bool_),
    )

    assert metrics["top1_index"] == COHORT_WIDTH - 1
    assert metrics["retained_winner_index"] == 1
    assert metrics["top1_retained_r4800_regret"] == 2.0
    assert metrics["top1_retained_r4800_winner_recalled"] is False


def test_pairwise_ordering_excludes_teacher_ties_and_half_credits_prediction_ties() -> None:
    correct, total = pairwise_ordering_stats(
        np.asarray([2.0, 2.0, 0.0]),
        np.asarray([3.0, 2.0, 2.0]),
        np.asarray([True, True, True]),
    )

    assert total == 2
    assert correct == 1.5


def test_group_metrics_report_unscorable_when_top64_has_no_r4800_labels() -> None:
    metrics = o1_group_metrics(
        scores=np.arange(COHORT_WIDTH, dtype=np.float32),
        action_hashes=_hashes(),
        r4800_mean=np.zeros(COHORT_WIDTH, dtype=np.float32),
        r4800_mask=np.zeros(COHORT_WIDTH, dtype=np.bool_),
        r1200_mean=np.arange(COHORT_WIDTH, dtype=np.float32),
        r1200_mask=np.ones(COHORT_WIDTH, dtype=np.bool_),
    )

    assert metrics["r4800_scorable"] is False
    assert metrics["retained_winner_index"] == -1
    assert metrics["top1_retained_r4800_regret"] is None
    assert metrics["top1_retained_r4800_winner_recalled"] is None
    assert metrics["r1200_pairwise_total"] > 0


def test_game_clustered_bootstrap_is_deterministic_and_directional() -> None:
    control = np.asarray([1.0, 1.2, 0.8, 1.1, 0.9, 1.3])
    treatment = control - 0.2
    games = np.asarray([10, 10, 11, 11, 12, 12], dtype=np.uint64)

    first = game_clustered_bootstrap(
        treatment,
        control,
        games,
        replicates=1_000,
        seed=71,
    )
    repeated = game_clustered_bootstrap(
        treatment,
        control,
        games,
        replicates=1_000,
        seed=71,
    )

    assert first == repeated
    assert first["mean_difference"] < 0
    assert first["ci95_upper"] < 0
