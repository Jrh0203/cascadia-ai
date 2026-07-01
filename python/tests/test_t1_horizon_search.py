from __future__ import annotations

import numpy as np
from cascadia_mlx.t1_horizon_search import (
    BOOTSTRAP_REPLICATES,
    COHORT_WIDTH,
    HORIZON_TURNS,
    _game_clustered_bootstrap,
    _holm_bonferroni,
    frozen_protocol,
)


def test_frozen_protocol_has_exact_horizon_budgets() -> None:
    protocol = frozen_protocol()
    budget = sum(
        active * samples
        for active, samples in zip(
            [64, 32, 16, 8],
            protocol["stage_additional_samples"],
            strict=True,
        )
    )
    assert protocol["h0_evaluations_per_group"] == COHORT_WIDTH == 64
    assert budget == protocol["trajectories_per_search_group"] == 640
    assert protocol["horizon_opponent_turns"] == HORIZON_TURNS
    assert (
        protocol["prefix_coupling"]
        == "h1-prefix-of-h2-prefix-of-h3-by-shared-opponent-uniforms"
    )


def test_holm_bonferroni_stops_after_first_nonrejection() -> None:
    result = _holm_bonferroni(
        {
            "a": 0.001,
            "b": 0.009,
            "c": 0.03,
            "d": 0.04,
            "e": 0.50,
            "f": 0.80,
        },
        alpha=0.05,
    )
    assert result["a"]["rejected"] is True
    assert result["b"]["rejected"] is True
    assert result["c"]["rejected"] is False
    assert result["d"]["rejected"] is False
    assert result["e"]["rejected"] is False
    assert result["f"]["rejected"] is False
    assert result["a"]["adjusted_p"] <= result["b"]["adjusted_p"]


def test_game_clustered_bootstrap_is_paired_and_deterministic() -> None:
    games = np.repeat(np.arange(7), 4)
    reference = np.linspace(0.0, 2.7, len(games))
    treatment = reference - 0.25
    first = _game_clustered_bootstrap(treatment, reference, games)
    second = _game_clustered_bootstrap(treatment, reference, games)
    assert first == second
    assert first["replicates"] == BOOTSTRAP_REPLICATES
    assert first["mean_difference"] == -0.25
    assert first["ci95_upper"] < 0.0
    assert first["one_sided_superiority_p"] < 0.001
