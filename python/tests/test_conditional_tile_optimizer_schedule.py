from __future__ import annotations

from itertools import pairwise

from cascadia_mlx.conditional_tile_extended_exposure import (
    frozen_config as exposure_config,
)
from cascadia_mlx.conditional_tile_optimizer_schedule import (
    FINAL_LEARNING_RATE,
    HOLD_EPOCHS,
    frozen_config,
    late_cosine_learning_rate,
)
from cascadia_mlx.full_legal_hierarchical_factor_retrieval import (
    LEARNING_RATE,
)


def test_schedule_changes_only_late_learning_rate() -> None:
    config = exposure_config()
    assert late_cosine_learning_rate(1, config.epochs) == LEARNING_RATE
    assert late_cosine_learning_rate(HOLD_EPOCHS, config.epochs) == LEARNING_RATE
    assert (
        late_cosine_learning_rate(config.epochs, config.epochs)
        == FINAL_LEARNING_RATE
    )
    rates = [
        late_cosine_learning_rate(epoch, config.epochs)
        for epoch in range(1, config.epochs + 1)
    ]
    assert all(left >= right for left, right in pairwise(rates))
    assert all(rate > 0 for rate in rates)


def test_schedule_contract_matches_extended_exposure() -> None:
    source = exposure_config()
    treatment = frozen_config()
    assert treatment == source
