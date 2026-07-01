from __future__ import annotations

from cascadia_mlx.conditional_tile_extended_exposure import (
    EPOCHS,
    frozen_config,
)
from cascadia_mlx.conditional_tile_target_only import (
    BATCH_SIZE,
    SEED,
)
from cascadia_mlx.conditional_tile_target_only import (
    frozen_config as source_config,
)


def test_extended_exposure_changes_only_epoch_budget() -> None:
    source = source_config()
    treatment = frozen_config()
    assert EPOCHS == 200
    assert treatment.epochs == 200
    assert source.epochs == 20
    assert treatment.stage == source.stage == "tile"
    assert treatment.seed == source.seed == SEED
    assert treatment.batch_size == source.batch_size == BATCH_SIZE
    assert treatment.learning_rate == source.learning_rate
    assert treatment.weight_decay == source.weight_decay
    assert treatment.hidden_dim == source.hidden_dim
