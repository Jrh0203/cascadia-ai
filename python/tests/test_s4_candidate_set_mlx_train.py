from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import pytest
from cascadia_mlx.checkpoint import TrainerState, save_checkpoint
from cascadia_mlx.r3_action_edit_mlx_cache import ARMS as R3_ARMS
from cascadia_mlx.r3_action_edit_mlx_model import (
    R3ActionEditModelConfig,
    R3ActionEditRanker,
)
from cascadia_mlx.s4_candidate_set_mlx_model import S4_ARMS
from cascadia_mlx.s4_candidate_set_mlx_train import (
    S4CandidateSetTrainingConfig,
    _warm_start_identity,
    cross_arm_initialization,
)


def _checkpoint(tmp_path: Path) -> Path:
    mx.random.seed(53)
    model = R3ActionEditRanker(
        R3ActionEditModelConfig(arm=R3_ARMS[3])
    )
    optimizer = optim.AdamW(learning_rate=1e-4, weight_decay=1e-4)
    state = TrainerState(global_step=10, batch_in_epoch=10)
    return save_checkpoint(tmp_path, model, optimizer, state)


def _config(
    tmp_path: Path,
    *,
    arm: str = S4_ARMS[0],
    smoke_steps: int | None = 1,
) -> S4CandidateSetTrainingConfig:
    checkpoint = _checkpoint(tmp_path / "warm")
    return S4CandidateSetTrainingConfig(
        train_dataset=tmp_path / "train",
        validation_dataset=tmp_path / "validation",
        cache=tmp_path / "r3",
        s1_cache=tmp_path / "s1",
        context_cache=tmp_path / "context",
        warm_start_checkpoint=checkpoint,
        run_dir=tmp_path / "run",
        output=tmp_path / "report.json",
        arm=arm,
        smoke_steps=smoke_steps,
    )


def test_warm_start_identity_and_cross_arm_initialization_match(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path)
    identity = _warm_start_identity(
        checkpoint,
        require_production=False,
    )
    initialization = cross_arm_initialization(checkpoint)

    assert identity["global_step"] == 10
    assert identity["model_config"]["arm"] == R3_ARMS[3]
    assert len(
        set(initialization["cross_arm_parameter_counts"].values())
    ) == 1
    assert len(
        set(
            initialization[
                "cross_arm_initial_parameter_tensor_blake3"
            ].values()
        )
    ) == 1


def test_production_warm_start_requires_completed_r3_checkpoint(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="warm-start checkpoint"):
        _warm_start_identity(
            _checkpoint(tmp_path),
            require_production=True,
        )


def test_training_config_fails_closed() -> None:
    placeholder = Path("/tmp/s4")
    with pytest.raises(ValueError, match="unknown"):
        S4CandidateSetTrainingConfig(
            train_dataset=placeholder,
            validation_dataset=placeholder,
            cache=placeholder,
            s1_cache=placeholder,
            context_cache=placeholder,
            warm_start_checkpoint=placeholder,
            run_dir=placeholder,
            output=placeholder,
            arm="unknown",
            smoke_steps=1,
        ).validate()
    with pytest.raises(ValueError, match="at most 10"):
        S4CandidateSetTrainingConfig(
            train_dataset=placeholder,
            validation_dataset=placeholder,
            cache=placeholder,
            s1_cache=placeholder,
            context_cache=placeholder,
            warm_start_checkpoint=placeholder,
            run_dir=placeholder,
            output=placeholder,
            arm=S4_ARMS[0],
            smoke_steps=11,
        ).validate()
    with pytest.raises(ValueError, match="authorization"):
        S4CandidateSetTrainingConfig(
            train_dataset=placeholder,
            validation_dataset=placeholder,
            cache=placeholder,
            s1_cache=placeholder,
            context_cache=placeholder,
            warm_start_checkpoint=placeholder,
            run_dir=placeholder,
            output=placeholder,
            arm=S4_ARMS[0],
        ).validate()
