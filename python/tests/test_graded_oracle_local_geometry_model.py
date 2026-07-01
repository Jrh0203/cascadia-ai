from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from cascadia_mlx.dataset import ENTITY_DIM, GLOBAL_DIM
from cascadia_mlx.graded_oracle_dataset import (
    GRADED_ORACLE_ACTION_DIM,
    GRADED_ORACLE_PRIOR_DIM,
    GRADED_ORACLE_PUBLIC_SUPPLY_SIZE,
)
from cascadia_mlx.graded_oracle_local_geometry_model import (
    LOCAL_GEOMETRY_ARCHITECTURE,
    LOCAL_GEOMETRY_CONTEXT_DIM,
    LOCAL_GEOMETRY_MODEL_SCHEMA_VERSION,
    LOCAL_GEOMETRY_RELATION_DIM,
    LocalGeometryModelConfig,
    LocalGeometryRanker,
    candidate_local_geometry,
)
from cascadia_mlx.graded_oracle_model import (
    GradedOracleModelConfig,
    GradedOracleRanker,
    graded_oracle_loss,
    predict_graded_oracle_batch,
)
from cascadia_mlx.hex_symmetry import rotate_axial, rotate_one_hot


def _batch() -> SimpleNamespace:
    groups = 1
    candidates = 3
    candidate_mask = mx.array([[True, True, False]])
    scored_mask = mx.array([[True, True, False]])
    action_features = mx.zeros((groups, candidates, GRADED_ORACLE_ACTION_DIM))
    action_features = mx.concatenate(
        [
            action_features[..., :36],
            mx.array([[[1.0, 0, 0, 0, 0, 0], [1.0, 0, 0, 0, 0, 0], [0.0] * 6]]),
            action_features[..., 42:],
        ],
        axis=-1,
    )
    return SimpleNamespace(
        board_entities=mx.zeros((groups, 4, 23, ENTITY_DIM)),
        board_mask=mx.zeros((groups, 4, 23), dtype=mx.bool_),
        market_entities=mx.zeros((groups, 4, ENTITY_DIM)),
        market_mask=mx.ones((groups, 4), dtype=mx.bool_),
        global_features=mx.zeros((groups, GLOBAL_DIM)),
        public_supply=mx.zeros((groups, GRADED_ORACLE_PUBLIC_SUPPLY_SIZE)),
        action_features=action_features,
        prior_features=mx.zeros((groups, candidates, GRADED_ORACLE_PRIOR_DIM)),
        staged_market_entities=mx.zeros((groups, candidates, 4, ENTITY_DIM)),
        staged_market_mask=mx.ones((groups, candidates, 4), dtype=mx.bool_),
        staged_public_supply=mx.zeros(
            (groups, candidates, GRADED_ORACLE_PUBLIC_SUPPLY_SIZE)
        ),
        candidate_mask=candidate_mask,
        screen_value=mx.array([[92.0, 91.0, 0.0]]),
        r600_mean=mx.array([[94.0, 93.0, 0.0]]),
        r600_stddev=mx.ones((groups, candidates)),
        r600_samples=mx.where(scored_mask, 600.0, 0.0),
        r600_mask=scored_mask,
        r1200_mean=mx.array([[96.0, 95.0, 0.0]]),
        r1200_stddev=mx.ones((groups, candidates)),
        r1200_samples=mx.where(scored_mask, 1200.0, 0.0),
        r1200_mask=scored_mask,
        r4800_mean=mx.array([[97.0, 96.0, 0.0]]),
        r4800_stddev=mx.ones((groups, candidates)),
        r4800_samples=mx.where(scored_mask, 4800.0, 0.0),
        r4800_mask=scored_mask,
        selected_index=mx.array([0]),
    )


def _rotate_inputs(
    board_entities: mx.array,
    board_mask: mx.array,
    action_features: mx.array,
    candidate_mask: mx.array,
    steps: int,
) -> tuple[mx.array, mx.array]:
    group_steps = mx.array([steps])
    board_q, board_r = rotate_axial(
        board_entities[..., 0],
        board_entities[..., 1],
        group_steps,
    )
    board_rotation = rotate_one_hot(
        board_entities[..., 13:19],
        group_steps,
        board_mask,
    )
    rotated_board = mx.concatenate(
        [
            board_q[..., None],
            board_r[..., None],
            board_entities[..., 2:13],
            board_rotation,
            board_entities[..., 19:],
        ],
        axis=-1,
    )
    tile_q, tile_r = rotate_axial(
        action_features[..., 34],
        action_features[..., 35],
        group_steps,
    )
    wildlife_q, wildlife_r = rotate_axial(
        action_features[..., 43],
        action_features[..., 44],
        group_steps,
    )
    action_rotation = rotate_one_hot(
        action_features[..., 36:42],
        group_steps,
        candidate_mask,
    )
    rotated_action = mx.concatenate(
        [
            action_features[..., :34],
            tile_q[..., None],
            tile_r[..., None],
            action_rotation,
            action_features[..., 42:43],
            wildlife_q[..., None],
            wildlife_r[..., None],
            action_features[..., 45:],
        ],
        axis=-1,
    )
    return rotated_board, rotated_action


def _geometry_inputs() -> tuple[mx.array, mx.array, mx.array, mx.array]:
    board = np.zeros((1, 4, 23, ENTITY_DIM), dtype=np.float32)
    mask = np.zeros((1, 4, 23), dtype=np.bool_)
    action = np.zeros((1, 1, GRADED_ORACLE_ACTION_DIM), dtype=np.float32)
    candidate = np.ones((1, 1), dtype=np.bool_)
    action_rotation = 2
    action[0, 0, 36 + action_rotation] = 1.0
    action[0, 0, 42] = 1.0
    action[0, 0, 43] = 4.0 / 24.0

    directions = ((1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1))

    def rotated_direction(index: int) -> tuple[int, int]:
        return directions[(index + action_rotation) % 6]

    rows: list[tuple[int, int, int]] = []
    for local_direction in range(6):
        q, r = rotated_direction(local_direction)
        rows.append((q, r, (action_rotation + local_direction) % 6))
    rows.append((4, 0, action_rotation))
    for local_direction in range(6):
        q, r = rotated_direction(local_direction)
        rows.append((4 + q, r, (action_rotation + local_direction + 1) % 6))
    for index, (q, r, rotation) in enumerate(rows):
        board[0, 0, index, 0] = q / 24.0
        board[0, 0, index, 1] = r / 24.0
        board[0, 0, index, 2 + index % 5] = 1.0
        board[0, 0, index, 13 + rotation] = 1.0
        mask[0, 0, index] = True
    return mx.array(board), mx.array(mask), mx.array(action), mx.array(candidate)


def test_local_geometry_configuration_is_frozen() -> None:
    config = LocalGeometryModelConfig()
    assert config.schema_version == LOCAL_GEOMETRY_MODEL_SCHEMA_VERSION == 1
    assert config.architecture == LOCAL_GEOMETRY_ARCHITECTURE
    assert config.hidden_dim == 192
    assert config.local_hidden_dim == 192


def test_local_geometry_has_exact_relation_occupancy_and_padding() -> None:
    board, mask, action, _candidate = _geometry_inputs()
    padded_action = mx.concatenate([action, mx.zeros_like(action)], axis=1)
    padded_candidate = mx.array([[True, False]])
    context = candidate_local_geometry(board, mask, padded_action, padded_candidate)
    mx.eval(context)
    values = np.asarray(context).reshape(1, 2, 13, LOCAL_GEOMETRY_RELATION_DIM)

    np.testing.assert_array_equal(values[0, 0, :, -1], np.ones(13))
    np.testing.assert_array_equal(values[0, 1], np.zeros_like(values[0, 1]))
    assert context.shape[-1] == LOCAL_GEOMETRY_CONTEXT_DIM


def test_local_geometry_is_exactly_rotation_invariant() -> None:
    board, mask, action, candidate = _geometry_inputs()
    reference = candidate_local_geometry(board, mask, action, candidate)
    mx.eval(reference)
    reference_values = np.asarray(reference)
    for steps in range(1, 6):
        rotated_board, rotated_action = _rotate_inputs(
            board,
            mask,
            action,
            candidate,
            steps,
        )
        rotated = candidate_local_geometry(
            rotated_board,
            mask,
            rotated_action,
            candidate,
        )
        mx.eval(rotated)
        np.testing.assert_array_equal(np.asarray(rotated), reference_values)


def test_zero_initialized_treatment_matches_paired_base_exactly() -> None:
    batch = _batch()
    base_config = GradedOracleModelConfig(
        hidden_dim=24,
        attention_heads=4,
        board_blocks=0,
        market_blocks=0,
    )
    treatment_config = LocalGeometryModelConfig(
        hidden_dim=24,
        attention_heads=4,
        board_blocks=0,
        market_blocks=0,
        local_hidden_dim=24,
    )
    mx.random.seed(2026061601)
    base = GradedOracleRanker(base_config)
    mx.random.seed(2026061601)
    treatment = LocalGeometryRanker(treatment_config)
    base_prediction = predict_graded_oracle_batch(base, batch)
    treatment_prediction = predict_graded_oracle_batch(treatment, batch)
    mx.eval(
        base_prediction.scores,
        base_prediction.residuals,
        base_prediction.standard_errors,
        treatment_prediction.scores,
        treatment_prediction.residuals,
        treatment_prediction.standard_errors,
    )

    np.testing.assert_array_equal(
        np.asarray(treatment_prediction.scores),
        np.asarray(base_prediction.scores),
    )
    np.testing.assert_array_equal(
        np.asarray(treatment_prediction.residuals),
        np.asarray(base_prediction.residuals),
    )
    np.testing.assert_array_equal(
        np.asarray(treatment_prediction.standard_errors),
        np.asarray(base_prediction.standard_errors),
    )


def test_local_geometry_head_receives_finite_gradient() -> None:
    batch = _batch()
    model = LocalGeometryRanker(
        LocalGeometryModelConfig(
            hidden_dim=24,
            attention_heads=4,
            board_blocks=0,
            market_blocks=0,
            local_hidden_dim=24,
        )
    )
    before = np.asarray(model.local_residual_head.weight).copy()
    optimizer = optim.AdamW(learning_rate=1e-4, weight_decay=1e-4)
    loss_and_grad = nn.value_and_grad(model, graded_oracle_loss)
    loss, gradients = loss_and_grad(model, batch)
    optimizer.update(model, gradients)
    mx.eval(model.parameters(), optimizer.state, loss)

    assert np.isfinite(float(loss.item()))
    assert not np.array_equal(
        np.asarray(model.local_residual_head.weight),
        before,
    )
