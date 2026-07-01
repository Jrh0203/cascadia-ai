from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
from cascadia_mlx.dataset import ENTITY_DIM, GLOBAL_DIM
from cascadia_mlx.graded_oracle_dataset import (
    GRADED_ORACLE_ACTION_DIM,
    GRADED_ORACLE_PRIOR_DIM,
    GRADED_ORACLE_PRIOR_SCHEMA,
    GRADED_ORACLE_PUBLIC_SUPPLY_SIZE,
)
from cascadia_mlx.graded_oracle_model import (
    GRADED_ORACLE_ARCHITECTURE,
    GRADED_ORACLE_CANDIDATE_FACTOR_NAMES,
    GRADED_ORACLE_MODEL_SCHEMA_VERSION,
    GradedOracleModelConfig,
    GradedOracleRanker,
    encode_graded_oracle_batch,
    encode_graded_oracle_factor_batch,
    encode_graded_oracle_prepool_batch,
    graded_oracle_loss,
    graded_oracle_loss_components,
    predict_graded_oracle_batch,
)


def _batch() -> SimpleNamespace:
    groups = 2
    candidates = 5
    candidate_mask = mx.array(
        [
            [True, True, True, True, True],
            [True, True, True, False, False],
        ]
    )
    r1200_mask = mx.array(
        [
            [True, True, True, True, False],
            [True, True, True, False, False],
        ]
    )
    r4800_mask = mx.array(
        [
            [True, True, False, False, False],
            [True, True, False, False, False],
        ]
    )
    r600_mask = mx.array(
        [
            [True, True, True, True, False],
            [True, True, True, False, False],
        ]
    )
    return SimpleNamespace(
        board_entities=mx.zeros((groups, 4, 23, ENTITY_DIM)),
        board_mask=mx.ones((groups, 4, 23), dtype=mx.bool_),
        market_entities=mx.zeros((groups, 4, ENTITY_DIM)),
        market_mask=mx.ones((groups, 4), dtype=mx.bool_),
        global_features=mx.zeros((groups, GLOBAL_DIM)),
        public_supply=mx.zeros((groups, GRADED_ORACLE_PUBLIC_SUPPLY_SIZE)),
        action_features=mx.zeros((groups, candidates, GRADED_ORACLE_ACTION_DIM)),
        prior_features=mx.zeros((groups, candidates, GRADED_ORACLE_PRIOR_DIM)),
        staged_market_entities=mx.zeros((groups, candidates, 4, ENTITY_DIM)),
        staged_market_mask=mx.ones((groups, candidates, 4), dtype=mx.bool_),
        staged_public_supply=mx.zeros(
            (groups, candidates, GRADED_ORACLE_PUBLIC_SUPPLY_SIZE)
        ),
        candidate_mask=candidate_mask,
        screen_value=mx.array(
            [
                [92.0, 91.0, 90.0, 89.0, 88.0],
                [93.0, 92.0, 91.0, 0.0, 0.0],
            ]
        ),
        r600_mean=mx.array(
            [
                [94.0, 93.0, 92.0, 91.0, 0.0],
                [95.0, 94.0, 93.0, 0.0, 0.0],
            ]
        ),
        r600_stddev=mx.ones((groups, candidates)),
        r600_samples=mx.where(r600_mask, 600.0, 0.0),
        r600_mask=r600_mask,
        r1200_mean=mx.array(
            [
                [96.0, 95.0, 94.0, 93.0, 0.0],
                [97.0, 96.0, 95.0, 0.0, 0.0],
            ]
        ),
        r1200_stddev=mx.ones((groups, candidates)),
        r1200_samples=mx.where(r1200_mask, 1200.0, 0.0),
        r1200_mask=r1200_mask,
        r4800_mean=mx.array(
            [
                [97.0, 96.0, 0.0, 0.0, 0.0],
                [98.0, 97.0, 0.0, 0.0, 0.0],
            ]
        ),
        r4800_stddev=mx.ones((groups, candidates)),
        r4800_samples=mx.where(r4800_mask, 4800.0, 0.0),
        r4800_mask=r4800_mask,
        selected_index=mx.array([0, 0]),
    )


def test_frozen_graded_oracle_configuration_matches_adr_0081() -> None:
    config = GradedOracleModelConfig()
    assert config.schema_version == GRADED_ORACLE_MODEL_SCHEMA_VERSION == 2
    assert config.architecture == GRADED_ORACLE_ARCHITECTURE
    assert config.prior_feature_schema == GRADED_ORACLE_PRIOR_SCHEMA
    assert config.hidden_dim == 192
    assert config.attention_heads == 6
    assert config.board_blocks == 3
    assert config.market_blocks == 2
    assert config.feed_forward_multiplier == 4


def test_invalid_provenance_leaking_model_schema_is_rejected() -> None:
    with np.testing.assert_raises_regex(
        ValueError,
        "unsupported graded-oracle model configuration",
    ):
        GradedOracleModelConfig.from_dict(
            {
                "schema_version": 1,
                "architecture": GRADED_ORACLE_ARCHITECTURE,
                "hidden_dim": 192,
                "attention_heads": 6,
                "board_blocks": 3,
                "market_blocks": 2,
                "feed_forward_multiplier": 4,
            }
        )


def test_ranker_starts_as_exact_screen_and_has_finite_frozen_loss() -> None:
    batch = _batch()
    model = GradedOracleRanker(
        GradedOracleModelConfig(
            hidden_dim=24,
            attention_heads=4,
            board_blocks=0,
            market_blocks=0,
        )
    )

    prediction = predict_graded_oracle_batch(model, batch)
    components = graded_oracle_loss_components(model, batch)
    loss = graded_oracle_loss(model, batch)
    mx.eval(
        prediction.scores,
        prediction.residuals,
        prediction.standard_errors,
        *components.values(),
        loss,
    )

    np.testing.assert_allclose(
        np.asarray(prediction.scores),
        np.asarray(batch.screen_value),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(np.asarray(prediction.residuals), 0.0)
    assert np.all(np.asarray(prediction.standard_errors)[np.asarray(batch.candidate_mask)] > 0)
    assert set(components) == {
        "r1200_huber",
        "r4800_huber",
        "r1200_listwise",
        "r4800_winner",
        "standard_error_calibration",
        "screen_only_regularization",
    }
    assert np.isfinite(float(loss.item()))
    assert float(loss.item()) > 0.0


def test_exported_candidate_embeddings_exactly_feed_the_residual_head() -> None:
    batch = _batch()
    model = GradedOracleRanker(
        GradedOracleModelConfig(
            hidden_dim=24,
            attention_heads=4,
            board_blocks=0,
            market_blocks=0,
        )
    )
    model.residual_head.weight = mx.ones_like(model.residual_head.weight) * 0.01
    embeddings = encode_graded_oracle_batch(model, batch)
    prediction = predict_graded_oracle_batch(model, batch)
    reconstructed = (
        12.0
        * mx.tanh(model.residual_head(embeddings).reshape(2, 5))
        * batch.candidate_mask
    )
    mx.eval(embeddings, prediction.residuals, reconstructed)
    assert embeddings.shape == (2, 5, 24)
    np.testing.assert_allclose(
        np.asarray(reconstructed),
        np.asarray(prediction.residuals),
        atol=0.0,
        rtol=0.0,
    )


def test_prepool_candidates_reconstruct_exported_embeddings_bit_exactly() -> None:
    batch = _batch()
    model = GradedOracleRanker(
        GradedOracleModelConfig(
            hidden_dim=24,
            attention_heads=4,
            board_blocks=0,
            market_blocks=0,
        )
    )
    prepool = encode_graded_oracle_prepool_batch(model, batch)
    reconstructed = model.encode_output_from_prepool(
        prepool,
        batch.candidate_mask,
    )
    embeddings = encode_graded_oracle_batch(model, batch)
    mx.eval(prepool, reconstructed, embeddings)
    assert prepool.shape == (2, 5, 24)
    np.testing.assert_array_equal(
        np.asarray(reconstructed),
        np.asarray(embeddings),
    )


def test_candidate_factors_reconstruct_prepool_candidates_bit_exactly() -> None:
    batch = _batch()
    model = GradedOracleRanker(
        GradedOracleModelConfig(
            hidden_dim=24,
            attention_heads=4,
            board_blocks=0,
            market_blocks=0,
        )
    )
    factors = encode_graded_oracle_factor_batch(model, batch)
    reconstructed = model.candidate_projection(
        factors.reshape(2, 5, -1)
    ) * batch.candidate_mask[..., None]
    prepool = encode_graded_oracle_prepool_batch(model, batch)
    mx.eval(factors, reconstructed, prepool)
    assert factors.shape == (
        2,
        5,
        len(GRADED_ORACLE_CANDIDATE_FACTOR_NAMES),
        24,
    )
    np.testing.assert_array_equal(
        np.asarray(reconstructed),
        np.asarray(prepool),
    )
