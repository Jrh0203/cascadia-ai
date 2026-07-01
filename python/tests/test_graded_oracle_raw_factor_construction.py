from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from cascadia_mlx.dataset import ENTITY_DIM, GLOBAL_DIM
from cascadia_mlx.graded_oracle_dataset import (
    GRADED_ORACLE_ACTION_DIM,
    GRADED_ORACLE_PRIOR_DIM,
    GRADED_ORACLE_PUBLIC_SUPPLY_SIZE,
)
from cascadia_mlx.graded_oracle_raw_factor_construction import (
    CANDIDATE_RAW_DIM,
    COMPLETE_RAW_DIM,
    COMPLETE_RAW_FLAT,
    CONSTRUCTION_DIM,
    EXACT_LOCAL_RELATION,
    EXPLICIT_MARKET_TRANSITION,
    FRESH_ENTITY_CROSS,
    MARKET_TRANSITION_DIM,
    PARENT_RAW_DIM,
    PROBE_KINDS,
    PROBE_SEEDS,
    RawFactorProbeConfig,
    balanced_score_binary_loss,
    batch_counts,
    build_raw_factor_probe,
    parameter_count,
    raw_candidate_features,
    raw_factor_construction_classification,
    raw_parent_features,
    score_raw_factor_batch,
)
from mlx.utils import tree_flatten


def _batch(
    *,
    groups: int = 1,
    candidates: int = 4,
    valid: int | None = None,
    seed: int = 7,
) -> SimpleNamespace:
    rng = np.random.default_rng(seed)
    valid = candidates if valid is None else valid
    board_entities = rng.normal(size=(groups, 4, 23, ENTITY_DIM)).astype(np.float32)
    board_mask = np.zeros((groups, 4, 23), dtype=np.bool_)
    board_mask[..., :3] = True
    board_entities *= board_mask[..., None]
    market_entities = rng.normal(size=(groups, 4, ENTITY_DIM)).astype(np.float32)
    market_mask = np.ones((groups, 4), dtype=np.bool_)
    global_features = rng.normal(size=(groups, GLOBAL_DIM)).astype(np.float32)
    public_supply = rng.uniform(size=(groups, GRADED_ORACLE_PUBLIC_SUPPLY_SIZE)).astype(np.float32)
    action_features = rng.normal(size=(groups, candidates, GRADED_ORACLE_ACTION_DIM)).astype(
        np.float32
    )
    action_features[..., 34:36] = 0.0
    action_features[..., 36:42] = 0.0
    action_features[..., 36] = 1.0
    action_features[..., 42] = 1.0
    action_features[..., 43:45] = 0.0
    prior_features = rng.normal(size=(groups, candidates, GRADED_ORACLE_PRIOR_DIM)).astype(
        np.float32
    )
    staged_market_entities = rng.normal(size=(groups, candidates, 4, ENTITY_DIM)).astype(np.float32)
    staged_market_mask = np.ones(
        (groups, candidates, 4),
        dtype=np.bool_,
    )
    staged_public_supply = rng.uniform(
        size=(
            groups,
            candidates,
            GRADED_ORACLE_PUBLIC_SUPPLY_SIZE,
        )
    ).astype(np.float32)
    candidate_mask = np.zeros((groups, candidates), dtype=np.bool_)
    candidate_mask[:, :valid] = True
    action_features *= candidate_mask[..., None]
    prior_features *= candidate_mask[..., None]
    staged_market_entities *= candidate_mask[..., None, None]
    staged_market_mask &= candidate_mask[..., None]
    staged_public_supply *= candidate_mask[..., None]
    screen_rank = np.broadcast_to(
        np.arange(1, candidates + 1, dtype=np.float32),
        (groups, candidates),
    ).copy()
    action_hash = np.zeros((groups, candidates, 32), dtype=np.uint8)
    for index in range(candidates):
        action_hash[:, index, 0] = index + 1
    return SimpleNamespace(
        board_entities=mx.array(board_entities),
        board_mask=mx.array(board_mask),
        market_entities=mx.array(market_entities),
        market_mask=mx.array(market_mask),
        global_features=mx.array(global_features),
        public_supply=mx.array(public_supply),
        action_features=mx.array(action_features),
        prior_features=mx.array(prior_features),
        staged_market_entities=mx.array(staged_market_entities),
        staged_market_mask=mx.array(staged_market_mask),
        staged_public_supply=mx.array(staged_public_supply),
        candidate_mask=mx.array(candidate_mask),
        screen_rank=mx.array(screen_rank),
        action_hash=action_hash,
    )


def _permute(batch: SimpleNamespace, permutation: np.ndarray) -> SimpleNamespace:
    indices = mx.array(permutation)
    return SimpleNamespace(
        **{
            **batch.__dict__,
            "action_features": batch.action_features[:, indices],
            "prior_features": batch.prior_features[:, indices],
            "staged_market_entities": (batch.staged_market_entities[:, indices]),
            "staged_market_mask": batch.staged_market_mask[:, indices],
            "staged_public_supply": (batch.staged_public_supply[:, indices]),
            "candidate_mask": batch.candidate_mask[:, indices],
            "screen_rank": batch.screen_rank[:, indices],
            "action_hash": batch.action_hash[:, permutation],
        }
    )


def test_raw_dimensions_and_probe_shapes() -> None:
    batch = _batch()
    parent = raw_parent_features(
        batch.board_entities,
        batch.board_mask,
        batch.market_entities,
        batch.market_mask,
        batch.global_features,
        batch.public_supply,
    )
    candidate = raw_candidate_features(
        batch.action_features,
        batch.prior_features,
        batch.staged_market_entities,
        batch.staged_market_mask,
        batch.staged_public_supply,
    )
    assert parent.shape == (1, PARENT_RAW_DIM)
    assert candidate.shape == (1, 4, CANDIDATE_RAW_DIM)
    assert COMPLETE_RAW_DIM == PARENT_RAW_DIM + CANDIDATE_RAW_DIM
    assert MARKET_TRANSITION_DIM == 772
    for kind in PROBE_KINDS:
        model = build_raw_factor_probe(kind)
        scores = score_raw_factor_batch(model, batch)
        encoded = model.encode_candidates(
            batch.board_entities,
            batch.board_mask,
            batch.market_entities,
            batch.market_mask,
            batch.global_features,
            batch.public_supply,
            batch.action_features,
            batch.prior_features,
            batch.staged_market_entities,
            batch.staged_market_mask,
            batch.staged_public_supply,
            batch.candidate_mask,
        )
        mx.eval(scores, encoded)
        assert scores.shape == (1, 4)
        assert encoded.shape == (1, 4, CONSTRUCTION_DIM)
        assert parameter_count(model) > 6_000_000
        assert np.all(np.isfinite(np.asarray(scores)))


def test_all_probes_have_finite_gradients() -> None:
    batch = _batch()
    target = mx.array([[True, False, True, False]])
    eligible = mx.ones((1, 4), dtype=mx.bool_)
    counts = batch_counts(batch)
    for kind in PROBE_KINDS:
        model = build_raw_factor_probe(kind)

        def loss_fn(candidate_model: nn.Module) -> mx.array:
            scores = score_raw_factor_batch(candidate_model, batch, counts)
            return balanced_score_binary_loss(
                scores,
                target,
                eligible,
                counts,
            )

        loss, gradients = nn.value_and_grad(model, loss_fn)(model)
        mx.eval(loss, gradients)
        assert np.isfinite(float(loss.item()))
        flattened = tree_flatten(gradients)
        assert flattened
        assert all(np.all(np.isfinite(np.asarray(value))) for _, value in flattened)


def test_probes_are_candidate_permutation_equivariant() -> None:
    batch = _batch()
    permutation = np.array([2, 0, 3, 1])
    permuted = _permute(batch, permutation)
    for kind in PROBE_KINDS:
        model = build_raw_factor_probe(kind)
        original = score_raw_factor_batch(model, batch)
        reordered = score_raw_factor_batch(model, permuted)
        mx.eval(original, reordered)
        np.testing.assert_allclose(
            np.asarray(reordered)[0],
            np.asarray(original)[0, permutation],
            atol=1e-5,
            rtol=1e-5,
        )


def test_padding_cannot_change_valid_scores() -> None:
    compact = _batch(candidates=3, valid=3, seed=11)
    padded = _batch(candidates=5, valid=3, seed=11)
    padded.board_entities = compact.board_entities
    padded.board_mask = compact.board_mask
    padded.market_entities = compact.market_entities
    padded.market_mask = compact.market_mask
    padded.global_features = compact.global_features
    padded.public_supply = compact.public_supply
    padded.action_features = mx.concatenate(
        [
            compact.action_features,
            mx.ones((1, 2, GRADED_ORACLE_ACTION_DIM)) * 999.0,
        ],
        axis=1,
    )
    padded.prior_features = mx.concatenate(
        [
            compact.prior_features,
            mx.ones((1, 2, GRADED_ORACLE_PRIOR_DIM)) * 999.0,
        ],
        axis=1,
    )
    padded.staged_market_entities = mx.concatenate(
        [
            compact.staged_market_entities,
            mx.ones((1, 2, 4, ENTITY_DIM)) * 999.0,
        ],
        axis=1,
    )
    padded.staged_market_mask = mx.concatenate(
        [
            compact.staged_market_mask,
            mx.ones((1, 2, 4), dtype=mx.bool_),
        ],
        axis=1,
    )
    padded.staged_public_supply = mx.concatenate(
        [
            compact.staged_public_supply,
            mx.ones((1, 2, GRADED_ORACLE_PUBLIC_SUPPLY_SIZE)) * 999.0,
        ],
        axis=1,
    )
    padded.candidate_mask = mx.array([[True, True, True, False, False]])
    padded.screen_rank = mx.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    padded.action_hash[:, :3] = compact.action_hash

    for kind in PROBE_KINDS:
        model = build_raw_factor_probe(kind)
        compact_scores = score_raw_factor_batch(model, compact)
        padded_scores = score_raw_factor_batch(model, padded)
        mx.eval(compact_scores, padded_scores)
        np.testing.assert_allclose(
            np.asarray(padded_scores)[0, :3],
            np.asarray(compact_scores)[0],
            atol=1e-5,
            rtol=1e-5,
        )


def test_classification_selects_best_passing_construction() -> None:
    failed = {
        "train": {
            "target_positive_recall": 0.3,
            "target_set_exact_fraction": 0.0,
        },
        "validation": {
            "target_positive_recall": 0.3,
            "target_set_exact_fraction": 0.0,
        },
        "execution": {
            "mlx_memory_before_clear": {
                "peak_active_memory_bytes": 100,
            }
        },
    }
    passed = {
        "train": {
            "target_positive_recall": 0.9,
            "target_set_exact_fraction": 0.3,
        },
        "validation": {
            "target_positive_recall": 0.6,
            "target_set_exact_fraction": 0.02,
        },
        "execution": {
            "mlx_memory_before_clear": {
                "peak_active_memory_bytes": 100,
            }
        },
    }
    reports = {kind: failed for kind in PROBE_KINDS}
    reports[EXACT_LOCAL_RELATION] = passed
    reports[EXPLICIT_MARKET_TRANSITION] = passed
    result = raw_factor_construction_classification(reports)
    assert result["classification"] == "raw_factor_construction_sufficient"
    assert result["selected_kind"] == EXPLICIT_MARKET_TRANSITION

    reports = {kind: failed for kind in PROBE_KINDS}
    assert (
        raw_factor_construction_classification(reports)["classification"]
        == "raw_factor_construction_insufficient"
    )


def test_configuration_and_save_load_are_frozen(tmp_path) -> None:
    for kind in PROBE_KINDS:
        RawFactorProbeConfig(kind=kind, seed=PROBE_SEEDS[kind]).validate()
    with np.testing.assert_raises_regex(ValueError, "configuration drifted"):
        RawFactorProbeConfig(
            kind=COMPLETE_RAW_FLAT,
            seed=PROBE_SEEDS[COMPLETE_RAW_FLAT],
            epochs=21,
        ).validate()

    batch = _batch()
    model = build_raw_factor_probe(COMPLETE_RAW_FLAT)
    before = score_raw_factor_batch(model, batch)
    path = tmp_path / "weights.safetensors"
    mx.save_safetensors(str(path), dict(tree_flatten(model.parameters())))
    reloaded = build_raw_factor_probe(COMPLETE_RAW_FLAT)
    reloaded.load_weights(str(path))
    after = score_raw_factor_batch(reloaded, batch)
    mx.eval(before, after)
    assert np.array_equal(np.asarray(before), np.asarray(after))


def test_arm_constants_remain_distinct() -> None:
    assert set(PROBE_KINDS) == {
        COMPLETE_RAW_FLAT,
        EXACT_LOCAL_RELATION,
        EXPLICIT_MARKET_TRANSITION,
        FRESH_ENTITY_CROSS,
    }
    assert len(set(PROBE_SEEDS.values())) == 4
