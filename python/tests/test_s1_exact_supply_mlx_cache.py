from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
from cascadia_mlx.d6_contract import D6_CONTRACT
from cascadia_mlx.dataset import ENTITY_DIM
from cascadia_mlx.graded_oracle_dataset import GRADED_ORACLE_ACTION_DIM
from cascadia_mlx.s1_exact_supply_mlx_cache import (
    ADR_ID,
    ARCHETYPE_COUNT,
    ARMS,
    CACHE_SCHEMA,
    CACHE_SCHEMA_VERSION,
    CATALOG_BLAKE3,
    EXACT_SUPPLY_DIM,
    EXACT_TOKEN_COUNT,
    EXPERIMENT_ID,
    FRONTIER_FEATURE_DIM,
    LEGACY_SUPPLY_DIM,
    NORMALIZATION_CONTRACT,
    PROTOCOL_ID,
    S1ExactSupplyBatch,
    S1ExactSupplyCache,
    _candidate_identity_hash,
    _canonical_blake3,
    _route_arm_inputs,
    _Split,
    _validate_collision_witness,
    collision_witness_arm_inputs,
    transform_s1_exact_supply_batch,
)


def _routed_inputs() -> dict[str, S1ExactSupplyBatch]:
    public_supply = np.linspace(0.0, 1.0, LEGACY_SUPPLY_DIM, dtype=np.float32)[None, :]
    staged_public_supply = np.broadcast_to(public_supply[:, None, :], (1, 2, 30)).copy()
    base = SimpleNamespace(
        public_supply=mx.array(public_supply),
        staged_public_supply=mx.array(staged_public_supply),
        candidate_mask=mx.array([[True, True]]),
    )
    exact_values = np.zeros((1, EXACT_SUPPLY_DIM), dtype=np.uint8)
    exact_values[:, :5] = 10
    exact_values[:, 5:45] = 2
    exact_values[:, 45] = 1
    exact_values[:, 80:] = [81, 79, 2]
    staged_wildlife = np.asarray([[[9, 10, 10, 10, 10], [10, 9, 10, 10, 10]]])
    selected = np.asarray([[3, 17]], dtype=np.int32)
    frontier = np.ones((1, 2, FRONTIER_FEATURE_DIM), dtype=np.float32)
    catalog = np.zeros((ARCHETYPE_COUNT, 23), dtype=np.float32)
    return {
        arm: _route_arm_inputs(
            arm=arm,
            batch=base,
            exact_values=exact_values,
            staged_wildlife=staged_wildlife,
            selected_archetype=selected.copy(),
            frontier_features=frontier.copy(),
            catalog_features=catalog,
        )
        for arm in ARMS
    }


def test_c0_is_a_real_30_marginal_control_and_all_arms_share_shapes() -> None:
    batches = _routed_inputs()
    c0, t1, t2 = (batches[arm] for arm in ARMS)
    for batch in batches.values():
        assert batch.supply_vector.shape == (1, EXACT_SUPPLY_DIM)
        assert batch.supply_tokens.shape == (1, EXACT_TOKEN_COUNT, 32)
        assert batch.supply_mask.shape == (1, EXACT_TOKEN_COUNT)
        assert bool(np.asarray(batch.supply_mask).all())

    c0_vector = np.asarray(c0.supply_vector)
    np.testing.assert_allclose(
        c0_vector[:, :LEGACY_SUPPLY_DIM],
        np.asarray(c0.base.public_supply),
    )
    assert np.count_nonzero(c0_vector[:, LEGACY_SUPPLY_DIM:]) == 0
    assert np.count_nonzero(np.asarray(c0.supply_tokens)[:, LEGACY_SUPPLY_DIM:]) == 0
    assert np.count_nonzero(np.asarray(c0.selected_archetype)) == 0
    assert np.count_nonzero(np.asarray(c0.frontier_features)) == 0

    assert np.count_nonzero(np.asarray(t1.supply_vector)[:, LEGACY_SUPPLY_DIM:]) > 0
    assert np.count_nonzero(np.asarray(t1.selected_archetype)) == 0
    assert np.count_nonzero(np.asarray(t1.frontier_features)) == 0
    np.testing.assert_array_equal(np.asarray(t2.selected_archetype), [[3, 17]])
    assert np.count_nonzero(np.asarray(t2.frontier_features)) > 0
    assert NORMALIZATION_CONTRACT["unused_c0_exact_slots"] == "zero"


def _collision_witness() -> dict[str, object]:
    left = [0] * ARCHETYPE_COUNT
    right = [0] * ARCHETYPE_COUNT
    left[26] = 1
    left[72] = 1
    right[24] = 1
    right[74] = 1
    identity = {
        "schema": "adr-0143-factual-legacy-collision-v1",
        "left_physical_tile_ids": [0, 23],
        "right_physical_tile_ids": [2, 20],
        "left_archetype_ids": [26, 72],
        "right_archetype_ids": [24, 74],
        "legacy_marginals_equal": True,
        "refill_laws_differ": True,
        "legacy_supply_values": [0] * LEGACY_SUPPLY_DIM,
        "left_refill_numerators": left,
        "right_refill_numerators": right,
        "refill_denominator": 2,
        "left_supply_blake3": "1" * 64,
        "right_supply_blake3": "2" * 64,
    }
    return {"witness_id": _canonical_blake3(identity), "identity": identity}


def test_factual_collision_aliases_c0_but_separates_exact_refill_laws() -> None:
    witness = _collision_witness()
    _validate_collision_witness(witness)
    cache = SimpleNamespace(manifest={"collision_witness": witness})
    c0 = collision_witness_arm_inputs(cache, ARMS[0])
    t1 = collision_witness_arm_inputs(cache, ARMS[1])
    np.testing.assert_array_equal(
        c0["left_supply_vector"],
        c0["right_supply_vector"],
    )
    assert not np.array_equal(
        t1["left_supply_vector"],
        t1["right_supply_vector"],
    )
    assert not np.array_equal(
        t1["left_refill_target"],
        t1["right_refill_target"],
    )


@dataclass(frozen=True)
class _GeometryBase:
    board_entities: mx.array
    board_mask: mx.array
    action_features: mx.array
    candidate_mask: mx.array


def test_d6_transforms_geometry_but_not_supply_or_relational_facts() -> None:
    board = np.zeros((1, 4, 23, ENTITY_DIM), dtype=np.float32)
    board_mask = np.zeros((1, 4, 23), dtype=np.bool_)
    board_mask[0, 0, 0] = True
    board[0, 0, 0, :2] = [2, -1]
    board[0, 0, 0, 12] = 1
    board[0, 0, 0, 13] = 1
    action = np.zeros((1, 1, GRADED_ORACLE_ACTION_DIM), dtype=np.float32)
    action[0, 0, 22] = 1
    action[0, 0, 34:36] = [1, -2]
    action[0, 0, 36] = 1
    action[0, 0, 43:45] = [-1, 1]
    base = _GeometryBase(
        board_entities=mx.array(board),
        board_mask=mx.array(board_mask),
        action_features=mx.array(action),
        candidate_mask=mx.array([[True]]),
    )
    supply = mx.array(np.arange(EXACT_SUPPLY_DIM, dtype=np.float32)[None, :])
    batch = S1ExactSupplyBatch(
        base=base,
        supply_vector=supply,
        staged_supply_vector=supply[:, None, :],
        supply_tokens=mx.ones((1, EXACT_TOKEN_COUNT, 32)),
        supply_mask=mx.ones((1, EXACT_TOKEN_COUNT), dtype=mx.bool_),
        refill_target=mx.ones((1, ARCHETYPE_COUNT)) / ARCHETYPE_COUNT,
        selected_archetype=mx.array([[17]]),
        frontier_features=mx.ones((1, 1, FRONTIER_FEATURE_DIM)),
    )
    transformed = transform_s1_exact_supply_batch(batch, 7)
    restored = transform_s1_exact_supply_batch(
        transformed,
        D6_CONTRACT.inverse_table[7],
    )
    np.testing.assert_array_equal(
        np.asarray(transformed.supply_vector),
        np.asarray(batch.supply_vector),
    )
    np.testing.assert_array_equal(
        np.asarray(transformed.frontier_features),
        np.asarray(batch.frontier_features),
    )
    np.testing.assert_allclose(
        np.asarray(restored.board_entities),
        np.asarray(batch.board_entities),
    )
    np.testing.assert_allclose(
        np.asarray(restored.action_features),
        np.asarray(batch.action_features),
    )


def test_materialize_preserves_unsigned_high_bit_group_ids() -> None:
    group_id = 2**63 + 5
    action_hash = np.zeros((1, 1, 32), dtype=np.uint8)
    staged = np.asarray([[9, 10, 10, 10, 10]], dtype=np.uint8)
    archetypes = np.asarray([17], dtype=np.uint8)
    requirements = np.full((1, 6), 5, dtype=np.uint8)
    compatibility = np.zeros((1, 8), dtype=np.uint8)
    candidate_identity = _candidate_identity_hash(
        group_id,
        action_hash[0],
        staged,
        archetypes,
        requirements,
        compatibility,
    )
    exact = np.zeros((1, EXACT_SUPPLY_DIM), dtype=np.uint8)
    exact[:, 5:45] = 2
    exact[:, 45] = 1
    exact[:, 80:] = [81, 79, 2]
    source = _Split(
        groups=1,
        candidates=1,
        tensors={
            "public_state_hashes": np.zeros((1, 32), dtype=np.uint8),
            "exact_supply_values": exact,
            "candidate_offsets": np.asarray([0, 1], dtype=np.uint64),
            "staged_wildlife_counts": staged,
            "selected_archetype_ids": archetypes,
            "frontier_requirements": requirements,
            "selected_compatibility": compatibility,
            "candidate_identity_hashes": np.frombuffer(
                candidate_identity,
                dtype=np.uint8,
            )[None, :],
        },
        group_rows={group_id: 0},
    )
    cache = object.__new__(S1ExactSupplyCache)
    cache.splits = {"train": source}
    cache.catalog_features = np.zeros((ARCHETYPE_COUNT, 23), dtype=np.float32)
    action_features = np.zeros((1, 1, GRADED_ORACLE_ACTION_DIM), dtype=np.float32)
    action_features[0, 0, 36] = 1
    base = SimpleNamespace(
        group_id=mx.array([group_id - 2**64], dtype=mx.int64),
        public_state_hash=np.zeros((1, 32), dtype=np.uint8),
        candidate_mask=mx.array([[True]]),
        action_hash=action_hash,
        action_features=mx.array(action_features),
        public_supply=mx.zeros((1, LEGACY_SUPPLY_DIM)),
        staged_public_supply=mx.zeros((1, 1, LEGACY_SUPPLY_DIM)),
    )
    batch = cache.materialize("train", ARMS[2], base)
    assert batch.supply_vector.shape == (1, EXACT_SUPPLY_DIM)
    assert int(np.asarray(batch.selected_archetype)[0, 0]) == 17


def test_production_cache_cannot_claim_complete_with_incomplete_splits(
    tmp_path,
) -> None:
    identity = {"bounded": True}
    cache_id = _canonical_blake3(identity)
    cache = object.__new__(S1ExactSupplyCache)
    cache.root = tmp_path / cache_id
    cache.manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_schema": CACHE_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "catalog_blake3": CATALOG_BLAKE3,
        "cache_id": cache_id,
        "scientific_identity": identity,
        "complete_open_corpus": True,
        "splits": {
            "train": {
                "complete_open_split": False,
                "groups": 1,
                "candidates": 1,
            },
            "validation": {
                "complete_open_split": False,
                "groups": 1,
                "candidates": 1,
            },
        },
        "catalog": [{} for _ in range(ARCHETYPE_COUNT)],
        "collision_witness": _collision_witness(),
        "hidden_information": {
            "public_position_records_only": True,
            "public_supply_only": True,
            "hidden_stack_order_read": False,
            "hidden_wildlife_order_read": False,
            "excluded_tile_identities_read": False,
            "future_refills_read": False,
            "sealed_test_opened": False,
            "gameplay_opened": False,
        },
        "exporter": {
            "executable_blake3": "a" * 64,
            "source": {"v2_source_blake3": "b" * 64},
        },
    }
    with np.testing.assert_raises_regex(
        ValueError,
        "does not fully cover train",
    ):
        cache._validate_envelope(require_complete=True)
