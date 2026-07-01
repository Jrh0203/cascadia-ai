from __future__ import annotations

import json
import subprocess
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from cascadia_mlx.d6_contract import (
    D6_CONTRACT,
    D6_CONTRACT_ARTIFACT,
    D6_CONTRACT_ID,
    D6_CONTRACT_SCHEMA_VERSION,
    D6_SCIENTIFIC_BLAKE3,
    D6ContractError,
    load_d6_contract,
)
from cascadia_mlx.hex_symmetry import (
    compose_transform_ids,
    d6_transform_ids,
    inverse_transform_id,
    rotate_axial,
    rotate_one_hot,
    rotation_steps,
    transform_axial,
    transform_coord,
    transform_direction,
    transform_direction_indices,
    transform_orientation,
    transform_orientation_one_hot,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPORT_COMMAND = [
    "cargo",
    "run",
    "--quiet",
    "-p",
    "cascadia-game",
    "--bin",
    "d6_contract_metadata",
    "--",
]


def _radius(q: int, r: int) -> int:
    return max(abs(q), abs(r), abs(-q - r))


RADIUS_8_COORDINATES = tuple(
    (q, r) for q in range(-8, 9) for r in range(-8, 9) if _radius(q, r) <= 8
)


def test_fresh_rust_export_matches_bundled_artifact_byte_for_byte() -> None:
    exported = subprocess.run(
        [*EXPORT_COMMAND, "--stdout"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    bundled = D6_CONTRACT_ARTIFACT.read_bytes()
    assert exported == bundled

    subprocess.run(
        [*EXPORT_COMMAND, "--check", str(D6_CONTRACT_ARTIFACT)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )

    raw = json.loads(exported)
    assert tuple(tuple(tuple(row) for row in matrix) for matrix in raw["coordinate_matrices"]) == (
        D6_CONTRACT.coordinate_matrices
    )
    assert tuple(tuple(row) for row in raw["direction_tables"]) == D6_CONTRACT.direction_tables
    assert tuple(tuple(row) for row in raw["dual_tile_rotation_tables"]) == (
        D6_CONTRACT.dual_tile_rotation_tables
    )
    assert tuple(tuple(row) for row in raw["single_tile_rotation_tables"]) == (
        D6_CONTRACT.single_tile_rotation_tables
    )
    assert tuple(raw["inverse_table"]) == D6_CONTRACT.inverse_table
    assert tuple(tuple(row) for row in raw["composition_table"]) == (D6_CONTRACT.composition_table)


def test_metadata_schema_ids_tables_and_scientific_hash_are_frozen() -> None:
    assert D6_CONTRACT.schema_version == D6_CONTRACT_SCHEMA_VERSION == 1
    assert D6_CONTRACT.contract_id == D6_CONTRACT_ID == "cascadia-game-exact-d6-v1"
    assert D6_CONTRACT.scientific_blake3 == D6_SCIENTIFIC_BLAKE3
    assert D6_SCIENTIFIC_BLAKE3 == (
        "db6ac2f9f6ebe2daaa2db603c6c16183512b5d989aed6979e1991e167737633f"
    )
    assert D6_CONTRACT.edge_order == ("E", "NE", "NW", "W", "SW", "SE")
    assert tuple(transform.id for transform in D6_CONTRACT.transforms) == tuple(range(12))
    assert tuple(transform.rotation_steps for transform in D6_CONTRACT.transforms) == (
        0,
        1,
        2,
        3,
        4,
        5,
        0,
        1,
        2,
        3,
        4,
        5,
    )
    assert tuple(transform.reflected for transform in D6_CONTRACT.transforms) == (
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
    )
    assert tuple(transform.name for transform in D6_CONTRACT.transforms) == (
        "R0",
        "R1",
        "R2",
        "R3",
        "R4",
        "R5",
        "R0S",
        "R1S",
        "R2S",
        "R3S",
        "R4S",
        "R5S",
    )


def test_all_radius_8_coordinates_round_trip_and_compose_for_all_transforms() -> None:
    for transform_id in range(12):
        inverse = inverse_transform_id(transform_id)
        assert compose_transform_ids(transform_id, inverse) == 0
        assert compose_transform_ids(inverse, transform_id) == 0
        for q, r in RADIUS_8_COORDINATES:
            transformed = transform_coord(q, r, transform_id)
            assert _radius(*transformed) == _radius(q, r)
            assert transform_coord(*transformed, inverse) == (q, r)

            for right in range(12):
                composed = compose_transform_ids(transform_id, right)
                after_right = transform_coord(q, r, right)
                assert transform_coord(*after_right, transform_id) == transform_coord(
                    q,
                    r,
                    composed,
                )


def test_all_direction_rotation_inverse_and_composition_tables() -> None:
    for transform_id in range(12):
        inverse = inverse_transform_id(transform_id)
        for direction in range(6):
            transformed = transform_direction(direction, transform_id)
            assert transform_direction(transformed, inverse) == direction
            for right in range(12):
                composed = compose_transform_ids(transform_id, right)
                assert transform_direction(
                    transform_direction(direction, right),
                    transform_id,
                ) == transform_direction(direction, composed)


def test_all_tile_orientations_include_reflections_and_single_terrain_canonicalization() -> None:
    for transform_id in range(12):
        inverse = inverse_transform_id(transform_id)
        for rotation in range(6):
            dual = transform_orientation(
                rotation,
                transform_id,
                is_dual_terrain=True,
            )
            assert transform_orientation(dual, inverse, is_dual_terrain=True) == rotation
            assert (
                transform_orientation(
                    rotation,
                    transform_id,
                    is_dual_terrain=False,
                )
                == 0
            )
            for right in range(12):
                composed = compose_transform_ids(transform_id, right)
                after_right = transform_orientation(
                    rotation,
                    right,
                    is_dual_terrain=True,
                )
                assert transform_orientation(
                    after_right,
                    transform_id,
                    is_dual_terrain=True,
                ) == transform_orientation(
                    rotation,
                    composed,
                    is_dual_terrain=True,
                )


def test_mlx_d6_apis_match_every_rust_generated_table_entry() -> None:
    transform_ids = d6_transform_ids(range(12), 12)
    coordinates = np.asarray(RADIUS_8_COORDINATES, dtype=np.int32)
    q = mx.array(np.broadcast_to(coordinates[:, 0], (12, len(coordinates))))
    r = mx.array(np.broadcast_to(coordinates[:, 1], (12, len(coordinates))))
    transformed_q, transformed_r = transform_axial(q, r, transform_ids)
    actual_coordinates = np.stack(
        [np.asarray(transformed_q), np.asarray(transformed_r)],
        axis=-1,
    )
    expected_coordinates = np.asarray(
        [
            [transform_coord(q_value, r_value, transform_id) for q_value, r_value in coordinates]
            for transform_id in range(12)
        ],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(actual_coordinates, expected_coordinates)

    directions = mx.array(np.broadcast_to(np.arange(6, dtype=np.int32), (12, 6)))
    np.testing.assert_array_equal(
        np.asarray(transform_direction_indices(directions, transform_ids)),
        np.asarray(D6_CONTRACT.direction_tables, dtype=np.int32),
    )

    orientations = mx.eye(6)[directions]
    mask = mx.ones((12, 6))
    transformed_orientations = transform_orientation_one_hot(
        orientations,
        transform_ids,
        mask,
    )
    np.testing.assert_array_equal(
        np.asarray(mx.argmax(transformed_orientations, axis=-1)),
        np.asarray(D6_CONTRACT.dual_tile_rotation_tables, dtype=np.int32),
    )

    alternating_semantics = mx.array(np.broadcast_to(np.arange(6) % 2 == 0, (12, 6)))
    semantic_orientations = transform_orientation_one_hot(
        orientations,
        transform_ids,
        mask,
        is_dual_terrain=alternating_semantics,
    )
    expected_semantic = np.where(
        np.asarray(alternating_semantics),
        np.asarray(D6_CONTRACT.dual_tile_rotation_tables),
        np.asarray(D6_CONTRACT.single_tile_rotation_tables),
    )
    np.testing.assert_array_equal(
        np.asarray(mx.argmax(semantic_orientations, axis=-1)),
        expected_semantic,
    )


def test_legacy_c6_public_api_is_unchanged_and_table_driven() -> None:
    steps = rotation_steps(range(6), 6)
    q = mx.array(np.full((6, 1), 2, dtype=np.int32))
    r = mx.array(np.full((6, 1), -1, dtype=np.int32))
    rotated_q, rotated_r = rotate_axial(q, r, steps)
    np.testing.assert_array_equal(
        np.stack([np.asarray(rotated_q)[:, 0], np.asarray(rotated_r)[:, 0]], axis=-1),
        np.asarray(
            [
                [2, -1],
                [1, -2],
                [-1, -1],
                [-2, 1],
                [-1, 2],
                [1, 1],
            ],
            dtype=np.int32,
        ),
    )

    source_rotations = mx.eye(6)[mx.array([[0], [1], [2], [3], [4], [5]])]
    mask = mx.array([[1], [1], [0], [1], [1], [1]])
    rotated = rotate_one_hot(source_rotations, steps, mask)
    np.testing.assert_array_equal(
        np.asarray(mx.argmax(rotated, axis=-1))[:, 0],
        np.asarray([0, 2, 0, 0, 2, 4]),
    )
    np.testing.assert_array_equal(np.asarray(rotated)[2], np.zeros((1, 6)))

    np.testing.assert_array_equal(np.asarray(rotation_steps(3, 4)), np.full(4, 3))
    with pytest.raises(ValueError, match=r"\[0, 5\]"):
        rotation_steps([0, 6], 2)
    with pytest.raises(ValueError, match="one value"):
        rotation_steps([0], 2)


def test_invalid_transform_inputs_and_artifact_drift_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"\[0, 11\]"):
        d6_transform_ids([0, 12], 2)
    with pytest.raises(ValueError, match="transform_id"):
        transform_coord(0, 0, 12)
    with pytest.raises(ValueError, match="direction"):
        transform_direction(6, 0)
    with pytest.raises(ValueError, match="rotation"):
        transform_orientation(6, 0, is_dual_terrain=True)

    raw = json.loads(D6_CONTRACT_ARTIFACT.read_text(encoding="utf-8"))
    raw["scientific_blake3"] = "0" * 64
    invalid_hash = tmp_path / "invalid-hash.json"
    invalid_hash.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(D6ContractError, match="scientific hash"):
        load_d6_contract(invalid_hash)

    raw = json.loads(D6_CONTRACT_ARTIFACT.read_text(encoding="utf-8"))
    raw["composition_table"][1][1] = 0
    invalid_composition = tmp_path / "invalid-composition.json"
    invalid_composition.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(D6ContractError, match="composition_table"):
        load_d6_contract(invalid_composition)

    drifted = tmp_path / "drifted.json"
    drifted.write_bytes(D6_CONTRACT_ARTIFACT.read_bytes() + b"\n")
    checked = subprocess.run(
        [*EXPORT_COMMAND, "--check", str(drifted)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert checked.returncode != 0
    assert "D6 metadata drift" in checked.stderr
