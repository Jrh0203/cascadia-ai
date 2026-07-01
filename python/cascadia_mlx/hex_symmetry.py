"""Table-driven C6 and D6 helpers bound to the Rust rules contract."""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx
import numpy as np

from cascadia_mlx.d6_contract import D6_CONTRACT

_COORDINATE_MATRICES = mx.array(np.asarray(D6_CONTRACT.coordinate_matrices, dtype=np.int32))
_DIRECTION_TABLES = mx.array(np.asarray(D6_CONTRACT.direction_tables, dtype=np.int32))
_DUAL_ROTATION_TABLES = mx.array(np.asarray(D6_CONTRACT.dual_tile_rotation_tables, dtype=np.int32))
_SINGLE_ROTATION_TABLES = mx.array(
    np.asarray(D6_CONTRACT.single_tile_rotation_tables, dtype=np.int32)
)


def d6_transform_ids(transforms: int | Sequence[int], count: int) -> mx.array:
    """Validate one stable Rust D6 transform ID per example."""
    transform_ids = np.asarray(transforms, dtype=np.int32)
    if transform_ids.ndim == 0:
        transform_ids = np.full(count, int(transform_ids), dtype=np.int32)
    if transform_ids.shape != (count,) or np.any((transform_ids < 0) | (transform_ids >= 12)):
        raise ValueError("transforms must provide one value in [0, 11] per example")
    return mx.array(transform_ids)


def rotation_steps(rotations: int | Sequence[int], count: int) -> mx.array:
    """Validate one 60-degree rotation step per example."""
    steps = np.asarray(rotations, dtype=np.int32)
    if steps.ndim == 0:
        steps = np.full(count, int(steps), dtype=np.int32)
    if steps.shape != (count,) or np.any((steps < 0) | (steps >= 6)):
        raise ValueError("rotations must provide one value in [0, 5] per example")
    return mx.array(steps)


def transform_coord(q: int, r: int, transform_id: int) -> tuple[int, int]:
    """Transform one axial coordinate with a stable Rust D6 transform ID."""
    _validate_index(transform_id, 12, "transform_id")
    matrix = D6_CONTRACT.coordinate_matrices[transform_id]
    return (
        matrix[0][0] * q + matrix[0][1] * r,
        matrix[1][0] * q + matrix[1][1] * r,
    )


def transform_direction(direction: int, transform_id: int) -> int:
    """Transform one canonical direction index."""
    _validate_index(transform_id, 12, "transform_id")
    _validate_index(direction, 6, "direction")
    return D6_CONTRACT.direction_tables[transform_id][direction]


def transform_orientation(
    rotation: int,
    transform_id: int,
    *,
    is_dual_terrain: bool,
) -> int:
    """Transform one tile orientation, canonicalizing single-terrain tiles."""
    _validate_index(transform_id, 12, "transform_id")
    _validate_index(rotation, 6, "rotation")
    tables = (
        D6_CONTRACT.dual_tile_rotation_tables
        if is_dual_terrain
        else D6_CONTRACT.single_tile_rotation_tables
    )
    return tables[transform_id][rotation]


def inverse_transform_id(transform_id: int) -> int:
    """Return the stable ID of a transform's inverse."""
    _validate_index(transform_id, 12, "transform_id")
    return D6_CONTRACT.inverse_table[transform_id]


def compose_transform_ids(left: int, right: int) -> int:
    """Return the stable ID for applying ``right`` and then ``left``."""
    _validate_index(left, 12, "left")
    _validate_index(right, 12, "right")
    return D6_CONTRACT.composition_table[left][right]


def transform_axial(
    q: mx.array,
    r: mx.array,
    transforms: mx.array,
) -> tuple[mx.array, mx.array]:
    """Apply exact D6 coordinate matrices to batched axial coordinates."""
    matrices = mx.take(_COORDINATE_MATRICES, transforms, axis=0)
    matrix_shape = (transforms.shape[0],) + (1,) * (q.ndim - 1) + (2, 2)
    matrices = matrices.reshape(matrix_shape)
    return (
        matrices[..., 0, 0] * q + matrices[..., 0, 1] * r,
        matrices[..., 1, 0] * q + matrices[..., 1, 1] * r,
    )


def rotate_axial(
    q: mx.array,
    r: mx.array,
    steps: mx.array,
) -> tuple[mx.array, mx.array]:
    """Rotate normalized axial coordinates by exact 60-degree steps."""
    return transform_axial(q, r, steps)


def transform_direction_indices(
    directions: mx.array,
    transforms: mx.array,
) -> mx.array:
    """Apply exact D6 direction permutations to batched direction indices."""
    tables = mx.take(_DIRECTION_TABLES, transforms, axis=0)
    table_shape = (transforms.shape[0],) + (1,) * (directions.ndim - 1) + (6,)
    tables = tables.reshape(table_shape)
    return mx.squeeze(mx.take_along_axis(tables, directions[..., None], axis=-1), axis=-1)


def transform_orientation_one_hot(
    values: mx.array,
    transforms: mx.array,
    mask: mx.array,
    *,
    is_dual_terrain: bool | mx.array | None = None,
) -> mx.array:
    """Transform six-way tile orientations and preserve padding.

    Omitting ``is_dual_terrain`` retains the historical dual-terrain behavior.
    Supplying false, either globally or per tile, canonicalizes orientation to
    the Rust single-terrain representation.
    """
    dual_tables = mx.take(_DUAL_ROTATION_TABLES, transforms, axis=0)
    single_tables = mx.take(_SINGLE_ROTATION_TABLES, transforms, axis=0)
    table_shape = (transforms.shape[0],) + (1,) * (values.ndim - 2) + (6,)
    source = mx.argmax(values, axis=-1)[..., None]
    dual_indices = mx.squeeze(
        mx.take_along_axis(dual_tables.reshape(table_shape), source, axis=-1),
        axis=-1,
    )
    single_indices = mx.squeeze(
        mx.take_along_axis(single_tables.reshape(table_shape), source, axis=-1),
        axis=-1,
    )
    if is_dual_terrain is None or is_dual_terrain is True:
        indices = dual_indices
    elif is_dual_terrain is False:
        indices = single_indices
    else:
        indices = mx.where(is_dual_terrain, dual_indices, single_indices)
    return mx.eye(6)[indices] * mask[..., None]


def rotate_one_hot(
    values: mx.array,
    steps: mx.array,
    mask: mx.array,
) -> mx.array:
    """Rotate six-way orientation one-hots and preserve padding."""
    return transform_orientation_one_hot(values, steps, mask)


def _validate_index(value: int, upper_bound: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < upper_bound:
        raise ValueError(f"{name} must be an integer in [0, {upper_bound - 1}]")
