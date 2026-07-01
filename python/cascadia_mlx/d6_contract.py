"""Validated Python binding for the Rust-owned exact D6 contract."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

D6_CONTRACT_ARTIFACT = Path(__file__).with_name("d6_contract_metadata.v1.json")
D6_CONTRACT_SCHEMA_VERSION = 1
D6_CONTRACT_ID = "cascadia-game-exact-d6-v1"
D6_SCIENTIFIC_BLAKE3 = "db6ac2f9f6ebe2daaa2db603c6c16183512b5d989aed6979e1991e167737633f"

_EXPECTED_KEYS = {
    "schema_version",
    "contract_id",
    "edge_order",
    "coordinate_matrices",
    "direction_tables",
    "dual_tile_rotation_tables",
    "single_tile_rotation_tables",
    "inverse_table",
    "composition_table",
    "transforms",
    "scientific_blake3",
}


@dataclass(frozen=True)
class D6TransformMetadata:
    id: int
    rotation_steps: int
    reflected: bool
    name: str


@dataclass(frozen=True)
class D6Contract:
    schema_version: int
    contract_id: str
    edge_order: tuple[str, ...]
    coordinate_matrices: tuple[tuple[tuple[int, int], tuple[int, int]], ...]
    direction_tables: tuple[tuple[int, ...], ...]
    dual_tile_rotation_tables: tuple[tuple[int, ...], ...]
    single_tile_rotation_tables: tuple[tuple[int, ...], ...]
    inverse_table: tuple[int, ...]
    composition_table: tuple[tuple[int, ...], ...]
    transforms: tuple[D6TransformMetadata, ...]
    scientific_blake3: str


class D6ContractError(ValueError):
    """The bundled Rust metadata does not satisfy the frozen D6 schema."""


def _integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise D6ContractError(f"{path} must be an integer")
    return value


def _integer_row(value: Any, length: int, path: str, upper_bound: int) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise D6ContractError(f"{path} must contain exactly {length} entries")
    row = tuple(_integer(item, f"{path}[{index}]") for index, item in enumerate(value))
    if any(item < 0 or item >= upper_bound for item in row):
        raise D6ContractError(f"{path} entries must be in [0, {upper_bound - 1}]")
    return row


def _integer_table(
    value: Any,
    rows: int,
    columns: int,
    path: str,
    upper_bound: int,
) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, list) or len(value) != rows:
        raise D6ContractError(f"{path} must contain exactly {rows} rows")
    return tuple(
        _integer_row(row, columns, f"{path}[{index}]", upper_bound)
        for index, row in enumerate(value)
    )


def _coordinate_matrices(
    value: Any,
) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    if not isinstance(value, list) or len(value) != 12:
        raise D6ContractError("coordinate_matrices must contain exactly 12 matrices")
    matrices = []
    for transform_id, matrix in enumerate(value):
        if not isinstance(matrix, list) or len(matrix) != 2:
            raise D6ContractError(f"coordinate_matrices[{transform_id}] must be 2x2")
        signed_rows = []
        for row_index, row in enumerate(matrix):
            if not isinstance(row, list) or len(row) != 2:
                raise D6ContractError(
                    f"coordinate_matrices[{transform_id}][{row_index}] "
                    "must contain exactly 2 entries"
                )
            signed_rows.append(
                tuple(
                    _integer(
                        item,
                        f"coordinate_matrices[{transform_id}][{row_index}][{column}]",
                    )
                    for column, item in enumerate(row)
                )
            )
        signed_matrix = (signed_rows[0], signed_rows[1])
        if any(abs(item) > 1 for row in signed_rows for item in row):
            raise D6ContractError(f"coordinate_matrices[{transform_id}] entries must be in [-1, 1]")
        determinant = (
            signed_matrix[0][0] * signed_matrix[1][1] - signed_matrix[0][1] * signed_matrix[1][0]
        )
        if abs(determinant) != 1:
            raise D6ContractError(f"coordinate_matrices[{transform_id}] must be unimodular")
        matrices.append(signed_matrix)
    return tuple(matrices)


def _matrix_product(
    left: tuple[tuple[int, int], tuple[int, int]],
    right: tuple[tuple[int, int], tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]]:
    return (
        (
            left[0][0] * right[0][0] + left[0][1] * right[1][0],
            left[0][0] * right[0][1] + left[0][1] * right[1][1],
        ),
        (
            left[1][0] * right[0][0] + left[1][1] * right[1][0],
            left[1][0] * right[0][1] + left[1][1] * right[1][1],
        ),
    )


def _parse_transforms(value: Any) -> tuple[D6TransformMetadata, ...]:
    if not isinstance(value, list) or len(value) != 12:
        raise D6ContractError("transforms must contain exactly 12 entries")
    transforms = []
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != {
            "id",
            "rotation_steps",
            "reflected",
            "name",
        }:
            raise D6ContractError(f"transforms[{index}] has an invalid schema")
        transform_id = _integer(item["id"], f"transforms[{index}].id")
        rotation = _integer(item["rotation_steps"], f"transforms[{index}].rotation_steps")
        reflected = item["reflected"]
        name = item["name"]
        if not isinstance(reflected, bool):
            raise D6ContractError(f"transforms[{index}].reflected must be boolean")
        if not isinstance(name, str):
            raise D6ContractError(f"transforms[{index}].name must be a string")
        transforms.append(
            D6TransformMetadata(
                id=transform_id,
                rotation_steps=rotation,
                reflected=reflected,
                name=name,
            )
        )
    return tuple(transforms)


def _validate_group_action(contract: D6Contract) -> None:
    identity_matrix = ((1, 0), (0, 1))
    if contract.coordinate_matrices[0] != identity_matrix:
        raise D6ContractError("transform zero must use the identity coordinate matrix")
    if len(set(contract.coordinate_matrices)) != 12:
        raise D6ContractError("coordinate_matrices must identify 12 unique transforms")
    if set(contract.inverse_table) != set(range(12)):
        raise D6ContractError("inverse_table must be a permutation of the transform IDs")
    if contract.inverse_table[0] != 0:
        raise D6ContractError("transform zero must be its own inverse")
    if contract.composition_table[0] != tuple(range(12)):
        raise D6ContractError("composition_table row zero must be the left identity")
    if tuple(row[0] for row in contract.composition_table) != tuple(range(12)):
        raise D6ContractError("composition_table column zero must be the right identity")
    if any(set(row) != set(range(12)) for row in contract.composition_table):
        raise D6ContractError("every composition_table row must be a permutation")

    for transform_id in range(12):
        inverse = contract.inverse_table[transform_id]
        if contract.composition_table[transform_id][inverse] != 0:
            raise D6ContractError(f"inverse_table[{transform_id}] is not a right inverse")
        if contract.composition_table[inverse][transform_id] != 0:
            raise D6ContractError(f"inverse_table[{transform_id}] is not a left inverse")

        for right in range(12):
            composed = contract.composition_table[transform_id][right]
            if (
                _matrix_product(
                    contract.coordinate_matrices[transform_id],
                    contract.coordinate_matrices[right],
                )
                != contract.coordinate_matrices[composed]
            ):
                raise D6ContractError(
                    f"coordinate matrix composition disagrees at ({transform_id}, {right})"
                )
            for table_name, table in (
                ("direction_tables", contract.direction_tables),
                ("dual_tile_rotation_tables", contract.dual_tile_rotation_tables),
                ("single_tile_rotation_tables", contract.single_tile_rotation_tables),
            ):
                for value in range(6):
                    if table[transform_id][table[right][value]] != table[composed][value]:
                        raise D6ContractError(
                            f"{table_name} composition disagrees at "
                            f"({transform_id}, {right}, {value})"
                        )

            for third in range(12):
                left_associated = contract.composition_table[composed][third]
                right_associated = contract.composition_table[transform_id][
                    contract.composition_table[right][third]
                ]
                if left_associated != right_associated:
                    raise D6ContractError(
                        f"composition_table is not associative at "
                        f"({transform_id}, {right}, {third})"
                    )


def _parse_contract(raw: Any) -> D6Contract:
    if not isinstance(raw, dict) or set(raw) != _EXPECTED_KEYS:
        raise D6ContractError("D6 metadata root has an invalid schema")

    schema_version = _integer(raw["schema_version"], "schema_version")
    if schema_version != D6_CONTRACT_SCHEMA_VERSION:
        raise D6ContractError(
            f"unsupported D6 schema {schema_version}; expected {D6_CONTRACT_SCHEMA_VERSION}"
        )
    if raw["contract_id"] != D6_CONTRACT_ID:
        raise D6ContractError(f"unexpected D6 contract id {raw['contract_id']!r}")
    if raw["scientific_blake3"] != D6_SCIENTIFIC_BLAKE3:
        raise D6ContractError(f"unexpected D6 scientific hash {raw['scientific_blake3']!r}")

    edge_order = raw["edge_order"]
    if edge_order != ["E", "NE", "NW", "W", "SW", "SE"]:
        raise D6ContractError("edge_order does not match the frozen Rust contract")

    transforms = _parse_transforms(raw["transforms"])
    expected_transforms = tuple(
        D6TransformMetadata(
            id=transform_id,
            rotation_steps=transform_id % 6,
            reflected=transform_id >= 6,
            name=f"R{transform_id % 6}{'S' if transform_id >= 6 else ''}",
        )
        for transform_id in range(12)
    )
    if transforms != expected_transforms:
        raise D6ContractError("transform IDs or descriptors are not the frozen 0..11 sequence")

    direction_tables = _integer_table(raw["direction_tables"], 12, 6, "direction_tables", 6)
    dual_rotation_tables = _integer_table(
        raw["dual_tile_rotation_tables"],
        12,
        6,
        "dual_tile_rotation_tables",
        6,
    )
    single_rotation_tables = _integer_table(
        raw["single_tile_rotation_tables"],
        12,
        6,
        "single_tile_rotation_tables",
        6,
    )
    if any(set(row) != set(range(6)) for row in direction_tables):
        raise D6ContractError("every direction table row must be a permutation")
    if any(set(row) != set(range(6)) for row in dual_rotation_tables):
        raise D6ContractError("every dual-terrain rotation row must be a permutation")
    if any(any(value != 0 for value in row) for row in single_rotation_tables):
        raise D6ContractError("single-terrain rotation tables must canonicalize to zero")

    contract = D6Contract(
        schema_version=schema_version,
        contract_id=raw["contract_id"],
        edge_order=tuple(edge_order),
        coordinate_matrices=_coordinate_matrices(raw["coordinate_matrices"]),
        direction_tables=direction_tables,
        dual_tile_rotation_tables=dual_rotation_tables,
        single_tile_rotation_tables=single_rotation_tables,
        inverse_table=_integer_row(raw["inverse_table"], 12, "inverse_table", 12),
        composition_table=_integer_table(
            raw["composition_table"],
            12,
            12,
            "composition_table",
            12,
        ),
        transforms=transforms,
        scientific_blake3=raw["scientific_blake3"],
    )
    _validate_group_action(contract)
    return contract


def load_d6_contract(path: Path | str = D6_CONTRACT_ARTIFACT) -> D6Contract:
    """Load and strictly validate one Rust-generated D6 metadata artifact."""
    artifact = Path(path)
    try:
        raw = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise D6ContractError(f"failed to load D6 contract at {artifact}: {error}") from error
    return _parse_contract(raw)


D6_CONTRACT = load_d6_contract()
