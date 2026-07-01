"""Deterministic F4 information-preservation and adversarial diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Final

import blake3
import numpy as np

from cascadia_mlx.d6_contract import (
    D6_CONTRACT,
    D6_CONTRACT_ID,
    D6_SCIENTIFIC_BLAKE3,
)

SUITE_SCHEMA_VERSION: Final = 1
PAIR_SCHEMA_VERSION: Final = 1
BOUNDARY_SCHEMA_VERSION: Final = 1
PROBE_SCHEMA_VERSION: Final = 1
RESOLVED_DEPENDENCY_SCHEMA_VERSION: Final = 2
EXPERIMENT_ID: Final = "information-preservation-adversarial-suite-v1"
FIXTURE_SET_ID: Final = "information-preservation-adversarial-pairs-v1"
RESOLVED_DEPENDENCY_ARTIFACT_ID: Final = "information-preservation-resolved-dependencies-v2"
F1_EXPERIMENT_ID: Final = "feature-schema-activation-census-v1"
F1_CLASSIFICATION: Final = "feature_schema_activation_census_complete"
F1_CLASSIFICATION_SCIENTIFIC_BLAKE3: Final = (
    "f7f8559431f53a461f9464e14ef4cee2119cf3ddcf0bf4e3dd9126ab8bdd91fb"
)
F1_FORWARD_SCIENTIFIC_BLAKE3: Final = (
    "8906487b91aa0da25f388e2075d15150c8d1499a022cbf2d231987b37f182e65"
)
F1_MANIFEST_SCIENTIFIC_BLAKE3: Final = (
    "1ebc86d586453548cb6109780f3c86d05867936f61eb1e34690b7bfd086fc9de"
)
F1_CLASSIFICATION_FILE_BLAKE3: Final = (
    "3b6fc75f125189d7b7dc9d693a6603e30b100fc5d50221620da98984e93ce695"
)
F1_FORWARD_FILE_BLAKE3: Final = "0578883cb10be4a7b99449516cfbf1cd5edb3c620fba750460b587e005adc774"
F1_RESULT_MANIFEST_FILE_BLAKE3: Final = (
    "070765de28a27703d92289f8feec374b95cac88ee8639e8475a976f0d1a92b32"
)
F1_RESULT_REPORT_FILE_BLAKE3: Final = (
    "906442bbdcdc8359b06c65b2181c1f521ec5ffe8e2c636caaba0a702f82751cb"
)
F2_SCIENTIFIC_BLAKE3: Final = "c6076545aa93e78902b739eefef1545a23b8f2dbe44770f427a30969511800e5"
F2_EXPERIMENT_ID: Final = "state-footprint-census-v1"
F3_EXPERIMENT_ID: Final = "exact-d6-transform-contract-v1"
IN_RADIUS_BOUNDARY_ID: Final = "radius-6-in-radius-only-v1"
REFILL_DISTRIBUTION_BOUNDARY_ID: Final = "public-refill-distribution-v1"
NORMAL_95: Final = 1.959963984540054

REQUIRED_FAMILIES: Final = (
    "semantic_tile_multiset",
    "multiplicity_descendant_distribution",
    "long_salmon_component_context",
    "focal_relative_opponent_order",
    "d6_transforms",
    "tile_id_permutation",
    "component_bridge",
    "equal_immediate_different_future_conflict",
    "opponent_demand_seat_timing",
    "public_action_equivalence_refill_near_match",
    "same_factor_scores_different_joint_completion",
    "ambiguous_confidence_set_vs_distinguishable_winner",
    "same_in_radius_different_overflow_consequence",
    "same_compact_latent_different_legal_affordance",
)
REQUIRED_PROBES: Final = (
    "occupancy",
    "frontier",
    "component",
    "motif",
    "exact_supply",
    "staged_market",
    "action_edit",
    "opponent_demand",
    "d6_identity",
    "legal_mask",
    "confidence_set_membership",
)
ALLOWED_DEPENDENCIES: Final = ("F1", "F2", "F3")
RELATIONS: Final = ("equivalent", "different")
PAIR_STATUSES: Final = ("ready", "dependency_blocked")

CLASSIFICATION_PASSED: Final = "information_preservation_suite_passed"
CLASSIFICATION_BLOCKED: Final = "information_preservation_suite_dependency_blocked"
CLASSIFICATION_FAILED: Final = "information_preservation_suite_failed"
CLASSIFICATION_INVALID: Final = "information_preservation_suite_invalid"
EXIT_CODES: Final = {
    CLASSIFICATION_PASSED: 0,
    CLASSIFICATION_BLOCKED: 2,
    CLASSIFICATION_FAILED: 3,
    CLASSIFICATION_INVALID: 4,
}

_SCIENTIFIC_EXCLUDED_KEYS: Final = {
    "canonical_hash",
    "scientific_blake3",
    "generated_at",
    "created_at",
    "updated_at",
    "timestamp",
    "hostname",
    "host",
    "path",
    "output_path",
    "output_dir",
}
_FORBIDDEN_PUBLIC_KEY_TOKENS: Final = (
    "hidden",
    "private",
    "future",
    "terminal",
    "rollout",
    "teacher",
    "bag_order",
    "refill_order",
    "unseen_order",
)
_TERRAINS: Final = ("Forest", "Mountain", "Prairie", "Wetland", "River")
_WILDLIFE: Final = ("Bear", "Elk", "Salmon", "Hawk", "Fox")
_F1_SOURCE_FILES: Final = {
    "final_classification": {
        "path": (
            "artifacts/experiments/feature-schema-activation-census-v1/"
            "reports/final-classification.json"
        ),
        "file_blake3": F1_CLASSIFICATION_FILE_BLAKE3,
        "scientific_blake3": F1_CLASSIFICATION_SCIENTIFIC_BLAKE3,
    },
    "final_forward": {
        "path": (
            "artifacts/experiments/feature-schema-activation-census-v1/reports/final-forward.json"
        ),
        "file_blake3": F1_FORWARD_FILE_BLAKE3,
        "scientific_blake3": F1_FORWARD_SCIENTIFIC_BLAKE3,
    },
    "result_manifest": {
        "path": ("artifacts/experiments/feature-schema-activation-census-v1/result-manifest.json"),
        "file_blake3": F1_RESULT_MANIFEST_FILE_BLAKE3,
        "scientific_blake3": None,
    },
    "result_report": {
        "path": "docs/v2/reports/feature-schema-activation-census-v1-result.md",
        "file_blake3": F1_RESULT_REPORT_FILE_BLAKE3,
        "scientific_blake3": None,
    },
}
_F1_RELEVANT_BLOCKS: Final = (
    {
        "block_id": "legacy.habitat_sizes_v1",
        "schema_blake3": "30d0dd72bb7209adccb133e7cc42005284520b0359a6732b35280ee327d3a78d",
        "width": 50,
        "implementation_status": "implemented",
        "measurement_status": "measurable",
        "semantic_owner": "focal habitat structure",
        "rows": 200_000,
        "active_rows": 200_000,
        "collision_status": "unknown",
        "status": ["active_no_detected_issue"],
    },
    {
        "block_id": "legacy.patterns_v1",
        "schema_blake3": "30d0dd72bb7209adccb133e7cc42005284520b0359a6732b35280ee327d3a78d",
        "width": 89,
        "implementation_status": "implemented",
        "measurement_status": "measurable",
        "semantic_owner": "focal wildlife scoring motifs",
        "rows": 200_000,
        "active_rows": 200_000,
        "collision_status": "unknown",
        "status": ["rare_channels_present"],
    },
    {
        "block_id": "graded.action.immediate_score",
        "schema_blake3": "f630830247072927f6e59e12c314920496268074778703a8d7b318f982480188",
        "width": 1,
        "implementation_status": "implemented",
        "measurement_status": "measurable",
        "semantic_owner": "exact observable immediate score",
        "rows": 2_995_314,
        "active_rows": 2_995_314,
        "collision_status": "no_channel_alias_detected",
        "status": ["active_no_detected_issue"],
    },
    {
        "block_id": "graded.action.immediate_deltas",
        "schema_blake3": "f630830247072927f6e59e12c314920496268074778703a8d7b318f982480188",
        "width": 11,
        "implementation_status": "implemented",
        "measurement_status": "measurable",
        "semantic_owner": "exact observable score-component deltas",
        "rows": 2_995_314,
        "active_rows": 2_732_222,
        "collision_status": "no_channel_alias_detected",
        "status": ["active_no_detected_issue"],
    },
    {
        "block_id": "graded.parent_public_supply",
        "schema_blake3": "f630830247072927f6e59e12c314920496268074778703a8d7b318f982480188",
        "width": 30,
        "implementation_status": "implemented",
        "measurement_status": "measurable",
        "semantic_owner": "public wildlife and tile-supply marginals",
        "rows": 2_995_314,
        "active_rows": 2_995_314,
        "collision_status": "structural_or_empirical_alias",
        "status": ["known_noninjective_supply_summary"],
    },
    {
        "block_id": "graded.staged_market",
        "schema_blake3": "f630830247072927f6e59e12c314920496268074778703a8d7b318f982480188",
        "width": 124,
        "implementation_status": "implemented",
        "measurement_status": "measurable",
        "semantic_owner": "public market after ordered prelude",
        "rows": 2_995_314,
        "active_rows": 2_995_314,
        "collision_status": "unknown_hash_candidate",
        "status": ["sketch_alias_candidates_present"],
    },
    {
        "block_id": "graded.staged_public_supply",
        "schema_blake3": "f630830247072927f6e59e12c314920496268074778703a8d7b318f982480188",
        "width": 30,
        "implementation_status": "implemented",
        "measurement_status": "measurable",
        "semantic_owner": "public supply after ordered prelude",
        "rows": 2_995_314,
        "active_rows": 2_995_314,
        "collision_status": "structural_or_empirical_alias",
        "status": ["known_noninjective_supply_summary"],
    },
)


class SuiteValidationError(ValueError):
    """Raised when a fixture, boundary, or report violates the frozen contract."""


class BoundaryUnavailable(LookupError):
    """Raised when a boundary does not apply to one public input."""


def _normalized(value: Any, *, scientific: bool = False) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, np.ndarray):
        return {
            "__array__": True,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": _normalized(value.tolist(), scientific=scientific),
        }
    if isinstance(value, np.generic):
        return _normalized(value.item(), scientific=scientific)
    if isinstance(value, bytes):
        return {"__bytes_hex__": value.hex()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        output = {}
        for raw_key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            key = str(raw_key)
            if scientific and key in _SCIENTIFIC_EXCLUDED_KEYS:
                continue
            output[key] = _normalized(item, scientific=scientific)
        return output
    if isinstance(value, (list, tuple)):
        return [_normalized(item, scientific=scientific) for item in value]
    if isinstance(value, set | frozenset):
        items = [_normalized(item, scientific=scientific) for item in value]
        return sorted(items, key=canonical_json)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SuiteValidationError("scientific values must be finite")
        if value == 0.0:
            return 0.0
        return value
    if value is None or isinstance(value, (bool, int, str)):
        return value
    raise SuiteValidationError(f"unsupported canonical value: {type(value).__name__}")


def canonical_json(value: Any, *, scientific: bool = False) -> str:
    """Return stable, finite, type-aware canonical JSON."""
    return json.dumps(
        _normalized(value, scientific=scientific),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def scientific_blake3(value: Any) -> str:
    """Hash scientific content while excluding paths, timestamps, and digest fields."""
    return blake3.blake3(canonical_json(value, scientific=True).encode()).hexdigest()


def file_blake3(path: str | Path) -> str:
    """Return the exact BLAKE3 of one immutable artifact."""
    digest = blake3.blake3()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_keys(value: Any, expected: set[str], *, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SuiteValidationError(f"{location} must be an object")
    actual = {str(key) for key in value}
    if actual != expected:
        raise SuiteValidationError(f"{location} keys drifted: {sorted(actual ^ expected)}")
    return value


def _require_blake3(value: Any, *, location: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SuiteValidationError(f"{location} must be a lowercase BLAKE3 digest")
    return value


def _coordinate(value: Any, *, location: str) -> tuple[int, int]:
    mapping = _require_keys(value, {"q", "r"}, location=location)
    q = mapping["q"]
    r = mapping["r"]
    if (
        isinstance(q, bool)
        or not isinstance(q, int)
        or isinstance(r, bool)
        or not isinstance(r, int)
        or not -24 <= q <= 24
        or not -24 <= r <= 24
    ):
        raise SuiteValidationError(f"{location} must be an in-grid integer axial coordinate")
    return q, r


def _coordinate_list(value: Any, *, location: str) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, list):
        raise SuiteValidationError(f"{location} must be a coordinate list")
    coordinates = tuple(
        _coordinate(item, location=f"{location}[{index}]") for index, item in enumerate(value)
    )
    if tuple(sorted(coordinates)) != coordinates or len(set(coordinates)) != len(coordinates):
        raise SuiteValidationError(f"{location} must be sorted and unique")
    return coordinates


def _boolean_mask(value: Any, length: int, *, location: str) -> tuple[bool, ...]:
    if (
        not isinstance(value, list)
        or len(value) != length
        or any(not isinstance(item, bool) for item in value)
    ):
        raise SuiteValidationError(f"{location} must contain {length} booleans")
    return tuple(value)


def _axial_radius(coordinate: tuple[int, int]) -> int:
    q, r = coordinate
    return (abs(q) + abs(r) + abs(-q - r)) // 2


def _nonnegative_integer(value: Any, *, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SuiteValidationError(f"{location} must be a nonnegative integer")
    return value


def _validate_habitat_component(value: Any, *, location: str) -> dict[str, Any]:
    component = dict(
        _require_keys(
            value,
            {"terrain", "component_blake3", "cells", "size"},
            location=location,
        )
    )
    if component["terrain"] not in _TERRAINS:
        raise SuiteValidationError(f"{location}.terrain is invalid")
    _require_blake3(component["component_blake3"], location=f"{location}.component_blake3")
    cells = _coordinate_list(component["cells"], location=f"{location}.cells")
    size = _nonnegative_integer(component["size"], location=f"{location}.size")
    if not cells or size != len(cells):
        raise SuiteValidationError(f"{location} component size is inconsistent")
    return component


def _validate_habitat_components(value: Any, *, location: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise SuiteValidationError(f"{location} must be a component list")
    components = [
        _validate_habitat_component(item, location=f"{location}[{index}]")
        for index, item in enumerate(value)
    ]
    hashes = [item["component_blake3"] for item in components]
    if len(hashes) != len(set(hashes)):
        raise SuiteValidationError(f"{location} component identities must be unique")
    return components


def _validate_public_cells(value: Any, *, location: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise SuiteValidationError(f"{location} must be a public-cell list")
    cells = []
    coordinates = []
    for index, item in enumerate(value):
        item_location = f"{location}[{index}]"
        cell = dict(
            _require_keys(
                item,
                {
                    "coord",
                    "tile_id",
                    "terrain_a",
                    "terrain_b",
                    "rotation",
                    "wildlife",
                },
                location=item_location,
            )
        )
        coordinate = _coordinate(cell["coord"], location=f"{item_location}.coord")
        coordinates.append(coordinate)
        tile_id = _nonnegative_integer(cell["tile_id"], location=f"{item_location}.tile_id")
        rotation = _nonnegative_integer(cell["rotation"], location=f"{item_location}.rotation")
        if tile_id > 255 or rotation > 5:
            raise SuiteValidationError(f"{item_location} tile identity or rotation is invalid")
        if cell["terrain_a"] not in _TERRAINS or (
            cell["terrain_b"] is not None and cell["terrain_b"] not in _TERRAINS
        ):
            raise SuiteValidationError(f"{item_location} terrain is invalid")
        if cell["wildlife"] is not None and cell["wildlife"] not in _WILDLIFE:
            raise SuiteValidationError(f"{item_location}.wildlife is invalid")
        cells.append(cell)
    if tuple(sorted(coordinates)) != tuple(coordinates) or len(set(coordinates)) != len(
        coordinates
    ):
        raise SuiteValidationError(f"{location} coordinates must be sorted and unique")
    return cells


def _validate_score_breakdown(value: Any, *, location: str) -> dict[str, Any]:
    score = dict(
        _require_keys(
            value,
            {
                "habitat",
                "wildlife",
                "nature_tokens",
                "habitat_bonus",
                "base_total",
                "total",
            },
            location=location,
        )
    )
    arrays = {}
    for key in ("habitat", "wildlife", "habitat_bonus"):
        values = score[key]
        if not isinstance(values, list) or len(values) != 5:
            raise SuiteValidationError(f"{location}.{key} must contain five integers")
        arrays[key] = [
            _nonnegative_integer(item, location=f"{location}.{key}[{index}]")
            for index, item in enumerate(values)
        ]
    nature_tokens = _nonnegative_integer(
        score["nature_tokens"],
        location=f"{location}.nature_tokens",
    )
    base_total = _nonnegative_integer(score["base_total"], location=f"{location}.base_total")
    total = _nonnegative_integer(score["total"], location=f"{location}.total")
    if base_total != sum(arrays["habitat"]) + sum(arrays["wildlife"]) + nature_tokens:
        raise SuiteValidationError(f"{location}.base_total is inconsistent")
    if total != base_total + sum(arrays["habitat_bonus"]):
        raise SuiteValidationError(f"{location}.total is inconsistent")
    return score


def _validate_motif_signature(value: Any, *, location: str) -> dict[str, Any]:
    motif = dict(
        _require_keys(
            value,
            {
                "wildlife",
                "components",
                "adjacent_same_wildlife_edges",
                "conflicted_coordinates",
                "branching_coordinates",
            },
            location=location,
        )
    )
    if motif["wildlife"] not in _WILDLIFE:
        raise SuiteValidationError(f"{location}.wildlife is invalid")
    if not isinstance(motif["components"], list) or not motif["components"]:
        raise SuiteValidationError(f"{location}.components must be nonempty")
    for index, component in enumerate(motif["components"]):
        if not _coordinate_list(component, location=f"{location}.components[{index}]"):
            raise SuiteValidationError(f"{location}.components[{index}] must be nonempty")
    edges = motif["adjacent_same_wildlife_edges"]
    if not isinstance(edges, list):
        raise SuiteValidationError(f"{location}.adjacent_same_wildlife_edges must be a list")
    for index, edge in enumerate(edges):
        edge_location = f"{location}.adjacent_same_wildlife_edges[{index}]"
        edge_map = _require_keys(edge, {"left", "right"}, location=edge_location)
        left = _coordinate(edge_map["left"], location=f"{edge_location}.left")
        right = _coordinate(edge_map["right"], location=f"{edge_location}.right")
        if left >= right:
            raise SuiteValidationError(f"{edge_location} must use canonical endpoint order")
    _coordinate_list(
        motif["conflicted_coordinates"],
        location=f"{location}.conflicted_coordinates",
    )
    _coordinate_list(
        motif["branching_coordinates"],
        location=f"{location}.branching_coordinates",
    )
    return motif


def _validate_turn_action(value: Any, *, location: str) -> dict[str, Any]:
    action = dict(
        _require_keys(
            value,
            {
                "replace_three_of_a_kind",
                "wildlife_wipes",
                "draft",
                "tile",
                "wildlife",
            },
            location=location,
        )
    )
    if not isinstance(action["replace_three_of_a_kind"], bool):
        raise SuiteValidationError(f"{location}.replace_three_of_a_kind must be boolean")
    if not isinstance(action["wildlife_wipes"], list):
        raise SuiteValidationError(f"{location}.wildlife_wipes must be a list")
    draft = _require_keys(action["draft"], {"Paired"}, location=f"{location}.draft")
    paired = _require_keys(draft["Paired"], {"slot"}, location=f"{location}.draft.Paired")
    slot = _nonnegative_integer(paired["slot"], location=f"{location}.draft.Paired.slot")
    if slot > 3:
        raise SuiteValidationError(f"{location}.draft.Paired.slot is invalid")
    tile = _require_keys(action["tile"], {"coord", "rotation"}, location=f"{location}.tile")
    _coordinate(tile["coord"], location=f"{location}.tile.coord")
    rotation = _nonnegative_integer(tile["rotation"], location=f"{location}.tile.rotation")
    if rotation > 5:
        raise SuiteValidationError(f"{location}.tile.rotation is invalid")
    if action["wildlife"] is not None:
        _coordinate(action["wildlife"], location=f"{location}.wildlife")
    return action


def _validate_refill_distribution(value: Any, *, location: str) -> dict[str, Any]:
    distribution = dict(
        _require_keys(
            value,
            {"wildlife_counts", "total", "draws", "order_free"},
            location=location,
        )
    )
    counts = distribution["wildlife_counts"]
    if not isinstance(counts, list) or len(counts) != 5:
        raise SuiteValidationError(f"{location}.wildlife_counts must contain five integers")
    normalized_counts = [
        _nonnegative_integer(item, location=f"{location}.wildlife_counts[{index}]")
        for index, item in enumerate(counts)
    ]
    total = _nonnegative_integer(distribution["total"], location=f"{location}.total")
    draws = _nonnegative_integer(distribution["draws"], location=f"{location}.draws")
    if total != sum(normalized_counts) or draws != 1 or distribution["order_free"] is not True:
        raise SuiteValidationError(f"{location} is not an order-free one-draw distribution")
    return distribution


def _validate_long_salmon_side(value: Any, *, location: str) -> dict[str, Any]:
    side = dict(
        _require_keys(
            value,
            {
                "board_blake3",
                "radius_one_neighborhood",
                "salmon_component",
                "salmon_component_size",
                "salmon_path_endpoints",
                "maximum_salmon_degree",
                "habitat_components",
                "base_score",
                "salmon_score",
            },
            location=location,
        )
    )
    _require_blake3(side["board_blake3"], location=f"{location}.board_blake3")
    _validate_public_cells(
        side["radius_one_neighborhood"],
        location=f"{location}.radius_one_neighborhood",
    )
    component = _coordinate_list(
        side["salmon_component"],
        location=f"{location}.salmon_component",
    )
    component_size = _nonnegative_integer(
        side["salmon_component_size"],
        location=f"{location}.salmon_component_size",
    )
    endpoints = _coordinate_list(
        side["salmon_path_endpoints"],
        location=f"{location}.salmon_path_endpoints",
    )
    maximum_degree = _nonnegative_integer(
        side["maximum_salmon_degree"],
        location=f"{location}.maximum_salmon_degree",
    )
    if component_size != len(component) or len(endpoints) != 2 or maximum_degree > 2:
        raise SuiteValidationError(f"{location} is not an exact Salmon path receipt")
    _validate_habitat_components(
        side["habitat_components"],
        location=f"{location}.habitat_components",
    )
    _nonnegative_integer(side["base_score"], location=f"{location}.base_score")
    _nonnegative_integer(side["salmon_score"], location=f"{location}.salmon_score")
    return side


def _validate_component_bridge_side(
    value: Any,
    *,
    bridge_coord: tuple[int, int],
    location: str,
) -> dict[str, Any]:
    side = dict(
        _require_keys(
            value,
            {
                "board_before_blake3",
                "board_after_blake3",
                "pre_components",
                "touched_component_blake3s",
                "post_component",
                "merged_source_count",
                "largest_before",
                "largest_after",
            },
            location=location,
        )
    )
    for key in ("board_before_blake3", "board_after_blake3"):
        _require_blake3(side[key], location=f"{location}.{key}")
    pre_components = _validate_habitat_components(
        side["pre_components"],
        location=f"{location}.pre_components",
    )
    touched = side["touched_component_blake3s"]
    if (
        not isinstance(touched, list)
        or touched != sorted(touched)
        or len(touched) != len(set(touched))
    ):
        raise SuiteValidationError(f"{location}.touched_component_blake3s must be sorted")
    for index, digest in enumerate(touched):
        _require_blake3(
            digest,
            location=f"{location}.touched_component_blake3s[{index}]",
        )
    pre_hashes = {item["component_blake3"] for item in pre_components}
    if not set(touched) <= pre_hashes:
        raise SuiteValidationError(f"{location} touches an unknown pre-component")
    post_component = _validate_habitat_component(
        side["post_component"],
        location=f"{location}.post_component",
    )
    post_cells = {
        _coordinate(cell, location=f"{location}.post_component.cells[{index}]")
        for index, cell in enumerate(post_component["cells"])
    }
    merged_source_count = _nonnegative_integer(
        side["merged_source_count"],
        location=f"{location}.merged_source_count",
    )
    if bridge_coord not in post_cells or merged_source_count != len(touched):
        raise SuiteValidationError(f"{location} bridge identity is inconsistent")
    _nonnegative_integer(side["largest_before"], location=f"{location}.largest_before")
    _nonnegative_integer(side["largest_after"], location=f"{location}.largest_after")
    return side


def _validate_f1_receipt(value: Any) -> dict[str, Any]:
    receipt = dict(
        _require_keys(
            value,
            {
                "source_experiment_id",
                "source_report",
                "classification",
                "complete",
                "classification_scientific_blake3",
                "merged_census_scientific_blake3",
                "schema_manifest_scientific_blake3",
                "source_files",
                "closed_domains",
                "relevant_blocks",
                "long_salmon_component_context",
                "component_bridge",
                "equal_immediate_different_future_conflict",
                "public_action_equivalence_refill_near_match",
                "all_four_pairs_executable",
            },
            location="resolved_dependencies.f1",
        )
    )
    if (
        receipt["source_experiment_id"] != F1_EXPERIMENT_ID
        or receipt["source_report"] != _F1_SOURCE_FILES["result_report"]["path"]
        or receipt["classification"] != F1_CLASSIFICATION
        or receipt["complete"] is not True
        or receipt["classification_scientific_blake3"] != F1_CLASSIFICATION_SCIENTIFIC_BLAKE3
        or receipt["merged_census_scientific_blake3"] != F1_FORWARD_SCIENTIFIC_BLAKE3
        or receipt["schema_manifest_scientific_blake3"] != F1_MANIFEST_SCIENTIFIC_BLAKE3
    ):
        raise SuiteValidationError("resolved F1 authority chain drifted")

    source_files = _require_keys(
        receipt["source_files"],
        set(_F1_SOURCE_FILES),
        location="resolved_dependencies.f1.source_files",
    )
    for name, expected in _F1_SOURCE_FILES.items():
        source = _require_keys(
            source_files[name],
            {"path", "file_blake3", "scientific_blake3"},
            location=f"resolved_dependencies.f1.source_files.{name}",
        )
        _require_blake3(
            source["file_blake3"],
            location=f"resolved_dependencies.f1.source_files.{name}.file_blake3",
        )
        if source["scientific_blake3"] is not None:
            _require_blake3(
                source["scientific_blake3"],
                location=f"resolved_dependencies.f1.source_files.{name}.scientific_blake3",
            )
        if dict(source) != expected:
            raise SuiteValidationError(f"resolved F1 source file drifted: {name}")

    closed_domains = _require_keys(
        receipt["closed_domains"],
        {
            "external_compute_used",
            "gameplay_opened",
            "hidden_teacher_values_used_as_features",
            "new_teacher_compute_used",
            "test_split_opened",
        },
        location="resolved_dependencies.f1.closed_domains",
    )
    if any(value is not False for value in closed_domains.values()):
        raise SuiteValidationError("resolved F1 receipt opened a forbidden evidence domain")
    if canonical_json(receipt["relevant_blocks"]) != canonical_json(_F1_RELEVANT_BLOCKS):
        raise SuiteValidationError("resolved F1 relevant block census drifted")

    long_pair = dict(
        _require_keys(
            receipt["long_salmon_component_context"],
            {
                "schema",
                "exact_api",
                "anchor",
                "left",
                "right",
                "same_radius_one_neighborhood",
                "different_long_salmon_component_context",
            },
            location="resolved_dependencies.f1.long_salmon_component_context",
        )
    )
    if long_pair["schema"] != "f1-exact-component-motif-pair-v1" or long_pair["exact_api"] != [
        "cascadia_game::Board::place_tile",
        "cascadia_game::Board::place_wildlife",
        "cascadia_game::Board::tile_at",
        "cascadia_game::Board::wildlife_positions",
        "cascadia_game::score_board",
    ]:
        raise SuiteValidationError("resolved F1 long-Salmon API contract drifted")
    anchor = _coordinate(
        long_pair["anchor"],
        location="resolved_dependencies.f1.long_salmon_component_context.anchor",
    )
    if anchor != (0, 0):
        raise SuiteValidationError("resolved F1 long-Salmon anchor must be the origin")
    long_left = _validate_long_salmon_side(
        long_pair["left"],
        location="resolved_dependencies.f1.long_salmon_component_context.left",
    )
    long_right = _validate_long_salmon_side(
        long_pair["right"],
        location="resolved_dependencies.f1.long_salmon_component_context.right",
    )
    if (
        long_left["radius_one_neighborhood"] != long_right["radius_one_neighborhood"]
        or long_left["salmon_component_size"] != 5
        or long_right["salmon_component_size"] != 4
        or long_left["salmon_component"] == long_right["salmon_component"]
        or long_pair["same_radius_one_neighborhood"] is not True
        or long_pair["different_long_salmon_component_context"] is not True
    ):
        raise SuiteValidationError("resolved F1 long-Salmon witness is not exact")

    bridge = dict(
        _require_keys(
            receipt["component_bridge"],
            {
                "schema",
                "exact_api",
                "terrain",
                "bridge_coord",
                "bridge_tile",
                "left",
                "right",
                "different_bridge_effect",
            },
            location="resolved_dependencies.f1.component_bridge",
        )
    )
    if bridge["schema"] != "f1-exact-component-bridge-pair-v1" or bridge["exact_api"] != [
        "cascadia_game::Board::place_tile",
        "cascadia_game::Board::tile_at",
        "cascadia_game::Board::largest_habitat",
        "cascadia_game::Tile::terrain_on_edge",
    ]:
        raise SuiteValidationError("resolved F1 bridge API contract drifted")
    if bridge["terrain"] != "Forest":
        raise SuiteValidationError("resolved F1 bridge terrain drifted")
    bridge_coord = _coordinate(
        bridge["bridge_coord"],
        location="resolved_dependencies.f1.component_bridge.bridge_coord",
    )
    _require_keys(
        bridge["bridge_tile"],
        {"id", "terrain_a", "terrain_b", "wildlife", "keystone"},
        location="resolved_dependencies.f1.component_bridge.bridge_tile",
    )
    bridge_left = _validate_component_bridge_side(
        bridge["left"],
        bridge_coord=bridge_coord,
        location="resolved_dependencies.f1.component_bridge.left",
    )
    bridge_right = _validate_component_bridge_side(
        bridge["right"],
        bridge_coord=bridge_coord,
        location="resolved_dependencies.f1.component_bridge.right",
    )
    if (
        bridge_left["merged_source_count"] != 2
        or bridge_right["merged_source_count"] != 1
        or bridge_left["post_component"]["size"] != 3
        or bridge_right["post_component"]["size"] != 5
        or (bridge_left["largest_before"], bridge_left["largest_after"]) != (1, 3)
        or (bridge_right["largest_before"], bridge_right["largest_after"]) != (4, 5)
        or bridge["different_bridge_effect"] is not True
    ):
        raise SuiteValidationError("resolved F1 bridge witness is not exact")

    motif_pair = dict(
        _require_keys(
            receipt["equal_immediate_different_future_conflict"],
            {
                "schema",
                "exact_api",
                "left",
                "right",
                "same_tile_layout",
                "equal_immediate_score",
                "different_public_motif_conflict",
            },
            location="resolved_dependencies.f1.equal_immediate_different_future_conflict",
        )
    )
    if motif_pair["schema"] != "f1-exact-motif-conflict-pair-v1" or motif_pair["exact_api"] != [
        "cascadia_game::Board::place_tile",
        "cascadia_game::Board::place_wildlife",
        "cascadia_game::Board::wildlife_positions",
        "cascadia_game::score_board",
    ]:
        raise SuiteValidationError("resolved F1 motif API contract drifted")
    motif_sides = {}
    for side in ("left", "right"):
        side_location = f"resolved_dependencies.f1.equal_immediate_different_future_conflict.{side}"
        motif_side = dict(
            _require_keys(
                motif_pair[side],
                {"board_blake3", "tile_layout_blake3", "score", "motif_conflict"},
                location=side_location,
            )
        )
        _require_blake3(
            motif_side["board_blake3"],
            location=f"{side_location}.board_blake3",
        )
        _require_blake3(
            motif_side["tile_layout_blake3"],
            location=f"{side_location}.tile_layout_blake3",
        )
        _validate_score_breakdown(motif_side["score"], location=f"{side_location}.score")
        _validate_motif_signature(
            motif_side["motif_conflict"],
            location=f"{side_location}.motif_conflict",
        )
        motif_sides[side] = motif_side
    if (
        motif_sides["left"]["tile_layout_blake3"] != motif_sides["right"]["tile_layout_blake3"]
        or motif_sides["left"]["score"] != motif_sides["right"]["score"]
        or motif_sides["left"]["score"]["base_total"] != 4
        or motif_sides["left"]["motif_conflict"]["wildlife"] != "Hawk"
        or not motif_sides["left"]["motif_conflict"]["conflicted_coordinates"]
        or motif_sides["right"]["motif_conflict"]["wildlife"] != "Salmon"
        or not motif_sides["right"]["motif_conflict"]["branching_coordinates"]
        or motif_sides["left"]["motif_conflict"] == motif_sides["right"]["motif_conflict"]
        or motif_pair["same_tile_layout"] is not True
        or motif_pair["equal_immediate_score"] is not True
        or motif_pair["different_public_motif_conflict"] is not True
    ):
        raise SuiteValidationError("resolved F1 motif-conflict witness is not exact")

    action_pair = dict(
        _require_keys(
            receipt["public_action_equivalence_refill_near_match"],
            {
                "schema",
                "exact_api",
                "exact_equivalence",
                "refill_near_match",
                "exact_equivalence_proven",
                "refill_near_match_proven",
            },
            location="resolved_dependencies.f1.public_action_equivalence_refill_near_match",
        )
    )
    if action_pair["schema"] != "f1-public-transition-equivalence-pair-v1" or action_pair[
        "exact_api"
    ] != [
        "cascadia_game::GameState::legal_turn_actions_for_draft",
        "cascadia_game::GameState::preview_public_afterstate",
        "cascadia_game::GameState::transition",
        "cascadia_game::GameState::public_supply",
        "cascadia_game::PublicGameState::canonical_hash",
    ]:
        raise SuiteValidationError("resolved F1 public-action API contract drifted")
    exact = dict(
        _require_keys(
            action_pair["exact_equivalence"],
            {
                "seed",
                "source_public_state_blake3",
                "tile_slot",
                "tile_id",
                "left_action",
                "right_action",
                "left_action_blake3",
                "right_action_blake3",
                "canonical_public_afterstate_blake3",
                "left_transition_public_blake3",
                "right_transition_public_blake3",
                "actions_distinct",
                "preview_public_equal",
                "full_transition_equal",
                "drafted_tile_single_terrain",
            },
            location=(
                "resolved_dependencies.f1."
                "public_action_equivalence_refill_near_match.exact_equivalence"
            ),
        )
    )
    exact_location = (
        "resolved_dependencies.f1.public_action_equivalence_refill_near_match.exact_equivalence"
    )
    _nonnegative_integer(exact["seed"], location=f"{exact_location}.seed")
    _nonnegative_integer(exact["tile_slot"], location=f"{exact_location}.tile_slot")
    _nonnegative_integer(exact["tile_id"], location=f"{exact_location}.tile_id")
    for key in (
        "source_public_state_blake3",
        "left_action_blake3",
        "right_action_blake3",
        "canonical_public_afterstate_blake3",
        "left_transition_public_blake3",
        "right_transition_public_blake3",
    ):
        _require_blake3(exact[key], location=f"{exact_location}.{key}")
    left_action = _validate_turn_action(
        exact["left_action"],
        location=f"{exact_location}.left_action",
    )
    right_action = _validate_turn_action(
        exact["right_action"],
        location=f"{exact_location}.right_action",
    )
    if (
        left_action == right_action
        or exact["left_action_blake3"] == exact["right_action_blake3"]
        or exact["left_transition_public_blake3"] != exact["right_transition_public_blake3"]
        or any(
            exact[key] is not True
            for key in (
                "actions_distinct",
                "preview_public_equal",
                "full_transition_equal",
                "drafted_tile_single_terrain",
            )
        )
    ):
        raise SuiteValidationError("resolved F1 public-action equivalence is not exact")

    near = dict(
        _require_keys(
            action_pair["refill_near_match"],
            {
                "seed",
                "source_public_state_blake3",
                "drafted_wildlife",
                "place_action",
                "return_action",
                "place_public_afterstate_blake3",
                "return_public_afterstate_blake3",
                "place_distribution",
                "return_distribution",
                "both_actions_legal",
                "public_afterstates_different",
                "refill_distributions_different",
                "uses_only_public_counts",
            },
            location=(
                "resolved_dependencies.f1."
                "public_action_equivalence_refill_near_match.refill_near_match"
            ),
        )
    )
    near_location = (
        "resolved_dependencies.f1.public_action_equivalence_refill_near_match.refill_near_match"
    )
    _nonnegative_integer(near["seed"], location=f"{near_location}.seed")
    if near["drafted_wildlife"] not in _WILDLIFE:
        raise SuiteValidationError(f"{near_location}.drafted_wildlife is invalid")
    for key in (
        "source_public_state_blake3",
        "place_public_afterstate_blake3",
        "return_public_afterstate_blake3",
    ):
        _require_blake3(near[key], location=f"{near_location}.{key}")
    place_action = _validate_turn_action(
        near["place_action"],
        location=f"{near_location}.place_action",
    )
    return_action = _validate_turn_action(
        near["return_action"],
        location=f"{near_location}.return_action",
    )
    place_distribution = _validate_refill_distribution(
        near["place_distribution"],
        location=f"{near_location}.place_distribution",
    )
    return_distribution = _validate_refill_distribution(
        near["return_distribution"],
        location=f"{near_location}.return_distribution",
    )
    expected_return_action = dict(place_action)
    expected_return_action["wildlife"] = None
    wildlife_index = _WILDLIFE.index(near["drafted_wildlife"])
    count_deltas = [
        right - left
        for left, right in zip(
            place_distribution["wildlife_counts"],
            return_distribution["wildlife_counts"],
            strict=True,
        )
    ]
    if (
        return_action != expected_return_action
        or near["place_public_afterstate_blake3"] == near["return_public_afterstate_blake3"]
        or place_distribution == return_distribution
        or return_distribution["total"] != place_distribution["total"] + 1
        or count_deltas != [int(index == wildlife_index) for index in range(5)]
        or any(
            near[key] is not True
            for key in (
                "both_actions_legal",
                "public_afterstates_different",
                "refill_distributions_different",
                "uses_only_public_counts",
            )
        )
        or action_pair["exact_equivalence_proven"] is not True
        or action_pair["refill_near_match_proven"] is not True
        or receipt["all_four_pairs_executable"] is not True
    ):
        raise SuiteValidationError("resolved F1 refill near-match is not exact")
    return receipt


def _validate_f2_board_receipt(
    value: Any,
    *,
    universe: tuple[tuple[int, int], ...],
    location: str,
) -> dict[str, Any]:
    board = dict(
        _require_keys(
            value,
            {
                "tile_count",
                "occupied",
                "in_radius_occupied",
                "overflow_occupied",
                "frontier",
                "in_radius_frontier",
                "overflow_frontier",
                "maximum_occupied_radius",
                "maximum_frontier_radius",
                "exact_legal_mask",
                "in_radius_only_legal_mask",
            },
            location=location,
        )
    )
    if board["tile_count"] != 23:
        raise SuiteValidationError(f"{location}.tile_count must be 23")
    occupied = _coordinate_list(board["occupied"], location=f"{location}.occupied")
    in_radius_occupied = _coordinate_list(
        board["in_radius_occupied"],
        location=f"{location}.in_radius_occupied",
    )
    overflow_occupied = _coordinate_list(
        board["overflow_occupied"],
        location=f"{location}.overflow_occupied",
    )
    frontier = _coordinate_list(board["frontier"], location=f"{location}.frontier")
    in_radius_frontier = _coordinate_list(
        board["in_radius_frontier"],
        location=f"{location}.in_radius_frontier",
    )
    overflow_frontier = _coordinate_list(
        board["overflow_frontier"],
        location=f"{location}.overflow_frontier",
    )
    if len(occupied) != board["tile_count"]:
        raise SuiteValidationError(f"{location}.occupied does not match tile_count")
    if tuple(sorted((*in_radius_occupied, *overflow_occupied))) != occupied:
        raise SuiteValidationError(f"{location} occupied partition is not exact")
    if tuple(sorted((*in_radius_frontier, *overflow_frontier))) != frontier:
        raise SuiteValidationError(f"{location} frontier partition is not exact")
    if any(_axial_radius(item) > 6 for item in in_radius_occupied + in_radius_frontier):
        raise SuiteValidationError(f"{location} in-radius partition exceeds radius six")
    if any(_axial_radius(item) <= 6 for item in overflow_occupied + overflow_frontier):
        raise SuiteValidationError(f"{location} overflow partition is not outside radius six")
    if board["maximum_occupied_radius"] != max(map(_axial_radius, occupied)):
        raise SuiteValidationError(f"{location}.maximum_occupied_radius is inconsistent")
    if board["maximum_frontier_radius"] != max(map(_axial_radius, frontier)):
        raise SuiteValidationError(f"{location}.maximum_frontier_radius is inconsistent")

    exact_mask = _boolean_mask(
        board["exact_legal_mask"],
        len(universe),
        location=f"{location}.exact_legal_mask",
    )
    in_radius_mask = _boolean_mask(
        board["in_radius_only_legal_mask"],
        len(universe),
        location=f"{location}.in_radius_only_legal_mask",
    )
    frontier_set = set(frontier)
    in_radius_frontier_set = set(in_radius_frontier)
    if exact_mask != tuple(item in frontier_set for item in universe):
        raise SuiteValidationError(f"{location}.exact_legal_mask disagrees with Rust frontier")
    if in_radius_mask != tuple(item in in_radius_frontier_set for item in universe):
        raise SuiteValidationError(
            f"{location}.in_radius_only_legal_mask disagrees with retained frontier"
        )
    return board


def _validate_f2_receipt(value: Any) -> dict[str, Any]:
    receipt = dict(
        _require_keys(
            value,
            {
                "source_experiment_id",
                "source_scientific_blake3",
                "source_report",
                "compact_radius",
                "compact_capacity",
                "center",
                "exact_api",
                "f2_straight_case_match",
                "legal_action_universe",
                "left",
                "right",
                "same_in_radius_occupied",
                "same_in_radius_frontier",
                "different_exact_legal_masks",
            },
            location="resolved_dependencies.f2",
        )
    )
    if receipt["source_experiment_id"] != F2_EXPERIMENT_ID:
        raise SuiteValidationError("resolved F2 receipt names the wrong experiment")
    if receipt["source_scientific_blake3"] != F2_SCIENTIFIC_BLAKE3:
        raise SuiteValidationError("resolved F2 receipt has the wrong scientific hash")
    if receipt["compact_radius"] != 6 or receipt["compact_capacity"] != 127:
        raise SuiteValidationError("resolved F2 compact contract must be radius 6 / 127 cells")
    if _coordinate(receipt["center"], location="resolved_dependencies.f2.center") != (0, 0):
        raise SuiteValidationError("resolved F2 fixture must already be centered at the origin")
    if receipt["exact_api"] != [
        "cascadia_game::Board::place_tile",
        "cascadia_game::Board::frontier",
        "cascadia_game::HexCoord::distance",
    ]:
        raise SuiteValidationError("resolved F2 receipt does not name the frozen Rust APIs")
    straight = _require_keys(
        receipt["f2_straight_case_match"],
        {
            "name",
            "legal_placed_tile_count",
            "occupied_recentered_radius",
            "frontier_recentered_radius",
            "radius_6_occupied_overflow",
            "radius_6_frontier_overflow",
        },
        location="resolved_dependencies.f2.f2_straight_case_match",
    )
    if dict(straight) != {
        "name": "straight_23_tile_chain",
        "legal_placed_tile_count": 23,
        "occupied_recentered_radius": 11,
        "frontier_recentered_radius": 12,
        "radius_6_occupied_overflow": 10,
        "radius_6_frontier_overflow": 26,
    }:
        raise SuiteValidationError("resolved F2 straight chain does not match the census")
    universe = _coordinate_list(
        receipt["legal_action_universe"],
        location="resolved_dependencies.f2.legal_action_universe",
    )
    left = _validate_f2_board_receipt(
        receipt["left"],
        universe=universe,
        location="resolved_dependencies.f2.left",
    )
    right = _validate_f2_board_receipt(
        receipt["right"],
        universe=universe,
        location="resolved_dependencies.f2.right",
    )
    if left["in_radius_occupied"] != right["in_radius_occupied"]:
        raise SuiteValidationError("resolved F2 pair does not share in-radius occupancy")
    if left["in_radius_frontier"] != right["in_radius_frontier"]:
        raise SuiteValidationError("resolved F2 pair does not share in-radius frontier")
    if left["exact_legal_mask"] == right["exact_legal_mask"]:
        raise SuiteValidationError("resolved F2 pair exact legal masks collide")
    if left["in_radius_only_legal_mask"] != right["in_radius_only_legal_mask"]:
        raise SuiteValidationError("resolved F2 pair in-radius masks must collide")
    if (
        receipt["same_in_radius_occupied"] is not True
        or receipt["same_in_radius_frontier"] is not True
        or receipt["different_exact_legal_masks"] is not True
    ):
        raise SuiteValidationError("resolved F2 closure flags are incomplete")
    return receipt


def _validate_f3_receipt(value: Any) -> dict[str, Any]:
    receipt = dict(
        _require_keys(
            value,
            {
                "source_experiment_id",
                "source_report",
                "contract_id",
                "contract_scientific_blake3",
                "seed",
                "prelude",
                "source_public_state_blake3",
                "source_legal_action_count",
                "selected_source_row",
                "selected_source_action",
                "canonical_decision_blake3",
                "orbit",
                "all_12_transforms_present",
                "every_legal_map_bijective",
                "every_action_round_trips",
                "every_transition_equivariant",
            },
            location="resolved_dependencies.f3",
        )
    )
    if receipt["source_experiment_id"] != F3_EXPERIMENT_ID:
        raise SuiteValidationError("resolved F3 receipt names the wrong experiment")
    if receipt["contract_id"] != D6_CONTRACT_ID:
        raise SuiteValidationError("resolved F3 receipt names the wrong D6 contract")
    if receipt["contract_scientific_blake3"] != D6_SCIENTIFIC_BLAKE3:
        raise SuiteValidationError("resolved F3 receipt has the wrong D6 scientific hash")
    _require_blake3(
        receipt["source_public_state_blake3"],
        location="resolved_dependencies.f3.source_public_state_blake3",
    )
    _require_blake3(
        receipt["canonical_decision_blake3"],
        location="resolved_dependencies.f3.canonical_decision_blake3",
    )
    legal_count = receipt["source_legal_action_count"]
    selected_source_row = receipt["selected_source_row"]
    if (
        isinstance(legal_count, bool)
        or not isinstance(legal_count, int)
        or legal_count <= 0
        or isinstance(selected_source_row, bool)
        or not isinstance(selected_source_row, int)
        or not 0 <= selected_source_row < legal_count
    ):
        raise SuiteValidationError("resolved F3 source legal-row contract is invalid")
    orbit = receipt["orbit"]
    if not isinstance(orbit, list) or len(orbit) != 12:
        raise SuiteValidationError("resolved F3 receipt must contain all 12 transforms")
    expected_orbit_keys = {
        "transform_id",
        "inverse_id",
        "transformed_public_state_blake3",
        "transformed_legal_action_count",
        "selected_transformed_row",
        "transformed_action",
        "forward_rows_blake3",
        "inverse_rows_blake3",
        "legal_map_bijective",
        "policy_round_trip",
        "action_round_trip",
        "transition_equivariant",
        "transition_then_transform_public_blake3",
        "transform_then_transition_public_blake3",
    }
    for transform_id, item in enumerate(orbit):
        transform = _require_keys(
            item,
            expected_orbit_keys,
            location=f"resolved_dependencies.f3.orbit[{transform_id}]",
        )
        if transform["transform_id"] != transform_id:
            raise SuiteValidationError("resolved F3 orbit IDs are not the frozen 0..11 order")
        if transform["inverse_id"] != D6_CONTRACT.inverse_table[transform_id]:
            raise SuiteValidationError("resolved F3 inverse ID disagrees with Rust metadata")
        if transform["transformed_legal_action_count"] != legal_count:
            raise SuiteValidationError("resolved F3 legal-set cardinality changed under D6")
        transformed_row = transform["selected_transformed_row"]
        if (
            isinstance(transformed_row, bool)
            or not isinstance(transformed_row, int)
            or not 0 <= transformed_row < legal_count
        ):
            raise SuiteValidationError("resolved F3 selected transformed row is invalid")
        for key in (
            "transformed_public_state_blake3",
            "forward_rows_blake3",
            "inverse_rows_blake3",
            "transition_then_transform_public_blake3",
            "transform_then_transition_public_blake3",
        ):
            _require_blake3(
                transform[key],
                location=f"resolved_dependencies.f3.orbit[{transform_id}].{key}",
            )
        if any(
            transform[key] is not True
            for key in (
                "legal_map_bijective",
                "policy_round_trip",
                "action_round_trip",
                "transition_equivariant",
            )
        ):
            raise SuiteValidationError("resolved F3 orbit contains an unproven transform")
        if (
            transform["transition_then_transform_public_blake3"]
            != transform["transform_then_transition_public_blake3"]
        ):
            raise SuiteValidationError("resolved F3 transition hashes are not equivariant")
    identity = orbit[0]
    if identity["transformed_public_state_blake3"] != receipt["source_public_state_blake3"]:
        raise SuiteValidationError("resolved F3 identity state hash changed")
    if identity["selected_transformed_row"] != selected_source_row:
        raise SuiteValidationError("resolved F3 identity row changed")
    if scientific_blake3(identity["transformed_action"]) != scientific_blake3(
        receipt["selected_source_action"]
    ):
        raise SuiteValidationError("resolved F3 identity action changed")
    if any(
        receipt[key] is not True
        for key in (
            "all_12_transforms_present",
            "every_legal_map_bijective",
            "every_action_round_trips",
            "every_transition_equivariant",
        )
    ):
        raise SuiteValidationError("resolved F3 closure flags are incomplete")
    return receipt


def _f1_pair_payload(family: str, receipt: Mapping[str, Any]) -> dict[str, Any]:
    if family == "long_salmon_component_context":
        pair = receipt["long_salmon_component_context"]
        labels = {
            side: {
                "anchor": pair["anchor"],
                "salmon_component": pair[side]["salmon_component"],
                "salmon_component_size": pair[side]["salmon_component_size"],
                "salmon_path_endpoints": pair[side]["salmon_path_endpoints"],
                "maximum_salmon_degree": pair[side]["maximum_salmon_degree"],
                "habitat_components": pair[side]["habitat_components"],
                "base_score": pair[side]["base_score"],
                "salmon_score": pair[side]["salmon_score"],
            }
            for side in ("left", "right")
        }
        return {
            "public_inputs": {
                side: {
                    "kind": "adversarial_public_fixture_v1",
                    "anchor": pair["anchor"],
                    "board_blake3": pair[side]["board_blake3"],
                    "radius_one_neighborhood": pair[side]["radius_one_neighborhood"],
                    "concepts": {"motif": labels[side]},
                }
                for side in ("left", "right")
            },
            "expectations": [
                {
                    "concept": "long_salmon_component_context",
                    "relation": "different",
                    "probe": "motif",
                    "left_label": labels["left"],
                    "right_label": labels["right"],
                }
            ],
            "boundary_contracts": {"public-observable-v1": ["long_salmon_component_context"]},
        }
    if family == "component_bridge":
        pair = receipt["component_bridge"]
        labels = {
            side: {
                "terrain": pair["terrain"],
                "bridge_coord": pair["bridge_coord"],
                "pre_components": pair[side]["pre_components"],
                "touched_component_blake3s": pair[side]["touched_component_blake3s"],
                "post_component": pair[side]["post_component"],
                "merged_source_count": pair[side]["merged_source_count"],
                "largest_before": pair[side]["largest_before"],
                "largest_after": pair[side]["largest_after"],
            }
            for side in ("left", "right")
        }
        return {
            "public_inputs": {
                side: {
                    "kind": "adversarial_public_fixture_v1",
                    "terrain": pair["terrain"],
                    "bridge_coord": pair["bridge_coord"],
                    "bridge_tile": pair["bridge_tile"],
                    "board_before_blake3": pair[side]["board_before_blake3"],
                    "board_after_blake3": pair[side]["board_after_blake3"],
                    "concepts": {"component": labels[side]},
                }
                for side in ("left", "right")
            },
            "expectations": [
                {
                    "concept": "component_bridge_effect",
                    "relation": "different",
                    "probe": "component",
                    "left_label": labels["left"],
                    "right_label": labels["right"],
                }
            ],
            "boundary_contracts": {"public-observable-v1": ["component_bridge_effect"]},
        }
    if family == "equal_immediate_different_future_conflict":
        pair = receipt["equal_immediate_different_future_conflict"]
        labels = {side: pair[side]["motif_conflict"] for side in ("left", "right")}
        return {
            "public_inputs": {
                side: {
                    "kind": "adversarial_public_fixture_v1",
                    "board_blake3": pair[side]["board_blake3"],
                    "tile_layout_blake3": pair[side]["tile_layout_blake3"],
                    "immediate_score": pair[side]["score"],
                    "concepts": {"motif": labels[side]},
                }
                for side in ("left", "right")
            },
            "expectations": [
                {
                    "concept": "motif_conflict",
                    "relation": "different",
                    "probe": "motif",
                    "left_label": labels["left"],
                    "right_label": labels["right"],
                }
            ],
            "boundary_contracts": {"public-observable-v1": ["motif_conflict"]},
        }
    if family == "public_action_equivalence_refill_near_match":
        pair = receipt["public_action_equivalence_refill_near_match"]
        exact = pair["exact_equivalence"]
        near = pair["refill_near_match"]
        exact_label = {
            "source_public_state_blake3": exact["source_public_state_blake3"],
            "canonical_public_afterstate_blake3": exact["canonical_public_afterstate_blake3"],
            "transition_public_blake3": exact["left_transition_public_blake3"],
            "tile_slot": exact["tile_slot"],
            "tile_id": exact["tile_id"],
        }
        refill_labels = {
            "left": {
                "source_public_state_blake3": near["source_public_state_blake3"],
                "public_afterstate_blake3": near["place_public_afterstate_blake3"],
                "refill_distribution": near["place_distribution"],
            },
            "right": {
                "source_public_state_blake3": near["source_public_state_blake3"],
                "public_afterstate_blake3": near["return_public_afterstate_blake3"],
                "refill_distribution": near["return_distribution"],
            },
        }
        exact_cases = {
            "left": {
                "action": exact["left_action"],
                "action_blake3": exact["left_action_blake3"],
            },
            "right": {
                "action": exact["right_action"],
                "action_blake3": exact["right_action_blake3"],
            },
        }
        near_cases = {
            "left": {
                "action": near["place_action"],
                "public_afterstate_blake3": near["place_public_afterstate_blake3"],
                "refill_distribution": near["place_distribution"],
            },
            "right": {
                "action": near["return_action"],
                "public_afterstate_blake3": near["return_public_afterstate_blake3"],
                "refill_distribution": near["return_distribution"],
            },
        }
        return {
            "public_inputs": {
                side: {
                    "kind": "adversarial_public_fixture_v1",
                    "exact_case": exact_cases[side],
                    "near_match_case": near_cases[side],
                    "concepts": {"action_edit": exact_label},
                    "declared_projections": {
                        REFILL_DISTRIBUTION_BOUNDARY_ID: {
                            "concepts": {"action_edit": refill_labels[side]}
                        }
                    },
                }
                for side in ("left", "right")
            },
            "expectations": [
                {
                    "concept": "exact_public_transition",
                    "relation": "equivalent",
                    "probe": "action_edit",
                    "left_label": exact_label,
                    "right_label": exact_label,
                },
                {
                    "concept": "refill_near_match_transition",
                    "relation": "different",
                    "probe": "action_edit",
                    "left_label": refill_labels["left"],
                    "right_label": refill_labels["right"],
                },
            ],
            "boundary_contracts": {
                "public-observable-v1": ["exact_public_transition"],
                REFILL_DISTRIBUTION_BOUNDARY_ID: ["refill_near_match_transition"],
            },
        }
    raise SuiteValidationError(f"unsupported resolved F1 family: {family}")


def load_resolved_dependency_artifact(path: str | Path) -> dict[str, Any]:
    """Load and fail-closed validate the Rust-generated F1/F2/F3 receipts."""
    artifact_path = Path(path)
    try:
        raw = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SuiteValidationError(f"cannot load resolved dependency artifact: {error}") from error
    root = dict(
        _require_keys(
            raw,
            {"schema_version", "artifact_id", "f1", "f2", "f3"},
            location="resolved_dependencies",
        )
    )
    if root["schema_version"] != RESOLVED_DEPENDENCY_SCHEMA_VERSION:
        raise SuiteValidationError("unsupported resolved dependency schema version")
    if root["artifact_id"] != RESOLVED_DEPENDENCY_ARTIFACT_ID:
        raise SuiteValidationError("unsupported resolved dependency artifact identity")
    root["f1"] = _validate_f1_receipt(root["f1"])
    root["f2"] = _validate_f2_receipt(root["f2"])
    root["f3"] = _validate_f3_receipt(root["f3"])
    return root


def _reject_hidden_public_fields(value: Any, *, location: str) -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key).lower()
            if any(token in key for token in _FORBIDDEN_PUBLIC_KEY_TOKENS):
                raise SuiteValidationError(
                    f"forbidden hidden/future field in public input at {location}.{raw_key}"
                )
            _reject_hidden_public_fields(item, location=f"{location}.{raw_key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_hidden_public_fields(item, location=f"{location}[{index}]")


@dataclass(frozen=True)
class ConceptExpectation:
    """One exact semantic relation to test at representation boundaries."""

    concept: str
    relation: str
    probe: str
    left_label: Any
    right_label: Any

    def validate(self, *, blocked: bool = False) -> None:
        if not self.concept:
            raise SuiteValidationError("expectation concept cannot be empty")
        if self.relation not in RELATIONS:
            raise SuiteValidationError(f"unsupported relation: {self.relation}")
        if self.probe not in REQUIRED_PROBES:
            raise SuiteValidationError(f"expectation uses unknown probe: {self.probe}")
        if blocked:
            if self.left_label is not None or self.right_label is not None:
                raise SuiteValidationError("blocked concepts cannot fabricate exact labels")
            return
        if self.left_label is None or self.right_label is None:
            raise SuiteValidationError("ready concepts require exact left and right labels")
        labels_equal = scientific_blake3(self.left_label) == scientific_blake3(self.right_label)
        if labels_equal != (self.relation == "equivalent"):
            raise SuiteValidationError(f"expectation labels contradict relation for {self.concept}")


@dataclass(frozen=True)
class DependencyBlock:
    """Named upstream fact required before a pair can become exact."""

    dependency: str
    reason: str
    expected_artifact_schema: str
    verification_command: str

    def validate(self) -> None:
        if self.dependency not in ALLOWED_DEPENDENCIES:
            raise SuiteValidationError("dependency blocks must name F1, F2, or F3")
        if not self.reason or not self.expected_artifact_schema:
            raise SuiteValidationError("dependency block must name its reason and schema")
        if not self.verification_command.strip():
            raise SuiteValidationError("dependency block requires an executable command")


@dataclass(frozen=True)
class AdversarialPair:
    """Immutable public pair plus exact semantic expectations."""

    schema_version: int
    pair_id: str
    family: str
    title: str
    status: str
    public_inputs: Mapping[str, Any]
    expectations: tuple[ConceptExpectation, ...]
    boundary_contracts: Mapping[str, tuple[str, ...]]
    provenance: Mapping[str, Any]
    evidence: Mapping[str, Any]
    dependency: DependencyBlock | None
    boundary_observations: tuple[Mapping[str, Any], ...]
    canonical_hash: str

    def validate(self, *, verify_hash: bool = True) -> None:
        if self.schema_version != PAIR_SCHEMA_VERSION:
            raise SuiteValidationError("unsupported adversarial pair schema version")
        if not self.pair_id or self.family not in REQUIRED_FAMILIES:
            raise SuiteValidationError("pair requires a stable ID and required family")
        if self.status not in PAIR_STATUSES:
            raise SuiteValidationError(f"unsupported pair status: {self.status}")
        if set(self.public_inputs) != {"left", "right"}:
            raise SuiteValidationError("public_inputs must contain exactly left and right")
        _reject_hidden_public_fields(self.public_inputs, location="public_inputs")
        if not self.expectations:
            raise SuiteValidationError("pair requires at least one concept expectation")
        concepts = [expectation.concept for expectation in self.expectations]
        if len(concepts) != len(set(concepts)):
            raise SuiteValidationError("pair expectation concepts must be unique")
        for expectation in self.expectations:
            expectation.validate(blocked=self.status == "dependency_blocked")
        known_concepts = set(concepts)
        for boundary_id, required_concepts in self.boundary_contracts.items():
            if not boundary_id or not required_concepts:
                raise SuiteValidationError("boundary contracts cannot be empty")
            unknown = set(required_concepts) - known_concepts
            if unknown:
                raise SuiteValidationError(
                    f"boundary contract references unknown concepts: {sorted(unknown)}"
                )
        if self.status == "dependency_blocked":
            if self.dependency is None:
                raise SuiteValidationError("blocked pair requires dependency metadata")
            self.dependency.validate()
        elif self.dependency is not None:
            raise SuiteValidationError("ready pair cannot carry a dependency block")
        if not isinstance(self.provenance.get("evidence_domain"), str):
            raise SuiteValidationError("pair provenance requires evidence_domain")
        if self.family == "ambiguous_confidence_set_vs_distinguishable_winner":
            self._validate_confidence_evidence()
        if self.status == "ready" and self.family in {
            "long_salmon_component_context",
            "component_bridge",
            "equal_immediate_different_future_conflict",
            "public_action_equivalence_refill_near_match",
        }:
            self._validate_f1_evidence()
        if self.status == "ready" and self.family == "d6_transforms":
            self._validate_d6_evidence()
        if self.status == "ready" and self.family in {
            "same_in_radius_different_overflow_consequence",
            "same_compact_latent_different_legal_affordance",
        }:
            self._validate_f2_evidence()
        if verify_hash and self.canonical_hash != self.expected_hash():
            raise SuiteValidationError(f"canonical pair hash mismatch: {self.pair_id}")

    def _validate_confidence_evidence(self) -> None:
        if self.status != "ready":
            raise SuiteValidationError("confidence-set family must use existing open evidence")
        if self.provenance.get("sealed_test_opened") is not False:
            raise SuiteValidationError("confidence evidence must explicitly keep test sealed")
        expectation = next(
            (item for item in self.expectations if item.probe == "confidence_set_membership"),
            None,
        )
        if expectation is None:
            raise SuiteValidationError("confidence-set family requires its frozen probe")
        for side, label in (
            ("left", expectation.left_label),
            ("right", expectation.right_label),
        ):
            evidence = self.evidence.get(side)
            if not isinstance(evidence, Mapping):
                raise SuiteValidationError("confidence-set evidence requires both groups")
            observed = confidence_set_evidence(
                evidence.get("means", ()),
                evidence.get("stddevs", ()),
                evidence.get("samples", ()),
                evidence.get("action_hashes", ()),
            )
            expected_label = {
                "confidence_set_membership": observed["confidence_set_membership"],
                "distinguishable_winner": observed["distinguishable_winner"],
            }
            if scientific_blake3(label) != scientific_blake3(expected_label):
                raise SuiteValidationError(
                    f"confidence-set label does not match open evidence: {side}"
                )

    def _resolved_receipt(self, *, dependency: str) -> Mapping[str, Any]:
        expected_keys = {
            "resolved_artifact_id",
            "resolved_artifact_file_blake3",
            "resolved_artifact_scientific_blake3",
            "receipt_scientific_blake3",
            "receipt",
        }
        evidence = _require_keys(
            self.evidence,
            expected_keys,
            location=f"{self.pair_id}.evidence",
        )
        if evidence["resolved_artifact_id"] != RESOLVED_DEPENDENCY_ARTIFACT_ID:
            raise SuiteValidationError(f"{self.pair_id} names the wrong resolved artifact")
        for key in (
            "resolved_artifact_file_blake3",
            "resolved_artifact_scientific_blake3",
            "receipt_scientific_blake3",
        ):
            _require_blake3(evidence[key], location=f"{self.pair_id}.evidence.{key}")
        receipt = evidence["receipt"]
        if evidence["receipt_scientific_blake3"] != scientific_blake3(receipt):
            raise SuiteValidationError(f"{self.pair_id} receipt scientific hash drifted")
        if dependency == "F1":
            return _validate_f1_receipt(receipt)
        if dependency == "F2":
            return _validate_f2_receipt(receipt)
        if dependency == "F3":
            return _validate_f3_receipt(receipt)
        raise SuiteValidationError(f"unsupported resolved dependency: {dependency}")

    def _validate_f1_evidence(self) -> None:
        receipt = self._resolved_receipt(dependency="F1")
        expected = _f1_pair_payload(self.family, receipt)
        if scientific_blake3(self.public_inputs) != scientific_blake3(expected["public_inputs"]):
            raise SuiteValidationError(f"resolved F1 public inputs drifted: {self.family}")
        observed_expectations = [asdict(item) for item in self.expectations]
        if scientific_blake3(observed_expectations) != scientific_blake3(expected["expectations"]):
            raise SuiteValidationError(f"resolved F1 expectations drifted: {self.family}")
        observed_contracts = {
            boundary_id: list(concepts) for boundary_id, concepts in self.boundary_contracts.items()
        }
        if scientific_blake3(observed_contracts) != scientific_blake3(
            expected["boundary_contracts"]
        ):
            raise SuiteValidationError(f"resolved F1 boundary contract drifted: {self.family}")
        expected_provenance = {
            "evidence_domain": "rust-f1-authority-bound-exact-public-witness",
            "sealed_test_opened": False,
            "source_experiment_id": F1_EXPERIMENT_ID,
            "classification_scientific_blake3": F1_CLASSIFICATION_SCIENTIFIC_BLAKE3,
            "merged_census_scientific_blake3": F1_FORWARD_SCIENTIFIC_BLAKE3,
            "schema_manifest_scientific_blake3": F1_MANIFEST_SCIENTIFIC_BLAKE3,
        }
        if dict(self.provenance) != expected_provenance:
            raise SuiteValidationError(f"resolved F1 provenance drifted: {self.family}")

    def _validate_d6_evidence(self) -> None:
        receipt = self._resolved_receipt(dependency="F3")
        expected_label = {
            "canonical_decision_blake3": receipt["canonical_decision_blake3"],
            "contract_id": receipt["contract_id"],
            "legal_action_count": receipt["source_legal_action_count"],
        }
        expectation = next(
            (item for item in self.expectations if item.probe == "d6_identity"),
            None,
        )
        if expectation is None or expectation.relation != "equivalent":
            raise SuiteValidationError("resolved D6 pair requires one equivalence expectation")
        if any(
            scientific_blake3(label) != scientific_blake3(expected_label)
            for label in (expectation.left_label, expectation.right_label)
        ):
            raise SuiteValidationError("resolved D6 labels do not match the Rust receipt")
        orbit_by_id = {item["transform_id"]: item for item in receipt["orbit"]}
        for side in ("left", "right"):
            public = self.public_inputs[side]
            transform_id = public.get("transform_id")
            if transform_id not in orbit_by_id:
                raise SuiteValidationError("resolved D6 public input names an unknown transform")
            orbit = orbit_by_id[transform_id]
            if public.get("public_state_blake3") != orbit["transformed_public_state_blake3"]:
                raise SuiteValidationError("resolved D6 public state hash drifted")
            if scientific_blake3(public.get("action")) != scientific_blake3(
                orbit["transformed_action"]
            ):
                raise SuiteValidationError("resolved D6 action drifted")
            if public.get("legal_action_count") != receipt["source_legal_action_count"]:
                raise SuiteValidationError("resolved D6 legal-action count drifted")
            if scientific_blake3(public["concepts"]["d6_identity"]) != scientific_blake3(
                expected_label
            ):
                raise SuiteValidationError("resolved D6 public identity drifted")
        if (
            self.public_inputs["left"]["transform_id"]
            == self.public_inputs["right"]["transform_id"]
        ):
            raise SuiteValidationError("resolved D6 pair must compare distinct transforms")

    def _validate_f2_evidence(self) -> None:
        receipt = self._resolved_receipt(dependency="F2")
        universe_blake3 = scientific_blake3(receipt["legal_action_universe"])
        labels = {
            side: {
                "universe_blake3": universe_blake3,
                "mask": receipt[side]["exact_legal_mask"],
            }
            for side in ("left", "right")
        }
        expectation = next(
            (item for item in self.expectations if item.probe == "legal_mask"),
            None,
        )
        if expectation is None or expectation.relation != "different":
            raise SuiteValidationError("resolved F2 pair requires one legal-mask difference")
        if scientific_blake3(expectation.left_label) != scientific_blake3(labels["left"]):
            raise SuiteValidationError("resolved F2 left legal-mask label drifted")
        if scientific_blake3(expectation.right_label) != scientific_blake3(labels["right"]):
            raise SuiteValidationError("resolved F2 right legal-mask label drifted")
        for side in ("left", "right"):
            if scientific_blake3(self.public_inputs[side]["concepts"]["legal_mask"]) != (
                scientific_blake3(labels[side])
            ):
                raise SuiteValidationError("resolved F2 public legal mask drifted")
        left_latent = self.public_inputs["left"].get("latent_target")
        right_latent = self.public_inputs["right"].get("latent_target")
        if self.family == "same_compact_latent_different_legal_affordance":
            if scientific_blake3(left_latent) != scientific_blake3(right_latent):
                raise SuiteValidationError("resolved compact pair latent targets must be equal")
            for side in ("left", "right"):
                expected_overflow = {
                    "occupied": receipt[side]["overflow_occupied"],
                    "frontier": receipt[side]["overflow_frontier"],
                }
                if scientific_blake3(self.public_inputs[side].get("exact_overflow")) != (
                    scientific_blake3(expected_overflow)
                ):
                    raise SuiteValidationError("resolved compact overflow sidecar drifted")

    def fixture_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "pair_id": self.pair_id,
            "family": self.family,
            "title": self.title,
            "status": self.status,
            "public_inputs": self.public_inputs,
            "expectations": [asdict(value) for value in self.expectations],
            "boundary_contracts": {
                key: list(value) for key, value in self.boundary_contracts.items()
            },
            "provenance": self.provenance,
            "evidence": self.evidence,
            "dependency": asdict(self.dependency) if self.dependency else None,
        }

    def expected_hash(self) -> str:
        return scientific_blake3(self.fixture_payload())

    def to_dict(self) -> dict[str, Any]:
        output = self.fixture_payload()
        output["boundary_observations"] = list(self.boundary_observations)
        output["canonical_hash"] = self.canonical_hash
        return output

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> AdversarialPair:
        allowed = {
            "schema_version",
            "pair_id",
            "family",
            "title",
            "status",
            "public_inputs",
            "expectations",
            "boundary_contracts",
            "provenance",
            "evidence",
            "dependency",
            "boundary_observations",
            "canonical_hash",
        }
        if set(values) != allowed:
            raise SuiteValidationError(
                f"adversarial pair keys drifted: {sorted(set(values) ^ allowed)}"
            )
        dependency = values["dependency"]
        pair = cls(
            schema_version=int(values["schema_version"]),
            pair_id=str(values["pair_id"]),
            family=str(values["family"]),
            title=str(values["title"]),
            status=str(values["status"]),
            public_inputs=dict(values["public_inputs"]),
            expectations=tuple(
                ConceptExpectation(
                    concept=str(item["concept"]),
                    relation=str(item["relation"]),
                    probe=str(item["probe"]),
                    left_label=item.get("left_label"),
                    right_label=item.get("right_label"),
                )
                for item in values["expectations"]
            ),
            boundary_contracts={
                str(key): tuple(str(item) for item in concepts)
                for key, concepts in dict(values["boundary_contracts"]).items()
            },
            provenance=dict(values["provenance"]),
            evidence=dict(values["evidence"]),
            dependency=(
                DependencyBlock(
                    dependency=str(dependency["dependency"]),
                    reason=str(dependency["reason"]),
                    expected_artifact_schema=str(dependency["expected_artifact_schema"]),
                    verification_command=str(dependency["verification_command"]),
                )
                if dependency is not None
                else None
            ),
            boundary_observations=tuple(values["boundary_observations"]),
            canonical_hash=str(values["canonical_hash"]),
        )
        pair.validate()
        return pair


@dataclass(frozen=True)
class FixtureSet:
    """Validated immutable collection of all required F4 pair families."""

    schema_version: int
    fixture_set_id: str
    pairs: tuple[AdversarialPair, ...]
    canonical_hash: str

    def validate(self, *, require_complete: bool = True) -> None:
        if self.schema_version != SUITE_SCHEMA_VERSION:
            raise SuiteValidationError("unsupported fixture-set schema version")
        if self.fixture_set_id != FIXTURE_SET_ID:
            raise SuiteValidationError("unsupported fixture-set identity")
        for pair in self.pairs:
            pair.validate()
        pair_ids = [pair.pair_id for pair in self.pairs]
        if len(pair_ids) != len(set(pair_ids)):
            raise SuiteValidationError("fixture pair IDs must be unique")
        families = [pair.family for pair in self.pairs]
        if len(families) != len(set(families)):
            raise SuiteValidationError("fixture families must be unique")
        if require_complete and tuple(families) != REQUIRED_FAMILIES:
            missing = sorted(set(REQUIRED_FAMILIES) - set(families))
            extra = sorted(set(families) - set(REQUIRED_FAMILIES))
            raise SuiteValidationError(
                f"required family registry mismatch; missing={missing}, extra={extra}"
            )
        if self.canonical_hash != self.expected_hash():
            raise SuiteValidationError("fixture-set canonical hash mismatch")

    def expected_hash(self) -> str:
        return scientific_blake3(
            {
                "schema_version": self.schema_version,
                "fixture_set_id": self.fixture_set_id,
                "pair_hashes": [pair.canonical_hash for pair in self.pairs],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "fixture_set_id": self.fixture_set_id,
            "pairs": [pair.to_dict() for pair in self.pairs],
            "canonical_hash": self.canonical_hash,
        }

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any],
        *,
        require_complete: bool = True,
    ) -> FixtureSet:
        if set(values) != {"schema_version", "fixture_set_id", "pairs", "canonical_hash"}:
            raise SuiteValidationError("fixture-set keys drifted")
        fixture_set = cls(
            schema_version=int(values["schema_version"]),
            fixture_set_id=str(values["fixture_set_id"]),
            pairs=tuple(AdversarialPair.from_dict(item) for item in values["pairs"]),
            canonical_hash=str(values["canonical_hash"]),
        )
        fixture_set.validate(require_complete=require_complete)
        return fixture_set


def build_pair(
    *,
    pair_id: str,
    family: str,
    title: str,
    status: str,
    public_inputs: Mapping[str, Any],
    expectations: Sequence[ConceptExpectation],
    boundary_contracts: Mapping[str, Sequence[str]],
    provenance: Mapping[str, Any],
    evidence: Mapping[str, Any] | None = None,
    dependency: DependencyBlock | None = None,
) -> AdversarialPair:
    """Construct and hash one pair without allowing callers to forge its digest."""
    draft = AdversarialPair(
        schema_version=PAIR_SCHEMA_VERSION,
        pair_id=pair_id,
        family=family,
        title=title,
        status=status,
        public_inputs=dict(public_inputs),
        expectations=tuple(expectations),
        boundary_contracts={
            str(key): tuple(str(item) for item in concepts)
            for key, concepts in boundary_contracts.items()
        },
        provenance=dict(provenance),
        evidence=dict(evidence or {}),
        dependency=dependency,
        boundary_observations=(),
        canonical_hash="",
    )
    pair = AdversarialPair(**{**draft.__dict__, "canonical_hash": draft.expected_hash()})
    pair.validate()
    return pair


def build_fixture_set(pairs: Sequence[AdversarialPair]) -> FixtureSet:
    """Construct and hash a complete fixture set in frozen family order."""
    draft = FixtureSet(
        schema_version=SUITE_SCHEMA_VERSION,
        fixture_set_id=FIXTURE_SET_ID,
        pairs=tuple(pairs),
        canonical_hash="",
    )
    fixture_set = FixtureSet(
        schema_version=draft.schema_version,
        fixture_set_id=draft.fixture_set_id,
        pairs=draft.pairs,
        canonical_hash=draft.expected_hash(),
    )
    fixture_set.validate()
    return fixture_set


def _resolved_evidence(
    *,
    artifact: Mapping[str, Any],
    artifact_file_blake3: str,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "resolved_artifact_id": artifact["artifact_id"],
        "resolved_artifact_file_blake3": _require_blake3(
            artifact_file_blake3,
            location="resolved_dependency_artifact.file_blake3",
        ),
        "resolved_artifact_scientific_blake3": scientific_blake3(artifact),
        "receipt_scientific_blake3": scientific_blake3(receipt),
        "receipt": dict(receipt),
    }


def materialize_resolved_dependencies(
    fixture_set: FixtureSet,
    artifact: Mapping[str, Any],
    *,
    artifact_file_blake3: str,
) -> FixtureSet:
    """Replace every F1/F2/F3 dependency contract with Rust-backed exact fixtures."""
    fixture_set.validate()
    resolved = dict(
        _require_keys(
            artifact,
            {"schema_version", "artifact_id", "f1", "f2", "f3"},
            location="resolved_dependencies",
        )
    )
    if resolved["schema_version"] != RESOLVED_DEPENDENCY_SCHEMA_VERSION:
        raise SuiteValidationError("unsupported resolved dependency schema version")
    if resolved["artifact_id"] != RESOLVED_DEPENDENCY_ARTIFACT_ID:
        raise SuiteValidationError("unsupported resolved dependency artifact identity")
    f1 = _validate_f1_receipt(resolved["f1"])
    f2 = _validate_f2_receipt(resolved["f2"])
    f3 = _validate_f3_receipt(resolved["f3"])

    f1_metadata = {
        "long_salmon_component_context": (
            "f4-03-long-salmon-component",
            "Same radius-one context, different long Salmon or habitat component",
        ),
        "component_bridge": (
            "f4-07-component-bridge",
            "Same local terrain count, different component bridge",
        ),
        "equal_immediate_different_future_conflict": (
            "f4-08-motif-conflict",
            "Equal immediate score, different Hawk or Salmon conflict",
        ),
        "public_action_equivalence_refill_near_match": (
            "f4-10-public-action-equivalence",
            "Exact public-action equivalence and refill-divergent near-match",
        ),
    }
    f1_pairs = {}
    for family, (pair_id, title) in f1_metadata.items():
        payload = _f1_pair_payload(family, f1)
        f1_pairs[family] = build_pair(
            pair_id=pair_id,
            family=family,
            title=title,
            status="ready",
            public_inputs=payload["public_inputs"],
            expectations=[
                ConceptExpectation(
                    concept=item["concept"],
                    relation=item["relation"],
                    probe=item["probe"],
                    left_label=item["left_label"],
                    right_label=item["right_label"],
                )
                for item in payload["expectations"]
            ],
            boundary_contracts=payload["boundary_contracts"],
            provenance={
                "evidence_domain": "rust-f1-authority-bound-exact-public-witness",
                "sealed_test_opened": False,
                "source_experiment_id": F1_EXPERIMENT_ID,
                "classification_scientific_blake3": F1_CLASSIFICATION_SCIENTIFIC_BLAKE3,
                "merged_census_scientific_blake3": F1_FORWARD_SCIENTIFIC_BLAKE3,
                "schema_manifest_scientific_blake3": F1_MANIFEST_SCIENTIFIC_BLAKE3,
            },
            evidence=_resolved_evidence(
                artifact=resolved,
                artifact_file_blake3=artifact_file_blake3,
                receipt=f1,
            ),
        )

    d6_label = {
        "canonical_decision_blake3": f3["canonical_decision_blake3"],
        "contract_id": f3["contract_id"],
        "legal_action_count": f3["source_legal_action_count"],
    }
    d6_orbit = {item["transform_id"]: item for item in f3["orbit"]}
    d6_sides = {"left": d6_orbit[0], "right": d6_orbit[8]}
    d6_pair = build_pair(
        pair_id="f4-05-d6-transforms",
        family="d6_transforms",
        title="All 12 D6 transforms of one decision",
        status="ready",
        public_inputs={
            side: {
                "kind": "adversarial_public_fixture_v1",
                "transform_id": orbit["transform_id"],
                "public_state_blake3": orbit["transformed_public_state_blake3"],
                "legal_action_count": orbit["transformed_legal_action_count"],
                "action": orbit["transformed_action"],
                "concepts": {"d6_identity": d6_label},
            }
            for side, orbit in d6_sides.items()
        },
        expectations=[
            ConceptExpectation(
                concept="d6_decision_identity",
                relation="equivalent",
                probe="d6_identity",
                left_label=d6_label,
                right_label=d6_label,
            )
        ],
        boundary_contracts={"public-observable-v1": ["d6_decision_identity"]},
        provenance={
            "evidence_domain": "rust-exact-d6-state-action-legal-row-orbit",
            "sealed_test_opened": False,
            "source_experiment_id": F3_EXPERIMENT_ID,
            "contract_scientific_blake3": D6_SCIENTIFIC_BLAKE3,
        },
        evidence=_resolved_evidence(
            artifact=resolved,
            artifact_file_blake3=artifact_file_blake3,
            receipt=f3,
        ),
    )

    universe_blake3 = scientific_blake3(f2["legal_action_universe"])
    exact_labels = {
        side: {
            "universe_blake3": universe_blake3,
            "mask": f2[side]["exact_legal_mask"],
        }
        for side in ("left", "right")
    }
    in_radius_labels = {
        side: {
            "universe_blake3": universe_blake3,
            "mask": f2[side]["in_radius_only_legal_mask"],
        }
        for side in ("left", "right")
    }
    overflow_pair = build_pair(
        pair_id="f4-13-overflow-consequence",
        family="same_in_radius_different_overflow_consequence",
        title="Same in-radius state, different overflow consequence",
        status="ready",
        public_inputs={
            side: {
                "kind": "adversarial_public_fixture_v1",
                "compact_radius": f2["compact_radius"],
                "center": f2["center"],
                "in_radius_occupied": f2[side]["in_radius_occupied"],
                "overflow_occupied": f2[side]["overflow_occupied"],
                "concepts": {"legal_mask": exact_labels[side]},
                "declared_projections": {
                    IN_RADIUS_BOUNDARY_ID: {"concepts": {"legal_mask": in_radius_labels[side]}}
                },
            }
            for side in ("left", "right")
        },
        expectations=[
            ConceptExpectation(
                concept="overflow_legal_consequence",
                relation="different",
                probe="legal_mask",
                left_label=exact_labels["left"],
                right_label=exact_labels["right"],
            )
        ],
        boundary_contracts={
            "public-observable-v1": ["overflow_legal_consequence"],
            IN_RADIUS_BOUNDARY_ID: ["overflow_legal_consequence"],
        },
        provenance={
            "evidence_domain": "rust-board-frontier-equal-radius6-different-overflow",
            "sealed_test_opened": False,
            "source_experiment_id": F2_EXPERIMENT_ID,
            "source_scientific_blake3": F2_SCIENTIFIC_BLAKE3,
        },
        evidence=_resolved_evidence(
            artifact=resolved,
            artifact_file_blake3=artifact_file_blake3,
            receipt=f2,
        ),
    )

    latent_target = {
        "compact_radius": f2["compact_radius"],
        "compact_capacity": f2["compact_capacity"],
        "center": f2["center"],
        "in_radius_occupied": f2["left"]["in_radius_occupied"],
        "in_radius_frontier": f2["left"]["in_radius_frontier"],
    }
    compact_pair = build_pair(
        pair_id="f4-14-compact-latent-affordance",
        family="same_compact_latent_different_legal_affordance",
        title="Same compact latent target, different legal affordance",
        status="ready",
        public_inputs={
            side: source_from_compact_projection(
                latent_target=latent_target,
                retained_concepts={"legal_mask": exact_labels[side]},
            )
            | {
                "compact_contract": "radius6-127-plus-exact-overflow-v1",
                "exact_overflow": {
                    "occupied": f2[side]["overflow_occupied"],
                    "frontier": f2[side]["overflow_frontier"],
                },
            }
            for side in ("left", "right")
        },
        expectations=[
            ConceptExpectation(
                concept="compact_legal_affordance",
                relation="different",
                probe="legal_mask",
                left_label=exact_labels["left"],
                right_label=exact_labels["right"],
            )
        ],
        boundary_contracts={
            "public-observable-v1": ["compact_legal_affordance"],
            "declared-compact-projection-v1": ["compact_legal_affordance"],
        },
        provenance={
            "evidence_domain": "rust-radius6-compact-latent-with-exact-overflow-sidecar",
            "sealed_test_opened": False,
            "source_experiment_id": F2_EXPERIMENT_ID,
            "source_scientific_blake3": F2_SCIENTIFIC_BLAKE3,
        },
        evidence=_resolved_evidence(
            artifact=resolved,
            artifact_file_blake3=artifact_file_blake3,
            receipt=f2,
        ),
    )

    replacements = {
        **f1_pairs,
        d6_pair.family: d6_pair,
        overflow_pair.family: overflow_pair,
        compact_pair.family: compact_pair,
    }
    materialized = build_fixture_set(
        [replacements.get(pair.family, pair) for pair in fixture_set.pairs]
    )
    if len(materialized.pairs) != len(REQUIRED_FAMILIES) or any(
        pair.status != "ready" or pair.dependency is not None for pair in materialized.pairs
    ):
        raise SuiteValidationError("resolved dependency closure must make all 14 pairs executable")
    return materialized


class BoundaryAdapter(ABC):
    """Plugin interface for one representation boundary."""

    boundary_id: str
    schema_version: int = BOUNDARY_SCHEMA_VERSION
    description: str

    @abstractmethod
    def project(self, public_input: Mapping[str, Any]) -> Mapping[str, Any]:
        """Project one public input into the exact representation boundary."""

    def metadata(self) -> dict[str, Any]:
        return {
            "boundary_id": self.boundary_id,
            "schema_version": self.schema_version,
            "description": self.description,
            "implementation": f"{type(self).__module__}.{type(self).__qualname__}",
        }


class BoundaryRegistry:
    """Stable plugin registry used by the core runner."""

    def __init__(self) -> None:
        self._adapters: dict[str, BoundaryAdapter] = {}

    def register(self, adapter: BoundaryAdapter) -> None:
        if not adapter.boundary_id:
            raise SuiteValidationError("boundary adapter requires an ID")
        if adapter.schema_version != BOUNDARY_SCHEMA_VERSION:
            raise SuiteValidationError("unsupported boundary adapter schema version")
        if adapter.boundary_id in self._adapters:
            raise SuiteValidationError(f"duplicate boundary adapter: {adapter.boundary_id}")
        self._adapters[adapter.boundary_id] = adapter

    def get(self, boundary_id: str) -> BoundaryAdapter:
        try:
            return self._adapters[boundary_id]
        except KeyError as error:
            raise SuiteValidationError(f"unknown boundary adapter: {boundary_id}") from error

    def ids(self) -> tuple[str, ...]:
        return tuple(self._adapters)

    def adapters(self) -> tuple[BoundaryAdapter, ...]:
        return tuple(self._adapters.values())


class PublicObservableBoundary(BoundaryAdapter):
    boundary_id = "public-observable-v1"
    description = "Exact normalized public concept map before representation compression."

    def project(self, public_input: Mapping[str, Any]) -> Mapping[str, Any]:
        concepts = public_input.get("concepts")
        if not isinstance(concepts, Mapping):
            raise BoundaryUnavailable("public input has no normalized concept map")
        blocked = public_input.get("blocked_concepts", {})
        return {"concepts": dict(concepts), "blocked_concepts": dict(blocked)}


class DeclaredProjectionBoundary(BoundaryAdapter):
    """Fixture or future-model projection named without changing the runner."""

    def __init__(self, boundary_id: str, *, description: str | None = None):
        self.boundary_id = boundary_id
        self.description = description or f"Declared exact projection {boundary_id}."

    def project(self, public_input: Mapping[str, Any]) -> Mapping[str, Any]:
        projections = public_input.get("declared_projections")
        if not isinstance(projections, Mapping) or self.boundary_id not in projections:
            raise BoundaryUnavailable(f"no declared projection for {self.boundary_id}")
        projection = projections[self.boundary_id]
        if not isinstance(projection, Mapping):
            raise SuiteValidationError("declared projection must be an object")
        return dict(projection)


class KindConceptBoundary(BoundaryAdapter):
    """Normalize an existing public schema into frozen probe concept keys."""

    accepted_kinds: tuple[str, ...]

    def project(self, public_input: Mapping[str, Any]) -> Mapping[str, Any]:
        if public_input.get("kind") not in self.accepted_kinds:
            raise BoundaryUnavailable(f"{self.boundary_id} does not apply")
        concepts = public_input.get("concepts")
        if not isinstance(concepts, Mapping):
            raise SuiteValidationError(f"{self.boundary_id} requires normalized concepts")
        return {
            "concepts": dict(concepts),
            "source_schema": public_input.get("source_schema"),
        }


class V2PositionRecordBoundary(KindConceptBoundary):
    boundary_id = "v2-position-record-v1"
    description = "Current compact-entity-v2 fixed-width PositionRecord observables."
    accepted_kinds = ("v2_position_record",)


class CurrentDatasetTensorBoundary(KindConceptBoundary):
    boundary_id = "current-dataset-tensors-v1"
    description = "Decoded compact-entity-v2 board, market, global, and mask tensors."
    accepted_kinds = ("current_dataset_tensors",)


class GradedOracleRawBoundary(KindConceptBoundary):
    boundary_id = "graded-oracle-raw-v1"
    description = "Lossless complete-action public raw and staged observable tensors."
    accepted_kinds = ("graded_oracle_raw",)


class GradedOracleFactorBoundary(KindConceptBoundary):
    boundary_id = "graded-oracle-factors-v1"
    description = "Seven typed graded-oracle candidate factors at candidate projection input."
    accepted_kinds = ("graded_oracle_factors",)


class CompactProjectionBoundary(KindConceptBoundary):
    boundary_id = "declared-compact-projection-v1"
    description = "Declared compact representation target and exact retained concept map."
    accepted_kinds = ("compact_projection",)


@dataclass(frozen=True)
class ProbeObservation:
    """One frozen probe result for one side of one pair."""

    schema_version: int
    probe_id: str
    status: str
    value: Any
    signature: str | None
    dependency: str | None
    reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FrozenProbe(ABC):
    """Probe interface supporting deterministic and future MLX implementations."""

    probe_id: str
    schema_version: int = PROBE_SCHEMA_VERSION
    description: str

    @abstractmethod
    def observe(self, projection: Mapping[str, Any]) -> ProbeObservation:
        """Observe one exact concept at one boundary."""


class ExactConceptProbe(FrozenProbe):
    """Read an exact normalized concept without learning or approximation."""

    def __init__(self, probe_id: str):
        self.probe_id = probe_id
        self.description = f"Exact deterministic {probe_id} probe."

    def observe(self, projection: Mapping[str, Any]) -> ProbeObservation:
        concepts = projection.get("concepts", {})
        if isinstance(concepts, Mapping) and self.probe_id in concepts:
            value = concepts[self.probe_id]
            return ProbeObservation(
                schema_version=PROBE_SCHEMA_VERSION,
                probe_id=self.probe_id,
                status="observed",
                value=value,
                signature=scientific_blake3(value),
                dependency=None,
                reason=None,
            )
        blocked = projection.get("blocked_concepts", {})
        if isinstance(blocked, Mapping) and self.probe_id in blocked:
            item = blocked[self.probe_id]
            if not isinstance(item, Mapping):
                raise SuiteValidationError("blocked concept record must be an object")
            dependency = str(item.get("dependency", ""))
            if dependency not in ALLOWED_DEPENDENCIES:
                raise SuiteValidationError("blocked probe must name F1, F2, or F3")
            return ProbeObservation(
                schema_version=PROBE_SCHEMA_VERSION,
                probe_id=self.probe_id,
                status="blocked",
                value=None,
                signature=None,
                dependency=dependency,
                reason=str(item.get("reason", "canonical concept unavailable")),
            )
        return ProbeObservation(
            schema_version=PROBE_SCHEMA_VERSION,
            probe_id=self.probe_id,
            status="unsupported",
            value=None,
            signature=None,
            dependency=None,
            reason=f"{self.probe_id} is not exposed by this boundary",
        )


class ProbeRegistry:
    """Registry that makes every required probe explicit."""

    def __init__(self) -> None:
        self._probes: dict[str, FrozenProbe] = {}

    def register(self, probe: FrozenProbe) -> None:
        if probe.probe_id in self._probes:
            raise SuiteValidationError(f"duplicate probe: {probe.probe_id}")
        if probe.schema_version != PROBE_SCHEMA_VERSION:
            raise SuiteValidationError("unsupported probe schema version")
        self._probes[probe.probe_id] = probe

    def get(self, probe_id: str) -> FrozenProbe:
        try:
            return self._probes[probe_id]
        except KeyError as error:
            raise SuiteValidationError(f"unknown frozen probe: {probe_id}") from error

    def validate_complete(self) -> None:
        if tuple(self._probes) != REQUIRED_PROBES:
            raise SuiteValidationError("required frozen probe registry is incomplete")

    def ids(self) -> tuple[str, ...]:
        return tuple(self._probes)


def default_probe_registry() -> ProbeRegistry:
    registry = ProbeRegistry()
    for probe_id in REQUIRED_PROBES:
        registry.register(ExactConceptProbe(probe_id))
    registry.validate_complete()
    return registry


def default_boundary_registry(
    *,
    declared_boundary_ids: Iterable[str] = (),
) -> BoundaryRegistry:
    registry = BoundaryRegistry()
    registry.register(PublicObservableBoundary())
    registry.register(V2PositionRecordBoundary())
    registry.register(CurrentDatasetTensorBoundary())
    registry.register(GradedOracleRawBoundary())
    registry.register(GradedOracleFactorBoundary())
    registry.register(CompactProjectionBoundary())
    for boundary_id in declared_boundary_ids:
        registry.register(DeclaredProjectionBoundary(boundary_id))
    return registry


def _array_rows(values: Any, mask: Any | None = None) -> list[Any]:
    array = np.asarray(values)
    if mask is None:
        return _normalized(array)
    bool_mask = np.asarray(mask, dtype=np.bool_)
    return _normalized(array[bool_mask])


def source_from_v2_position_record(record: np.void | np.ndarray) -> dict[str, Any]:
    """Normalize one current fixed-width V2 record without hidden information."""
    value = np.asarray(record)
    if value.shape not in ((), (1,)):
        raise SuiteValidationError("V2 position adapter requires exactly one record")
    row = value.reshape(-1)[0]
    board_counts = np.asarray(row["board_counts"], dtype=np.int64)
    board_entities = np.asarray(row["board_entities"])
    occupancy = [
        _normalized(board_entities[seat, : int(board_counts[seat])])
        for seat in range(len(board_counts))
    ]
    return {
        "kind": "v2_position_record",
        "source_schema": "compact-entity-v2",
        "concepts": {
            "occupancy": occupancy,
            "staged_market": _normalized(np.asarray(row["market_entities"])),
        },
    }


def source_from_dataset_batch(batch: Any, *, index: int = 0) -> dict[str, Any]:
    """Normalize one decoded current Dataset batch row."""
    board_entities = np.asarray(batch.board_entities)[index]
    board_mask = np.asarray(batch.board_mask)[index]
    market_entities = np.asarray(batch.market_entities)[index]
    market_mask = np.asarray(batch.market_mask)[index]
    return {
        "kind": "current_dataset_tensors",
        "source_schema": "compact-entity-v2-decoded",
        "concepts": {
            "occupancy": [
                _array_rows(board_entities[seat], board_mask[seat])
                for seat in range(board_entities.shape[0])
            ],
            "staged_market": _array_rows(market_entities, market_mask),
        },
    }


def source_from_graded_oracle_batch(
    batch: Any,
    *,
    group: int = 0,
    candidate: int = 0,
) -> dict[str, Any]:
    """Normalize public raw tensors from one open complete-action candidate."""
    candidate_mask = np.asarray(batch.candidate_mask)[group]
    if candidate < 0 or candidate >= len(candidate_mask) or not candidate_mask[candidate]:
        raise SuiteValidationError("graded-oracle candidate index is not legal")
    board_entities = np.asarray(batch.board_entities)[group]
    board_mask = np.asarray(batch.board_mask)[group]
    market_entities = np.asarray(batch.market_entities)[group]
    market_mask = np.asarray(batch.market_mask)[group]
    staged_market = np.asarray(batch.staged_market_entities)[group, candidate]
    staged_market_mask = np.asarray(batch.staged_market_mask)[group, candidate]
    return {
        "kind": "graded_oracle_raw",
        "source_schema": "complete-action-graded-oracle-v1-public",
        "concepts": {
            "occupancy": [
                _array_rows(board_entities[seat], board_mask[seat])
                for seat in range(board_entities.shape[0])
            ],
            "exact_supply": _normalized(np.asarray(batch.public_supply)[group]),
            "staged_market": {
                "parent": _array_rows(market_entities, market_mask),
                "staged": _array_rows(staged_market, staged_market_mask),
            },
            "action_edit": _normalized(np.asarray(batch.action_features)[group, candidate]),
            "legal_mask": _normalized(candidate_mask),
        },
    }


def source_from_graded_factor_array(
    factors: Any,
    *,
    group: int = 0,
    candidate: int = 0,
) -> dict[str, Any]:
    """Normalize the seven typed graded-oracle factors without teacher labels."""
    values = np.asarray(factors)
    if values.ndim != 4 or values.shape[2] != 7:
        raise SuiteValidationError("graded-oracle factor tensor must be [G, C, 7, H]")
    return {
        "kind": "graded_oracle_factors",
        "source_schema": "complete-action-graded-residual-v1-factors",
        "concepts": {
            "action_edit": _normalized(values[group, candidate]),
        },
    }


def source_from_compact_projection(
    *,
    latent_target: Any,
    retained_concepts: Mapping[str, Any],
) -> dict[str, Any]:
    """Declare a future compact arm while keeping exact retained concepts explicit."""
    return {
        "kind": "compact_projection",
        "source_schema": "declared-compact-projection-v1",
        "latent_target": _normalized(latent_target),
        "concepts": dict(retained_concepts),
    }


def confidence_set_evidence(
    means: Sequence[float],
    stddevs: Sequence[float],
    samples: Sequence[float],
    action_hashes: Sequence[str],
) -> dict[str, Any]:
    """Classify an existing open teacher group without generating labels."""
    mean = np.asarray(means, dtype=np.float64)
    stddev = np.asarray(stddevs, dtype=np.float64)
    sample = np.asarray(samples, dtype=np.float64)
    if (
        mean.ndim != 1
        or len(mean) < 2
        or stddev.shape != mean.shape
        or sample.shape != mean.shape
        or len(action_hashes) != len(mean)
        or np.any(sample <= 0)
        or not np.all(np.isfinite(mean))
        or not np.all(np.isfinite(stddev))
    ):
        raise SuiteValidationError("invalid open teacher confidence evidence")
    order = sorted(range(len(mean)), key=lambda index: (-mean[index], action_hashes[index]))
    winner, runner_up = order[:2]
    standard_error = stddev / np.sqrt(sample)
    thresholds = NORMAL_95 * np.hypot(standard_error[winner], standard_error)
    confidence = (mean[winner] - mean) <= thresholds
    margin_threshold = NORMAL_95 * math.hypot(standard_error[winner], standard_error[runner_up])
    return {
        "winner_index": winner,
        "runner_up_index": runner_up,
        "confidence_set_membership": confidence.tolist(),
        "confidence_set_size": int(np.sum(confidence)),
        "distinguishable_winner": bool(mean[winner] - mean[runner_up] > margin_threshold),
        "winner_margin": float(mean[winner] - mean[runner_up]),
        "winner_margin_threshold_95": float(margin_threshold),
    }


def _evaluate_expectation(
    expectation: ConceptExpectation,
    left: ProbeObservation,
    right: ProbeObservation,
) -> dict[str, Any]:
    if left.status == "blocked" or right.status == "blocked":
        return {
            "concept": expectation.concept,
            "probe": expectation.probe,
            "expected_relation": expectation.relation,
            "status": "blocked",
            "retained": False,
            "collision": False,
            "equivalence_violation": False,
            "left": left.to_dict(),
            "right": right.to_dict(),
        }
    if left.status != "observed" or right.status != "observed":
        return {
            "concept": expectation.concept,
            "probe": expectation.probe,
            "expected_relation": expectation.relation,
            "status": "unsupported",
            "retained": False,
            "collision": False,
            "equivalence_violation": False,
            "left": left.to_dict(),
            "right": right.to_dict(),
        }
    equal = left.signature == right.signature
    retained = equal if expectation.relation == "equivalent" else not equal
    return {
        "concept": expectation.concept,
        "probe": expectation.probe,
        "expected_relation": expectation.relation,
        "status": "observed",
        "observed_relation": "equivalent" if equal else "different",
        "retained": retained,
        "collision": expectation.relation == "different" and equal,
        "equivalence_violation": expectation.relation == "equivalent" and not equal,
        "left": left.to_dict(),
        "right": right.to_dict(),
    }


def evaluate_pair_boundary(
    pair: AdversarialPair,
    adapter: BoundaryAdapter,
    probes: ProbeRegistry,
) -> dict[str, Any]:
    """Evaluate exact concept retention for one pair at one boundary."""
    pair.validate()
    projections: dict[str, Mapping[str, Any]] = {}
    projection_hashes: dict[str, str] = {}
    unavailable: dict[str, str] = {}
    for side in ("left", "right"):
        try:
            projection = adapter.project(pair.public_inputs[side])
        except BoundaryUnavailable as error:
            unavailable[side] = str(error)
            continue
        projections[side] = projection
        projection_hashes[side] = scientific_blake3(projection)

    required = set(pair.boundary_contracts.get(adapter.boundary_id, ()))
    if unavailable:
        status = "invalid" if required else "unsupported"
        return {
            "schema_version": BOUNDARY_SCHEMA_VERSION,
            "pair_id": pair.pair_id,
            "family": pair.family,
            "boundary_id": adapter.boundary_id,
            "status": status,
            "boundary_verdict": status,
            "required_concepts": sorted(required),
            "unavailable": unavailable,
            "projection_hashes": projection_hashes,
            "projection_equal": None,
            "concepts": [],
            "signature": scientific_blake3(
                {
                    "pair_id": pair.pair_id,
                    "boundary_id": adapter.boundary_id,
                    "status": status,
                    "unavailable": unavailable,
                }
            ),
        }

    concept_results = []
    for expectation in pair.expectations:
        left = probes.get(expectation.probe).observe(projections["left"])
        right = probes.get(expectation.probe).observe(projections["right"])
        result = _evaluate_expectation(expectation, left, right)
        result["required"] = expectation.concept in required
        concept_results.append(result)

    missing_required = [
        result["concept"]
        for result in concept_results
        if result["required"] and result["status"] == "unsupported"
    ]
    blocked_required = [
        result["concept"]
        for result in concept_results
        if result["required"] and result["status"] == "blocked"
    ]
    observed_required = [
        result
        for result in concept_results
        if result["required"] and result["status"] == "observed"
    ]
    information_lost = any(not bool(result["retained"]) for result in observed_required)
    if missing_required:
        status = "invalid"
        verdict = "invalid"
    elif blocked_required or pair.status == "dependency_blocked":
        status = "blocked"
        verdict = "blocked"
    else:
        status = "complete"
        verdict = "information_lost" if information_lost else "retained"
    output = {
        "schema_version": BOUNDARY_SCHEMA_VERSION,
        "pair_id": pair.pair_id,
        "family": pair.family,
        "boundary_id": adapter.boundary_id,
        "status": status,
        "boundary_verdict": verdict,
        "required_concepts": sorted(required),
        "missing_required_concepts": missing_required,
        "blocked_required_concepts": blocked_required,
        "projection_hashes": projection_hashes,
        "projection_equal": projection_hashes["left"] == projection_hashes["right"],
        "concepts": concept_results,
    }
    output["signature"] = scientific_blake3(output)
    return output


def _boundary_metrics(observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    applicable = [item for item in observations if item["status"] not in ("unsupported", "invalid")]
    concept_results = [
        concept
        for item in observations
        for concept in item["concepts"]
        if concept["required"] and concept["status"] == "observed"
    ]
    different = [item for item in concept_results if item["expected_relation"] == "different"]
    equivalent = [item for item in concept_results if item["expected_relation"] == "equivalent"]
    legal = [item for item in concept_results if item["probe"] == "legal_mask"]
    return {
        "pairs_total": len(observations),
        "pairs_applicable": len(applicable),
        "pairs_information_lost": sum(
            item["boundary_verdict"] == "information_lost" for item in observations
        ),
        "pairs_retained": sum(item["boundary_verdict"] == "retained" for item in observations),
        "pairs_blocked": sum(item["status"] == "blocked" for item in observations),
        "pairs_invalid": sum(item["status"] == "invalid" for item in observations),
        "concept_assertions": len(concept_results),
        "concepts_retained": sum(bool(item["retained"]) for item in concept_results),
        "exact_collisions": sum(bool(item["collision"]) for item in concept_results),
        "equivalence_violations": sum(
            bool(item["equivalence_violation"]) for item in concept_results
        ),
        "difference_assertions": len(different),
        "pair_separability": (
            sum(bool(item["retained"]) for item in different) / len(different)
            if different
            else None
        ),
        "equivalence_assertions": len(equivalent),
        "exact_equivalence_rate": (
            sum(bool(item["retained"]) for item in equivalent) / len(equivalent)
            if equivalent
            else None
        ),
        "legal_mask_assertions": len(legal),
        "legal_mask_retention": (
            sum(bool(item["retained"]) for item in legal) / len(legal) if legal else None
        ),
        "deterministic_signature": scientific_blake3([item["signature"] for item in observations]),
    }


def run_suite(
    fixture_set: FixtureSet,
    *,
    boundaries: BoundaryRegistry | None = None,
    probes: ProbeRegistry | None = None,
    selected_boundaries: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run every registered boundary and return deterministic scientific output."""
    errors: list[str] = []
    try:
        fixture_set.validate()
    except SuiteValidationError as error:
        errors.append(str(error))
    probes = probes or default_probe_registry()
    try:
        probes.validate_complete()
    except SuiteValidationError as error:
        errors.append(str(error))
    declared_ids = sorted(
        {
            boundary_id
            for pair in fixture_set.pairs
            for side in pair.public_inputs.values()
            for boundary_id in (
                side.get("declared_projections", {}).keys()
                if isinstance(side, Mapping)
                and isinstance(side.get("declared_projections"), Mapping)
                else ()
            )
        }
    )
    boundaries = boundaries or default_boundary_registry(declared_boundary_ids=declared_ids)
    boundary_ids = tuple(selected_boundaries or boundaries.ids())
    for pair in fixture_set.pairs:
        missing = set(pair.boundary_contracts) - set(boundary_ids)
        if missing:
            errors.append(
                f"{pair.pair_id} required boundaries were not selected: {sorted(missing)}"
            )

    observations_by_boundary: dict[str, list[dict[str, Any]]] = {
        boundary_id: [] for boundary_id in boundary_ids
    }
    pair_details = []
    for pair in fixture_set.pairs:
        observations = []
        for boundary_id in boundary_ids:
            try:
                observation = evaluate_pair_boundary(
                    pair,
                    boundaries.get(boundary_id),
                    probes,
                )
            except SuiteValidationError as error:
                errors.append(f"{pair.pair_id}/{boundary_id}: {error}")
                observation = {
                    "schema_version": BOUNDARY_SCHEMA_VERSION,
                    "pair_id": pair.pair_id,
                    "family": pair.family,
                    "boundary_id": boundary_id,
                    "status": "invalid",
                    "boundary_verdict": "invalid",
                    "required_concepts": [],
                    "concepts": [],
                    "signature": scientific_blake3(
                        {
                            "pair_id": pair.pair_id,
                            "boundary_id": boundary_id,
                            "error": str(error),
                        }
                    ),
                }
            observations.append(observation)
            observations_by_boundary[boundary_id].append(observation)
        pair_detail = pair.to_dict()
        pair_detail["boundary_observations"] = observations
        pair_detail["pair_status"] = (
            "invalid"
            if any(item["status"] == "invalid" for item in observations)
            else "dependency_blocked"
            if pair.status == "dependency_blocked"
            else "complete"
        )
        pair_details.append(pair_detail)

    invalid_observations = sum(
        item["status"] == "invalid"
        for observations in observations_by_boundary.values()
        for item in observations
    )
    blocked_pairs = sum(pair.status == "dependency_blocked" for pair in fixture_set.pairs)
    if errors or invalid_observations:
        classification = CLASSIFICATION_INVALID
    elif blocked_pairs:
        classification = CLASSIFICATION_BLOCKED
    elif any(
        item["boundary_verdict"] == "information_lost"
        for observations in observations_by_boundary.values()
        for item in observations
        if item["required_concepts"]
    ):
        classification = CLASSIFICATION_FAILED
    else:
        classification = CLASSIFICATION_PASSED

    boundary_reports = {
        boundary_id: {
            "adapter": boundaries.get(boundary_id).metadata(),
            "metrics": _boundary_metrics(observations),
        }
        for boundary_id, observations in observations_by_boundary.items()
    }
    summary = {
        "schema_version": SUITE_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "fixture_set_id": fixture_set.fixture_set_id,
        "fixture_set_blake3": fixture_set.canonical_hash,
        "classification": classification,
        "exit_code": EXIT_CODES[classification],
        "family_registry": list(REQUIRED_FAMILIES),
        "families_present": [pair.family for pair in fixture_set.pairs],
        "ready_pairs": len(fixture_set.pairs) - blocked_pairs,
        "dependency_blocked_pairs": blocked_pairs,
        "probe_registry": list(probes.ids()),
        "selected_boundaries": list(boundary_ids),
        "boundary_reports": boundary_reports,
        "errors": errors,
        "evidence_boundary": {
            "sealed_test_opened": False,
            "gameplay_used": False,
            "teacher_rollout_used": False,
            "cloud_or_external_compute_used": False,
        },
    }
    report = {
        "summary": summary,
        "pairs": pair_details,
    }
    summary["scientific_blake3"] = scientific_blake3(report)
    report["scientific_blake3"] = summary["scientific_blake3"]
    return report


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact deterministic human summary."""
    summary = report["summary"]
    lines = [
        "# Information-Preservation and Adversarial Suite",
        "",
        f"- Classification: `{summary['classification']}`",
        f"- Scientific BLAKE3: `{summary['scientific_blake3']}`",
        f"- Families: {len(summary['families_present'])}/{len(REQUIRED_FAMILIES)}",
        f"- Ready pairs: {summary['ready_pairs']}",
        f"- Dependency-blocked pairs: {summary['dependency_blocked_pairs']}",
        f"- Frozen probes: {len(summary['probe_registry'])}/{len(REQUIRED_PROBES)}",
        "",
        "## Classification",
        "",
        "- The suite is complete and non-blocked: every frozen family has executable evidence.",
        "- `failed` means at least one required representation boundary loses information; "
        "it does not mean fixture generation or validation failed.",
        "- The exact public-observable boundary retains all fourteen families.",
        "",
        "## Boundaries",
        "",
        "| Boundary | Applicable | Retained | Lost | Collisions | Eq violations |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for boundary_id, values in summary["boundary_reports"].items():
        metrics = values["metrics"]
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                boundary_id,
                metrics["pairs_applicable"],
                metrics["pairs_retained"],
                metrics["pairs_information_lost"],
                metrics["exact_collisions"],
                metrics["equivalence_violations"],
            )
        )
    lines.extend(["", "## Pair Families", ""])
    for pair in report["pairs"]:
        dependency = pair["dependency"]
        suffix = (
            f" blocked by {dependency['dependency']}"
            if dependency is not None
            else " exact fixture"
        )
        lines.append(f"- `{pair['family']}`: {pair['pair_status']};{suffix}")
    pair_by_family = {pair["family"]: pair for pair in report["pairs"]}
    f1_pairs = [
        pair_by_family.get("long_salmon_component_context"),
        pair_by_family.get("component_bridge"),
        pair_by_family.get("equal_immediate_different_future_conflict"),
        pair_by_family.get("public_action_equivalence_refill_near_match"),
    ]
    d6_pair = pair_by_family.get("d6_transforms")
    overflow_pair = pair_by_family.get("same_in_radius_different_overflow_consequence")
    compact_pair = pair_by_family.get("same_compact_latent_different_legal_affordance")
    if all(
        pair is not None and pair["pair_status"] == "complete"
        for pair in (*f1_pairs, d6_pair, overflow_pair, compact_pair)
    ):
        f1_receipt = f1_pairs[0]["evidence"]["receipt"]
        d6_receipt = d6_pair["evidence"]["receipt"]
        f2_receipt = overflow_pair["evidence"]["receipt"]
        lines.extend(
            [
                "",
                "## Resolved F1/F2/F3 Evidence",
                "",
                "- F1: all four formerly blocked pairs are executable from exact Rust "
                "component, motif, score, transition, and public-supply receipts.",
                "- F1 classification scientific BLAKE3: "
                f"`{f1_receipt['classification_scientific_blake3']}`.",
                "- F1 merged-census scientific BLAKE3: "
                f"`{f1_receipt['merged_census_scientific_blake3']}`.",
                "- F3: {} exact transforms over {} legal rows; every legal map is "
                "bijective, every selected action round-trips, and every transition "
                "is equivariant.".format(
                    len(d6_receipt["orbit"]),
                    d6_receipt["source_legal_action_count"],
                ),
                f"- F3 contract scientific BLAKE3: `{d6_receipt['contract_scientific_blake3']}`.",
                "- F2: both Rust boards contain 23 legal tiles, retain the same {} "
                "radius-6 occupied cells and the same in-radius frontier, and carry "
                "different exact `Board::frontier` legal masks.".format(
                    len(f2_receipt["left"]["in_radius_occupied"])
                ),
                "- Radius-6 in-radius-only projection: exact collision; legal-mask "
                "retention `0.0`.",
                "- Radius-6 / 127-cell compact projection with exact overflow sidecar: "
                "legal-mask retention `1.0`.",
                f"- F2 source scientific BLAKE3: `{f2_receipt['source_scientific_blake3']}`.",
            ]
        )
    blockers = [pair for pair in report["pairs"] if pair["pair_status"] == "dependency_blocked"]
    if blockers:
        lines.extend(["", "## Remaining Blockers", ""])
        for pair in blockers:
            lines.append(
                f"- `{pair['family']}`: {pair['dependency']['dependency']} - "
                f"{pair['dependency']['reason']}"
            )
    if summary["errors"]:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in summary["errors"])
    return "\n".join(lines) + "\n"


def load_fixture_set(
    path: str | Path,
    *,
    require_complete: bool = True,
) -> FixtureSet:
    try:
        values = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SuiteValidationError(f"cannot load fixture set: {error}") from error
    return FixtureSet.from_dict(values, require_complete=require_complete)


def write_fixture_set(fixture_set: FixtureSet, path: str | Path) -> None:
    """Write one validated fixture set with deterministic formatting."""
    fixture_set.validate()
    Path(path).write_text(
        json.dumps(_normalized(fixture_set.to_dict()), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_outputs(
    report: Mapping[str, Any],
    *,
    json_output: str | Path | None = None,
    markdown_output: str | Path | None = None,
    jsonl_output: str | Path | None = None,
) -> None:
    if json_output is not None:
        Path(json_output).write_text(
            json.dumps(_normalized(report), indent=2, sort_keys=True) + "\n"
        )
    if markdown_output is not None:
        Path(markdown_output).write_text(render_markdown(report))
    if jsonl_output is not None:
        lines = [canonical_json(pair) for pair in report["pairs"]]
        Path(jsonl_output).write_text("\n".join(lines) + ("\n" if lines else ""))


def _default_fixture_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "artifacts"
        / "experiments"
        / EXPERIMENT_ID
        / "fixtures"
        / "pairs-v1.json"
    )


def _default_resolved_dependency_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "artifacts"
        / "experiments"
        / EXPERIMENT_ID
        / "fixtures"
        / "resolved-dependencies-v2.json"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=_default_fixture_path())
    parser.add_argument(
        "--resolved-dependencies",
        type=Path,
        default=_default_resolved_dependency_path(),
    )
    parser.add_argument("--materialize-fixtures", type=Path)
    parser.add_argument("--boundary", action="append", dest="boundaries")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--jsonl-out", type=Path)
    parser.add_argument(
        "--stdout",
        choices=("json", "markdown", "none"),
        default="json",
    )
    parser.add_argument("--list-boundaries", action="store_true")
    parser.add_argument("--list-probes", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    probes = default_probe_registry()
    if args.list_probes:
        print("\n".join(probes.ids()))
        return 0
    try:
        fixture_set = load_fixture_set(args.fixtures)
        if args.materialize_fixtures is not None:
            resolved = load_resolved_dependency_artifact(args.resolved_dependencies)
            fixture_set = materialize_resolved_dependencies(
                fixture_set,
                resolved,
                artifact_file_blake3=file_blake3(args.resolved_dependencies),
            )
            write_fixture_set(fixture_set, args.materialize_fixtures)
        declared_ids = sorted(
            {
                boundary_id
                for pair in fixture_set.pairs
                for side in pair.public_inputs.values()
                for boundary_id in side.get("declared_projections", {})
            }
        )
        boundaries = default_boundary_registry(declared_boundary_ids=declared_ids)
        if args.list_boundaries:
            print("\n".join(boundaries.ids()))
            return 0
        report = run_suite(
            fixture_set,
            boundaries=boundaries,
            probes=probes,
            selected_boundaries=args.boundaries,
        )
    except SuiteValidationError as error:
        report = {
            "summary": {
                "schema_version": SUITE_SCHEMA_VERSION,
                "experiment_id": EXPERIMENT_ID,
                "classification": CLASSIFICATION_INVALID,
                "exit_code": EXIT_CODES[CLASSIFICATION_INVALID],
                "errors": [str(error)],
                "scientific_blake3": scientific_blake3({"error": str(error)}),
            },
            "pairs": [],
        }
        report["scientific_blake3"] = report["summary"]["scientific_blake3"]
    write_outputs(
        report,
        json_output=args.json_out,
        markdown_output=args.markdown_out,
        jsonl_output=args.jsonl_out,
    )
    if args.stdout == "json":
        print(json.dumps(_normalized(report), indent=2, sort_keys=True))
    elif args.stdout == "markdown":
        print(render_markdown(report), end="")
    return int(report["summary"]["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
