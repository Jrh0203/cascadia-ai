#!/usr/bin/env python3
"""Deterministic feature manifests and streaming activation/collision census.

The tool intentionally has no MLX dependency. It reads the repository's
checksum-bound binary/NumPy formats directly so million-candidate censuses stay
streaming, reproducible, and usable on machines that are not training models.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import struct
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import blake3
import numpy as np

EXPERIMENT_ID = "feature-schema-activation-census-v1"
MANIFEST_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1

PHASES = ("opening", "early", "middle", "late")
PHASE_LABELS = (*PHASES, "unknown")
SEATS = ("0", "1", "2", "3", "unknown")
DEFAULT_RARE_THRESHOLD = 1e-4
DEFAULT_BATCH_ROWS = 8_192
DEFAULT_COLLISION_SAMPLE_MODULUS = 64
DEFAULT_COLLISION_SIGNATURE_LIMIT = 100_000
DEFAULT_EXACT_ALIAS_CELL_LIMIT = 2_000_000

POSITION_FEATURE_SCHEMA = "compact-entity-v2"
POSITION_TARGET_SCHEMA = "base-score-components-v1"
POSITION_MAGIC = b"CSD2REC\0"
POSITION_HEADER_SIZE = 80
POSITION_RECORD_SIZE = 864
POSITION_MAX_BOARD_TILES = 23

GRADED_FEATURE_SCHEMA = "complete-action-graded-oracle-v1"
GRADED_TARGET_SCHEMA = "screen-r600-r1200-r4800-graded-v1"
GRADED_MAGIC = b"CSD2GOV\0"
GRADED_HEADER_SIZE = 112
GRADED_GROUP_HEADER_SIZE = 960
GRADED_CANDIDATE_SIZE = 224
GRADED_ACTION_STORAGE_SIZE = 128
GRADED_ACTION_DIM = 140
GRADED_PUBLIC_SUPPLY_SIZE = 30
GRADED_MAX_WIPES = 20

FACTOR_CACHE_SCHEMA = "graded-oracle-frozen-candidate-factors-v1"
FACTOR_NAMES = (
    "action",
    "prior",
    "parent",
    "staged",
    "board_cross",
    "staged_cross",
    "action_parent_product",
)
FACTOR_DIM = 192

HIERARCHICAL_CACHE_SCHEMA = "hierarchical-factor-retrieval-cache-v1"
HIERARCHICAL_EXPERIMENT_ID = "full-legal-hierarchical-factor-retrieval-pilot-v1"
SCHEMA_VERSIONS = {
    "legacy-nnue-v1-5197": 1,
    "legacy-mid-v4opp-11231": 1,
    POSITION_FEATURE_SCHEMA: 2,
    GRADED_FEATURE_SCHEMA: 1,
    FACTOR_CACHE_SCHEMA: 1,
    HIERARCHICAL_CACHE_SCHEMA: 1,
    "legacy-mid-v4-fixed-v1": 1,
    "relational-opportunity-graph-v0": 0,
}

_POSITION_HEADER = struct.Struct("<8sHHHHIIQBBBB5s7s32s")
_GRADED_HEADER = struct.Struct("<8sHHHHIIIBBBBQ32s32s8s")

_POSITION_DTYPE = np.dtype(
    {
        "names": [
            "game_index",
            "turn",
            "active_seat",
            "player_count",
            "total_turns",
            "board_counts",
            "nature_tokens",
            "scoring_cards",
            "habitat_bonuses",
            "wildlife_counts",
            "habitat_sizes",
            "board_entities",
            "market_entities",
            "targets",
        ],
        "formats": [
            "<u8",
            "u1",
            "u1",
            "u1",
            "u1",
            ("u1", (4,)),
            ("u1", (4,)),
            ("u1", (5,)),
            "u1",
            ("u1", (4, 5)),
            ("u1", (4, 5)),
            ("u1", (4, 23, 8)),
            ("u1", (4, 8)),
            ("<u2", (11,)),
        ],
        "offsets": [0, 8, 9, 10, 11, 12, 16, 20, 25, 32, 52, 72, 808, 840],
        "itemsize": POSITION_RECORD_SIZE,
    }
)

_GRADED_GROUP_DTYPE = np.dtype(
    {
        "names": [
            "group_id",
            "raw_seed",
            "candidate_count",
            "selected_index",
            "champion_index",
            "completed_turns",
            "current_player",
            "personal_turn",
            "phase",
            "public_state_hash",
            "position",
            "public_supply",
        ],
        "formats": [
            "<u8",
            "<u8",
            "<u2",
            "<u2",
            "<u2",
            "<u2",
            "u1",
            "u1",
            "u1",
            ("u1", (32,)),
            _POSITION_DTYPE,
            ("u1", (GRADED_PUBLIC_SUPPLY_SIZE,)),
        ],
        "offsets": [0, 8, 16, 18, 20, 22, 24, 25, 26, 28, 60, 924],
        "itemsize": GRADED_GROUP_HEADER_SIZE,
    }
)

_GRADED_ESTIMATE_DTYPE = np.dtype(
    {
        "names": ["mean", "stddev", "samples"],
        "formats": ["<f4", "<f4", "<u2"],
        "offsets": [0, 4, 8],
        "itemsize": 12,
    }
)

_GRADED_ACTION_DTYPE = np.dtype(
    {
        "names": [
            "same_slot_independent",
            "draft_kind",
            "tile_slot",
            "wildlife_slot",
            "tile_id",
            "tile_terrain_a",
            "tile_terrain_b",
            "tile_wildlife_mask",
            "tile_keystone",
            "drafted_wildlife",
            "tile_q",
            "tile_r",
            "rotation",
            "wildlife_present",
            "wildlife_q",
            "wildlife_r",
            "replace_three_of_a_kind",
            "wipe_count",
            "wipe_masks",
            "staged_active_nature_tokens",
            "staged_market_entities",
            "staged_public_supply",
            "immediate_score",
            "immediate_deltas",
        ],
        "formats": [
            "u1",
            "u1",
            "u1",
            "u1",
            "u1",
            "u1",
            "u1",
            "u1",
            "u1",
            "u1",
            "i1",
            "i1",
            "u1",
            "u1",
            "i1",
            "i1",
            "u1",
            "u1",
            ("u1", (GRADED_MAX_WIPES,)),
            "u1",
            ("u1", (4, 8)),
            ("u1", (GRADED_PUBLIC_SUPPLY_SIZE,)),
            "<u2",
            ("<i2", (11,)),
        ],
        "offsets": [*range(18), 18, 38, 42, 74, 104, 106],
        "itemsize": GRADED_ACTION_STORAGE_SIZE,
    }
)

_GRADED_CANDIDATE_DTYPE = np.dtype(
    {
        "names": [
            "action_hash",
            "canonical_index",
            "screen_rank",
            "source_flags",
            "fidelity_mask",
            "model_immediate_score",
            "model_remaining_value",
            "screen_value",
            "uniform_market_survival_proxy",
            "visible_wildlife_count",
            "public_bag_wildlife_count",
            "action",
            "r600",
            "r1200",
            "r4800",
        ],
        "formats": [
            ("u1", (32,)),
            "<u2",
            "<u2",
            "<u2",
            "<u2",
            "<f4",
            "<f4",
            "<f4",
            "<f4",
            "u1",
            "u1",
            _GRADED_ACTION_DTYPE,
            _GRADED_ESTIMATE_DTYPE,
            _GRADED_ESTIMATE_DTYPE,
            _GRADED_ESTIMATE_DTYPE,
        ],
        "offsets": [0, 32, 34, 36, 38, 40, 44, 48, 52, 56, 57, 60, 188, 200, 212],
        "itemsize": GRADED_CANDIDATE_SIZE,
    }
)


class CensusError(ValueError):
    """Raised when an input or report violates the frozen census contract."""


@dataclasses.dataclass(frozen=True)
class BlockSpec:
    block_id: str
    schema: str
    name: str
    ownership: dict[str, Any]
    semantic_owner: str
    value_domain: str
    expected_d6_behavior: str
    perspective_convention: str
    incremental_dependencies: tuple[str, ...]
    compatibility: str
    row_domain: str
    width: int
    implementation_status: str = "implemented"
    measurement_status: str = "measurable"
    known_status: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "schema": self.schema,
            "name": self.name,
            "ownership": self.ownership,
            "semantic_owner": self.semantic_owner,
            "value_domain": self.value_domain,
            "expected_d6_behavior": self.expected_d6_behavior,
            "perspective_convention": self.perspective_convention,
            "incremental_dependencies": list(self.incremental_dependencies),
            "compatibility": self.compatibility,
            "row_domain": self.row_domain,
            "width": self.width,
            "implementation_status": self.implementation_status,
            "measurement_status": self.measurement_status,
            "known_status": list(self.known_status),
            "evidence": list(self.evidence),
        }


def _legacy_block(
    block_id: str,
    name: str,
    start: int,
    stop: int,
    owner: str,
    domain: str,
    d6: str,
    perspective: str,
    dependencies: Sequence[str],
    *,
    known_status: Sequence[str] = (),
) -> BlockSpec:
    return BlockSpec(
        block_id=block_id,
        schema="legacy-mid-v4opp-11231",
        name=name,
        ownership={"kind": "feature_index_range", "start": start, "stop": stop},
        semantic_owner=owner,
        value_domain=domain,
        expected_d6_behavior=d6,
        perspective_convention=perspective,
        incremental_dependencies=tuple(dependencies),
        compatibility="Exact historical mid-features,v4-opp checkpoint columns; append-only.",
        row_domain="legacy_sparse_position",
        width=stop - start,
        known_status=tuple(known_status),
        evidence=(
            "legacy/crates/cascadia-ai/src/nnue.rs",
            "python/cascadia_mlx/legacy_nnue.py",
        ),
    )


def _spec(
    block_id: str,
    schema: str,
    name: str,
    ownership: dict[str, Any],
    owner: str,
    domain: str,
    d6: str,
    perspective: str,
    dependencies: Sequence[str],
    compatibility: str,
    row_domain: str,
    width: int,
    *,
    known_status: Sequence[str] = (),
    evidence: Sequence[str] = (),
    implementation_status: str = "implemented",
    measurement_status: str = "measurable",
) -> BlockSpec:
    return BlockSpec(
        block_id=block_id,
        schema=schema,
        name=name,
        ownership=ownership,
        semantic_owner=owner,
        value_domain=domain,
        expected_d6_behavior=d6,
        perspective_convention=perspective,
        incremental_dependencies=tuple(dependencies),
        compatibility=compatibility,
        row_domain=row_domain,
        width=width,
        implementation_status=implementation_status,
        measurement_status=measurement_status,
        known_status=tuple(known_status),
        evidence=tuple(evidence),
    )


def block_specs() -> tuple[BlockSpec, ...]:
    """Return the frozen declarative registry for implemented and future schemas."""
    blocks: list[BlockSpec] = [
        _legacy_block(
            "legacy.cell_core",
            "441-cell wildlife, vacancy, and primary terrain",
            0,
            4_851,
            "focal board absolute lattice",
            "sparse binary; 441 cells x 11 mutually constrained channels",
            "Not invariant: rotations/reflections permute absolute cell ownership.",
            "Focal player's board only.",
            ("placed tiles", "placed wildlife", "grid index"),
        ),
        _legacy_block(
            "legacy.turn",
            "global completed-turn bucket",
            4_851,
            4_872,
            "game phase",
            "one-hot binary, turns 0..20",
            "Invariant.",
            "Focal personal-turn convention in legacy extractor.",
            ("completed focal turns",),
        ),
        _legacy_block(
            "legacy.nature_tokens",
            "focal Nature Token bucket",
            4_872,
            4_881,
            "focal resource state",
            "one-hot binary, 0..7 and 8+",
            "Invariant.",
            "Focal player.",
            ("focal Nature Tokens",),
        ),
        _legacy_block(
            "legacy.wildlife_counts_v1",
            "focal wildlife counts, legacy bins",
            4_881,
            4_911,
            "focal wildlife inventory",
            "5 x 6 one-hot binary bins",
            "Invariant.",
            "Focal player.",
            ("focal wildlife positions",),
        ),
        _legacy_block(
            "legacy.habitat_sizes_v1",
            "focal largest habitats, legacy bins",
            4_911,
            4_961,
            "focal habitat structure",
            "5 x 10 one-hot binary bins",
            "Invariant.",
            "Focal player.",
            ("largest habitat components",),
        ),
        _legacy_block(
            "legacy.wildlife_pairs",
            "wildlife pair states in three line directions",
            4_961,
            5_108,
            "focal wildlife local geometry",
            "3 x 7 x 7 sparse binary states",
            "Rotations permute directions; reflections reverse directed ownership.",
            "Focal player.",
            ("placed wildlife", "three canonical line directions"),
        ),
        _legacy_block(
            "legacy.patterns_v1",
            "Card A wildlife pattern summaries",
            5_108,
            5_197,
            "focal wildlife scoring motifs",
            "mixed one-hot bins and sparse bits",
            "Invariant when motif extraction is exact.",
            "Focal player.",
            ("bear", "elk", "salmon", "hawk", "fox", "empty eligible slots"),
        ),
        _legacy_block(
            "legacy.bag_wildlife_v1",
            "public wildlife bag marginals",
            5_197,
            5_252,
            "public wildlife supply",
            "5 x 11 one-hot binary bins",
            "Invariant.",
            "Public state shared by all focal seats.",
            ("public wildlife bag counts",),
        ),
        _legacy_block(
            "legacy.opponent_habitat_max_v1",
            "maximum opponent habitat by terrain",
            5_252,
            5_307,
            "compressed opponent habitat threat",
            "5 x 11 one-hot binary bins",
            "Invariant.",
            "Maximum over opponents; opponent identity discarded.",
            ("all opponent largest habitats",),
        ),
        _legacy_block(
            "legacy.allowed_wildlife",
            "per-cell allowed-wildlife mask",
            5_307,
            7_512,
            "focal board tile affordance",
            "441 cells x 5 sparse binary mask bits",
            "Not invariant: D6 permutes absolute cell ownership.",
            "Focal player's board only.",
            ("placed tiles", "tile wildlife mask"),
        ),
        _legacy_block(
            "legacy.wildlife_counts_ext",
            "focal wildlife counts, extended bins",
            7_512,
            7_562,
            "focal wildlife inventory",
            "5 x 10 one-hot binary bins",
            "Invariant.",
            "Focal player.",
            ("focal wildlife positions",),
        ),
        _legacy_block(
            "legacy.terrain_pairs",
            "terrain pair states in three line directions",
            7_562,
            7_670,
            "focal habitat local geometry",
            "3 x 6 x 6 sparse binary states",
            "Rotations permute directions; reflections reverse directed ownership.",
            "Focal player.",
            ("placed tile edges", "three canonical line directions"),
        ),
        _legacy_block(
            "legacy.secondary_terrain",
            "per-cell secondary terrain",
            7_670,
            9_875,
            "focal board dual-terrain semantics",
            "441 cells x 5 sparse binary channels",
            "Not invariant: D6 permutes absolute cell ownership.",
            "Focal player's board only.",
            ("placed dual-terrain tiles",),
        ),
        _legacy_block(
            "legacy.habitat_sizes_ext",
            "focal largest habitats, extended bins",
            9_875,
            9_945,
            "focal habitat structure",
            "5 x 14 one-hot binary bins",
            "Invariant.",
            "Focal player.",
            ("largest habitat components",),
        ),
        _legacy_block(
            "legacy.wildlife_counts_ext2",
            "focal wildlife counts, 0..10+ bins",
            9_945,
            10_000,
            "focal wildlife inventory",
            "5 x 11 one-hot binary bins",
            "Invariant.",
            "Focal player.",
            ("focal wildlife positions",),
        ),
        _legacy_block(
            "legacy.extension_capacity",
            "wildlife extension-capacity buckets",
            10_000,
            10_040,
            "focal wildlife opportunity",
            "5 x 8 one-hot binary bins",
            "Invariant.",
            "Focal player.",
            ("placed wildlife", "empty eligible neighboring tiles"),
        ),
        _legacy_block(
            "legacy.patterns_v2",
            "extended Card A pattern opportunity summaries",
            10_040,
            10_088,
            "focal wildlife motif opportunity",
            "mixed one-hot bins and sparse bits",
            "Invariant when motif extraction is exact.",
            "Focal player.",
            ("wildlife motifs", "eligible empty slots"),
        ),
        _legacy_block(
            "legacy.bag_wildlife_ext",
            "public wildlife bag marginals, extended",
            10_088,
            10_193,
            "public wildlife supply",
            "5 x 21 one-hot binary bins",
            "Invariant.",
            "Public state shared by all focal seats.",
            ("public wildlife bag counts",),
        ),
        _legacy_block(
            "legacy.opponent_habitat_max_ext",
            "maximum opponent habitat, extended",
            10_193,
            10_263,
            "compressed opponent habitat threat",
            "5 x 14 one-hot binary bins",
            "Invariant.",
            "Maximum over opponents; opponent identity discarded.",
            ("all opponent largest habitats",),
        ),
        _legacy_block(
            "legacy.market",
            "four visible tile/wildlife market slots",
            10_263,
            10_351,
            "public market",
            "4 x 22 mixed one-hot and mask bits",
            "Invariant except tile orientation semantics are absent.",
            "Public state shared by all focal seats.",
            ("visible tile market", "visible wildlife market"),
        ),
        _legacy_block(
            "legacy.tile_bag_terrain_marginal",
            "unseen tile terrain marginals",
            10_351,
            10_456,
            "public tile supply marginal",
            "5 x 21 one-hot binary bins",
            "Invariant.",
            "Public state shared by all focal seats.",
            ("unseen tile terrain capacities",),
        ),
        _legacy_block(
            "legacy.tile_bag_wildlife_marginal",
            "unseen tile wildlife-capacity marginals",
            10_456,
            10_561,
            "public tile supply marginal",
            "5 x 21 one-hot binary bins",
            "Invariant.",
            "Public state shared by all focal seats.",
            ("unseen tile wildlife capacities",),
        ),
        _legacy_block(
            "legacy.mid_tail_historical_adjacency_prefix",
            "historical mid-features tail: accidental adjacency prefix",
            10_561,
            10_862,
            "first 301 columns of per-cell six-direction adjacency",
            "sparse binary; cells 0..2 complete plus 67 columns of cell 3",
            "Not invariant and not the documented supply tail.",
            "Focal player's board only.",
            ("placed tile adjacency", "historical mid-features truncation"),
            known_status=("schema_defect", "structurally_misdocumented"),
        ),
        _legacy_block(
            "legacy.v4opp",
            "three detailed opponent slots",
            10_862,
            11_231,
            "opponent wildlife, habitat, tokens, and Card A threat bits",
            "3 x 123 mixed one-hot bins and sparse bits",
            "Invariant within each opponent slot.",
            (
                "Historical BagInfo slot order; characterize all focal seats because "
                "old extraction is not guaranteed focal-relative."
            ),
            ("three opponent summaries",),
            known_status=("historical_perspective_requires_census",),
        ),
    ]
    v1_blocks = [
        dataclasses.replace(
            spec,
            block_id=spec.block_id.replace("legacy.", "legacy_v1.", 1),
            schema="legacy-nnue-v1-5197",
            compatibility="Exact historical 5,197-column V1 checkpoint layout.",
        )
        for spec in blocks
        if int(spec.ownership.get("stop", 0)) <= 5_197
    ]
    blocks = [*v1_blocks, *blocks]

    position_evidence = (
        "crates/cascadia-data/src/lib.rs",
        "python/cascadia_mlx/dataset.py",
    )
    compatibility = "compact-entity-v2 fixed 864-byte PositionRecord and decoded tensors."
    blocks.extend(
        [
            _spec(
                "v2.board.coordinates",
                POSITION_FEATURE_SCHEMA,
                "board axial coordinates",
                {"kind": "tensor_slice", "tensor": "board_entities", "slice": "[:,0:4,0:23,0:2]"},
                "all four public boards",
                "float32 q/24 and r/24; masked padding is zero",
                "D6 transforms q/r exactly and permutes token rows only by canonical sort.",
                "Board slot 0 is focal; slots 1..3 are relative clockwise seats.",
                ("board tile coordinates", "board_counts"),
                compatibility,
                "position_record",
                4 * 23 * 2,
                evidence=position_evidence,
            ),
            _spec(
                "v2.board.primary_terrain",
                POSITION_FEATURE_SCHEMA,
                "board primary terrain",
                {"kind": "tensor_slice", "tensor": "board_entities", "slice": "[:,0:4,0:23,2:7]"},
                "all four public boards",
                "five-way one-hot; masked padding is zero",
                "Invariant values; token coordinates transform.",
                "Focal-relative board slots.",
                ("board tile terrain_a", "board_counts"),
                compatibility,
                "position_record",
                4 * 23 * 5,
                evidence=position_evidence,
            ),
            _spec(
                "v2.board.secondary_terrain",
                POSITION_FEATURE_SCHEMA,
                "board secondary terrain or none",
                {"kind": "tensor_slice", "tensor": "board_entities", "slice": "[:,0:4,0:23,7:13]"},
                "all four public boards",
                "six-way one-hot including none; masked padding is zero",
                "Invariant values; token coordinates transform.",
                "Focal-relative board slots.",
                ("board tile terrain_b", "board_counts"),
                compatibility,
                "position_record",
                4 * 23 * 6,
                evidence=position_evidence,
            ),
            _spec(
                "v2.board.rotation",
                POSITION_FEATURE_SCHEMA,
                "board tile rotation",
                {"kind": "tensor_slice", "tensor": "board_entities", "slice": "[:,0:4,0:23,13:19]"},
                "all four public boards",
                "six-way one-hot; masked padding is zero",
                "Rotations add modulo six; reflections reverse orientation.",
                "Focal-relative board slots.",
                ("board tile rotation", "board_counts"),
                compatibility,
                "position_record",
                4 * 23 * 6,
                evidence=position_evidence,
            ),
            _spec(
                "v2.board.allowed_wildlife",
                POSITION_FEATURE_SCHEMA,
                "board allowed-wildlife mask",
                {"kind": "tensor_slice", "tensor": "board_entities", "slice": "[:,0:4,0:23,19:24]"},
                "all four public boards",
                "five independent binary mask bits; masked padding is zero",
                "Invariant values; token coordinates transform.",
                "Focal-relative board slots.",
                ("board tile wildlife mask", "board_counts"),
                compatibility,
                "position_record",
                4 * 23 * 5,
                evidence=position_evidence,
            ),
            _spec(
                "v2.board.placed_wildlife",
                POSITION_FEATURE_SCHEMA,
                "board placed wildlife or none",
                {"kind": "tensor_slice", "tensor": "board_entities", "slice": "[:,0:4,0:23,24:30]"},
                "all four public boards",
                "six-way one-hot including none; masked padding is zero",
                "Invariant values; token coordinates transform.",
                "Focal-relative board slots.",
                ("placed wildlife", "board_counts"),
                compatibility,
                "position_record",
                4 * 23 * 6,
                evidence=position_evidence,
            ),
            _spec(
                "v2.board.keystone",
                POSITION_FEATURE_SCHEMA,
                "board keystone bit",
                {"kind": "tensor_slice", "tensor": "board_entities", "slice": "[:,0:4,0:23,30:31]"},
                "all four public boards",
                "binary; masked padding is zero",
                "Invariant value; token coordinates transform.",
                "Focal-relative board slots.",
                ("tile keystone", "board_counts"),
                compatibility,
                "position_record",
                4 * 23,
                evidence=position_evidence,
            ),
            _spec(
                "v2.market.coordinates_rotation",
                POSITION_FEATURE_SCHEMA,
                "market coordinate and rotation placeholders",
                {
                    "kind": "tensor_slices",
                    "tensor": "market_entities",
                    "slices": ["[:,:,0:2]", "[:,:,13:19]"],
                },
                "market compatibility padding",
                "structural zero",
                "Invariant.",
                "Public market shared by all focal seats.",
                (),
                compatibility,
                "position_record",
                4 * 8,
                known_status=("structural_constant_zero",),
                evidence=position_evidence,
            ),
            _spec(
                "v2.market.semantic",
                POSITION_FEATURE_SCHEMA,
                "market tile and wildlife semantics",
                {
                    "kind": "tensor_slices",
                    "tensor": "market_entities",
                    "slices": ["[:,:,2:13]", "[:,:,19:31]"],
                },
                "four public market slots",
                "terrain one-hots, allowed mask, wildlife one-hot, keystone",
                "Invariant; tile orientation is not represented.",
                "Public market shared by all focal seats.",
                ("market tile entities", "market wildlife entities"),
                compatibility,
                "position_record",
                4 * 23,
                evidence=position_evidence,
            ),
            _spec(
                "v2.global.phase",
                POSITION_FEATURE_SCHEMA,
                "normalized phase and turns remaining",
                {"kind": "tensor_slice", "tensor": "global_features", "slice": "[:,0:2]"},
                "game clock",
                "two float32 values in [0,1]",
                "Invariant.",
                "Same public clock for all focal seats.",
                ("turn", "total_turns"),
                compatibility,
                "position_record",
                2,
                evidence=position_evidence,
            ),
            _spec(
                "v2.global.player_count",
                POSITION_FEATURE_SCHEMA,
                "player-count one-hot",
                {"kind": "tensor_slice", "tensor": "global_features", "slice": "[:,2:6]"},
                "rules configuration",
                "four-way one-hot for one through four players",
                "Invariant.",
                "Public configuration.",
                ("player_count",),
                compatibility,
                "position_record",
                4,
                known_status=("constant_under_frozen_4p_domain",),
                evidence=position_evidence,
            ),
            _spec(
                "v2.global.nature_tokens",
                POSITION_FEATURE_SCHEMA,
                "all-player Nature Tokens",
                {"kind": "tensor_slice", "tensor": "global_features", "slice": "[:,6:10]"},
                "public player resources",
                "four float32 counts divided by 20",
                "Invariant.",
                "Focal-relative player order.",
                ("nature_tokens",),
                compatibility,
                "position_record",
                4,
                evidence=position_evidence,
            ),
            _spec(
                "v2.global.board_counts",
                POSITION_FEATURE_SCHEMA,
                "all-player board tile counts",
                {"kind": "tensor_slice", "tensor": "global_features", "slice": "[:,10:14]"},
                "public board sizes",
                "four float32 counts divided by 23",
                "Invariant.",
                "Focal-relative player order.",
                ("board_counts",),
                compatibility,
                "position_record",
                4,
                evidence=position_evidence,
            ),
            _spec(
                "v2.global.wildlife_counts",
                POSITION_FEATURE_SCHEMA,
                "all-player wildlife counts",
                {"kind": "tensor_slice", "tensor": "global_features", "slice": "[:,14:34]"},
                "public wildlife inventory",
                "4 x 5 float32 counts divided by 20",
                "Invariant.",
                "Focal-relative player order.",
                ("wildlife_counts",),
                compatibility,
                "position_record",
                20,
                evidence=position_evidence,
            ),
            _spec(
                "v2.global.habitat_sizes",
                POSITION_FEATURE_SCHEMA,
                "all-player largest habitats",
                {"kind": "tensor_slice", "tensor": "global_features", "slice": "[:,34:54]"},
                "public habitat structure",
                "4 x 5 float32 counts divided by 23",
                "Invariant.",
                "Focal-relative player order.",
                ("habitat_sizes",),
                compatibility,
                "position_record",
                20,
                evidence=position_evidence,
            ),
            _spec(
                "v2.global.market_wildlife",
                POSITION_FEATURE_SCHEMA,
                "market wildlife one-hots",
                {"kind": "tensor_slice", "tensor": "global_features", "slice": "[:,54:74]"},
                "public wildlife market",
                "4 x 5 one-hot; empty slot is all zero",
                "Invariant.",
                "Public market shared by all focal seats.",
                ("market wildlife",),
                compatibility,
                "position_record",
                20,
                evidence=position_evidence,
            ),
            _spec(
                "v2.global.scoring_cards",
                POSITION_FEATURE_SCHEMA,
                "five scoring-card one-hots",
                {"kind": "tensor_slice", "tensor": "global_features", "slice": "[:,74:94]"},
                "rules configuration",
                "5 x 4 one-hot",
                "Invariant.",
                "Public configuration.",
                ("scoring_cards",),
                compatibility,
                "position_record",
                20,
                known_status=("constant_under_frozen_AAAAA_domain",),
                evidence=position_evidence,
            ),
            _spec(
                "v2.global.habitat_bonus",
                POSITION_FEATURE_SCHEMA,
                "habitat-bonus enabled bit",
                {"kind": "tensor_slice", "tensor": "global_features", "slice": "[:,94:95]"},
                "rules configuration",
                "binary",
                "Invariant.",
                "Public configuration.",
                ("habitat_bonuses",),
                compatibility,
                "position_record",
                1,
                known_status=("constant_zero_under_frozen_domain",),
                evidence=position_evidence,
            ),
            _spec(
                "v2.global.market_diversity",
                POSITION_FEATURE_SCHEMA,
                "market wildlife diversity",
                {"kind": "tensor_slice", "tensor": "global_features", "slice": "[:,95:96]"},
                "public market summary",
                "distinct visible wildlife types divided by four",
                "Invariant.",
                "Public market shared by all focal seats.",
                ("market wildlife",),
                compatibility,
                "position_record",
                1,
                evidence=position_evidence,
            ),
        ]
    )

    graded_evidence = (
        "crates/cascadia-data/src/graded_oracle.rs",
        "python/cascadia_mlx/graded_oracle_dataset.py",
    )
    graded_compat = "complete-action-graded-oracle-v1 storage and decoded model inputs."
    action_slices = (
        ("same_slot_independent", "0:1", 1, "draft coupling", "binary"),
        ("draft_kind", "1:3", 2, "draft kind", "two-way one-hot"),
        ("tile_slot", "3:7", 4, "tile market slot", "four-way one-hot"),
        ("wildlife_slot", "7:11", 4, "wildlife market slot", "four-way one-hot"),
        ("tile_id", "11:12", 1, "stable serialization tile ID", "tile_id/84; non-semantic ordinal"),
        ("tile_terrain_a", "12:17", 5, "drafted tile primary terrain", "five-way one-hot"),
        (
            "tile_terrain_b",
            "17:23",
            6,
            "drafted tile secondary terrain",
            "six-way one-hot including none",
        ),
        ("tile_wildlife_mask", "23:28", 5, "drafted tile wildlife eligibility", "five mask bits"),
        ("tile_keystone", "28:29", 1, "drafted tile keystone", "binary"),
        ("drafted_wildlife", "29:34", 5, "drafted wildlife", "five-way one-hot"),
        ("tile_coordinates", "34:36", 2, "tile placement", "q/24 and r/24"),
        ("tile_rotation", "36:42", 6, "tile orientation", "six-way one-hot"),
        ("wildlife_present", "42:43", 1, "wildlife placement presence", "binary"),
        (
            "wildlife_coordinates",
            "43:45",
            2,
            "wildlife placement",
            "q/24 and r/24 gated by presence",
        ),
        ("replace_three", "45:46", 1, "free replacement choice", "binary"),
        ("wipe_count", "46:47", 1, "ordered paid-wipe count", "count/20"),
        ("wipe_masks", "47:127", 80, "ordered paid-wipe masks", "20 x 4 binary bits"),
        ("staged_nature_tokens", "127:128", 1, "post-prelude focal Nature Tokens", "count/20"),
        ("immediate_score", "128:129", 1, "exact observable immediate score", "score/100"),
        (
            "immediate_deltas",
            "129:140",
            11,
            "exact observable score-component deltas",
            "11 signed values/20",
        ),
    )
    for suffix, tensor_slice, width, owner, domain in action_slices:
        blocks.append(
            _spec(
                f"graded.action.{suffix}",
                GRADED_FEATURE_SCHEMA,
                suffix.replace("_", " "),
                {
                    "kind": "tensor_slice",
                    "tensor": "action_features",
                    "slice": f"[:,:,{tensor_slice}]",
                },
                owner,
                domain,
                (
                    "Coordinates and rotation transform under D6; all other semantic values "
                    "are invariant. Reflections require the shared D6 contract."
                    if suffix in {"tile_coordinates", "tile_rotation", "wildlife_coordinates"}
                    else "Invariant."
                ),
                "Candidate belongs to the current focal seat; parent boards are focal-relative.",
                ("lossless TurnAction", "staged public transition"),
                graded_compat,
                "graded_candidate",
                width,
                known_status=("nonsemantic_scalar_id",) if suffix == "tile_id" else (),
                evidence=graded_evidence,
            )
        )
    blocks.extend(
        [
            _spec(
                "graded.prior.observable",
                GRADED_FEATURE_SCHEMA,
                "observable candidate priors",
                {"kind": "tensor_slice", "tensor": "prior_features", "slice": "[:,:,0:8]"},
                "live-computable screen and market priors",
                "eight normalized float32 values",
                "Invariant.",
                "Current focal decision.",
                ("screen model", "public market", "public wildlife bag"),
                graded_compat,
                "graded_candidate",
                8,
                evidence=graded_evidence,
            ),
            _spec(
                "graded.parent_public_supply",
                GRADED_FEATURE_SCHEMA,
                "parent public-supply marginals",
                {"kind": "tensor_slice", "tensor": "public_supply", "slice": "[:,0:30]"},
                "public wildlife and tile-supply marginals",
                "5 counts/20 plus 25 counts/81",
                "Invariant.",
                "Public state shared by all focal seats.",
                ("public bag", "unseen tile marginals"),
                graded_compat,
                "graded_candidate",
                30,
                known_status=("known_noninjective_supply_summary",),
                evidence=graded_evidence,
            ),
            _spec(
                "graded.staged_market",
                GRADED_FEATURE_SCHEMA,
                "candidate-specific staged market",
                {"kind": "tensor", "tensor": "staged_market_entities", "shape": "[candidate,4,31]"},
                "public market after ordered prelude",
                "four compact entity rows",
                "Invariant; tile orientation is not represented.",
                "Candidate-specific public state.",
                ("ordered wipes", "free replacement", "draft kind"),
                graded_compat,
                "graded_candidate",
                4 * 31,
                evidence=graded_evidence,
            ),
            _spec(
                "graded.staged_public_supply",
                GRADED_FEATURE_SCHEMA,
                "candidate-specific staged public-supply marginals",
                {"kind": "tensor_slice", "tensor": "staged_public_supply", "slice": "[:,:,0:30]"},
                "public supply after ordered prelude",
                "5 counts/20 plus 25 counts/81",
                "Invariant.",
                "Candidate-specific public state.",
                ("ordered wipes", "free replacement", "public supply"),
                graded_compat,
                "graded_candidate",
                30,
                known_status=("known_noninjective_supply_summary",),
                evidence=graded_evidence,
            ),
        ]
    )

    for factor_index, factor_name in enumerate(FACTOR_NAMES):
        blocks.append(
            _spec(
                f"factor_cache.{factor_name}",
                FACTOR_CACHE_SCHEMA,
                f"{factor_name} pre-compression factor",
                {
                    "kind": "tensor_slice",
                    "tensor": "factors",
                    "slice": f"[:,{factor_index}:{factor_index + 1},0:192]",
                },
                f"frozen graded-oracle {factor_name} latent",
                "192 float32 learned activations",
                "Inherited from the frozen encoder; no exact D6 guarantee.",
                "Candidate-specific, parent encoder uses focal-relative boards.",
                ("frozen ADR 0081 checkpoint", "observable candidate inputs"),
                "graded-oracle-frozen-candidate-factors-v1 cache only.",
                "factor_candidate",
                FACTOR_DIM,
                evidence=(
                    "python/cascadia_mlx/graded_oracle_model.py",
                    "python/cascadia_mlx/graded_oracle_factor_integration.py",
                ),
            )
        )

    hierarchical_slices = (
        ("group_state.board_entities", "group_state", 0, 2_852, "flattened board entities"),
        ("group_state.board_mask", "group_state", 2_852, 2_944, "flattened board masks"),
        (
            "group_state.market_entities",
            "group_state",
            2_944,
            3_068,
            "flattened market entities",
        ),
        ("group_state.market_mask", "group_state", 3_068, 3_072, "market masks"),
        ("group_state.global", "group_state", 3_072, 3_168, "global features"),
        ("group_state.public_supply", "group_state", 3_168, 3_198, "public supply"),
        (
            "draft_query_context.constant",
            "draft_query_context",
            0,
            1,
            "structural draft-query placeholder",
        ),
        (
            "draft_item.draft_factor",
            "draft_item_features",
            0,
            117,
            "complete-action draft prefix",
        ),
        (
            "draft_item.staged_public",
            "draft_item_features",
            117,
            275,
            "staged market, mask, and supply",
        ),
        (
            "draft_item.descendant_min",
            "draft_item_features",
            275,
            295,
            "minimum observable descendant values",
        ),
        (
            "draft_item.descendant_mean",
            "draft_item_features",
            295,
            315,
            "mean observable descendant values",
        ),
        (
            "draft_item.descendant_max",
            "draft_item_features",
            315,
            335,
            "maximum observable descendant values",
        ),
        (
            "draft_item.descendant_count",
            "draft_item_features",
            335,
            336,
            "log-scaled descendant count",
        ),
        (
            "tile_query.draft_factor",
            "tile_query_context",
            0,
            117,
            "selected draft prefix",
        ),
        (
            "tile_query.staged_public",
            "tile_query_context",
            117,
            275,
            "staged market, mask, and supply",
        ),
        ("tile_item.tile_factor", "tile_item_features", 0, 8, "tile placement factor"),
        (
            "tile_item.local_geometry",
            "tile_item_features",
            8,
            188,
            "six tile-neighbor relation rows",
        ),
        (
            "tile_item.descendant_min",
            "tile_item_features",
            188,
            208,
            "minimum observable descendant values",
        ),
        (
            "tile_item.descendant_mean",
            "tile_item_features",
            208,
            228,
            "mean observable descendant values",
        ),
        (
            "tile_item.descendant_max",
            "tile_item_features",
            228,
            248,
            "maximum observable descendant values",
        ),
        (
            "tile_item.descendant_count",
            "tile_item_features",
            248,
            249,
            "log-scaled descendant count",
        ),
        (
            "wildlife_query.draft_factor",
            "wildlife_query_context",
            0,
            117,
            "selected draft prefix",
        ),
        (
            "wildlife_query.staged_public",
            "wildlife_query_context",
            117,
            275,
            "staged market, mask, and supply",
        ),
        (
            "wildlife_query.tile_factor",
            "wildlife_query_context",
            275,
            283,
            "selected tile placement factor",
        ),
        (
            "wildlife_query.tile_local_geometry",
            "wildlife_query_context",
            283,
            463,
            "selected tile local geometry",
        ),
        (
            "wildlife_item.wildlife_factor",
            "wildlife_item_features",
            0,
            3,
            "wildlife placement factor",
        ),
        (
            "wildlife_item.local_geometry",
            "wildlife_item_features",
            3,
            213,
            "seven wildlife-neighborhood relation rows",
        ),
        (
            "wildlife_item.observable_descendant",
            "wildlife_item_features",
            213,
            233,
            "observable prior plus immediate action values",
        ),
    )
    for suffix, array, start, stop, owner in hierarchical_slices:
        known_status = (
            ("structural_constant_zero",)
            if suffix == "draft_query_context.constant"
            else ()
        )
        blocks.append(
            _spec(
                f"hierarchical.{suffix}",
                HIERARCHICAL_CACHE_SCHEMA,
                suffix.replace("_", " ").replace(".", " "),
                {
                    "kind": "npz_array_slice",
                    "array": array,
                    "slice": f"[:,{start}:{stop}]",
                },
                owner,
                "float32",
                (
                    "No exact D6 guarantee; local geometry is rotation-canonical only "
                    "where documented."
                ),
                (
                    "Focal-relative state; cache does not retain absolute focal seat, so "
                    "seat census "
                    "is unavailable without source alignment."
                ),
                ("observable parent/action features", "hierarchical prefix construction"),
                "hierarchical-factor-retrieval-cache-v1 only.",
                array,
                stop - start,
                known_status=known_status,
                evidence=("python/cascadia_mlx/full_legal_hierarchical_factor_retrieval.py",),
            )
        )

    future_evidence = ("docs/v2/reports/feature-representation-audit-2026-06-16.md",)
    blocks.extend(
        [
            _spec(
                "future.corrected_mid_tail",
                "legacy-mid-v4-fixed-v1",
                "intended extended tile-supply tail and overflow bit",
                {"kind": "proposed_feature_index_range", "start": 10_561, "stop": 10_862},
                "exact intended supply tail",
                "150 terrain bins + 150 wildlife-capacity bins + overflow bit",
                "Invariant.",
                "Public state shared by all focal seats.",
                ("unseen tile bag", "overflow refresh state"),
                "New schema and checkpoint only; historical columns must not be reinterpreted.",
                "unimplemented",
                301,
                implementation_status="unimplemented",
                measurement_status="unmeasurable",
                known_status=("manifest_only",),
                evidence=future_evidence,
            ),
            _spec(
                "future.relational_opportunity_graph",
                "relational-opportunity-graph-v0",
                "typed multi-resolution relational opportunity graph",
                {
                    "kind": "proposed_token_relation_schema",
                    "tokens": [
                        "tiles",
                        "frontiers",
                        "habitat_components",
                        "wildlife_motifs",
                        "supply_archetypes",
                        "market_items",
                        "opponents",
                        "complete_actions",
                    ],
                    "relations": [
                        "hex_adjacency",
                        "component_membership",
                        "motif_membership",
                        "action_edit",
                        "demand",
                        "supply",
                        "candidate_interaction",
                    ],
                },
                "future CascadiaZero representation",
                "not yet frozen",
                "Must implement exact shared D6 contract.",
                "Must be focal-relative with explicit absolute-seat metadata.",
                ("F2 footprint", "F3 D6", "S1-S4 semantics", "P0-P2 policy"),
                "No checkpoint compatibility; proposed schema only.",
                "unimplemented",
                0,
                implementation_status="unimplemented",
                measurement_status="unmeasurable",
                known_status=("manifest_only",),
                evidence=future_evidence,
            ),
        ]
    )
    return tuple(blocks)


def _schema_hash(schema: dict[str, Any]) -> str:
    return scientific_blake3(schema)


def build_manifest() -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for block in block_specs():
        grouped[block.schema].append(block.to_dict())
    schemas = []
    for name in sorted(grouped):
        entry = {
            "name": name,
            "version": SCHEMA_VERSIONS[name],
            "blocks": grouped[name],
        }
        entry["schema_blake3"] = _schema_hash(entry)
        for block in entry["blocks"]:
            block["schema_version"] = entry["version"]
            block["schema_blake3"] = entry["schema_blake3"]
        schemas.append(entry)
    manifest = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "phase_contract": {
            "opening": "personal_turn == 1",
            "early": "2 <= personal_turn <= 5",
            "middle": "6 <= personal_turn <= 13",
            "late": "14 <= personal_turn <= 20",
        },
        "perspective_contract": {
            "v2": "absolute focal seat is recorded; player/board tensors are focal-relative",
            "legacy": (
                "input must record focal seat; historical opponent ordering is measured, "
                "not repaired"
            ),
            "factor_cache": "seat/phase are unknown unless exposed by the cache itself",
        },
        "collision_contract": {
            "structural_alias": "declared by schema construction",
            "empirical_alias": "byte-for-byte equal channel streams on the observed row domain",
            "sketch_alias": "matching deterministic BLAKE3 streams without retained exact cells",
            "representation_collision": (
                "different stable row identities with byte-identical feature rows, verified after "
                "deterministic feature-fingerprint sampling"
            ),
            "unknown": "not inferable from the supplied open evidence",
        },
        "teacher_feature_policy": {
            "allowed": [
                "observable screen priors",
                "exact immediate public score and component deltas",
                "public staged state",
            ],
            "excluded": [
                "r600/r1200/r4800 means",
                "teacher standard deviations and samples",
                "expected-rank labels",
                "target masks",
                "selected/champion labels",
            ],
        },
        "experiment_contract": {
            "open_domains": [
                "complete-action-graded-oracle-v1 train",
                "complete-action-graded-oracle-v1 validation",
                "compact-entity-v2 train/validation datasets when supplied",
                "manifested legacy sparse extraction streams",
                "frozen observable candidate-factor caches",
                "frozen hierarchical feature arrays",
            ],
            "minimum_candidate_rows": 1_000_000,
            "required_focal_seats": [0, 1, 2, 3],
            "success_gates": [
                "every implemented active channel is named and within its boundary",
                "at least one million open candidate rows are scanned",
                "all four focal seats and four frozen phases are reported where exposed",
                "every input manifest, header, size, and payload checksum validates",
                "dead, constant, rare, structural alias, empirical alias, and unknown are distinct",
                "legacy historical extraction semantics are not reinterpreted",
                "scientific BLAKE3 is deterministic and excludes timestamps/output paths",
                "four shard reports contain unique nonduplicative evidence IDs and merge exactly",
            ],
            "closed_domains": [
                "sealed test split",
                "gameplay",
                "new teacher rollout compute",
                "cloud, Modal, or external compute",
                "hidden teacher values as features",
            ],
            "four_machine_sharding": {
                "assignment": "BLAKE3(evidence_id) mod 4",
                "john1": 0,
                "john2": 1,
                "john3": 2,
                "john4": 3,
                "merge_requirement": (
                    "exact shard set 0/4, 1/4, 2/4, 3/4 with no duplicate evidence IDs"
                ),
            },
        },
        "schemas": schemas,
    }
    manifest["scientific_blake3"] = scientific_blake3(manifest)
    return manifest


def _block_schema_identities(manifest: Mapping[str, Any]) -> dict[str, tuple[int, str]]:
    return {
        block["block_id"]: (int(schema["version"]), str(schema["schema_blake3"]))
        for schema in manifest["schemas"]
        for block in schema["blocks"]
    }


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def scientific_blake3(value: object) -> str:
    return blake3.blake3(canonical_json(value)).hexdigest()


def _scientific_manifest_view(value: object) -> object:
    excluded = {
        "created_unix_seconds",
        "updated_unix_seconds",
        "created_unix_ms",
        "updated_unix_ms",
        "started_unix_ms",
        "completed_unix_ms",
        "execution",
        "host",
        "hardware",
        "dataset_root",
        "output_root",
        "output_path",
    }
    if isinstance(value, dict):
        return {
            key: _scientific_manifest_view(item)
            for key, item in value.items()
            if key not in excluded
        }
    if isinstance(value, list):
        return [_scientific_manifest_view(item) for item in value]
    return value


def manifest_scientific_blake3(manifest: Mapping[str, Any]) -> str:
    return scientific_blake3(_scientific_manifest_view(dict(manifest)))


def checksum(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _phase_from_personal_turn(personal_turn: int) -> str:
    if personal_turn <= 1:
        return "opening"
    if personal_turn <= 5:
        return "early"
    if personal_turn <= 13:
        return "middle"
    return "late"


def _phase_from_position(record: np.void) -> str:
    players = max(int(record["player_count"]), 1)
    personal_turn = int(record["turn"]) // players + 1
    return _phase_from_personal_turn(personal_turn)


def _one_hot(values: np.ndarray, classes: int) -> np.ndarray:
    valid = values < classes
    clipped = np.where(valid, values, 0)
    return np.eye(classes, dtype=np.float32)[clipped] * valid[..., None]


def _one_hot_with_none(values: np.ndarray, classes: int) -> np.ndarray:
    mapped = np.where(values < classes, values, classes)
    return np.eye(classes + 1, dtype=np.float32)[mapped]


def _mask_bits(values: np.ndarray, bits: int) -> np.ndarray:
    shifts = np.arange(bits, dtype=np.uint8)
    return ((values[..., None] >> shifts) & 1).astype(np.float32)


def _decode_market(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw)
    mask = (raw[..., 0] < 5) | (raw[..., 3] < 5)
    prefix = raw.shape[:-1]
    values = np.concatenate(
        [
            np.zeros((*prefix, 2), dtype=np.float32),
            _one_hot(raw[..., 0], 5),
            _one_hot_with_none(raw[..., 1], 5),
            np.zeros((*prefix, 6), dtype=np.float32),
            _mask_bits(raw[..., 2], 5),
            _one_hot_with_none(raw[..., 3], 5),
            raw[..., 4, None].astype(np.float32),
        ],
        axis=-1,
    )
    values *= mask[..., None]
    return values


def decode_position_blocks(records: np.ndarray) -> dict[str, np.ndarray]:
    records = np.asarray(records)
    rows = len(records)
    board_raw = records["board_entities"]
    counts = records["board_counts"].astype(np.int32)
    mask = np.arange(POSITION_MAX_BOARD_TILES)[None, None, :] < counts[:, :, None]
    q = board_raw[..., 0].view(np.int8).astype(np.float32) / 24.0
    r = board_raw[..., 1].view(np.int8).astype(np.float32) / 24.0
    coordinates = np.stack([q, r], axis=-1) * mask[..., None]
    primary = _one_hot(board_raw[..., 2], 5) * mask[..., None]
    secondary = _one_hot_with_none(board_raw[..., 3], 5) * mask[..., None]
    rotation = _one_hot(board_raw[..., 4], 6) * mask[..., None]
    allowed = _mask_bits(board_raw[..., 5], 5) * mask[..., None]
    wildlife = _one_hot_with_none(board_raw[..., 6], 5) * mask[..., None]
    keystone = board_raw[..., 7, None].astype(np.float32) * mask[..., None]

    market = _decode_market(records["market_entities"])
    market_placeholders = np.concatenate([market[..., 0:2], market[..., 13:19]], axis=-1)
    market_semantic = np.concatenate([market[..., 2:13], market[..., 19:31]], axis=-1)

    turns = records["turn"].astype(np.float32)
    total_turns = np.maximum(records["total_turns"].astype(np.float32), 1.0)
    phase = np.stack([turns / total_turns, (total_turns - turns) / total_turns], axis=-1)
    market_wildlife = records["market_entities"][..., 3]
    diversity = np.asarray(
        [
            len({int(value) for value in row if value < 5}) / 4.0
            for row in market_wildlife
        ],
        dtype=np.float32,
    )[:, None]
    global_values = {
        "v2.global.phase": phase,
        "v2.global.player_count": _one_hot(records["player_count"] - 1, 4),
        "v2.global.nature_tokens": records["nature_tokens"].astype(np.float32) / 20.0,
        "v2.global.board_counts": counts.astype(np.float32) / 23.0,
        "v2.global.wildlife_counts": records["wildlife_counts"].astype(np.float32).reshape(rows, -1)
        / 20.0,
        "v2.global.habitat_sizes": records["habitat_sizes"].astype(np.float32).reshape(rows, -1)
        / 23.0,
        "v2.global.market_wildlife": _one_hot(market_wildlife, 5).reshape(rows, -1),
        "v2.global.scoring_cards": _one_hot(records["scoring_cards"], 4).reshape(rows, -1),
        "v2.global.habitat_bonus": records["habitat_bonuses"].astype(np.float32)[:, None],
        "v2.global.market_diversity": diversity,
    }
    return {
        "v2.board.coordinates": coordinates.reshape(rows, -1),
        "v2.board.primary_terrain": primary.reshape(rows, -1),
        "v2.board.secondary_terrain": secondary.reshape(rows, -1),
        "v2.board.rotation": rotation.reshape(rows, -1),
        "v2.board.allowed_wildlife": allowed.reshape(rows, -1),
        "v2.board.placed_wildlife": wildlife.reshape(rows, -1),
        "v2.board.keystone": keystone.reshape(rows, -1),
        "v2.market.coordinates_rotation": market_placeholders.reshape(rows, -1),
        "v2.market.semantic": market_semantic.reshape(rows, -1),
        **global_values,
    }


def decode_action_blocks(candidates: np.ndarray) -> dict[str, np.ndarray]:
    candidates = np.asarray(candidates)
    actions = candidates["action"]
    presence = actions["wildlife_present"].astype(np.float32)[:, None]
    wipe_bits = _mask_bits(actions["wipe_masks"], 4).reshape(len(actions), -1)
    action = np.concatenate(
        [
            actions["same_slot_independent"].astype(np.float32)[:, None],
            _one_hot(actions["draft_kind"], 2),
            _one_hot(actions["tile_slot"], 4),
            _one_hot(actions["wildlife_slot"], 4),
            actions["tile_id"].astype(np.float32)[:, None] / 84.0,
            _one_hot(actions["tile_terrain_a"], 5),
            _one_hot_with_none(actions["tile_terrain_b"], 5),
            _mask_bits(actions["tile_wildlife_mask"], 5),
            actions["tile_keystone"].astype(np.float32)[:, None],
            _one_hot(actions["drafted_wildlife"], 5),
            actions["tile_q"].astype(np.float32)[:, None] / 24.0,
            actions["tile_r"].astype(np.float32)[:, None] / 24.0,
            _one_hot(actions["rotation"], 6),
            presence,
            actions["wildlife_q"].astype(np.float32)[:, None] / 24.0 * presence,
            actions["wildlife_r"].astype(np.float32)[:, None] / 24.0 * presence,
            actions["replace_three_of_a_kind"].astype(np.float32)[:, None],
            actions["wipe_count"].astype(np.float32)[:, None] / GRADED_MAX_WIPES,
            wipe_bits,
            actions["staged_active_nature_tokens"].astype(np.float32)[:, None] / 20.0,
            actions["immediate_score"].astype(np.float32)[:, None] / 100.0,
            actions["immediate_deltas"].astype(np.float32) / 20.0,
        ],
        axis=-1,
    )
    if action.shape[1] != GRADED_ACTION_DIM:
        raise CensusError("graded action decoder dimension drifted")
    slices = {
        "same_slot_independent": (0, 1),
        "draft_kind": (1, 3),
        "tile_slot": (3, 7),
        "wildlife_slot": (7, 11),
        "tile_id": (11, 12),
        "tile_terrain_a": (12, 17),
        "tile_terrain_b": (17, 23),
        "tile_wildlife_mask": (23, 28),
        "tile_keystone": (28, 29),
        "drafted_wildlife": (29, 34),
        "tile_coordinates": (34, 36),
        "tile_rotation": (36, 42),
        "wildlife_present": (42, 43),
        "wildlife_coordinates": (43, 45),
        "replace_three": (45, 46),
        "wipe_count": (46, 47),
        "wipe_masks": (47, 127),
        "staged_nature_tokens": (127, 128),
        "immediate_score": (128, 129),
        "immediate_deltas": (129, 140),
    }
    rank = candidates["screen_rank"].astype(np.float32)
    priors = np.concatenate(
        [
            candidates["model_immediate_score"].astype(np.float32)[:, None] / 100.0,
            candidates["model_remaining_value"].astype(np.float32)[:, None] / 100.0,
            candidates["screen_value"].astype(np.float32)[:, None] / 100.0,
            rank[:, None] / 4096.0,
            1.0 / np.maximum(rank[:, None], 1.0),
            candidates["uniform_market_survival_proxy"].astype(np.float32)[:, None],
            candidates["visible_wildlife_count"].astype(np.float32)[:, None] / 4.0,
            candidates["public_bag_wildlife_count"].astype(np.float32)[:, None] / 20.0,
        ],
        axis=-1,
    )
    staged_market = _decode_market(actions["staged_market_entities"]).reshape(len(actions), -1)
    supply_scales = np.asarray([20.0] * 5 + [81.0] * 25, dtype=np.float32)
    staged_supply = actions["staged_public_supply"].astype(np.float32) / supply_scales
    result = {
        f"graded.action.{name}": action[:, left:right]
        for name, (left, right) in slices.items()
    }
    result["graded.prior.observable"] = priors
    result["graded.staged_market"] = staged_market
    result["graded.staged_public_supply"] = staged_supply
    return result


def _empty_phase_seat() -> dict[str, dict[str, dict[str, int]]]:
    return {
        phase: {
            seat: {"rows": 0, "active_rows": 0, "nonzero_values": 0, "values": 0}
            for seat in SEATS
        }
        for phase in PHASE_LABELS
    }


class BlockAccumulator:
    """Mergeable per-channel activation statistics for one fixed-width block."""

    def __init__(
        self,
        spec: BlockSpec,
        *,
        rare_threshold: float,
        exact_alias_cell_limit: int,
    ):
        self.spec = spec
        self.rare_threshold = rare_threshold
        self.exact_alias_cell_limit = exact_alias_cell_limit
        self.rows = 0
        self.active_rows = 0
        self.nonzero_values = 0
        self.values = 0
        self.channel_nonzero = np.zeros(spec.width, dtype=np.uint64)
        self.channel_min = np.full(spec.width, np.inf, dtype=np.float64)
        self.channel_max = np.full(spec.width, -np.inf, dtype=np.float64)
        self.channel_hashers = [blake3.blake3() for _ in range(spec.width)]
        self.phase_seat = _empty_phase_seat()
        self._exact_chunks: list[np.ndarray] | None = []

    def update(
        self,
        values: np.ndarray,
        *,
        phases: Sequence[str] | str,
        seats: Sequence[str] | str,
    ) -> None:
        array = np.asarray(values)
        if array.ndim != 2 or array.shape[1] != self.spec.width:
            raise CensusError(
                f"{self.spec.block_id} expected width {self.spec.width}, got {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise CensusError(f"{self.spec.block_id} contains non-finite values")
        rows = len(array)
        if rows == 0:
            return
        array = np.ascontiguousarray(array, dtype=np.float32)
        nonzero = array != 0
        active = np.any(nonzero, axis=1)
        self.rows += rows
        self.active_rows += int(np.count_nonzero(active))
        self.nonzero_values += int(np.count_nonzero(nonzero))
        self.values += int(array.size)
        self.channel_nonzero += np.count_nonzero(nonzero, axis=0).astype(np.uint64)
        self.channel_min = np.minimum(self.channel_min, np.min(array, axis=0))
        self.channel_max = np.maximum(self.channel_max, np.max(array, axis=0))
        for index, hasher in enumerate(self.channel_hashers):
            hasher.update(np.ascontiguousarray(array[:, index]).tobytes())

        phase_values = [phases] * rows if isinstance(phases, str) else list(phases)
        seat_values = [seats] * rows if isinstance(seats, str) else list(seats)
        if len(phase_values) != rows or len(seat_values) != rows:
            raise CensusError("phase/seat metadata length does not match feature rows")
        for phase in PHASE_LABELS:
            phase_mask = np.asarray([value == phase for value in phase_values])
            if not np.any(phase_mask):
                continue
            for seat in SEATS:
                mask = phase_mask & np.asarray([value == seat for value in seat_values])
                count = int(np.count_nonzero(mask))
                if not count:
                    continue
                cell = self.phase_seat[phase][seat]
                cell["rows"] += count
                cell["active_rows"] += int(np.count_nonzero(active[mask]))
                cell["nonzero_values"] += int(np.count_nonzero(nonzero[mask]))
                cell["values"] += count * self.spec.width

        if self._exact_chunks is not None:
            if (self.rows * self.spec.width) <= self.exact_alias_cell_limit:
                self._exact_chunks.append(array.copy())
            else:
                self._exact_chunks = None

    def update_repeated(
        self,
        values: np.ndarray,
        count: int,
        *,
        phase: str,
        seat: str,
    ) -> None:
        if count <= 0:
            return
        row = np.asarray(values, dtype=np.float32).reshape(1, -1)
        if row.shape[1] != self.spec.width:
            raise CensusError(f"{self.spec.block_id} repeated row width drifted")
        chunk = min(count, DEFAULT_BATCH_ROWS)
        remaining = count
        while remaining:
            take = min(remaining, chunk)
            self.update(np.broadcast_to(row, (take, self.spec.width)), phases=phase, seats=seat)
            remaining -= take

    def merge(self, other: Mapping[str, Any]) -> None:
        if int(other["width"]) != self.spec.width:
            raise CensusError(f"cannot merge width drift for {self.spec.block_id}")
        rows = int(other["rows"])
        self.rows += rows
        self.active_rows += int(other["active_rows"])
        self.nonzero_values += int(other["nonzero_values"])
        self.values += int(other["values"])
        self.channel_nonzero += np.asarray(other["channel_nonzero"], dtype=np.uint64)
        if rows:
            self.channel_min = np.minimum(
                self.channel_min, np.asarray(other["channel_min"], dtype=np.float64)
            )
            self.channel_max = np.maximum(
                self.channel_max, np.asarray(other["channel_max"], dtype=np.float64)
            )
        for phase in PHASE_LABELS:
            for seat in SEATS:
                for key in ("rows", "active_rows", "nonzero_values", "values"):
                    self.phase_seat[phase][seat][key] += int(
                        other["phase_seat"][phase][seat][key]
                    )
        self._exact_chunks = None
        self.channel_hashers = []

    def raw_state(self) -> dict[str, Any]:
        minimum = np.where(np.isfinite(self.channel_min), self.channel_min, 0.0)
        maximum = np.where(np.isfinite(self.channel_max), self.channel_max, 0.0)
        return {
            "block_id": self.spec.block_id,
            "width": self.spec.width,
            "rows": self.rows,
            "active_rows": self.active_rows,
            "nonzero_values": self.nonzero_values,
            "values": self.values,
            "channel_nonzero": self.channel_nonzero.astype(int).tolist(),
            "channel_min": minimum.tolist(),
            "channel_max": maximum.tolist(),
            "channel_blake3": [hasher.hexdigest() for hasher in self.channel_hashers],
            "phase_seat": self.phase_seat,
        }

    def finish(self) -> dict[str, Any]:
        state = self.raw_state()
        rows = max(self.rows, 1)
        dead = np.flatnonzero(self.channel_nonzero == 0)
        constant = np.flatnonzero(
            np.isfinite(self.channel_min) & (self.channel_min == self.channel_max)
        )
        rates = self.channel_nonzero.astype(np.float64) / rows
        rare = np.flatnonzero((rates > 0) & (rates < self.rare_threshold))
        hash_groups: dict[str, list[int]] = defaultdict(list)
        if self.channel_hashers:
            for index, hasher in enumerate(self.channel_hashers):
                hash_groups[hasher.hexdigest()].append(index)
        sketch_aliases = [indices for indices in hash_groups.values() if len(indices) > 1]
        exact_aliases: list[list[int]] = []
        if self._exact_chunks is not None and self._exact_chunks:
            exact = np.concatenate(self._exact_chunks, axis=0)
            for group in sketch_aliases:
                remaining = set(group)
                while remaining:
                    first = min(remaining)
                    aliases = [
                        index
                        for index in sorted(remaining)
                        if np.array_equal(exact[:, first], exact[:, index])
                    ]
                    remaining.difference_update(aliases)
                    if len(aliases) > 1:
                        exact_aliases.append(aliases)
        alias_mode = (
            "exact"
            if self._exact_chunks is not None
            else ("deterministic_blake3_sketch" if self.channel_hashers else "merge_unknown")
        )
        phase_seat = {}
        for phase in PHASE_LABELS:
            phase_seat[phase] = {}
            for seat in SEATS:
                cell = self.phase_seat[phase][seat]
                phase_seat[phase][seat] = {
                    **cell,
                    "active_row_rate": cell["active_rows"] / cell["rows"] if cell["rows"] else None,
                    "value_activation_rate": (
                        cell["nonzero_values"] / cell["values"] if cell["values"] else None
                    ),
                }
        status = []
        if len(dead) == self.spec.width and self.spec.width:
            status.append("dead")
        elif len(constant) == self.spec.width and self.spec.width:
            status.append("constant")
        if len(rare):
            status.append("rare_channels_present")
        if exact_aliases:
            status.append("empirical_aliases_present")
        elif sketch_aliases:
            status.append("sketch_alias_candidates_present")
        status.extend(self.spec.known_status)
        if not status:
            status.append("active_no_detected_issue")
        return {
            "block_id": self.spec.block_id,
            "schema": self.spec.schema,
            "name": self.spec.name,
            "width": self.spec.width,
            "rows": self.rows,
            "active_rows": self.active_rows,
            "active_row_rate": self.active_rows / self.rows if self.rows else None,
            "nonzero_values": self.nonzero_values,
            "values": self.values,
            "value_activation_rate": self.nonzero_values / self.values if self.values else None,
            "dead_channel_count": len(dead),
            "constant_channel_count": len(constant),
            "rare_channel_count": len(rare),
            "dead_channels": dead.astype(int).tolist(),
            "constant_channels": constant.astype(int).tolist(),
            "rare_channels": rare.astype(int).tolist(),
            "alias_analysis": {
                "mode": alias_mode,
                "exact_alias_groups": exact_aliases,
                "sketch_alias_groups": sketch_aliases if alias_mode != "exact" else [],
                "structural_status": list(self.spec.known_status),
            },
            "collision_status": (
                "structural_or_empirical_alias"
                if exact_aliases or self.spec.known_status
                else ("unknown" if alias_mode == "merge_unknown" else "no_channel_alias_detected")
            ),
            "status": sorted(set(status)),
            "phase_seat": phase_seat,
            "evidence_provenance": list(self.spec.evidence),
            "_merge": state,
        }


class LegacyAccumulator:
    """Sparse binary activation census for historical NNUE index streams."""

    def __init__(
        self,
        specs: Sequence[BlockSpec],
        rare_threshold: float,
        *,
        feature_count: int,
    ):
        self.specs = tuple(specs)
        self.rare_threshold = rare_threshold
        self.rows = 0
        self.counts = np.zeros(feature_count, dtype=np.uint64)
        self.phase_seat_block_rows = {
            spec.block_id: _empty_phase_seat() for spec in self.specs
        }

    def update(self, indices: Sequence[int], phase: str, seat: str) -> None:
        values = np.asarray(sorted(set(int(index) for index in indices)), dtype=np.int64)
        if np.any(values < 0) or np.any(values >= len(self.counts)):
            raise CensusError("legacy sparse feature index crosses schema boundary")
        self.rows += 1
        self.counts[values] += 1
        for spec in self.specs:
            start = int(spec.ownership["start"])
            stop = int(spec.ownership["stop"])
            active = values[(values >= start) & (values < stop)]
            cell = self.phase_seat_block_rows[spec.block_id][phase][seat]
            cell["rows"] += 1
            cell["active_rows"] += int(len(active) > 0)
            cell["nonzero_values"] += len(active)
            cell["values"] += spec.width

    def finish(self) -> list[dict[str, Any]]:
        results = []
        for spec in self.specs:
            start = int(spec.ownership["start"])
            stop = int(spec.ownership["stop"])
            counts = self.counts[start:stop]
            rows = max(self.rows, 1)
            dead = np.flatnonzero(counts == 0)
            constant = np.flatnonzero(counts == self.rows) if self.rows else np.array([], dtype=int)
            rare = np.flatnonzero((counts > 0) & ((counts / rows) < self.rare_threshold))
            active_rows_upper = min(self.rows, int(np.sum(counts)))
            status = list(spec.known_status)
            if len(dead) == spec.width:
                status.append("dead")
            if len(constant) == spec.width and spec.width:
                status.append("constant")
            if len(rare):
                status.append("rare_channels_present")
            if not status:
                status.append("active_no_detected_issue")
            phase_seat = {}
            for phase in PHASE_LABELS:
                phase_seat[phase] = {}
                for seat in SEATS:
                    cell = self.phase_seat_block_rows[spec.block_id][phase][seat]
                    phase_seat[phase][seat] = {
                        **cell,
                        "active_row_rate": (
                            cell["active_rows"] / cell["rows"] if cell["rows"] else None
                        ),
                        "value_activation_rate": (
                            cell["nonzero_values"] / cell["values"]
                            if cell["values"]
                            else None
                        ),
                    }
            results.append(
                {
                    "block_id": spec.block_id,
                    "schema": spec.schema,
                    "name": spec.name,
                    "width": spec.width,
                    "rows": self.rows,
                    "active_rows": active_rows_upper,
                    "active_row_rate": active_rows_upper / self.rows if self.rows else None,
                    "nonzero_values": int(np.sum(counts)),
                    "values": self.rows * spec.width,
                    "value_activation_rate": (
                        int(np.sum(counts)) / (self.rows * spec.width)
                        if self.rows and spec.width
                        else None
                    ),
                    "dead_channel_count": len(dead),
                    "constant_channel_count": len(constant),
                    "rare_channel_count": len(rare),
                    "dead_channels": dead.astype(int).tolist(),
                    "constant_channels": constant.astype(int).tolist(),
                    "rare_channels": rare.astype(int).tolist(),
                    "alias_analysis": {
                        "mode": "bounded_sparse_activation_sketch",
                        "exact_alias_groups": [],
                        "sketch_alias_groups": [],
                        "structural_status": list(spec.known_status),
                    },
                    "collision_status": "unknown",
                    "status": sorted(set(status)),
                    "phase_seat": phase_seat,
                    "evidence_provenance": list(spec.evidence),
                    "_merge": {
                        "block_id": spec.block_id,
                        "width": spec.width,
                        "rows": self.rows,
                        "active_rows": active_rows_upper,
                        "nonzero_values": int(np.sum(counts)),
                        "values": self.rows * spec.width,
                        "channel_nonzero": counts.astype(int).tolist(),
                        "channel_min": [0.0] * spec.width,
                        "channel_max": [1.0 if count else 0.0 for count in counts],
                        "phase_seat": phase_seat,
                    },
                }
            )
        return results


class CollisionTracker:
    """Bounded deterministic row-collision detector with byte verification."""

    def __init__(self, *, sample_modulus: int, signature_limit: int):
        if sample_modulus <= 0 or signature_limit <= 0:
            raise CensusError("collision sampling bounds must be positive")
        self.sample_modulus = sample_modulus
        self.signature_limit = signature_limit
        self.rows = 0
        self.sampled_rows = 0
        self.collisions = 0
        self.hash_collisions_rejected = 0
        self.truncated = False
        self._seen: dict[str, tuple[bytes, bytes]] = {}
        self.examples: list[dict[str, str]] = []

    def update(self, features: np.ndarray, identities: np.ndarray) -> None:
        array = np.ascontiguousarray(features, dtype="<f4")
        identity = np.ascontiguousarray(identities, dtype=np.uint8)
        if array.ndim != 2 or identity.ndim != 2 or len(array) != len(identity):
            raise CensusError("collision tracker row shape mismatch")
        self.rows += len(array)
        words = array.view("<u4").reshape(len(array), -1)
        multipliers = (
            np.arange(1, words.shape[1] + 1, dtype=np.uint64) * np.uint64(0x9E3779B185EBCA87)
        ) | np.uint64(1)
        cheap = np.bitwise_xor.reduce(words.astype(np.uint64) * multipliers, axis=1)
        selected = np.flatnonzero((cheap % self.sample_modulus) == 0)
        for row_index in selected:
            payload = array[row_index].tobytes()
            row_identity = identity[row_index].tobytes()
            signature = blake3.blake3(payload).hexdigest()
            self.sampled_rows += 1
            prior = self._seen.get(signature)
            if prior is None:
                if len(self._seen) >= self.signature_limit:
                    self.truncated = True
                    continue
                self._seen[signature] = (payload, row_identity)
                continue
            prior_payload, prior_identity = prior
            if prior_payload != payload:
                self.hash_collisions_rejected += 1
                continue
            if prior_identity != row_identity:
                self.collisions += 1
                if len(self.examples) < 20:
                    self.examples.append(
                        {
                            "feature_blake3": signature,
                            "first_identity": prior_identity.hex(),
                            "second_identity": row_identity.hex(),
                        }
                    )

    def finish(self) -> dict[str, Any]:
        return {
            "mode": "deterministic_feature_fingerprint_sample_with_byte_verification",
            "sample_modulus": self.sample_modulus,
            "signature_limit": self.signature_limit,
            "rows": self.rows,
            "sampled_rows": self.sampled_rows,
            "sample_rate": self.sampled_rows / self.rows if self.rows else None,
            "verified_representation_collisions": self.collisions,
            "cryptographic_hash_collisions_rejected": self.hash_collisions_rejected,
            "signature_table_truncated": self.truncated,
            "examples": self.examples,
            "status": (
                "verified_collisions_present"
                if self.collisions
                else ("bounded_unknown_due_to_truncation" if self.truncated else "none_in_sample")
            ),
        }


class Census:
    def __init__(
        self,
        *,
        rare_threshold: float,
        exact_alias_cell_limit: int,
        collision_sample_modulus: int,
        collision_signature_limit: int,
    ):
        self.specs = {spec.block_id: spec for spec in block_specs()}
        self.accumulators: dict[str, BlockAccumulator] = {}
        self.legacy_results: list[dict[str, Any]] = []
        self.evidence: list[dict[str, Any]] = []
        self.provenance: list[dict[str, Any]] = []
        self.rare_threshold = rare_threshold
        self.exact_alias_cell_limit = exact_alias_cell_limit
        self.collisions = {
            "graded_candidate_bundle": CollisionTracker(
                sample_modulus=collision_sample_modulus,
                signature_limit=collision_signature_limit,
            ),
            "factor_candidate_bundle": CollisionTracker(
                sample_modulus=collision_sample_modulus,
                signature_limit=collision_signature_limit,
            ),
        }

    def accumulator(self, block_id: str) -> BlockAccumulator:
        if block_id not in self.accumulators:
            spec = self.specs[block_id]
            self.accumulators[block_id] = BlockAccumulator(
                spec,
                rare_threshold=self.rare_threshold,
                exact_alias_cell_limit=self.exact_alias_cell_limit,
            )
        return self.accumulators[block_id]

    def update_blocks(
        self,
        blocks: Mapping[str, np.ndarray],
        *,
        phases: Sequence[str] | str,
        seats: Sequence[str] | str,
    ) -> None:
        for block_id, values in blocks.items():
            self.accumulator(block_id).update(values, phases=phases, seats=seats)

    def finish(self, config: dict[str, Any]) -> dict[str, Any]:
        manifest = build_manifest()
        schema_identities = _block_schema_identities(manifest)
        measured = [accumulator.finish() for accumulator in self.accumulators.values()]
        measured.extend(self.legacy_results)
        measured_by_id = {entry["block_id"]: entry for entry in measured}
        blocks = []
        for spec in block_specs():
            entry = spec.to_dict()
            entry["schema_version"], entry["schema_blake3"] = schema_identities[spec.block_id]
            if spec.block_id in measured_by_id:
                entry["census"] = measured_by_id[spec.block_id]
            else:
                entry["census"] = {
                    "status": (
                        ["unimplemented_unmeasurable"]
                        if spec.implementation_status != "implemented"
                        else ["not_measured_in_this_shard"]
                    ),
                    "rows": 0,
                    "collision_status": "unknown",
                }
            blocks.append(entry)
        scientific = {
            "report_schema_version": REPORT_SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "manifest_scientific_blake3": manifest["scientific_blake3"],
            "config": config,
            "evidence": sorted(self.evidence, key=lambda value: value["evidence_id"]),
            "blocks": blocks,
            "representation_collisions": {
                name: tracker.finish() for name, tracker in self.collisions.items()
            },
            "closed_domains": {
                "test_split_opened": False,
                "gameplay_opened": False,
                "new_teacher_compute_used": False,
                "external_compute_used": False,
                "hidden_teacher_values_used_as_features": False,
            },
        }
        return {
            "scientific": scientific,
            "scientific_blake3": scientific_blake3(scientific),
            "provenance": {
                "input_manifests": sorted(
                    self.provenance, key=lambda value: value["evidence_id"]
                )
            },
        }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CensusError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise CensusError(f"JSON root must be an object: {path}")
    return value


def _selected(index: int, count: int, key: str) -> bool:
    if count <= 0 or not 0 <= index < count:
        raise CensusError("invalid shard selection")
    digest = blake3.blake3(key.encode()).digest(length=8)
    return int.from_bytes(digest, "little") % count == index


def _validate_split(split: str) -> None:
    if split not in {"train", "validation"}:
        raise CensusError(f"only open train/validation splits are allowed, got {split!r}")


def _position_evidence_id(manifest: Mapping[str, Any], entry: Mapping[str, Any]) -> str:
    return f"position:{manifest.get('dataset_id', 'unknown')}:{entry['file']}"


def _graded_evidence_id(manifest: Mapping[str, Any], entry: Mapping[str, Any]) -> str:
    return f"graded:{manifest.get('dataset_id', 'unknown')}:{entry['file']}"


def scan_dataset_root(
    census: Census,
    root: Path,
    *,
    expected_split: str | None,
    row_limit: int | None,
    batch_rows: int,
    shard_index: int,
    shard_count: int,
) -> None:
    manifest_path = root / "dataset.json"
    manifest = _load_json(manifest_path)
    _validate_split(str(manifest.get("split")))
    if expected_split is not None and manifest.get("split") != expected_split:
        raise CensusError(
            f"dataset root {root} is {manifest.get('split')!r}, expected {expected_split!r}"
        )
    feature_schema = manifest.get("feature_schema")
    manifest_file_blake3 = checksum(manifest_path)
    manifest_identity_blake3 = manifest_scientific_blake3(manifest)
    if feature_schema == POSITION_FEATURE_SCHEMA:
        _scan_position_dataset(
            census,
            root,
            manifest,
            manifest_identity_blake3=manifest_identity_blake3,
            manifest_file_blake3=manifest_file_blake3,
            row_limit=row_limit,
            batch_rows=batch_rows,
            shard_index=shard_index,
            shard_count=shard_count,
        )
    elif feature_schema == GRADED_FEATURE_SCHEMA:
        _scan_graded_dataset(
            census,
            root,
            manifest,
            manifest_identity_blake3=manifest_identity_blake3,
            manifest_file_blake3=manifest_file_blake3,
            row_limit=row_limit,
            batch_rows=batch_rows,
            shard_index=shard_index,
            shard_count=shard_count,
        )
    else:
        raise CensusError(f"unsupported dataset feature schema {feature_schema!r}")


def _scan_position_dataset(
    census: Census,
    root: Path,
    manifest: Mapping[str, Any],
    *,
    manifest_identity_blake3: str,
    manifest_file_blake3: str,
    row_limit: int | None,
    batch_rows: int,
    shard_index: int,
    shard_count: int,
) -> None:
    required = {
        "schema_version": 1,
        "feature_schema": POSITION_FEATURE_SCHEMA,
        "target_schema": POSITION_TARGET_SCHEMA,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise CensusError(f"position manifest {key} drifted")
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise CensusError("position manifest shards must be a list")
    if sum(int(entry["record_count"]) for entry in shards) != int(manifest["total_records"]):
        raise CensusError("position manifest record total drifted")
    remaining = row_limit
    for entry in shards:
        evidence_id = _position_evidence_id(manifest, entry)
        if not _selected(shard_index, shard_count, evidence_id):
            continue
        path = root / str(entry["file"])
        if path.stat().st_size != int(entry["byte_count"]) or checksum(path) != entry["blake3"]:
            raise CensusError(f"position shard checksum/size mismatch: {path}")
        with path.open("rb") as handle:
            header = handle.read(POSITION_HEADER_SIZE)
        (
            magic,
            schema,
            header_size,
            record_size,
            target_dim,
            record_count,
            game_count,
            first_game_index,
            _split,
            _strategy,
            _players,
            _bonuses,
            _cards,
            _reserved,
            feature_hash,
        ) = _POSITION_HEADER.unpack(header)
        if (
            magic != POSITION_MAGIC
            or schema != 1
            or header_size != POSITION_HEADER_SIZE
            or record_size != POSITION_RECORD_SIZE
            or target_dim != 11
            or feature_hash != blake3.blake3(POSITION_FEATURE_SCHEMA.encode()).digest()
            or record_count != int(entry["record_count"])
            or game_count != int(entry["game_count"])
            or first_game_index != int(entry["first_game_index"])
        ):
            raise CensusError(f"position shard header mismatch: {path}")
        take_total = record_count if remaining is None else min(record_count, remaining)
        records = np.memmap(
            path,
            mode="r",
            dtype=_POSITION_DTYPE,
            offset=POSITION_HEADER_SIZE,
            shape=(record_count,),
        )
        for start in range(0, take_total, batch_rows):
            batch = np.asarray(records[start : min(start + batch_rows, take_total)])
            phases = [_phase_from_position(record) for record in batch]
            seats = [str(int(record["active_seat"])) for record in batch]
            census.update_blocks(decode_position_blocks(batch), phases=phases, seats=seats)
        census.evidence.append(
            {
                "evidence_id": evidence_id,
                "kind": "position_dataset_shard",
                "split": manifest["split"],
                "manifest_scientific_blake3": manifest_identity_blake3,
                "payload_blake3": entry["blake3"],
                "rows_available": record_count,
                "rows_scanned": take_total,
            }
        )
        census.provenance.append(
            {
                "evidence_id": evidence_id,
                "manifest_file_blake3": manifest_file_blake3,
            }
        )
        if remaining is not None:
            remaining -= take_total
            if remaining <= 0:
                break


def _scan_graded_dataset(
    census: Census,
    root: Path,
    manifest: Mapping[str, Any],
    *,
    manifest_identity_blake3: str,
    manifest_file_blake3: str,
    row_limit: int | None,
    batch_rows: int,
    shard_index: int,
    shard_count: int,
) -> None:
    required = {
        "schema_version": 1,
        "feature_schema": GRADED_FEATURE_SCHEMA,
        "position_feature_schema": POSITION_FEATURE_SCHEMA,
        "target_schema": GRADED_TARGET_SCHEMA,
        "group_header_size": GRADED_GROUP_HEADER_SIZE,
        "candidate_record_size": GRADED_CANDIDATE_SIZE,
        "action_feature_size": GRADED_ACTION_STORAGE_SIZE,
        "public_supply_size": GRADED_PUBLIC_SUPPLY_SIZE,
        "maximum_wildlife_wipes": GRADED_MAX_WIPES,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise CensusError(f"graded manifest {key} drifted")
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise CensusError("graded manifest shards must be a list")
    if sum(int(entry["record_count"]) for entry in shards) != int(manifest["total_records"]):
        raise CensusError("graded manifest record total drifted")
    remaining = row_limit
    supply_scales = np.asarray([20.0] * 5 + [81.0] * 25, dtype=np.float32)
    for entry in shards:
        evidence_id = _graded_evidence_id(manifest, entry)
        if not _selected(shard_index, shard_count, evidence_id):
            continue
        path = root / str(entry["file"])
        if path.stat().st_size != int(entry["byte_count"]) or checksum(path) != entry["blake3"]:
            raise CensusError(f"graded shard checksum/size mismatch: {path}")
        raw = np.memmap(path, mode="r", dtype=np.uint8)
        header = bytes(raw[:GRADED_HEADER_SIZE])
        (
            magic,
            schema,
            header_size,
            group_header_size,
            candidate_size,
            record_count,
            group_count,
            game_count,
            _split,
            _players,
            _bonuses,
            _reserved,
            first_game_index,
            feature_hash,
            target_hash,
            _tail,
        ) = _GRADED_HEADER.unpack(header)
        if (
            magic != GRADED_MAGIC
            or schema != 1
            or header_size != GRADED_HEADER_SIZE
            or group_header_size != GRADED_GROUP_HEADER_SIZE
            or candidate_size != GRADED_CANDIDATE_SIZE
            or feature_hash != blake3.blake3(GRADED_FEATURE_SCHEMA.encode()).digest()
            or target_hash != blake3.blake3(GRADED_TARGET_SCHEMA.encode()).digest()
            or record_count != int(entry["record_count"])
            or group_count != int(entry["group_count"])
            or game_count != int(entry["game_count"])
            or first_game_index != int(entry["first_game_index"])
        ):
            raise CensusError(f"graded shard header mismatch: {path}")
        offset = GRADED_HEADER_SIZE
        scanned = 0
        observed_records = 0
        for _ in range(group_count):
            group = np.frombuffer(raw, dtype=_GRADED_GROUP_DTYPE, count=1, offset=offset)[0]
            count = int(group["candidate_count"])
            if count < 2:
                raise CensusError(f"invalid graded group width: {path}")
            candidate_offset = offset + GRADED_GROUP_HEADER_SIZE
            observed_records += count
            take = count if remaining is None else min(count, remaining)
            if take:
                phase = _phase_from_personal_turn(int(group["personal_turn"]))
                seat = str(int(group["current_player"]))
                parent_blocks = decode_position_blocks(
                    np.asarray([group["position"]], dtype=_POSITION_DTYPE)
                )
                for block_id, values in parent_blocks.items():
                    census.accumulator(block_id).update_repeated(
                        values[0], take, phase=phase, seat=seat
                    )
                supply = group["public_supply"].astype(np.float32) / supply_scales
                census.accumulator("graded.parent_public_supply").update_repeated(
                    supply, take, phase=phase, seat=seat
                )
                candidates = np.frombuffer(
                    raw,
                    dtype=_GRADED_CANDIDATE_DTYPE,
                    count=count,
                    offset=candidate_offset,
                )
                for start in range(0, take, batch_rows):
                    batch = np.asarray(candidates[start : min(start + batch_rows, take)])
                    decoded = decode_action_blocks(batch)
                    census.update_blocks(decoded, phases=phase, seats=seat)
                    collision_bundle = np.concatenate(
                        [
                            *[
                                decoded[f"graded.action.{name}"]
                                for name in (
                                    "same_slot_independent",
                                    "draft_kind",
                                    "tile_slot",
                                    "wildlife_slot",
                                    "tile_id",
                                    "tile_terrain_a",
                                    "tile_terrain_b",
                                    "tile_wildlife_mask",
                                    "tile_keystone",
                                    "drafted_wildlife",
                                    "tile_coordinates",
                                    "tile_rotation",
                                    "wildlife_present",
                                    "wildlife_coordinates",
                                    "replace_three",
                                    "wipe_count",
                                    "wipe_masks",
                                    "staged_nature_tokens",
                                    "immediate_score",
                                    "immediate_deltas",
                                )
                            ],
                            decoded["graded.prior.observable"],
                            decoded["graded.staged_market"],
                            decoded["graded.staged_public_supply"],
                        ],
                        axis=1,
                    )
                    census.collisions["graded_candidate_bundle"].update(
                        collision_bundle, batch["action_hash"]
                    )
                scanned += take
                if remaining is not None:
                    remaining -= take
            offset = candidate_offset + count * GRADED_CANDIDATE_SIZE
            if remaining is not None and remaining <= 0:
                break
        if remaining is None and (
            offset != path.stat().st_size or observed_records != record_count
        ):
            raise CensusError(f"graded shard framing mismatch: {path}")
        census.evidence.append(
            {
                "evidence_id": evidence_id,
                "kind": "graded_dataset_shard",
                "split": manifest["split"],
                "manifest_scientific_blake3": manifest_identity_blake3,
                "payload_blake3": entry["blake3"],
                "rows_available": record_count,
                "rows_scanned": scanned,
            }
        )
        census.provenance.append(
            {
                "evidence_id": evidence_id,
                "manifest_file_blake3": manifest_file_blake3,
            }
        )
        if remaining is not None and remaining <= 0:
            break


def _factor_payload_blake3(manifest: Mapping[str, Any]) -> str:
    payload = {
        "cache_schema": manifest["cache_schema"],
        "split": manifest["split"],
        "dataset_id": manifest["dataset_id"],
        "dataset_manifest_blake3": manifest["dataset_manifest_blake3"],
        "checkpoint": manifest["checkpoint"],
        "checkpoint_manifest_blake3": manifest["checkpoint_manifest_blake3"],
        "model_blake3": manifest["model_blake3"],
        "factor_names": manifest["factor_names"],
        "factor_count": manifest["factor_count"],
        "factor_dim": manifest["factor_dim"],
        "groups": manifest["groups"],
        "candidates": manifest["candidates"],
        "batches": [
            {
                "index": entry["index"],
                "factors_file": entry["factors_file"],
                "metadata_file": entry["metadata_file"],
                "factors_blake3": entry["factors_blake3"],
                "metadata_blake3": entry["metadata_blake3"],
                "groups": entry["groups"],
                "candidates": entry["candidates"],
            }
            for entry in manifest["batches"]
        ],
    }
    return scientific_blake3(payload)


def scan_factor_cache(
    census: Census,
    root: Path,
    *,
    row_limit: int | None,
    batch_rows: int,
    shard_index: int,
    shard_count: int,
) -> None:
    manifest_path = root / "cache.json"
    manifest = _load_json(manifest_path)
    manifest_file_blake3 = checksum(manifest_path)
    manifest_identity_blake3 = manifest_scientific_blake3(manifest)
    _validate_split(str(manifest.get("split")))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("cache_schema") != FACTOR_CACHE_SCHEMA
        or tuple(manifest.get("factor_names", ())) != FACTOR_NAMES
        or manifest.get("factor_count") != len(FACTOR_NAMES)
        or manifest.get("factor_dim") != FACTOR_DIM
        or manifest.get("features_contain_targets_or_teacher_values") is not False
    ):
        raise CensusError("unsupported or unsafe factor cache")
    if _factor_payload_blake3(manifest) != manifest.get("payload_blake3"):
        raise CensusError("factor cache payload identity drifted")
    if sum(int(entry["candidates"]) for entry in manifest["batches"]) != int(
        manifest["candidates"]
    ):
        raise CensusError("factor cache candidate total drifted")
    remaining = row_limit
    for entry in manifest["batches"]:
        evidence_id = (
            f"factor:{manifest['dataset_id']}:{manifest['split']}:{int(entry['index']):06d}"
        )
        if not _selected(shard_index, shard_count, evidence_id):
            continue
        factors_path = root / str(entry["factors_file"])
        metadata_path = root / str(entry["metadata_file"])
        if (
            checksum(factors_path) != entry["factors_blake3"]
            or checksum(metadata_path) != entry["metadata_blake3"]
        ):
            raise CensusError("candidate factor cache checksum mismatch")
        factors = np.load(factors_path, mmap_mode="r")
        if factors.shape != (int(entry["candidates"]), len(FACTOR_NAMES), FACTOR_DIM):
            raise CensusError("candidate factor cache shape drifted")
        take = len(factors) if remaining is None else min(len(factors), remaining)
        with np.load(metadata_path) as metadata:
            if len(metadata["action_hash"]) != len(factors):
                raise CensusError("factor metadata row count drifted")
            identities = metadata["action_hash"][:take].copy()
        for start in range(0, take, batch_rows):
            right = min(start + batch_rows, take)
            batch = np.asarray(factors[start:right], dtype=np.float32)
            for index, name in enumerate(FACTOR_NAMES):
                census.accumulator(f"factor_cache.{name}").update(
                    batch[:, index, :], phases="unknown", seats="unknown"
                )
            census.collisions["factor_candidate_bundle"].update(
                batch.reshape(len(batch), -1), identities[start:right]
            )
        census.evidence.append(
            {
                "evidence_id": evidence_id,
                "kind": "candidate_factor_cache_batch",
                "split": manifest["split"],
                "manifest_scientific_blake3": manifest_identity_blake3,
                "payload_blake3": entry["factors_blake3"],
                "metadata_blake3": entry["metadata_blake3"],
                "rows_available": len(factors),
                "rows_scanned": take,
                "phase_and_seat_exposed": False,
            }
        )
        census.provenance.append(
            {
                "evidence_id": evidence_id,
                "manifest_file_blake3": manifest_file_blake3,
            }
        )
        if remaining is not None:
            remaining -= take
            if remaining <= 0:
                break


def _hierarchical_scientific_blake3(value: Mapping[str, Any]) -> str:
    scientific = {
        key: item
        for key, item in value.items()
        if key not in {"host", "execution", "cache_file", "payload_blake3"}
    }
    return scientific_blake3(scientific)


def _phase_vector_from_groups(phases: np.ndarray, group_indices: np.ndarray) -> list[str]:
    labels = ("early", "middle", "late")
    return [labels[min(max(int(phases[index]), 0), 2)] for index in group_indices]


def scan_hierarchical_cache(
    census: Census,
    root: Path,
    *,
    row_limit: int | None,
    batch_rows: int,
    shard_index: int,
    shard_count: int,
) -> None:
    manifest_path = root / "manifest.json"
    manifest = _load_json(manifest_path)
    manifest_file_blake3 = checksum(manifest_path)
    manifest_identity_blake3 = manifest_scientific_blake3(manifest)
    _validate_split(str(manifest.get("split")))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("cache_schema") != HIERARCHICAL_CACHE_SCHEMA
        or manifest.get("experiment_id") != HIERARCHICAL_EXPERIMENT_ID
    ):
        raise CensusError("unsupported hierarchical factor cache")
    if _hierarchical_scientific_blake3(manifest) != manifest.get("payload_blake3"):
        raise CensusError("hierarchical cache payload identity drifted")
    hierarchical_specs = [
        spec for spec in block_specs() if spec.schema == HIERARCHICAL_CACHE_SCHEMA
    ]
    arrays_needed = sorted({str(spec.ownership["array"]) for spec in hierarchical_specs})
    remaining_by_array = {
        name: row_limit for name in arrays_needed
    }
    for entry in manifest["shards"]:
        evidence_id = (
            f"hierarchical:{manifest['split']}:{int(entry['shard_index']):02d}:"
            f"{entry['source_blake3']}"
        )
        if not _selected(shard_index, shard_count, evidence_id):
            continue
        path = root / str(entry["cache_file"])
        if checksum(path) != entry["cache_blake3"]:
            raise CensusError("hierarchical factor cache checksum mismatch")
        with np.load(path) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}
        phases = arrays["phase"]
        phase_by_array: dict[str, list[str]] = {
            "group_state": _phase_vector_from_groups(phases, np.arange(len(phases)))
        }
        for stage in ("draft", "tile", "wildlife"):
            query_groups = arrays[f"{stage}_query_group"]
            query_phases = _phase_vector_from_groups(phases, query_groups)
            phase_by_array[f"{stage}_query_context"] = query_phases
            offsets = arrays[f"{stage}_query_offsets"]
            item_group_indices = np.repeat(
                query_groups,
                np.diff(offsets).astype(np.int64),
            )
            phase_by_array[f"{stage}_item_features"] = _phase_vector_from_groups(
                phases, item_group_indices
            )
        takes = {
            name: (
                len(arrays[name])
                if remaining_by_array[name] is None
                else min(len(arrays[name]), int(remaining_by_array[name]))
            )
            for name in arrays_needed
        }
        scanned_for_evidence = 0
        for spec in hierarchical_specs:
            array_name = str(spec.ownership["array"])
            tensor_slice = str(spec.ownership["slice"])
            left, right = (
                int(value)
                for value in tensor_slice.removeprefix("[:,").removesuffix("]").split(":")
            )
            values = arrays[array_name][:, left:right]
            phase_values = phase_by_array[array_name]
            take = takes[array_name]
            for start in range(0, take, batch_rows):
                right = min(start + batch_rows, take)
                census.accumulator(spec.block_id).update(
                    values[start:right],
                    phases=phase_values[start:right],
                    seats="unknown",
                )
        scanned_for_evidence = sum(takes.values())
        census.evidence.append(
            {
                "evidence_id": evidence_id,
                "kind": "hierarchical_factor_cache_shard",
                "split": manifest["split"],
                "manifest_scientific_blake3": manifest_identity_blake3,
                "payload_blake3": entry["cache_blake3"],
                "rows_available": sum(len(arrays[name]) for name in arrays_needed),
                "rows_scanned": scanned_for_evidence,
                "seat_exposed": False,
                "teacher_label_arrays_excluded": True,
            }
        )
        census.provenance.append(
            {
                "evidence_id": evidence_id,
                "manifest_file_blake3": manifest_file_blake3,
            }
        )
        if row_limit is not None:
            for name in arrays_needed:
                remaining_by_array[name] = int(remaining_by_array[name]) - takes[name]
            if all(int(value) <= 0 for value in remaining_by_array.values()):
                break


def scan_legacy_root(
    census: Census,
    root: Path,
    *,
    row_limit: int | None,
    shard_index: int,
    shard_count: int,
) -> None:
    manifest_path = root / "manifest.json"
    manifest = _load_json(manifest_path)
    manifest_file_blake3 = checksum(manifest_path)
    manifest_identity_blake3 = manifest_scientific_blake3(manifest)
    _validate_split(str(manifest.get("split")))
    feature_schema = str(manifest.get("feature_schema"))
    expected_feature_counts = {
        "legacy-nnue-v1-5197": 5_197,
        "legacy-mid-v4opp-11231": 11_231,
    }
    if (
        manifest.get("schema_version") != 1
        or feature_schema not in expected_feature_counts
        or manifest.get("feature_count") != expected_feature_counts[feature_schema]
    ):
        raise CensusError("unsupported legacy sparse feature stream")
    entries = manifest.get("shards")
    if not isinstance(entries, list):
        raise CensusError("legacy manifest shards must be a list")
    if sum(int(entry["row_count"]) for entry in entries) != int(manifest["rows"]):
        raise CensusError("legacy manifest row total drifted")
    included_schemas = {feature_schema}
    if feature_schema == "legacy-mid-v4opp-11231":
        included_schemas.add("legacy-nnue-v1-5197")
    legacy_specs = [spec for spec in block_specs() if spec.schema in included_schemas]
    accumulator = LegacyAccumulator(
        legacy_specs,
        census.rare_threshold,
        feature_count=expected_feature_counts[feature_schema],
    )
    remaining = row_limit
    for entry in entries:
        evidence_id = f"legacy:{manifest.get('dataset_id', 'unknown')}:{entry['file']}"
        if not _selected(shard_index, shard_count, evidence_id):
            continue
        path = root / str(entry["file"])
        if checksum(path) != entry["blake3"]:
            raise CensusError("legacy sparse shard checksum mismatch")
        scanned = 0
        with path.open() as handle:
            for line in handle:
                if remaining is not None and remaining <= 0:
                    break
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise CensusError(f"invalid legacy JSONL row: {error}") from error
                if not isinstance(row, dict) or not isinstance(row.get("features"), list):
                    raise CensusError("legacy JSONL row must contain a feature list")
                phase = row.get("phase")
                if phase is None:
                    phase = _phase_from_personal_turn(int(row["personal_turn"]))
                if phase not in PHASES:
                    raise CensusError("legacy JSONL phase is invalid")
                seat = str(int(row["focal_seat"]))
                if seat not in SEATS[:-1]:
                    raise CensusError("legacy JSONL focal seat is invalid")
                accumulator.update(row["features"], phase, seat)
                scanned += 1
                if remaining is not None:
                    remaining -= 1
        census.evidence.append(
            {
                "evidence_id": evidence_id,
                "kind": "legacy_sparse_feature_shard",
                "split": manifest["split"],
                "manifest_scientific_blake3": manifest_identity_blake3,
                "payload_blake3": entry["blake3"],
                "rows_available": entry["row_count"],
                "rows_scanned": scanned,
            }
        )
        census.provenance.append(
            {
                "evidence_id": evidence_id,
                "manifest_file_blake3": manifest_file_blake3,
            }
        )
        if remaining is not None and remaining <= 0:
            break
    census.legacy_results.extend(accumulator.finish())


def _public_report(report: dict[str, Any]) -> dict[str, Any]:
    return report


def write_details_jsonl(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for block in report["scientific"]["blocks"]:
            census = block.get("census", {})
            merge = census.get("_merge")
            if not merge:
                continue
            rows = max(int(merge["rows"]), 1)
            for index, (nonzero, minimum, maximum) in enumerate(
                zip(
                    merge["channel_nonzero"],
                    merge["channel_min"],
                    merge["channel_max"],
                    strict=True,
                )
            ):
                status = []
                if nonzero == 0:
                    status.append("dead")
                if minimum == maximum:
                    status.append("constant")
                rate = nonzero / rows
                if 0 < rate < report["scientific"]["config"]["rare_threshold"]:
                    status.append("rare")
                handle.write(
                    json.dumps(
                        {
                            "block_id": block["block_id"],
                            "channel": index,
                            "rows": merge["rows"],
                            "nonzero": nonzero,
                            "activation_rate": rate,
                            "minimum": minimum,
                            "maximum": maximum,
                            "status": status or ["active"],
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )


def merge_reports(reports: Sequence[Path], *, require_shards: int | None = None) -> dict[str, Any]:
    if not reports:
        raise CensusError("merge requires at least one report")
    values = [_load_json(path) for path in reports]
    for value in values:
        if scientific_blake3(value["scientific"]) != value.get("scientific_blake3"):
            raise CensusError("input report scientific BLAKE3 mismatch")
        if value["scientific"].get("experiment_id") != EXPERIMENT_ID:
            raise CensusError("input report experiment identity drifted")
    evidence_ids: set[str] = set()
    for value in values:
        for evidence in value["scientific"]["evidence"]:
            evidence_id = str(evidence["evidence_id"])
            if evidence_id in evidence_ids:
                raise CensusError(f"duplicate evidence in merge: {evidence_id}")
            evidence_ids.add(evidence_id)
    shard_pairs = {
        (
            int(value["scientific"]["config"]["shard_index"]),
            int(value["scientific"]["config"]["shard_count"]),
        )
        for value in values
    }
    if require_shards is not None:
        expected = {(index, require_shards) for index in range(require_shards)}
        if shard_pairs != expected:
            raise CensusError(f"merge requires exact shard set {sorted(expected)}")

    manifest = build_manifest()
    schema_identities = _block_schema_identities(manifest)
    specs = {spec.block_id: spec for spec in block_specs()}
    rare_threshold = float(values[0]["scientific"]["config"]["rare_threshold"])
    merged_accumulators: dict[str, BlockAccumulator] = {}
    local_blocks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for value in values:
        config = value["scientific"]["config"]
        if float(config["rare_threshold"]) != rare_threshold:
            raise CensusError("cannot merge reports with different rare thresholds")
        for block in value["scientific"]["blocks"]:
            state = block.get("census", {}).get("_merge")
            if not state:
                continue
            block_id = block["block_id"]
            local_blocks[block_id].append(block)
            accumulator = merged_accumulators.setdefault(
                block_id,
                BlockAccumulator(
                    specs[block_id],
                    rare_threshold=rare_threshold,
                    exact_alias_cell_limit=0,
                ),
            )
            accumulator.merge(state)
    measured = {
        block_id: accumulator.finish()
        for block_id, accumulator in merged_accumulators.items()
    }
    for block_id, result in measured.items():
        local = local_blocks[block_id]
        states = [block["census"]["_merge"] for block in local]
        if not states or not all(state.get("channel_blake3") for state in states):
            continue
        width = int(result["width"])
        fingerprints = [
            tuple(str(state["channel_blake3"][channel]) for state in states)
            for channel in range(width)
        ]
        groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for channel, fingerprint in enumerate(fingerprints):
            groups[fingerprint].append(channel)
        candidates = [channels for channels in groups.values() if len(channels) > 1]
        all_exact = all(
            block["census"]["alias_analysis"]["mode"] == "exact" for block in local
        )
        exact_groups: list[list[int]] = []
        sketch_groups = candidates
        if all_exact:
            local_partitions = []
            for block in local:
                partition = {
                    channel: group
                    for group in block["census"]["alias_analysis"]["exact_alias_groups"]
                    for channel in group
                }
                local_partitions.append(partition)
            exact_groups = [
                group
                for group in candidates
                if all(
                    all(partition.get(channel) == group for channel in group)
                    for partition in local_partitions
                )
            ]
            sketch_groups = [group for group in candidates if group not in exact_groups]
        result["alias_analysis"] = {
            "mode": (
                "exact_per_shard_intersection"
                if all_exact
                else "deterministic_per_shard_blake3_intersection"
            ),
            "exact_alias_groups": exact_groups,
            "sketch_alias_groups": sketch_groups,
            "structural_status": list(specs[block_id].known_status),
        }
        result["_merge"]["channel_blake3"] = [
            scientific_blake3(sorted(fingerprint)) for fingerprint in fingerprints
        ]
        status = set(result["status"])
        status.discard("active_no_detected_issue")
        if exact_groups:
            status.add("empirical_aliases_present")
        if sketch_groups:
            status.add("sketch_alias_candidates_present")
        if not status:
            status.add("active_no_detected_issue")
        result["status"] = sorted(status)
        result["collision_status"] = (
            "structural_or_empirical_alias"
            if exact_groups or specs[block_id].known_status
            else (
                "unknown_hash_candidate"
                if sketch_groups
                else "no_channel_alias_detected"
            )
        )
    blocks = []
    for spec in block_specs():
        entry = spec.to_dict()
        entry["schema_version"], entry["schema_blake3"] = schema_identities[spec.block_id]
        entry["census"] = measured.get(
            spec.block_id,
            {
                "status": (
                    ["unimplemented_unmeasurable"]
                    if spec.implementation_status != "implemented"
                    else ["not_measured_in_merged_shards"]
                ),
                "rows": 0,
                "collision_status": "unknown",
            },
        )
        blocks.append(entry)
    config = {
        key: value
        for key, value in values[0]["scientific"]["config"].items()
        if key not in {"shard_index", "shard_count", "row_limit"}
    }
    config["merged_shards"] = sorted([list(pair) for pair in shard_pairs])
    scientific = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "manifest_scientific_blake3": manifest["scientific_blake3"],
        "config": config,
        "evidence": sorted(
            [
                evidence
                for value in values
                for evidence in value["scientific"]["evidence"]
            ],
            key=lambda evidence: evidence["evidence_id"],
        ),
        "blocks": blocks,
        "representation_collisions": {
            "merge_status": "per_shard_byte_verified_cross_shard_unknown",
            "cross_shard_status": (
                "unknown: identical representations split across source-evidence shards "
                "are not byte-compared during merge"
            ),
            "inputs": [
                value["scientific"]["representation_collisions"]
                for value in sorted(values, key=lambda item: item["scientific_blake3"])
            ],
        },
        "closed_domains": {
            "test_split_opened": False,
            "gameplay_opened": False,
            "new_teacher_compute_used": False,
            "external_compute_used": False,
            "hidden_teacher_values_used_as_features": False,
        },
    }
    provenance = sorted(
        [
            item
            for value in values
            for item in value.get("provenance", {}).get("input_manifests", [])
        ],
        key=lambda item: (item["evidence_id"], item["manifest_file_blake3"]),
    )
    return {
        "scientific": scientific,
        "scientific_blake3": scientific_blake3(scientific),
        "provenance": {"input_manifests": provenance},
    }


def _add_common_census_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-root", action="append", type=Path, default=[])
    parser.add_argument("--validation-root", action="append", type=Path, default=[])
    parser.add_argument("--dataset-root", action="append", type=Path, default=[])
    parser.add_argument("--legacy-root", action="append", type=Path, default=[])
    parser.add_argument("--factor-cache-root", action="append", type=Path, default=[])
    parser.add_argument("--hierarchical-cache-root", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--details-jsonl", type=Path)
    parser.add_argument("--row-limit", type=int)
    parser.add_argument("--batch-rows", type=int, default=DEFAULT_BATCH_ROWS)
    parser.add_argument("--rare-threshold", type=float, default=DEFAULT_RARE_THRESHOLD)
    parser.add_argument(
        "--collision-sample-modulus",
        type=int,
        default=DEFAULT_COLLISION_SAMPLE_MODULUS,
    )
    parser.add_argument(
        "--collision-signature-limit",
        type=int,
        default=DEFAULT_COLLISION_SIGNATURE_LIMIT,
    )
    parser.add_argument(
        "--exact-alias-cell-limit",
        type=int,
        default=DEFAULT_EXACT_ALIAS_CELL_LIMIT,
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    manifest = subparsers.add_parser("manifest", help="emit the deterministic schema manifest")
    manifest.add_argument("--output", type=Path, required=True)
    census = subparsers.add_parser("census", help="stream open data and emit a census report")
    _add_common_census_arguments(census)
    merge = subparsers.add_parser("merge", help="merge disjoint shard reports")
    merge.add_argument("--input", action="append", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--details-jsonl", type=Path)
    merge.add_argument("--require-shards", type=int)
    return parser


def _run_census(args: argparse.Namespace) -> dict[str, Any]:
    if args.row_limit is not None and args.row_limit <= 0:
        raise CensusError("row limit must be positive")
    if args.batch_rows <= 0:
        raise CensusError("batch rows must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        raise CensusError("shard index must be within shard count")
    if not 0 <= args.rare_threshold < 1:
        raise CensusError("rare threshold must be in [0,1)")
    census = Census(
        rare_threshold=args.rare_threshold,
        exact_alias_cell_limit=args.exact_alias_cell_limit,
        collision_sample_modulus=args.collision_sample_modulus,
        collision_signature_limit=args.collision_signature_limit,
    )
    for root in args.dataset_root:
        scan_dataset_root(
            census,
            root,
            expected_split=None,
            row_limit=args.row_limit,
            batch_rows=args.batch_rows,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
    for root in args.train_root:
        scan_dataset_root(
            census,
            root,
            expected_split="train",
            row_limit=args.row_limit,
            batch_rows=args.batch_rows,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
    for root in args.validation_root:
        scan_dataset_root(
            census,
            root,
            expected_split="validation",
            row_limit=args.row_limit,
            batch_rows=args.batch_rows,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
    for root in args.legacy_root:
        scan_legacy_root(
            census,
            root,
            row_limit=args.row_limit,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
    for root in args.factor_cache_root:
        scan_factor_cache(
            census,
            root,
            row_limit=args.row_limit,
            batch_rows=args.batch_rows,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
    for root in args.hierarchical_cache_root:
        scan_hierarchical_cache(
            census,
            root,
            row_limit=args.row_limit,
            batch_rows=args.batch_rows,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
    config = {
        "row_limit": args.row_limit,
        "batch_rows": args.batch_rows,
        "rare_threshold": args.rare_threshold,
        "collision_sample_modulus": args.collision_sample_modulus,
        "collision_signature_limit": args.collision_signature_limit,
        "exact_alias_cell_limit": args.exact_alias_cell_limit,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
    }
    return census.finish(config)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "manifest":
            _write_json(args.output, build_manifest())
            return 0
        if args.command == "census":
            report = _run_census(args)
            if args.details_jsonl:
                write_details_jsonl(args.details_jsonl, report)
            _write_json(args.output, _public_report(report))
            return 0
        if args.command == "merge":
            report = merge_reports(args.input, require_shards=args.require_shards)
            if args.details_jsonl:
                write_details_jsonl(args.details_jsonl, report)
            _write_json(args.output, _public_report(report))
            return 0
    except (CensusError, OSError, KeyError, ValueError) as error:
        print(f"feature-schema-activation-census: {error}", file=sys.stderr)
        return 2
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
