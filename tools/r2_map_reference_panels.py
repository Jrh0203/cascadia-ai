#!/usr/bin/env python3
"""Freeze and verify the open W0 reference panels for R2-MAP.

The 100-game performance seeds are deliberately open. Candidate-gate and final
strength seeds are represented only by domain commitments; this tool has no
interface that can derive, accept, print, or persist their seed values.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import blake3

REPOSITORY = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPOSITORY / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from cascadia_mlx.r2_map_contracts import (  # noqa: E402
    CAMPAIGN_ROOT,
    require_local_storage_authority,
)
from cascadia_mlx.r2_map_market_decision import (  # noqa: E402
    MARKET_ACTION_SCHEMA_BLAKE3,
)
from cascadia_mlx.r2_map_model import (  # noqa: E402
    EXACT_PARAMETER_COUNT,
    R2MapModelConfig,
    tensor_contract_manifest,
)
from cascadia_mlx.r2_map_serve import (  # noqa: E402
    MARKET_REQUEST_SCHEMA,
    MARKET_REQUEST_SCHEMA_BLAKE3,
    MARKET_RESPONSE_SCHEMA,
    MARKET_RESPONSE_SCHEMA_BLAKE3,
)

SCHEMA_ID = "cascadia.r2-map.reference-panel-manifest.v1"
SCHEMA_ID_V1_1 = "cascadia.r2-map.reference-panel-manifest.v1.1"
CAMPAIGN_ID = "r2-map-expert-iteration-v1"
PERFORMANCE_DOMAIN = "r2-map-open-reference-performance-100-v1"
# v1 outcomes were never opened. The v1.1 implementation binding therefore
# reuses the already-preregistered open seeds exactly; only live source,
# protocol, and model identities change.
PERFORMANCE_DOMAIN_V1_1 = PERFORMANCE_DOMAIN
PERFORMANCE_GAME_COUNT = 100
FROZEN_V1_FORMATTED_SHA256 = (
    "12555a92ab337eca8d299210e19f5c4bb52298822e82f688ad967ceeaed1f7ec"
)
FROZEN_V1_CANONICAL_SHA256 = (
    "5d88e296810eb5f8c5abc67ebc317ce987a2edb11d97b0c4e55ea873d96e5a65"
)
FROZEN_V1_REGISTRATION_SHA256 = (
    "7d0336714a1e520c9c99f0d488e48577848f6c0b336ca6257ae987f2548e0d51"
)
FROZEN_V1_REPOSITORY_PATH = "docs/v2/reports/r2-map-w0-reference-panel-manifest-v1.json"
REGISTRATION_SCHEMA_ID_V1_1 = "cascadia.r2-map.w0-preregistration-registration.v1.1"
CAMPAIGN_CONTROL_ROOT = CAMPAIGN_ROOT / "control/w0-preregistration"

SOURCE_BINDINGS: dict[str, tuple[str, ...]] = {
    "maximum-width-service": (
        "python/cascadia_mlx/r2_map_serve.py",
        "python/tests/test_r2_map_serve.py",
        "tests/fixtures/r2_map/public-market-decision-protocol-v3.json",
    ),
    "d6-public-only": (
        "crates/cascadia-game/src/symmetry.rs",
        "python/cascadia_mlx/r2_map_dataset.py",
        "python/cascadia_mlx/r2_map_model.py",
        "python/tests/test_d6_contract.py",
        "python/tests/test_r2_map_dataset.py",
        "python/tests/test_r2_map_model.py",
    ),
    "replay-pinecone": (
        "crates/cascadia-data/src/r2_map_experience.rs",
        "crates/cascadia-eval/src/focal.rs",
    ),
    "checkpoint-resume": (
        "python/cascadia_mlx/checkpoint.py",
        "python/cascadia_mlx/r2_map_train.py",
        "python/cascadia_mlx/r2_map_verify.py",
        "python/tests/test_r2_map_checkpoint_train.py",
    ),
    "open-performance-100": (
        "crates/cascadia-eval/src/focal.rs",
        "crates/cascadia-eval/src/focal_campaign.rs",
        "crates/cascadia-eval/src/focal_gameplay.rs",
    ),
}

# v1.1 is an append-only implementation binding for the sequential public
# market contract.  It intentionally expands, rather than mutates, the v1
# binding surface.  The predecessor file remains byte-for-byte immutable and
# is expected to fail the live-source gate after the v1.1 repair.
SOURCE_BINDINGS_V1_1: dict[str, tuple[str, ...]] = {
    "maximum-width-service": (
        "crates/cascadia-game/src/game.rs",
        "crates/cascadia-r2/src/r2_map_runtime.rs",
        "crates/cascadia-model/src/r2_map.rs",
        "crates/cascadia-search/src/r2_map_direct.rs",
        "crates/cascadia-search/src/r2_map_runner.rs",
        "python/cascadia_mlx/r2_map_market_decision.py",
        "python/cascadia_mlx/r2_map_model.py",
        "python/cascadia_mlx/r2_map_protocol_fixture.py",
        "python/cascadia_mlx/r2_map_serve.py",
        "python/tests/test_r2_map_market_decision.py",
        "python/tests/test_r2_map_serve.py",
        "tools/r2_map_market_protocol_fixture.py",
    ),
    "d6-public-only": (
        "crates/cascadia-game/src/game.rs",
        "crates/cascadia-game/src/symmetry.rs",
        "crates/cascadia-r2/src/r2_map_runtime.rs",
        "python/cascadia_mlx/r2_map_dataset.py",
        "python/cascadia_mlx/r2_map_model.py",
        "python/tests/test_d6_contract.py",
        "python/tests/test_r2_map_dataset.py",
        "python/tests/test_r2_map_model.py",
    ),
    "replay-pinecone": (
        "crates/cascadia-game/src/game.rs",
        "crates/cascadia-data/src/r2_map_collector.rs",
        "crates/cascadia-data/src/r2_map_experience.rs",
        "crates/cascadia-eval/src/focal.rs",
        "crates/cascadia-eval/src/r2_map_gameplay.rs",
    ),
    "checkpoint-resume": (
        *SOURCE_BINDINGS["checkpoint-resume"],
        "python/cascadia_mlx/r2_map_training_contract.py",
        "python/cascadia_mlx/r2_map_remote_training.py",
        "python/tests/test_r2_map_remote_training.py",
        "tools/r2_map_john1_train.py",
    ),
    "open-performance-100": (
        "crates/cascadia-game/src/game.rs",
        "crates/cascadia-data/src/r2_map_collector.rs",
        "crates/cascadia-data/src/r2_map_experience.rs",
        "crates/cascadia-r2/src/r2_map_runtime.rs",
        "crates/cascadia-model/src/r2_map.rs",
        "crates/cascadia-search/src/r2_map_direct.rs",
        "crates/cascadia-search/src/r2_map_runner.rs",
        "crates/cascadia-eval/src/focal.rs",
        "crates/cascadia-eval/src/focal_campaign.rs",
        "crates/cascadia-eval/src/longitudinal.rs",
        "crates/cascadia-eval/src/r2_map_binding.rs",
        "crates/cascadia-eval/src/r2_map_gameplay.rs",
        "crates/cascadia-cli-v2/src/r2_map_commands.rs",
        "python/cascadia_mlx/r2_map_market_decision.py",
        "python/cascadia_mlx/r2_map_model.py",
        "python/cascadia_mlx/r2_map_serve.py",
        "python/tests/test_r2_map_market_decision.py",
        "tests/fixtures/r2_map/public-market-decision-protocol-v3.json",
        "tools/r2_map_reference_panels.py",
    ),
}


class ReferencePanelError(ValueError):
    """A reference-panel commitment or source binding is invalid."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def performance_seeds(domain: str = PERFORMANCE_DOMAIN) -> list[int]:
    """Return the only seed material this module is capable of deriving."""

    return [
        int.from_bytes(
            hashlib.sha256(f"{domain}:{index:03d}".encode("ascii")).digest()[:8],
            byteorder="little",
            signed=False,
        )
        for index in range(PERFORMANCE_GAME_COUNT)
    ]


def _source_binding(repository: Path, paths: Sequence[str]) -> list[dict[str, str]]:
    bindings = []
    for relative in paths:
        path = repository / relative
        if not path.is_file():
            raise ReferencePanelError(f"required source binding is missing: {relative}")
        bindings.append({"path": relative, "sha256": sha256_file(path)})
    return bindings


def _panel(
    repository: Path,
    panel_id: str,
    definition: Mapping[str, Any],
    source_bindings: Mapping[str, Sequence[str]] = SOURCE_BINDINGS,
) -> dict[str, Any]:
    value = {
        "panel_id": panel_id,
        "definition": dict(definition),
        "source_bindings": _source_binding(repository, source_bindings[panel_id]),
    }
    value["panel_sha256"] = sha256_bytes(canonical_json(value))
    return value


def _protected_domain(domain_id: str, pair_or_game_count: int) -> dict[str, Any]:
    public_descriptor = {
        "domain_id": domain_id,
        "count": pair_or_game_count,
        "opened": False,
        "seed_material_present": False,
        "provisioning": "separate sealed workflow only after its registered phase barrier",
    }
    return {
        **public_descriptor,
        "domain_commitment_sha256": sha256_bytes(canonical_json(public_descriptor)),
    }


def _serving_protocol_schema(repository: Path) -> dict[str, Any]:
    fixture_relative = "tests/fixtures/r2_map/public-market-decision-protocol-v3.json"
    fixture_path = repository / fixture_relative
    try:
        fixture_bytes = fixture_path.read_bytes()
        fixture = json.loads(fixture_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise ReferencePanelError(f"cannot read public market protocol fixture: {error}") from error
    claimed_canonical = fixture.pop("fixture_blake3", None)
    observed_canonical = blake3.blake3(canonical_json(fixture)).hexdigest()
    expected = {
        "action_schema_blake3": MARKET_ACTION_SCHEMA_BLAKE3,
        "request_schema": MARKET_REQUEST_SCHEMA,
        "request_schema_blake3": MARKET_REQUEST_SCHEMA_BLAKE3,
        "response_schema": MARKET_RESPONSE_SCHEMA,
        "response_schema_blake3": MARKET_RESPONSE_SCHEMA_BLAKE3,
    }
    if claimed_canonical != observed_canonical or any(
        fixture.get(key) != value for key, value in expected.items()
    ):
        raise ReferencePanelError(
            "public market protocol fixture differs from the live serving schemas"
        )
    return {
        "schema_id": "cascadia.r2-map.sequential-public-serving.v1.1",
        "market_action_schema_blake3": MARKET_ACTION_SCHEMA_BLAKE3,
        "request_schema_blake3": MARKET_REQUEST_SCHEMA_BLAKE3,
        "response_schema_blake3": MARKET_RESPONSE_SCHEMA_BLAKE3,
        "fixture_canonical_blake3": observed_canonical,
        "fixture_file_blake3": blake3.blake3(fixture_bytes).hexdigest(),
        "fixture_path": fixture_relative,
        "market_action_bytes": 8,
        "stage_order": ["free-three-of-a-kind", "paid-wipes", "draft"],
        "choice_order": "canonical-engine-order",
        "score_invalidation": "after-every-committed-public-reveal",
        "draft_protocol": "r2-map-grouped-exhaustive-request-v1.1",
    }


def build_manifest(repository: Path) -> dict[str, Any]:
    repository = repository.resolve()
    open_seeds = performance_seeds()
    if len(set(open_seeds)) != PERFORMANCE_GAME_COUNT:
        raise ReferencePanelError("open performance seed derivation collided")

    panels = [
        _panel(
            repository,
            "maximum-width-service",
            {
                "protocol": "r2-map-grouped-exhaustive-request-v1",
                "reference_candidate_count": 6372,
                "expected_action_evaluations": 6372,
                "complete_cardinality_required": True,
                "ordered_action_id_derivation": (
                    "sha256('r2-map-max-width-service-v1:' || zero-padded action index)"
                ),
                "truncation_allowed": False,
            },
        ),
        _panel(
            repository,
            "d6-public-only",
            {
                "d6_schema": "exact-r2-d6-v1",
                "transform_ids": list(range(12)),
                "round_trip_required": True,
                "prediction_permutation_or_invariance_required": True,
                "forbidden_tensor_metadata": [
                    "campaign_id",
                    "checkpoint_id",
                    "collector_host",
                    "game_seed",
                    "global_game_index",
                    "iteration",
                    "policy_id",
                    "promotion_index",
                    "seed_domain",
                    "split",
                ],
                "hidden_order_mutation_independence_required": True,
                "future_refill_mutation_independence_required": True,
            },
        ),
        _panel(
            repository,
            "replay-pinecone",
            {
                "game_config": "research-aaaaa-4p-no-habitat-bonus",
                "experience_schema": "CSDR2XP-v1",
                "replay_seal_required": True,
                "score_reconciliation_required": True,
                "focal_decisions_per_iterative_game": 20,
                "bootstrap_decisions_per_game": 80,
                "pinecone_identity": (
                    "earned - independent_draft_spend - paid_wipe_spend = remaining"
                ),
                "free_three_of_a_kind_replacements_are_context_only": True,
            },
        ),
        _panel(
            repository,
            "checkpoint-resume",
            {
                "checkpoint_schema_version": 2,
                "prediction_panel_id": "r2-map-fixed-panel-v1",
                "synthetic_seed": 42,
                "checkpoint_after_global_step": 1,
                "expected_next_batch_identity": "synthetic-batch-0001",
                "model_optimizer_rng_cursor_sampler_and_loss_head_exact": True,
                "fault_stages": "all registered R2_MAP_WRITE_STAGES",
                "corruption_rejection_required": True,
            },
        ),
        _panel(
            repository,
            "open-performance-100",
            {
                "seed_domain": PERFORMANCE_DOMAIN,
                "seeds": open_seeds,
                "game_count": PERFORMANCE_GAME_COUNT,
                "focal_seat": "game_index mod 4",
                "game_config": "research-aaaaa-4p-no-habitat-bonus",
                "reference_and_optimized_order": [
                    "reference-then-optimized",
                    "optimized-then-reference",
                ],
                "strength_claim_authorized": False,
                "protected_domain": False,
            },
        ),
    ]
    manifest = {
        "schema_id": SCHEMA_ID,
        "campaign_id": CAMPAIGN_ID,
        "status": "frozen-open-reference-panels",
        "panels": panels,
        "protected_seed_domains": [
            _protected_domain("r2-map-strength-blinded-smoke-20-v1", 20),
            _protected_domain("r2-map-fixed-development-gate-250-v1", 250),
            _protected_domain("r2-map-final-domain-1000-v1", 1000),
        ],
        "protected_seed_handling": {
            "values_in_manifest": False,
            "values_accepted_by_tool": False,
            "values_derivable_by_tool": False,
            "opening_authorized": False,
        },
    }
    manifest["manifest_sha256"] = sha256_bytes(canonical_json(manifest))
    return manifest


def _frozen_v1_predecessor(repository: Path) -> dict[str, Any]:
    path = repository.resolve() / FROZEN_V1_REPOSITORY_PATH
    if not path.is_file():
        raise ReferencePanelError("frozen v1 predecessor manifest is missing")
    formatted_sha256 = sha256_file(path)
    if formatted_sha256 != FROZEN_V1_FORMATTED_SHA256:
        raise ReferencePanelError("frozen v1 predecessor manifest bytes changed")
    try:
        predecessor = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReferencePanelError(f"cannot read frozen v1 predecessor: {error}") from error
    if predecessor.get("manifest_sha256") != FROZEN_V1_CANONICAL_SHA256:
        raise ReferencePanelError("frozen v1 predecessor canonical identity changed")
    return {
        "schema_id": SCHEMA_ID,
        "repository_path": FROZEN_V1_REPOSITORY_PATH,
        "formatted_file_sha256": FROZEN_V1_FORMATTED_SHA256,
        "canonical_manifest_sha256": FROZEN_V1_CANONICAL_SHA256,
        "registration_file_sha256": FROZEN_V1_REGISTRATION_SHA256,
        "execution_status": "immutable-stale-negative",
        "open_panel_outcomes_opened": False,
        "open_seed_domain_reused_by_successor": True,
    }


def build_manifest_v1_1(repository: Path) -> dict[str, Any]:
    """Build the append-only v1.1 binding after the public-market repair.

    Protected-domain descriptors are inherited without exposing seed material.
    The open reference seeds remain byte-identical because no v1 outcomes were
    opened; v1.1 changes only implementation bindings.
    """

    repository = repository.resolve()
    predecessor = _frozen_v1_predecessor(repository)
    base = build_manifest(repository)
    definitions = {
        panel["panel_id"]: dict(panel["definition"]) for panel in base["panels"]
    }

    definitions["maximum-width-service"].update(
        {
            "protocol": "r2-map-grouped-exhaustive-request-v1.1",
            "market_protocol": "r2-map-sequential-public-market-v1.1",
            "market_stage_order": ["free-three-of-a-kind", "paid-wipes", "draft"],
            "all_legal_market_choices_scored_exactly_once": True,
            "all_legal_draft_actions_scored_exactly_once": True,
            "legal_market_choice_feasibility": (
                "public-universal-visible-market-and-species-counts-v1"
            ),
            "market_screen_hidden_permutation_invariant": True,
            "every_advertised_market_choice_commits": True,
            "independent_python_complete_screen_validation": True,
            "scores_invalidated_after_each_public_reveal": True,
            "future_wipe_vectors_allowed": False,
            "conditional_hidden_outcome_resampling_allowed": False,
        }
    )
    definitions["d6-public-only"].update(
        {
            "market_parent_is_current_public_state": True,
            "replacement_visible_only_after_committed_choice": True,
            "legal_choice_set_independent_of_hidden_refill_order": True,
        }
    )
    definitions["replay-pinecone"].update(
        {
            "market_trace_required": True,
            "market_trace_fields": [
                "stage",
                "ordered_legal_choice_hashes",
                "parent_public_state_hash",
                "selected_choice_hash",
                "resulting_public_state_hash",
            ],
            "one_public_reveal_per_committed_paid_wipe": True,
            "paid_wipe_spend_per_committed_wipe": 1,
            "free_replacement_spend": 0,
            "bundled_turn_action_must_replay_exact_trace": True,
        }
    )
    definitions["checkpoint-resume"].update(
        {
            "model_contract": "r2-map-v1.1",
            "sequential_market_head_resume_exact": True,
        }
    )
    open_seeds = performance_seeds(PERFORMANCE_DOMAIN_V1_1)
    if len(set(open_seeds)) != PERFORMANCE_GAME_COUNT:
        raise ReferencePanelError("v1.1 open performance seed derivation collided")
    definitions["open-performance-100"].update(
        {
            "seed_domain": PERFORMANCE_DOMAIN_V1_1,
            "seeds": open_seeds,
            "runtime_contract": "r2-map-sequential-public-market-v1.1",
            "old_v1_outcomes_reused": False,
            "predecessor_outcomes_opened": False,
            "seed_domain_changed": False,
            "market_trace_and_pinecone_validation_required": True,
        }
    )

    panels = [
        _panel(
            repository,
            panel_id,
            definitions[panel_id],
            SOURCE_BINDINGS_V1_1,
        )
        for panel_id in (
            "maximum-width-service",
            "d6-public-only",
            "replay-pinecone",
            "checkpoint-resume",
            "open-performance-100",
        )
    ]
    unique_bindings: dict[str, str] = {}
    for panel in panels:
        for binding in panel["source_bindings"]:
            previous = unique_bindings.setdefault(binding["path"], binding["sha256"])
            if previous != binding["sha256"]:
                raise ReferencePanelError("one source path acquired conflicting hashes")
    source_bundle = [
        {"path": path, "sha256": digest}
        for path, digest in sorted(unique_bindings.items())
    ]
    model_config = R2MapModelConfig()
    model_config_values = model_config.to_dict()
    tensor_contract = tensor_contract_manifest(model_config)
    model_schema = {
        "schema_id": "cascadia.r2-map.model-contract.v1.1",
        "model_config": model_config_values,
        "tensor_contract": tensor_contract,
        "hidden_dimension": model_config.hidden_dim,
        "attention_heads": model_config.attention_heads,
        "board_latents": model_config.board_latents,
        "board_latent_blocks": model_config.board_latent_blocks,
        "cross_board_blocks": model_config.cross_board_blocks,
        "feed_forward_multiplier": model_config.feed_forward_multiplier,
        "action_fusion_expansion": model_config.action_fusion_expansion,
        "multitask_bottleneck": model_config.multitask_dim,
        "expected_float32_parameters": EXACT_PARAMETER_COUNT,
        "reference_precision": model_config.precision,
        "candidate_semantics": "exact-full-public-afterstate",
        "legal_action_coverage": "exhaustive",
        "sequential_market_head": True,
    }
    serving_protocol_schema = _serving_protocol_schema(repository)
    panel_hashes = {panel["panel_id"]: panel["panel_sha256"] for panel in panels}
    implementation_identity = {
        "source_bundle_sha256": sha256_bytes(canonical_json(source_bundle)),
        "serving_protocol_schema_sha256": sha256_bytes(
            canonical_json(serving_protocol_schema)
        ),
        "market_action_schema_blake3": serving_protocol_schema[
            "market_action_schema_blake3"
        ],
        "request_schema_blake3": serving_protocol_schema["request_schema_blake3"],
        "response_schema_blake3": serving_protocol_schema["response_schema_blake3"],
        "protocol_fixture_canonical_blake3": serving_protocol_schema[
            "fixture_canonical_blake3"
        ],
        "protocol_fixture_file_blake3": serving_protocol_schema[
            "fixture_file_blake3"
        ],
        "model_schema_sha256": sha256_bytes(canonical_json(model_schema)),
        "maximum_width_panel_sha256": panel_hashes["maximum-width-service"],
        "replay_pinecone_panel_sha256": panel_hashes["replay-pinecone"],
        "open_reference_seed_domain_id": PERFORMANCE_DOMAIN_V1_1,
    }
    manifest = {
        "schema_id": SCHEMA_ID_V1_1,
        "campaign_id": CAMPAIGN_ID,
        "contract_revision": "sequential-public-market-v1.1",
        "status": "frozen-open-reference-panels-v1.1",
        "predecessor": predecessor,
        "implementation_identity": implementation_identity,
        "source_bundle": source_bundle,
        "model_schema": model_schema,
        "serving_protocol_schema": serving_protocol_schema,
        "panels": panels,
        "protected_seed_domains": base["protected_seed_domains"],
        "protected_seed_handling": base["protected_seed_handling"],
    }
    manifest["manifest_sha256"] = sha256_bytes(canonical_json(manifest))
    return manifest


def verify_manifest(repository: Path, path: Path) -> dict[str, Any]:
    try:
        observed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReferencePanelError(f"cannot read frozen manifest: {error}") from error
    expected = build_manifest(repository)
    if observed != expected:
        raise ReferencePanelError("frozen reference-panel manifest differs from current bindings")
    return {
        "valid": True,
        "manifest_sha256": expected["manifest_sha256"],
        "panel_sha256": {panel["panel_id"]: panel["panel_sha256"] for panel in expected["panels"]},
        "open_performance_games": PERFORMANCE_GAME_COUNT,
        "protected_seed_values_opened": False,
    }


def verify_manifest_v1_1(repository: Path, path: Path) -> dict[str, Any]:
    try:
        observed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReferencePanelError(f"cannot read frozen v1.1 manifest: {error}") from error
    expected = build_manifest_v1_1(repository)
    if observed != expected:
        raise ReferencePanelError(
            "frozen v1.1 reference-panel manifest differs from current bindings"
        )
    return {
        "valid": True,
        "schema_id": SCHEMA_ID_V1_1,
        "manifest_sha256": expected["manifest_sha256"],
        "predecessor_formatted_sha256": FROZEN_V1_FORMATTED_SHA256,
        "panel_sha256": {panel["panel_id"]: panel["panel_sha256"] for panel in expected["panels"]},
        "open_performance_games": PERFORMANCE_GAME_COUNT,
        "protected_seed_values_opened": False,
    }


def build_registration_v1_1(
    repository: Path,
    repository_manifest: Path,
    ssd_manifest: Path,
    predecessor_registration: Path,
    registered_at: str,
    *,
    control_root: Path = CAMPAIGN_CONTROL_ROOT,
    predecessor_registration_sha256: str = FROZEN_V1_REGISTRATION_SHA256,
) -> dict[str, Any]:
    """Bind identical repository/john2-storage v1.1 bytes to the immutable v1 chain."""

    if control_root == CAMPAIGN_CONTROL_ROOT:
        require_local_storage_authority()
    repository = repository.resolve()
    repository_manifest = repository_manifest.resolve()
    ssd_manifest = ssd_manifest.resolve()
    predecessor_registration = predecessor_registration.resolve()
    control_root = control_root.resolve()
    expected_repository_manifest = (
        repository / "docs/v2/reports/r2-map-w0-reference-panel-manifest-v1.1.json"
    )
    expected_ssd_manifest = control_root / "reference-panel-manifest-v1.1.json"
    expected_predecessor = control_root / "registration.json"
    if (
        repository_manifest != expected_repository_manifest
        or ssd_manifest != expected_ssd_manifest
        or predecessor_registration != expected_predecessor
    ):
        raise ReferencePanelError("v1.1 registration paths do not match the append-only layout")
    verification = verify_manifest_v1_1(repository, repository_manifest)
    repository_bytes = repository_manifest.read_bytes()
    if ssd_manifest.read_bytes() != repository_bytes:
        raise ReferencePanelError("repository and canonical john2 v1.1 manifest bytes differ")
    if sha256_file(predecessor_registration) != predecessor_registration_sha256:
        raise ReferencePanelError("frozen v1 registration bytes changed")
    if not registered_at or not registered_at.isascii():
        raise ReferencePanelError("registered_at must be a nonempty ASCII timestamp")
    formatted_sha256 = sha256_bytes(repository_bytes)
    manifest = json.loads(repository_bytes)
    return {
        "schema_id": REGISTRATION_SCHEMA_ID_V1_1,
        "campaign_id": CAMPAIGN_ID,
        "contract_revision": "sequential-public-market-v1.1",
        "registered_at": registered_at,
        "append_only_predecessor": {
            "path": str(predecessor_registration),
            "formatted_file_sha256": predecessor_registration_sha256,
            "execution_status": "immutable-stale-negative",
        },
        "artifacts": {
            "reference_panels": {
                "repository_path": str(repository_manifest),
                "ssd_path": str(ssd_manifest),
                "formatted_file_sha256": formatted_sha256,
                "canonical_manifest_sha256": verification["manifest_sha256"],
            }
        },
        "implementation_identity": manifest["implementation_identity"],
        "independent_verification": {
            "python_exact_regeneration_required": True,
            "rust_source_rehash_required": True,
            "rust_initializer_must_reject_v1": True,
            "all_live_source_bindings_required": True,
        },
        "protected_seed_values_opened": False,
        "john4_used": False,
    }


def verify_registration_v1_1(
    repository: Path,
    registration_path: Path,
    *,
    control_root: Path = CAMPAIGN_CONTROL_ROOT,
    predecessor_registration_sha256: str = FROZEN_V1_REGISTRATION_SHA256,
) -> dict[str, Any]:
    try:
        observed = json.loads(registration_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReferencePanelError(f"cannot read v1.1 registration: {error}") from error
    reference = observed.get("artifacts", {}).get("reference_panels", {})
    registered_at = observed.get("registered_at")
    if not isinstance(reference, dict) or not isinstance(registered_at, str):
        raise ReferencePanelError("v1.1 registration shape is incomplete")
    expected = build_registration_v1_1(
        repository,
        Path(str(reference.get("repository_path", ""))),
        Path(str(reference.get("ssd_path", ""))),
        control_root / "registration.json",
        registered_at,
        control_root=control_root,
        predecessor_registration_sha256=predecessor_registration_sha256,
    )
    if observed != expected:
        raise ReferencePanelError("v1.1 registration differs from current immutable bindings")
    identity = observed["implementation_identity"]
    implementation_binding = {
        "schema_id": "cascadia.r2-map.implementation-binding.v1.1",
        "contract_revision": "sequential-public-market-v1.1",
        "w0_registration_sha256": sha256_file(registration_path),
        "reference_manifest_sha256": reference["canonical_manifest_sha256"],
        "maximum_width_panel_sha256": identity["maximum_width_panel_sha256"],
        "replay_pinecone_panel_sha256": identity["replay_pinecone_panel_sha256"],
        "source_bundle_sha256": identity["source_bundle_sha256"],
        "serving_protocol_schema_sha256": identity[
            "serving_protocol_schema_sha256"
        ],
        "market_action_schema_blake3": identity["market_action_schema_blake3"],
        "request_schema_blake3": identity["request_schema_blake3"],
        "response_schema_blake3": identity["response_schema_blake3"],
        "protocol_fixture_canonical_blake3": identity[
            "protocol_fixture_canonical_blake3"
        ],
        "protocol_fixture_file_blake3": identity["protocol_fixture_file_blake3"],
        "model_schema_sha256": identity["model_schema_sha256"],
        "open_reference_seed_domain_id": identity[
            "open_reference_seed_domain_id"
        ],
        "protocols": {
            "collector_hash": list(
                bytes.fromhex(identity["replay_pinecone_panel_sha256"])
            ),
            "source_hash": list(bytes.fromhex(identity["source_bundle_sha256"])),
            "serving_protocol_hash": list(
                bytes.fromhex(identity["serving_protocol_schema_sha256"])
            ),
        },
    }
    return {
        "valid": True,
        "schema_id": REGISTRATION_SCHEMA_ID_V1_1,
        "manifest_sha256": reference["canonical_manifest_sha256"],
        "implementation_binding": implementation_binding,
        "protected_seed_values_opened": False,
    }


def _write_immutable(path: Path, rendered: str) -> None:
    """Publish complete bytes without overwriting an existing artifact."""

    encoded = rendered.encode("utf-8")
    parent = path.parent
    if not parent.is_dir():
        raise ReferencePanelError(f"immutable output parent is missing: {parent}")
    lock_path = parent / f".{path.name}.lock"
    lock = lock_path.open("a+b")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if path.exists():
            if path.read_bytes() != encoded:
                raise ReferencePanelError(f"immutable output already differs: {path}")
            return
        temporary = parent / f".{path.name}.{os.getpid()}.tmp"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            # The exclusive advisory lock makes the absent-destination check
            # and rename one transaction for every campaign writer.
            os.rename(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        directory = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    result.add_argument("--output", type=Path, help="write rendered JSON to this exact path")
    result.add_argument(
        "--revision",
        choices=("v1", "v1.1"),
        default="v1",
        help="render or verify one immutable implementation binding",
    )
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("render")
    verify = commands.add_parser("verify")
    verify.add_argument("manifest", type=Path)
    registration = commands.add_parser("render-registration")
    registration.add_argument("--repository-manifest", type=Path, required=True)
    registration.add_argument(
        "--ssd-manifest",
        type=Path,
        required=True,
        help="compatibility name for the canonical john2 storage manifest",
    )
    registration.add_argument("--predecessor-registration", type=Path, required=True)
    registration.add_argument("--registered-at", required=True)
    verify_registration = commands.add_parser("verify-registration")
    verify_registration.add_argument("registration", type=Path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        if arguments.command == "render":
            value = (
                build_manifest(arguments.repository)
                if arguments.revision == "v1"
                else build_manifest_v1_1(arguments.repository)
            )
        elif arguments.command == "verify":
            value = (
                verify_manifest(arguments.repository, arguments.manifest)
                if arguments.revision == "v1"
                else verify_manifest_v1_1(arguments.repository, arguments.manifest)
            )
        elif arguments.command == "render-registration":
            if arguments.revision != "v1.1":
                raise ReferencePanelError("registration rendering requires --revision v1.1")
            value = build_registration_v1_1(
                arguments.repository,
                arguments.repository_manifest,
                arguments.ssd_manifest,
                arguments.predecessor_registration,
                arguments.registered_at,
            )
        else:
            if arguments.revision != "v1.1":
                raise ReferencePanelError("registration verification requires --revision v1.1")
            value = verify_registration_v1_1(
                arguments.repository,
                arguments.registration,
            )
        rendered = json.dumps(value, sort_keys=True, indent=2) + "\n"
        if arguments.output is None:
            sys.stdout.write(rendered)
        else:
            _write_immutable(arguments.output, rendered)
    except (OSError, ReferencePanelError) as error:
        print(f"R2-MAP reference-panel verification refused: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
