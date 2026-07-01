from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from cascadia_mlx.r2_map_market_decision import MARKET_ACTION_SCHEMA_BLAKE3
from cascadia_mlx.r2_map_model import (
    EXACT_PARAMETER_COUNT,
    R2MapModelConfig,
    tensor_contract_manifest,
)
from cascadia_mlx.r2_map_serve import (
    MARKET_REQUEST_SCHEMA_BLAKE3,
    MARKET_RESPONSE_SCHEMA_BLAKE3,
)
from r2_map_reference_panels import (
    FROZEN_V1_CANONICAL_SHA256,
    FROZEN_V1_FORMATTED_SHA256,
    FROZEN_V1_REGISTRATION_SHA256,
    FROZEN_V1_REPOSITORY_PATH,
    PERFORMANCE_DOMAIN,
    PERFORMANCE_DOMAIN_V1_1,
    PERFORMANCE_GAME_COUNT,
    SCHEMA_ID_V1_1,
    SOURCE_BINDINGS,
    SOURCE_BINDINGS_V1_1,
    ReferencePanelError,
    build_manifest,
    build_manifest_v1_1,
    build_registration_v1_1,
    sha256_file,
    verify_manifest,
    verify_manifest_v1_1,
    verify_registration_v1_1,
)

REPOSITORY = Path(__file__).resolve().parents[1]


def test_manifest_freezes_all_required_panels_and_only_open_performance_seeds() -> None:
    manifest = build_manifest(REPOSITORY)
    panels = {panel["panel_id"]: panel for panel in manifest["panels"]}
    assert set(panels) == {
        "maximum-width-service",
        "d6-public-only",
        "replay-pinecone",
        "checkpoint-resume",
        "open-performance-100",
    }
    performance = panels["open-performance-100"]["definition"]
    assert performance["seed_domain"] == PERFORMANCE_DOMAIN
    assert len(performance["seeds"]) == PERFORMANCE_GAME_COUNT
    assert len(set(performance["seeds"])) == PERFORMANCE_GAME_COUNT
    assert performance["strength_claim_authorized"] is False
    assert panels["maximum-width-service"]["definition"]["truncation_allowed"] is False

    for domain in manifest["protected_seed_domains"]:
        assert domain["opened"] is False
        assert domain["seed_material_present"] is False
        assert "seeds" not in domain
        assert "first_seed" not in domain
    assert manifest["protected_seed_handling"]["values_accepted_by_tool"] is False


def test_frozen_manifest_verification_detects_any_drift(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    manifest = build_manifest(REPOSITORY)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    receipt = verify_manifest(REPOSITORY, path)
    assert receipt["valid"] is True
    assert receipt["protected_seed_values_opened"] is False

    manifest["panels"][0]["definition"]["reference_candidate_count"] -= 1
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ReferencePanelError, match="differs"):
        verify_manifest(REPOSITORY, path)


def test_panel_hashes_and_manifest_are_deterministic() -> None:
    assert build_manifest(REPOSITORY) == build_manifest(REPOSITORY)


def test_v1_1_is_append_only_reuses_unopened_seeds_and_binds_market_surface() -> None:
    manifest = build_manifest_v1_1(REPOSITORY)
    assert manifest["schema_id"] == SCHEMA_ID_V1_1
    assert manifest["status"] == "frozen-open-reference-panels-v1.1"
    assert manifest["predecessor"] == {
        "schema_id": "cascadia.r2-map.reference-panel-manifest.v1",
        "repository_path": FROZEN_V1_REPOSITORY_PATH,
        "formatted_file_sha256": FROZEN_V1_FORMATTED_SHA256,
        "canonical_manifest_sha256": FROZEN_V1_CANONICAL_SHA256,
        "registration_file_sha256": FROZEN_V1_REGISTRATION_SHA256,
        "execution_status": "immutable-stale-negative",
        "open_panel_outcomes_opened": False,
        "open_seed_domain_reused_by_successor": True,
    }
    panels = {panel["panel_id"]: panel for panel in manifest["panels"]}
    maximum = panels["maximum-width-service"]["definition"]
    assert maximum["all_legal_market_choices_scored_exactly_once"] is True
    assert maximum["all_legal_draft_actions_scored_exactly_once"] is True
    assert maximum["legal_market_choice_feasibility"] == (
        "public-universal-visible-market-and-species-counts-v1"
    )
    assert maximum["market_screen_hidden_permutation_invariant"] is True
    assert maximum["every_advertised_market_choice_commits"] is True
    assert maximum["independent_python_complete_screen_validation"] is True
    assert maximum["conditional_hidden_outcome_resampling_allowed"] is False
    assert maximum["scores_invalidated_after_each_public_reveal"] is True
    assert maximum["future_wipe_vectors_allowed"] is False
    replay = panels["replay-pinecone"]["definition"]
    assert replay["one_public_reveal_per_committed_paid_wipe"] is True
    assert replay["paid_wipe_spend_per_committed_wipe"] == 1
    assert replay["free_replacement_spend"] == 0
    implementation = manifest["implementation_identity"]
    assert manifest["model_schema"]["model_config"] == R2MapModelConfig().to_dict()
    assert manifest["model_schema"]["tensor_contract"] == tensor_contract_manifest()
    assert manifest["model_schema"]["expected_float32_parameters"] == EXACT_PARAMETER_COUNT
    assert implementation["market_action_schema_blake3"] == MARKET_ACTION_SCHEMA_BLAKE3
    assert implementation["request_schema_blake3"] == MARKET_REQUEST_SCHEMA_BLAKE3
    assert implementation["response_schema_blake3"] == MARKET_RESPONSE_SCHEMA_BLAKE3

    performance = panels["open-performance-100"]["definition"]
    assert performance["seed_domain"] == PERFORMANCE_DOMAIN_V1_1
    v1_seeds = build_manifest(REPOSITORY)["panels"][4]["definition"]["seeds"]
    assert performance["seeds"] == v1_seeds
    assert performance["predecessor_outcomes_opened"] is False
    assert performance["seed_domain_changed"] is False
    observed_bindings = {
        binding["path"]
        for binding in panels["open-performance-100"]["source_bindings"]
    }
    assert observed_bindings == set(SOURCE_BINDINGS_V1_1["open-performance-100"])
    assert "python/cascadia_mlx/r2_map_market_decision.py" in observed_bindings
    assert "python/tests/test_r2_map_market_decision.py" in observed_bindings
    assert "crates/cascadia-data/src/r2_map_collector.rs" in observed_bindings
    assert manifest["protected_seed_handling"]["opening_authorized"] is False


def test_v1_1_verifier_rejects_drift_and_v1_predecessor_is_byte_immutable(
    tmp_path: Path,
) -> None:
    predecessor = REPOSITORY / FROZEN_V1_REPOSITORY_PATH
    assert sha256_file(predecessor) == FROZEN_V1_FORMATTED_SHA256
    assert json.loads(predecessor.read_text(encoding="utf-8"))["manifest_sha256"] == (
        FROZEN_V1_CANONICAL_SHA256
    )

    manifest = build_manifest_v1_1(REPOSITORY)
    path = tmp_path / "manifest-v1.1.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert verify_manifest_v1_1(REPOSITORY, path)["valid"] is True
    manifest["panels"][2]["definition"]["free_replacement_spend"] = 1
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ReferencePanelError, match="differs"):
        verify_manifest_v1_1(REPOSITORY, path)


def test_v1_1_registration_is_append_only_and_binds_identical_live_bytes(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    required = {
        relative
        for bindings in (SOURCE_BINDINGS, SOURCE_BINDINGS_V1_1)
        for paths in bindings.values()
        for relative in paths
    }
    required.add(FROZEN_V1_REPOSITORY_PATH)
    for relative in sorted(required):
        source = REPOSITORY / relative
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    repository_manifest = (
        repository / "docs/v2/reports/r2-map-w0-reference-panel-manifest-v1.1.json"
    )
    rendered = json.dumps(build_manifest_v1_1(repository), sort_keys=True, indent=2) + "\n"
    repository_manifest.write_text(rendered, encoding="utf-8")
    control_root = tmp_path / "control" / "w0-preregistration"
    control_root.mkdir(parents=True)
    ssd_manifest = control_root / "reference-panel-manifest-v1.1.json"
    ssd_manifest.write_text(rendered, encoding="utf-8")
    predecessor_registration = control_root / "registration.json"
    predecessor_registration.write_text('{"immutable":"v1"}\n', encoding="utf-8")
    predecessor_sha256 = sha256_file(predecessor_registration)

    registration = build_registration_v1_1(
        repository,
        repository_manifest,
        ssd_manifest,
        predecessor_registration,
        "2026-06-18T00:00:00Z",
        control_root=control_root,
        predecessor_registration_sha256=predecessor_sha256,
    )
    registration_path = control_root / "registration-v1.1.json"
    registration_path.write_text(
        json.dumps(registration, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    receipt = verify_registration_v1_1(
        repository,
        registration_path,
        control_root=control_root,
        predecessor_registration_sha256=predecessor_sha256,
    )
    assert receipt["valid"] is True
    binding = receipt["implementation_binding"]
    assert binding["w0_registration_sha256"] == sha256_file(registration_path)
    assert binding["protocols"]["collector_hash"] == list(
        bytes.fromhex(binding["replay_pinecone_panel_sha256"])
    )

    ssd_manifest.write_text(rendered + "\n", encoding="utf-8")
    with pytest.raises(ReferencePanelError, match="bytes differ"):
        verify_registration_v1_1(
            repository,
            registration_path,
            control_root=control_root,
            predecessor_registration_sha256=predecessor_sha256,
        )
