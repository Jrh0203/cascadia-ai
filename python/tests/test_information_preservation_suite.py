from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from cascadia_mlx.dataset import _RECORD_DTYPE
from cascadia_mlx.information_preservation_suite import (
    CLASSIFICATION_FAILED,
    CLASSIFICATION_INVALID,
    F1_CLASSIFICATION_SCIENTIFIC_BLAKE3,
    F1_FORWARD_SCIENTIFIC_BLAKE3,
    IN_RADIUS_BOUNDARY_ID,
    REFILL_DISTRIBUTION_BOUNDARY_ID,
    REQUIRED_FAMILIES,
    REQUIRED_PROBES,
    AdversarialPair,
    BoundaryAdapter,
    BoundaryRegistry,
    ConceptExpectation,
    DependencyBlock,
    FixtureSet,
    SuiteValidationError,
    build_pair,
    canonical_json,
    confidence_set_evidence,
    default_boundary_registry,
    default_probe_registry,
    evaluate_pair_boundary,
    file_blake3,
    load_fixture_set,
    load_resolved_dependency_artifact,
    main,
    materialize_resolved_dependencies,
    render_markdown,
    run_suite,
    scientific_blake3,
    source_from_compact_projection,
    source_from_dataset_batch,
    source_from_graded_factor_array,
    source_from_graded_oracle_batch,
    source_from_v2_position_record,
    write_outputs,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = (
    ROOT
    / "artifacts"
    / "experiments"
    / "information-preservation-adversarial-suite-v1"
    / "fixtures"
    / "pairs-v1.json"
)
RESOLVED_DEPENDENCY_PATH = FIXTURE_PATH.with_name("resolved-dependencies-v2.json")
GENERATOR_MANIFEST = FIXTURE_PATH.parents[1] / "fixture-generator" / "Cargo.toml"


@pytest.fixture(scope="module")
def fixture_set() -> FixtureSet:
    return load_fixture_set(FIXTURE_PATH)


def _pair(fixture_set: FixtureSet, family: str) -> AdversarialPair:
    return next(pair for pair in fixture_set.pairs if pair.family == family)


def _registry_for(fixture_set: FixtureSet) -> BoundaryRegistry:
    declared = sorted(
        {
            boundary_id
            for pair in fixture_set.pairs
            for side in pair.public_inputs.values()
            for boundary_id in side.get("declared_projections", {})
        }
    )
    return default_boundary_registry(declared_boundary_ids=declared)


def test_fixture_schema_covers_all_required_families_and_probes(
    fixture_set: FixtureSet,
) -> None:
    assert tuple(pair.family for pair in fixture_set.pairs) == REQUIRED_FAMILIES
    assert len({pair.pair_id for pair in fixture_set.pairs}) == 14
    assert default_probe_registry().ids() == REQUIRED_PROBES
    assert fixture_set.canonical_hash == fixture_set.expected_hash()
    assert sum(pair.status == "ready" for pair in fixture_set.pairs) == 14
    assert sum(pair.status == "dependency_blocked" for pair in fixture_set.pairs) == 0
    assert all(pair.dependency is None for pair in fixture_set.pairs)


def test_pair_schema_rejects_unknown_fields_and_hash_drift(
    fixture_set: FixtureSet,
) -> None:
    values = _pair(fixture_set, "semantic_tile_multiset").to_dict()
    values["unexpected"] = True
    with pytest.raises(SuiteValidationError, match="keys drifted"):
        AdversarialPair.from_dict(values)

    values.pop("unexpected")
    values["title"] = "tampered"
    with pytest.raises(SuiteValidationError, match="hash mismatch"):
        AdversarialPair.from_dict(values)


def test_ready_labels_must_exactly_match_the_declared_relation() -> None:
    with pytest.raises(SuiteValidationError, match="contradict relation"):
        build_pair(
            pair_id="bad-label-relation",
            family=REQUIRED_FAMILIES[0],
            title="bad",
            status="ready",
            public_inputs={
                "left": {"concepts": {"exact_supply": [1]}},
                "right": {"concepts": {"exact_supply": [2]}},
            },
            expectations=[
                ConceptExpectation(
                    "supply",
                    "equivalent",
                    "exact_supply",
                    [1],
                    [2],
                )
            ],
            boundary_contracts={"public-observable-v1": ["supply"]},
            provenance={"evidence_domain": "unit-test"},
        )


def test_blocked_pair_cannot_fabricate_labels_or_omit_dependency() -> None:
    common: dict[str, Any] = {
        "pair_id": "blocked-test",
        "family": REQUIRED_FAMILIES[2],
        "title": "blocked",
        "status": "dependency_blocked",
        "public_inputs": {
            "left": {
                "concepts": {},
                "blocked_concepts": {"motif": {"dependency": "F1", "reason": "missing"}},
            },
            "right": {
                "concepts": {},
                "blocked_concepts": {"motif": {"dependency": "F1", "reason": "missing"}},
            },
        },
        "boundary_contracts": {"public-observable-v1": ["motif"]},
        "provenance": {"evidence_domain": "unit-test"},
    }
    with pytest.raises(SuiteValidationError, match="cannot fabricate"):
        build_pair(
            **common,
            expectations=[ConceptExpectation("motif", "different", "motif", [1], [2])],
            dependency=DependencyBlock("F1", "missing", "fixture-v1", "true"),
        )
    with pytest.raises(SuiteValidationError, match="requires dependency"):
        build_pair(
            **common,
            expectations=[ConceptExpectation("motif", "different", "motif", None, None)],
        )


def test_hidden_or_future_public_fields_are_rejected() -> None:
    with pytest.raises(SuiteValidationError, match="forbidden hidden/future field"):
        build_pair(
            pair_id="hidden-test",
            family=REQUIRED_FAMILIES[0],
            title="hidden",
            status="ready",
            public_inputs={
                "left": {
                    "concepts": {"exact_supply": [1]},
                    "hidden_refill_order": [1, 2],
                },
                "right": {"concepts": {"exact_supply": [2]}},
            },
            expectations=[ConceptExpectation("supply", "different", "exact_supply", [1], [2])],
            boundary_contracts={"public-observable-v1": ["supply"]},
            provenance={"evidence_domain": "unit-test"},
        )


def test_exact_difference_and_collision_detection(fixture_set: FixtureSet) -> None:
    pair = _pair(fixture_set, "semantic_tile_multiset")
    boundaries = _registry_for(fixture_set)
    probes = default_probe_registry()

    exact = evaluate_pair_boundary(
        pair,
        boundaries.get("public-observable-v1"),
        probes,
    )
    weak = evaluate_pair_boundary(
        pair,
        boundaries.get("public-supply-marginals-v1"),
        probes,
    )

    assert exact["boundary_verdict"] == "retained"
    assert exact["concepts"][0]["observed_relation"] == "different"
    assert exact["concepts"][0]["retained"]
    assert weak["boundary_verdict"] == "information_lost"
    assert weak["concepts"][0]["collision"]
    assert weak["projection_equal"]


def test_exact_equivalence_and_equivalence_violation(fixture_set: FixtureSet) -> None:
    pair = _pair(fixture_set, "tile_id_permutation")
    boundaries = _registry_for(fixture_set)
    probes = default_probe_registry()

    exact = evaluate_pair_boundary(
        pair,
        boundaries.get("public-observable-v1"),
        probes,
    )
    scalar_id = evaluate_pair_boundary(
        pair,
        boundaries.get("scalar-tile-id-v1"),
        probes,
    )

    assert exact["concepts"][0]["observed_relation"] == "equivalent"
    assert exact["concepts"][0]["retained"]
    assert scalar_id["concepts"][0]["equivalence_violation"]
    assert scalar_id["boundary_verdict"] == "information_lost"


def test_all_pairs_are_executable_and_classification_is_complete(
    fixture_set: FixtureSet,
) -> None:
    report = run_suite(fixture_set)
    assert report["summary"]["ready_pairs"] == 14
    assert report["summary"]["dependency_blocked_pairs"] == 0
    assert report["summary"]["classification"] == CLASSIFICATION_FAILED
    assert report["summary"]["exit_code"] == 3
    assert not report["summary"]["errors"]


def test_rust_generated_resolved_dependency_artifact_is_byte_reproducible(
    tmp_path: Path,
) -> None:
    regenerated = tmp_path / "resolved-dependencies-v2.json"
    environment = os.environ.copy()
    environment["CARGO_TARGET_DIR"] = str(ROOT / "target" / "f4-fixture-generator")
    subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "--manifest-path",
            str(GENERATOR_MANIFEST),
            "--",
            str(regenerated),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
    )
    assert regenerated.read_bytes() == RESOLVED_DEPENDENCY_PATH.read_bytes()


def test_resolved_dependency_materialization_is_exact_and_idempotent(
    fixture_set: FixtureSet,
) -> None:
    artifact = load_resolved_dependency_artifact(RESOLVED_DEPENDENCY_PATH)
    materialized = materialize_resolved_dependencies(
        fixture_set,
        artifact,
        artifact_file_blake3=file_blake3(RESOLVED_DEPENDENCY_PATH),
    )

    assert materialized.canonical_hash == fixture_set.canonical_hash
    assert canonical_json(materialized.to_dict()) == canonical_json(fixture_set.to_dict())
    assert artifact["f1"]["classification_scientific_blake3"] == (
        F1_CLASSIFICATION_SCIENTIFIC_BLAKE3
    )
    assert artifact["f1"]["merged_census_scientific_blake3"] == (F1_FORWARD_SCIENTIFIC_BLAKE3)
    assert artifact["f1"]["all_four_pairs_executable"]
    assert artifact["f2"]["source_scientific_blake3"] == (
        "c6076545aa93e78902b739eefef1545a23b8f2dbe44770f427a30969511800e5"
    )
    assert artifact["f3"]["contract_scientific_blake3"] == (
        "db6ac2f9f6ebe2daaa2db603c6c16183512b5d989aed6979e1991e167737633f"
    )


def test_resolved_dependency_artifact_rejects_upstream_drift(tmp_path: Path) -> None:
    values = json.loads(RESOLVED_DEPENDENCY_PATH.read_text())
    values["f1"]["classification_scientific_blake3"] = "0" * 64
    drifted_f1 = tmp_path / "drifted-f1.json"
    drifted_f1.write_text(json.dumps(values))
    with pytest.raises(SuiteValidationError, match="authority chain"):
        load_resolved_dependency_artifact(drifted_f1)

    values = json.loads(RESOLVED_DEPENDENCY_PATH.read_text())
    values["f1"]["relevant_blocks"][0]["active_rows"] -= 1
    drifted_f1_census = tmp_path / "drifted-f1-census.json"
    drifted_f1_census.write_text(json.dumps(values))
    with pytest.raises(SuiteValidationError, match="block census"):
        load_resolved_dependency_artifact(drifted_f1_census)

    values = json.loads(RESOLVED_DEPENDENCY_PATH.read_text())
    values["f1"]["public_action_equivalence_refill_near_match"]["refill_near_match"][
        "return_distribution"
    ]["wildlife_counts"][0] += 1
    drifted_f1_refill = tmp_path / "drifted-f1-refill.json"
    drifted_f1_refill.write_text(json.dumps(values))
    with pytest.raises(SuiteValidationError, match="distribution"):
        load_resolved_dependency_artifact(drifted_f1_refill)

    values = json.loads(RESOLVED_DEPENDENCY_PATH.read_text())
    values["f2"]["source_scientific_blake3"] = "0" * 64
    drifted_f2 = tmp_path / "drifted-f2.json"
    drifted_f2.write_text(json.dumps(values))
    with pytest.raises(SuiteValidationError, match="wrong scientific hash"):
        load_resolved_dependency_artifact(drifted_f2)

    values = json.loads(RESOLVED_DEPENDENCY_PATH.read_text())
    values["f3"]["orbit"][1]["inverse_id"] = 0
    drifted_f3 = tmp_path / "drifted-f3.json"
    drifted_f3.write_text(json.dumps(values))
    with pytest.raises(SuiteValidationError, match="inverse ID"):
        load_resolved_dependency_artifact(drifted_f3)


def test_resolved_long_salmon_pair_uses_exact_remote_component_context(
    fixture_set: FixtureSet,
) -> None:
    pair = _pair(fixture_set, "long_salmon_component_context")
    receipt = pair.evidence["receipt"]["long_salmon_component_context"]
    observation = evaluate_pair_boundary(
        pair,
        _registry_for(fixture_set).get("public-observable-v1"),
        default_probe_registry(),
    )

    assert pair.status == "ready"
    assert pair.dependency is None
    assert (
        receipt["left"]["radius_one_neighborhood"] == (receipt["right"]["radius_one_neighborhood"])
    )
    assert receipt["left"]["salmon_component_size"] == 5
    assert receipt["right"]["salmon_component_size"] == 4
    assert receipt["left"]["maximum_salmon_degree"] <= 2
    assert receipt["right"]["maximum_salmon_degree"] <= 2
    assert observation["boundary_verdict"] == "retained"
    assert observation["concepts"][0]["observed_relation"] == "different"


def test_resolved_component_bridge_pair_distinguishes_merge_from_extension(
    fixture_set: FixtureSet,
) -> None:
    pair = _pair(fixture_set, "component_bridge")
    receipt = pair.evidence["receipt"]["component_bridge"]
    observation = evaluate_pair_boundary(
        pair,
        _registry_for(fixture_set).get("public-observable-v1"),
        default_probe_registry(),
    )

    assert receipt["left"]["merged_source_count"] == 2
    assert receipt["right"]["merged_source_count"] == 1
    assert receipt["left"]["post_component"]["size"] == 3
    assert receipt["right"]["post_component"]["size"] == 5
    assert observation["boundary_verdict"] == "retained"


def test_resolved_motif_conflict_pair_has_equal_score_and_distinct_public_motif(
    fixture_set: FixtureSet,
) -> None:
    pair = _pair(fixture_set, "equal_immediate_different_future_conflict")
    receipt = pair.evidence["receipt"]["equal_immediate_different_future_conflict"]
    observation = evaluate_pair_boundary(
        pair,
        _registry_for(fixture_set).get("public-observable-v1"),
        default_probe_registry(),
    )

    assert receipt["left"]["tile_layout_blake3"] == receipt["right"]["tile_layout_blake3"]
    assert receipt["left"]["score"] == receipt["right"]["score"]
    assert receipt["left"]["score"]["base_total"] == 4
    assert receipt["left"]["motif_conflict"]["wildlife"] == "Hawk"
    assert receipt["left"]["motif_conflict"]["conflicted_coordinates"]
    assert receipt["right"]["motif_conflict"]["wildlife"] == "Salmon"
    assert receipt["right"]["motif_conflict"]["branching_coordinates"]
    assert observation["boundary_verdict"] == "retained"


def test_resolved_public_action_pair_tests_equivalence_and_refill_separately(
    fixture_set: FixtureSet,
) -> None:
    pair = _pair(fixture_set, "public_action_equivalence_refill_near_match")
    receipt = pair.evidence["receipt"]["public_action_equivalence_refill_near_match"]
    registry = _registry_for(fixture_set)
    probes = default_probe_registry()
    exact = evaluate_pair_boundary(
        pair,
        registry.get("public-observable-v1"),
        probes,
    )
    refill = evaluate_pair_boundary(
        pair,
        registry.get(REFILL_DISTRIBUTION_BOUNDARY_ID),
        probes,
    )

    assert (
        receipt["exact_equivalence"]["left_action"]
        != (receipt["exact_equivalence"]["right_action"])
    )
    assert (
        receipt["exact_equivalence"]["left_transition_public_blake3"]
        == (receipt["exact_equivalence"]["right_transition_public_blake3"])
    )
    assert (
        receipt["refill_near_match"]["place_distribution"]
        != (receipt["refill_near_match"]["return_distribution"])
    )
    assert receipt["refill_near_match"]["place_distribution"]["order_free"]
    assert receipt["refill_near_match"]["return_distribution"]["order_free"]
    assert exact["boundary_verdict"] == "retained"
    assert refill["boundary_verdict"] == "retained"
    assert next(
        concept for concept in exact["concepts"] if concept["concept"] == "exact_public_transition"
    )["retained"]
    assert next(
        concept
        for concept in refill["concepts"]
        if concept["concept"] == "refill_near_match_transition"
    )["retained"]


def test_resolved_d6_pair_executes_the_rust_orbit_contract(
    fixture_set: FixtureSet,
) -> None:
    pair = _pair(fixture_set, "d6_transforms")
    receipt = pair.evidence["receipt"]
    observation = evaluate_pair_boundary(
        pair,
        _registry_for(fixture_set).get("public-observable-v1"),
        default_probe_registry(),
    )

    assert pair.status == "ready"
    assert pair.dependency is None
    assert len(receipt["orbit"]) == 12
    assert receipt["source_legal_action_count"] == 540
    assert receipt["every_legal_map_bijective"]
    assert receipt["every_action_round_trips"]
    assert receipt["every_transition_equivariant"]
    assert observation["boundary_verdict"] == "retained"
    assert observation["concepts"][0]["observed_relation"] == "equivalent"


def test_resolved_f2_pairs_distinguish_overflow_and_preserve_exact_sidecar(
    fixture_set: FixtureSet,
) -> None:
    registry = _registry_for(fixture_set)
    probes = default_probe_registry()
    overflow = _pair(
        fixture_set,
        "same_in_radius_different_overflow_consequence",
    )
    compact = _pair(
        fixture_set,
        "same_compact_latent_different_legal_affordance",
    )

    exact = evaluate_pair_boundary(
        overflow,
        registry.get("public-observable-v1"),
        probes,
    )
    clipped = evaluate_pair_boundary(
        overflow,
        registry.get(IN_RADIUS_BOUNDARY_ID),
        probes,
    )
    compact_exact = evaluate_pair_boundary(
        compact,
        registry.get("declared-compact-projection-v1"),
        probes,
    )

    assert (
        overflow.public_inputs["left"]["in_radius_occupied"]
        == (overflow.public_inputs["right"]["in_radius_occupied"])
    )
    assert exact["boundary_verdict"] == "retained"
    assert clipped["boundary_verdict"] == "information_lost"
    assert clipped["concepts"][0]["collision"]
    assert (
        compact.public_inputs["left"]["latent_target"]
        == (compact.public_inputs["right"]["latent_target"])
    )
    assert (
        compact.public_inputs["left"]["exact_overflow"]
        != (compact.public_inputs["right"]["exact_overflow"])
    )
    assert compact_exact["boundary_verdict"] == "retained"
    assert compact_exact["concepts"][0]["observed_relation"] == "different"


def test_open_confidence_labels_recompute_from_exact_evidence(
    fixture_set: FixtureSet,
) -> None:
    pair = _pair(
        fixture_set,
        "ambiguous_confidence_set_vs_distinguishable_winner",
    )
    expectation = pair.expectations[0]
    left = confidence_set_evidence(
        **{
            key: pair.evidence["left"][key]
            for key in ("means", "stddevs", "samples", "action_hashes")
        }
    )
    right = confidence_set_evidence(
        **{
            key: pair.evidence["right"][key]
            for key in ("means", "stddevs", "samples", "action_hashes")
        }
    )

    assert left["confidence_set_size"] == 4
    assert not left["distinguishable_winner"]
    assert right["confidence_set_size"] == 1
    assert right["distinguishable_winner"]
    assert expectation.left_label["confidence_set_membership"] == left["confidence_set_membership"]
    assert (
        expectation.right_label["confidence_set_membership"] == right["confidence_set_membership"]
    )
    assert "teacher" not in canonical_json(pair.public_inputs).lower()
    assert pair.provenance["sealed_test_opened"] is False


class CopyPluginBoundary(BoundaryAdapter):
    boundary_id = "unit-copy-plugin-v1"
    description = "Unit-test plugin boundary."

    def project(self, public_input: dict[str, Any]) -> dict[str, Any]:
        return {"concepts": public_input["concepts"]}


def test_boundary_plugin_registration_requires_no_runner_change(
    fixture_set: FixtureSet,
) -> None:
    registry = BoundaryRegistry()
    registry.register(CopyPluginBoundary())
    with pytest.raises(SuiteValidationError, match="duplicate"):
        registry.register(CopyPluginBoundary())

    pair = _pair(fixture_set, "multiplicity_descendant_distribution")
    result = evaluate_pair_boundary(
        pair,
        registry.get("unit-copy-plugin-v1"),
        default_probe_registry(),
    )
    assert result["boundary_verdict"] == "retained"
    assert result["concepts"][0]["probe"] == "component"


def test_current_v2_and_graded_boundary_source_adapters() -> None:
    record = np.zeros(1, dtype=_RECORD_DTYPE)
    record["board_counts"][0] = [1, 0, 0, 0]
    record["board_entities"][0, 0, 0] = [1, 2, 3, 255, 0, 1, 255, 0]
    record["market_entities"][0] = 255
    position = source_from_v2_position_record(record)
    assert position["kind"] == "v2_position_record"
    assert len(position["concepts"]["occupancy"][0]["values"]) == 1

    batch = SimpleNamespace(
        board_entities=np.zeros((1, 4, 23, 31), dtype=np.float32),
        board_mask=np.zeros((1, 4, 23), dtype=np.bool_),
        market_entities=np.zeros((1, 4, 31), dtype=np.float32),
        market_mask=np.array([[True, False, False, False]]),
    )
    tensors = source_from_dataset_batch(batch)
    assert tensors["kind"] == "current_dataset_tensors"
    assert tensors["concepts"]["staged_market"]["shape"] == [1, 31]

    graded = SimpleNamespace(
        candidate_mask=np.array([[True, False]]),
        board_entities=batch.board_entities,
        board_mask=batch.board_mask,
        market_entities=batch.market_entities,
        market_mask=batch.market_mask,
        staged_market_entities=np.zeros((1, 2, 4, 31), dtype=np.float32),
        staged_market_mask=np.array([[[True, False, False, False]] * 2]).reshape(1, 2, 4),
        public_supply=np.zeros((1, 30), dtype=np.float32),
        action_features=np.zeros((1, 2, 140), dtype=np.float32),
    )
    raw = source_from_graded_oracle_batch(graded)
    assert raw["kind"] == "graded_oracle_raw"
    assert raw["concepts"]["legal_mask"]["values"] == [True, False]
    assert set(raw["concepts"]["staged_market"]) == {"parent", "staged"}

    factors = source_from_graded_factor_array(np.zeros((1, 2, 7, 192), dtype=np.float32))
    assert factors["kind"] == "graded_oracle_factors"
    assert factors["concepts"]["action_edit"]["shape"] == [7, 192]

    compact = source_from_compact_projection(
        latent_target=[1, 2],
        retained_concepts={"legal_mask": [True, False]},
    )
    assert compact["kind"] == "compact_projection"
    assert compact["concepts"]["legal_mask"] == [True, False]


def test_boundary_adapters_normalize_existing_schema_kinds() -> None:
    registry = default_boundary_registry()
    cases = (
        ("v2-position-record-v1", {"kind": "v2_position_record"}),
        ("current-dataset-tensors-v1", {"kind": "current_dataset_tensors"}),
        ("graded-oracle-raw-v1", {"kind": "graded_oracle_raw"}),
        ("graded-oracle-factors-v1", {"kind": "graded_oracle_factors"}),
        ("declared-compact-projection-v1", {"kind": "compact_projection"}),
    )
    for boundary_id, source in cases:
        source["concepts"] = {"legal_mask": [True]}
        projected = registry.get(boundary_id).project(source)
        assert projected["concepts"]["legal_mask"] == [True]


def test_deterministic_report_hash_excludes_paths_and_timestamps(
    fixture_set: FixtureSet,
) -> None:
    first = run_suite(fixture_set)
    second = run_suite(fixture_set)
    assert first["scientific_blake3"] == second["scientific_blake3"]
    assert canonical_json(first) == canonical_json(second)

    left = {
        "result": {"score": 1},
        "path": "/tmp/one",
        "timestamp": 1,
        "hostname": "john1",
    }
    right = {
        "result": {"score": 1},
        "path": "/elsewhere/two",
        "timestamp": 999,
        "hostname": "john4",
    }
    assert scientific_blake3(left) == scientific_blake3(right)


def test_json_markdown_and_jsonl_outputs_are_deterministic(
    fixture_set: FixtureSet,
    tmp_path: Path,
) -> None:
    report = run_suite(fixture_set)
    json_path = tmp_path / "summary.json"
    markdown_path = tmp_path / "summary.md"
    jsonl_path = tmp_path / "pairs.jsonl"
    write_outputs(
        report,
        json_output=json_path,
        markdown_output=markdown_path,
        jsonl_output=jsonl_path,
    )

    assert json.loads(json_path.read_text())["scientific_blake3"] == report["scientific_blake3"]
    assert markdown_path.read_text() == render_markdown(report)
    rows = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
    assert len(rows) == 14
    assert rows[0]["family"] == REQUIRED_FAMILIES[0]


def test_missing_required_family_is_invalid_not_false_success(
    fixture_set: FixtureSet,
) -> None:
    draft = FixtureSet(
        schema_version=fixture_set.schema_version,
        fixture_set_id=fixture_set.fixture_set_id,
        pairs=fixture_set.pairs[:-1],
        canonical_hash="",
    )
    incomplete = FixtureSet(
        schema_version=draft.schema_version,
        fixture_set_id=draft.fixture_set_id,
        pairs=draft.pairs,
        canonical_hash=draft.expected_hash(),
    )
    report = run_suite(incomplete)
    assert report["summary"]["classification"] == CLASSIFICATION_INVALID
    assert report["summary"]["exit_code"] == 4
    assert any(
        "required family registry mismatch" in error for error in report["summary"]["errors"]
    )


def test_missing_required_concept_is_invalid_not_false_success() -> None:
    pair = build_pair(
        pair_id="missing-required-concept",
        family=REQUIRED_FAMILIES[0],
        title="missing concept",
        status="ready",
        public_inputs={
            "left": {"concepts": {}},
            "right": {"concepts": {}},
        },
        expectations=[ConceptExpectation("supply", "different", "exact_supply", [1], [2])],
        boundary_contracts={"public-observable-v1": ["supply"]},
        provenance={"evidence_domain": "unit-test"},
    )
    observation = evaluate_pair_boundary(
        pair,
        default_boundary_registry().get("public-observable-v1"),
        default_probe_registry(),
    )
    assert observation["status"] == "invalid"
    assert observation["missing_required_concepts"] == ["supply"]


def test_cli_returns_nonzero_for_complete_failure_and_invalid_suites(
    fixture_set: FixtureSet,
    tmp_path: Path,
) -> None:
    json_path = tmp_path / "report.json"
    assert (
        main(
            [
                "--fixtures",
                str(FIXTURE_PATH),
                "--json-out",
                str(json_path),
                "--stdout",
                "none",
            ]
        )
        == 3
    )
    assert json.loads(json_path.read_text())["summary"]["classification"] == (CLASSIFICATION_FAILED)

    malformed = tmp_path / "malformed.json"
    values = fixture_set.to_dict()
    values["pairs"] = values["pairs"][:-1]
    malformed.write_text(json.dumps(values))
    assert main(["--fixtures", str(malformed), "--stdout", "none"]) == 4
