"""Fault-injection tests for exact, pre-outcome Rival panel plans."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any

from cascadiav3.rival.manifest import (
    deployment_design_identity,
    load_root_manifest,
    validate_root_manifest,
)
from cascadiav3.rival.panel_plan import (
    PanelPlanError,
    TerminalPanelUnit,
    TerminalUnitExpectation,
    load_terminal_panel_plan,
    validate_terminal_panel_plan,
)
from cascadiav3.rival.schema import canonical_json_bytes, sha256_hex

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "cascadiav3/tests/fixtures/rival/panel_manifest.json"


def _digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def _unit(
    record: dict[str, Any],
    *,
    unit_index: int,
    fidelity: str,
    target_seat: int = 2,
) -> dict[str, Any]:
    challenger = record["candidate_selection_entries"][1]
    return {
        "unit_index": unit_index,
        "fidelity": fidelity,
        "target_seat": target_seat,
        "challenger_candidate_occurrence_id": (challenger["candidate_action_occurrence_id"]),
        "challenger_action_content_id": challenger["action_content_id"],
        "incumbent_post_action_memory_sha256": _digest(f"incumbent-memory:{unit_index}"),
        "challenger_post_action_memory_sha256": _digest(f"challenger-memory:{unit_index}"),
    }


def _plan_record(
    record: dict[str, Any], *, panel_kind: str, fidelity: str, count: int
) -> dict[str, Any]:
    plan = {
        "schema_id": "cascadiav3.rival_terminal_panel_plan.v1",
        "plan_id": f"fixture:{panel_kind.lower()}-plan",
        "manifest_id": record["manifest_id"],
        "root_id": record["root_id"],
        "ruleset_identity": record["ruleset_identity"],
        "source_game_identity_sha256": record["source_game_identity_sha256"],
        "candidate_set_identity": record["candidate_set_identity"],
        "incumbent_policy_identity": record["incumbent_policy_identity"],
        "incumbent_candidate_occurrence_id": (record["incumbent_candidate_occurrence_id"]),
        "incumbent_action_content_id": record["incumbent_action_id"],
        "sampler_identity": record["sampler_identity"],
        "policy_rng_factory_identity": record["policy_rng_factory_identity"],
        "panel_kind": panel_kind,
        "units": [_unit(record, unit_index=index, fidelity=fidelity) for index in range(count)],
    }
    plan["content_sha256"] = sha256_hex(plan)
    return plan


def _rehash_manifest(record: dict[str, Any]) -> dict[str, Any]:
    record.pop("content_sha256", None)
    record["deployment_design_sha256"] = deployment_design_identity(record)
    record["content_sha256"] = sha256_hex(record)
    return record


def _high_only_fixture() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    record = json.loads(FIXTURE.read_text(encoding="utf-8"))
    record.update(
        {
            "inference_mode": "high_fidelity_only",
            "required_panels": ["S", "H"],
            "forbidden_panels": ["L", "A"],
            "beta_cv": 0.0,
            "multifidelity_claim": False,
            "low_policy_identity": None,
            "coefficient_identity": None,
            "expected_l": 0,
            "low_expectation_id": None,
            "low_law_h_id": None,
            "low_law_l_id": None,
            "max_abs_beta": None,
        }
    )
    plans = {
        "S": _plan_record(record, panel_kind="S", fidelity="high", count=2),
        "H": _plan_record(record, panel_kind="H", fidelity="high", count=2),
    }
    record["panel_identities"] = {
        "S": "sha256:" + plans["S"]["content_sha256"],
        "H": "sha256:" + plans["H"]["content_sha256"],
        "L": None,
        "A": None,
    }
    return _rehash_manifest(record), plans


class RivalTerminalPanelPlanTest(unittest.TestCase):
    def test_exact_high_only_plans_are_manifest_bound_and_issue_expectations(self) -> None:
        record, plan_records = _high_only_fixture()
        manifest = validate_root_manifest(record)
        plans = {
            panel: validate_terminal_panel_plan(plan, manifest=manifest)
            for panel, plan in plan_records.items()
        }
        self.assertNotEqual(plans["S"].identity, plans["H"].identity)
        self.assertEqual(plans["S"].identity, manifest.panel_identity("S"))
        expectation = plans["H"].high_fidelity_expectation(1)
        expectation.validate(manifest)
        self.assertEqual(expectation.panel_kind, "H")
        self.assertEqual(expectation.unit_index, 1)
        self.assertNotEqual(
            plans["S"].high_fidelity_expectation(1).unit_id,
            expectation.unit_id,
        )

    def test_posthoc_unit_or_plan_mutation_is_not_a_valid_capability(self) -> None:
        record, plan_records = _high_only_fixture()
        manifest = validate_root_manifest(record)
        plan = validate_terminal_panel_plan(plan_records["S"], manifest=manifest)
        with self.assertRaisesRegex(PanelPlanError, "does not match"):
            replace(plan.units[0], target_seat=3)

        direct_unit = TerminalPanelUnit(
            unit_index=plan.units[0].unit_index,
            fidelity=plan.units[0].fidelity,
            target_seat=3,
            challenger_candidate_occurrence_id=(plan.units[0].challenger_candidate_occurrence_id),
            challenger_action_content_id=plan.units[0].challenger_action_content_id,
            incumbent_post_action_memory_sha256=(plan.units[0].incumbent_post_action_memory_sha256),
            challenger_post_action_memory_sha256=(
                plan.units[0].challenger_post_action_memory_sha256
            ),
        )
        with self.assertRaisesRegex(PanelPlanError, "artifact validator"):
            replace(plan, units=(direct_unit, *plan.units[1:]))

        with self.assertRaises(PanelPlanError):
            TerminalPanelUnit(  # type: ignore[arg-type]
                unit_index=0,
                fidelity="high",
                target_seat=2,
                challenger_candidate_occurrence_id=(
                    plan.units[0].challenger_candidate_occurrence_id
                ),
                challenger_action_content_id=plan.units[0].challenger_action_content_id,
                incumbent_post_action_memory_sha256=_digest("i"),
                challenger_post_action_memory_sha256=_digest("c"),
            ).require_validated_artifact()

    def test_importable_sentinels_cannot_forge_units_plans_or_expectations(self) -> None:
        import cascadiav3.rival.panel_plan as panel_module

        self.assertFalse(hasattr(panel_module, "_PLAN_PROOF"))
        self.assertFalse(hasattr(panel_module, "_EXPECTATION_PROOF"))

        record, plan_records = _high_only_fixture()
        manifest = validate_root_manifest(record)
        plan = validate_terminal_panel_plan(plan_records["H"], manifest=manifest)
        expectation = plan.high_fidelity_expectation(0)

        with self.assertRaisesRegex(PanelPlanError, "does not match"):
            replace(expectation, target_seat=(expectation.target_seat + 1) % 4)

        direct = TerminalUnitExpectation(
            panel_kind=expectation.panel_kind,
            panel_id=expectation.panel_id,
            unit_index=expectation.unit_index,
            fidelity="high",
            target_seat=expectation.target_seat,
            challenger_candidate_occurrence_id=(expectation.challenger_candidate_occurrence_id),
            challenger_action_content_id=expectation.challenger_action_content_id,
            incumbent_post_action_memory_sha256=(expectation.incumbent_post_action_memory_sha256),
            challenger_post_action_memory_sha256=(expectation.challenger_post_action_memory_sha256),
            _unit_record_sha256=expectation._unit_record_sha256,
        )
        with self.assertRaisesRegex(PanelPlanError, "artifact validator"):
            direct.validate(manifest)

    def test_rehashed_posthoc_plan_still_fails_semantic_allocation(self) -> None:
        record, plan_records = _high_only_fixture()
        changed_plan = copy.deepcopy(plan_records["H"])
        changed_plan["units"].pop()
        changed_plan.pop("content_sha256")
        changed_plan["content_sha256"] = sha256_hex(changed_plan)
        record["panel_identities"]["H"] = "sha256:" + changed_plan["content_sha256"]
        _rehash_manifest(record)
        with self.assertRaisesRegex(PanelPlanError, "exact conditional allocation"):
            validate_terminal_panel_plan(
                changed_plan,
                manifest=validate_root_manifest(record),
            )

    def test_every_cross_manifest_identity_is_rechecked_after_rehash(self) -> None:
        for field in (
            "manifest_id",
            "root_id",
            "ruleset_identity",
            "source_game_identity_sha256",
            "candidate_set_identity",
            "incumbent_policy_identity",
            "incumbent_candidate_occurrence_id",
            "incumbent_action_content_id",
            "sampler_identity",
            "policy_rng_factory_identity",
        ):
            record, plans = _high_only_fixture()
            changed = copy.deepcopy(plans["S"])
            changed[field] = _digest(field)
            changed.pop("content_sha256")
            changed["content_sha256"] = sha256_hex(changed)
            record["panel_identities"]["S"] = "sha256:" + changed["content_sha256"]
            _rehash_manifest(record)
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(PanelPlanError, "does not join"),
            ):
                validate_terminal_panel_plan(
                    changed,
                    manifest=validate_root_manifest(record),
                )

    def test_multifidelity_plans_freeze_all_conditional_units(self) -> None:
        record = json.loads(FIXTURE.read_text(encoding="utf-8"))
        plans = {
            "S": _plan_record(record, panel_kind="S", fidelity="low", count=2),
            "H": _plan_record(
                record,
                panel_kind="H",
                fidelity="paired_high_low",
                count=2,
            ),
            "L": _plan_record(record, panel_kind="L", fidelity="low", count=4),
        }
        record["panel_identities"] = {
            panel: "sha256:" + plan["content_sha256"] for panel, plan in plans.items()
        } | {"A": None}
        _rehash_manifest(record)
        manifest = validate_root_manifest(record)
        validated = {
            panel: validate_terminal_panel_plan(plan, manifest=manifest)
            for panel, plan in plans.items()
        }
        self.assertEqual(len(validated["S"].units), manifest.expected_s)
        self.assertEqual(len(validated["H"].units), manifest.expected_h)
        self.assertEqual(len(validated["L"].units), manifest.expected_l)
        with self.assertRaisesRegex(PanelPlanError, "only high-fidelity"):
            validated["H"].high_fidelity_expectation(0)

    def test_strict_loader_rejects_duplicate_keys(self) -> None:
        record, plans = _high_only_fixture()
        rendered = json.dumps(plans["S"], sort_keys=True, separators=(",", ":"))
        canonical = rendered + "\n"
        duplicate = rendered[:-1] + ',"schema_id":"duplicate"}'
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "manifest.json"
            manifest_bytes = canonical_json_bytes(record) + b"\n"
            manifest_path.write_bytes(manifest_bytes)
            manifest = load_root_manifest(
                manifest_path,
                expected_file_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                expected_content_sha256=record["content_sha256"],
            )
            valid_path = Path(temporary) / "valid-plan.json"
            valid_path.write_text(canonical, encoding="utf-8")
            loaded = load_terminal_panel_plan(
                valid_path,
                manifest=manifest,
                expected_file_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            )
            self.assertEqual(loaded.identity, manifest.panel_identity("S"))

            path = Path(temporary) / "plan.json"
            path.write_text(duplicate, encoding="utf-8")
            with self.assertRaisesRegex(PanelPlanError, "duplicate JSON key"):
                load_terminal_panel_plan(
                    path,
                    manifest=manifest,
                    expected_file_sha256=hashlib.sha256(duplicate.encode("utf-8")).hexdigest(),
                )


if __name__ == "__main__":
    unittest.main()
