"""Adversarial CPU tests for the Rust terminal-evidence join."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any

from cascadiav3.rival.appeals import (
    AppealError,
    EvidenceDomain,
    HighFidelityAppealStateMachine,
)
from cascadiav3.rival.manifest import (
    CandidateSelectionEntry,
    candidate_set_identity,
    deployment_design_identity,
    load_root_manifest,
    validate_root_manifest,
)
from cascadiav3.rival.panel_plan import (
    PanelPlanError,
    TerminalPanelPlan,
    TerminalUnitExpectation,
    validate_terminal_panel_plan,
)
from cascadiav3.rival.schema import canonical_json_bytes, sha256_hex
from cascadiav3.rival.terminal_evidence import (
    MAX_TERMINAL_PAIR_LEDGER_BYTES,
    MAX_VERIFIER_STDERR_BYTES,
    MAX_VERIFIER_STDOUT_BYTES,
    RustTerminalRowResolver,
    RustTerminalVerifier,
    TerminalEvidenceError,
    TerminalEvidenceReference,
    VerifiedTerminalPairEvidence,
    receipt_identity,
)
from test_rival_cohorts import CHALLENGER_OCCURRENCE_ID, verified_high_design

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "cascadiav3/tests/fixtures/rival/panel_manifest.json"
SHA_A = "sha256:" + "a" * 64


def _file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fake_verifier(path: Path) -> None:
    source = f"""#!{sys.executable}
import hashlib
import json
import pathlib
import os
import sys
import time

expected_environment = {{
    "CASCADIA_CPU_ONLY_TESTS": "1",
    "CASCADIA_DEVICE": "cpu",
    "CUDA_VISIBLE_DEVICES": "",
    "LC_ALL": "C",
}}
if any(os.environ.get(key) != value for key, value in expected_environment.items()):
    print("unsafe verifier environment", file=sys.stderr)
    raise SystemExit(63)
if os.environ.get("HOME") is not None:
    print("inherited verifier environment", file=sys.stderr)
    raise SystemExit(63)
if len(sys.argv) != 5 or sys.argv[1] != "verify-terminal-pair":
    print("bad invocation", file=sys.stderr)
    raise SystemExit(64)
ledger_path = pathlib.Path(sys.argv[2])
ledger_bytes = ledger_path.read_bytes()
ledger = json.loads(ledger_bytes)
if ledger.get("sleep_seconds"):
    time.sleep(ledger["sleep_seconds"])
if ledger.get("exit_code"):
    print("fixture rejection", file=sys.stderr)
    raise SystemExit(ledger["exit_code"])
if ledger.get("fork_descendant"):
    descendant = os.fork()
    pathlib.Path(ledger["descendant_started_path"]).write_text("started")
    if descendant == 0:
        time.sleep(ledger["descendant_survival_delay"])
        pathlib.Path(ledger["descendant_survived_path"]).write_text("survived")
        raise SystemExit(0)
    time.sleep(60)
if ledger.get("stdout_bytes"):
    sys.stdout.buffer.write(b"x" * ledger["stdout_bytes"])
    sys.stdout.buffer.flush()
if ledger.get("stderr_bytes"):
    sys.stderr.buffer.write(b"x" * ledger["stderr_bytes"])
    sys.stderr.buffer.flush()
receipt = dict(ledger["receipt_fields"])
receipt["schema_id"] = "cascadiav3.rival_verified_terminal_pair_receipt.v1"
receipt["verifier_contract_id"] = "cascadia-rival.verify-terminal-pair.v1"
receipt["verifier_executable_sha256"] = "sha256:" + hashlib.sha256(
    pathlib.Path(sys.argv[0]).read_bytes()
).hexdigest()
receipt["ledger_file_sha256"] = "sha256:" + hashlib.sha256(ledger_bytes).hexdigest()
encoded = json.dumps(
    receipt, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
).encode("utf-8")
receipt["receipt_sha256"] = "sha256:" + hashlib.sha256(encoded).hexdigest()
if (
    sys.argv[3] != receipt["pair_sha256"]
    or sys.argv[4] != receipt["parent_manifest_sha256"]
):
    print("fixture pin mismatch", file=sys.stderr)
    raise SystemExit(65)
if ledger.get("corrupt_receipt_hash"):
    receipt["receipt_sha256"] = "sha256:" + "0" * 64
if ledger.get("unknown_field"):
    receipt["unknown"] = True
if ledger.get("mutate_ledger"):
    with ledger_path.open("ab") as handle:
        handle.write(b" ")
if ledger.get("write_stderr"):
    print("unexpected warning", file=sys.stderr)
rendered = json.dumps(receipt, sort_keys=True, separators=(",", ":"))
if ledger.get("duplicate_output"):
    rendered = rendered[:-1] + ',"schema_id":"duplicate"}}'
print(rendered)
"""
    path.write_text(source, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _panel_plan_record(
    record: dict[str, Any],
    panel: str,
    *,
    unit_index: int = 0,
    target_seat: int = 2,
    incumbent_memory_sha256: str = "sha256:" + "5" * 64,
    challenger_memory_sha256: str = "sha256:" + "6" * 64,
) -> dict[str, Any]:
    challenger = record["candidate_selection_entries"][1]
    unit = {
        "unit_index": unit_index,
        "fidelity": "high",
        "target_seat": target_seat,
        "challenger_candidate_occurrence_id": (challenger["candidate_action_occurrence_id"]),
        "challenger_action_content_id": challenger["action_content_id"],
        "incumbent_post_action_memory_sha256": incumbent_memory_sha256,
        "challenger_post_action_memory_sha256": challenger_memory_sha256,
    }
    plan = {
        "schema_id": "cascadiav3.rival_terminal_panel_plan.v1",
        "plan_id": f"fixture:{panel.lower()}-panel-plan",
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
        "panel_kind": panel,
        "units": [unit],
    }
    plan["content_sha256"] = sha256_hex(plan)
    return plan


def _manifest_record_for_executable(
    executable: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    record = json.loads(FIXTURE.read_text(encoding="utf-8"))
    record.pop("content_sha256")
    record.pop("deployment_design_sha256")
    record.update(
        {
            "inference_mode": "high_fidelity_only",
            "required_panels": ["S", "H"],
            "forbidden_panels": ["L", "A"],
            "panel_identities": {
                "S": "sha256:" + "1" * 64,
                "H": "sha256:" + "2" * 64,
                "L": None,
                "A": None,
            },
            "beta_cv": 0.0,
            "multifidelity_claim": False,
            "low_policy_identity": None,
            "coefficient_identity": None,
            "expected_l": 0,
            "expected_s": 1,
            "expected_h": 1,
            "low_expectation_id": None,
            "low_law_h_id": None,
            "low_law_l_id": None,
            "max_abs_beta": None,
            "terminal_verifier_executable_sha256": _file_sha256(executable),
        }
    )
    record["candidate_selection_entries"][1]["expected_s"] = 1
    plans = {panel: _panel_plan_record(record, panel) for panel in ("S", "H")}
    record["panel_identities"] = {
        "S": "sha256:" + plans["S"]["content_sha256"],
        "H": "sha256:" + plans["H"]["content_sha256"],
        "L": None,
        "A": None,
    }
    record["deployment_design_sha256"] = deployment_design_identity(record)
    record["content_sha256"] = sha256_hex(record)
    return record, plans


def _expectation(plans: dict[str, TerminalPanelPlan], panel: str = "S") -> TerminalUnitExpectation:
    return plans[panel].high_fidelity_expectation(0)


def _receipt_fields(record: dict[str, Any], expectation: TerminalUnitExpectation) -> dict[str, Any]:
    world_prefix = "7" if expectation.panel_kind == "S" else "9"
    other_world_prefix = "8" if expectation.panel_kind == "S" else "0"
    challenger_branch_ordinal = next(
        index
        for index, candidate in enumerate(record["candidate_selection_entries"])
        if candidate["candidate_action_occurrence_id"]
        == expectation.challenger_candidate_occurrence_id
    )
    return {
        "pair_sha256": "sha256:" + "4" * 64,
        "parent_manifest_sha256": "sha256:" + record["content_sha256"],
        "ruleset_identity_sha256": record["ruleset_identity"],
        "source_game_identity_sha256": record["source_game_identity_sha256"],
        "scenario_sampler_identity_sha256": record["sampler_identity"],
        "continuation_policy_identity_sha256": record["incumbent_policy_identity"],
        "policy_rng_factory_identity_sha256": record["policy_rng_factory_identity"],
        "source_public_root_id": record["root_id"],
        "source_rules_menu_hash": record["rules_menu_hash"],
        "source_candidate_menu_hash": record["incumbent_menu_hash"],
        "panel_id": expectation.panel_id,
        "unit_index": expectation.unit_index,
        "fidelity": expectation.fidelity,
        "target_seat": expectation.target_seat,
        "incumbent_candidate_occurrence_id": record["incumbent_candidate_occurrence_id"],
        "challenger_candidate_occurrence_id": (expectation.challenger_candidate_occurrence_id),
        "challenger_branch_ordinal": challenger_branch_ordinal,
        "incumbent_action_content_id": record["incumbent_action_id"],
        "challenger_action_content_id": expectation.challenger_action_content_id,
        "incumbent_post_action_memory_sha256": (expectation.incumbent_post_action_memory_sha256),
        "challenger_post_action_memory_sha256": (expectation.challenger_post_action_memory_sha256),
        "incumbent_world_redetermination_seed_sha256": ("sha256:" + world_prefix * 64),
        "challenger_world_redetermination_seed_sha256": ("sha256:" + other_world_prefix * 64),
        "target_score_difference": 11,
        "proxy_policy": True,
        "beta_cv_required": 0,
    }


class RustTerminalVerifierTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.executable = self.root / "fake-rival-contract"
        _write_fake_verifier(self.executable)
        self.record, plan_records = _manifest_record_for_executable(self.executable)
        self.manifest = self._load_manifest_record(self.record, "manifest.json")
        self.plans = {
            panel: validate_terminal_panel_plan(plan, manifest=self.manifest)
            for panel, plan in plan_records.items()
        }
        self.verifier = RustTerminalVerifier(
            executable=self.executable,
            manifest=self.manifest,
            timeout_seconds=2.0,
        )
        self.addCleanup(self.verifier.close)

    def _load_manifest_record(self, record: dict[str, Any], name: str):
        path = self.root / name
        path.write_bytes(canonical_json_bytes(record) + b"\n")
        return load_root_manifest(
            path,
            expected_file_sha256=_file_sha256(path),
            expected_content_sha256="sha256:" + record["content_sha256"],
        )

    def _write_ledger(
        self,
        expectation: TerminalUnitExpectation,
        *,
        receipt_overrides: dict[str, Any] | None = None,
        controls: dict[str, Any] | None = None,
        name: str = "pair.json",
    ) -> Path:
        receipt = _receipt_fields(self.record, expectation)
        receipt.update(receipt_overrides or {})
        payload = {"receipt_fields": receipt, **(controls or {})}
        path = self.root / name
        path.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        return path

    def test_receipt_identity_validates_and_returns_declared_canonical_self_hash(self) -> None:
        expectation = _expectation(self.plans)
        ledger = self._write_ledger(expectation, name="receipt-identity.json")
        receipt = _receipt_fields(self.record, expectation)
        receipt.update(
            {
                "schema_id": "cascadiav3.rival_verified_terminal_pair_receipt.v1",
                "verifier_contract_id": "cascadia-rival.verify-terminal-pair.v1",
                "verifier_executable_sha256": _file_sha256(self.executable),
                "ledger_file_sha256": _file_sha256(ledger),
            }
        )
        receipt["receipt_sha256"] = (
            "sha256:" + hashlib.sha256(canonical_json_bytes(receipt)).hexdigest()
        )

        declared = receipt["receipt_sha256"]
        self.assertEqual(receipt_identity(receipt), declared)
        self.assertNotEqual(
            declared,
            "sha256:" + hashlib.sha256(canonical_json_bytes(receipt)).hexdigest(),
        )

        corrupt = dict(receipt)
        corrupt["receipt_sha256"] = "sha256:" + "0" * 64
        with self.assertRaisesRegex(TerminalEvidenceError, "content hash mismatch"):
            receipt_identity(corrupt)

    def test_verified_proxy_receipts_are_the_only_numeric_row_path(self) -> None:
        selection_expectation = _expectation(self.plans, "S")
        selection = self.verifier.verify(
            self._write_ledger(selection_expectation, name="selection.json"),
            expectation=selection_expectation,
            expected_pair_sha256="sha256:" + "4" * 64,
        )
        selection_row = selection.as_selection_row()
        self.assertEqual(selection.evidence_domain, EvidenceDomain.CPU_PROXY_REFERENCE)
        self.assertEqual(selection_row.selection_score, 11.0)
        self.assertEqual(
            selection_row.challenger_id,
            selection_expectation.challenger_candidate_occurrence_id,
        )
        self.assertFalse(selection_row.evidence_domain is EvidenceDomain.PRODUCTION_TERMINAL)
        with self.assertRaisesRegex(TerminalEvidenceError, "only an H receipt"):
            selection.as_high_only_h_row()

        high_expectation = _expectation(self.plans, "H")
        high = self.verifier.verify(
            self._write_ledger(high_expectation, name="high.json"),
            expectation=high_expectation,
            expected_pair_sha256="sha256:" + "4" * 64,
        )
        high_row = high.as_high_only_h_row()
        self.assertEqual(high_row.high_difference, 11.0)
        self.assertEqual(high_row.evidence_domain, EvidenceDomain.CPU_PROXY_REFERENCE)
        with self.assertRaisesRegex(TerminalEvidenceError, "only an S receipt"):
            high.as_selection_row()

        unsealed = VerifiedTerminalPairEvidence(
            expectation=selection.expectation,
            ledger_file_sha256=selection.ledger_file_sha256,
            pair_sha256=selection.pair_sha256,
            receipt_sha256=selection.receipt_sha256,
            target_score_difference=selection.target_score_difference,
            world_redetermination_seed_sha256s=(selection.world_redetermination_seed_sha256s),
            evidence_domain=selection.evidence_domain,
            _validation_capability=None,
        )
        with self.assertRaisesRegex(TerminalEvidenceError, "must be produced"):
            unsealed.as_selection_row()
        with self.assertRaisesRegex(TerminalEvidenceError, "does not match"):
            replace(selection, target_score_difference=12)

    def test_s_and_h_receipts_cannot_reuse_a_redetermination_seed_commitment(
        self,
    ) -> None:
        selection_expectation = _expectation(self.plans, "S")
        selection = self.verifier.verify(
            self._write_ledger(selection_expectation, name="world-selection.json"),
            expectation=selection_expectation,
            expected_pair_sha256="sha256:" + "4" * 64,
        )
        self.assertEqual(
            selection.expectation.challenger_candidate_occurrence_id,
            CHALLENGER_OCCURRENCE_ID,
        )

        high_expectation = _expectation(self.plans, "H")
        distinct_high = self.verifier.verify(
            self._write_ledger(
                high_expectation,
                receipt_overrides={"pair_sha256": "sha256:" + "3" * 64},
                name="world-high-distinct.json",
            ),
            expectation=high_expectation,
            expected_pair_sha256="sha256:" + "3" * 64,
        )
        self.assertNotEqual(selection.pair_sha256, distinct_high.pair_sha256)
        self.assertNotEqual(selection.receipt_sha256, distinct_high.receipt_sha256)
        normal = HighFidelityAppealStateMachine(
            design=verified_high_design(expected_s=1, expected_h=1)
        )
        normal.add_selection(selection.as_selection_row())
        normal.freeze_challenger(CHALLENGER_OCCURRENCE_ID)
        normal.add_h(distinct_high.as_high_only_h_row())
        self.assertEqual(normal.operational_accounting().attempted_total, 2)

        reused_seed = selection.world_redetermination_seed_sha256s[0]
        overlapping_high = self.verifier.verify(
            self._write_ledger(
                high_expectation,
                receipt_overrides={
                    "pair_sha256": "sha256:" + "2" * 64,
                    "incumbent_world_redetermination_seed_sha256": reused_seed,
                },
                name="world-high-overlap.json",
            ),
            expectation=high_expectation,
            expected_pair_sha256="sha256:" + "2" * 64,
        )
        self.assertNotEqual(selection.pair_sha256, overlapping_high.pair_sha256)
        self.assertNotEqual(selection.receipt_sha256, overlapping_high.receipt_sha256)
        overlapping = HighFidelityAppealStateMachine(
            design=verified_high_design(expected_s=1, expected_h=1)
        )
        overlapping.add_selection(selection.as_selection_row())
        overlapping.freeze_challenger(CHALLENGER_OCCURRENCE_ID)
        with self.assertRaisesRegex(AppealError, "world-redetermination seed commitment reused"):
            overlapping.add_h(overlapping_high.as_high_only_h_row())

    def test_every_manifest_and_unit_join_field_fails_closed(self) -> None:
        expectation = _expectation(self.plans)
        substitutions: dict[str, Any] = {
            "parent_manifest_sha256": SHA_A,
            "ruleset_identity_sha256": "sha256:" + "b" * 64,
            "source_game_identity_sha256": "sha256:" + "b" * 64,
            "scenario_sampler_identity_sha256": "sha256:" + "b" * 64,
            "continuation_policy_identity_sha256": "sha256:" + "b" * 64,
            "policy_rng_factory_identity_sha256": "sha256:" + "b" * 64,
            "source_public_root_id": ("cascadiav3.rival_public_root.v1:sha256:" + "b" * 64),
            "source_rules_menu_hash": ("cascadiav3.rival_rules_menu.v1:sha256:" + "b" * 64),
            "source_candidate_menu_hash": ("cascadiav3.rival_incumbent_menu.v1:sha256:" + "b" * 64),
            "panel_id": "sha256:" + "b" * 64,
            "unit_index": 8,
            "fidelity": "low",
            "target_seat": 1,
            "incumbent_candidate_occurrence_id": (
                "cascadiav3.rival_candidate_action_occurrence.v1:sha256:" + "c" * 64
            ),
            "challenger_candidate_occurrence_id": (
                "cascadiav3.rival_candidate_action_occurrence.v1:sha256:" + "c" * 64
            ),
            "challenger_branch_ordinal": 0,
            "incumbent_action_content_id": (
                "cascadiav3.rival_action_content.v1:sha256:" + "c" * 64
            ),
            "challenger_action_content_id": (
                "cascadiav3.rival_action_content.v1:sha256:" + "c" * 64
            ),
            "incumbent_post_action_memory_sha256": "sha256:" + "c" * 64,
            "challenger_post_action_memory_sha256": "sha256:" + "c" * 64,
        }
        for index, (field, value) in enumerate(substitutions.items()):
            ledger = self._write_ledger(
                expectation,
                receipt_overrides={field: value},
                name=f"substitution-{index}.json",
            )
            expected_error = (
                "pin mismatch" if field == "parent_manifest_sha256" else "does not join"
            )
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(TerminalEvidenceError, expected_error),
            ):
                self.verifier.verify(
                    ledger,
                    expectation=expectation,
                    expected_pair_sha256="sha256:" + "4" * 64,
                )

    def test_receipt_shape_hash_domain_and_stability_are_enforced(self) -> None:
        expectation = _expectation(self.plans)
        cases = (
            ({"corrupt_receipt_hash": True}, "content hash"),
            ({"unknown_field": True}, "unknown"),
            ({"duplicate_output": True}, "duplicate JSON key"),
            ({"write_stderr": True}, "wrote to stderr"),
            ({"mutate_ledger": True}, "changed during verification"),
        )
        for index, (controls, reason) in enumerate(cases):
            ledger = self._write_ledger(
                expectation, controls=controls, name=f"invalid-{index}.json"
            )
            with (
                self.subTest(controls=controls),
                self.assertRaisesRegex(TerminalEvidenceError, reason),
            ):
                self.verifier.verify(
                    ledger,
                    expectation=expectation,
                    expected_pair_sha256="sha256:" + "4" * 64,
                )

        for index, (overrides, reason) in enumerate(
            (
                ({"proxy_policy": False}, "cannot authenticate production"),
                ({"beta_cv_required": 1}, "beta_cv_required=0"),
                ({"target_score_difference": True}, "must be an integer"),
                ({"challenger_branch_ordinal": True}, "must be an integer"),
                ({"challenger_branch_ordinal": -1}, "outside"),
                ({"challenger_branch_ordinal": 2**16}, "outside"),
                (
                    {"challenger_world_redetermination_seed_sha256": ("sha256:" + "7" * 64)},
                    "repeats a world-redetermination seed commitment",
                ),
            )
        ):
            ledger = self._write_ledger(
                expectation,
                receipt_overrides=overrides,
                name=f"domain-{index}.json",
            )
            with (
                self.subTest(overrides=overrides),
                self.assertRaisesRegex(TerminalEvidenceError, reason),
            ):
                self.verifier.verify(
                    ledger,
                    expectation=expectation,
                    expected_pair_sha256="sha256:" + "4" * 64,
                )

    def test_symlinks_wrong_executable_and_multifidelity_manifest_reject(self) -> None:
        expectation = _expectation(self.plans)
        ledger = self._write_ledger(expectation)
        symlink = self.root / "pair-link.json"
        symlink.symlink_to(ledger)
        with self.assertRaisesRegex(TerminalEvidenceError, "symbolic link"):
            self.verifier.verify(
                symlink,
                expectation=expectation,
                expected_pair_sha256="sha256:" + "4" * 64,
            )

        executable_symlink = self.root / "fake-rival-contract-link"
        executable_symlink.symlink_to(self.executable)
        with self.assertRaisesRegex(TerminalEvidenceError, "without following links"):
            RustTerminalVerifier(
                executable=executable_symlink,
                manifest=self.manifest,
            )

        wrong = copy.deepcopy(self.record)
        wrong["terminal_verifier_executable_sha256"] = SHA_A
        wrong.pop("content_sha256")
        wrong["deployment_design_sha256"] = deployment_design_identity(wrong)
        wrong["content_sha256"] = sha256_hex(wrong)
        with self.assertRaisesRegex(TerminalEvidenceError, "does not match"):
            RustTerminalVerifier(
                executable=self.executable,
                manifest=self._load_manifest_record(wrong, "wrong-manifest.json"),
            )

        multifidelity = json.loads(FIXTURE.read_text(encoding="utf-8"))
        multifidelity["terminal_verifier_executable_sha256"] = _file_sha256(self.executable)
        multifidelity.pop("content_sha256")
        multifidelity["deployment_design_sha256"] = deployment_design_identity(multifidelity)
        multifidelity["content_sha256"] = sha256_hex(multifidelity)
        with self.assertRaisesRegex(TerminalEvidenceError, "high-fidelity-only"):
            RustTerminalVerifier(
                executable=self.executable,
                manifest=self._load_manifest_record(multifidelity, "multifidelity-manifest.json"),
            )

        with self.assertRaisesRegex(TerminalEvidenceError, "externally byte/content-pinned"):
            RustTerminalVerifier(
                executable=self.executable,
                manifest=validate_root_manifest(self.record),
            )

    def test_timeout_and_nonzero_exit_are_denials(self) -> None:
        expectation = _expectation(self.plans)
        verifier = RustTerminalVerifier(
            executable=self.executable,
            manifest=self.manifest,
            timeout_seconds=0.05,
        )
        slow = self._write_ledger(expectation, controls={"sleep_seconds": 0.2}, name="slow.json")
        with self.assertRaisesRegex(TerminalEvidenceError, "did not complete"):
            verifier.verify(
                slow,
                expectation=expectation,
                expected_pair_sha256="sha256:" + "4" * 64,
            )
        rejected = self._write_ledger(expectation, controls={"exit_code": 9}, name="rejected.json")
        with self.assertRaisesRegex(TerminalEvidenceError, "exit 9"):
            self.verifier.verify(
                rejected,
                expectation=expectation,
                expected_pair_sha256="sha256:" + "4" * 64,
            )

    def test_oversized_ledger_is_denied_before_hashing_or_spawn(self) -> None:
        expectation = _expectation(self.plans)
        oversized = self.root / "oversized-pair.json"
        with oversized.open("wb") as handle:
            handle.truncate(MAX_TERMINAL_PAIR_LEDGER_BYTES + 1)
        with self.assertRaisesRegex(
            TerminalEvidenceError,
            f"Rust contract byte limit of {MAX_TERMINAL_PAIR_LEDGER_BYTES}",
        ):
            self.verifier.verify(
                oversized,
                expectation=expectation,
                expected_pair_sha256="sha256:" + "4" * 64,
            )

    @unittest.skipUnless(hasattr(os, "fork"), "process-group test requires os.fork")
    def test_timeout_kills_the_verifier_process_group_and_descendants(self) -> None:
        expectation = _expectation(self.plans)
        started = self.root / "descendant-started"
        survived = self.root / "descendant-survived"
        verifier = RustTerminalVerifier(
            executable=self.executable,
            manifest=self.manifest,
            timeout_seconds=0.5,
        )
        self.addCleanup(verifier.close)
        ledger = self._write_ledger(
            expectation,
            controls={
                "fork_descendant": True,
                "descendant_started_path": str(started),
                "descendant_survived_path": str(survived),
                "descendant_survival_delay": 1.5,
            },
            name="forked-descendant.json",
        )
        with self.assertRaisesRegex(TerminalEvidenceError, "timed out"):
            verifier.verify(
                ledger,
                expectation=expectation,
                expected_pair_sha256="sha256:" + "4" * 64,
            )
        self.assertTrue(started.is_file())
        time.sleep(1.75)
        self.assertFalse(survived.exists())

    def test_output_caps_are_enforced_while_both_pipes_are_streamed(self) -> None:
        expectation = _expectation(self.plans)
        cases = (
            (
                {"stdout_bytes": MAX_VERIFIER_STDOUT_BYTES + 1},
                "stdout exceeded its limit",
            ),
            (
                {"stderr_bytes": MAX_VERIFIER_STDERR_BYTES + 1},
                "stderr exceeded its limit",
            ),
        )
        for index, (controls, expected_error) in enumerate(cases):
            ledger = self._write_ledger(
                expectation,
                controls=controls,
                name=f"oversized-{index}.json",
            )
            with (
                self.subTest(controls=controls),
                self.assertRaisesRegex(TerminalEvidenceError, expected_error),
            ):
                self.verifier.verify(
                    ledger,
                    expectation=expectation,
                    expected_pair_sha256="sha256:" + "4" * 64,
                )

    def test_arguments_are_not_shell_interpreted_and_environment_is_sanitized(self) -> None:
        expectation = _expectation(self.plans)
        ledger = self._write_ledger(
            expectation,
            name="pair ; exit 93 ; shell metacharacters.json",
        )
        evidence = self.verifier.verify(
            ledger,
            expectation=expectation,
            expected_pair_sha256="sha256:" + "4" * 64,
        )
        self.assertEqual(evidence.target_score_difference, 11)

    def test_source_mutation_and_removal_cannot_change_executed_bytes(self) -> None:
        expectation = _expectation(self.plans)
        ledger = self._write_ledger(expectation, name="source-isolation.json")
        snapshot = self.verifier.executable
        self.assertNotEqual(snapshot.parent, self.executable.parent)
        self.assertEqual(stat.S_IMODE(snapshot.parent.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(snapshot.stat().st_mode), 0o700)
        self.assertEqual(
            snapshot.name,
            "verifier-" + self.manifest.terminal_verifier_executable_sha256.removeprefix("sha256:"),
        )

        self.executable.write_text(
            f"#!{sys.executable}\nraise SystemExit(91)\n",
            encoding="utf-8",
        )
        self.executable.chmod(0o700)
        first = self.verifier.verify(
            ledger,
            expectation=expectation,
            expected_pair_sha256="sha256:" + "4" * 64,
        )
        self.assertEqual(first.target_score_difference, 11)

        self.executable.unlink()
        second = self.verifier.verify(
            ledger,
            expectation=expectation,
            expected_pair_sha256="sha256:" + "4" * 64,
        )
        self.assertEqual(second.receipt_sha256, first.receipt_sha256)

    def test_executable_snapshot_substitution_fails_closed(self) -> None:
        expectation = _expectation(self.plans)
        ledger = self._write_ledger(expectation, name="snapshot-substitution.json")
        snapshot = self.verifier.executable
        displaced = snapshot.with_name("displaced-snapshot")
        snapshot.replace(displaced)
        snapshot.write_text(
            f"#!{sys.executable}\nraise SystemExit(92)\n",
            encoding="utf-8",
        )
        snapshot.chmod(0o700)

        with self.assertRaisesRegex(TerminalEvidenceError, "snapshot identity was substituted"):
            self.verifier.verify(
                ledger,
                expectation=expectation,
                expected_pair_sha256="sha256:" + "4" * 64,
            )

    def test_concrete_journal_resolver_reverifies_ledger_on_every_replay(self) -> None:
        expectation = _expectation(self.plans)
        ledger = self._write_ledger(expectation, name="resolver.json")
        evidence = self.verifier.verify(
            ledger,
            expectation=expectation,
            expected_pair_sha256="sha256:" + "4" * 64,
        )
        reference = TerminalEvidenceReference(
            evidence.receipt_sha256,
            evidence.pair_sha256,
            ledger,
            expectation,
        )
        resolver = RustTerminalRowResolver(
            verifier=self.verifier,
            references=(reference,),
        )
        row = resolver(
            "selection",
            {"evidence_receipt_sha256": evidence.receipt_sha256},
        )
        self.assertEqual(row, evidence.as_selection_row())
        with self.assertRaisesRegex(TerminalEvidenceError, "unregistered"):
            resolver(
                "selection",
                {"evidence_receipt_sha256": "sha256:" + "9" * 64},
            )
        with self.assertRaisesRegex(TerminalEvidenceError, "duplicate"):
            RustTerminalRowResolver(
                verifier=self.verifier,
                references=(reference, reference),
            )

        ledger.write_bytes(ledger.read_bytes() + b" ")
        with self.assertRaisesRegex(TerminalEvidenceError, "differs"):
            resolver(
                "selection",
                {"evidence_receipt_sha256": evidence.receipt_sha256},
            )

    def test_expectation_itself_is_bound_to_registered_menu_and_panel(self) -> None:
        valid = _expectation(self.plans)
        with self.assertRaises(TypeError):
            TerminalUnitExpectation(  # type: ignore[call-arg]
                panel_kind="S",
                panel_id=self.record["panel_identities"]["S"],
                unit_index=0,
                fidelity="high",
                target_seat=2,
                challenger_candidate_occurrence_id=(valid.challenger_candidate_occurrence_id),
                challenger_action_content_id=valid.challenger_action_content_id,
                incumbent_post_action_memory_sha256=(valid.incumbent_post_action_memory_sha256),
                challenger_post_action_memory_sha256=(valid.challenger_post_action_memory_sha256),
            )

        changed_plan = copy.deepcopy(self.plans["S"])
        object.__setattr__(changed_plan.units[0], "target_seat", 1)
        with self.assertRaisesRegex(PanelPlanError, "validation capability|forged or mutated"):
            changed_plan.high_fidelity_expectation(0)

    def test_real_rust_fixture_verifier_round_trip_when_binary_is_built(self) -> None:
        configured = os.environ.get("CASCADIA_RIVAL_CONTRACT_BIN")
        binary = Path(configured) if configured else REPO_ROOT / "target/debug/rival-contract"
        if not binary.is_file():
            self.skipTest("build rival-contract first for the cross-language integration test")
        environment = {
            "CASCADIA_CPU_ONLY_TESTS": "1",
            "CASCADIA_DEVICE": "cpu",
            "CUDA_VISIBLE_DEVICES": "",
            "LC_ALL": "C",
            "PATH": os.defpath,
        }

        fixture_process = subprocess.run(
            [str(binary), "proxy-terminal-pair-fixture"],
            check=False,
            capture_output=True,
            env=environment,
        )
        if fixture_process.returncode != 0:
            if configured:
                self.fail(
                    "configured rival-contract could not emit its fixture: "
                    + fixture_process.stderr.decode("utf-8", errors="replace")
                )
            self.skipTest("rebuild stale rival-contract for the integration test")
        default_pair = fixture_process.stdout
        default_path = self.root / "rust-default-pair.json"
        default_path.write_bytes(default_pair)
        default_record = json.loads(default_pair)
        context_receipt = json.loads(
            subprocess.run(
                [
                    str(binary),
                    "verify-terminal-pair",
                    str(default_path),
                    default_record["pair_sha256"],
                    default_record["parent_manifest_sha256"],
                ],
                check=True,
                capture_output=True,
                env=environment,
            ).stdout
        )
        self.assertEqual(receipt_identity(context_receipt), context_receipt["receipt_sha256"])

        incumbent = CandidateSelectionEntry(
            context_receipt["incumbent_candidate_occurrence_id"],
            context_receipt["incumbent_action_content_id"],
            0,
        )
        challenger = CandidateSelectionEntry(
            context_receipt["challenger_candidate_occurrence_id"],
            context_receipt["challenger_action_content_id"],
            1,
        )
        record: dict[str, Any] = {
            "schema_id": "cascadiav3.rival_root_manifest.v1",
            "manifest_id": "fixture:rust-python-terminal-join",
            "ruleset_identity": context_receipt["ruleset_identity_sha256"],
            "source_revision": "fixture:rust-cpu-reference",
            "root_id": context_receipt["source_public_root_id"],
            "source_game_id": "fixture:rust-source-game",
            "source_game_identity_sha256": context_receipt["source_game_identity_sha256"],
            "root_kind": "draft_policy_root",
            "root_cohort_role": "design_tomography",
            "complete_game_seed_role": None,
            "inference_mode": "high_fidelity_only",
            "required_panels": ["S", "H"],
            "forbidden_panels": ["L", "A"],
            "panel_identities": {"S": None, "H": None, "L": None, "A": None},
            "beta_cv": 0.0,
            "multifidelity_claim": False,
            "incumbent_policy_identity": context_receipt["continuation_policy_identity_sha256"],
            "incumbent_action_id": incumbent.action_content_id,
            "incumbent_candidate_occurrence_id": (incumbent.candidate_action_occurrence_id),
            "rules_menu_hash": context_receipt["source_rules_menu_hash"],
            "incumbent_menu_hash": context_receipt["source_candidate_menu_hash"],
            "low_policy_identity": None,
            "candidate_set_identity": candidate_set_identity((incumbent, challenger)),
            "candidate_selection_entries": [
                {
                    "candidate_action_occurrence_id": (row.candidate_action_occurrence_id),
                    "action_content_id": row.action_content_id,
                    "expected_s": row.expected_s,
                }
                for row in (incumbent, challenger)
            ],
            "sampler_identity": context_receipt["scenario_sampler_identity_sha256"],
            "policy_rng_factory_identity": context_receipt["policy_rng_factory_identity_sha256"],
            "terminal_verifier_executable_sha256": _file_sha256(binary),
            "terminal_verifier_contract_id": ("cascadia-rival.verify-terminal-pair.v1"),
            "coefficient_identity": None,
            "allocation_identity": "sha256:" + "5" * 64,
            "bound_certificate_identity": "sha256:" + "6" * 64,
            "error_ledger_identity": "sha256:" + "7" * 64,
            "expected_s": 1,
            "expected_h": 1,
            "expected_l": 0,
            "practical_margin": 0.25,
            "preference_weight": 2.0,
            "selection_rule": "highest_mean_then_lexicographic_action_id",
            "low_expectation_id": None,
            "low_law_h_id": None,
            "low_law_l_id": None,
            "max_abs_beta": None,
            "a_panel_enabled": False,
            "quantitative_target_enabled": False,
        }
        plan_records = {
            panel: _panel_plan_record(
                record,
                panel,
                target_seat=context_receipt["target_seat"],
                incumbent_memory_sha256=(context_receipt["incumbent_post_action_memory_sha256"]),
                challenger_memory_sha256=(context_receipt["challenger_post_action_memory_sha256"]),
            )
            for panel in ("S", "H")
        }
        record["panel_identities"] = {
            "S": "sha256:" + plan_records["S"]["content_sha256"],
            "H": "sha256:" + plan_records["H"]["content_sha256"],
            "L": None,
            "A": None,
        }
        record["deployment_design_sha256"] = deployment_design_identity(record)
        record["content_sha256"] = sha256_hex(record)
        manifest = self._load_manifest_record(record, "rust-manifest.json")
        plans = {
            panel: validate_terminal_panel_plan(plan, manifest=manifest)
            for panel, plan in plan_records.items()
        }

        pinned_pair = subprocess.run(
            [
                str(binary),
                "proxy-terminal-pair-fixture",
                "sha256:" + manifest.content_sha256,
                plans["H"].identity,
                "0",
            ],
            check=True,
            capture_output=True,
            env=environment,
        ).stdout
        pair_path = self.root / "rust-pinned-pair.json"
        pair_path.write_bytes(pinned_pair)
        expectation = plans["H"].high_fidelity_expectation(0)
        pinned_pair_record = json.loads(pinned_pair)
        expected_pair_sha256 = pinned_pair_record["pair_sha256"]
        evidence = RustTerminalVerifier(
            executable=binary,
            manifest=manifest,
        ).verify(
            pair_path,
            expectation=expectation,
            expected_pair_sha256=expected_pair_sha256,
        )
        self.assertEqual(evidence.evidence_domain, EvidenceDomain.CPU_PROXY_REFERENCE)
        self.assertEqual(
            evidence.as_high_only_h_row().high_difference,
            float(pinned_pair_record["target_score_difference"]),
        )


if __name__ == "__main__":
    unittest.main()
