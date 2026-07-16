"""CPU-only fault-injection tests for the immutable Rival appeal journal."""

from __future__ import annotations

import json
import os
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from cascadiav3.rival.appeal_journal import (
    AppealEventJournal,
    AppealJournalError,
)
from cascadiav3.rival.appeals import (
    EvidenceDomain,
    HighOnlyHRow,
    HRow,
    LRow,
    SelectionRow,
    UnitStatus,
    _verified_selection_row,
)
from cascadiav3.rival.schema import (
    attach_content_hash,
    canonical_json_bytes,
    sha256_hex,
)
from test_rival_cohorts import (
    CHALLENGER_OCCURRENCE_ID,
    ROOT_ID,
    verified_design,
    verified_high_design,
)


def _selection(
    *,
    unit_id: str = "s0",
    score: float | None = 7.0,
    status: UnitStatus = UnitStatus.COMPLETE,
    rng_key: str = "rng:s0",
) -> SelectionRow:
    return SelectionRow.contract_test(
        unit_id,
        CHALLENGER_OCCURRENCE_ID,
        score,
        status,
        rng_key,
    )


def _h(
    *,
    unit_id: str = "h0",
    high: float | None = 10.0,
    low: float | None = 2.0,
    status: UnitStatus = UnitStatus.COMPLETE,
) -> HRow:
    return HRow.contract_test(
        unit_id,
        CHALLENGER_OCCURRENCE_ID,
        high,
        low,
        status,
        f"physical:{unit_id}",
        (f"inner:{unit_id}:inc", f"inner:{unit_id}:challenger"),
    )


def _l(
    *,
    unit_id: str = "l0",
    low: float | None = 1.0,
    status: UnitStatus = UnitStatus.COMPLETE,
) -> LRow:
    return LRow.contract_test(
        unit_id,
        CHALLENGER_OCCURRENCE_ID,
        low,
        status,
        f"physical:{unit_id}",
        (f"inner:{unit_id}:inc", f"inner:{unit_id}:challenger"),
    )


def _write_record_for_tamper(path: Path, record: dict[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(record) + b"\n")


class RivalAppealJournalTest(unittest.TestCase):
    def setUp(self) -> None:
        # This suite is a reference-contract suite.  It must never acquire an
        # accelerator merely because the host happens to have one.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["CASCADIA_CPU_ONLY_TESTS"] = "1"
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def _journal(self, name: str = "journal", **kwargs: Any) -> AppealEventJournal:
        return AppealEventJournal.create(
            self.root / name,
            design=verified_design(expected_s=1, expected_h=1, expected_l=1),
            **kwargs,
        )

    def test_complete_multifidelity_history_replays_and_finalizes_once(self) -> None:
        journal = self._journal()
        opening_bytes = (journal.events_directory / "0000000000000000.json").read_bytes()
        journal.add_selection(_selection())
        journal.freeze_challenger(CHALLENGER_OCCURRENCE_ID)
        journal.add_h(_h())
        before_final = journal.add_l(_l())

        self.assertEqual(before_final.event_count, 5)
        self.assertFalse(before_final.finalized)
        self.assertEqual(before_final.operational.attempted_total, 3)
        self.assertFalse(hasattr(before_final, "finalize"))
        decision = journal.finalize()
        self.assertIn(decision.status, {"not_confirmed", "no_label"})
        self.assertTrue(journal.final_path.is_file())
        self.assertEqual(
            opening_bytes,
            (journal.events_directory / "0000000000000000.json").read_bytes(),
        )

        replayed = AppealEventJournal.open(
            journal.directory,
            design=verified_design(expected_s=1, expected_h=1, expected_l=1),
        ).replay()
        self.assertTrue(replayed.finalized)
        self.assertEqual(replayed.decision, decision)
        self.assertEqual(replayed.operational.completed_total, 3)
        with self.assertRaisesRegex(AppealJournalError, "already consumed"):
            journal.finalize()
        with self.assertRaisesRegex(AppealJournalError, "post-final"):
            journal.add_l(_l(unit_id="l1"))

    def test_failed_unit_is_durably_preserved_as_operational_no_label(self) -> None:
        journal = self._journal()
        journal.add_selection(_selection(score=None, status=UnitStatus.TIMEOUT))
        decision = journal.finalize()
        self.assertEqual(decision.status, "no_label")
        self.assertEqual(decision.operational.attempted_s, 1)
        self.assertEqual(decision.operational.completed_s, 0)
        self.assertEqual(decision.operational.timeouts, 1)
        replayed = journal.replay()
        self.assertEqual(replayed.decision, decision)
        self.assertEqual(replayed.operational.timeouts, 1)

    def test_panel_batches_are_linear_and_validate_before_any_publication(self) -> None:
        design = verified_design(expected_s=2, expected_h=2, expected_l=2)
        journal = AppealEventJournal.create(self.root / "batches", design=design)
        duplicate = _selection()
        with self.assertRaisesRegex(AppealJournalError, "duplicate"):
            journal.add_selections((duplicate, duplicate))
        self.assertEqual(journal.replay().event_count, 1)

        journal.add_selections(
            (
                _selection(),
                _selection(unit_id="s1", score=6.0, rng_key="rng:s1"),
            )
        )
        journal.freeze_challenger(CHALLENGER_OCCURRENCE_ID)
        journal.add_h_rows((_h(), _h(unit_id="h1")))
        snapshot = journal.add_l_rows((_l(), _l(unit_id="l1")))
        self.assertEqual(snapshot.event_count, 8)
        self.assertEqual(snapshot.operational.attempted_total, 6)
        self.assertEqual(journal.finalize().operational.completed_total, 6)

    def test_high_fidelity_machine_replays_without_opening_an_l_panel(self) -> None:
        design = verified_high_design(expected_s=1, expected_h=1)
        journal = AppealEventJournal.create(self.root / "high", design=design)
        journal.add_selection(_selection())
        journal.freeze_challenger(CHALLENGER_OCCURRENCE_ID)
        journal.add_h(
            HighOnlyHRow.contract_test(
                "h0",
                CHALLENGER_OCCURRENCE_ID,
                None,
                UnitStatus.TIMEOUT,
                "physical:h0",
                ("inner:h0:inc", "inner:h0:challenger"),
            )
        )
        with self.assertRaisesRegex(AppealJournalError, "forbidden"):
            journal.add_l(_l())
        decision = journal.finalize()
        self.assertEqual(decision.status, "no_label")
        self.assertEqual(decision.operational.timeouts, 1)
        self.assertEqual(journal.replay().decision, decision)

    def test_gap_and_unexpected_alias_are_rejected(self) -> None:
        journal = self._journal("gap")
        journal.add_selection(_selection())
        event_one = journal.events_directory / "0000000000000001.json"
        event_one.rename(journal.events_directory / "0000000000000002.json")
        with self.assertRaisesRegex(AppealJournalError, "gapped"):
            journal.replay()

        alias = self._journal("alias")
        (alias.events_directory / "0.json").write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(AppealJournalError, "unexpected event artifact"):
            alias.replay()

    def test_content_mutation_and_duplicate_json_keys_are_rejected(self) -> None:
        journal = self._journal("mutation")
        journal.add_selection(_selection())
        event = journal.events_directory / "0000000000000001.json"
        record = json.loads(event.read_text(encoding="utf-8"))
        record["payload"]["selection_score"] = 99.0
        _write_record_for_tamper(event, record)
        with self.assertRaisesRegex(AppealJournalError, "content_sha256 mismatch"):
            journal.replay()

        duplicate = self._journal("duplicate-json")
        opening = duplicate.events_directory / "0000000000000000.json"
        raw = opening.read_text(encoding="utf-8")
        opening.write_text(raw[:-2] + ',"schema_id":"duplicate"}\n', encoding="utf-8")
        with self.assertRaisesRegex(AppealJournalError, "duplicate JSON key"):
            duplicate.replay()

        wrong_primitive = self._journal("wrong-primitive")
        opening = wrong_primitive.events_directory / "0000000000000000.json"
        primitive_record = json.loads(opening.read_text(encoding="utf-8"))
        primitive_record.pop("content_sha256")
        primitive_record["format_version"] = 1.0
        _write_record_for_tamper(opening, attach_content_hash(primitive_record))
        with self.assertRaisesRegex(AppealJournalError, "wrong format version"):
            wrong_primitive.replay()

        seed_tamper = self._journal("redetermination-seed-tamper")
        seed_tamper.add_selection(_selection())
        event = seed_tamper.events_directory / "0000000000000001.json"
        record = json.loads(event.read_text(encoding="utf-8"))
        record.pop("content_sha256")
        record["payload"]["world_redetermination_seed_sha256s"][0] = "sha256:" + "9" * 64
        _write_record_for_tamper(event, attach_content_hash(record))
        with self.assertRaisesRegex(AppealJournalError, "derived persisted evidence"):
            seed_tamper.replay()

        production_tamper = self._journal("production-domain-tamper")
        production_tamper.add_selection(_selection())
        event = production_tamper.events_directory / "0000000000000001.json"
        record = json.loads(event.read_text(encoding="utf-8"))
        record.pop("content_sha256")
        record["payload"]["evidence_domain"] = EvidenceDomain.PRODUCTION_TERMINAL.value
        record["payload"]["evidence_receipt_sha256"] = "sha256:" + "8" * 64
        _write_record_for_tamper(event, attach_content_hash(record))
        with self.assertRaisesRegex(AppealJournalError, "structurally unavailable"):
            production_tamper.replay()

    def test_semantic_duplicates_and_frozen_design_substitution_are_rejected(self) -> None:
        design = verified_design(expected_s=2, expected_h=1, expected_l=1)
        journal = AppealEventJournal.create(self.root / "duplicates", design=design)
        journal.add_selection(_selection())
        with self.assertRaisesRegex(AppealJournalError, "duplicate"):
            journal.add_selection(_selection())

        other_design = verified_design(
            root_id=ROOT_ID.rsplit(":", 1)[0] + ":" + "9" * 64,
            expected_s=2,
            expected_h=1,
            expected_l=1,
        )
        with self.assertRaisesRegex(AppealJournalError, "root_id"):
            AppealEventJournal.open(journal.directory, design=other_design)

    def test_final_receipt_binds_event_tip_and_deterministic_decision(self) -> None:
        journal = self._journal("final-tamper")
        journal.add_selection(_selection(score=None, status=UnitStatus.INVALID))
        journal.finalize()
        record = json.loads(journal.final_path.read_text(encoding="utf-8"))
        record.pop("content_sha256")
        record["decision"]["reason"] = "forged but self-consistently rehashed"
        record["decision_sha256"] = "sha256:" + sha256_hex(record["decision"])
        _write_record_for_tamper(journal.final_path, attach_content_hash(record))
        with self.assertRaisesRegex(AppealJournalError, "deterministic replay"):
            journal.replay()

        tip_journal = self._journal("tip-tamper")
        tip_journal.add_selection(_selection(score=None, status=UnitStatus.TIMEOUT))
        tip_journal.finalize()
        tip_record = json.loads(tip_journal.final_path.read_text(encoding="utf-8"))
        tip_record.pop("content_sha256")
        tip_record["event_tip_sha256"] = "sha256:" + "0" * 64
        _write_record_for_tamper(tip_journal.final_path, attach_content_hash(tip_record))
        with self.assertRaisesRegex(AppealJournalError, "event_tip_sha256"):
            tip_journal.replay()

    def test_terminal_rows_require_a_trusted_resolver_on_write_and_replay(self) -> None:
        receipt = "sha256:" + "7" * 64
        terminal_row = _verified_selection_row(
            unit_id="s-terminal",
            challenger_id=CHALLENGER_OCCURRENCE_ID,
            selection_score=3.5,
            rng_key="rng:terminal",
            world_redetermination_seed_sha256s=(
                "sha256:" + "1" * 64,
                "sha256:" + "2" * 64,
            ),
            evidence_domain=EvidenceDomain.CPU_PROXY_REFERENCE,
            receipt_sha256=receipt,
        )
        untrusted = self._journal("untrusted")
        with self.assertRaisesRegex(AppealJournalError, "trusted row resolver"):
            untrusted.add_selection(terminal_row)

        resolver_calls: list[tuple[str, str]] = []

        def resolver(kind: str, payload: Any) -> SelectionRow:
            resolver_calls.append((kind, payload["evidence_receipt_sha256"]))
            return _verified_selection_row(
                unit_id=payload["unit_id"],
                challenger_id=payload["challenger_id"],
                selection_score=payload["selection_score"],
                rng_key=payload["rng_key"],
                world_redetermination_seed_sha256s=tuple(
                    payload["world_redetermination_seed_sha256s"]
                ),
                evidence_domain=EvidenceDomain(payload["evidence_domain"]),
                receipt_sha256=payload["evidence_receipt_sha256"],
            )

        design = verified_design(expected_s=1, expected_h=1, expected_l=1)
        resolver.root_id = design.root_id  # type: ignore[attr-defined]
        resolver.deployment_design_sha256 = (  # type: ignore[attr-defined]
            design.deployment_design_sha256
        )
        resolver.manifest_content_sha256 = (  # type: ignore[attr-defined]
            "sha256:" + design.manifest_content_sha256
        )
        trusted = AppealEventJournal.create(
            self.root / "trusted",
            design=design,
            terminal_row_resolver=resolver,
        )
        trusted.add_selection(terminal_row)
        event = json.loads(
            (trusted.events_directory / "0000000000000001.json").read_text(encoding="utf-8")
        )
        self.assertEqual(event["payload"]["evidence_receipt_sha256"], receipt)
        self.assertEqual(
            event["payload"]["evidence_domain"],
            EvidenceDomain.CPU_PROXY_REFERENCE.value,
        )
        self.assertEqual(
            event["payload"]["world_redetermination_seed_sha256s"],
            list(terminal_row.world_redetermination_seed_sha256s),
        )
        trusted.replay()
        self.assertGreaterEqual(len(resolver_calls), 2)  # initial admission + replay
        with self.assertRaisesRegex(AppealJournalError, "trusted row resolver"):
            AppealEventJournal.open(trusted.directory, design=trusted.design)

        def lying_resolver(kind: str, payload: Any) -> SelectionRow:
            del kind
            return _verified_selection_row(
                unit_id=payload["unit_id"],
                challenger_id=payload["challenger_id"],
                selection_score=payload["selection_score"] + 1.0,
                rng_key=payload["rng_key"],
                world_redetermination_seed_sha256s=tuple(
                    payload["world_redetermination_seed_sha256s"]
                ),
                evidence_domain=EvidenceDomain(payload["evidence_domain"]),
                receipt_sha256=payload["evidence_receipt_sha256"],
            )

        lying_resolver.root_id = design.root_id  # type: ignore[attr-defined]
        lying_resolver.deployment_design_sha256 = (  # type: ignore[attr-defined]
            design.deployment_design_sha256
        )
        lying_resolver.manifest_content_sha256 = (  # type: ignore[attr-defined]
            "sha256:" + design.manifest_content_sha256
        )

        with self.assertRaisesRegex(AppealJournalError, "differs from the journal"):
            AppealEventJournal.open(
                trusted.directory,
                design=trusted.design,
                terminal_row_resolver=lying_resolver,
            )

        def unsealed_resolver(kind: str, payload: Any) -> SelectionRow:
            del kind
            return SelectionRow(
                payload["unit_id"],
                payload["challenger_id"],
                payload["selection_score"],
                UnitStatus(payload["status"]),
                payload["rng_key"],
                tuple(payload["world_redetermination_seed_sha256s"]),
                EvidenceDomain(payload["evidence_domain"]),
                payload["evidence_receipt_sha256"],
                None,
            )

        unsealed_resolver.root_id = design.root_id  # type: ignore[attr-defined]
        unsealed_resolver.deployment_design_sha256 = (  # type: ignore[attr-defined]
            design.deployment_design_sha256
        )
        unsealed_resolver.manifest_content_sha256 = (  # type: ignore[attr-defined]
            "sha256:" + design.manifest_content_sha256
        )
        with self.assertRaisesRegex(AppealJournalError, "must be produced"):
            AppealEventJournal.open(
                trusted.directory,
                design=trusted.design,
                terminal_row_resolver=unsealed_resolver,
            )

    def test_concurrent_finalize_publishes_exactly_one_final_receipt(self) -> None:
        journal = self._journal("race")
        journal.add_selection(_selection(score=None, status=UnitStatus.TIMEOUT))
        first = AppealEventJournal.open(journal.directory, design=journal.design)
        second = AppealEventJournal.open(journal.directory, design=journal.design)
        barrier = threading.Barrier(2)
        results: list[str] = []
        failures: list[Exception] = []

        def finalize(candidate: AppealEventJournal) -> None:
            try:
                barrier.wait(timeout=5)
                results.append(candidate.finalize().status)
            except Exception as exc:  # the losing writer must fail closed
                failures.append(exc)

        threads = [
            threading.Thread(target=finalize, args=(first,)),
            threading.Thread(target=finalize, args=(second,)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
            self.assertFalse(thread.is_alive())

        self.assertEqual(results, ["no_label"])
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], AppealJournalError)
        self.assertIn("already consumed", str(failures[0]))
        self.assertEqual(journal.replay().decision.status, "no_label")

    def test_create_is_race_safe_and_layout_fails_closed(self) -> None:
        directory = self.root / "create-race"
        design = verified_design(expected_s=1, expected_h=1, expected_l=1)
        barrier = threading.Barrier(2)
        successes: list[AppealEventJournal] = []
        failures: list[Exception] = []

        def create() -> None:
            try:
                barrier.wait(timeout=5)
                successes.append(AppealEventJournal.create(directory, design=design))
            except Exception as exc:
                failures.append(exc)

        threads = [threading.Thread(target=create), threading.Thread(target=create)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
            self.assertFalse(thread.is_alive())
        self.assertEqual(len(successes), 1)
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], AppealJournalError)

        (directory / "shadow-final.json").write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(AppealJournalError, "unexpected journal artifacts"):
            successes[0].replay()

    def test_broken_final_symlink_is_a_fail_closed_finality_fence(self) -> None:
        journal = self._journal("broken-final")
        journal.add_selection(_selection(score=None, status=UnitStatus.TIMEOUT))
        journal.final_path.symlink_to(self.root / "missing-final-target")
        with self.assertRaisesRegex(AppealJournalError, "already consumed"):
            journal.finalize()
        with self.assertRaisesRegex(AppealJournalError, "post-final"):
            journal.add_selection(_selection(unit_id="s1", rng_key="rng:s1"))
        with self.assertRaisesRegex(AppealJournalError, "cannot safely open"):
            journal.replay()


if __name__ == "__main__":
    unittest.main()
