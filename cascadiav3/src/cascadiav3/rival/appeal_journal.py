"""Durable, immutable event journals for Rival appeal state machines.

The statistical state machines in :mod:`cascadiav3.rival.appeals` deliberately
allow exactly one inferential look.  This module makes that property survive
process crashes and restarts:

* every transition is a create-new, canonical-JSON event;
* events form a contiguous SHA-256 chain bound to the root and frozen design;
* replay reconstructs the state machine and rejects malformed history;
* ``FINAL.json`` is a create-new receipt binding the decision to the event tip;
* callers never receive an unfinalized state machine on which they could peek.

Contract-test rows are reconstructed only through their explicit fixture
constructors.  CPU-proxy and production-terminal rows are reconstructed only
through a caller-supplied trusted resolver.  The journal never turns persisted
floating-point fields into terminal evidence by itself.
"""

from __future__ import annotations

import copy
import dataclasses
import fcntl
import json
import os
import re
import stat
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Protocol

from .appeals import (
    AppealDecision,
    AppealError,
    AppealStateMachine,
    EvidenceDomain,
    HighFidelityAppealDecision,
    HighFidelityAppealStateMachine,
    HighOnlyHRow,
    HRow,
    LRow,
    OperationalAccounting,
    SelectionRow,
    UnitStatus,
    VerifiedHighFidelityDesign,
    VerifiedMultifidelityDesign,
    require_validated_appeal_design,
    require_validated_evidence_row,
)
from .schema import (
    RivalSchemaError,
    attach_content_hash,
    canonical_json_bytes,
    require_exact_keys,
    require_sha256,
    sha256_hex,
    verify_content_hash,
    write_new_canonical_json,
)

APPEAL_EVENT_SCHEMA_ID = "cascadiav3.rival_appeal_event.v1"
APPEAL_FINAL_SCHEMA_ID = "cascadiav3.rival_appeal_final.v1"
_FORMAT_VERSION = 1
_EVENT_NAME = re.compile(r"^(?P<sequence>[0-9]{16})\.json$")
_EVENT_TYPES = frozenset({"journal_open", "selection_row", "freeze_challenger", "h_row", "l_row"})

AppealMode = Literal["multifidelity", "high_fidelity_only"]
RowKind = Literal["selection", "h_multifidelity", "h_high_only", "l"]
type AppealRow = SelectionRow | HRow | HighOnlyHRow | LRow
type AppealMachine = AppealStateMachine | HighFidelityAppealStateMachine
type AppealResult = AppealDecision | HighFidelityAppealDecision
type AppealDesign = VerifiedMultifidelityDesign | VerifiedHighFidelityDesign


class AppealJournalError(ValueError):
    """Raised when journal history, finality, or evidence provenance is invalid."""


class TrustedTerminalRowResolver(Protocol):
    """Re-verify one persisted non-fixture row against trusted receipt storage.

    Implementations should resolve ``evidence_receipt_sha256`` to the immutable
    verifier receipt, validate that receipt and its source ledger, and return a
    row produced by the terminal-evidence adapter.  The journal checks that the
    returned row exactly matches every persisted field before admitting it.
    """

    root_id: str
    deployment_design_sha256: str
    manifest_content_sha256: str

    def __call__(self, row_kind: RowKind, serialized_row: Mapping[str, Any]) -> AppealRow: ...


@dataclass(frozen=True)
class JournalSnapshot:
    """Read-only replay result that intentionally exposes no inference method."""

    mode: AppealMode
    root_id: str
    deployment_design_sha256: str
    manifest_content_sha256: str
    event_count: int
    event_tip_sha256: str
    selected_challenger_id: str | None
    operational: OperationalAccounting
    finalized: bool
    decision: AppealResult | None


@dataclass(frozen=True)
class _Binding:
    mode: AppealMode
    root_id: str
    deployment_design_sha256: str
    manifest_content_sha256: str


def _qualified_sha256(value: str, field: str) -> str:
    try:
        return "sha256:" + require_sha256(value, field)
    except RivalSchemaError as exc:
        raise AppealJournalError(str(exc)) from exc


def _artifact_present(path: Path) -> bool:
    """Return true for every directory entry, including a broken symlink."""

    return os.path.lexists(path)


def _binding_for_design(design: AppealDesign) -> _Binding:
    if isinstance(design, VerifiedMultifidelityDesign):
        mode: AppealMode = "multifidelity"
    elif isinstance(design, VerifiedHighFidelityDesign):
        mode = "high_fidelity_only"
    else:
        raise AppealJournalError("journal requires a verified Rival appeal design")
    try:
        require_validated_appeal_design(design)
    except AppealError as exc:
        raise AppealJournalError(
            f"journal requires a validated Rival appeal design: {exc}"
        ) from exc
    if not isinstance(design.root_id, str) or not design.root_id:
        raise AppealJournalError("verified design root_id is invalid")
    return _Binding(
        mode=mode,
        root_id=design.root_id,
        deployment_design_sha256=_qualified_sha256(
            design.deployment_design_sha256, "deployment_design_sha256"
        ),
        manifest_content_sha256=_qualified_sha256(
            design.manifest_content_sha256, "manifest_content_sha256"
        ),
    )


def _strict_json_record(path: Path) -> dict[str, Any]:
    """Read one canonical JSON object without following links or racing mutation."""

    flags = os.O_RDONLY | os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise AppealJournalError(f"cannot safely open journal artifact {path}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise AppealJournalError(f"journal artifact must be a single-link regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    signature_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    signature_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if signature_before != signature_after:
        raise AppealJournalError(f"journal artifact changed while being read: {path}")
    raw = b"".join(chunks)

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AppealJournalError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    def reject_nonfinite_constant(value: str) -> Any:
        raise AppealJournalError(f"non-finite JSON constant {value!r} in {path}")

    try:
        parsed = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AppealJournalError(f"invalid JSON in journal artifact {path}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise AppealJournalError(f"journal artifact must be a JSON object: {path}")
    try:
        canonical = canonical_json_bytes(parsed) + b"\n"
    except RivalSchemaError as exc:
        raise AppealJournalError(f"journal artifact is not encodable: {path}: {exc}") from exc
    if raw != canonical:
        raise AppealJournalError(f"journal artifact is not canonical JSON: {path}")
    return parsed


def _canonical_value(value: Any) -> Any:
    """Round-trip through the journal encoding to obtain JSON-native values."""

    return json.loads(canonical_json_bytes(value))


def _decision_record(decision: AppealResult) -> dict[str, Any]:
    if not isinstance(decision, (AppealDecision, HighFidelityAppealDecision)):
        raise AppealJournalError("state machine returned an unknown decision type")
    record = _canonical_value(dataclasses.asdict(decision))
    assert isinstance(record, dict)
    return record


def _row_kind(row: AppealRow) -> RowKind:
    if isinstance(row, SelectionRow):
        return "selection"
    if isinstance(row, HRow):
        return "h_multifidelity"
    if isinstance(row, HighOnlyHRow):
        return "h_high_only"
    if isinstance(row, LRow):
        return "l"
    raise AppealJournalError(f"unsupported appeal row type {type(row).__name__}")


def _row_record(row: AppealRow) -> dict[str, Any]:
    """Serialize every public outcome, key, status, domain, and receipt binding."""

    # Re-run the row's evidence check so an object forged without dataclass
    # construction cannot cross the persistence boundary.
    try:
        require_validated_evidence_row(row)
    except (AppealError, AttributeError, TypeError, ValueError) as exc:
        raise AppealJournalError(f"invalid evidence row: {exc}") from exc
    if not isinstance(row.status, UnitStatus):
        raise AppealJournalError("evidence row status must be a typed UnitStatus")
    if not isinstance(row.evidence_domain, EvidenceDomain):
        raise AppealJournalError("evidence row domain must be a typed EvidenceDomain")
    common: dict[str, Any] = {
        "unit_id": row.unit_id,
        "challenger_id": row.challenger_id,
        "status": row.status.value,
        "world_redetermination_seed_sha256s": list(row.world_redetermination_seed_sha256s),
        "evidence_domain": row.evidence_domain.value,
        "evidence_receipt_sha256": row.evidence_receipt_sha256,
    }
    if isinstance(row, SelectionRow):
        common.update(selection_score=row.selection_score, rng_key=row.rng_key)
    elif isinstance(row, HRow):
        common.update(
            high_difference=row.high_difference,
            low_difference=row.low_difference,
            physical_coupling_key=row.physical_coupling_key,
            inner_rng_keys=list(row.inner_rng_keys),
        )
    elif isinstance(row, HighOnlyHRow):
        common.update(
            high_difference=row.high_difference,
            physical_key=row.physical_key,
            inner_rng_keys=list(row.inner_rng_keys),
        )
    elif isinstance(row, LRow):
        common.update(
            low_difference=row.low_difference,
            physical_key=row.physical_key,
            inner_rng_keys=list(row.inner_rng_keys),
        )
    else:  # pragma: no cover - narrowed by the union and checked above
        raise AppealJournalError("unsupported row")
    canonical = _canonical_value(common)
    assert isinstance(canonical, dict)
    return canonical


def _enum_fields(payload: Mapping[str, Any]) -> tuple[UnitStatus, EvidenceDomain]:
    try:
        status = UnitStatus(payload["status"])
        domain = EvidenceDomain(payload["evidence_domain"])
    except (KeyError, ValueError, TypeError) as exc:
        raise AppealJournalError(f"invalid row status/evidence domain: {exc}") from exc
    if domain is EvidenceDomain.PRODUCTION_TERMINAL:
        raise AppealJournalError(
            "production-terminal evidence is structurally unavailable before "
            "a production Rust adapter is admitted"
        )
    receipt = payload.get("evidence_receipt_sha256")
    if domain is EvidenceDomain.CONTRACT_TEST:
        if receipt is not None:
            raise AppealJournalError("contract-test row cannot claim a terminal receipt")
    else:
        _qualified_sha256(receipt, "evidence_receipt_sha256")
    return status, domain


def _require_string_list(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or any(not isinstance(item, str) for item in value):
        raise AppealJournalError(f"{field} must be a non-empty JSON string array")
    return tuple(value)


def _deserialize_row(
    kind: RowKind,
    payload: Mapping[str, Any],
    resolver: TrustedTerminalRowResolver | None,
) -> AppealRow:
    common = (
        "unit_id",
        "challenger_id",
        "status",
        "world_redetermination_seed_sha256s",
        "evidence_domain",
        "evidence_receipt_sha256",
    )
    fields: dict[RowKind, tuple[str, ...]] = {
        "selection": (*common, "selection_score", "rng_key"),
        "h_multifidelity": (
            *common,
            "high_difference",
            "low_difference",
            "physical_coupling_key",
            "inner_rng_keys",
        ),
        "h_high_only": (*common, "high_difference", "physical_key", "inner_rng_keys"),
        "l": (*common, "low_difference", "physical_key", "inner_rng_keys"),
    }
    try:
        require_exact_keys(payload, required=fields[kind], where=f"{kind} journal row")
    except (KeyError, RivalSchemaError) as exc:
        raise AppealJournalError(str(exc)) from exc
    status, domain = _enum_fields(payload)
    world_redetermination_seed_sha256s = _require_string_list(
        payload["world_redetermination_seed_sha256s"],
        "world_redetermination_seed_sha256s",
    )
    for index, identity in enumerate(world_redetermination_seed_sha256s):
        _qualified_sha256(identity, f"world_redetermination_seed_sha256s[{index}]")
    if len(set(world_redetermination_seed_sha256s)) != len(world_redetermination_seed_sha256s):
        raise AppealJournalError("world-redetermination seed commitment reused within one row")
    expected = copy.deepcopy(dict(payload))
    if domain is not EvidenceDomain.CONTRACT_TEST:
        if resolver is None:
            raise AppealJournalError(
                "terminal/proxy journal replay requires a trusted row resolver"
            )
        resolver_payload = MappingProxyType(copy.deepcopy(expected))
        try:
            resolved = resolver(kind, resolver_payload)
        except Exception as exc:
            raise AppealJournalError(
                f"trusted terminal row resolver rejected {kind} row: {exc}"
            ) from exc
        if _row_kind(resolved) != kind or _row_record(resolved) != expected:
            raise AppealJournalError(
                "trusted terminal row resolver returned data that differs from the journal"
            )
        return resolved

    try:
        if kind == "selection":
            row: AppealRow = SelectionRow.contract_test(
                payload["unit_id"],
                payload["challenger_id"],
                payload["selection_score"],
                status,
                payload["rng_key"],
            )
        else:
            inner_rng_keys = _require_string_list(payload["inner_rng_keys"], "inner_rng_keys")
            if kind == "h_multifidelity":
                row = HRow.contract_test(
                    payload["unit_id"],
                    payload["challenger_id"],
                    payload["high_difference"],
                    payload["low_difference"],
                    status,
                    payload["physical_coupling_key"],
                    inner_rng_keys,
                )
            elif kind == "h_high_only":
                row = HighOnlyHRow.contract_test(
                    payload["unit_id"],
                    payload["challenger_id"],
                    payload["high_difference"],
                    status,
                    payload["physical_key"],
                    inner_rng_keys,
                )
            else:
                row = LRow.contract_test(
                    payload["unit_id"],
                    payload["challenger_id"],
                    payload["low_difference"],
                    status,
                    payload["physical_key"],
                    inner_rng_keys,
                )
        if _row_record(row) != expected:
            raise AppealJournalError(
                "contract-test row differs from its derived persisted evidence"
            )
        return row
    except (AppealError, KeyError, TypeError, ValueError) as exc:
        raise AppealJournalError(f"invalid contract-test {kind} row: {exc}") from exc


class AppealEventJournal:
    """Crash-reconstructible, single-look persistence for one appeal root."""

    def __init__(
        self,
        directory: str | Path,
        *,
        design: AppealDesign,
        terminal_row_resolver: TrustedTerminalRowResolver | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.events_directory = self.directory / "events"
        self.final_path = self.directory / "FINAL.json"
        self.lock_path = self.directory / ".journal.lock"
        self.design = design
        self.binding = _binding_for_design(design)
        self.terminal_row_resolver = terminal_row_resolver
        if terminal_row_resolver is not None:
            for field, expected in (
                ("root_id", self.binding.root_id),
                ("deployment_design_sha256", self.binding.deployment_design_sha256),
                ("manifest_content_sha256", self.binding.manifest_content_sha256),
            ):
                if getattr(terminal_row_resolver, field, None) != expected:
                    raise AppealJournalError(
                        f"trusted terminal row resolver {field} does not match journal design"
                    )

    @classmethod
    def create(
        cls,
        directory: str | Path,
        *,
        design: AppealDesign,
        terminal_row_resolver: TrustedTerminalRowResolver | None = None,
    ) -> AppealEventJournal:
        journal = cls(
            directory,
            design=design,
            terminal_row_resolver=terminal_row_resolver,
        )
        if journal.directory.is_symlink():
            raise AppealJournalError("journal directory cannot be a symbolic link")
        journal.directory.mkdir(parents=True, exist_ok=True)
        with journal._locked():
            journal._ensure_layout(create_events=True)
            if any(journal.events_directory.iterdir()) or _artifact_present(journal.final_path):
                raise AppealJournalError("refusing to reuse a non-empty appeal journal")
            header = journal._event_record(
                sequence=0,
                event_type="journal_open",
                payload={"format_version": _FORMAT_VERSION},
                previous_event_sha256=None,
            )
            journal._publish(journal._event_path(0), header)
            journal._load_events()
        return journal

    @classmethod
    def open(
        cls,
        directory: str | Path,
        *,
        design: AppealDesign,
        terminal_row_resolver: TrustedTerminalRowResolver | None = None,
    ) -> AppealEventJournal:
        journal = cls(
            directory,
            design=design,
            terminal_row_resolver=terminal_row_resolver,
        )
        journal.replay()
        return journal

    @contextmanager
    def _locked(self):
        if self.directory.is_symlink():
            raise AppealJournalError("journal directory cannot be a symbolic link")
        self.directory.mkdir(parents=True, exist_ok=True)
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(self.lock_path, flags, 0o600)
        except OSError as exc:
            raise AppealJournalError(f"cannot open journal lock safely: {exc}") from exc
        try:
            lock_stat = os.fstat(descriptor)
            if not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_nlink != 1:
                raise AppealJournalError("journal lock must be a single-link regular file")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _ensure_layout(self, *, create_events: bool = False) -> None:
        if not self.directory.is_dir() or self.directory.is_symlink():
            raise AppealJournalError("journal path must be a real directory")
        if create_events:
            self.events_directory.mkdir(exist_ok=True)
        if not self.events_directory.is_dir() or self.events_directory.is_symlink():
            raise AppealJournalError("journal events path must be a real directory")
        allowed = {"events", ".journal.lock", "FINAL.json"}
        unexpected = sorted(
            path.name for path in self.directory.iterdir() if path.name not in allowed
        )
        if unexpected:
            raise AppealJournalError(f"unexpected journal artifacts: {unexpected}")

    def _event_path(self, sequence: int) -> Path:
        return self.events_directory / f"{sequence:016d}.json"

    @staticmethod
    def _publish(path: Path, record: Mapping[str, Any]) -> None:
        try:
            write_new_canonical_json(path, record)
        except (OSError, RivalSchemaError) as exc:
            raise AppealJournalError(
                f"failed to publish immutable journal artifact {path}: {exc}"
            ) from exc

    def _event_record(
        self,
        *,
        sequence: int,
        event_type: str,
        payload: Mapping[str, Any],
        previous_event_sha256: str | None,
    ) -> dict[str, Any]:
        return attach_content_hash(
            {
                "schema_id": APPEAL_EVENT_SCHEMA_ID,
                "format_version": _FORMAT_VERSION,
                "sequence": sequence,
                "event_type": event_type,
                "appeal_mode": self.binding.mode,
                "root_id": self.binding.root_id,
                "deployment_design_sha256": self.binding.deployment_design_sha256,
                "manifest_content_sha256": self.binding.manifest_content_sha256,
                "previous_event_sha256": previous_event_sha256,
                "payload": dict(payload),
            }
        )

    def _load_events(self) -> list[dict[str, Any]]:
        self._ensure_layout()
        paths: list[tuple[int, Path]] = []
        for path in self.events_directory.iterdir():
            match = _EVENT_NAME.fullmatch(path.name)
            if match is None:
                raise AppealJournalError(f"unexpected event artifact name: {path.name}")
            paths.append((int(match.group("sequence")), path))
        paths.sort(key=lambda pair: pair[0])
        if not paths:
            raise AppealJournalError("appeal journal has no opening event")
        observed_sequences = [sequence for sequence, _ in paths]
        expected_sequences = list(range(len(paths)))
        if observed_sequences != expected_sequences:
            raise AppealJournalError(
                "event sequence is gapped, duplicated, or does not begin at zero: "
                f"observed={observed_sequences}"
            )

        events: list[dict[str, Any]] = []
        previous: str | None = None
        event_fields = (
            "schema_id",
            "format_version",
            "sequence",
            "event_type",
            "appeal_mode",
            "root_id",
            "deployment_design_sha256",
            "manifest_content_sha256",
            "previous_event_sha256",
            "payload",
            "content_sha256",
        )
        for sequence, path in paths:
            record = _strict_json_record(path)
            try:
                require_exact_keys(record, required=event_fields, where=f"event {sequence}")
                digest = verify_content_hash(record)
            except RivalSchemaError as exc:
                raise AppealJournalError(
                    f"event {sequence} failed schema validation: {exc}"
                ) from exc
            if record["schema_id"] != APPEAL_EVENT_SCHEMA_ID:
                raise AppealJournalError(f"event {sequence} has wrong schema_id")
            if type(record["format_version"]) is not int or record["format_version"] != (
                _FORMAT_VERSION
            ):
                raise AppealJournalError(f"event {sequence} has wrong format version")
            if type(record["sequence"]) is not int or record["sequence"] != sequence:
                raise AppealJournalError(f"event {sequence} sequence field does not match filename")
            if (
                not isinstance(record["event_type"], str)
                or record["event_type"] not in _EVENT_TYPES
            ):
                raise AppealJournalError(f"event {sequence} has unknown event_type")
            for field, expected in (
                ("appeal_mode", self.binding.mode),
                ("root_id", self.binding.root_id),
                ("deployment_design_sha256", self.binding.deployment_design_sha256),
                ("manifest_content_sha256", self.binding.manifest_content_sha256),
                ("previous_event_sha256", previous),
            ):
                if record[field] != expected:
                    raise AppealJournalError(
                        f"event {sequence} {field} does not match its frozen binding"
                    )
            if not isinstance(record["payload"], dict):
                raise AppealJournalError(f"event {sequence} payload must be an object")
            previous = "sha256:" + digest
            events.append(record)
        if events[0]["event_type"] != "journal_open":
            raise AppealJournalError("event zero must be journal_open")
        try:
            require_exact_keys(
                events[0]["payload"], required=("format_version",), where="opening payload"
            )
        except RivalSchemaError as exc:
            raise AppealJournalError(str(exc)) from exc
        if (
            type(events[0]["payload"]["format_version"]) is not int
            or events[0]["payload"]["format_version"] != _FORMAT_VERSION
        ):
            raise AppealJournalError("opening payload has wrong format version")
        if any(event["event_type"] == "journal_open" for event in events[1:]):
            raise AppealJournalError("journal_open event may occur only at sequence zero")
        return events

    def _new_machine(self) -> AppealMachine:
        if self.binding.mode == "multifidelity":
            assert isinstance(self.design, VerifiedMultifidelityDesign)
            return AppealStateMachine(design=self.design)
        assert isinstance(self.design, VerifiedHighFidelityDesign)
        return HighFidelityAppealStateMachine(design=self.design)

    def _apply_event(self, machine: AppealMachine, event: Mapping[str, Any]) -> None:
        event_type = event["event_type"]
        payload = event["payload"]
        assert isinstance(payload, Mapping)
        if event_type == "journal_open":
            return
        try:
            if event_type == "selection_row":
                machine.add_selection(
                    _deserialize_row("selection", payload, self.terminal_row_resolver)  # type: ignore[arg-type]
                )
            elif event_type == "freeze_challenger":
                require_exact_keys(
                    payload,
                    required=("challenger_id",),
                    where="freeze_challenger payload",
                )
                machine.freeze_challenger(payload["challenger_id"])
            elif event_type == "h_row":
                kind: RowKind = (
                    "h_multifidelity" if self.binding.mode == "multifidelity" else "h_high_only"
                )
                row = _deserialize_row(kind, payload, self.terminal_row_resolver)
                machine.add_h(row)  # type: ignore[arg-type]
            elif event_type == "l_row":
                if self.binding.mode != "multifidelity":
                    raise AppealJournalError("L event is forbidden in high-fidelity mode")
                machine.add_l(
                    _deserialize_row("l", payload, self.terminal_row_resolver)  # type: ignore[arg-type]
                )
            else:  # guarded by the event schema
                raise AppealJournalError(f"unknown event type {event_type!r}")
        except (AppealError, RivalSchemaError, KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, AppealJournalError):
                raise
            raise AppealJournalError(f"event transition rejected: {exc}") from exc

    def _replay_machine(self, events: list[dict[str, Any]]) -> AppealMachine:
        machine = self._new_machine()
        for event in events:
            try:
                self._apply_event(machine, event)
            except AppealJournalError as exc:
                raise AppealJournalError(
                    f"event {event['sequence']} cannot be replayed: {exc}"
                ) from exc
        return machine

    def _final_record(self, events: list[dict[str, Any]], decision: AppealResult) -> dict[str, Any]:
        decision_record = _decision_record(decision)
        return attach_content_hash(
            {
                "schema_id": APPEAL_FINAL_SCHEMA_ID,
                "format_version": _FORMAT_VERSION,
                "appeal_mode": self.binding.mode,
                "root_id": self.binding.root_id,
                "deployment_design_sha256": self.binding.deployment_design_sha256,
                "manifest_content_sha256": self.binding.manifest_content_sha256,
                "event_count": len(events),
                "event_tip_sha256": "sha256:" + events[-1]["content_sha256"],
                "decision_sha256": "sha256:" + sha256_hex(decision_record),
                "decision": decision_record,
            }
        )

    def _load_final(
        self, events: list[dict[str, Any]], machine: AppealMachine
    ) -> AppealResult | None:
        if not _artifact_present(self.final_path):
            return None
        record = _strict_json_record(self.final_path)
        fields = (
            "schema_id",
            "format_version",
            "appeal_mode",
            "root_id",
            "deployment_design_sha256",
            "manifest_content_sha256",
            "event_count",
            "event_tip_sha256",
            "decision_sha256",
            "decision",
            "content_sha256",
        )
        try:
            require_exact_keys(record, required=fields, where="FINAL.json")
            verify_content_hash(record)
        except RivalSchemaError as exc:
            raise AppealJournalError(f"FINAL.json failed schema validation: {exc}") from exc
        expected_binding = {
            "schema_id": APPEAL_FINAL_SCHEMA_ID,
            "format_version": _FORMAT_VERSION,
            "appeal_mode": self.binding.mode,
            "root_id": self.binding.root_id,
            "deployment_design_sha256": self.binding.deployment_design_sha256,
            "manifest_content_sha256": self.binding.manifest_content_sha256,
            "event_count": len(events),
            "event_tip_sha256": "sha256:" + events[-1]["content_sha256"],
        }
        for field, expected in expected_binding.items():
            if record[field] != expected:
                raise AppealJournalError(f"FINAL.json {field} does not match journal history")
        if type(record["format_version"]) is not int:
            raise AppealJournalError("FINAL.json format_version must be an integer")
        if type(record["event_count"]) is not int:
            raise AppealJournalError("FINAL.json event_count must be an integer")
        if not isinstance(record["decision"], dict):
            raise AppealJournalError("FINAL.json decision must be an object")
        expected_decision_hash = "sha256:" + sha256_hex(record["decision"])
        if record["decision_sha256"] != expected_decision_hash:
            raise AppealJournalError("FINAL.json decision hash mismatch")
        try:
            recomputed = machine.finalize()
        except AppealError as exc:
            raise AppealJournalError(
                f"FINAL.json exists for a non-finalizable history: {exc}"
            ) from exc
        if _decision_record(recomputed) != record["decision"]:
            raise AppealJournalError("FINAL.json decision differs from deterministic replay")
        return recomputed

    def _snapshot(
        self,
        events: list[dict[str, Any]],
        machine: AppealMachine,
        decision: AppealResult | None,
    ) -> JournalSnapshot:
        return JournalSnapshot(
            mode=self.binding.mode,
            root_id=self.binding.root_id,
            deployment_design_sha256=self.binding.deployment_design_sha256,
            manifest_content_sha256=self.binding.manifest_content_sha256,
            event_count=len(events),
            event_tip_sha256="sha256:" + events[-1]["content_sha256"],
            selected_challenger_id=machine._selected,
            operational=machine.operational_accounting(),
            finalized=decision is not None,
            decision=decision,
        )

    def replay(self) -> JournalSnapshot:
        """Validate the whole chain and return a non-peekable state snapshot."""

        with self._locked():
            events = self._load_events()
            machine = self._replay_machine(events)
            decision = self._load_final(events, machine)
            return self._snapshot(events, machine, decision)

    def _append_many(self, transitions: Iterable[tuple[str, Mapping[str, Any]]]) -> JournalSnapshot:
        """Validate then durably append a panel batch with one history replay.

        Validation is all-or-nothing: no event is published unless every
        transition is accepted by the reconstructed state machine.  Publication
        is deliberately prefix-durable, so a process crash can leave only a
        valid, replayable prefix rather than an opaque aggregate file.
        """

        pending = tuple(transitions)
        if not pending:
            raise AppealJournalError("appeal journal append batch cannot be empty")
        with self._locked():
            events = self._load_events()
            if _artifact_present(self.final_path):
                raise AppealJournalError("appeal journal is final; post-final writes are forbidden")
            machine = self._replay_machine(events)
            candidates: list[dict[str, Any]] = []
            previous = "sha256:" + events[-1]["content_sha256"]
            for offset, (event_type, payload) in enumerate(pending):
                if event_type not in _EVENT_TYPES - {"journal_open"}:
                    raise AppealJournalError(f"cannot append event type {event_type!r}")
                candidate = self._event_record(
                    sequence=len(events) + offset,
                    event_type=event_type,
                    payload=payload,
                    previous_event_sha256=previous,
                )
                # Validate every exact persisted representation before the
                # first create-new publication occurs.
                self._apply_event(machine, candidate)
                candidates.append(candidate)
                previous = "sha256:" + candidate["content_sha256"]
            for candidate in candidates:
                self._publish(self._event_path(candidate["sequence"]), candidate)
            events.extend(candidates)
            return self._snapshot(events, machine, None)

    def add_selection(self, row: SelectionRow) -> JournalSnapshot:
        return self.add_selections((row,))

    def add_selections(self, rows: Iterable[SelectionRow]) -> JournalSnapshot:
        """Append an S panel or S-panel tranche with one verified replay."""

        materialized = tuple(rows)
        if not materialized:
            raise AppealJournalError("selection row batch cannot be empty")
        transitions: list[tuple[str, Mapping[str, Any]]] = []
        for row in materialized:
            if not isinstance(row, SelectionRow):
                raise AppealJournalError("selection event requires SelectionRow")
            transitions.append(("selection_row", _row_record(row)))
        return self._append_many(transitions)

    def freeze_challenger(self, challenger_id: str) -> JournalSnapshot:
        if not isinstance(challenger_id, str):
            raise AppealJournalError("challenger_id must be a string")
        return self._append_many((("freeze_challenger", {"challenger_id": challenger_id}),))

    def add_h(self, row: HRow | HighOnlyHRow) -> JournalSnapshot:
        return self.add_h_rows((row,))

    def add_h_rows(self, rows: Iterable[HRow | HighOnlyHRow]) -> JournalSnapshot:
        """Append an H panel or H-panel tranche with one verified replay."""

        materialized = tuple(rows)
        if not materialized:
            raise AppealJournalError("H row batch cannot be empty")
        expected_kind: RowKind = (
            "h_multifidelity" if self.binding.mode == "multifidelity" else "h_high_only"
        )
        transitions: list[tuple[str, Mapping[str, Any]]] = []
        for row in materialized:
            if _row_kind(row) != expected_kind:
                raise AppealJournalError(f"{self.binding.mode} journal rejects this H row type")
            transitions.append(("h_row", _row_record(row)))
        return self._append_many(transitions)

    def add_l(self, row: LRow) -> JournalSnapshot:
        return self.add_l_rows((row,))

    def add_l_rows(self, rows: Iterable[LRow]) -> JournalSnapshot:
        """Append an L panel or L-panel tranche with one verified replay."""

        if self.binding.mode != "multifidelity":
            raise AppealJournalError("L events are forbidden in high-fidelity-only mode")
        materialized = tuple(rows)
        if not materialized:
            raise AppealJournalError("L row batch cannot be empty")
        transitions: list[tuple[str, Mapping[str, Any]]] = []
        for row in materialized:
            if not isinstance(row, LRow):
                raise AppealJournalError("L event requires LRow")
            transitions.append(("l_row", _row_record(row)))
        return self._append_many(transitions)

    def finalize(self) -> AppealResult:
        """Consume and durably publish the journal's sole inferential look."""

        with self._locked():
            events = self._load_events()
            # Check finality before replaying or evaluating any outcomes.  A
            # second call therefore cannot become a second inferential peek.
            if _artifact_present(self.final_path):
                raise AppealJournalError("appeal journal has already consumed its inferential look")
            machine = self._replay_machine(events)
            try:
                decision = machine.finalize()
            except AppealError as exc:
                raise AppealJournalError(f"appeal is not finalizable: {exc}") from exc
            self._publish(self.final_path, self._final_record(events, decision))
            return decision


__all__ = [
    "APPEAL_EVENT_SCHEMA_ID",
    "APPEAL_FINAL_SCHEMA_ID",
    "AppealEventJournal",
    "AppealJournalError",
    "JournalSnapshot",
    "TrustedTerminalRowResolver",
]
