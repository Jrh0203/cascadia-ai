"""Immutable, exact unit plans for Rival's within-root panels.

Panel identities in a root manifest are content hashes, not decorative digest
slots.  This module validates the artifact behind each identity and is the
only constructor for terminal-unit expectations consumed by the Rust evidence
adapter.  Consequently a caller cannot choose a seat, challenger, unit index,
or branch-local policy memory after observing terminal outcomes.

The plan contains only inputs known before a panel runs.  A terminal pair's
``pair_sha256`` is deliberately *not* part of the plan because it commits to
the resulting trajectories and scores; that post-run content pin belongs to
the evidence reference instead.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from .manifest import (
    ACTION_CONTENT_ID_PREFIX,
    CANDIDATE_OCCURRENCE_ID_PREFIX,
    RootManifest,
    require_externally_pinned_root_manifest,
)
from .schema import (
    RIVAL_TERMINAL_PANEL_PLAN_SCHEMA_ID,
    RivalSchemaError,
    _issue_validation_capability,
    _require_validation_capability,
    read_pinned_canonical_json_object,
    require_exact_keys,
    require_nonempty_string,
    require_schema,
    require_sha256,
    sha256_hex,
    verify_content_hash,
)

PanelKind = Literal["S", "H", "L"]
PanelFidelity = Literal["high", "low", "paired_high_low"]

_PLAN_FIELDS = (
    "schema_id",
    "plan_id",
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
    "panel_kind",
    "units",
    "content_sha256",
)
_UNIT_FIELDS = (
    "unit_index",
    "fidelity",
    "target_seat",
    "challenger_candidate_occurrence_id",
    "challenger_action_content_id",
    "incumbent_post_action_memory_sha256",
    "challenger_post_action_memory_sha256",
)


class PanelPlanError(ValueError):
    """Raised when a panel plan is incomplete, adaptive, or substituted."""


def _qualified_sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise PanelPlanError(f"{field_name} must use the 'sha256:' wire")
    try:
        require_sha256(value, field_name)
    except RivalSchemaError as exc:
        raise PanelPlanError(str(exc)) from exc
    return value


def _namespaced(value: Any, field_name: str, prefix: str) -> str:
    if not isinstance(value, str) or not value.startswith(prefix):
        raise PanelPlanError(f"{field_name} must use namespace {prefix!r}")
    try:
        require_sha256(value.removeprefix(prefix), field_name)
    except RivalSchemaError as exc:
        raise PanelPlanError(str(exc)) from exc
    return value


def _uint32(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**32:
        raise PanelPlanError(f"{field_name} must be a uint32 integer")
    return value


def _seat(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 3:
        raise PanelPlanError("target_seat must be an integer in [0, 3]")
    return value


@dataclass(frozen=True)
class TerminalPanelUnit:
    """One pre-outcome unit in an immutable panel plan."""

    unit_index: int
    fidelity: PanelFidelity
    target_seat: int
    challenger_candidate_occurrence_id: str
    challenger_action_content_id: str
    incumbent_post_action_memory_sha256: str
    challenger_post_action_memory_sha256: str
    _validation_capability: object | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self._validation_capability is not None:
            self.require_validated_artifact()

    def public_record(self) -> dict[str, Any]:
        return {
            "unit_index": self.unit_index,
            "fidelity": self.fidelity,
            "target_seat": self.target_seat,
            "challenger_candidate_occurrence_id": self.challenger_candidate_occurrence_id,
            "challenger_action_content_id": self.challenger_action_content_id,
            "incumbent_post_action_memory_sha256": (self.incumbent_post_action_memory_sha256),
            "challenger_post_action_memory_sha256": (self.challenger_post_action_memory_sha256),
        }

    def require_validated_artifact(self) -> None:
        try:
            _require_validation_capability(
                self._validation_capability,
                artifact_kind="TerminalPanelUnit",
                content_sha256=_unit_runtime_fingerprint(self),
            )
        except RivalSchemaError as exc:
            raise PanelPlanError(str(exc)) from exc


def _unit_runtime_fingerprint(unit: TerminalPanelUnit) -> str:
    return sha256_hex(
        {
            "schema_id": "cascadiav3.rival_validated_terminal_panel_unit_runtime.v1",
            "fields": unit.public_record(),
        }
    )


@dataclass(frozen=True)
class TerminalUnitExpectation:
    """Validated plan membership for one high-fidelity Rust terminal pair."""

    panel_kind: Literal["S", "H"]
    panel_id: str
    unit_index: int
    fidelity: Literal["high"]
    target_seat: int
    challenger_candidate_occurrence_id: str
    challenger_action_content_id: str
    incumbent_post_action_memory_sha256: str
    challenger_post_action_memory_sha256: str
    _unit_record_sha256: str = field(repr=False, compare=False)
    _validation_capability: object | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self._validation_capability is not None:
            self._require_validated_artifact()

    @property
    def unit_id(self) -> str:
        public = self._public_record()
        return f"cascadiav3.rival_verified_terminal_unit.v1:sha256:{sha256_hex(public)}"

    def _public_record(self) -> dict[str, Any]:
        return {
            "panel_kind": self.panel_kind,
            "panel_id": self.panel_id,
            "unit_index": self.unit_index,
            "fidelity": self.fidelity,
            "target_seat": self.target_seat,
            "challenger_candidate_occurrence_id": (self.challenger_candidate_occurrence_id),
            "challenger_action_content_id": self.challenger_action_content_id,
            "incumbent_post_action_memory_sha256": (self.incumbent_post_action_memory_sha256),
            "challenger_post_action_memory_sha256": (self.challenger_post_action_memory_sha256),
        }

    def validate(self, manifest: RootManifest) -> None:
        """Rejoin this plan-issued capability to the verifier's manifest."""

        self._require_validated_artifact()
        if not isinstance(manifest, RootManifest) or not manifest.validated:
            raise PanelPlanError("terminal expectation requires a validated RootManifest")
        if sha256_hex(self._public_record()) != self._unit_record_sha256:
            raise PanelPlanError("terminal expectation was mutated after plan selection")
        if self.panel_kind not in manifest.required_panels:
            raise PanelPlanError("terminal expectation panel is not required by the manifest")
        if manifest.panel_identity(self.panel_kind) != self.panel_id:
            raise PanelPlanError("terminal expectation belongs to a different panel plan")
        matching = [
            row
            for row in manifest.candidate_selection_entries
            if row.candidate_action_occurrence_id == self.challenger_candidate_occurrence_id
        ]
        if len(matching) != 1:
            raise PanelPlanError(
                "terminal expectation challenger is absent from the frozen candidate menu"
            )
        if matching[0].action_content_id != self.challenger_action_content_id:
            raise PanelPlanError("terminal expectation challenger/action binding was substituted")
        if self.challenger_candidate_occurrence_id == manifest.incumbent_candidate_occurrence_id:
            raise PanelPlanError("terminal expectation challenger equals the incumbent")

    def _require_validated_artifact(self) -> None:
        try:
            _require_validation_capability(
                self._validation_capability,
                artifact_kind="TerminalUnitExpectation",
                content_sha256=_expectation_runtime_fingerprint(self),
            )
        except RivalSchemaError as exc:
            raise PanelPlanError(str(exc)) from exc


def _expectation_runtime_fingerprint(expectation: TerminalUnitExpectation) -> str:
    return sha256_hex(
        {
            "schema_id": "cascadiav3.rival_validated_terminal_expectation_runtime.v1",
            "plan_identity": expectation.panel_id,
            "unit_record_sha256": expectation._unit_record_sha256,
            "fields": expectation._public_record(),
        }
    )


@dataclass(frozen=True)
class TerminalPanelPlan:
    """A validated, manifest-bound, ordered panel schedule."""

    plan_id: str
    manifest_id: str
    root_id: str
    ruleset_identity: str
    source_game_identity_sha256: str
    candidate_set_identity: str
    incumbent_policy_identity: str
    incumbent_candidate_occurrence_id: str
    incumbent_action_content_id: str
    sampler_identity: str
    policy_rng_factory_identity: str
    panel_kind: PanelKind
    units: tuple[TerminalPanelUnit, ...]
    content_sha256: str
    _validation_capability: object | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self._validation_capability is not None:
            self.validate_integrity()

    @property
    def identity(self) -> str:
        return "sha256:" + self.content_sha256

    def _content_record(self) -> dict[str, Any]:
        return {
            "schema_id": RIVAL_TERMINAL_PANEL_PLAN_SCHEMA_ID,
            "plan_id": self.plan_id,
            "manifest_id": self.manifest_id,
            "root_id": self.root_id,
            "ruleset_identity": self.ruleset_identity,
            "source_game_identity_sha256": self.source_game_identity_sha256,
            "candidate_set_identity": self.candidate_set_identity,
            "incumbent_policy_identity": self.incumbent_policy_identity,
            "incumbent_candidate_occurrence_id": (self.incumbent_candidate_occurrence_id),
            "incumbent_action_content_id": self.incumbent_action_content_id,
            "sampler_identity": self.sampler_identity,
            "policy_rng_factory_identity": self.policy_rng_factory_identity,
            "panel_kind": self.panel_kind,
            "units": [unit.public_record() for unit in self.units],
        }

    def validate_integrity(self) -> None:
        for unit in self.units:
            unit.require_validated_artifact()
        try:
            _require_validation_capability(
                self._validation_capability,
                artifact_kind="TerminalPanelPlan",
                content_sha256=_plan_runtime_fingerprint(self),
            )
        except RivalSchemaError as exc:
            raise PanelPlanError(str(exc)) from exc
        if sha256_hex(self._content_record()) != self.content_sha256:
            raise PanelPlanError("terminal panel plan was forged or mutated after validation")

    def high_fidelity_expectation(self, unit_index: int) -> TerminalUnitExpectation:
        """Return the one plan-proven high-fidelity unit at ``unit_index``."""

        self.validate_integrity()
        matches = [unit for unit in self.units if unit.unit_index == unit_index]
        if len(matches) != 1:
            raise PanelPlanError(f"panel has no unique unit_index {unit_index}")
        unit = matches[0]
        if self.panel_kind not in {"S", "H"} or unit.fidelity != "high":
            raise PanelPlanError(
                "the v1 Rust terminal adapter accepts only high-fidelity S/H units"
            )
        public = {
            "panel_kind": self.panel_kind,
            "panel_id": self.identity,
            **unit.public_record(),
        }
        expectation = TerminalUnitExpectation(
            panel_kind=self.panel_kind,
            panel_id=self.identity,
            unit_index=unit.unit_index,
            fidelity="high",
            target_seat=unit.target_seat,
            challenger_candidate_occurrence_id=(unit.challenger_candidate_occurrence_id),
            challenger_action_content_id=unit.challenger_action_content_id,
            incumbent_post_action_memory_sha256=(unit.incumbent_post_action_memory_sha256),
            challenger_post_action_memory_sha256=(unit.challenger_post_action_memory_sha256),
            _unit_record_sha256=sha256_hex(public),
        )
        return replace(
            expectation,
            _validation_capability=_issue_validation_capability(
                "TerminalUnitExpectation",
                _expectation_runtime_fingerprint(expectation),
            ),
        )


def _plan_runtime_fingerprint(plan: TerminalPanelPlan) -> str:
    return sha256_hex(
        {
            "schema_id": "cascadiav3.rival_validated_terminal_panel_plan_runtime.v1",
            "content_sha256": plan.content_sha256,
            "fields": plan._content_record(),
        }
    )


def _expected_fidelity(manifest: RootManifest, panel_kind: PanelKind) -> PanelFidelity:
    if manifest.inference_mode == "high_fidelity_only":
        if panel_kind not in {"S", "H"}:
            raise PanelPlanError("high-fidelity-only designs cannot preregister an L panel")
        return "high"
    return {"S": "low", "H": "paired_high_low", "L": "low"}[panel_kind]


def validate_terminal_panel_plan(
    record: Mapping[str, Any], *, manifest: RootManifest
) -> TerminalPanelPlan:
    """Validate one canonical panel plan and join it to a validated root manifest."""

    if not isinstance(manifest, RootManifest):
        raise PanelPlanError("panel plan requires a validated RootManifest")
    # ``RootManifest`` has a validator capability in addition to its type.  A
    # local import-free getattr keeps this module compatible while callers are
    # migrated and fails closed for hand-constructed dataclasses.
    if not manifest.validated:
        raise PanelPlanError("panel plan rejects a hand-constructed RootManifest")
    try:
        require_schema(record, RIVAL_TERMINAL_PANEL_PLAN_SCHEMA_ID)
        require_exact_keys(record, required=_PLAN_FIELDS, where="terminal panel plan")
        content_sha256 = verify_content_hash(record)
    except RivalSchemaError as exc:
        raise PanelPlanError(str(exc)) from exc

    try:
        plan_id = require_nonempty_string(record["plan_id"], "plan_id")
        manifest_id = require_nonempty_string(record["manifest_id"], "manifest_id")
    except RivalSchemaError as exc:
        raise PanelPlanError(str(exc)) from exc
    panel_kind_raw = record["panel_kind"]
    if panel_kind_raw not in {"S", "H", "L"}:
        raise PanelPlanError("panel_kind must be S, H, or L")
    panel_kind: PanelKind = panel_kind_raw

    identity = "sha256:" + content_sha256
    if panel_kind not in manifest.required_panels:
        raise PanelPlanError("panel plan kind is not required by the root manifest")
    if manifest.panel_identity(panel_kind) != identity:
        raise PanelPlanError("panel plan content hash does not match manifest panel identity")

    expected_manifest_fields = {
        "manifest_id": manifest.manifest_id,
        "root_id": manifest.root_id,
        "ruleset_identity": manifest.ruleset_identity,
        "source_game_identity_sha256": manifest.source_game_identity_sha256,
        "candidate_set_identity": manifest.candidate_set_identity,
        "incumbent_policy_identity": manifest.incumbent_policy_identity,
        "incumbent_candidate_occurrence_id": (manifest.incumbent_candidate_occurrence_id),
        "incumbent_action_content_id": manifest.incumbent_action_id,
        "sampler_identity": manifest.sampler_identity,
        "policy_rng_factory_identity": manifest.policy_rng_factory_identity,
    }
    mismatches = [
        key for key, expected in expected_manifest_fields.items() if record[key] != expected
    ]
    if mismatches:
        raise PanelPlanError(
            "panel plan does not join to the frozen root manifest: " + ", ".join(sorted(mismatches))
        )

    units_raw = record["units"]
    if not isinstance(units_raw, list) or not units_raw:
        raise PanelPlanError("terminal panel plan units must be a non-empty list")
    menu = {row.candidate_action_occurrence_id: row for row in manifest.candidate_selection_entries}
    eligible = {
        occurrence: row
        for occurrence, row in menu.items()
        if occurrence != manifest.incumbent_candidate_occurrence_id
    }
    expected_fidelity = _expected_fidelity(manifest, panel_kind)
    units: list[TerminalPanelUnit] = []
    for position, raw in enumerate(units_raw):
        if not isinstance(raw, Mapping):
            raise PanelPlanError(f"units[{position}] must be an object")
        try:
            require_exact_keys(
                raw, required=_UNIT_FIELDS, where=f"terminal panel units[{position}]"
            )
        except RivalSchemaError as exc:
            raise PanelPlanError(str(exc)) from exc
        unit_index = _uint32(raw["unit_index"], f"units[{position}].unit_index")
        if unit_index != position:
            raise PanelPlanError("unit_index values must be the canonical contiguous order 0..n-1")
        if raw["fidelity"] != expected_fidelity:
            raise PanelPlanError(
                f"{panel_kind} units require fidelity {expected_fidelity!r} "
                f"under {manifest.inference_mode}"
            )
        challenger = _namespaced(
            raw["challenger_candidate_occurrence_id"],
            f"units[{position}].challenger_candidate_occurrence_id",
            CANDIDATE_OCCURRENCE_ID_PREFIX,
        )
        if challenger not in eligible:
            raise PanelPlanError("panel unit challenger is not an eligible manifest candidate")
        action = _namespaced(
            raw["challenger_action_content_id"],
            f"units[{position}].challenger_action_content_id",
            ACTION_CONTENT_ID_PREFIX,
        )
        if eligible[challenger].action_content_id != action:
            raise PanelPlanError("panel unit challenger/action binding was substituted")
        unit = TerminalPanelUnit(
            unit_index=unit_index,
            fidelity=expected_fidelity,
            target_seat=_seat(raw["target_seat"]),
            challenger_candidate_occurrence_id=challenger,
            challenger_action_content_id=action,
            incumbent_post_action_memory_sha256=_qualified_sha256(
                raw["incumbent_post_action_memory_sha256"],
                f"units[{position}].incumbent_post_action_memory_sha256",
            ),
            challenger_post_action_memory_sha256=_qualified_sha256(
                raw["challenger_post_action_memory_sha256"],
                f"units[{position}].challenger_post_action_memory_sha256",
            ),
        )
        units.append(
            replace(
                unit,
                _validation_capability=_issue_validation_capability(
                    "TerminalPanelUnit",
                    _unit_runtime_fingerprint(unit),
                ),
            )
        )

    observed = Counter(unit.challenger_candidate_occurrence_id for unit in units)
    if panel_kind == "S":
        expected = {occurrence: row.expected_s for occurrence, row in eligible.items()}
    else:
        per_candidate = manifest.expected_h if panel_kind == "H" else manifest.expected_l
        expected = {occurrence: per_candidate for occurrence in eligible}
    if dict(observed) != expected:
        raise PanelPlanError(
            f"{panel_kind} plan does not contain the exact conditional allocation: "
            f"observed={dict(observed)!r}; expected={expected!r}"
        )

    plan = TerminalPanelPlan(
        plan_id=plan_id,
        manifest_id=manifest_id,
        root_id=manifest.root_id,
        ruleset_identity=manifest.ruleset_identity,
        source_game_identity_sha256=manifest.source_game_identity_sha256,
        candidate_set_identity=manifest.candidate_set_identity,
        incumbent_policy_identity=manifest.incumbent_policy_identity,
        incumbent_candidate_occurrence_id=manifest.incumbent_candidate_occurrence_id,
        incumbent_action_content_id=manifest.incumbent_action_id,
        sampler_identity=manifest.sampler_identity,
        policy_rng_factory_identity=manifest.policy_rng_factory_identity,
        panel_kind=panel_kind,
        units=tuple(units),
        content_sha256=content_sha256,
    )
    return replace(
        plan,
        _validation_capability=_issue_validation_capability(
            "TerminalPanelPlan",
            _plan_runtime_fingerprint(plan),
        ),
    )


def load_terminal_panel_plan(
    path: str | Path,
    *,
    manifest: RootManifest,
    expected_file_sha256: str,
) -> TerminalPanelPlan:
    """Load one byte-pinned canonical plan and join its semantic content hash."""

    try:
        require_externally_pinned_root_manifest(manifest)
    except RivalSchemaError as exc:
        raise PanelPlanError(str(exc)) from exc
    try:
        record = read_pinned_canonical_json_object(
            path,
            expected_file_sha256=expected_file_sha256,
            field="terminal panel plan",
        )
    except RivalSchemaError as exc:
        raise PanelPlanError(str(exc)) from exc
    return validate_terminal_panel_plan(record, manifest=manifest)
