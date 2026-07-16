"""Single-challenger S/H/L appeal protocol and categorical label emission."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from statistics import fmean
from typing import Literal

from .bounds import (
    BoundCertificate,
    CertifiedRange,
    HighOnlyErrorAllocation,
    HighOnlyHoeffdingBound,
    HoeffdingBound,
    RootErrorAllocation,
    TransformedWidths,
    fixed_high_only_hoeffding_lower_bound,
    fixed_hoeffding_lower_bound,
    require_validated_bound_certificate,
    transformed_widths_from_certificate,
)
from .coverage import ErrorFamilyLedger, HighOnlyRootErrorEntry, RootErrorEntry
from .manifest import (
    CandidateSelectionEntry,
    RootManifest,
    require_validated_root_manifest,
)
from .multifidelity import (
    CoefficientBinding,
    HighDifference,
    HighFidelityEstimate,
    LowDifference,
    MultifidelityEstimate,
    PairedDifference,
    estimate_fixed_panels,
    estimate_high_fidelity_only,
)
from .schema import (
    RivalSchemaError,
    _issue_validation_capability,
    _require_validation_capability,
    require_sha256,
    sha256_hex,
)


class AppealError(ValueError):
    """Raised when the preregistered appeal state machine is violated."""


def _identity(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise AppealError(f"{name} must be a non-empty, whitespace-trimmed string")
    return value


class UnitStatus(StrEnum):
    COMPLETE = "complete"
    TIMEOUT = "timeout"
    INVALID = "invalid"


class EvidenceDomain(StrEnum):
    CONTRACT_TEST = "synthetic_contract_test"
    CPU_PROXY_REFERENCE = "cpu_proxy_reference"
    PRODUCTION_TERMINAL = "production_terminal"


def _synthetic_world_redetermination_seed_sha256s(source_key: str) -> tuple[str, str]:
    """Derive unmistakably synthetic pair commitments for contract fixtures."""

    source_key = _identity(source_key, "contract-test redetermination-seed source key")
    return tuple(
        "sha256:"
        + sha256_hex(
            {
                "schema_id": "cascadiav3.rival_synthetic_world_redetermination_seed.v1",
                "source_key": source_key,
                "side": side,
            }
        )
        for side in ("incumbent", "challenger")
    )


def _validate_world_redetermination_seed_sha256s(values: tuple[str, ...]) -> None:
    if not isinstance(values, tuple) or not values:
        raise AppealError("world_redetermination_seed_sha256s must be a non-empty immutable tuple")
    for index, value in enumerate(values):
        try:
            require_sha256(value, f"world_redetermination_seed_sha256s[{index}]")
        except RivalSchemaError as exc:
            raise AppealError(str(exc)) from exc
    if len(set(values)) != len(values):
        raise AppealError("world-redetermination seed commitment reused within one row")


def _validate_evidence_binding(
    domain: EvidenceDomain,
    receipt_sha256: str | None,
) -> None:
    if not isinstance(domain, EvidenceDomain):
        raise AppealError("evidence_domain must be a typed EvidenceDomain")
    if domain is EvidenceDomain.PRODUCTION_TERMINAL:
        raise AppealError(
            "production-terminal evidence is structurally unavailable before "
            "a production Rust adapter is admitted"
        )
    if domain is EvidenceDomain.CONTRACT_TEST:
        if receipt_sha256 is not None:
            raise AppealError("contract-test evidence requires its explicit fixture constructor")
        return
    if not isinstance(receipt_sha256, str) or not receipt_sha256.startswith("sha256:"):
        raise AppealError("verified terminal evidence requires a qualified receipt SHA-256")
    try:
        require_sha256(receipt_sha256, "evidence_receipt_sha256")
    except RivalSchemaError as exc:
        raise AppealError(str(exc)) from exc


def _row_runtime_fingerprint(row: SelectionRow | HRow | LRow | HighOnlyHRow) -> str:
    payload = asdict(row)
    payload.pop("_validation_capability", None)
    return sha256_hex(
        {
            "schema_id": "cascadiav3.rival_evidence_row_runtime.v1",
            "row_type": type(row).__name__,
            "fields": payload,
        }
    )


def _row_artifact_kind(row: SelectionRow | HRow | LRow | HighOnlyHRow) -> str:
    return f"RivalEvidenceRow:{type(row).__name__}"


def _seal_evidence_row(
    row: SelectionRow | HRow | LRow | HighOnlyHRow,
) -> SelectionRow | HRow | LRow | HighOnlyHRow:
    return replace(
        row,
        _validation_capability=_issue_validation_capability(
            _row_artifact_kind(row),
            _row_runtime_fingerprint(row),
        ),
    )


def require_validated_evidence_row(
    row: SelectionRow | HRow | LRow | HighOnlyHRow,
) -> None:
    if not isinstance(row, (SelectionRow, HRow, LRow, HighOnlyHRow)):
        raise AppealError("appeal evidence must be a typed Rival evidence row")
    row.__post_init__()
    try:
        _require_validation_capability(
            row._validation_capability,
            artifact_kind=_row_artifact_kind(row),
            content_sha256=_row_runtime_fingerprint(row),
        )
    except RivalSchemaError as exc:
        raise AppealError(str(exc)) from exc


@dataclass(frozen=True)
class SelectionRow:
    """One S unit; ``challenger_id`` is a candidate occurrence, not free text."""

    unit_id: str
    challenger_id: str
    selection_score: float | None
    status: UnitStatus
    rng_key: str
    world_redetermination_seed_sha256s: tuple[str, ...]
    evidence_domain: EvidenceDomain
    evidence_receipt_sha256: str | None
    _validation_capability: object | None = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _validate_world_redetermination_seed_sha256s(self.world_redetermination_seed_sha256s)
        _validate_evidence_binding(self.evidence_domain, self.evidence_receipt_sha256)
        if self._validation_capability is not None:
            try:
                _require_validation_capability(
                    self._validation_capability,
                    artifact_kind=_row_artifact_kind(self),
                    content_sha256=_row_runtime_fingerprint(self),
                )
            except RivalSchemaError as exc:
                raise AppealError(str(exc)) from exc

    @classmethod
    def contract_test(
        cls,
        unit_id: str,
        challenger_id: str,
        selection_score: float | None,
        status: UnitStatus,
        rng_key: str,
    ) -> SelectionRow:
        row = cls(
            unit_id,
            challenger_id,
            selection_score,
            status,
            rng_key,
            _synthetic_world_redetermination_seed_sha256s(rng_key),
            EvidenceDomain.CONTRACT_TEST,
            None,
            None,
        )
        sealed = _seal_evidence_row(row)
        assert isinstance(sealed, SelectionRow)
        return sealed


@dataclass(frozen=True)
class HRow:
    unit_id: str
    challenger_id: str
    high_difference: float | None
    low_difference: float | None
    status: UnitStatus
    physical_coupling_key: str
    inner_rng_keys: tuple[str, ...]
    world_redetermination_seed_sha256s: tuple[str, ...]
    evidence_domain: EvidenceDomain
    evidence_receipt_sha256: str | None
    _validation_capability: object | None = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _validate_world_redetermination_seed_sha256s(self.world_redetermination_seed_sha256s)
        _validate_evidence_binding(self.evidence_domain, self.evidence_receipt_sha256)
        if self._validation_capability is not None:
            try:
                _require_validation_capability(
                    self._validation_capability,
                    artifact_kind=_row_artifact_kind(self),
                    content_sha256=_row_runtime_fingerprint(self),
                )
            except RivalSchemaError as exc:
                raise AppealError(str(exc)) from exc

    @classmethod
    def contract_test(
        cls,
        unit_id: str,
        challenger_id: str,
        high_difference: float | None,
        low_difference: float | None,
        status: UnitStatus,
        physical_coupling_key: str,
        inner_rng_keys: tuple[str, ...],
    ) -> HRow:
        row = cls(
            unit_id,
            challenger_id,
            high_difference,
            low_difference,
            status,
            physical_coupling_key,
            inner_rng_keys,
            _synthetic_world_redetermination_seed_sha256s(physical_coupling_key),
            EvidenceDomain.CONTRACT_TEST,
            None,
            None,
        )
        sealed = _seal_evidence_row(row)
        assert isinstance(sealed, HRow)
        return sealed


@dataclass(frozen=True)
class LRow:
    unit_id: str
    challenger_id: str
    low_difference: float | None
    status: UnitStatus
    physical_key: str
    inner_rng_keys: tuple[str, ...]
    world_redetermination_seed_sha256s: tuple[str, ...]
    evidence_domain: EvidenceDomain
    evidence_receipt_sha256: str | None
    _validation_capability: object | None = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _validate_world_redetermination_seed_sha256s(self.world_redetermination_seed_sha256s)
        _validate_evidence_binding(self.evidence_domain, self.evidence_receipt_sha256)
        if self._validation_capability is not None:
            try:
                _require_validation_capability(
                    self._validation_capability,
                    artifact_kind=_row_artifact_kind(self),
                    content_sha256=_row_runtime_fingerprint(self),
                )
            except RivalSchemaError as exc:
                raise AppealError(str(exc)) from exc

    @classmethod
    def contract_test(
        cls,
        unit_id: str,
        challenger_id: str,
        low_difference: float | None,
        status: UnitStatus,
        physical_key: str,
        inner_rng_keys: tuple[str, ...],
    ) -> LRow:
        row = cls(
            unit_id,
            challenger_id,
            low_difference,
            status,
            physical_key,
            inner_rng_keys,
            _synthetic_world_redetermination_seed_sha256s(physical_key),
            EvidenceDomain.CONTRACT_TEST,
            None,
            None,
        )
        sealed = _seal_evidence_row(row)
        assert isinstance(sealed, LRow)
        return sealed


@dataclass(frozen=True)
class HighOnlyHRow:
    """One high-fidelity action-pair outcome under the S/H control design."""

    unit_id: str
    challenger_id: str
    high_difference: float | None
    status: UnitStatus
    physical_key: str
    inner_rng_keys: tuple[str, ...]
    world_redetermination_seed_sha256s: tuple[str, ...]
    evidence_domain: EvidenceDomain
    evidence_receipt_sha256: str | None
    _validation_capability: object | None = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _validate_world_redetermination_seed_sha256s(self.world_redetermination_seed_sha256s)
        _validate_evidence_binding(self.evidence_domain, self.evidence_receipt_sha256)
        if self._validation_capability is not None:
            try:
                _require_validation_capability(
                    self._validation_capability,
                    artifact_kind=_row_artifact_kind(self),
                    content_sha256=_row_runtime_fingerprint(self),
                )
            except RivalSchemaError as exc:
                raise AppealError(str(exc)) from exc

    @classmethod
    def contract_test(
        cls,
        unit_id: str,
        challenger_id: str,
        high_difference: float | None,
        status: UnitStatus,
        physical_key: str,
        inner_rng_keys: tuple[str, ...],
    ) -> HighOnlyHRow:
        row = cls(
            unit_id,
            challenger_id,
            high_difference,
            status,
            physical_key,
            inner_rng_keys,
            _synthetic_world_redetermination_seed_sha256s(physical_key),
            EvidenceDomain.CONTRACT_TEST,
            None,
            None,
        )
        sealed = _seal_evidence_row(row)
        assert isinstance(sealed, HighOnlyHRow)
        return sealed


def _verified_selection_row(
    *,
    unit_id: str,
    challenger_id: str,
    selection_score: float,
    rng_key: str,
    world_redetermination_seed_sha256s: tuple[str, ...],
    evidence_domain: EvidenceDomain,
    receipt_sha256: str,
) -> SelectionRow:
    row = SelectionRow(
        unit_id,
        challenger_id,
        selection_score,
        UnitStatus.COMPLETE,
        rng_key,
        world_redetermination_seed_sha256s,
        evidence_domain,
        receipt_sha256,
        None,
    )
    sealed = _seal_evidence_row(row)
    assert isinstance(sealed, SelectionRow)
    return sealed


def _verified_h_row(
    *,
    unit_id: str,
    challenger_id: str,
    high_difference: float,
    low_difference: float,
    physical_coupling_key: str,
    inner_rng_keys: tuple[str, ...],
    world_redetermination_seed_sha256s: tuple[str, ...],
    evidence_domain: EvidenceDomain,
    receipt_bundle_sha256: str,
) -> HRow:
    row = HRow(
        unit_id,
        challenger_id,
        high_difference,
        low_difference,
        UnitStatus.COMPLETE,
        physical_coupling_key,
        inner_rng_keys,
        world_redetermination_seed_sha256s,
        evidence_domain,
        receipt_bundle_sha256,
        None,
    )
    sealed = _seal_evidence_row(row)
    assert isinstance(sealed, HRow)
    return sealed


def _verified_l_row(
    *,
    unit_id: str,
    challenger_id: str,
    low_difference: float,
    physical_key: str,
    inner_rng_keys: tuple[str, ...],
    world_redetermination_seed_sha256s: tuple[str, ...],
    evidence_domain: EvidenceDomain,
    receipt_sha256: str,
) -> LRow:
    row = LRow(
        unit_id,
        challenger_id,
        low_difference,
        UnitStatus.COMPLETE,
        physical_key,
        inner_rng_keys,
        world_redetermination_seed_sha256s,
        evidence_domain,
        receipt_sha256,
        None,
    )
    sealed = _seal_evidence_row(row)
    assert isinstance(sealed, LRow)
    return sealed


def _verified_high_only_h_row(
    *,
    unit_id: str,
    challenger_id: str,
    high_difference: float,
    physical_key: str,
    inner_rng_keys: tuple[str, ...],
    world_redetermination_seed_sha256s: tuple[str, ...],
    evidence_domain: EvidenceDomain,
    receipt_sha256: str,
) -> HighOnlyHRow:
    row = HighOnlyHRow(
        unit_id,
        challenger_id,
        high_difference,
        UnitStatus.COMPLETE,
        physical_key,
        inner_rng_keys,
        world_redetermination_seed_sha256s,
        evidence_domain,
        receipt_sha256,
        None,
    )
    sealed = _seal_evidence_row(row)
    assert isinstance(sealed, HighOnlyHRow)
    return sealed


@dataclass(frozen=True)
class OperationalAccounting:
    attempted_s: int
    attempted_h: int
    attempted_l: int
    completed_s: int
    completed_h: int
    completed_l: int
    timeouts: int
    invalid: int

    @property
    def attempted_total(self) -> int:
        return self.attempted_s + self.attempted_h + self.attempted_l

    @property
    def completed_total(self) -> int:
        return self.completed_s + self.completed_h + self.completed_l


@dataclass(frozen=True)
class CategoricalPreference:
    incumbent_action_id: str
    challenger_action_id: str
    preference: Literal["challenger_over_incumbent"]
    preference_valid: Literal[True]
    preference_weight: float


@dataclass(frozen=True)
class AppealDecision:
    status: Literal["confirmed", "not_confirmed", "no_label"]
    reason: str
    preference: CategoricalPreference | None
    estimate_audit: MultifidelityEstimate | None
    bound_audit: HoeffdingBound | None
    operational: OperationalAccounting
    evidence_domain: EvidenceDomain | None
    scientific_evidence: bool


@dataclass(frozen=True)
class HighFidelityAppealDecision:
    """Decision from the distinct beta-zero S/H-only control experiment."""

    status: Literal["confirmed", "not_confirmed", "no_label"]
    reason: str
    preference: CategoricalPreference | None
    estimate_audit: HighFidelityEstimate | None
    bound_audit: HighOnlyHoeffdingBound | None
    operational: OperationalAccounting
    evidence_domain: EvidenceDomain | None
    scientific_evidence: bool


@dataclass(frozen=True)
class VerifiedMultifidelityDesign:
    """Identity-bound inputs admitted by the one-look appeal state machine."""

    root_id: str
    incumbent_action_id: str
    incumbent_candidate_occurrence_id: str
    candidate_selection_entries: tuple[CandidateSelectionEntry, ...]
    expected_s: int
    expected_h: int
    expected_l: int
    beta_cv: float
    high_range: CertifiedRange
    low_range: CertifiedRange
    widths: TransformedWidths
    error_allocation: RootErrorAllocation
    practical_margin: float
    preference_weight: float
    selection_rule: str
    manifest_content_sha256: str
    deployment_design_sha256: str
    coefficient_id: str
    bound_certificate_sha256: str
    error_family_id: str
    _validation_capability: object | None = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._validation_capability is not None:
            try:
                _require_validation_capability(
                    self._validation_capability,
                    artifact_kind=_design_artifact_kind(self),
                    content_sha256=_design_runtime_fingerprint(self),
                )
            except RivalSchemaError as exc:
                raise AppealError(str(exc)) from exc


@dataclass(frozen=True)
class VerifiedHighFidelityDesign:
    """Identity-bound inputs for the non-multifidelity S/H control path."""

    root_id: str
    incumbent_action_id: str
    incumbent_candidate_occurrence_id: str
    candidate_selection_entries: tuple[CandidateSelectionEntry, ...]
    expected_s: int
    expected_h: int
    high_range: CertifiedRange
    error_allocation: HighOnlyErrorAllocation
    practical_margin: float
    preference_weight: float
    selection_rule: str
    manifest_content_sha256: str
    deployment_design_sha256: str
    bound_certificate_sha256: str
    error_family_id: str
    _validation_capability: object | None = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._validation_capability is not None:
            try:
                _require_validation_capability(
                    self._validation_capability,
                    artifact_kind=_design_artifact_kind(self),
                    content_sha256=_design_runtime_fingerprint(self),
                )
            except RivalSchemaError as exc:
                raise AppealError(str(exc)) from exc


def _design_artifact_kind(
    design: VerifiedMultifidelityDesign | VerifiedHighFidelityDesign,
) -> str:
    return f"RivalAppealDesign:{type(design).__name__}"


def _design_runtime_fingerprint(
    design: VerifiedMultifidelityDesign | VerifiedHighFidelityDesign,
) -> str:
    payload = asdict(design)
    payload.pop("_validation_capability", None)
    return sha256_hex(
        {
            "schema_id": "cascadiav3.rival_appeal_design_runtime.v1",
            "design_type": type(design).__name__,
            "fields": payload,
        }
    )


def _seal_appeal_design(
    design: VerifiedMultifidelityDesign | VerifiedHighFidelityDesign,
) -> VerifiedMultifidelityDesign | VerifiedHighFidelityDesign:
    return replace(
        design,
        _validation_capability=_issue_validation_capability(
            _design_artifact_kind(design),
            _design_runtime_fingerprint(design),
        ),
    )


def require_validated_appeal_design(
    design: VerifiedMultifidelityDesign | VerifiedHighFidelityDesign,
) -> None:
    if not isinstance(design, (VerifiedMultifidelityDesign, VerifiedHighFidelityDesign)):
        raise AppealError("appeal state machine requires a typed verified design")
    design.__post_init__()
    try:
        _require_validation_capability(
            design._validation_capability,
            artifact_kind=_design_artifact_kind(design),
            content_sha256=_design_runtime_fingerprint(design),
        )
    except RivalSchemaError as exc:
        raise AppealError(str(exc)) from exc


def _validate_common_binding_inputs(
    *,
    manifest: RootManifest,
    bound_certificate: BoundCertificate,
    error_family: ErrorFamilyLedger,
) -> None:
    if not isinstance(error_family, ErrorFamilyLedger):
        raise AppealError("error family must be an ErrorFamilyLedger")
    try:
        require_validated_root_manifest(manifest)
        require_validated_bound_certificate(bound_certificate)
        error_family.require_validated_artifact()
    except ValueError as exc:
        raise AppealError(f"unvalidated scientific design input: {exc}") from exc
    _identity(manifest.incumbent_action_id, "incumbent_action_id")
    expected_ruleset_identity = "sha256:" + sha256_hex(bound_certificate.ruleset)
    if manifest.ruleset_identity != expected_ruleset_identity:
        raise AppealError("root manifest and bound certificate ruleset identities differ")
    expected_bound_identity = "sha256:" + bound_certificate.content_sha256
    if manifest.bound_certificate_identity != expected_bound_identity:
        raise AppealError("root manifest does not pin the supplied bound certificate")
    if manifest.error_ledger_identity != error_family.identity:
        raise AppealError("root manifest does not pin the supplied error family")
    error_family.validate()
    if manifest.selection_rule != "highest_mean_then_lexicographic_action_id":
        raise AppealError("unsupported or adaptive selection rule")


def bind_high_fidelity_design(
    *,
    manifest: RootManifest,
    bound_certificate: BoundCertificate,
    error_family: ErrorFamilyLedger,
) -> VerifiedHighFidelityDesign:
    """Bind the separate beta-zero, S/H-only control experiment."""
    _validate_common_binding_inputs(
        manifest=manifest,
        bound_certificate=bound_certificate,
        error_family=error_family,
    )
    if (
        manifest.inference_mode != "high_fidelity_only"
        or manifest.multifidelity_claim
        or manifest.beta_cv != 0.0
        or manifest.expected_l != 0
        or manifest.low_policy_identity is not None
    ):
        raise AppealError("high-fidelity appeal requires the separate S/H-only manifest")
    matching = [row for row in error_family.roots if row.root_id == manifest.root_id]
    if len(matching) != 1 or not isinstance(matching[0], HighOnlyRootErrorEntry):
        raise AppealError("error family lacks exactly one high-only budget for this root")
    error_entry = matching[0]
    if not error_entry.potentially_eligible:
        raise AppealError("appeal root was not included in the potential-error budget")
    if manifest.allocation_identity != error_entry.allocation_identity:
        raise AppealError("root manifest does not pin its high-only error allocation")
    design = VerifiedHighFidelityDesign(
        root_id=manifest.root_id,
        incumbent_action_id=manifest.incumbent_action_id,
        incumbent_candidate_occurrence_id=manifest.incumbent_candidate_occurrence_id,
        candidate_selection_entries=manifest.candidate_selection_entries,
        expected_s=manifest.expected_s,
        expected_h=manifest.expected_h,
        high_range=bound_certificate.high,
        error_allocation=error_entry.allocation(),
        practical_margin=manifest.practical_margin,
        preference_weight=manifest.preference_weight,
        selection_rule=manifest.selection_rule,
        manifest_content_sha256=manifest.content_sha256,
        deployment_design_sha256=manifest.deployment_design_sha256,
        bound_certificate_sha256=bound_certificate.content_sha256,
        error_family_id=error_family.identity,
        _validation_capability=None,
    )
    sealed = _seal_appeal_design(design)
    assert isinstance(sealed, VerifiedHighFidelityDesign)
    return sealed


def bind_multifidelity_design(
    *,
    manifest: RootManifest,
    coefficient: CoefficientBinding,
    bound_certificate: BoundCertificate,
    error_family: ErrorFamilyLedger,
) -> VerifiedMultifidelityDesign:
    """Bind every numeric input to verified manifest/certificate identities."""
    if not isinstance(manifest, RootManifest):
        raise AppealError("manifest must be a RootManifest")
    if not isinstance(coefficient, CoefficientBinding):
        raise AppealError("coefficient must be a CoefficientBinding")
    if manifest.inference_mode != "multifidelity" or not manifest.multifidelity_claim:
        raise AppealError("multifidelity appeal requires a Rival-MF root manifest")
    try:
        coefficient.require_validated_artifact()
    except ValueError as exc:
        raise AppealError(f"unvalidated scientific coefficient: {exc}") from exc
    _validate_common_binding_inputs(
        manifest=manifest,
        bound_certificate=bound_certificate,
        error_family=error_family,
    )
    if manifest.coefficient_identity != coefficient.identity:
        raise AppealError("root manifest does not pin the supplied coefficient")
    if manifest.manifest_id != coefficient.deployment_design_id:
        raise AppealError("coefficient is not calibrated for this deployment design")
    coefficient.require_design(
        deployment_design_id=manifest.manifest_id,
        deployment_design_sha256=manifest.deployment_design_sha256,
        incumbent_policy_id=manifest.incumbent_policy_identity,
        low_policy_id=manifest.low_policy_identity or "",
        sampler_id=manifest.sampler_identity,
        allocation_id=manifest.allocation_identity,
    )
    if (
        coefficient.low_expectation_h_id != manifest.low_expectation_id
        or coefficient.low_expectation_l_id != manifest.low_expectation_id
        or coefficient.low_law_h_id != manifest.low_law_h_id
        or coefficient.low_law_l_id != manifest.low_law_l_id
        or coefficient.max_abs_beta != manifest.max_abs_beta
    ):
        raise AppealError("coefficient is not bound to the manifest low-law design")
    if not math.isclose(manifest.beta_cv, coefficient.beta_cv, rel_tol=0.0, abs_tol=0.0):
        raise AppealError("manifest beta_cv differs from the frozen coefficient")
    matching = [row for row in error_family.roots if row.root_id == manifest.root_id]
    if len(matching) != 1 or not isinstance(matching[0], RootErrorEntry):
        raise AppealError("error family lacks exactly one multifidelity budget for this root")
    error_entry = matching[0]
    if not error_entry.potentially_eligible:
        raise AppealError("appeal root was not included in the potential-error budget")
    if manifest.allocation_identity != error_entry.allocation_identity:
        raise AppealError("root manifest does not pin its multifidelity error allocation")
    design = VerifiedMultifidelityDesign(
        root_id=manifest.root_id,
        incumbent_action_id=manifest.incumbent_action_id,
        incumbent_candidate_occurrence_id=manifest.incumbent_candidate_occurrence_id,
        candidate_selection_entries=manifest.candidate_selection_entries,
        expected_s=manifest.expected_s,
        expected_h=manifest.expected_h,
        expected_l=manifest.expected_l,
        beta_cv=coefficient.beta_cv,
        high_range=bound_certificate.high,
        low_range=bound_certificate.low,
        widths=transformed_widths_from_certificate(bound_certificate, beta_cv=coefficient.beta_cv),
        error_allocation=error_entry.allocation(),
        practical_margin=manifest.practical_margin,
        preference_weight=manifest.preference_weight,
        selection_rule=manifest.selection_rule,
        manifest_content_sha256=manifest.content_sha256,
        deployment_design_sha256=manifest.deployment_design_sha256,
        coefficient_id=coefficient.coefficient_id,
        bound_certificate_sha256=bound_certificate.content_sha256,
        error_family_id=error_family.identity,
        _validation_capability=None,
    )
    sealed = _seal_appeal_design(design)
    assert isinstance(sealed, VerifiedMultifidelityDesign)
    return sealed


class AppealStateMachine:
    """Enforce frozen selection, complete panels, and exactly one inference look."""

    def __init__(
        self,
        *,
        design: VerifiedMultifidelityDesign,
    ) -> None:
        if not isinstance(design, VerifiedMultifidelityDesign):
            raise AppealError("appeal state machine requires a verified design")
        require_validated_appeal_design(design)
        root_id = design.root_id
        incumbent_action_id = design.incumbent_action_id
        _identity(root_id, "root_id")
        _identity(incumbent_action_id, "incumbent_action_id")
        expected_s = design.expected_s
        expected_h = design.expected_h
        expected_l = design.expected_l
        beta_cv = design.beta_cv
        widths = design.widths
        error_allocation = design.error_allocation
        practical_margin = design.practical_margin
        preference_weight = design.preference_weight
        selection_rule = design.selection_rule
        for value, name in (
            (expected_s, "expected_s"),
            (expected_h, "expected_h"),
            (expected_l, "expected_l"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise AppealError(f"{name} must be a positive integer")
        for value, name in (
            (beta_cv, "beta_cv"),
            (practical_margin, "practical_margin"),
            (preference_weight, "preference_weight"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise AppealError(f"{name} must be finite")
        if practical_margin < 0.0:
            raise AppealError("practical_margin must be non-negative")
        if preference_weight <= 0.0:
            raise AppealError("preference_weight must be positive")
        if selection_rule != "highest_mean_then_lexicographic_action_id":
            raise AppealError("unsupported or adaptive selection rule")
        self.root_id = root_id
        self.incumbent_action_id = incumbent_action_id
        self.incumbent_candidate_occurrence_id = design.incumbent_candidate_occurrence_id
        self.candidate_selection_entries = design.candidate_selection_entries
        self._candidate_by_occurrence = {
            row.candidate_action_occurrence_id: row for row in self.candidate_selection_entries
        }
        self._eligible_allocations = {
            row.candidate_action_occurrence_id: row.expected_s
            for row in self.candidate_selection_entries
            if row.candidate_action_occurrence_id != self.incumbent_candidate_occurrence_id
        }
        if (
            len(self._candidate_by_occurrence) != len(self.candidate_selection_entries)
            or sum(self._eligible_allocations.values()) != expected_s
            or not self._eligible_allocations
        ):
            raise AppealError("verified design has an inconsistent candidate/S allocation")
        self.expected_s = expected_s
        self.expected_h = expected_h
        self.expected_l = expected_l
        self.beta_cv = float(beta_cv)
        self.high_range = design.high_range
        self.low_range = design.low_range
        self.widths = widths
        self.error_allocation = error_allocation
        self.practical_margin = float(practical_margin)
        self.preference_weight = float(preference_weight)
        self.selection_rule = selection_rule
        self.design = design
        self._s: list[SelectionRow] = []
        self._h: list[HRow] = []
        self._l: list[LRow] = []
        self._selected: str | None = None
        self._seen_units: set[str] = set()
        self._seen_inner_keys: set[str] = set()
        self._seen_world_redetermination_seeds: set[str] = set()
        self._evidence_domain: EvidenceDomain | None = None
        self._seen_evidence_receipts: set[str] = set()
        self._finalized = False

    def _validate_evidence_row(
        self,
        domain: EvidenceDomain,
        receipt_sha256: str | None,
        world_redetermination_seed_sha256s: tuple[str, ...],
    ) -> None:
        if domain is EvidenceDomain.PRODUCTION_TERMINAL:
            raise AppealError(
                "production-terminal evidence is structurally unavailable before "
                "a production Rust adapter is admitted"
            )
        if self._evidence_domain is not None and domain is not self._evidence_domain:
            raise AppealError("one appeal cannot mix evidence domains")
        if receipt_sha256 is not None and receipt_sha256 in self._seen_evidence_receipts:
            raise AppealError("verified terminal receipt reused across panel rows")
        _validate_world_redetermination_seed_sha256s(world_redetermination_seed_sha256s)
        overlap = self._seen_world_redetermination_seeds & set(world_redetermination_seed_sha256s)
        if overlap:
            raise AppealError(
                "duplicate world-redetermination seed commitment reused across panel rows: "
                + ", ".join(sorted(overlap))
            )

    def _commit_evidence_row(
        self,
        domain: EvidenceDomain,
        receipt_sha256: str | None,
        world_redetermination_seed_sha256s: tuple[str, ...],
    ) -> None:
        if self._evidence_domain is None:
            self._evidence_domain = domain
        if receipt_sha256 is not None:
            self._seen_evidence_receipts.add(receipt_sha256)
        self._seen_world_redetermination_seeds.update(world_redetermination_seed_sha256s)

    @staticmethod
    def _validate_status_payload(status: UnitStatus, *values: float | None) -> None:
        if not isinstance(status, UnitStatus):
            raise AppealError("unit status must be a UnitStatus value")
        if status is UnitStatus.COMPLETE:
            if any(
                value is None
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in values
            ):
                raise AppealError("complete rows require every finite outcome")
        elif any(value is not None for value in values):
            raise AppealError("timeout/invalid rows cannot carry inferential outcomes")

    def _validate_unclaimed_unit(self, unit_id: str) -> None:
        _identity(unit_id, "unit_id")
        if unit_id in self._seen_units:
            raise AppealError(f"duplicate or cross-panel unit_id: {unit_id}")

    def _validate_unclaimed_inner_keys(self, keys: tuple[str, ...]) -> None:
        if not isinstance(keys, tuple) or not keys:
            raise AppealError("inner RNG keys must be a non-empty tuple")
        if any(not isinstance(key, str) or not key or key != key.strip() for key in keys):
            raise AppealError("each confirmation row needs non-empty inner RNG keys")
        if len(set(keys)) != len(keys):
            raise AppealError("inner RNG key reused within one row")
        overlap = self._seen_inner_keys & set(keys)
        if overlap:
            raise AppealError(f"inner RNG key reused across rows: {sorted(overlap)}")

    def add_selection(self, row: SelectionRow) -> None:
        if not isinstance(row, SelectionRow):
            raise AppealError("selection panel requires SelectionRow")
        require_validated_evidence_row(row)
        if self._selected is not None or self._h or self._l:
            raise AppealError("selection panel is frozen")
        if len(self._s) >= self.expected_s:
            raise AppealError("selection panel exceeds preregistered count")
        self._validate_evidence_row(
            row.evidence_domain,
            row.evidence_receipt_sha256,
            row.world_redetermination_seed_sha256s,
        )
        self._validate_status_payload(row.status, row.selection_score)
        self._validate_unclaimed_unit(row.unit_id)
        _identity(row.challenger_id, "selection challenger_id")
        expected_for_candidate = self._eligible_allocations.get(row.challenger_id)
        if expected_for_candidate is None:
            raise AppealError("selection row is not an eligible registered challenger occurrence")
        observed_for_candidate = sum(
            existing.challenger_id == row.challenger_id for existing in self._s
        )
        if observed_for_candidate >= expected_for_candidate:
            raise AppealError("selection row exceeds the challenger-specific S allocation")
        _identity(row.rng_key, "selection rng_key")
        if any(existing.rng_key == row.rng_key for existing in self._s):
            raise AppealError(f"selection RNG key reused: {row.rng_key}")
        self._seen_units.add(row.unit_id)
        self._commit_evidence_row(
            row.evidence_domain,
            row.evidence_receipt_sha256,
            row.world_redetermination_seed_sha256s,
        )
        self._s.append(row)

    def freeze_challenger(self, challenger_id: str) -> None:
        _identity(challenger_id, "challenger_id")
        if self._selected is not None:
            raise AppealError("exactly one challenger may be frozen")
        if len(self._s) != self.expected_s:
            raise AppealError("cannot freeze before the complete fixed S panel exists")
        if any(row.status is not UnitStatus.COMPLETE for row in self._s):
            raise AppealError("cannot select from a failed S panel")
        observed_allocations = {
            candidate: sum(row.challenger_id == candidate for row in self._s)
            for candidate in self._eligible_allocations
        }
        if observed_allocations != self._eligible_allocations:
            raise AppealError("S panel does not match the registered per-candidate allocation")
        candidates = {row.challenger_id for row in self._s}
        if challenger_id not in candidates:
            raise AppealError("frozen challenger was not evaluated in S")
        if challenger_id == self.incumbent_candidate_occurrence_id:
            raise AppealError("challenger occurrence must differ from incumbent")
        scores_by_candidate = {
            candidate: fmean(
                float(row.selection_score) for row in self._s if row.challenger_id == candidate
            )
            for candidate in candidates
        }
        expected = min(
            candidates,
            key=lambda candidate: (
                -scores_by_candidate[candidate],
                self._candidate_by_occurrence[candidate].action_content_id,
            ),
        )
        if challenger_id != expected:
            raise AppealError(
                "frozen challenger violates registered S winner rule: "
                f"{challenger_id!r} != {expected!r}"
            )
        self._selected = challenger_id

    def add_h(self, row: HRow) -> None:
        if not isinstance(row, HRow):
            raise AppealError("multifidelity H panel requires HRow")
        require_validated_evidence_row(row)
        self._require_frozen_challenger(row.challenger_id)
        if self._l:
            # Ordering H before L is a simple way to make accidental access
            # patterns inspectable.  It does not imply any inferential peek.
            raise AppealError("H panel must be recorded before L panel begins")
        if len(self._h) >= self.expected_h:
            raise AppealError("H panel exceeds preregistered count")
        self._validate_evidence_row(
            row.evidence_domain,
            row.evidence_receipt_sha256,
            row.world_redetermination_seed_sha256s,
        )
        self._validate_status_payload(row.status, row.high_difference, row.low_difference)
        if row.status is UnitStatus.COMPLETE:
            assert row.high_difference is not None and row.low_difference is not None
            self._require_in_certified_range(
                row.high_difference, self.high_range, "H high difference"
            )
            self._require_in_certified_range(row.low_difference, self.low_range, "H low difference")
        self._validate_unclaimed_unit(row.unit_id)
        _identity(row.physical_coupling_key, "H physical_coupling_key")
        if any(existing.physical_coupling_key == row.physical_coupling_key for existing in self._h):
            raise AppealError("H physical coupling key reused across units")
        self._validate_unclaimed_inner_keys(row.inner_rng_keys)
        self._seen_units.add(row.unit_id)
        self._seen_inner_keys.update(row.inner_rng_keys)
        self._commit_evidence_row(
            row.evidence_domain,
            row.evidence_receipt_sha256,
            row.world_redetermination_seed_sha256s,
        )
        self._h.append(row)

    def add_l(self, row: LRow) -> None:
        if not isinstance(row, LRow):
            raise AppealError("multifidelity L panel requires LRow")
        require_validated_evidence_row(row)
        self._require_frozen_challenger(row.challenger_id)
        if len(self._h) != self.expected_h:
            raise AppealError("L panel cannot begin before all H attempts exist")
        if len(self._l) >= self.expected_l:
            raise AppealError("L panel exceeds preregistered count")
        self._validate_evidence_row(
            row.evidence_domain,
            row.evidence_receipt_sha256,
            row.world_redetermination_seed_sha256s,
        )
        self._validate_status_payload(row.status, row.low_difference)
        if row.status is UnitStatus.COMPLETE:
            assert row.low_difference is not None
            self._require_in_certified_range(row.low_difference, self.low_range, "L low difference")
        self._validate_unclaimed_unit(row.unit_id)
        _identity(row.physical_key, "L physical_key")
        h_keys = {existing.physical_coupling_key for existing in self._h}
        if row.physical_key in h_keys or any(
            existing.physical_key == row.physical_key for existing in self._l
        ):
            raise AppealError("physical key overlaps H/L independent panels")
        self._validate_unclaimed_inner_keys(row.inner_rng_keys)
        self._seen_units.add(row.unit_id)
        self._seen_inner_keys.update(row.inner_rng_keys)
        self._commit_evidence_row(
            row.evidence_domain,
            row.evidence_receipt_sha256,
            row.world_redetermination_seed_sha256s,
        )
        self._l.append(row)

    def _require_frozen_challenger(self, challenger_id: str) -> None:
        _identity(challenger_id, "confirmation challenger_id")
        if self._selected is None:
            raise AppealError("H/L access is forbidden before S freezes one challenger")
        if challenger_id != self._selected:
            raise AppealError("confirmation row does not use the one frozen challenger")

    @staticmethod
    def _require_in_certified_range(value: float, certified: CertifiedRange, name: str) -> None:
        if value < certified.minimum or value > certified.maximum:
            raise AppealError(
                f"{name} {value} is outside certified range "
                f"[{certified.minimum}, {certified.maximum}]"
            )

    def operational_accounting(self) -> OperationalAccounting:
        statuses = [row.status for row in (*self._s, *self._h, *self._l)]
        return OperationalAccounting(
            attempted_s=len(self._s),
            attempted_h=len(self._h),
            attempted_l=len(self._l),
            completed_s=sum(row.status is UnitStatus.COMPLETE for row in self._s),
            completed_h=sum(row.status is UnitStatus.COMPLETE for row in self._h),
            completed_l=sum(row.status is UnitStatus.COMPLETE for row in self._l),
            timeouts=sum(status is UnitStatus.TIMEOUT for status in statuses),
            invalid=sum(status is UnitStatus.INVALID for status in statuses),
        )

    def finalize(self) -> AppealDecision:
        """Consume the single planned look; a second call is an invalid peek."""
        if self._finalized:
            raise AppealError(
                "inferential result has already been consumed; repeated peek rejected"
            )
        if self._evidence_domain is EvidenceDomain.PRODUCTION_TERMINAL:
            raise AppealError(
                "production-terminal inference is structurally unavailable before "
                "a production Rust adapter is admitted"
            )
        if self._selected is None:
            if len(self._s) != self.expected_s:
                raise AppealError("cannot finalize before every fixed S attempt exists")
            if all(row.status is UnitStatus.COMPLETE for row in self._s):
                raise AppealError("complete S panel requires its deterministic winner to be frozen")
            self._finalized = True
            return AppealDecision(
                status="no_label",
                reason="failed S unit; no challenger selected and no confirmation panel opened",
                preference=None,
                estimate_audit=None,
                bound_audit=None,
                operational=self.operational_accounting(),
                evidence_domain=self._evidence_domain,
                scientific_evidence=False,
            )
        if len(self._h) != self.expected_h or len(self._l) != self.expected_l:
            raise AppealError("cannot inspect inference before complete fixed H/L attempts exist")
        self._finalized = True
        operational = self.operational_accounting()
        if operational.completed_h != self.expected_h or operational.completed_l != self.expected_l:
            return AppealDecision(
                status="no_label",
                reason="failed H/L unit; failure retained only in operational accounting",
                preference=None,
                estimate_audit=None,
                bound_audit=None,
                operational=operational,
                evidence_domain=self._evidence_domain,
                scientific_evidence=False,
            )

        paired = tuple(
            PairedDifference(
                row.unit_id,
                float(row.high_difference),
                float(row.low_difference),
            )
            for row in self._h
        )
        extra = tuple(LowDifference(row.unit_id, float(row.low_difference)) for row in self._l)
        estimate = estimate_fixed_panels(
            paired,
            extra,
            beta_cv=self.beta_cv,
            expected_n_h=self.expected_h,
            expected_n_l=self.expected_l,
        )
        bound = fixed_hoeffding_lower_bound(
            high_corrected_mean=estimate.high_corrected_mean,
            low_correction_mean=estimate.low_correction_mean,
            widths=self.widths,
            allocation=self.error_allocation,
            n_h=self.expected_h,
            n_l=self.expected_l,
        )
        if bound.lower_bound <= self.practical_margin:
            return AppealDecision(
                status="not_confirmed",
                reason="fixed lower bound did not clear the preregistered practical margin",
                preference=None,
                estimate_audit=estimate,
                bound_audit=bound,
                operational=operational,
                evidence_domain=self._evidence_domain,
                scientific_evidence=False,
            )
        return AppealDecision(
            status="no_label",
            reason=(
                "bound cleared only in a synthetic/proxy evidence domain; "
                "scientific preference emission is forbidden"
            ),
            preference=None,
            estimate_audit=estimate,
            bound_audit=bound,
            operational=operational,
            evidence_domain=self._evidence_domain,
            scientific_evidence=False,
        )


class HighFidelityAppealStateMachine(AppealStateMachine):
    """Separate fixed-sample S/H control with no low panel or MF claim.

    Selection and identity/key accounting deliberately reuse the audited S/H/L
    protocol.  The constructor accepts only a high-fidelity verified design,
    ``add_l`` is structurally rejected, and the inference result has a distinct
    type so it cannot be mislabeled as multifidelity evidence.
    """

    def __init__(self, *, design: VerifiedHighFidelityDesign) -> None:
        if not isinstance(design, VerifiedHighFidelityDesign):
            raise AppealError("high-fidelity state machine requires a verified design")
        require_validated_appeal_design(design)
        self.root_id = _identity(design.root_id, "root_id")
        self.incumbent_action_id = _identity(design.incumbent_action_id, "incumbent_action_id")
        self.incumbent_candidate_occurrence_id = design.incumbent_candidate_occurrence_id
        self.candidate_selection_entries = design.candidate_selection_entries
        self._candidate_by_occurrence = {
            row.candidate_action_occurrence_id: row for row in self.candidate_selection_entries
        }
        self._eligible_allocations = {
            row.candidate_action_occurrence_id: row.expected_s
            for row in self.candidate_selection_entries
            if row.candidate_action_occurrence_id != self.incumbent_candidate_occurrence_id
        }
        self.expected_s = design.expected_s
        if (
            len(self._candidate_by_occurrence) != len(self.candidate_selection_entries)
            or sum(self._eligible_allocations.values()) != self.expected_s
            or not self._eligible_allocations
        ):
            raise AppealError("verified design has an inconsistent candidate/S allocation")
        self.expected_h = design.expected_h
        self.expected_l = 0
        self.high_range = design.high_range
        self.error_allocation = design.error_allocation
        self.practical_margin = design.practical_margin
        self.preference_weight = design.preference_weight
        self.selection_rule = design.selection_rule
        self.design = design
        self._s: list[SelectionRow] = []
        self._h: list[HighOnlyHRow] = []
        self._l: list[LRow] = []
        self._selected: str | None = None
        self._seen_units: set[str] = set()
        self._seen_inner_keys: set[str] = set()
        self._seen_world_redetermination_seeds: set[str] = set()
        self._evidence_domain: EvidenceDomain | None = None
        self._seen_evidence_receipts: set[str] = set()
        self._finalized = False

    def add_h(self, row: HighOnlyHRow) -> None:  # type: ignore[override]
        if not isinstance(row, HighOnlyHRow):
            raise AppealError("high-fidelity H panel requires HighOnlyHRow")
        require_validated_evidence_row(row)
        self._require_frozen_challenger(row.challenger_id)
        if len(self._h) >= self.expected_h:
            raise AppealError("H panel exceeds preregistered count")
        self._validate_evidence_row(
            row.evidence_domain,
            row.evidence_receipt_sha256,
            row.world_redetermination_seed_sha256s,
        )
        self._validate_status_payload(row.status, row.high_difference)
        if row.status is UnitStatus.COMPLETE:
            assert row.high_difference is not None
            self._require_in_certified_range(
                row.high_difference, self.high_range, "H high difference"
            )
        self._validate_unclaimed_unit(row.unit_id)
        _identity(row.physical_key, "H physical_key")
        if any(existing.physical_key == row.physical_key for existing in self._h):
            raise AppealError("H physical key reused across units")
        self._validate_unclaimed_inner_keys(row.inner_rng_keys)
        self._seen_units.add(row.unit_id)
        self._seen_inner_keys.update(row.inner_rng_keys)
        self._commit_evidence_row(
            row.evidence_domain,
            row.evidence_receipt_sha256,
            row.world_redetermination_seed_sha256s,
        )
        self._h.append(row)

    def add_l(self, row: LRow) -> None:
        del row
        raise AppealError("L panel is structurally forbidden in high-fidelity-only mode")

    def finalize(self) -> HighFidelityAppealDecision:  # type: ignore[override]
        """Consume the one fixed H-only look and emit only categorical labels."""
        if self._finalized:
            raise AppealError(
                "inferential result has already been consumed; repeated peek rejected"
            )
        if self._evidence_domain is EvidenceDomain.PRODUCTION_TERMINAL:
            raise AppealError(
                "production-terminal inference is structurally unavailable before "
                "a production Rust adapter is admitted"
            )
        if self._selected is None:
            if len(self._s) != self.expected_s:
                raise AppealError("cannot finalize before every fixed S attempt exists")
            if all(row.status is UnitStatus.COMPLETE for row in self._s):
                raise AppealError("complete S panel requires its deterministic winner to be frozen")
            self._finalized = True
            return HighFidelityAppealDecision(
                status="no_label",
                reason="failed S unit; no challenger selected and no H panel opened",
                preference=None,
                estimate_audit=None,
                bound_audit=None,
                operational=self.operational_accounting(),
                evidence_domain=self._evidence_domain,
                scientific_evidence=False,
            )
        if len(self._h) != self.expected_h:
            raise AppealError("cannot inspect inference before complete fixed H attempts exist")
        self._finalized = True
        operational = self.operational_accounting()
        if operational.completed_h != self.expected_h:
            return HighFidelityAppealDecision(
                status="no_label",
                reason="failed H unit; failure retained only in operational accounting",
                preference=None,
                estimate_audit=None,
                bound_audit=None,
                operational=operational,
                evidence_domain=self._evidence_domain,
                scientific_evidence=False,
            )
        estimate = estimate_high_fidelity_only(
            tuple(HighDifference(row.unit_id, float(row.high_difference)) for row in self._h),
            expected_n_h=self.expected_h,
        )
        bound = fixed_high_only_hoeffding_lower_bound(
            high_mean=estimate.estimate,
            certified_width=self.high_range.width,
            allocation=self.error_allocation,
            n_h=self.expected_h,
        )
        if bound.lower_bound <= self.practical_margin:
            return HighFidelityAppealDecision(
                status="not_confirmed",
                reason="fixed high-only lower bound did not clear the practical margin",
                preference=None,
                estimate_audit=estimate,
                bound_audit=bound,
                operational=operational,
                evidence_domain=self._evidence_domain,
                scientific_evidence=False,
            )
        return HighFidelityAppealDecision(
            status="no_label",
            reason=(
                "bound cleared only in a synthetic/proxy evidence domain; "
                "scientific preference emission is forbidden"
            ),
            preference=None,
            estimate_audit=estimate,
            bound_audit=bound,
            operational=operational,
            evidence_domain=self._evidence_domain,
            scientific_evidence=False,
        )
