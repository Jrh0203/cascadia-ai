"""Error ledgers and exact/synthetic coverage checks for Rival v1 bounds."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from dataclasses import field as dataclass_field
from itertools import product
from pathlib import Path
from typing import Any, Literal

from .bounds import (
    HighOnlyErrorAllocation,
    RootErrorAllocation,
    fixed_hoeffding_lower_bound,
    transformed_widths,
)
from .cohorts import AllocationRegistry, CohortError
from .manifest import PUBLIC_ROOT_ID_PREFIX
from .multifidelity import LowDifference, PairedDifference, estimate_fixed_panels
from .schema import (
    RIVAL_ERROR_FAMILY_LEDGER_SCHEMA_ID,
    RIVAL_POTENTIAL_ROOT_CENSUS_SCHEMA_ID,
    RivalSchemaError,
    _issue_validation_capability,
    _require_validation_capability,
    attach_content_hash,
    read_pinned_canonical_json_object,
    read_strict_json_object,
    require_exact_keys,
    require_finite,
    require_nonempty_string,
    require_positive_int,
    require_schema,
    require_sha256,
    sha256_hex,
    verify_content_hash,
    write_new_canonical_json,
)


class CoverageError(ValueError):
    """Raised when multiplicity or a coverage design is invalid."""


ErrorFamilyKind = Literal["finite_training_corpus", "one_seat_instrument"]


@dataclass(frozen=True)
class PotentialRootCensus:
    """Complete preregistered root universe for one multiplicity family."""

    census_id: str
    family_kind: ErrorFamilyKind
    source_root_set_sha256: str
    allocation_registry_identity: str
    eligible_root_ids: tuple[str, ...]
    content_sha256: str
    _validation_capability: object | None = dataclass_field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self._validation_capability is not None:
            try:
                _require_validation_capability(
                    self._validation_capability,
                    artifact_kind="PotentialRootCensus",
                    content_sha256=_census_runtime_fingerprint(self),
                )
            except RivalSchemaError as exc:
                raise CoverageError(str(exc)) from exc

    @property
    def identity(self) -> str:
        return "sha256:" + self.content_sha256

    def require_validated_artifact(self) -> None:
        try:
            _require_validation_capability(
                self._validation_capability,
                artifact_kind="PotentialRootCensus",
                content_sha256=_census_runtime_fingerprint(self),
            )
        except RivalSchemaError as exc:
            raise CoverageError(str(exc)) from exc


_CENSUS_FIELDS = (
    "schema_id",
    "census_id",
    "family_kind",
    "source_root_set_sha256",
    "allocation_registry_identity",
    "eligible_root_ids",
    "content_sha256",
)


def _census_runtime_fingerprint(census: PotentialRootCensus) -> str:
    payload = asdict(census)
    payload.pop("_validation_capability", None)
    return sha256_hex(
        {
            "schema_id": "cascadiav3.rival_validated_potential_root_census_runtime.v1",
            "fields": payload,
        }
    )


def _qualified_sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise RivalSchemaError(f"{field_name} must use the 'sha256:' wire")
    return "sha256:" + require_sha256(value, field_name)


def _public_root_id(value: Any, field_name: str) -> str:
    text = require_nonempty_string(value, field_name)
    if not text.startswith(PUBLIC_ROOT_ID_PREFIX):
        raise RivalSchemaError(f"{field_name} must use {PUBLIC_ROOT_ID_PREFIX!r}")
    require_sha256(text.removeprefix(PUBLIC_ROOT_ID_PREFIX), field_name)
    return text


def validate_potential_root_census(
    record: Mapping[str, Any],
    *,
    allocation_registry: AllocationRegistry,
    expected_allocation_registry_identity: str,
    expected_content_sha256: str,
) -> PotentialRootCensus:
    if not isinstance(allocation_registry, AllocationRegistry):
        raise RivalSchemaError(
            "potential-root census requires an externally pinned AllocationRegistry"
        )
    require_schema(record, RIVAL_POTENTIAL_ROOT_CENSUS_SCHEMA_ID)
    require_exact_keys(record, required=_CENSUS_FIELDS, where="potential-root census")
    content_sha256 = verify_content_hash(record)
    expected = require_sha256(expected_content_sha256, "expected_content_sha256")
    if content_sha256 != expected:
        raise RivalSchemaError("potential-root census differs from its preregistered pin")
    family_kind = record["family_kind"]
    if family_kind not in {"finite_training_corpus", "one_seat_instrument"}:
        raise RivalSchemaError(f"unknown error family kind {family_kind!r}")
    try:
        expected_roots = allocation_registry.eligible_root_ids_for_family(
            family_kind,
            expected_allocation_registry_identity=expected_allocation_registry_identity,
        )
    except CohortError as exc:
        raise RivalSchemaError(str(exc)) from exc
    if not expected_roots:
        raise RivalSchemaError(
            f"allocation registry contains no roots for error family {family_kind!r}"
        )
    roots_raw = record["eligible_root_ids"]
    if not isinstance(roots_raw, list) or not roots_raw:
        raise RivalSchemaError("eligible_root_ids must be a non-empty ordered list")
    roots = tuple(
        _public_root_id(value, f"eligible_root_ids[{index}]")
        for index, value in enumerate(roots_raw)
    )
    if roots != tuple(sorted(set(roots))):
        raise RivalSchemaError(
            "eligible_root_ids must be unique and sorted in canonical lexical order"
        )
    if roots != expected_roots:
        raise RivalSchemaError(
            "eligible_root_ids must exactly equal the externally pinned allocation "
            "registry family universe"
        )
    source_root_set = _qualified_sha256(
        record["source_root_set_sha256"],
        "source_root_set_sha256",
    )
    if source_root_set != allocation_registry.root_source_set_sha256:
        raise RivalSchemaError(
            "source_root_set_sha256 must equal the allocation registry root-source set"
        )
    source_registry = _qualified_sha256(
        record["allocation_registry_identity"], "allocation_registry_identity"
    )
    if source_registry != allocation_registry.identity:
        raise RivalSchemaError(
            "allocation_registry_identity must equal the supplied registry identity"
        )
    census = PotentialRootCensus(
        census_id=require_nonempty_string(record["census_id"], "census_id"),
        family_kind=family_kind,
        source_root_set_sha256=source_root_set,
        allocation_registry_identity=source_registry,
        eligible_root_ids=roots,
        content_sha256=content_sha256,
    )
    return replace(
        census,
        _validation_capability=_issue_validation_capability(
            "PotentialRootCensus",
            _census_runtime_fingerprint(census),
        ),
    )


def load_potential_root_census(
    path: str | Path,
    *,
    allocation_registry: AllocationRegistry,
    expected_allocation_registry_identity: str,
    expected_file_sha256: str,
    expected_content_sha256: str,
) -> PotentialRootCensus:
    record = read_pinned_canonical_json_object(
        path,
        expected_file_sha256=expected_file_sha256,
        field="potential-root census",
    )
    return validate_potential_root_census(
        record,
        allocation_registry=allocation_registry,
        expected_allocation_registry_identity=expected_allocation_registry_identity,
        expected_content_sha256=expected_content_sha256,
    )


@dataclass(frozen=True)
class RootErrorEntry:
    root_id: str
    delta_root: float
    delta_h: float
    delta_l: float
    potentially_eligible: bool = True

    def allocation(self) -> RootErrorAllocation:
        return RootErrorAllocation(self.delta_h, self.delta_l, self.delta_root)

    @property
    def allocation_identity(self) -> str:
        return "sha256:" + sha256_hex(
            {
                "schema_id": "cascadiav3.rival_root_error_allocation.v1",
                "inference_mode": "multifidelity",
                "root_id": self.root_id,
                "delta_root": self.delta_root,
                "delta_h": self.delta_h,
                "delta_l": self.delta_l,
                "potentially_eligible": self.potentially_eligible,
            }
        )


@dataclass(frozen=True)
class HighOnlyRootErrorEntry:
    root_id: str
    delta_root: float
    delta_h: float
    potentially_eligible: bool = True

    def allocation(self) -> HighOnlyErrorAllocation:
        return HighOnlyErrorAllocation(self.delta_h, self.delta_root)

    @property
    def allocation_identity(self) -> str:
        return "sha256:" + sha256_hex(
            {
                "schema_id": "cascadiav3.rival_root_error_allocation.v1",
                "inference_mode": "high_fidelity_only",
                "root_id": self.root_id,
                "delta_root": self.delta_root,
                "delta_h": self.delta_h,
                "potentially_eligible": self.potentially_eligible,
            }
        )


@dataclass(frozen=True)
class ErrorFamilyLedger:
    family_id: str
    family_kind: ErrorFamilyKind
    delta_family: float
    certified_potential_appeal_count: int
    roots: tuple[RootErrorEntry | HighOnlyRootErrorEntry, ...]
    source_census_identity: str | None = None
    artifact_content_sha256: str | None = dataclass_field(
        default=None,
        repr=False,
        compare=False,
    )
    _validation_capability: object | None = dataclass_field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self._validation_capability is not None:
            try:
                _require_validation_capability(
                    self._validation_capability,
                    artifact_kind="ErrorFamilyLedger",
                    content_sha256=_error_ledger_runtime_fingerprint(self),
                )
            except RivalSchemaError as exc:
                raise CoverageError(str(exc)) from exc

    @property
    def content_sha256(self) -> str:
        if self.artifact_content_sha256 is not None:
            return self.artifact_content_sha256
        roots = []
        for row in self.roots:
            if isinstance(row, RootErrorEntry):
                roots.append(
                    {
                        "inference_mode": "multifidelity",
                        "root_id": row.root_id,
                        "delta_root": row.delta_root,
                        "delta_h": row.delta_h,
                        "delta_l": row.delta_l,
                        "potentially_eligible": row.potentially_eligible,
                    }
                )
            elif isinstance(row, HighOnlyRootErrorEntry):
                roots.append(
                    {
                        "inference_mode": "high_fidelity_only",
                        "root_id": row.root_id,
                        "delta_root": row.delta_root,
                        "delta_h": row.delta_h,
                        "potentially_eligible": row.potentially_eligible,
                    }
                )
            else:  # Defensive: a forged dataclass-like row must not hash as valid.
                raise CoverageError(f"unknown root-error entry type {type(row).__name__}")
        return sha256_hex(
            {
                "schema_id": "cascadiav3.rival_error_family_ledger.v1",
                "family_id": self.family_id,
                "family_kind": self.family_kind,
                "delta_family": self.delta_family,
                "certified_potential_appeal_count": self.certified_potential_appeal_count,
                "roots": roots,
            }
        )

    @property
    def identity(self) -> str:
        return "sha256:" + self.content_sha256

    def validate(self) -> float:
        if not isinstance(self.family_id, str) or not self.family_id:
            raise CoverageError("family_id must be non-empty")
        if self.family_kind not in {"finite_training_corpus", "one_seat_instrument"}:
            raise CoverageError(f"unknown error family kind {self.family_kind!r}")
        delta_family = require_finite(self.delta_family, "delta_family")
        if not 0.0 < delta_family < 1.0:
            raise CoverageError("delta_family must be in (0, 1)")
        if (
            isinstance(self.certified_potential_appeal_count, bool)
            or not isinstance(self.certified_potential_appeal_count, int)
            or self.certified_potential_appeal_count <= 0
        ):
            raise CoverageError("certified potential appeal count must be positive")
        if not isinstance(self.roots, tuple):
            raise CoverageError("error-family roots must be a frozen tuple")
        ids: set[str] = set()
        eligible = 0
        total = 0.0
        for row in self.roots:
            if not isinstance(row, (RootErrorEntry, HighOnlyRootErrorEntry)):
                raise CoverageError("error-family root has an unknown typed allocation")
            if (
                not isinstance(row.root_id, str)
                or not row.root_id
                or row.root_id != row.root_id.strip()
                or row.root_id in ids
            ):
                raise CoverageError(f"empty or duplicate root error entry {row.root_id!r}")
            if not isinstance(row.potentially_eligible, bool):
                raise CoverageError("potentially_eligible must be boolean")
            ids.add(row.root_id)
            row.allocation()
            if row.potentially_eligible:
                eligible += 1
                total += row.delta_root
        if eligible != self.certified_potential_appeal_count:
            raise CoverageError(
                "error ledger must enumerate every potentially eligible appeal: "
                f"found {eligible}, certified {self.certified_potential_appeal_count}"
            )
        if total > delta_family + 1.0e-15:
            raise CoverageError(f"sum(delta_root)={total} exceeds family budget {delta_family}")
        return total

    def require_validated_artifact(
        self,
        *,
        census: PotentialRootCensus | None = None,
    ) -> None:
        if self.artifact_content_sha256 is None or self.source_census_identity is None:
            raise CoverageError(
                "error family must come from a pinned census-complete ledger validator"
            )
        try:
            _require_validation_capability(
                self._validation_capability,
                artifact_kind="ErrorFamilyLedger",
                content_sha256=_error_ledger_runtime_fingerprint(self),
            )
        except RivalSchemaError as exc:
            raise CoverageError(str(exc)) from exc
        self.validate()
        if census is not None:
            census.require_validated_artifact()
            if self.source_census_identity != census.identity:
                raise CoverageError("error ledger is bound to a different potential-root census")
            if self.family_kind != census.family_kind:
                raise CoverageError("error ledger and potential-root census family kinds differ")
            if tuple(row.root_id for row in self.roots) != census.eligible_root_ids:
                raise CoverageError("error ledger root set differs from its complete census")


_ERROR_LEDGER_FIELDS = (
    "schema_id",
    "family_id",
    "family_kind",
    "source_census_sha256",
    "delta_family",
    "roots",
    "content_sha256",
)
_MF_ERROR_ROOT_FIELDS = (
    "inference_mode",
    "root_id",
    "delta_root",
    "delta_h",
    "delta_l",
)
_HIGH_ONLY_ERROR_ROOT_FIELDS = (
    "inference_mode",
    "root_id",
    "delta_root",
    "delta_h",
)


def _error_ledger_runtime_fingerprint(ledger: ErrorFamilyLedger) -> str:
    payload = asdict(ledger)
    payload.pop("_validation_capability", None)
    return sha256_hex(
        {
            "schema_id": "cascadiav3.rival_validated_error_family_runtime.v1",
            "fields": payload,
        }
    )


def validate_error_family_ledger(
    record: Mapping[str, Any],
    *,
    census: PotentialRootCensus,
    expected_content_sha256: str,
) -> ErrorFamilyLedger:
    """Validate one census-complete family allocation with no omitted roots."""

    if not isinstance(census, PotentialRootCensus):
        raise RivalSchemaError("error ledger requires a PotentialRootCensus")
    try:
        census.require_validated_artifact()
    except CoverageError as exc:
        raise RivalSchemaError(str(exc)) from exc
    require_schema(record, RIVAL_ERROR_FAMILY_LEDGER_SCHEMA_ID)
    require_exact_keys(record, required=_ERROR_LEDGER_FIELDS, where="error family ledger")
    content_sha256 = verify_content_hash(record)
    expected = require_sha256(expected_content_sha256, "expected_content_sha256")
    if content_sha256 != expected:
        raise RivalSchemaError("error family ledger differs from its preregistered pin")
    if record["family_kind"] != census.family_kind:
        raise RivalSchemaError("error family ledger and census family kinds differ")
    if record["source_census_sha256"] != census.identity:
        raise RivalSchemaError("error family ledger does not pin the supplied census")
    delta_family = require_finite(record["delta_family"], "delta_family")
    if not 0.0 < delta_family < 1.0:
        raise RivalSchemaError("delta_family must be in (0, 1)")
    roots_raw = record["roots"]
    if not isinstance(roots_raw, list) or not roots_raw:
        raise RivalSchemaError("error family roots must be a non-empty ordered list")
    roots: list[RootErrorEntry | HighOnlyRootErrorEntry] = []
    for index, raw in enumerate(roots_raw):
        if not isinstance(raw, Mapping):
            raise RivalSchemaError(f"error family root {index} must be an object")
        mode = raw.get("inference_mode")
        if mode == "multifidelity":
            require_exact_keys(
                raw,
                required=_MF_ERROR_ROOT_FIELDS,
                where=f"error family root {index}",
            )
            row: RootErrorEntry | HighOnlyRootErrorEntry = RootErrorEntry(
                _public_root_id(raw["root_id"], f"roots[{index}].root_id"),
                require_finite(raw["delta_root"], f"roots[{index}].delta_root"),
                require_finite(raw["delta_h"], f"roots[{index}].delta_h"),
                require_finite(raw["delta_l"], f"roots[{index}].delta_l"),
            )
        elif mode == "high_fidelity_only":
            require_exact_keys(
                raw,
                required=_HIGH_ONLY_ERROR_ROOT_FIELDS,
                where=f"error family root {index}",
            )
            row = HighOnlyRootErrorEntry(
                _public_root_id(raw["root_id"], f"roots[{index}].root_id"),
                require_finite(raw["delta_root"], f"roots[{index}].delta_root"),
                require_finite(raw["delta_h"], f"roots[{index}].delta_h"),
            )
        else:
            raise RivalSchemaError(f"unknown inference_mode for error family root {index}")
        try:
            row.allocation()
        except (RivalSchemaError, ValueError) as exc:
            raise RivalSchemaError(f"invalid error family root {index}: {exc}") from exc
        roots.append(row)
    observed_root_ids = tuple(row.root_id for row in roots)
    if observed_root_ids != census.eligible_root_ids:
        raise RivalSchemaError(
            "error family roots must exactly equal the supplied census in canonical order"
        )
    ledger = ErrorFamilyLedger(
        family_id=require_nonempty_string(record["family_id"], "family_id"),
        family_kind=census.family_kind,
        delta_family=delta_family,
        certified_potential_appeal_count=len(census.eligible_root_ids),
        roots=tuple(roots),
        source_census_identity=census.identity,
        artifact_content_sha256=content_sha256,
    )
    try:
        ledger.validate()
    except CoverageError as exc:
        raise RivalSchemaError(str(exc)) from exc
    return replace(
        ledger,
        _validation_capability=_issue_validation_capability(
            "ErrorFamilyLedger",
            _error_ledger_runtime_fingerprint(ledger),
        ),
    )


def load_error_family_ledger(
    path: str | Path,
    *,
    census: PotentialRootCensus,
    expected_file_sha256: str,
    expected_content_sha256: str,
) -> ErrorFamilyLedger:
    record = read_pinned_canonical_json_object(
        path,
        expected_file_sha256=expected_file_sha256,
        field="error family ledger",
    )
    return validate_error_family_ledger(
        record,
        census=census,
        expected_content_sha256=expected_content_sha256,
    )


def validate_separate_error_families(families: Iterable[ErrorFamilyLedger]) -> None:
    rows = tuple(families)
    kinds = [family.family_kind for family in rows]
    if set(kinds) != {"finite_training_corpus", "one_seat_instrument"}:
        raise CoverageError(
            "validation requires separate finite-training and one-seat error families"
        )
    if len(set(kinds)) != len(kinds):
        raise CoverageError("each error-family kind may appear at most once")
    root_ids: set[str] = set()
    for family in rows:
        if not isinstance(family, ErrorFamilyLedger):
            raise CoverageError("error families must contain ErrorFamilyLedger artifacts")
        family.require_validated_artifact()
        current = {row.root_id for row in family.roots}
        overlap = root_ids & current
        if overlap:
            raise CoverageError(
                f"training and one-seat error families reuse root identities: {sorted(overlap)}"
            )
        root_ids.update(current)


@dataclass(frozen=True)
class DiscreteJointOutcome:
    high: float
    low: float
    probability: float


@dataclass(frozen=True)
class DiscreteLowOutcome:
    low: float
    probability: float


def _validate_distribution(probabilities: Iterable[float], name: str) -> None:
    values = tuple(probabilities)
    if not values:
        raise CoverageError(f"{name} distribution is empty")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0.0
        for value in values
    ):
        raise CoverageError(f"{name} probabilities must be finite and positive")
    if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise CoverageError(f"{name} probabilities must sum to one")


@dataclass(frozen=True)
class ExactCoverageResult:
    true_high_mean: float
    expected_estimate: float
    undercoverage_probability: float
    enumerated_sample_panels: int


def enumerate_exact_coverage(
    *,
    h_distribution: Sequence[DiscreteJointOutcome],
    l_distribution: Sequence[DiscreteLowOutcome],
    n_h: int,
    n_l: int,
    beta_cv: float,
    high_difference_width: float,
    low_difference_width: float,
    allocation: RootErrorAllocation,
    max_panels: int = 1_000_000,
) -> ExactCoverageResult:
    """Enumerate iid finite panels and compute exact bound undercoverage."""
    if not isinstance(h_distribution, Sequence) or not all(
        isinstance(row, DiscreteJointOutcome) for row in h_distribution
    ):
        raise CoverageError("H distribution requires typed DiscreteJointOutcome rows")
    if not isinstance(l_distribution, Sequence) or not all(
        isinstance(row, DiscreteLowOutcome) for row in l_distribution
    ):
        raise CoverageError("L distribution requires typed DiscreteLowOutcome rows")
    if not isinstance(allocation, RootErrorAllocation):
        raise CoverageError("exact coverage requires a RootErrorAllocation")
    if (
        isinstance(n_h, bool)
        or not isinstance(n_h, int)
        or isinstance(n_l, bool)
        or not isinstance(n_l, int)
        or n_h <= 0
        or n_l <= 0
    ):
        raise CoverageError("n_h and n_l must be positive")
    if isinstance(max_panels, bool) or not isinstance(max_panels, int) or max_panels <= 0:
        raise CoverageError("max_panels must be a positive integer")
    _validate_distribution((row.probability for row in h_distribution), "H")
    _validate_distribution((row.probability for row in l_distribution), "L")
    beta_cv = require_finite(beta_cv, "beta_cv")
    high_difference_width = require_finite(high_difference_width, "high_difference_width")
    low_difference_width = require_finite(low_difference_width, "low_difference_width")
    if high_difference_width < 0.0 or low_difference_width < 0.0:
        raise CoverageError("declared difference widths must be non-negative")
    high_values = [row.high for row in h_distribution]
    low_values = [row.low for row in h_distribution] + [row.low for row in l_distribution]
    if not all(
        not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
        for value in (*high_values, *low_values)
    ):
        raise CoverageError("finite-distribution outcomes must be finite")
    if high_difference_width + 1.0e-12 < max(high_values) - min(high_values):
        raise CoverageError("declared high difference width does not cover finite support")
    if low_difference_width + 1.0e-12 < max(low_values) - min(low_values):
        raise CoverageError("declared low difference width does not cover finite support")
    low_h_marginal: dict[float, float] = {}
    low_l_marginal: dict[float, float] = {}
    for row in h_distribution:
        low_h_marginal[row.low] = low_h_marginal.get(row.low, 0.0) + row.probability
    for row in l_distribution:
        low_l_marginal[row.low] = low_l_marginal.get(row.low, 0.0) + row.probability
    if set(low_h_marginal) != set(low_l_marginal) or any(
        not math.isclose(low_h_marginal[value], low_l_marginal[value], rel_tol=0.0, abs_tol=1.0e-12)
        for value in low_h_marginal
    ):
        raise CoverageError(
            "exact coverage requires the frozen H-low and independent L-low laws to match"
        )
    panel_count = len(h_distribution) ** n_h * len(l_distribution) ** n_l
    if panel_count > max_panels:
        raise CoverageError(f"exact design has {panel_count} panels, above max_panels={max_panels}")
    true_high_mean = sum(row.high * row.probability for row in h_distribution)
    expected_estimate = 0.0
    undercoverage = 0.0
    widths = transformed_widths(
        beta_cv=beta_cv,
        high_difference_width=high_difference_width,
        low_difference_width=low_difference_width,
    )
    for h_panel in product(h_distribution, repeat=n_h):
        probability_h = math.prod(row.probability for row in h_panel)
        paired = tuple(
            PairedDifference(f"h-{index}", row.high, row.low) for index, row in enumerate(h_panel)
        )
        for l_panel in product(l_distribution, repeat=n_l):
            probability = probability_h * math.prod(row.probability for row in l_panel)
            extra = tuple(LowDifference(f"l-{index}", row.low) for index, row in enumerate(l_panel))
            estimate = estimate_fixed_panels(
                paired,
                extra,
                beta_cv=beta_cv,
                expected_n_h=n_h,
                expected_n_l=n_l,
            )
            lower = fixed_hoeffding_lower_bound(
                high_corrected_mean=estimate.high_corrected_mean,
                low_correction_mean=estimate.low_correction_mean,
                widths=widths,
                allocation=allocation,
                n_h=n_h,
                n_l=n_l,
            ).lower_bound
            expected_estimate += probability * estimate.estimate
            if lower > true_high_mean:
                undercoverage += probability
    return ExactCoverageResult(
        true_high_mean=true_high_mean,
        expected_estimate=expected_estimate,
        undercoverage_probability=undercoverage,
        enumerated_sample_panels=panel_count,
    )


def binomial_upper_confidence_bound(failures: int, replications: int, *, alpha: float) -> float:
    """Exact one-sided Clopper-Pearson upper bound, dependency-free."""
    if isinstance(failures, bool) or not isinstance(failures, int):
        raise CoverageError("failures must be an integer")
    if isinstance(replications, bool) or not isinstance(replications, int):
        raise CoverageError("replications must be an integer")
    if replications <= 0 or not 0 <= failures <= replications:
        raise CoverageError("require 0 <= failures <= positive replications")
    alpha = require_finite(alpha, "alpha")
    if not 0.0 < alpha < 1.0:
        raise CoverageError("alpha must be in (0, 1)")
    if failures == replications:
        return 1.0
    if failures == 0:
        return 1.0 - alpha ** (1.0 / replications)

    def binomial_cdf(p: float) -> float:
        # Log-sum-exp remains stable when (1-p)**n would underflow.  This
        # branch has p strictly inside (0, 1), because bisection never evaluates
        # an endpoint.
        log_p = math.log(p)
        log_one_minus_p = math.log1p(-p)
        terms = [
            math.lgamma(replications + 1)
            - math.lgamma(k + 1)
            - math.lgamma(replications - k + 1)
            + k * log_p
            + (replications - k) * log_one_minus_p
            for k in range(failures + 1)
        ]
        maximum = max(terms)
        if maximum < math.log(float.fromhex("0x0.0000000000001p-1022")):
            return 0.0
        return math.exp(maximum) * sum(math.exp(term - maximum) for term in terms)

    low, high = 0.0, 1.0
    for _ in range(200):
        mid = (low + high) / 2.0
        if binomial_cdf(mid) > alpha:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def zero_failure_replications(*, tolerance: float, confidence: float) -> int:
    """Replications required for a zero-failure CP upper bound <= tolerance."""
    tolerance = require_finite(tolerance, "tolerance")
    confidence = require_finite(confidence, "confidence")
    if not 0.0 < tolerance < 1.0:
        raise CoverageError("tolerance must be in (0, 1)")
    if not 0.0 < confidence < 1.0:
        raise CoverageError("confidence must be in (0, 1)")
    alpha = 1.0 - confidence
    return math.ceil(math.log(alpha) / math.log(1.0 - tolerance))


def run_coverage_design(record: Mapping[str, object]) -> dict[str, object]:
    """Run compact exact fixtures from a JSON-compatible design."""
    require_exact_keys(
        record,
        required=("design_version", "exact_cases", "content_sha256"),
        where="coverage design",
    )
    design_sha256 = verify_content_hash(record)
    if record["design_version"] != 1 or isinstance(record["design_version"], bool):
        raise CoverageError("coverage design_version must be 1")
    cases = record["exact_cases"]
    if not isinstance(cases, list) or not cases:
        raise CoverageError("coverage design requires non-empty exact_cases")
    results: list[dict[str, object]] = []
    for case in cases:
        if not isinstance(case, Mapping):
            raise CoverageError("each exact case must be an object")
        require_exact_keys(
            case,
            required=(
                "case_id",
                "h_distribution",
                "l_distribution",
                "n_h",
                "n_l",
                "beta_cv",
                "high_difference_width",
                "low_difference_width",
                "allocation",
            ),
            where="coverage exact case",
        )
        case_id = require_nonempty_string(case["case_id"], "case_id")
        h_raw = case["h_distribution"]
        l_raw = case["l_distribution"]
        if not isinstance(h_raw, list) or not isinstance(l_raw, list):
            raise CoverageError("case distributions must be lists")
        for index, row in enumerate(h_raw):
            if not isinstance(row, Mapping):
                raise CoverageError(f"H distribution row {index} must be an object")
            require_exact_keys(
                row,
                required=("high", "low", "probability"),
                where=f"H distribution row {index}",
            )
        for index, row in enumerate(l_raw):
            if not isinstance(row, Mapping):
                raise CoverageError(f"L distribution row {index} must be an object")
            require_exact_keys(
                row,
                required=("low", "probability"),
                where=f"L distribution row {index}",
            )
        high_low_distribution = tuple(
            DiscreteJointOutcome(
                require_finite(row["high"], "H.high"),
                require_finite(row["low"], "H.low"),
                require_finite(row["probability"], "H.probability"),
            )
            for row in h_raw
        )
        extra_low_distribution = tuple(
            DiscreteLowOutcome(
                require_finite(row["low"], "L.low"),
                require_finite(row["probability"], "L.probability"),
            )
            for row in l_raw
        )
        allocation_raw = case["allocation"]
        if not isinstance(allocation_raw, Mapping):
            raise CoverageError("allocation must be an object")
        require_exact_keys(
            allocation_raw,
            required=("delta_h", "delta_l", "delta_root"),
            where="coverage allocation",
        )
        allocation = RootErrorAllocation(
            require_finite(allocation_raw["delta_h"], "allocation.delta_h"),
            require_finite(allocation_raw["delta_l"], "allocation.delta_l"),
            require_finite(allocation_raw["delta_root"], "allocation.delta_root"),
        )
        result = enumerate_exact_coverage(
            h_distribution=high_low_distribution,
            l_distribution=extra_low_distribution,
            n_h=require_positive_int(case["n_h"], "n_h"),
            n_l=require_positive_int(case["n_l"], "n_l"),
            beta_cv=require_finite(case["beta_cv"], "beta_cv"),
            high_difference_width=require_finite(
                case["high_difference_width"], "high_difference_width"
            ),
            low_difference_width=require_finite(
                case["low_difference_width"], "low_difference_width"
            ),
            allocation=allocation,
        )
        results.append({"case_id": case_id, **asdict(result)})
    return attach_content_hash(
        {
            "status": "PASS",
            "scope": "synthetic_or_enumerable_cpu_only",
            "strength_evidence": False,
            "coverage_design_sha256": "sha256:" + design_sha256,
            "cases": results,
        }
    )


def hash_fixture_tree(path: str | Path) -> tuple[str, int]:
    """Read and bind every regular file in a fixture directory."""

    root = Path(path)
    if not root.is_dir() or root.is_symlink():
        raise CoverageError("--fixtures must name a real, non-symlink directory")
    rows: list[dict[str, str]] = []
    for source in sorted(root.rglob("*")):
        if source.is_symlink():
            raise CoverageError(f"fixture tree contains a symlink: {source}")
        if source.is_file():
            rows.append(
                {
                    "path": source.relative_to(root).as_posix(),
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                }
            )
    if not rows:
        raise CoverageError("fixture tree contains no regular files")
    return sha256_hex({"schema_id": "cascadiav3.rival_fixture_tree.v1", "files": rows}), len(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--coverage-design", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.device != "cpu":
        parser.error("pre-GPU coverage accepts only explicit --device cpu")
    try:
        design = read_strict_json_object(args.coverage_design, field="coverage design")
        report = run_coverage_design(design)
        fixture_tree_sha256, fixture_file_count = hash_fixture_tree(args.fixtures)
        report.pop("content_sha256")
        report["fixture_tree_sha256"] = "sha256:" + fixture_tree_sha256
        report["fixture_file_count"] = fixture_file_count
        report = attach_content_hash(report)
        write_new_canonical_json(args.out, report)
    except (
        OSError,
        TypeError,
        KeyError,
        CoverageError,
        RivalSchemaError,
    ) as exc:
        print(json.dumps({"status": "DENIED", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
