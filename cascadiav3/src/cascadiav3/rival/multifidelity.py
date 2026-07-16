"""Fixed-panel multifidelity estimator for terminal own-score differences.

The target is always the high-fidelity one-deviation advantage.  Low-fidelity
returns enter only through a zero-mean control-variate correction.  This file
contains algebra, not adaptive stopping or rules-aware score bounds.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from dataclasses import field as dataclass_field
from pathlib import Path
from statistics import fmean
from typing import Any

from .schema import (
    RIVAL_COEFFICIENT_CALIBRATION_SCHEMA_ID,
    RivalSchemaError,
    _issue_validation_capability,
    _require_validation_capability,
    read_pinned_canonical_json_object,
    require_exact_keys,
    require_finite,
    require_nonempty_string,
    require_schema,
    require_sha256,
    sha256_hex,
    verify_content_hash,
)


class MultifidelityError(ValueError):
    """Raised when a fixed multifidelity design or panel is invalid."""


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MultifidelityError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise MultifidelityError(f"{name} must be finite")
    return result


def _count(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MultifidelityError(f"{name} must be a positive integer")
    return value


def _identity(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise MultifidelityError(f"{name} must be a non-empty, whitespace-trimmed string")
    return value


@dataclass(frozen=True)
class PairedDifference:
    """One complete H-panel unit: high and low differences on a coupled world."""

    unit_id: str
    high: float
    low: float

    def __post_init__(self) -> None:
        _identity(self.unit_id, "paired unit_id")
        _finite(self.high, "paired high difference")
        _finite(self.low, "paired low difference")


@dataclass(frozen=True)
class LowDifference:
    """One complete independent L-panel low-fidelity action difference."""

    unit_id: str
    low: float

    def __post_init__(self) -> None:
        _identity(self.unit_id, "low unit_id")
        _finite(self.low, "extra-low difference")


@dataclass(frozen=True)
class HighDifference:
    """One complete high-fidelity-only H-panel action difference."""

    unit_id: str
    high: float

    def __post_init__(self) -> None:
        _identity(self.unit_id, "high-only unit_id")
        _finite(self.high, "high-only difference")


@dataclass(frozen=True)
class CoefficientBinding:
    """A coefficient frozen on calibration and bound to a deployment design."""

    coefficient_id: str
    beta_cv: float
    calibration_cohort_id: str
    deployment_design_id: str
    deployment_design_sha256: str
    incumbent_policy_id: str
    low_policy_id: str
    sampler_id: str
    allocation_id: str
    low_expectation_h_id: str
    low_expectation_l_id: str
    low_law_h_id: str
    low_law_l_id: str
    max_abs_beta: float
    calibration_source_corpus_sha256: str | None = None
    calibration_root_index_sha256: str | None = None
    calibration_data_sha256: str | None = None
    estimator_identity: str | None = None
    _artifact_content_sha256: str | None = dataclass_field(
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
        for field in (
            "coefficient_id",
            "calibration_cohort_id",
            "deployment_design_id",
            "deployment_design_sha256",
            "incumbent_policy_id",
            "low_policy_id",
            "sampler_id",
            "allocation_id",
            "low_expectation_h_id",
            "low_expectation_l_id",
            "low_law_h_id",
            "low_law_l_id",
        ):
            _identity(getattr(self, field), field)
        if not self.deployment_design_sha256.startswith("sha256:"):
            raise MultifidelityError("deployment_design_sha256 must use the sha256: wire")
        require_sha256(self.deployment_design_sha256, "deployment_design_sha256")
        beta = _finite(self.beta_cv, "beta_cv")
        limit = _finite(self.max_abs_beta, "max_abs_beta")
        if limit <= 0.0:
            raise MultifidelityError("max_abs_beta must be positive")
        if abs(beta) > limit:
            raise MultifidelityError(f"unstable beta_cv {beta} exceeds frozen max_abs_beta {limit}")
        if self.calibration_cohort_id == self.deployment_design_id:
            raise MultifidelityError(
                "coefficient calibration and deployment/coverage identities must be disjoint"
            )
        if self.low_expectation_h_id != self.low_expectation_l_id:
            raise MultifidelityError(
                "H-low and L-low panels do not target the same frozen expectation"
            )
        optional_artifact_fields = (
            self.calibration_source_corpus_sha256,
            self.calibration_root_index_sha256,
            self.calibration_data_sha256,
            self.estimator_identity,
            self._artifact_content_sha256,
        )
        if any(value is not None for value in optional_artifact_fields) and any(
            value is None for value in optional_artifact_fields
        ):
            raise MultifidelityError(
                "coefficient artifact provenance fields must be all present or all absent"
            )
        if self._validation_capability is not None:
            try:
                _require_validation_capability(
                    self._validation_capability,
                    artifact_kind="CoefficientBinding",
                    content_sha256=_coefficient_runtime_fingerprint(self),
                )
            except RivalSchemaError as exc:
                raise MultifidelityError(str(exc)) from exc

    @property
    def equal_low_law(self) -> bool:
        return self.low_law_h_id == self.low_law_l_id

    @property
    def content_sha256(self) -> str:
        """Canonical, domain-separated digest of every coefficient input."""
        if self._artifact_content_sha256 is not None:
            return self._artifact_content_sha256
        return sha256_hex(
            {
                "schema_id": "cascadiav3.rival_coefficient_binding.v1",
                "coefficient_id": self.coefficient_id,
                "beta_cv": self.beta_cv,
                "calibration_cohort_id": self.calibration_cohort_id,
                "deployment_design_id": self.deployment_design_id,
                "deployment_design_sha256": self.deployment_design_sha256,
                "incumbent_policy_id": self.incumbent_policy_id,
                "low_policy_id": self.low_policy_id,
                "sampler_id": self.sampler_id,
                "allocation_id": self.allocation_id,
                "low_expectation_h_id": self.low_expectation_h_id,
                "low_expectation_l_id": self.low_expectation_l_id,
                "low_law_h_id": self.low_law_h_id,
                "low_law_l_id": self.low_law_l_id,
                "max_abs_beta": self.max_abs_beta,
            }
        )

    @property
    def identity(self) -> str:
        return "sha256:" + self.content_sha256

    @property
    def is_validated_artifact(self) -> bool:
        try:
            self.require_validated_artifact()
        except MultifidelityError:
            return False
        return True

    def require_validated_artifact(self) -> None:
        if self._artifact_content_sha256 is None:
            raise MultifidelityError(
                "coefficient must come from a pinned calibration artifact validator"
            )
        try:
            _require_validation_capability(
                self._validation_capability,
                artifact_kind="CoefficientBinding",
                content_sha256=_coefficient_runtime_fingerprint(self),
            )
        except RivalSchemaError as exc:
            raise MultifidelityError(str(exc)) from exc

    def require_design(
        self,
        *,
        deployment_design_id: str,
        deployment_design_sha256: str,
        incumbent_policy_id: str,
        low_policy_id: str,
        sampler_id: str,
        allocation_id: str,
    ) -> None:
        expected = {
            "deployment_design_id": deployment_design_id,
            "deployment_design_sha256": deployment_design_sha256,
            "incumbent_policy_id": incumbent_policy_id,
            "low_policy_id": low_policy_id,
            "sampler_id": sampler_id,
            "allocation_id": allocation_id,
        }
        for field, value in expected.items():
            if getattr(self, field) != value:
                raise MultifidelityError(
                    f"coefficient binding mismatch for {field}: "
                    f"{getattr(self, field)!r} != {value!r}"
                )


_COEFFICIENT_ARTIFACT_FIELDS = (
    "schema_id",
    "coefficient_id",
    "beta_cv",
    "calibration_cohort_id",
    "calibration_source_corpus_sha256",
    "calibration_root_index_sha256",
    "calibration_data_sha256",
    "deployment_design_id",
    "deployment_design_sha256",
    "incumbent_policy_id",
    "low_policy_id",
    "sampler_id",
    "allocation_id",
    "low_expectation_h_id",
    "low_expectation_l_id",
    "low_law_h_id",
    "low_law_l_id",
    "max_abs_beta",
    "estimator_identity",
    "content_sha256",
)


def _coefficient_runtime_fingerprint(binding: CoefficientBinding) -> str:
    payload = asdict(binding)
    payload.pop("_validation_capability", None)
    return sha256_hex(
        {
            "schema_id": "cascadiav3.rival_validated_coefficient_runtime.v1",
            "fields": payload,
        }
    )


def _qualified_sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise RivalSchemaError(f"{field_name} must use the 'sha256:' wire")
    return "sha256:" + require_sha256(value, field_name)


def validate_coefficient_calibration(
    record: Mapping[str, Any],
    *,
    expected_content_sha256: str,
) -> CoefficientBinding:
    """Validate one immutable, preregistered coefficient-calibration artifact."""

    require_schema(record, RIVAL_COEFFICIENT_CALIBRATION_SCHEMA_ID)
    require_exact_keys(
        record,
        required=_COEFFICIENT_ARTIFACT_FIELDS,
        where="coefficient calibration artifact",
    )
    content_sha256 = verify_content_hash(record)
    expected = require_sha256(expected_content_sha256, "expected_content_sha256")
    if content_sha256 != expected:
        raise RivalSchemaError("coefficient calibration content differs from its preregistered pin")
    binding = CoefficientBinding(
        coefficient_id=require_nonempty_string(record["coefficient_id"], "coefficient_id"),
        beta_cv=require_finite(record["beta_cv"], "beta_cv"),
        calibration_cohort_id=require_nonempty_string(
            record["calibration_cohort_id"], "calibration_cohort_id"
        ),
        deployment_design_id=require_nonempty_string(
            record["deployment_design_id"], "deployment_design_id"
        ),
        deployment_design_sha256=_qualified_sha256(
            record["deployment_design_sha256"], "deployment_design_sha256"
        ),
        incumbent_policy_id=_qualified_sha256(record["incumbent_policy_id"], "incumbent_policy_id"),
        low_policy_id=_qualified_sha256(record["low_policy_id"], "low_policy_id"),
        sampler_id=_qualified_sha256(record["sampler_id"], "sampler_id"),
        allocation_id=_qualified_sha256(record["allocation_id"], "allocation_id"),
        low_expectation_h_id=require_nonempty_string(
            record["low_expectation_h_id"], "low_expectation_h_id"
        ),
        low_expectation_l_id=require_nonempty_string(
            record["low_expectation_l_id"], "low_expectation_l_id"
        ),
        low_law_h_id=require_nonempty_string(record["low_law_h_id"], "low_law_h_id"),
        low_law_l_id=require_nonempty_string(record["low_law_l_id"], "low_law_l_id"),
        max_abs_beta=require_finite(record["max_abs_beta"], "max_abs_beta"),
        calibration_source_corpus_sha256=_qualified_sha256(
            record["calibration_source_corpus_sha256"],
            "calibration_source_corpus_sha256",
        ),
        calibration_root_index_sha256=_qualified_sha256(
            record["calibration_root_index_sha256"],
            "calibration_root_index_sha256",
        ),
        calibration_data_sha256=_qualified_sha256(
            record["calibration_data_sha256"],
            "calibration_data_sha256",
        ),
        estimator_identity=_qualified_sha256(record["estimator_identity"], "estimator_identity"),
        _artifact_content_sha256=content_sha256,
    )
    return replace(
        binding,
        _validation_capability=_issue_validation_capability(
            "CoefficientBinding",
            _coefficient_runtime_fingerprint(binding),
        ),
    )


def load_coefficient_calibration(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_content_sha256: str,
) -> CoefficientBinding:
    record = read_pinned_canonical_json_object(
        path,
        expected_file_sha256=expected_file_sha256,
        field="coefficient calibration artifact",
    )
    return validate_coefficient_calibration(
        record,
        expected_content_sha256=expected_content_sha256,
    )


@dataclass(frozen=True)
class MultifidelityEstimate:
    estimate: float
    mean_high: float
    mean_low_on_h: float
    mean_low_on_l: float
    high_corrected_mean: float
    low_correction_mean: float
    beta_cv: float
    n_h: int
    n_l: int


@dataclass(frozen=True)
class HighFidelityEstimate:
    estimate: float
    n_h: int


def estimate_high_fidelity_only(
    rows: Sequence[HighDifference], *, expected_n_h: int
) -> HighFidelityEstimate:
    """Compute the separate high-only control with no low panel or MF claim."""
    expected_n_h = _count(expected_n_h, "expected_n_h")
    if len(rows) != expected_n_h:
        raise MultifidelityError(
            f"incomplete high-only H panel: {len(rows)} rows; expected {expected_n_h}"
        )
    ids = [row.unit_id for row in rows]
    if len(set(ids)) != len(ids):
        raise MultifidelityError("duplicate unit_id in high-only H panel")
    return HighFidelityEstimate(fmean(row.high for row in rows), expected_n_h)


def optimal_beta_general(
    *,
    n_h: int,
    n_l: int,
    covariance_high_low_h: float,
    variance_low_h: float,
    variance_low_l: float,
) -> float:
    """Return the population variance-minimizing coefficient.

    This formula permits different low-panel variances.  It assumes panel H is
    independent of panel L and that only the H high/low pair is correlated.
    """
    n_h = _count(n_h, "n_h")
    n_l = _count(n_l, "n_l")
    covariance = _finite(covariance_high_low_h, "covariance_high_low_h")
    var_h = _finite(variance_low_h, "variance_low_h")
    var_l = _finite(variance_low_l, "variance_low_l")
    if var_h < 0.0 or var_l < 0.0:
        raise MultifidelityError("low-fidelity variances must be non-negative")
    denominator = var_h / n_h + var_l / n_l
    if denominator <= 0.0:
        raise MultifidelityError("low-fidelity control has zero total variance")
    return (covariance / n_h) / denominator


def optimal_beta_equal_law(
    *, n_h: int, n_l: int, covariance_high_low: float, variance_low: float
) -> float:
    """Special equal-law/equal-variance population coefficient."""
    n_h = _count(n_h, "n_h")
    n_l = _count(n_l, "n_l")
    covariance = _finite(covariance_high_low, "covariance_high_low")
    variance = _finite(variance_low, "variance_low")
    if variance <= 0.0:
        raise MultifidelityError("variance_low must be positive")
    return n_l / (n_h + n_l) * covariance / variance


@dataclass(frozen=True)
class BetaOptimization:
    beta_cv: float
    method: str


def optimal_beta_for_registered_design(
    *,
    n_h: int,
    n_l: int,
    covariance_high_low_h: float,
    variance_high: float,
    variance_low_h: float,
    variance_low_l: float,
    low_expectation_h_id: str,
    low_expectation_l_id: str,
    low_law_h_id: str,
    low_law_l_id: str,
    equal_law_assumptions_certified: bool,
) -> BetaOptimization:
    """Select the special formula only for an explicitly certified equal law."""
    _identity(low_expectation_h_id, "low_expectation_h_id")
    _identity(low_expectation_l_id, "low_expectation_l_id")
    _identity(low_law_h_id, "low_law_h_id")
    _identity(low_law_l_id, "low_law_l_id")
    if low_expectation_h_id != low_expectation_l_id:
        raise MultifidelityError("H-low and L-low expectations differ; estimator is biased")
    variance_high = _finite(variance_high, "variance_high")
    variance_low_h_checked = _finite(variance_low_h, "variance_low_h")
    covariance_checked = _finite(covariance_high_low_h, "covariance_high_low_h")
    if variance_high < 0.0 or variance_low_h_checked < 0.0:
        raise MultifidelityError("registered variances must be non-negative")
    if covariance_checked * covariance_checked > (
        variance_high * variance_low_h_checked
        + 1.0e-12 * max(1.0, variance_high * variance_low_h_checked)
    ):
        raise MultifidelityError("registered covariance violates the Cauchy-Schwarz bound")
    same_law = low_law_h_id == low_law_l_id
    if equal_law_assumptions_certified:
        if not same_law:
            raise MultifidelityError("equal-law certificate conflicts with low-panel law IDs")
        variance_h = _finite(variance_low_h, "variance_low_h")
        variance_l = _finite(variance_low_l, "variance_low_l")
        if not math.isclose(variance_h, variance_l, rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise MultifidelityError("equal-law certificate has unequal low-panel variances")
        special = optimal_beta_equal_law(
            n_h=n_h,
            n_l=n_l,
            covariance_high_low=covariance_high_low_h,
            variance_low=variance_h,
        )
        general = optimal_beta_general(
            n_h=n_h,
            n_l=n_l,
            covariance_high_low_h=covariance_high_low_h,
            variance_low_h=variance_h,
            variance_low_l=variance_l,
        )
        if not math.isclose(special, general, rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise MultifidelityError("special coefficient failed reduction-to-general check")
        return BetaOptimization(special, "certified_equal_low_law")
    return BetaOptimization(
        optimal_beta_general(
            n_h=n_h,
            n_l=n_l,
            covariance_high_low_h=covariance_high_low_h,
            variance_low_h=variance_low_h,
            variance_low_l=variance_low_l,
        ),
        "general_independent_panel_variance",
    )


def estimator_variance_general(
    *,
    n_h: int,
    n_l: int,
    beta_cv: float,
    variance_high: float,
    variance_low_h: float,
    variance_low_l: float,
    covariance_high_low_h: float,
) -> float:
    """Population variance of the independent two-panel estimator."""
    n_h = _count(n_h, "n_h")
    n_l = _count(n_l, "n_l")
    beta = _finite(beta_cv, "beta_cv")
    var_high = _finite(variance_high, "variance_high")
    var_low_h = _finite(variance_low_h, "variance_low_h")
    var_low_l = _finite(variance_low_l, "variance_low_l")
    covariance = _finite(covariance_high_low_h, "covariance_high_low_h")
    if min(var_high, var_low_h, var_low_l) < 0.0:
        raise MultifidelityError("variances must be non-negative")
    covariance_limit_squared = var_high * var_low_h
    if covariance * covariance > (
        covariance_limit_squared + 1.0e-12 * max(1.0, covariance_limit_squared)
    ):
        raise MultifidelityError("covariance violates the Cauchy-Schwarz bound")
    result = (
        var_high / n_h
        + beta * beta * (var_low_h / n_h + var_low_l / n_l)
        - 2.0 * beta * covariance / n_h
    )
    # A negative result beyond round-off means the supplied moments do not form
    # a positive-semidefinite covariance design.
    tolerance = 1.0e-12 * max(1.0, abs(var_high / n_h))
    if result < -tolerance:
        raise MultifidelityError(
            "supplied moments yield negative estimator variance; covariance is invalid"
        )
    return max(0.0, result)


def estimate_fixed_panels(
    paired_h: Sequence[PairedDifference],
    extra_l: Sequence[LowDifference],
    *,
    beta_cv: float,
    expected_n_h: int,
    expected_n_l: int,
) -> MultifidelityEstimate:
    """Compute exactly one complete fixed-panel estimate.

    This function accepts only successful rows.  Operational failures and the
    single-look state machine live in :mod:`cascadiav3.rival.appeals`; passing a
    shortened list here is rejected rather than silently changing allocation.
    """
    expected_n_h = _count(expected_n_h, "expected_n_h")
    expected_n_l = _count(expected_n_l, "expected_n_l")
    beta = _finite(beta_cv, "beta_cv")
    if len(paired_h) != expected_n_h:
        raise MultifidelityError(
            f"incomplete H panel: {len(paired_h)} rows; expected {expected_n_h}"
        )
    if len(extra_l) != expected_n_l:
        raise MultifidelityError(
            f"incomplete L panel: {len(extra_l)} rows; expected {expected_n_l}"
        )
    h_ids = [row.unit_id for row in paired_h]
    l_ids = [row.unit_id for row in extra_l]
    if len(set(h_ids)) != len(h_ids):
        raise MultifidelityError("duplicate unit_id in H panel")
    if len(set(l_ids)) != len(l_ids):
        raise MultifidelityError("duplicate unit_id in L panel")
    if set(h_ids) & set(l_ids):
        raise MultifidelityError("H and L panel unit identities overlap")

    mean_high = fmean(_finite(row.high, "D_H") for row in paired_h)
    mean_low_h = fmean(_finite(row.low, "D_L_on_H") for row in paired_h)
    mean_low_l = fmean(_finite(row.low, "D_L_on_L") for row in extra_l)
    high_corrected = mean_high - beta * mean_low_h
    low_correction = beta * mean_low_l
    return MultifidelityEstimate(
        estimate=high_corrected + low_correction,
        mean_high=mean_high,
        mean_low_on_h=mean_low_h,
        mean_low_on_l=mean_low_l,
        high_corrected_mean=high_corrected,
        low_correction_mean=low_correction,
        beta_cv=beta,
        n_h=expected_n_h,
        n_l=expected_n_l,
    )


def negate_paired(rows: Iterable[PairedDifference]) -> tuple[PairedDifference, ...]:
    """Swap challenger/incumbent orientation for an H panel."""
    return tuple(PairedDifference(row.unit_id, -row.high, -row.low) for row in rows)


def negate_low(rows: Iterable[LowDifference]) -> tuple[LowDifference, ...]:
    """Swap challenger/incumbent orientation for an L panel."""
    return tuple(LowDifference(row.unit_id, -row.low) for row in rows)
