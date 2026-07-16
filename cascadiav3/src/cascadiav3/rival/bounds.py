"""Verify Rust-authored score bounds and apply rules-agnostic interval algebra.

Python is intentionally not a Cascadia range authority.  It accepts only a
hash-pinned certificate naming ``crates/cascadia-rival::bounds`` as producer,
checks all caller-pinned identities, and then performs generic Hoeffding
arithmetic on the certified widths.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any

from .manifest import CANONICAL_RESEARCH_RULESET
from .schema import (
    RIVAL_BOUND_CERTIFICATE_SCHEMA_ID,
    RivalSchemaError,
    _issue_validation_capability,
    _require_validation_capability,
    read_strict_json_object,
    require_exact_keys,
    require_finite,
    require_nonempty_string,
    require_positive_int,
    require_probability,
    require_schema,
    require_sha256,
    verify_content_hash,
)

RUST_BOUND_AUTHORITY = "cascadia-rival/rust-global-aaaaa-score-relaxation-v1"
GLOBAL_BOUND_SCOPE = "global_terminal_own_score_difference"
TRUSTED_GLOBAL_CERTIFICATE_SHA256 = (
    "67fa21e1f4e887f73a1f0f4e22397ca23f79ca67972b44e34f94f385734eec64"
)


class BoundError(ValueError):
    """Raised when a certificate or confidence-bound design is invalid."""


@dataclass(frozen=True)
class CertifiedRange:
    minimum: float
    maximum: float
    width: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], name: str) -> CertifiedRange:
        require_exact_keys(value, required=("minimum", "maximum", "width"), where=f"{name} range")
        minimum = require_finite(value["minimum"], f"{name}.minimum")
        maximum = require_finite(value["maximum"], f"{name}.maximum")
        width = require_finite(value["width"], f"{name}.width")
        if maximum < minimum:
            raise RivalSchemaError(f"{name} maximum is below minimum")
        recomputed = maximum - minimum
        if width < 0.0 or not math.isclose(width, recomputed, rel_tol=0.0, abs_tol=1.0e-12):
            raise RivalSchemaError(
                f"{name}.width {width} does not equal maximum-minimum {recomputed}"
            )
        return cls(minimum, maximum, width)


@dataclass(frozen=True)
class BoundCertificate:
    authority_id: str
    ruleset: dict[str, str]
    rules_semantics_id: str
    game_config_sha256: str
    scope: str
    terminal_score: CertifiedRange
    high: CertifiedRange
    low: CertifiedRange
    derivation: dict[str, int]
    content_sha256: str
    _validation_capability: object | None = dataclass_field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self._validation_capability is not None:
            _require_validation_capability(
                self._validation_capability,
                artifact_kind="BoundCertificate",
                content_sha256=_bound_certificate_runtime_fingerprint(self),
            )


def _bound_certificate_runtime_fingerprint(certificate: BoundCertificate) -> str:
    from .schema import sha256_hex

    payload = asdict(certificate)
    payload.pop("_validation_capability", None)
    return sha256_hex(
        {
            "schema_id": "cascadiav3.rival_validated_bound_certificate_runtime.v1",
            "fields": payload,
        }
    )


def require_validated_bound_certificate(certificate: BoundCertificate) -> None:
    if not isinstance(certificate, BoundCertificate):
        raise BoundError("bound certificate must be a BoundCertificate")
    try:
        _require_validation_capability(
            certificate._validation_capability,
            artifact_kind="BoundCertificate",
            content_sha256=_bound_certificate_runtime_fingerprint(certificate),
        )
    except RivalSchemaError as exc:
        raise BoundError(str(exc)) from exc


_CERTIFICATE_FIELDS = (
    "schema_id",
    "authority_id",
    "ruleset",
    "scope",
    "terminal_score_min",
    "terminal_score_max",
    "score_difference_min",
    "score_difference_max",
    "score_difference_width",
    "derivation",
    "certificate_sha256",
)

_RULESET_FIELDS = (
    "schema_id",
    "legacy_ruleset_id",
    "rules_semantics_id",
    "game_config_sha256",
)

_DERIVATION_FIELDS = (
    "maximum_board_tiles",
    "terrain_memberships_per_tile",
    "maximum_personal_wildlife_placements",
    "maximum_points_per_wildlife_token",
    "maximum_nature_tokens_earned",
    "habitat_bonus_points",
)


def verify_bound_certificate(
    record: Mapping[str, Any],
    *,
    expected_certificate_sha256: str = TRUSTED_GLOBAL_CERTIFICATE_SHA256,
    expected_ruleset: Mapping[str, str] | None = None,
    expected_scope: str = GLOBAL_BOUND_SCOPE,
) -> BoundCertificate:
    """Verify an immutable certificate and every caller-pinned identity."""
    requested_hash = require_sha256(expected_certificate_sha256, "expected_certificate_sha256")
    if requested_hash != TRUSTED_GLOBAL_CERTIFICATE_SHA256:
        raise RivalSchemaError(
            "caller cannot substitute a different global bound certificate authority"
        )
    if expected_ruleset is None:
        expected_ruleset = CANONICAL_RESEARCH_RULESET
    if dict(expected_ruleset) != CANONICAL_RESEARCH_RULESET:
        raise RivalSchemaError("caller cannot substitute a different canonical ruleset")
    require_schema(record, RIVAL_BOUND_CERTIFICATE_SCHEMA_ID)
    require_exact_keys(record, required=_CERTIFICATE_FIELDS, where="bound certificate")
    authority = require_nonempty_string(record["authority_id"], "authority_id")
    if authority != RUST_BOUND_AUTHORITY:
        raise RivalSchemaError(f"Python does not accept non-Rust range authority {authority!r}")
    ruleset_raw = record["ruleset"]
    derivation_raw = record["derivation"]
    if not isinstance(ruleset_raw, Mapping):
        raise RivalSchemaError("ruleset must be a structured JSON object")
    if not isinstance(derivation_raw, Mapping):
        raise RivalSchemaError("derivation must be a JSON object")
    require_exact_keys(ruleset_raw, required=_RULESET_FIELDS, where="ruleset")
    require_exact_keys(derivation_raw, required=_DERIVATION_FIELDS, where="derivation")
    if not isinstance(ruleset_raw["game_config_sha256"], str) or not ruleset_raw[
        "game_config_sha256"
    ].startswith("sha256:"):
        raise RivalSchemaError("ruleset.game_config_sha256 must use the Rust 'sha256:' wire")
    ruleset = {
        "schema_id": require_nonempty_string(ruleset_raw["schema_id"], "ruleset.schema_id"),
        "legacy_ruleset_id": require_nonempty_string(
            ruleset_raw["legacy_ruleset_id"], "ruleset.legacy_ruleset_id"
        ),
        "rules_semantics_id": require_nonempty_string(
            ruleset_raw["rules_semantics_id"], "ruleset.rules_semantics_id"
        ),
        "game_config_sha256": "sha256:"
        + require_sha256(ruleset_raw["game_config_sha256"], "ruleset.game_config_sha256"),
    }
    if dict(expected_ruleset) != ruleset:
        raise RivalSchemaError("bound certificate structured ruleset identity mismatch")
    scope = require_nonempty_string(record["scope"], "scope")
    if scope != expected_scope:
        raise RivalSchemaError(f"bound certificate scope mismatch: {scope!r} != {expected_scope!r}")
    terminal_min = require_finite(record["terminal_score_min"], "terminal_score_min")
    terminal_max = require_finite(record["terminal_score_max"], "terminal_score_max")
    if terminal_max < terminal_min:
        raise RivalSchemaError("terminal score maximum is below minimum")
    difference = CertifiedRange.from_mapping(
        {
            "minimum": record["score_difference_min"],
            "maximum": record["score_difference_max"],
            "width": record["score_difference_width"],
        },
        "score_difference",
    )
    derivation: dict[str, int] = {}
    for field in _DERIVATION_FIELDS:
        value = derivation_raw[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RivalSchemaError(f"derivation.{field} must be a non-negative integer")
        derivation[field] = value
    # Python checks wire consistency but does not reproduce the rules-aware
    # derivation.  The caller-pinned Rust certificate hash is the authority.
    if not isinstance(record["certificate_sha256"], str) or not record[
        "certificate_sha256"
    ].startswith("sha256:"):
        raise RivalSchemaError("certificate_sha256 must use the Rust 'sha256:' wire")
    content_hash = verify_content_hash(record, "certificate_sha256")
    if content_hash != TRUSTED_GLOBAL_CERTIFICATE_SHA256:
        raise RivalSchemaError(
            "bound certificate is not the trusted Rust-authored global certificate: "
            f"{content_hash} != {TRUSTED_GLOBAL_CERTIFICATE_SHA256}"
        )
    certificate = BoundCertificate(
        authority_id=authority,
        ruleset=ruleset,
        rules_semantics_id=ruleset["rules_semantics_id"],
        game_config_sha256=ruleset["game_config_sha256"],
        scope=scope,
        terminal_score=CertifiedRange(terminal_min, terminal_max, terminal_max - terminal_min),
        high=difference,
        low=difference,
        derivation=derivation,
        content_sha256=content_hash,
    )
    return replace(
        certificate,
        _validation_capability=_issue_validation_capability(
            "BoundCertificate",
            _bound_certificate_runtime_fingerprint(certificate),
        ),
    )


def load_bound_certificate(path: str | Path, **expected: str) -> BoundCertificate:
    return verify_bound_certificate(
        read_strict_json_object(path, field="bound certificate"), **expected
    )


@dataclass(frozen=True)
class TransformedWidths:
    high_corrected: float
    low_correction: float


def transformed_widths(
    *, beta_cv: float, high_difference_width: float, low_difference_width: float
) -> TransformedWidths:
    beta = require_finite(beta_cv, "beta_cv")
    width_h = require_finite(high_difference_width, "high_difference_width")
    width_l = require_finite(low_difference_width, "low_difference_width")
    if width_h < 0.0 or width_l < 0.0:
        raise BoundError("certified difference widths must be non-negative")
    return TransformedWidths(width_h + abs(beta) * width_l, abs(beta) * width_l)


def transformed_widths_from_certificate(
    certificate: BoundCertificate, *, beta_cv: float
) -> TransformedWidths:
    require_validated_bound_certificate(certificate)
    return transformed_widths(
        beta_cv=beta_cv,
        high_difference_width=certificate.high.width,
        low_difference_width=certificate.low.width,
    )


@dataclass(frozen=True)
class RootErrorAllocation:
    delta_h: float
    delta_l: float
    delta_root: float

    def __post_init__(self) -> None:
        delta_h = require_probability(self.delta_h, "delta_h")
        delta_l = require_probability(self.delta_l, "delta_l")
        delta_root = require_probability(self.delta_root, "delta_root")
        if delta_h + delta_l > delta_root + 1.0e-15:
            raise BoundError(
                f"delta_h + delta_l = {delta_h + delta_l} exceeds delta_root {delta_root}"
            )


@dataclass(frozen=True)
class HighOnlyErrorAllocation:
    delta_h: float
    delta_root: float

    def __post_init__(self) -> None:
        delta_h = require_probability(self.delta_h, "delta_h")
        delta_root = require_probability(self.delta_root, "delta_root")
        if delta_h > delta_root:
            raise BoundError(f"delta_h {delta_h} exceeds delta_root {delta_root}")


@dataclass(frozen=True)
class HoeffdingBound:
    estimate: float
    lower_bound: float
    high_penalty: float
    low_penalty: float
    widths: TransformedWidths
    allocation: RootErrorAllocation
    n_h: int
    n_l: int


@dataclass(frozen=True)
class HighOnlyHoeffdingBound:
    estimate: float
    lower_bound: float
    penalty: float
    certified_width: float
    allocation: HighOnlyErrorAllocation
    n_h: int


def fixed_high_only_hoeffding_lower_bound(
    *,
    high_mean: float,
    certified_width: float,
    allocation: HighOnlyErrorAllocation,
    n_h: int,
) -> HighOnlyHoeffdingBound:
    """Separate S/H high-fidelity-only control; never labeled multifidelity."""
    if not isinstance(allocation, HighOnlyErrorAllocation):
        raise BoundError("high-only bound requires a HighOnlyErrorAllocation")
    estimate = require_finite(high_mean, "high_mean")
    width = require_finite(certified_width, "certified_width")
    n_h = require_positive_int(n_h, "n_h")
    if width < 0.0:
        raise BoundError("certified_width must be non-negative")
    penalty = width * math.sqrt(math.log(1.0 / allocation.delta_h) / (2.0 * n_h))
    return HighOnlyHoeffdingBound(
        estimate=estimate,
        lower_bound=estimate - penalty,
        penalty=penalty,
        certified_width=width,
        allocation=allocation,
        n_h=n_h,
    )


def fixed_hoeffding_lower_bound(
    *,
    high_corrected_mean: float,
    low_correction_mean: float,
    widths: TransformedWidths,
    allocation: RootErrorAllocation,
    n_h: int,
    n_l: int,
) -> HoeffdingBound:
    """One-sided fixed-sample lower bound for ``mean(X) + mean(Y)``."""
    if not isinstance(widths, TransformedWidths):
        raise BoundError("multifidelity bound requires typed TransformedWidths")
    if not isinstance(allocation, RootErrorAllocation):
        raise BoundError("multifidelity bound requires a RootErrorAllocation")
    high_mean = require_finite(high_corrected_mean, "high_corrected_mean")
    low_mean = require_finite(low_correction_mean, "low_correction_mean")
    n_h = require_positive_int(n_h, "n_h")
    n_l = require_positive_int(n_l, "n_l")
    high_width = require_finite(widths.high_corrected, "widths.high_corrected")
    low_width = require_finite(widths.low_correction, "widths.low_correction")
    if high_width < 0.0 or low_width < 0.0:
        raise BoundError("transformed widths must be non-negative")
    high_penalty = high_width * math.sqrt(math.log(1.0 / allocation.delta_h) / (2.0 * n_h))
    low_penalty = low_width * math.sqrt(math.log(1.0 / allocation.delta_l) / (2.0 * n_l))
    estimate = high_mean + low_mean
    return HoeffdingBound(
        estimate=estimate,
        lower_bound=estimate - high_penalty - low_penalty,
        high_penalty=high_penalty,
        low_penalty=low_penalty,
        widths=widths,
        allocation=allocation,
        n_h=n_h,
        n_l=n_l,
    )
