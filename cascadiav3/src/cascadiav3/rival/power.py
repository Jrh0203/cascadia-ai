"""Reproducible, symbolic pre-measurement Rival power envelopes.

Every rate, covariance, activation, and timeout input in this module is a
hypothesis.  Measured cost fields remain the literal string ``UNRESOLVED``;
therefore the output cannot fund or close the real program before P2a/P2b.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

from .bounds import RootErrorAllocation, fixed_hoeffding_lower_bound, transformed_widths
from .multifidelity import estimator_variance_general, optimal_beta_general
from .schema import (
    RIVAL_POWER_ENVELOPE_SCHEMA_ID,
    RivalSchemaError,
    attach_content_hash,
    require_exact_keys,
    require_finite,
    require_schema,
    require_sha256,
    verify_content_hash,
    write_new_canonical_json,
)

UNRESOLVED = "UNRESOLVED"
NON_FUNDING_STATUS = "NON_FUNDING_SYMBOLIC_ONLY"
NO_FINITE_HOURS = "NO_FINITE_HOURS_AT_FIXED_ROOT_ALLOCATION"


class PowerEnvelopeError(ValueError):
    """Raised when a symbolic envelope disguises an unresolved measurement."""


def _is_finite_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _require_ordered_unique_grid(values: tuple[Any, ...], name: str) -> None:
    if len(values) != len(set(values)):
        raise PowerEnvelopeError(f"{name} values must be unique")
    try:
        ordered = tuple(sorted(values))
    except TypeError as exc:
        raise PowerEnvelopeError(f"{name} values must have one numeric type") from exc
    if values != ordered:
        raise PowerEnvelopeError(f"{name} values must be in ascending canonical order")


@dataclass(frozen=True)
class CertifiedStratumRange:
    stratum: str
    bound_certificate_sha256: str
    high_difference_width: float
    low_difference_width: float


@dataclass(frozen=True)
class HypotheticalThroughput:
    scenario: str
    selection_pair_seconds: float
    high_pair_seconds: float
    low_pair_seconds: float
    fixed_root_seconds: float
    parallel_active_roots: int
    assumption_basis: str


@dataclass(frozen=True)
class MemoryAssumption:
    scenario: str
    available_memory_gib: float
    gib_per_active_root: float
    fixed_workspace_gib: float
    assumption_basis: str


@dataclass(frozen=True)
class PowerEnvelopeSpec:
    envelope_id: str
    source_revision: str
    certified_ranges: tuple[CertifiedStratumRange, ...]
    candidate_count: int
    finite_training_family_count: int
    one_seat_family_count: int
    certified_potential_appeals: int
    selection_units_per_candidate: int
    delta_game: float
    n_h_grid: tuple[int, ...]
    n_l_grid: tuple[int, ...]
    covariance_grid: tuple[float, ...]
    variance_high_assumption: float
    variance_low_h_assumption: float
    variance_low_l_assumption: float
    target_gap_grid: tuple[float, ...]
    activation_frequency_grid: tuple[float, ...]
    timeout_rate_grid: tuple[float, ...]
    practical_margin: float
    target_confirmed_roots: int
    calibration_root_requirement: int
    throughput_assumptions: tuple[HypotheticalThroughput, ...]
    memory_assumptions: tuple[MemoryAssumption, ...]


def _validate_spec(spec: PowerEnvelopeSpec) -> None:
    if not isinstance(spec, PowerEnvelopeSpec):
        raise PowerEnvelopeError("power envelope requires a typed PowerEnvelopeSpec")
    if any(
        not isinstance(value, str) or not value.strip() or value != value.strip()
        for value in (spec.envelope_id, spec.source_revision)
    ):
        raise PowerEnvelopeError(
            "envelope_id and source_revision must be non-empty trimmed strings"
        )
    if not isinstance(spec.certified_ranges, tuple) or not spec.certified_ranges:
        raise PowerEnvelopeError("at least one certified stratum range is required")
    if any(not isinstance(item, CertifiedStratumRange) for item in spec.certified_ranges):
        raise PowerEnvelopeError("certified_ranges must contain typed range records")
    for value, name in (
        (spec.candidate_count, "candidate_count"),
        (spec.finite_training_family_count, "finite_training_family_count"),
        (spec.one_seat_family_count, "one_seat_family_count"),
        (spec.certified_potential_appeals, "certified_potential_appeals"),
        (spec.selection_units_per_candidate, "selection_units_per_candidate"),
        (spec.target_confirmed_roots, "target_confirmed_roots"),
        (spec.calibration_root_requirement, "calibration_root_requirement"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise PowerEnvelopeError(f"{name} must be a positive integer")
    if spec.finite_training_family_count != 1 or spec.one_seat_family_count != 1:
        raise PowerEnvelopeError("v1 requires exactly two separate, named error families")
    if spec.candidate_count < 2:
        raise PowerEnvelopeError("candidate_count must include an incumbent and challenger")
    if not _is_finite_number(spec.delta_game) or not 0.0 < spec.delta_game < 1.0:
        raise PowerEnvelopeError("delta_game must be in (0, 1)")
    if not _is_finite_number(spec.practical_margin) or spec.practical_margin < 0.0:
        raise PowerEnvelopeError("practical_margin must be finite and non-negative")
    for grid, name in ((spec.n_h_grid, "n_h_grid"), (spec.n_l_grid, "n_l_grid")):
        if (
            not isinstance(grid, tuple)
            or not grid
            or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in grid)
        ):
            raise PowerEnvelopeError(f"{name} requires positive integer values")
        _require_ordered_unique_grid(grid, name)
    if (
        not isinstance(spec.covariance_grid, tuple)
        or not spec.covariance_grid
        or any(not _is_finite_number(value) for value in spec.covariance_grid)
    ):
        raise PowerEnvelopeError("covariance_grid values must be finite")
    _require_ordered_unique_grid(spec.covariance_grid, "covariance_grid")
    moment_variances = (
        spec.variance_high_assumption,
        spec.variance_low_h_assumption,
        spec.variance_low_l_assumption,
    )
    if any(not _is_finite_number(value) or value <= 0.0 for value in moment_variances):
        raise PowerEnvelopeError("symbolic population variances must be finite and positive")
    covariance_limit_squared = spec.variance_high_assumption * spec.variance_low_h_assumption
    if any(
        covariance * covariance
        > covariance_limit_squared + 1.0e-12 * max(1.0, covariance_limit_squared)
        for covariance in spec.covariance_grid
    ):
        raise PowerEnvelopeError("covariance grid violates the Cauchy-Schwarz bound")
    if (
        not isinstance(spec.target_gap_grid, tuple)
        or not spec.target_gap_grid
        or any(not _is_finite_number(value) or value < 0.0 for value in spec.target_gap_grid)
    ):
        raise PowerEnvelopeError("target_gap_grid values must be finite and non-negative")
    _require_ordered_unique_grid(spec.target_gap_grid, "target_gap_grid")
    if (
        not isinstance(spec.activation_frequency_grid, tuple)
        or not spec.activation_frequency_grid
        or any(
            not _is_finite_number(value) or not 0.0 < value <= 1.0
            for value in spec.activation_frequency_grid
        )
    ):
        raise PowerEnvelopeError("activation frequencies must be in (0, 1]")
    _require_ordered_unique_grid(spec.activation_frequency_grid, "activation_frequency_grid")
    if (
        not isinstance(spec.timeout_rate_grid, tuple)
        or not spec.timeout_rate_grid
        or any(
            not _is_finite_number(value) or not 0.0 <= value < 1.0
            for value in spec.timeout_rate_grid
        )
    ):
        raise PowerEnvelopeError("timeout rates must be in [0, 1)")
    _require_ordered_unique_grid(spec.timeout_rate_grid, "timeout_rate_grid")
    if not isinstance(spec.throughput_assumptions, tuple) or any(
        not isinstance(item, HypotheticalThroughput) for item in spec.throughput_assumptions
    ):
        raise PowerEnvelopeError("throughput_assumptions must contain typed throughput records")
    scenarios = [item.scenario for item in spec.throughput_assumptions]
    if any(not isinstance(scenario, str) for scenario in scenarios):
        raise PowerEnvelopeError("throughput scenario names must be strings")
    if len(scenarios) != 3 or set(scenarios) != {"optimistic", "central", "pessimistic"}:
        raise PowerEnvelopeError(
            "throughput assumptions require exactly one optimistic, central, and "
            "pessimistic scenario"
        )
    for item in spec.throughput_assumptions:
        numeric_costs = (
            item.selection_pair_seconds,
            item.high_pair_seconds,
            item.low_pair_seconds,
            item.fixed_root_seconds,
        )
        if (
            any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0.0
                for value in numeric_costs
            )
            or item.selection_pair_seconds <= 0.0
            or item.high_pair_seconds <= 0.0
            or item.low_pair_seconds <= 0.0
            or isinstance(item.parallel_active_roots, bool)
            or not isinstance(item.parallel_active_roots, int)
            or item.parallel_active_roots <= 0
            or not isinstance(item.assumption_basis, str)
            or not item.assumption_basis.strip()
            or item.assumption_basis != item.assumption_basis.strip()
        ):
            raise PowerEnvelopeError(
                "throughput unit costs and parallel roots must be positive and documented"
            )
    if not isinstance(spec.memory_assumptions, tuple) or not spec.memory_assumptions:
        raise PowerEnvelopeError("memory assumptions are required")
    if any(not isinstance(item, MemoryAssumption) for item in spec.memory_assumptions):
        raise PowerEnvelopeError("memory_assumptions must contain typed memory records")
    memory_scenarios = [item.scenario for item in spec.memory_assumptions]
    if any(
        not isinstance(scenario, str) or not scenario.strip() or scenario != scenario.strip()
        for scenario in memory_scenarios
    ) or len(memory_scenarios) != len(set(memory_scenarios)):
        raise PowerEnvelopeError("memory scenario names must be non-empty and unique")
    for item in spec.memory_assumptions:
        if (
            not isinstance(item.assumption_basis, str)
            or not item.assumption_basis.strip()
            or item.assumption_basis != item.assumption_basis.strip()
            or not _is_finite_number(item.available_memory_gib)
            or item.available_memory_gib <= 0.0
            or not _is_finite_number(item.gib_per_active_root)
            or item.gib_per_active_root <= 0.0
            or not _is_finite_number(item.fixed_workspace_gib)
            or item.fixed_workspace_gib < 0.0
            or item.fixed_workspace_gib >= item.available_memory_gib
        ):
            raise PowerEnvelopeError(
                "memory assumptions must leave positive documented active-root capacity"
            )
    seen_strata: set[str] = set()
    for item in spec.certified_ranges:
        if (
            not isinstance(item.stratum, str)
            or not item.stratum.strip()
            or item.stratum != item.stratum.strip()
            or item.stratum in seen_strata
        ):
            raise PowerEnvelopeError(f"empty or duplicate range stratum {item.stratum!r}")
        seen_strata.add(item.stratum)
        if not isinstance(
            item.bound_certificate_sha256, str
        ) or not item.bound_certificate_sha256.startswith("sha256:"):
            raise PowerEnvelopeError("certified range bound identity must use the 'sha256:' wire")
        try:
            require_sha256(item.bound_certificate_sha256, "bound_certificate_sha256")
        except RivalSchemaError as exc:
            raise PowerEnvelopeError(str(exc)) from exc
        if (
            not _is_finite_number(item.high_difference_width)
            or not _is_finite_number(item.low_difference_width)
            or item.high_difference_width < 0.0
            or item.low_difference_width < 0.0
        ):
            raise PowerEnvelopeError("certified widths must be finite and non-negative")


def build_power_envelope(spec: PowerEnvelopeSpec) -> dict[str, Any]:
    _validate_spec(spec)
    delta_root = spec.delta_game / spec.certified_potential_appeals
    allocation = RootErrorAllocation(delta_root / 2.0, delta_root / 2.0, delta_root)
    rows: list[dict[str, Any]] = []
    for (
        stratum,
        n_h,
        n_l,
        covariance,
        target_gap,
        activation,
        timeout,
        throughput,
        memory,
    ) in product(
        spec.certified_ranges,
        spec.n_h_grid,
        spec.n_l_grid,
        spec.covariance_grid,
        spec.target_gap_grid,
        spec.activation_frequency_grid,
        spec.timeout_rate_grid,
        spec.throughput_assumptions,
        spec.memory_assumptions,
    ):
        beta = optimal_beta_general(
            n_h=n_h,
            n_l=n_l,
            covariance_high_low_h=covariance,
            variance_low_h=spec.variance_low_h_assumption,
            variance_low_l=spec.variance_low_l_assumption,
        )
        hypothetical_variance = estimator_variance_general(
            n_h=n_h,
            n_l=n_l,
            beta_cv=beta,
            variance_high=spec.variance_high_assumption,
            variance_low_h=spec.variance_low_h_assumption,
            variance_low_l=spec.variance_low_l_assumption,
            covariance_high_low_h=covariance,
        )
        widths = transformed_widths(
            beta_cv=beta,
            high_difference_width=stratum.high_difference_width,
            low_difference_width=stratum.low_difference_width,
        )
        null_bound = fixed_hoeffding_lower_bound(
            high_corrected_mean=0.0,
            low_correction_mean=0.0,
            widths=widths,
            allocation=allocation,
            n_h=n_h,
            n_l=n_l,
        )
        half_width = -null_bound.lower_bound
        challenger_count = spec.candidate_count - 1
        selection_units = challenger_count * spec.selection_units_per_candidate
        terminal_pairs_per_root = 2 * n_h + n_l
        attempted_units_per_root = selection_units + terminal_pairs_per_root
        hypothetical_complete_probability = (1.0 - timeout) ** attempted_units_per_root
        label_probability = activation * hypothetical_complete_probability
        if label_probability == 0.0:
            raise PowerEnvelopeError(
                "timeout assumptions underflowed the symbolic completion probability"
            )
        memory_capacity = math.floor(
            (memory.available_memory_gib - memory.fixed_workspace_gib) / memory.gib_per_active_root
        )
        if memory_capacity <= 0:
            raise PowerEnvelopeError("memory assumption yields zero active-root capacity")
        effective_parallel_roots = min(throughput.parallel_active_roots, memory_capacity)
        root_work_seconds = (
            throughput.fixed_root_seconds
            + selection_units * throughput.selection_pair_seconds
            + n_h * (throughput.high_pair_seconds + throughput.low_pair_seconds)
            + n_l * throughput.low_pair_seconds
        )
        if root_work_seconds <= 0.0:
            raise PowerEnvelopeError("symbolic root work must be positive")
        roots_per_hour = effective_parallel_roots * 3600.0 / root_work_seconds
        resolves = target_gap > spec.practical_margin + half_width
        if resolves:
            attempted_roots: int | str = math.ceil(spec.target_confirmed_roots / label_probability)
            hypothetical_hours: float | str = attempted_roots / roots_per_hour
        else:
            attempted_roots = NO_FINITE_HOURS
            hypothetical_hours = NO_FINITE_HOURS
        rows.append(
            {
                "stratum": stratum.stratum,
                "bound_certificate_sha256": stratum.bound_certificate_sha256,
                "n_h": n_h,
                "n_l": n_l,
                "beta_cv_derived_from_covariance": beta,
                "covariance_assumption": covariance,
                "hypothetical_estimator_variance": hypothetical_variance,
                "target_gap_assumption": target_gap,
                "activation_frequency_assumption": activation,
                "timeout_rate_assumption": timeout,
                "throughput_scenario": throughput.scenario,
                "memory_scenario": memory.scenario,
                "selection_units_per_attempted_root": selection_units,
                "attempted_units_per_root": attempted_units_per_root,
                "hypothetical_root_work_seconds": root_work_seconds,
                "hypothetical_memory_capacity_roots": memory_capacity,
                "hypothetical_effective_parallel_roots": effective_parallel_roots,
                "roots_per_hour_assumption": roots_per_hour,
                "hoeffding_half_width": half_width,
                "symbolically_resolves_margin": resolves,
                "terminal_pairs_per_attempted_root": terminal_pairs_per_root,
                "required_attempted_roots_assuming_independent_timeouts": attempted_roots,
                "hypothetical_hours_not_decision_grade": hypothetical_hours,
                "decision_grade_gpu_hours": UNRESOLVED,
            }
        )
    base: dict[str, Any] = {
        "schema_id": RIVAL_POWER_ENVELOPE_SCHEMA_ID,
        "envelope_id": spec.envelope_id,
        "source_revision": spec.source_revision,
        "status": NON_FUNDING_STATUS,
        "can_fund_program": False,
        "can_close_program": False,
        "certified_range_by_stratum": [asdict(item) for item in spec.certified_ranges],
        "candidate_and_error_family_counts": {
            "candidate_count": spec.candidate_count,
            "finite_training_family_count": spec.finite_training_family_count,
            "one_seat_family_count": spec.one_seat_family_count,
            "certified_potential_appeals": spec.certified_potential_appeals,
            "selection_units_per_candidate": spec.selection_units_per_candidate,
        },
        "design_constants": {
            "delta_game": spec.delta_game,
            "practical_margin": spec.practical_margin,
            "target_confirmed_roots": spec.target_confirmed_roots,
        },
        "grid": {
            "n_h": list(spec.n_h_grid),
            "n_l": list(spec.n_l_grid),
            "beta_cv": "DERIVED_PER_ROW_FROM_COVARIANCE_AND_ALLOCATION",
            "covariance": list(spec.covariance_grid),
            "population_moment_assumptions": {
                "variance_high": spec.variance_high_assumption,
                "variance_low_h": spec.variance_low_h_assumption,
                "variance_low_l": spec.variance_low_l_assumption,
            },
            "target_gap": list(spec.target_gap_grid),
            "activation_frequency": list(spec.activation_frequency_grid),
            "timeout_rate": list(spec.timeout_rate_grid),
        },
        "calibration_requirements": {
            "root_count": spec.calibration_root_requirement,
            "must_be_disjoint_from_coverage": True,
            "selected_challenger_pipeline_must_match": True,
        },
        "throughput_assumptions": [asdict(item) for item in spec.throughput_assumptions],
        "memory_assumptions": [asdict(item) for item in spec.memory_assumptions],
        "measured_cost_fields": {
            "high_fidelity_terminal_pair_seconds": UNRESOLVED,
            "low_fidelity_terminal_pair_seconds": UNRESOLVED,
            "coupled_covariance_by_stratum": UNRESOLVED,
            "measured_activation_frequency": UNRESOLVED,
            "measured_timeout_rate": UNRESOLVED,
            "resolved_roots_per_hour": UNRESOLVED,
            "post_d1_john0_gpu_hours": UNRESOLVED,
        },
        "rows": rows,
        "interpretation": (
            "Synthetic no-go map only. Assumed rates and covariance cannot fund or close Rival; "
            "P2a/P2b must resolve every measured-cost field."
        ),
    }
    return attach_content_hash(base)


_ENVELOPE_FIELDS = (
    "schema_id",
    "envelope_id",
    "source_revision",
    "status",
    "can_fund_program",
    "can_close_program",
    "certified_range_by_stratum",
    "candidate_and_error_family_counts",
    "design_constants",
    "grid",
    "calibration_requirements",
    "throughput_assumptions",
    "memory_assumptions",
    "measured_cost_fields",
    "rows",
    "interpretation",
    "content_sha256",
)

_MEASURED_FIELDS = (
    "high_fidelity_terminal_pair_seconds",
    "low_fidelity_terminal_pair_seconds",
    "coupled_covariance_by_stratum",
    "measured_activation_frequency",
    "measured_timeout_rate",
    "resolved_roots_per_hour",
    "post_d1_john0_gpu_hours",
)

_GRID_FIELDS = (
    "n_h",
    "n_l",
    "beta_cv",
    "covariance",
    "population_moment_assumptions",
    "target_gap",
    "activation_frequency",
    "timeout_rate",
)

_ROW_FIELDS = (
    "stratum",
    "bound_certificate_sha256",
    "n_h",
    "n_l",
    "beta_cv_derived_from_covariance",
    "covariance_assumption",
    "hypothetical_estimator_variance",
    "target_gap_assumption",
    "activation_frequency_assumption",
    "timeout_rate_assumption",
    "throughput_scenario",
    "memory_scenario",
    "selection_units_per_attempted_root",
    "attempted_units_per_root",
    "hypothetical_root_work_seconds",
    "hypothetical_memory_capacity_roots",
    "hypothetical_effective_parallel_roots",
    "roots_per_hour_assumption",
    "hoeffding_half_width",
    "symbolically_resolves_margin",
    "terminal_pairs_per_attempted_root",
    "required_attempted_roots_assuming_independent_timeouts",
    "hypothetical_hours_not_decision_grade",
    "decision_grade_gpu_hours",
)

_COUNT_FIELDS = (
    "candidate_count",
    "finite_training_family_count",
    "one_seat_family_count",
    "certified_potential_appeals",
    "selection_units_per_candidate",
)

_DESIGN_FIELDS = ("delta_game", "practical_margin", "target_confirmed_roots")
_CALIBRATION_FIELDS = (
    "root_count",
    "must_be_disjoint_from_coverage",
    "selected_challenger_pipeline_must_match",
)
_CERTIFIED_RANGE_FIELDS = (
    "stratum",
    "bound_certificate_sha256",
    "high_difference_width",
    "low_difference_width",
)
_THROUGHPUT_FIELDS = (
    "scenario",
    "selection_pair_seconds",
    "high_pair_seconds",
    "low_pair_seconds",
    "fixed_root_seconds",
    "parallel_active_roots",
    "assumption_basis",
)
_MEMORY_FIELDS = (
    "scenario",
    "available_memory_gib",
    "gib_per_active_root",
    "fixed_workspace_gib",
    "assumption_basis",
)
_MOMENT_FIELDS = ("variance_high", "variance_low_h", "variance_low_l")


def _reconstruct_spec(record: Mapping[str, Any]) -> PowerEnvelopeSpec:
    counts = record["candidate_and_error_family_counts"]
    design = record["design_constants"]
    grid = record["grid"]
    calibration = record["calibration_requirements"]
    if not all(isinstance(value, Mapping) for value in (counts, design, grid, calibration)):
        raise RivalSchemaError("power envelope nested contracts must be objects")
    require_exact_keys(counts, required=_COUNT_FIELDS, where="power family counts")
    require_exact_keys(design, required=_DESIGN_FIELDS, where="power design constants")
    require_exact_keys(calibration, required=_CALIBRATION_FIELDS, where="calibration requirements")
    moments = grid["population_moment_assumptions"]
    if not isinstance(moments, Mapping):
        raise RivalSchemaError("population moment assumptions must be an object")
    require_exact_keys(moments, required=_MOMENT_FIELDS, where="population moments")

    ranges_raw = record["certified_range_by_stratum"]
    throughputs_raw = record["throughput_assumptions"]
    memories_raw = record["memory_assumptions"]
    if not all(isinstance(value, list) for value in (ranges_raw, throughputs_raw, memories_raw)):
        raise RivalSchemaError("range, throughput, and memory assumptions must be lists")
    for index, item in enumerate(ranges_raw):
        if not isinstance(item, Mapping):
            raise RivalSchemaError(f"certified range {index} must be an object")
        require_exact_keys(item, required=_CERTIFIED_RANGE_FIELDS, where=f"certified range {index}")
    for index, item in enumerate(throughputs_raw):
        if not isinstance(item, Mapping):
            raise RivalSchemaError(f"throughput assumption {index} must be an object")
        require_exact_keys(item, required=_THROUGHPUT_FIELDS, where=f"throughput {index}")
    for index, item in enumerate(memories_raw):
        if not isinstance(item, Mapping):
            raise RivalSchemaError(f"memory assumption {index} must be an object")
        require_exact_keys(item, required=_MEMORY_FIELDS, where=f"memory {index}")
    if calibration["must_be_disjoint_from_coverage"] is not True:
        raise RivalSchemaError("calibration must remain disjoint from coverage")
    if calibration["selected_challenger_pipeline_must_match"] is not True:
        raise RivalSchemaError("calibration must reproduce the selected-challenger pipeline")

    try:
        return PowerEnvelopeSpec(
            envelope_id=record["envelope_id"],
            source_revision=record["source_revision"],
            certified_ranges=tuple(CertifiedStratumRange(**dict(item)) for item in ranges_raw),
            candidate_count=counts["candidate_count"],
            finite_training_family_count=counts["finite_training_family_count"],
            one_seat_family_count=counts["one_seat_family_count"],
            certified_potential_appeals=counts["certified_potential_appeals"],
            selection_units_per_candidate=counts["selection_units_per_candidate"],
            delta_game=design["delta_game"],
            n_h_grid=tuple(grid["n_h"]),
            n_l_grid=tuple(grid["n_l"]),
            covariance_grid=tuple(grid["covariance"]),
            variance_high_assumption=moments["variance_high"],
            variance_low_h_assumption=moments["variance_low_h"],
            variance_low_l_assumption=moments["variance_low_l"],
            target_gap_grid=tuple(grid["target_gap"]),
            activation_frequency_grid=tuple(grid["activation_frequency"]),
            timeout_rate_grid=tuple(grid["timeout_rate"]),
            practical_margin=design["practical_margin"],
            target_confirmed_roots=design["target_confirmed_roots"],
            calibration_root_requirement=calibration["root_count"],
            throughput_assumptions=tuple(
                HypotheticalThroughput(**dict(item)) for item in throughputs_raw
            ),
            memory_assumptions=tuple(MemoryAssumption(**dict(item)) for item in memories_raw),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RivalSchemaError(
            f"power envelope cannot reconstruct its symbolic design: {exc}"
        ) from exc


def validate_power_envelope(record: Mapping[str, Any]) -> None:
    require_schema(record, RIVAL_POWER_ENVELOPE_SCHEMA_ID)
    require_exact_keys(record, required=_ENVELOPE_FIELDS, where="power envelope")
    if record["status"] != NON_FUNDING_STATUS:
        raise RivalSchemaError("pre-measurement power envelope must be non-funding")
    if record["can_fund_program"] is not False or record["can_close_program"] is not False:
        raise RivalSchemaError("symbolic envelope cannot fund or close the program")
    grid = record["grid"]
    if not isinstance(grid, Mapping):
        raise RivalSchemaError("power grid must be an object")
    require_exact_keys(grid, required=_GRID_FIELDS, where="power grid")
    if grid["beta_cv"] != "DERIVED_PER_ROW_FROM_COVARIANCE_AND_ALLOCATION":
        raise RivalSchemaError("symbolic beta must be derived from covariance and allocation")
    measured = record["measured_cost_fields"]
    if not isinstance(measured, Mapping) or not measured:
        raise RivalSchemaError("measured_cost_fields must be a non-empty object")
    require_exact_keys(measured, required=_MEASURED_FIELDS, where="measured cost fields")
    if any(value != UNRESOLVED for value in measured.values()):
        raise RivalSchemaError("all pre-P2 measured-cost fields must remain UNRESOLVED")
    rows = record["rows"]
    if not isinstance(rows, list) or not rows:
        raise RivalSchemaError("power envelope needs a non-empty symbolic grid")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise RivalSchemaError(f"power row {index} must be an object")
        require_exact_keys(row, required=_ROW_FIELDS, where=f"power row {index}")
        if row["decision_grade_gpu_hours"] != UNRESOLVED:
            raise RivalSchemaError("symbolic grid cannot contain decision-grade GPU hours")
        require_sha256(row["bound_certificate_sha256"], "bound_certificate_sha256")
        for field in (
            "beta_cv_derived_from_covariance",
            "covariance_assumption",
            "hypothetical_estimator_variance",
            "target_gap_assumption",
            "activation_frequency_assumption",
            "timeout_rate_assumption",
            "roots_per_hour_assumption",
            "hoeffding_half_width",
            "hypothetical_root_work_seconds",
        ):
            require_finite(row[field], f"row.{field}")
        if not isinstance(row["symbolically_resolves_margin"], bool):
            raise RivalSchemaError("symbolically_resolves_margin must be boolean")
        for field in (
            "selection_units_per_attempted_root",
            "attempted_units_per_root",
            "hypothetical_memory_capacity_roots",
            "hypothetical_effective_parallel_roots",
            "terminal_pairs_per_attempted_root",
        ):
            value = row[field]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise RivalSchemaError(f"row.{field} must be a positive integer")
        for field in (
            "required_attempted_roots_assuming_independent_timeouts",
            "hypothetical_hours_not_decision_grade",
        ):
            value = row[field]
            if row["symbolically_resolves_margin"]:
                require_finite(value, f"row.{field}")
            elif value != NO_FINITE_HOURS:
                raise RivalSchemaError(
                    f"non-resolving symbolic row must mark {field} as no finite work"
                )
    verify_content_hash(record)
    try:
        recomputed = build_power_envelope(_reconstruct_spec(record))
    except PowerEnvelopeError as exc:
        raise RivalSchemaError(f"invalid symbolic power design: {exc}") from exc
    if dict(record) != recomputed:
        raise RivalSchemaError("power envelope rows do not reproduce from their embedded design")


def write_power_envelope(path: str | Path, spec: PowerEnvelopeSpec) -> dict[str, Any]:
    envelope = build_power_envelope(spec)
    validate_power_envelope(envelope)
    write_new_canonical_json(path, envelope)
    return envelope
