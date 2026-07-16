"""Rules-agnostic Rival summaries with source-game clustered uncertainty."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from statistics import fmean, stdev

from .appeals import AppealDecision, HighFidelityAppealDecision
from .cohorts import AllocationRegistry, CohortError


class AnalysisError(ValueError):
    """Raised when an analysis would use the wrong statistical unit."""


@dataclass(frozen=True)
class RootMeasurement:
    root_id: str
    source_game_id: str
    value: float
    natural_frequency_weight: float = 1.0


@dataclass(frozen=True)
class ClusteredSummary:
    mean: float
    standard_error: float | None
    source_game_count: int
    root_count: int
    unit: str = "source_game_cluster"


def source_game_clustered_summary(
    rows: Iterable[RootMeasurement],
    *,
    allocation_registry: AllocationRegistry,
    expected_allocation_registry_identity: str,
    family_kind: str,
) -> ClusteredSummary:
    """Summarize one registry-complete family at the source-game level.

    Decision-grade use requires an externally byte-pinned allocation registry,
    its explicit expected identity, and an explicit error-family kind.  Rows
    must cover that family's registered root universe exactly and must preserve
    every registered root-to-source-game assignment.  Natural-frequency
    weights operate within a source game; complete registered games retain
    equal top-level weight.
    """
    if not isinstance(allocation_registry, AllocationRegistry):
        raise AnalysisError("clustered analysis requires a typed AllocationRegistry")
    try:
        eligible_root_ids = allocation_registry.eligible_root_ids_for_family(
            family_kind,
            expected_allocation_registry_identity=expected_allocation_registry_identity,
        )
    except CohortError as exc:
        raise AnalysisError(str(exc)) from exc
    if not eligible_root_ids:
        raise AnalysisError(f"allocation registry has no eligible roots for family {family_kind!r}")
    assignment_by_root = {row.root_id: row for row in allocation_registry.root_assignments}
    expected_source_game_by_root = {
        root_id: assignment_by_root[root_id].source_game_id for root_id in eligible_root_ids
    }

    records = tuple(rows)
    if not records:
        raise AnalysisError("at least one root measurement is required")
    seen_roots: set[str] = set()
    clusters: dict[str, list[RootMeasurement]] = {}
    for row in records:
        if not isinstance(row, RootMeasurement):
            raise AnalysisError("clustered analysis requires RootMeasurement records")
        if (
            not isinstance(row.root_id, str)
            or not row.root_id
            or row.root_id != row.root_id.strip()
            or row.root_id in seen_roots
        ):
            raise AnalysisError(f"empty or duplicate root_id {row.root_id!r}")
        if (
            not isinstance(row.source_game_id, str)
            or not row.source_game_id
            or row.source_game_id != row.source_game_id.strip()
        ):
            raise AnalysisError("source_game_id must be non-empty")
        if (
            isinstance(row.value, bool)
            or not isinstance(row.value, (int, float))
            or not math.isfinite(row.value)
        ):
            raise AnalysisError("root values must be finite")
        if (
            isinstance(row.natural_frequency_weight, bool)
            or not isinstance(row.natural_frequency_weight, (int, float))
            or not math.isfinite(row.natural_frequency_weight)
            or row.natural_frequency_weight <= 0.0
        ):
            raise AnalysisError("natural-frequency weights must be finite and positive")
        seen_roots.add(row.root_id)
        clusters.setdefault(row.source_game_id, []).append(row)
    expected_roots = set(eligible_root_ids)
    missing_roots = sorted(expected_roots - seen_roots)
    extra_roots = sorted(seen_roots - expected_roots)
    if missing_roots or extra_roots:
        details: list[str] = []
        if missing_roots:
            details.append(f"missing={missing_roots}")
        if extra_roots:
            details.append(f"extra={extra_roots}")
        raise AnalysisError(
            "root measurements do not exactly cover the pinned family registry: "
            + ", ".join(details)
        )
    for row in records:
        expected_source_game_id = expected_source_game_by_root[row.root_id]
        if row.source_game_id != expected_source_game_id:
            raise AnalysisError(
                "root measurement source_game_id differs from the pinned allocation registry: "
                f"root_id={row.root_id!r}, observed={row.source_game_id!r}, "
                f"expected={expected_source_game_id!r}"
            )
    cluster_means: list[float] = []
    for cluster in clusters.values():
        denominator = sum(row.natural_frequency_weight for row in cluster)
        cluster_means.append(
            sum(row.value * row.natural_frequency_weight for row in cluster) / denominator
        )
    mean = fmean(cluster_means)
    standard_error = (
        stdev(cluster_means) / math.sqrt(len(cluster_means)) if len(cluster_means) >= 2 else None
    )
    return ClusteredSummary(mean, standard_error, len(clusters), len(records))


def iid_root_standard_error_for_diagnostic(rows: Iterable[RootMeasurement]) -> float | None:
    """Return the forbidden iid-root comparison for explicit diagnostics only."""
    records = tuple(rows)
    for row in records:
        if (
            not isinstance(row, RootMeasurement)
            or isinstance(row.value, bool)
            or not isinstance(row.value, (int, float))
            or not math.isfinite(row.value)
        ):
            raise AnalysisError("iid diagnostic requires finite RootMeasurement records")
    if len(records) < 2:
        return None
    return stdev(row.value for row in records) / math.sqrt(len(records))


@dataclass(frozen=True)
class AppealOperationalSummary:
    root_count: int
    confirmed_labels: int
    no_labels: int
    attempted_terminal_units: int
    completed_terminal_units: int
    timeouts: int
    invalid: int
    completion_rate: float
    activation_rate: float


def summarize_appeals(
    decisions: Iterable[AppealDecision | HighFidelityAppealDecision],
) -> AppealOperationalSummary:
    rows = tuple(decisions)
    if any(not isinstance(row, (AppealDecision, HighFidelityAppealDecision)) for row in rows):
        raise AnalysisError("appeal summary requires typed appeal-decision records")
    attempted = sum(row.operational.attempted_total for row in rows)
    completed = sum(row.operational.completed_total for row in rows)
    confirmed = sum(row.preference is not None for row in rows)
    timeouts = sum(row.operational.timeouts for row in rows)
    invalid = sum(row.operational.invalid for row in rows)
    return AppealOperationalSummary(
        root_count=len(rows),
        confirmed_labels=confirmed,
        no_labels=len(rows) - confirmed,
        attempted_terminal_units=attempted,
        completed_terminal_units=completed,
        timeouts=timeouts,
        invalid=invalid,
        completion_rate=completed / attempted if attempted else 0.0,
        activation_rate=confirmed / len(rows) if rows else 0.0,
    )
