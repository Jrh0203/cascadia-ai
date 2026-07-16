"""Pinned root-cohort and complete-game seed-allocation contracts.

The registry stores commitments, never raw scientific seeds.  A manifest-only
collection check remains useful for structural validation, but it deliberately
does not claim seed disjointness.  A claim requires both a canonical,
byte-pinned ``cascadiav3.rival_allocation_registry.v1`` artifact and exact
runtime openings of every registered commitment.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .manifest import (
    COMPLETE_GAME_SEED_ROLES,
    PUBLIC_ROOT_ID_PREFIX,
    ROOT_COHORT_ROLES,
    RootManifest,
    require_validated_root_manifest,
    validate_root_manifest,
)
from .schema import (
    RIVAL_ALLOCATION_REGISTRY_SCHEMA_ID,
    RivalSchemaError,
    _issue_validation_capability,
    _require_validation_capability,
    read_pinned_canonical_json_object,
    read_strict_json_value,
    require_exact_keys,
    require_nonempty_string,
    require_schema,
    require_sha256,
    sha256_hex,
    verify_content_hash,
)

if TYPE_CHECKING:
    from .coverage import ErrorFamilyLedger, PotentialRootCensus

SEED_COMMITMENT_PREFIX = "cascadiav3.rival_seed_commitment.v1:sha256:"
SEED_COMMITMENT_PAYLOAD_SCHEMA_ID = "cascadiav3.rival_seed_commitment_payload.v1"
ROOT_SOURCE_SET_SCHEMA_ID = "cascadiav3.rival_root_source_set.v1"
MAX_SEED_U64 = (1 << 64) - 1
ERROR_FAMILY_ROOT_COHORT_ROLE = {
    "finite_training_corpus": "relabel_selection",
    "one_seat_instrument": "shadow_one_seat",
}


class CohortError(ValueError):
    """Raised when root or complete-game allocation axes are invalid."""


def _nonempty(value: object, field: str) -> str:
    try:
        return require_nonempty_string(value, field)
    except RivalSchemaError as exc:
        raise CohortError(str(exc)) from exc


def _public_root_id(value: object, field: str) -> str:
    text = _nonempty(value, field)
    if not text.startswith(PUBLIC_ROOT_ID_PREFIX):
        raise CohortError(f"{field} must use {PUBLIC_ROOT_ID_PREFIX!r}")
    try:
        require_sha256(text.removeprefix(PUBLIC_ROOT_ID_PREFIX), field)
    except RivalSchemaError as exc:
        raise CohortError(str(exc)) from exc
    return text


def _seed_commitment(value: object, field: str) -> str:
    text = _nonempty(value, field)
    if not text.startswith(SEED_COMMITMENT_PREFIX):
        raise CohortError(f"{field} must use {SEED_COMMITMENT_PREFIX!r}")
    try:
        require_sha256(text.removeprefix(SEED_COMMITMENT_PREFIX), field)
    except RivalSchemaError as exc:
        raise CohortError(str(exc)) from exc
    return text


def _qualified_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise CohortError(f"{field} must use the 'sha256:' wire")
    try:
        return "sha256:" + require_sha256(value, field)
    except RivalSchemaError as exc:
        raise CohortError(str(exc)) from exc


def _seed_u64(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CohortError(f"{field} must be an unsigned 64-bit integer")
    if not 0 <= value <= MAX_SEED_U64:
        raise CohortError(f"{field} must be an unsigned 64-bit integer")
    return value


def seed_commitment_for_value(seed_u64: int) -> str:
    """Commit one realized u64 seed using the v1 domain-separated encoding."""

    value = _seed_u64(seed_u64, "seed_u64")
    return SEED_COMMITMENT_PREFIX + sha256_hex(
        {
            "schema_id": SEED_COMMITMENT_PAYLOAD_SCHEMA_ID,
            # A decimal string keeps the full u64 meaning independent of a
            # JSON consumer's numeric precision.
            "seed_u64_decimal": str(value),
        }
    )


@dataclass(frozen=True)
class RootAssignment:
    root_id: str
    source_game_id: str
    cohort_role: str
    root_seed_commitment: str

    def __post_init__(self) -> None:
        _public_root_id(self.root_id, "root_id")
        _nonempty(self.source_game_id, "source_game_id")
        _seed_commitment(self.root_seed_commitment, "root_seed_commitment")
        if self.cohort_role not in ROOT_COHORT_ROLES:
            raise CohortError(f"invalid root cohort role: {self.cohort_role!r}")


@dataclass(frozen=True)
class CompleteGameAssignment:
    seed_commitment: str
    seed_role: str

    def __post_init__(self) -> None:
        _seed_commitment(self.seed_commitment, "seed_commitment")
        if self.seed_role not in COMPLETE_GAME_SEED_ROLES:
            raise CohortError(f"invalid complete-game seed role: {self.seed_role!r}")


@dataclass(frozen=True)
class RootSeedOpening:
    """Ephemeral per-run opening; never serialized into the registry."""

    root_id: str
    seed_u64: int = dataclass_field(repr=False)

    def __post_init__(self) -> None:
        _public_root_id(self.root_id, "root_id")
        _seed_u64(self.seed_u64, "seed_u64")


@dataclass(frozen=True)
class CompleteGameSeedOpening:
    """Ephemeral per-run opening for one promotion or target game."""

    seed_role: str
    seed_u64: int = dataclass_field(repr=False)

    def __post_init__(self) -> None:
        if self.seed_role not in COMPLETE_GAME_SEED_ROLES:
            raise CohortError(f"invalid complete-game seed role: {self.seed_role!r}")
        _seed_u64(self.seed_u64, "seed_u64")


@dataclass(frozen=True)
class AllocationSummary:
    root_count: int
    source_game_count: int
    complete_game_count: int
    root_counts_by_role: dict[str, int]
    game_counts_by_role: dict[str, int]


@dataclass(frozen=True)
class AllocationRegistry:
    """Validated immutable allocation registry; commitments are not raw seeds."""

    registry_id: str
    root_source_set_sha256: str
    root_assignments: tuple[RootAssignment, ...]
    complete_game_assignments: tuple[CompleteGameAssignment, ...]
    content_sha256: str
    _validation_capability: object | None = dataclass_field(
        default=None,
        repr=False,
        compare=False,
    )
    _external_pin_capability: object | None = dataclass_field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self._validation_capability is not None:
            try:
                _require_validation_capability(
                    self._validation_capability,
                    artifact_kind="AllocationRegistry",
                    content_sha256=_registry_runtime_fingerprint(self),
                )
            except RivalSchemaError as exc:
                raise CohortError(str(exc)) from exc
        if self._external_pin_capability is not None:
            try:
                _require_validation_capability(
                    self._external_pin_capability,
                    artifact_kind="ExternallyPinnedAllocationRegistry",
                    content_sha256=_registry_runtime_fingerprint(self),
                )
            except RivalSchemaError as exc:
                raise CohortError(str(exc)) from exc

    @property
    def identity(self) -> str:
        return "sha256:" + self.content_sha256

    def require_validated_artifact(self) -> None:
        try:
            _require_validation_capability(
                self._validation_capability,
                artifact_kind="AllocationRegistry",
                content_sha256=_registry_runtime_fingerprint(self),
            )
        except RivalSchemaError as exc:
            raise CohortError(str(exc)) from exc

    def require_externally_pinned_artifact(
        self,
        *,
        expected_allocation_registry_identity: str,
    ) -> None:
        """Require both semantic validation and canonical external byte pins."""

        _require_expected_registry(self, expected_allocation_registry_identity)
        try:
            _require_validation_capability(
                self._external_pin_capability,
                artifact_kind="ExternallyPinnedAllocationRegistry",
                content_sha256=_registry_runtime_fingerprint(self),
            )
        except RivalSchemaError as exc:
            raise CohortError(str(exc)) from exc

    def eligible_root_ids_for_family(
        self,
        family_kind: str,
        *,
        expected_allocation_registry_identity: str,
    ) -> tuple[str, ...]:
        """Return the exact pinned root universe for one error-family kind."""

        self.require_externally_pinned_artifact(
            expected_allocation_registry_identity=expected_allocation_registry_identity
        )
        try:
            cohort_role = ERROR_FAMILY_ROOT_COHORT_ROLE[family_kind]
        except KeyError as exc:
            raise CohortError(f"unknown error family kind {family_kind!r}") from exc
        return tuple(row.root_id for row in self.root_assignments if row.cohort_role == cohort_role)


@dataclass(frozen=True)
class ManifestCollectionSummary:
    root_count: int
    source_game_count: int
    root_source_set_sha256: str
    commitment_uniqueness_validated: bool
    allocation_registry_identity: str | None
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
                    artifact_kind="ManifestCollectionSummary",
                    content_sha256=_manifest_collection_runtime_fingerprint(self),
                )
            except RivalSchemaError as exc:
                raise CohortError(str(exc)) from exc

    def require_validated_registry_join(self, registry: AllocationRegistry) -> None:
        try:
            _require_validation_capability(
                self._validation_capability,
                artifact_kind="ManifestCollectionSummary",
                content_sha256=_manifest_collection_runtime_fingerprint(self),
            )
        except RivalSchemaError as exc:
            raise CohortError(str(exc)) from exc
        if not self.commitment_uniqueness_validated:
            raise CohortError("seed realization requires a validated manifest-to-registry join")
        if self.allocation_registry_identity != registry.identity:
            raise CohortError("manifest collection is bound to a different registry")
        if self.root_source_set_sha256 != registry.root_source_set_sha256:
            raise CohortError("manifest collection is bound to a different root-source set")


@dataclass(frozen=True)
class SeedRealizationSummary:
    allocation_registry_identity: str
    opened_root_count: int
    opened_complete_game_count: int
    realized_seed_disjointness_validated: bool = True


_REGISTRY_FIELDS = (
    "schema_id",
    "registry_id",
    "root_source_set_sha256",
    "root_assignments",
    "complete_game_assignments",
    "content_sha256",
)
_ROOT_ASSIGNMENT_FIELDS = (
    "root_id",
    "source_game_id",
    "cohort_role",
    "root_seed_commitment",
)
_COMPLETE_GAME_ASSIGNMENT_FIELDS = ("seed_commitment", "seed_role")


def _registry_runtime_fingerprint(registry: AllocationRegistry) -> str:
    payload = asdict(registry)
    payload.pop("_validation_capability", None)
    payload.pop("_external_pin_capability", None)
    return sha256_hex(
        {
            "schema_id": "cascadiav3.rival_validated_allocation_registry_runtime.v1",
            "fields": payload,
        }
    )


def _manifest_collection_runtime_fingerprint(summary: ManifestCollectionSummary) -> str:
    payload = asdict(summary)
    payload.pop("_validation_capability", None)
    return sha256_hex(
        {
            "schema_id": "cascadiav3.rival_validated_manifest_collection_runtime.v1",
            "fields": payload,
        }
    )


def root_source_set_identity(
    rows: Iterable[RootAssignment | RootManifest],
) -> str:
    """Hash exact preregistrable root/source/cohort projections."""

    values = tuple(rows)
    if not values:
        raise CohortError("root source set must be non-empty")
    entries: list[dict[str, str]] = []
    root_ids: set[str] = set()
    for row in values:
        if isinstance(row, RootManifest):
            try:
                require_validated_root_manifest(row)
            except RivalSchemaError as exc:
                raise CohortError(str(exc)) from exc
            root_id = row.root_id
            source_game_id = row.source_game_id
            cohort_role = row.root_cohort_role
        elif isinstance(row, RootAssignment):
            root_id = row.root_id
            source_game_id = row.source_game_id
            cohort_role = row.cohort_role
        else:
            raise CohortError(
                "root source set requires RootAssignment or validated RootManifest rows"
            )
        if root_id in root_ids:
            raise CohortError(f"root source set repeats root_id {root_id}")
        root_ids.add(root_id)
        entries.append(
            {
                "root_id": root_id,
                "source_game_id": source_game_id,
                "cohort_role": cohort_role,
            }
        )
    entries.sort(key=lambda row: row["root_id"])
    return "sha256:" + sha256_hex(
        {
            "schema_id": ROOT_SOURCE_SET_SCHEMA_ID,
            "roots": entries,
        }
    )


def validate_allocations(
    roots: Iterable[RootAssignment], games: Iterable[CompleteGameAssignment]
) -> AllocationSummary:
    """Validate uniqueness and cohort isolation for typed assignments.

    ``source_game_id`` is a cluster identity, not a row identity, so multiple
    roots from the same source game are valid.  The scientifically critical
    restriction is that coefficient calibration and untouched coverage never
    share a source-game cluster.
    """

    root_rows = tuple(roots)
    game_rows = tuple(games)
    root_ids: set[str] = set()
    commitments: set[str] = set()
    source_roles: dict[str, set[str]] = {}
    root_counts = {role: 0 for role in sorted(ROOT_COHORT_ROLES)}
    for row in root_rows:
        if not isinstance(row, RootAssignment):
            raise CohortError("root allocations must contain RootAssignment records")
        if row.root_id in root_ids:
            raise CohortError(f"root_id allocated more than once: {row.root_id}")
        if row.root_seed_commitment in commitments:
            raise CohortError(
                f"seed commitment reused across allocation registry: {row.root_seed_commitment}"
            )
        root_ids.add(row.root_id)
        commitments.add(row.root_seed_commitment)
        source_roles.setdefault(row.source_game_id, set()).add(row.cohort_role)
        root_counts[row.cohort_role] += 1

    for source_game_id, roles in source_roles.items():
        if {"coefficient_calibration", "untouched_coverage"} <= roles:
            raise CohortError(
                f"coefficient calibration and untouched coverage share source game {source_game_id}"
            )

    game_counts = {role: 0 for role in sorted(COMPLETE_GAME_SEED_ROLES)}
    for row in game_rows:
        if not isinstance(row, CompleteGameAssignment):
            raise CohortError(
                "complete-game allocations must contain CompleteGameAssignment records"
            )
        if row.seed_commitment in commitments:
            raise CohortError(
                f"seed commitment reused across allocation registry: {row.seed_commitment}"
            )
        commitments.add(row.seed_commitment)
        game_counts[row.seed_role] += 1

    return AllocationSummary(
        root_count=len(root_rows),
        source_game_count=len(source_roles),
        complete_game_count=len(game_rows),
        root_counts_by_role=root_counts,
        game_counts_by_role=game_counts,
    )


def validate_allocation_registry(
    record: Mapping[str, Any],
    *,
    expected_content_sha256: str,
) -> AllocationRegistry:
    """Validate an exact allocation registry against its preregistered pin."""

    require_schema(record, RIVAL_ALLOCATION_REGISTRY_SCHEMA_ID)
    require_exact_keys(record, required=_REGISTRY_FIELDS, where="allocation registry")
    content_sha256 = verify_content_hash(record)
    expected = require_sha256(expected_content_sha256, "expected_content_sha256")
    if content_sha256 != expected:
        raise RivalSchemaError("allocation registry differs from its preregistered pin")

    roots_raw = record["root_assignments"]
    if not isinstance(roots_raw, list) or not roots_raw:
        raise RivalSchemaError("root_assignments must be a non-empty ordered list")
    roots: list[RootAssignment] = []
    for index, raw in enumerate(roots_raw):
        if not isinstance(raw, Mapping):
            raise RivalSchemaError(f"root_assignments[{index}] must be an object")
        require_exact_keys(
            raw,
            required=_ROOT_ASSIGNMENT_FIELDS,
            where=f"root_assignments[{index}]",
        )
        try:
            roots.append(
                RootAssignment(
                    root_id=raw["root_id"],
                    source_game_id=raw["source_game_id"],
                    cohort_role=raw["cohort_role"],
                    root_seed_commitment=raw["root_seed_commitment"],
                )
            )
        except CohortError as exc:
            raise RivalSchemaError(f"invalid root assignment {index}: {exc}") from exc

    games_raw = record["complete_game_assignments"]
    if not isinstance(games_raw, list):
        raise RivalSchemaError("complete_game_assignments must be an ordered list")
    games: list[CompleteGameAssignment] = []
    for index, raw in enumerate(games_raw):
        if not isinstance(raw, Mapping):
            raise RivalSchemaError(f"complete_game_assignments[{index}] must be an object")
        require_exact_keys(
            raw,
            required=_COMPLETE_GAME_ASSIGNMENT_FIELDS,
            where=f"complete_game_assignments[{index}]",
        )
        try:
            games.append(
                CompleteGameAssignment(
                    seed_commitment=raw["seed_commitment"],
                    seed_role=raw["seed_role"],
                )
            )
        except CohortError as exc:
            raise RivalSchemaError(f"invalid complete-game assignment {index}: {exc}") from exc

    root_tuple = tuple(roots)
    canonical_roots = tuple(
        sorted(
            root_tuple,
            key=lambda row: (
                row.root_id,
                row.source_game_id,
                row.cohort_role,
                row.root_seed_commitment,
            ),
        )
    )
    if root_tuple != canonical_roots:
        raise RivalSchemaError("root_assignments must be in canonical lexical order")
    game_tuple = tuple(games)
    canonical_games = tuple(
        sorted(game_tuple, key=lambda row: (row.seed_commitment, row.seed_role))
    )
    if game_tuple != canonical_games:
        raise RivalSchemaError("complete_game_assignments must be in canonical lexical order")
    try:
        validate_allocations(root_tuple, game_tuple)
    except CohortError as exc:
        raise RivalSchemaError(str(exc)) from exc

    try:
        source_set_identity = _qualified_sha256(
            record["root_source_set_sha256"],
            "root_source_set_sha256",
        )
    except CohortError as exc:
        raise RivalSchemaError(str(exc)) from exc
    if source_set_identity != root_source_set_identity(root_tuple):
        raise RivalSchemaError(
            "root_source_set_sha256 does not bind the exact root/source/cohort projection"
        )
    registry = AllocationRegistry(
        registry_id=require_nonempty_string(record["registry_id"], "registry_id"),
        root_source_set_sha256=source_set_identity,
        root_assignments=root_tuple,
        complete_game_assignments=game_tuple,
        content_sha256=content_sha256,
    )
    return replace(
        registry,
        _validation_capability=_issue_validation_capability(
            "AllocationRegistry",
            _registry_runtime_fingerprint(registry),
        ),
    )


def _expected_registry_identity(value: object) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise CohortError("expected_allocation_registry_identity must use the 'sha256:' wire")
    try:
        return "sha256:" + require_sha256(
            value,
            "expected_allocation_registry_identity",
        )
    except RivalSchemaError as exc:
        raise CohortError(str(exc)) from exc


def _require_expected_registry(
    registry: AllocationRegistry,
    expected_allocation_registry_identity: str,
) -> None:
    if not isinstance(registry, AllocationRegistry):
        raise CohortError("allocation registry must be an AllocationRegistry")
    registry.require_validated_artifact()
    expected = _expected_registry_identity(expected_allocation_registry_identity)
    if registry.identity != expected:
        raise CohortError(
            "allocation registry differs from the explicit cross-artifact identity pin"
        )


def validate_seed_realizations(
    allocation_registry: AllocationRegistry,
    *,
    expected_allocation_registry_identity: str,
    manifest_collection: ManifestCollectionSummary,
    root_openings: Iterable[RootSeedOpening],
    complete_game_openings: Iterable[CompleteGameSeedOpening],
) -> SeedRealizationSummary:
    """Open every registered commitment and prove realized-seed disjointness.

    The summary contains no raw seeds.  Callers must run this check on the
    actual u64 values immediately before their use; commitment uniqueness by
    itself is only preregistration evidence.
    """

    allocation_registry.require_externally_pinned_artifact(
        expected_allocation_registry_identity=expected_allocation_registry_identity,
    )
    if not isinstance(manifest_collection, ManifestCollectionSummary):
        raise CohortError("seed realization requires a validated ManifestCollectionSummary")
    manifest_collection.require_validated_registry_join(allocation_registry)
    roots = tuple(root_openings)
    games = tuple(complete_game_openings)
    if not all(isinstance(row, RootSeedOpening) for row in roots):
        raise CohortError("root openings must contain RootSeedOpening records")
    if not all(isinstance(row, CompleteGameSeedOpening) for row in games):
        raise CohortError("complete-game openings must contain CompleteGameSeedOpening records")

    root_by_id: dict[str, RootSeedOpening] = {}
    realized_values: set[int] = set()
    for row in roots:
        if row.root_id in root_by_id:
            raise CohortError(f"root seed opened more than once: {row.root_id}")
        if row.seed_u64 in realized_values:
            raise CohortError(f"realized seed reused across allocation axes: {row.seed_u64}")
        root_by_id[row.root_id] = row
        realized_values.add(row.seed_u64)

    registered_roots = {row.root_id: row for row in allocation_registry.root_assignments}
    if set(root_by_id) != set(registered_roots):
        missing = sorted(set(registered_roots) - set(root_by_id))
        extra = sorted(set(root_by_id) - set(registered_roots))
        raise CohortError(
            "root seed openings must exactly equal the allocation registry: "
            f"missing={missing}, extra={extra}"
        )
    for root_id, opening in root_by_id.items():
        if (
            seed_commitment_for_value(opening.seed_u64)
            != registered_roots[root_id].root_seed_commitment
        ):
            raise CohortError(
                f"realized root seed does not open its registered commitment: {root_id}"
            )

    opened_games: set[tuple[str, str]] = set()
    for row in games:
        if row.seed_u64 in realized_values:
            raise CohortError(f"realized seed reused across allocation axes: {row.seed_u64}")
        realized_values.add(row.seed_u64)
        opened = (seed_commitment_for_value(row.seed_u64), row.seed_role)
        if opened in opened_games:
            raise CohortError("complete-game seed opening repeated")
        opened_games.add(opened)
    registered_games = {
        (row.seed_commitment, row.seed_role)
        for row in allocation_registry.complete_game_assignments
    }
    if opened_games != registered_games:
        missing = sorted(registered_games - opened_games)
        extra = sorted(opened_games - registered_games)
        raise CohortError(
            "complete-game seed openings must exactly equal the allocation registry: "
            f"missing={missing}, extra={extra}"
        )
    return SeedRealizationSummary(
        allocation_registry_identity=allocation_registry.identity,
        opened_root_count=len(roots),
        opened_complete_game_count=len(games),
    )


def load_allocation_registry(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_content_sha256: str,
) -> AllocationRegistry:
    """Load one canonical, single-link, byte-pinned allocation registry."""

    record = read_pinned_canonical_json_object(
        path,
        expected_file_sha256=expected_file_sha256,
        field="allocation registry",
    )
    registry = validate_allocation_registry(
        record,
        expected_content_sha256=expected_content_sha256,
    )
    return replace(
        registry,
        _external_pin_capability=_issue_validation_capability(
            "ExternallyPinnedAllocationRegistry",
            _registry_runtime_fingerprint(registry),
        ),
    )


def _validate_structural_manifest_axes(manifests: Sequence[RootManifest]) -> None:
    root_ids: set[str] = set()
    source_roles: dict[str, set[str]] = {}
    for manifest in manifests:
        if manifest.root_id in root_ids:
            raise CohortError(f"root_id allocated more than once: {manifest.root_id}")
        root_ids.add(manifest.root_id)
        source_roles.setdefault(manifest.source_game_id, set()).add(manifest.root_cohort_role)
    for source_game_id, roles in source_roles.items():
        if {"coefficient_calibration", "untouched_coverage"} <= roles:
            raise CohortError(
                f"coefficient calibration and untouched coverage share source game {source_game_id}"
            )


def validate_manifest_collection(
    records: Sequence[Mapping[str, object]],
    *,
    allocation_registry: AllocationRegistry | None = None,
    expected_allocation_registry_identity: str | None = None,
    required_panels: tuple[str, ...] | None = None,
    potential_root_census: PotentialRootCensus | None = None,
    error_family: ErrorFamilyLedger | None = None,
) -> ManifestCollectionSummary:
    """Validate manifests and optionally prove pinned commitment uniqueness.

    A registry does not prove that actual seeds opened its commitments; use
    :func:`validate_seed_realizations` immediately before execution.  This
    function never derives or invents a seed token from a root identity.
    """

    if not records:
        raise CohortError("manifest collection must be non-empty")
    manifests = [validate_root_manifest(record) for record in records]
    _validate_structural_manifest_axes(manifests)

    registry_identity: str | None = None
    if allocation_registry is not None:
        if expected_allocation_registry_identity is None:
            raise CohortError(
                "allocation registry joins require an explicit cross-artifact identity pin"
            )
        allocation_registry.require_externally_pinned_artifact(
            expected_allocation_registry_identity=expected_allocation_registry_identity,
        )
        if allocation_registry.root_source_set_sha256 != root_source_set_identity(manifests):
            raise CohortError("allocation registry does not bind the exact root-source set")
        registry_by_root = {row.root_id: row for row in allocation_registry.root_assignments}
        manifest_by_root = {manifest.root_id: manifest for manifest in manifests}
        if set(registry_by_root) != set(manifest_by_root):
            missing = sorted(set(manifest_by_root) - set(registry_by_root))
            extra = sorted(set(registry_by_root) - set(manifest_by_root))
            raise CohortError(
                "allocation registry root set must exactly equal the manifest collection: "
                f"missing={missing}, extra={extra}"
            )
        for root_id in sorted(manifest_by_root):
            manifest = manifest_by_root[root_id]
            assignment = registry_by_root[root_id]
            if assignment.source_game_id != manifest.source_game_id:
                raise CohortError(f"allocation registry source_game_id mismatch for root {root_id}")
            if assignment.cohort_role != manifest.root_cohort_role:
                raise CohortError(f"allocation registry cohort_role mismatch for root {root_id}")
        registry_identity = allocation_registry.identity
    elif expected_allocation_registry_identity is not None:
        raise CohortError("expected allocation registry identity was supplied without a registry")

    if (potential_root_census is None) != (error_family is None):
        raise CohortError(
            "census-complete collection validation requires both census and error ledger"
        )
    if potential_root_census is not None and error_family is not None:
        if allocation_registry is None:
            raise CohortError(
                "census-complete collection validation requires its pinned allocation registry"
            )
        try:
            potential_root_census.require_validated_artifact()
            error_family.require_validated_artifact(census=potential_root_census)
        except ValueError as exc:
            raise CohortError(str(exc)) from exc
        if potential_root_census.allocation_registry_identity != allocation_registry.identity:
            raise CohortError("potential-root census is bound to a different registry")
        if (
            potential_root_census.source_root_set_sha256
            != allocation_registry.root_source_set_sha256
        ):
            raise CohortError("potential-root census is bound to a different root-source set")
        try:
            family_roots = allocation_registry.eligible_root_ids_for_family(
                potential_root_census.family_kind,
                expected_allocation_registry_identity=allocation_registry.identity,
            )
        except CohortError as exc:
            raise CohortError(str(exc)) from exc
        if potential_root_census.eligible_root_ids != family_roots:
            raise CohortError(
                "potential-root census differs from the registry-derived family universe"
            )
        observed_roots = tuple(sorted(manifest.root_id for manifest in manifests))
        if observed_roots != potential_root_census.eligible_root_ids:
            raise CohortError(
                "manifest collection must exactly cover every root in the potential-root census"
            )
        if any(manifest.error_ledger_identity != error_family.identity for manifest in manifests):
            raise CohortError(
                "every manifest in one potential-root census must pin the same error ledger"
            )
    if required_panels is not None:
        mismatches = [
            manifest.root_id
            for manifest in manifests
            if manifest.required_panels != required_panels
        ]
        if mismatches:
            raise CohortError(
                f"manifest panel mode mismatch for roots {mismatches}; expected {required_panels}"
            )
    summary = ManifestCollectionSummary(
        root_count=len(manifests),
        source_game_count=len({manifest.source_game_id for manifest in manifests}),
        root_source_set_sha256=root_source_set_identity(manifests),
        commitment_uniqueness_validated=allocation_registry is not None,
        allocation_registry_identity=registry_identity,
    )
    return replace(
        summary,
        _validation_capability=_issue_validation_capability(
            "ManifestCollectionSummary",
            _manifest_collection_runtime_fingerprint(summary),
        ),
    )


def _parse_seed_u64(text: str, field: str) -> int:
    if (
        not text
        or (text != "0" and text.startswith("0"))
        or not text.isascii()
        or not text.isdigit()
    ):
        raise CohortError(f"{field} must use canonical unsigned decimal notation")
    return _seed_u64(int(text), field)


def _parse_root_opening(text: str) -> RootSeedOpening:
    root_id, separator, seed = text.rpartition("=")
    if not separator:
        raise CohortError("root seed opening must use ROOT_ID=SEED_U64")
    return RootSeedOpening(
        _public_root_id(root_id, "root opening root_id"),
        _parse_seed_u64(seed, "root opening seed_u64"),
    )


def _parse_game_opening(text: str) -> CompleteGameSeedOpening:
    role, separator, seed = text.partition("=")
    if not separator:
        raise CohortError("complete-game seed opening must use ROLE=SEED_U64")
    return CompleteGameSeedOpening(
        role,
        _parse_seed_u64(seed, "complete-game opening seed_u64"),
    )


def _load_manifest_records(path: Path) -> list[Mapping[str, object]]:
    try:
        value = read_strict_json_value(path, field="manifest collection")
    except RivalSchemaError as exc:
        raise CohortError(f"could not read {path}: {exc}") from exc
    records = value.get("root_manifests", [value]) if isinstance(value, dict) else value
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise CohortError("manifest input must be an object or list of objects")
    return records


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--manifest", type=Path, required=True)
    validate_parser.add_argument("--require-panels", default="S,H,L")
    validate_parser.add_argument("--require-disjoint", default="calibration,coverage")
    validate_parser.add_argument("--require-a-disabled", action="store_true")
    validate_parser.add_argument("--allocation-registry", type=Path)
    validate_parser.add_argument("--allocation-registry-file-sha256")
    validate_parser.add_argument("--allocation-registry-content-sha256")
    validate_parser.add_argument("--claim-seed-disjointness", action="store_true")
    validate_parser.add_argument("--root-seed-opening", action="append", default=[])
    validate_parser.add_argument(
        "--complete-game-seed-opening",
        action="append",
        default=[],
    )
    args = parser.parse_args(argv)
    if args.command != "validate":  # pragma: no cover - argparse owns this branch
        raise AssertionError(args.command)
    try:
        required_panels = tuple(part.strip() for part in args.require_panels.split(","))
        if required_panels not in {("S", "H", "L"), ("S", "H")}:
            raise CohortError("v1 requires either Rival-MF S,H,L or high-only S,H")
        if args.require_disjoint != "calibration,coverage":
            raise CohortError("v1 requires calibration,coverage source-game disjointness")
        if not args.require_a_disabled:
            raise CohortError("v1 validation must require A disabled")
        registry_args = (
            args.allocation_registry,
            args.allocation_registry_file_sha256,
            args.allocation_registry_content_sha256,
        )
        if any(value is not None for value in registry_args) and not all(
            value is not None for value in registry_args
        ):
            raise CohortError(
                "allocation registry path, file SHA-256, and content SHA-256 are all required"
            )
        if args.claim_seed_disjointness and not all(value is not None for value in registry_args):
            raise CohortError("seed-disjointness claims require a byte-pinned allocation registry")
        if (args.root_seed_opening or args.complete_game_seed_opening) and not (
            args.claim_seed_disjointness
        ):
            raise CohortError("seed openings are accepted only with --claim-seed-disjointness")
        registry = None
        expected_registry_identity = None
        if all(value is not None for value in registry_args):
            registry = load_allocation_registry(
                args.allocation_registry,
                expected_file_sha256=args.allocation_registry_file_sha256,
                expected_content_sha256=args.allocation_registry_content_sha256,
            )
            expected_registry_identity = "sha256:" + require_sha256(
                args.allocation_registry_content_sha256,
                "allocation_registry_content_sha256",
            )
        summary = validate_manifest_collection(
            _load_manifest_records(args.manifest),
            allocation_registry=registry,
            expected_allocation_registry_identity=expected_registry_identity,
            required_panels=required_panels,
        )
        realization = None
        if args.claim_seed_disjointness:
            if registry is None or expected_registry_identity is None:
                raise CohortError(
                    "seed-disjointness claims require a byte-pinned allocation registry"
                )
            realization = validate_seed_realizations(
                registry,
                expected_allocation_registry_identity=expected_registry_identity,
                manifest_collection=summary,
                root_openings=(_parse_root_opening(value) for value in args.root_seed_opening),
                complete_game_openings=(
                    _parse_game_opening(value) for value in args.complete_game_seed_opening
                ),
            )
    except (CohortError, RivalSchemaError) as exc:
        print(json.dumps({"status": "DENIED", "reason": str(exc)}, sort_keys=True))
        return 2
    result = {
        "status": "VALID",
        "mode": "cpu_contract_only",
        "structural_manifest_validation": "VALIDATED",
        "commitment_uniqueness": (
            "VALIDATED_FROM_PINNED_REGISTRY"
            if summary.commitment_uniqueness_validated
            else "NOT_VALIDATED"
        ),
        "seed_disjointness": (
            "VALIDATED_FROM_EXACT_OPENINGS"
            if realization is not None
            else "NOT_VALIDATED_UNTIL_OPENINGS"
        ),
    }
    if summary.allocation_registry_identity is not None:
        result["allocation_registry_identity"] = summary.allocation_registry_identity
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
