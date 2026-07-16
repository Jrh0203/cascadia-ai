"""Hash-bound Rival preference sidecars and derived expert training views.

The existing :class:`~cascadiav3.expert_tensor_shards.ExpertTensorShard`
formats (v1--v4) remain immutable. Rival labels live in a separate JSON shard
whose source-shard digest, record index, public root, ordered menus, and every
action occurrence are checked before a label can join onto an expert example.

Preference attachment is explicitly opt-in. With ``enable_preferences=False``
the view delegates directly to the legacy example and collator paths and adds
no keys, which keeps existing recipes and checkpoint identities unchanged.
"""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..expert_tensor_shards import (
    SUPPORTED_SHARD_VERSIONS,
    ExpertTensorShard,
    collate_expert_tensor_examples,
)
from .schema import (
    RIVAL_PREFERENCE_SHARD_SCHEMA_ID,
    RIVAL_TRAINING_VIEW_SCHEMA_ID,
    canonical_json_bytes,
    read_pinned_canonical_json_object,
)

ROOT_IDENTITY_INDEX_SCHEMA_ID = "cascadiav3.rival_expert_root_identity_index.v1"
PUBLIC_ROOT_PREFIX = "cascadiav3.rival_public_root.v1:sha256:"
RULES_MENU_PREFIX = "cascadiav3.rival_rules_menu.v1:sha256:"
INCUMBENT_MENU_PREFIX = "cascadiav3.rival_incumbent_menu.v1:sha256:"
ACTION_CONTENT_PREFIX = "cascadiav3.rival_action_content.v1:sha256:"
CANDIDATE_OCCURRENCE_PREFIX = "cascadiav3.rival_candidate_action_occurrence.v1:sha256:"
PREFERENCE_CHALLENGER = "challenger_over_incumbent"
PREFERENCE_UNLABELED = "unlabeled"
PREFERENCE_CATEGORIES = frozenset({PREFERENCE_CHALLENGER, PREFERENCE_UNLABELED})
RIVAL_TRAINER_INTEGRATION_PHASE = "P8"
RIVAL_TRAINER_INTEGRATION_HELD_REASON = (
    f"Rival preference training is held at {RIVAL_TRAINER_INTEGRATION_PHASE} until "
    "positive P7 evidence, an explicit TRAIN instruction, a phase-specific permit, "
    "and flag-off batch/loss/optimizer/checkpoint bit-identity tests exist"
)


class RivalTrainerIntegrationHeld(RuntimeError):
    """Raised before Torch import when held Rival preference training is requested."""


def enforce_rival_trainer_hold(*, requested: bool) -> None:
    """Fail closed if a caller attempts to cross the plan's P8 training gate."""

    if not isinstance(requested, bool):
        raise TypeError("requested must be boolean")
    if requested:
        raise RivalTrainerIntegrationHeld(RIVAL_TRAINER_INTEGRATION_HELD_REASON)


_BINDING_KEYS = frozenset(
    {
        "record_index",
        "public_root_id",
        "rules_legal_menu_hash",
        "incumbent_candidate_menu_hash",
        "ordered_action_content_ids",
        "candidate_action_occurrence_ids",
        "action_tensor_row_sha256",
        "selected_action_index",
    }
)
_PANEL_KEYS = frozenset({"S", "H", "L", "A"})
_RECORD_KEYS = _BINDING_KEYS | frozenset(
    {
        "incumbent_action_index",
        "challenger_action_index",
        "categorical_preference",
        "preference_valid",
        "preference_weight",
        "activation_stratum",
        "natural_frequency_weight",
        "sampling_probability",
        "root_cohort_role",
        "panel_identities",
        "advantage_target",
        "advantage_valid",
    }
)
_EXPERT_REF_KEYS = frozenset({"sha256", "schema_id", "record_count"})
_INDEX_KEYS = frozenset(
    {
        "schema_id",
        "source_revision",
        "expert_shard_sha256",
        "raw_root_ledger_sha256",
        "bindings",
        "content_sha256",
    }
)
_SHARD_KEYS = frozenset(
    {
        "schema_id",
        "source_revision",
        "ruleset_identity_sha256",
        "expert_shard",
        "root_identity_index_sha256",
        "incumbent_policy_identity_sha256",
        "challenger_policy_identity_sha256",
        "coefficient_identity_sha256",
        "allocation_identity_sha256",
        "bound_identity_sha256",
        "error_ledger_identity_sha256",
        "parent_manifest_sha256",
        "raw_root_ledger_sha256",
        "raw_world_ledger_sha256",
        "inference_mode",
        "a_panel_enabled",
        "preference_weight",
        "records",
        "content_sha256",
    }
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _bytes_sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _expect_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _expect_exact_keys(value: Mapping[str, Any], expected: frozenset[str], field: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise ValueError(f"{field} keys mismatch: missing={missing}, unknown={unknown}")


def _expect_sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or len(value) != 71
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ValueError(f"{field} must use the lowercase 'sha256:<hex>' wire")
    return value


def _expect_namespaced_id(value: Any, field: str, prefix: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith(prefix)
        or len(value) != len(prefix) + 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value[len(prefix) :])
    ):
        raise ValueError(f"{field} must use exact Rust namespace {prefix!r}")
    return value


def _expect_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value


def _expect_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean")
    return value


def _expect_int(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _expect_finite(value: Any, field: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    return result


def _expect_namespaced_list(value: Any, field: str, prefix: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    return tuple(
        _expect_namespaced_id(item, f"{field}[{index}]", prefix) for index, item in enumerate(value)
    )


@dataclass(frozen=True)
class ExpertRootBinding:
    """Identity-only index row derived from the immutable raw expert root."""

    record_index: int
    public_root_id: str
    rules_legal_menu_hash: str
    incumbent_candidate_menu_hash: str
    ordered_action_content_ids: tuple[str, ...]
    candidate_action_occurrence_ids: tuple[str, ...]
    action_tensor_row_sha256: tuple[str, ...]
    selected_action_index: int

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> ExpertRootBinding:
        _expect_exact_keys(raw, _BINDING_KEYS, "root identity binding")
        content_ids = _expect_namespaced_list(
            raw["ordered_action_content_ids"],
            "ordered_action_content_ids",
            ACTION_CONTENT_PREFIX,
        )
        occurrence_ids = _expect_namespaced_list(
            raw["candidate_action_occurrence_ids"],
            "candidate_action_occurrence_ids",
            CANDIDATE_OCCURRENCE_PREFIX,
        )
        tensor_hashes_raw = raw["action_tensor_row_sha256"]
        if not isinstance(tensor_hashes_raw, list) or not tensor_hashes_raw:
            raise ValueError("action_tensor_row_sha256 must be a non-empty list")
        tensor_hashes = tuple(
            _expect_sha256(value, f"action_tensor_row_sha256[{index}]")
            for index, value in enumerate(tensor_hashes_raw)
        )
        if len(content_ids) != len(occurrence_ids) or len(content_ids) != len(tensor_hashes):
            raise ValueError("action content, occurrence, and tensor identity counts differ")
        if len(set(occurrence_ids)) != len(occurrence_ids):
            raise ValueError("root action occurrence identities must be unique")
        selected_action_index = _expect_int(raw["selected_action_index"], "selected_action_index")
        if selected_action_index >= len(content_ids):
            raise ValueError("selected_action_index is outside the candidate menu")
        return cls(
            record_index=_expect_int(raw["record_index"], "record_index"),
            public_root_id=_expect_namespaced_id(
                raw["public_root_id"], "public_root_id", PUBLIC_ROOT_PREFIX
            ),
            rules_legal_menu_hash=_expect_namespaced_id(
                raw["rules_legal_menu_hash"], "rules_legal_menu_hash", RULES_MENU_PREFIX
            ),
            incumbent_candidate_menu_hash=_expect_namespaced_id(
                raw["incumbent_candidate_menu_hash"],
                "incumbent_candidate_menu_hash",
                INCUMBENT_MENU_PREFIX,
            ),
            ordered_action_content_ids=content_ids,
            candidate_action_occurrence_ids=occurrence_ids,
            action_tensor_row_sha256=tensor_hashes,
            selected_action_index=selected_action_index,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_index": self.record_index,
            "public_root_id": self.public_root_id,
            "rules_legal_menu_hash": self.rules_legal_menu_hash,
            "incumbent_candidate_menu_hash": self.incumbent_candidate_menu_hash,
            "ordered_action_content_ids": list(self.ordered_action_content_ids),
            "candidate_action_occurrence_ids": list(self.candidate_action_occurrence_ids),
            "action_tensor_row_sha256": list(self.action_tensor_row_sha256),
            "selected_action_index": self.selected_action_index,
        }


def root_identity_index_sha256(bindings: Sequence[ExpertRootBinding | Mapping[str, Any]]) -> str:
    """Hash an ordered, complete expert-record identity index."""

    normalized = [
        (
            binding
            if isinstance(binding, ExpertRootBinding)
            else ExpertRootBinding.from_mapping(binding)
        )
        for binding in bindings
    ]
    indices = [binding.record_index for binding in normalized]
    if indices != list(range(len(normalized))):
        raise ValueError("root identity bindings must be complete and ordered by record_index")
    return _canonical_sha256(
        {
            "schema_id": ROOT_IDENTITY_INDEX_SCHEMA_ID,
            "bindings": [binding.to_dict() for binding in normalized],
        }
    )


def _tensor_row_sha256(row: Any) -> str:
    """Hash one exact expert action row including dtype and shape."""
    import numpy as np

    array = np.ascontiguousarray(row)
    digest = hashlib.sha256()
    parts = (
        array.dtype.str.encode("ascii"),
        repr(array.shape).encode("ascii"),
        array.tobytes(),
    )
    for part in parts:
        digest.update(len(part).to_bytes(8, "little"))
        digest.update(part)
    return "sha256:" + digest.hexdigest()


def _verify_content_hash(payload: Mapping[str, Any], *, field: str = "content_sha256") -> str:
    observed = _expect_sha256(payload.get(field), field)
    content = {key: value for key, value in payload.items() if key != field}
    expected = _canonical_sha256(content)
    if observed != expected:
        raise ValueError(f"{field} mismatch: observed {observed}; recomputed {expected}")
    return observed


def attach_content_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "content_sha256" in payload:
        raise ValueError("refusing to overwrite content_sha256")
    result = dict(payload)
    result["content_sha256"] = _canonical_sha256(result)
    return result


class ExpertRootIdentityIndex:
    """Verified identity-to-tensor join emitted beside a raw root ledger."""

    def __init__(
        self,
        payload: Mapping[str, Any],
        *,
        expert_shard_path: Path,
        raw_root_ledger_path: Path,
        path: Path | None = None,
        expected_file_sha256: str | None = None,
    ) -> None:
        _expect_exact_keys(payload, _INDEX_KEYS, "root identity index")
        if payload["schema_id"] != ROOT_IDENTITY_INDEX_SCHEMA_ID:
            raise ValueError("unsupported root identity index schema")
        self.content_sha256 = _verify_content_hash(payload)
        self.path = Path(path) if path is not None else None
        if self.path is None:
            if expected_file_sha256 is not None:
                raise ValueError("an expected root-index file hash requires a path")
            self.index_file_sha256: str | None = None
        else:
            if expected_file_sha256 is None:
                raise ValueError("a persisted root identity index requires an expected file hash")
            self.index_file_sha256 = _expect_sha256(expected_file_sha256, "expected_file_sha256")
            persisted_payload = read_pinned_canonical_json_object(
                self.path,
                expected_file_sha256=self.index_file_sha256,
                field="root identity index",
            )
            if canonical_json_bytes(persisted_payload) != canonical_json_bytes(payload):
                raise ValueError("root identity index payload does not match its pinned file")
        self.expert_shard_path = Path(expert_shard_path)
        self.raw_root_ledger_path = Path(raw_root_ledger_path)
        self.source_revision = _expect_text(payload["source_revision"], "source_revision")
        self.expert_shard_sha256 = _expect_sha256(
            payload["expert_shard_sha256"], "expert_shard_sha256"
        )
        self.raw_root_ledger_sha256 = _expect_sha256(
            payload["raw_root_ledger_sha256"], "raw_root_ledger_sha256"
        )
        if file_sha256(self.expert_shard_path) != self.expert_shard_sha256:
            raise ValueError("root index expert shard SHA-256 mismatch")
        if file_sha256(self.raw_root_ledger_path) != self.raw_root_ledger_sha256:
            raise ValueError("root index raw root ledger SHA-256 mismatch")
        bindings_raw = payload["bindings"]
        if not isinstance(bindings_raw, list) or not bindings_raw:
            raise ValueError("root identity index bindings must be a non-empty list")
        bindings = tuple(
            ExpertRootBinding.from_mapping(_expect_mapping(value, f"bindings[{index}]"))
            for index, value in enumerate(bindings_raw)
        )
        if [binding.record_index for binding in bindings] != list(range(len(bindings))):
            raise ValueError("root identity index must be complete and ordered")

        expert = ExpertTensorShard(self.expert_shard_path)
        try:
            if len(expert) != len(bindings):
                raise ValueError("root identity index must cover every expert record")
            if expert.metadata.get("source_revision") != self.source_revision:
                raise ValueError("root identity index source_revision mismatch")
            for binding in bindings:
                record_index = binding.record_index
                start = int(expert.action_offsets[record_index])
                end = int(expert.action_offsets[record_index + 1])
                if end - start != len(binding.ordered_action_content_ids):
                    raise ValueError(f"root index action count mismatch at {record_index}")
                selected = int(expert.selected_action_index[record_index])
                if selected != binding.selected_action_index:
                    raise ValueError(f"root index selected action mismatch at {record_index}")
                observed_hashes = tuple(
                    _tensor_row_sha256(expert.actions[action_index])
                    for action_index in range(start, end)
                )
                if observed_hashes != binding.action_tensor_row_sha256:
                    raise ValueError(f"root index action tensor mismatch at {record_index}")
        finally:
            expert.close()
        self.bindings = bindings
        self.binding_by_index = {binding.record_index: binding for binding in bindings}

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        expected_file_sha256: str,
        expert_shard_path: Path,
        raw_root_ledger_path: Path,
    ) -> ExpertRootIdentityIndex:
        expected = _expect_sha256(expected_file_sha256, "expected_file_sha256")
        raw = read_pinned_canonical_json_object(
            Path(path),
            expected_file_sha256=expected,
            field="root identity index",
        )
        return cls(
            raw,
            expert_shard_path=expert_shard_path,
            raw_root_ledger_path=raw_root_ledger_path,
            path=Path(path),
            expected_file_sha256=expected,
        )

    @property
    def is_pinned(self) -> bool:
        return self.path is not None and self.index_file_sha256 is not None

    def assert_sources_unchanged(self) -> None:
        if not self.is_pinned:
            raise ValueError("root identity index was not loaded from a pinned immutable file")
        assert self.path is not None and self.index_file_sha256 is not None
        try:
            read_pinned_canonical_json_object(
                self.path,
                expected_file_sha256=self.index_file_sha256,
                field="root identity index",
            )
        except ValueError as exc:
            raise ValueError("root identity index changed after pinned validation") from exc
        if file_sha256(self.expert_shard_path) != self.expert_shard_sha256:
            raise ValueError("expert shard changed after root-index validation")
        if file_sha256(self.raw_root_ledger_path) != self.raw_root_ledger_sha256:
            raise ValueError("raw root ledger changed after root-index validation")


def _write_immutable_bytes(destination: Path, data: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".partial",
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
            # Read-only mode prevents accidental in-place edits. The digest
            # pin remains the authority because an owner can deliberately
            # change permissions again.
            os.fchmod(handle.fileno(), 0o444)
        os.link(temporary_name, destination)
        os.unlink(temporary_name)
        temporary_name = ""
        directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)


def write_expert_root_identity_index(
    path: Path,
    payload: Mapping[str, Any],
    *,
    expert_shard_path: Path,
    raw_root_ledger_path: Path,
) -> ExpertRootIdentityIndex:
    ExpertRootIdentityIndex(
        payload,
        expert_shard_path=expert_shard_path,
        raw_root_ledger_path=raw_root_ledger_path,
    )
    data = canonical_json_bytes(payload) + b"\n"
    expected_file_sha256 = _bytes_sha256(data)
    _write_immutable_bytes(Path(path), data)
    return ExpertRootIdentityIndex.load(
        Path(path),
        expected_file_sha256=expected_file_sha256,
        expert_shard_path=expert_shard_path,
        raw_root_ledger_path=raw_root_ledger_path,
    )


@dataclass(frozen=True)
class RivalPreferenceRecord:
    binding: ExpertRootBinding
    incumbent_action_index: int
    challenger_action_index: int
    categorical_preference: str
    preference_valid: bool
    preference_weight: float
    activation_stratum: str
    natural_frequency_weight: float
    sampling_probability: float
    root_cohort_role: str
    panel_identities: dict[str, str | None]
    advantage_target: float | None
    advantage_valid: bool

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        inference_mode: str,
        a_panel_enabled: bool,
        frozen_preference_weight: float,
    ) -> RivalPreferenceRecord:
        _expect_exact_keys(raw, _RECORD_KEYS, "preference record")
        binding = ExpertRootBinding.from_mapping({key: raw[key] for key in _BINDING_KEYS})
        action_count = len(binding.ordered_action_content_ids)
        incumbent = _expect_int(raw["incumbent_action_index"], "incumbent_action_index")
        challenger = _expect_int(raw["challenger_action_index"], "challenger_action_index")
        if incumbent >= action_count or challenger >= action_count:
            raise ValueError("incumbent/challenger action index is outside the ordered menu")
        if incumbent == challenger:
            raise ValueError("selected challenger must differ from the incumbent action")

        valid = _expect_bool(raw["preference_valid"], "preference_valid")
        category = raw["categorical_preference"]
        if category not in PREFERENCE_CATEGORIES:
            raise ValueError(f"unsupported categorical_preference {category!r}")
        expected_category = PREFERENCE_CHALLENGER if valid else PREFERENCE_UNLABELED
        if category != expected_category:
            raise ValueError(
                f"preference_valid={valid} requires categorical_preference={expected_category!r}"
            )
        weight = _expect_finite(raw["preference_weight"], "preference_weight", minimum=0.0)
        if weight <= 0.0:
            raise ValueError("preference_weight must be positive")
        if weight != frozen_preference_weight:
            raise ValueError("record preference_weight differs from the shard's frozen weight")

        natural_weight = _expect_finite(
            raw["natural_frequency_weight"], "natural_frequency_weight", minimum=0.0
        )
        if natural_weight <= 0.0:
            raise ValueError("natural_frequency_weight must be positive")
        sampling_probability = _expect_finite(
            raw["sampling_probability"], "sampling_probability", minimum=0.0
        )
        if not 0.0 < sampling_probability <= 1.0:
            raise ValueError("sampling_probability must be in (0, 1]")

        panels_raw = _expect_mapping(raw["panel_identities"], "panel_identities")
        _expect_exact_keys(panels_raw, _PANEL_KEYS, "panel_identities")
        panels: dict[str, str | None] = {
            "S": _expect_sha256(panels_raw["S"], "panel_identities.S"),
            "H": _expect_sha256(panels_raw["H"], "panel_identities.H"),
        }
        if inference_mode == "multifidelity":
            panels["L"] = _expect_sha256(panels_raw["L"], "panel_identities.L")
        elif inference_mode == "high_fidelity_only":
            if panels_raw["L"] is not None:
                raise ValueError("high-fidelity-only preference records require L=null")
            panels["L"] = None
        else:
            raise ValueError(f"unsupported preference inference_mode {inference_mode!r}")
        if a_panel_enabled:
            panels["A"] = _expect_sha256(panels_raw["A"], "panel_identities.A")
        else:
            if panels_raw["A"] is not None:
                raise ValueError("A panel identity must be null while a_panel_enabled is false")
            panels["A"] = None
        present_panel_ids = [value for value in panels.values() if value is not None]
        if len(set(present_panel_ids)) != len(present_panel_ids):
            raise ValueError("S/H/L/A panel identities must be pairwise disjoint")

        advantage_valid = _expect_bool(raw["advantage_valid"], "advantage_valid")
        advantage_raw = raw["advantage_target"]
        if not a_panel_enabled:
            if advantage_valid or advantage_raw is not None:
                raise ValueError("quantitative advantage must be absent while A is disabled")
            advantage = None
        elif advantage_valid:
            advantage = _expect_finite(advantage_raw, "advantage_target")
        else:
            if advantage_raw is not None:
                raise ValueError("invalid quantitative advantage target must be null")
            advantage = None

        cohort = _expect_text(raw["root_cohort_role"], "root_cohort_role")
        if cohort != "relabel_selection":
            raise ValueError("preference records must belong to the relabel_selection cohort")

        return cls(
            binding=binding,
            incumbent_action_index=incumbent,
            challenger_action_index=challenger,
            categorical_preference=category,
            preference_valid=valid,
            preference_weight=weight,
            activation_stratum=_expect_text(raw["activation_stratum"], "activation_stratum"),
            natural_frequency_weight=natural_weight,
            sampling_probability=sampling_probability,
            root_cohort_role=cohort,
            panel_identities=panels,
            advantage_target=advantage,
            advantage_valid=advantage_valid,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.binding.to_dict(),
            "incumbent_action_index": self.incumbent_action_index,
            "challenger_action_index": self.challenger_action_index,
            "categorical_preference": self.categorical_preference,
            "preference_valid": self.preference_valid,
            "preference_weight": self.preference_weight,
            "activation_stratum": self.activation_stratum,
            "natural_frequency_weight": self.natural_frequency_weight,
            "sampling_probability": self.sampling_probability,
            "root_cohort_role": self.root_cohort_role,
            "panel_identities": dict(self.panel_identities),
            "advantage_target": self.advantage_target,
            "advantage_valid": self.advantage_valid,
        }


class RivalPreferenceShard:
    """Validated, hash-bound preference labels for a subset of expert roots."""

    def __init__(
        self,
        payload: Mapping[str, Any],
        *,
        expert_shard_path: Path,
        root_identity_index: ExpertRootIdentityIndex,
        raw_world_ledger_path: Path,
        path: Path | None = None,
        expected_file_sha256: str | None = None,
    ) -> None:
        _expect_exact_keys(payload, _SHARD_KEYS, "preference shard")
        if payload["schema_id"] != RIVAL_PREFERENCE_SHARD_SCHEMA_ID:
            raise ValueError(
                f"unsupported preference shard schema {payload['schema_id']!r}; "
                f"expected {RIVAL_PREFERENCE_SHARD_SCHEMA_ID!r}"
            )
        self.path = Path(path) if path is not None else None
        if self.path is None:
            if expected_file_sha256 is not None:
                raise ValueError("an expected preference file hash requires a path")
            self.preference_file_sha256: str | None = None
        else:
            if expected_file_sha256 is None:
                raise ValueError("a persisted preference shard requires an expected file hash")
            self.preference_file_sha256 = _expect_sha256(
                expected_file_sha256, "expected_file_sha256"
            )
            persisted_payload = read_pinned_canonical_json_object(
                self.path,
                expected_file_sha256=self.preference_file_sha256,
                field="preference shard",
            )
            if canonical_json_bytes(persisted_payload) != canonical_json_bytes(payload):
                raise ValueError("preference shard payload does not match its pinned file")
        self.expert_shard_path = Path(expert_shard_path)
        self.raw_world_ledger_path = Path(raw_world_ledger_path)
        self.content_sha256 = _verify_content_hash(payload)
        self.source_revision = _expect_text(payload["source_revision"], "source_revision")
        for field in (
            "ruleset_identity_sha256",
            "root_identity_index_sha256",
            "incumbent_policy_identity_sha256",
            "challenger_policy_identity_sha256",
            "allocation_identity_sha256",
            "bound_identity_sha256",
            "error_ledger_identity_sha256",
            "parent_manifest_sha256",
            "raw_root_ledger_sha256",
            "raw_world_ledger_sha256",
        ):
            setattr(self, field, _expect_sha256(payload[field], field))
        self.inference_mode = _expect_text(payload["inference_mode"], "inference_mode")
        coefficient_raw = payload["coefficient_identity_sha256"]
        if self.inference_mode == "multifidelity":
            self.coefficient_identity_sha256 = _expect_sha256(
                coefficient_raw, "coefficient_identity_sha256"
            )
        elif self.inference_mode == "high_fidelity_only":
            if coefficient_raw is not None:
                raise ValueError("high-fidelity-only shard requires coefficient identity null")
            self.coefficient_identity_sha256 = None
        else:
            raise ValueError(f"unsupported inference_mode {self.inference_mode!r}")
        self.a_panel_enabled = _expect_bool(payload["a_panel_enabled"], "a_panel_enabled")
        if self.a_panel_enabled:
            raise ValueError(
                "Rival preference shard v1 has A disabled; quantitative targets require "
                "a future schema revision"
            )
        self.preference_weight = _expect_finite(
            payload["preference_weight"], "preference_weight", minimum=0.0
        )
        if self.preference_weight <= 0.0:
            raise ValueError("preference_weight must be positive")

        if not isinstance(root_identity_index, ExpertRootIdentityIndex):
            raise ValueError("preference shard requires a verified root identity index")
        if not root_identity_index.is_pinned:
            raise ValueError(
                "preference shard requires a root identity index loaded from a pinned file"
            )
        self.root_identity_index = root_identity_index
        root_identity_index.assert_sources_unchanged()
        if self.root_identity_index_sha256 != root_identity_index.content_sha256:
            raise ValueError("root identity index SHA-256 mismatch")
        if self.raw_root_ledger_sha256 != root_identity_index.raw_root_ledger_sha256:
            raise ValueError("preference raw root ledger does not match identity index")
        if file_sha256(self.raw_world_ledger_path) != self.raw_world_ledger_sha256:
            raise ValueError("preference raw world ledger SHA-256 mismatch")
        binding_by_index = root_identity_index.binding_by_index

        expert_ref = _expect_mapping(payload["expert_shard"], "expert_shard")
        _expect_exact_keys(expert_ref, _EXPERT_REF_KEYS, "expert_shard")
        expected_expert_hash = _expect_sha256(expert_ref["sha256"], "expert_shard.sha256")
        actual_expert_hash = file_sha256(self.expert_shard_path)
        if expected_expert_hash != actual_expert_hash:
            raise ValueError("expert shard SHA-256 mismatch")
        self.expert_shard_sha256 = actual_expert_hash

        if self.incumbent_policy_identity_sha256 == self.challenger_policy_identity_sha256:
            raise ValueError("incumbent and challenger policy identities must differ")

        expert = ExpertTensorShard(self.expert_shard_path)
        try:
            expert_schema = _expect_text(expert_ref["schema_id"], "expert_shard.schema_id")
            if expert_schema not in SUPPORTED_SHARD_VERSIONS or expert_schema != expert.version:
                raise ValueError("expert shard schema identity mismatch")
            expert_count = _expect_int(expert_ref["record_count"], "expert_shard.record_count")
            if expert_count != len(expert):
                raise ValueError("expert shard record count mismatch")
            if len(root_identity_index.bindings) != len(expert):
                raise ValueError("root identity index must cover every expert record")
            expert_source_revision = expert.metadata.get("source_revision")
            if expert_source_revision != self.source_revision:
                raise ValueError("preference source_revision does not match expert shard metadata")

            records_raw = payload["records"]
            if not isinstance(records_raw, list) or not records_raw:
                raise ValueError("records must be a non-empty list")
            records: dict[int, RivalPreferenceRecord] = {}
            panel_owners: dict[str, tuple[int, str]] = {}
            previous_record_index = -1
            for raw_index, raw_record in enumerate(records_raw):
                record = RivalPreferenceRecord.from_mapping(
                    _expect_mapping(raw_record, f"records[{raw_index}]"),
                    inference_mode=self.inference_mode,
                    a_panel_enabled=self.a_panel_enabled,
                    frozen_preference_weight=self.preference_weight,
                )
                record_index = record.binding.record_index
                if record_index in records:
                    raise ValueError(f"duplicate preference record_index {record_index}")
                if record_index <= previous_record_index:
                    raise ValueError("preference records must be ordered by expert record_index")
                previous_record_index = record_index
                for panel_name, panel_identity in record.panel_identities.items():
                    if panel_identity is None:
                        continue
                    previous_owner = panel_owners.get(panel_identity)
                    if previous_owner is not None:
                        raise ValueError(
                            "panel identity reused across preference records: "
                            f"{panel_identity} belongs to {previous_owner} and "
                            f"{(record_index, panel_name)}"
                        )
                    panel_owners[panel_identity] = (record_index, panel_name)
                expected_binding = binding_by_index.get(record_index)
                if expected_binding is None:
                    raise ValueError(
                        f"preference record_index {record_index} is outside expert shard"
                    )
                if record.binding != expected_binding:
                    raise ValueError(
                        f"preference identity mismatch at expert record {record_index}"
                    )
                action_start = int(expert.action_offsets[record_index])
                action_end = int(expert.action_offsets[record_index + 1])
                if len(record.binding.ordered_action_content_ids) != action_end - action_start:
                    raise ValueError(
                        f"ordered menu length mismatch at expert record {record_index}"
                    )
                if record.incumbent_action_index != record.binding.selected_action_index:
                    raise ValueError(
                        f"incumbent action is not the expert-selected action at {record_index}"
                    )
                records[record_index] = record
        finally:
            expert.close()
        self.records = records
        self.expert_schema_id = str(expert_ref["schema_id"])
        self.expert_record_count = int(expert_ref["record_count"])

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        expert_shard_path: Path,
        root_identity_index: ExpertRootIdentityIndex,
        raw_world_ledger_path: Path,
        expected_file_sha256: str,
    ) -> RivalPreferenceShard:
        expected = _expect_sha256(expected_file_sha256, "expected_file_sha256")
        raw = read_pinned_canonical_json_object(
            Path(path),
            expected_file_sha256=expected,
            field="preference shard",
        )
        return cls(
            raw,
            expert_shard_path=expert_shard_path,
            root_identity_index=root_identity_index,
            raw_world_ledger_path=raw_world_ledger_path,
            path=Path(path),
            expected_file_sha256=expected,
        )

    @property
    def is_pinned(self) -> bool:
        return self.path is not None and self.preference_file_sha256 is not None

    def assert_source_unchanged(self) -> None:
        if not self.is_pinned:
            raise ValueError("preference shard was not loaded from a pinned immutable file")
        assert self.path is not None and self.preference_file_sha256 is not None
        try:
            read_pinned_canonical_json_object(
                self.path,
                expected_file_sha256=self.preference_file_sha256,
                field="preference shard",
            )
        except ValueError as exc:
            raise ValueError("preference shard changed after pinned validation") from exc
        if file_sha256(self.expert_shard_path) != self.expert_shard_sha256:
            raise ValueError("expert shard changed after preference join validation")
        self.root_identity_index.assert_sources_unchanged()
        if file_sha256(self.raw_world_ledger_path) != self.raw_world_ledger_sha256:
            raise ValueError("raw world ledger changed after preference join validation")

    def record(self, index: int) -> RivalPreferenceRecord | None:
        return self.records.get(index)


def write_rival_preference_shard(
    path: Path,
    payload: Mapping[str, Any],
    *,
    expert_shard_path: Path,
    root_identity_index: ExpertRootIdentityIndex,
    raw_world_ledger_path: Path,
) -> RivalPreferenceShard:
    """Validate then create one immutable canonical preference shard."""

    RivalPreferenceShard(
        payload,
        expert_shard_path=expert_shard_path,
        root_identity_index=root_identity_index,
        raw_world_ledger_path=raw_world_ledger_path,
    )
    destination = Path(path)
    data = canonical_json_bytes(payload) + b"\n"
    expected_file_sha256 = _bytes_sha256(data)
    _write_immutable_bytes(destination, data)
    return RivalPreferenceShard.load(
        destination,
        expected_file_sha256=expected_file_sha256,
        expert_shard_path=expert_shard_path,
        root_identity_index=root_identity_index,
        raw_world_ledger_path=raw_world_ledger_path,
    )


def _attach_preference_to_example(
    example: dict[str, Any],
    record: RivalPreferenceRecord | None,
) -> dict[str, Any]:
    preference_valid = record is not None and record.preference_valid
    result = dict(example)
    result.update(
        {
            "training_view_schema_id": RIVAL_TRAINING_VIEW_SCHEMA_ID,
            # H establishes only challenger > incumbent. Untested actions are
            # absent from this pairwise target, never encoded as losers.
            "preference_incumbent_index": (
                record.incumbent_action_index if preference_valid else 0
            ),
            "preference_challenger_index": (
                record.challenger_action_index if preference_valid else 0
            ),
            "preference_valid": preference_valid,
            # Loss weight is preregistered and categorical. Sampling and
            # natural-frequency weights remain separate and cannot replace it.
            "policy_weight": record.preference_weight if preference_valid else 0.0,
            "natural_frequency_weight": (
                record.natural_frequency_weight if record is not None else 1.0
            ),
            "sampling_probability": (record.sampling_probability if record is not None else 1.0),
            "advantage_target": (
                record.advantage_target if record is not None and record.advantage_valid else 0.0
            ),
            "advantage_valid": bool(record is not None and record.advantage_valid),
        }
    )
    return result


def collate_rival_training_examples(
    examples: list[dict[str, Any]],
    *,
    enable_preferences: bool = False,
) -> dict[str, Any]:
    """Collate legacy examples, optionally adding padded Rival targets."""

    if not isinstance(enable_preferences, bool):
        raise TypeError("enable_preferences must be boolean")
    if not enable_preferences:
        return collate_expert_tensor_examples(examples)
    required = {
        "training_view_schema_id",
        "preference_incumbent_index",
        "preference_challenger_index",
        "preference_valid",
        "policy_weight",
        "natural_frequency_weight",
        "sampling_probability",
        "advantage_target",
        "advantage_valid",
    }
    for index, example in enumerate(examples):
        missing = required - set(example)
        if missing:
            raise ValueError(f"Rival example {index} is missing fields: {sorted(missing)}")
        if example["training_view_schema_id"] != RIVAL_TRAINING_VIEW_SCHEMA_ID:
            raise ValueError(f"Rival example {index} has an incompatible training view schema")

    for index, example in enumerate(examples):
        action_count = int(example["actions"].shape[0])
        preference_valid = _expect_bool(
            example["preference_valid"],
            f"preference_valid at example {index}",
        )
        for field in ("preference_incumbent_index", "preference_challenger_index"):
            value = example[field]
            invalid_index = (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value < action_count
            )
            if invalid_index:
                raise ValueError(f"{field} is outside the action menu at example {index}")
        incumbent = int(example["preference_incumbent_index"])
        challenger = int(example["preference_challenger_index"])
        policy_weight = _expect_finite(
            example["policy_weight"],
            f"policy_weight at example {index}",
            minimum=0.0,
        )
        if preference_valid:
            if incumbent == challenger:
                raise ValueError(f"valid preference indices must differ at example {index}")
            if policy_weight <= 0.0:
                raise ValueError(
                    f"valid preference requires positive policy_weight at example {index}"
                )
        elif incumbent != 0 or challenger != 0 or policy_weight != 0.0:
            raise ValueError(
                f"invalid preference must use zero indices and weight at example {index}"
            )
        natural_weight = _expect_finite(
            example["natural_frequency_weight"],
            f"natural_frequency_weight at example {index}",
            minimum=0.0,
        )
        if natural_weight <= 0.0:
            raise ValueError(f"natural_frequency_weight must be positive at example {index}")
        sampling_probability = _expect_finite(
            example["sampling_probability"],
            f"sampling_probability at example {index}",
            minimum=0.0,
        )
        if not 0.0 < sampling_probability <= 1.0:
            raise ValueError(f"sampling_probability must be in (0, 1] at example {index}")
        advantage_valid = _expect_bool(
            example["advantage_valid"],
            f"advantage_valid at example {index}",
        )
        advantage_target = _expect_finite(
            example["advantage_target"],
            f"advantage_target at example {index}",
        )
        if advantage_valid or advantage_target != 0.0:
            raise ValueError(
                f"Rival training view v1 requires disabled advantage at example {index}"
            )

    batch = collate_expert_tensor_examples(examples)
    import torch

    batch.update(
        {
            "training_view_schema_id": RIVAL_TRAINING_VIEW_SCHEMA_ID,
            "preference_incumbent_index": torch.tensor(
                [int(example["preference_incumbent_index"]) for example in examples],
                dtype=torch.long,
            ),
            "preference_challenger_index": torch.tensor(
                [int(example["preference_challenger_index"]) for example in examples],
                dtype=torch.long,
            ),
            "preference_valid": torch.tensor(
                [bool(example["preference_valid"]) for example in examples], dtype=torch.bool
            ),
            "policy_weight": torch.tensor(
                [float(example["policy_weight"]) for example in examples], dtype=torch.float32
            ),
            "natural_frequency_weight": torch.tensor(
                [float(example["natural_frequency_weight"]) for example in examples],
                dtype=torch.float32,
            ),
            "sampling_probability": torch.tensor(
                [float(example["sampling_probability"]) for example in examples],
                dtype=torch.float32,
            ),
            "advantage_target": torch.tensor(
                [float(example["advantage_target"]) for example in examples],
                dtype=torch.float32,
            ),
            "advantage_valid": torch.tensor(
                [bool(example["advantage_valid"]) for example in examples], dtype=torch.bool
            ),
        }
    )
    return batch


class RivalTrainingView:
    """Opt-in derived view over one immutable ExpertTensorShard."""

    def __init__(
        self,
        expert_shard_path: Path,
        *,
        preference_shard: RivalPreferenceShard | None = None,
        enable_preferences: bool = False,
    ) -> None:
        if not isinstance(enable_preferences, bool):
            raise TypeError("enable_preferences must be boolean")
        if enable_preferences and preference_shard is None:
            raise ValueError("enable_preferences=True requires a validated preference shard")
        self.expert = ExpertTensorShard(Path(expert_shard_path))
        try:
            self.preference_shard = preference_shard
            self.enable_preferences = enable_preferences
            if preference_shard is not None:
                if not preference_shard.is_pinned:
                    raise ValueError(
                        "training requires a preference shard loaded from a pinned file"
                    )
                if (
                    Path(expert_shard_path).resolve()
                    != preference_shard.expert_shard_path.resolve()
                ):
                    raise ValueError("preference shard is bound to a different expert shard path")
                preference_shard.assert_source_unchanged()
                if self.expert.version != preference_shard.expert_schema_id:
                    raise ValueError("preference shard expert schema changed")
        except BaseException:
            self.expert.close()
            raise

    def __len__(self) -> int:
        return len(self.expert)

    def example(self, index: int) -> dict[str, Any]:
        example = self.expert.example(index)
        if not self.enable_preferences:
            return example
        assert self.preference_shard is not None
        return _attach_preference_to_example(example, self.preference_shard.record(index))

    def collate(self, indices: Sequence[int]) -> dict[str, Any]:
        if not indices:
            raise ValueError("cannot collate an empty Rival training view")
        examples = [self.example(index) for index in indices]
        return collate_rival_training_examples(
            examples,
            enable_preferences=self.enable_preferences,
        )

    def close(self) -> None:
        self.expert.close()

    def __enter__(self) -> RivalTrainingView:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
