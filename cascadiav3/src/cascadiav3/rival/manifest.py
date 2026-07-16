"""Frozen identity and root-manifest validation for Rival experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any

from .schema import (
    RIVAL_POLICY_IDENTITY_SCHEMA_ID,
    RIVAL_ROOT_MANIFEST_SCHEMA_ID,
    RivalSchemaError,
    _issue_validation_capability,
    _require_validation_capability,
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
)

POLICY_KINDS = frozenset({"B_k", "pi_L", "W_k", "M_(k+1)"})
CANONICAL_RESEARCH_RULESET = {
    "schema_id": "cascadiav3.research_ruleset_identity.v1",
    "legacy_ruleset_id": ("cascadia_research_aaaaa_4p_card_a_no_habitat_bonus_rules_2026_07_16"),
    "rules_semantics_id": "cascadia-base-official-2026-07-16",
    "game_config_sha256": (
        "sha256:f5b2c782a483db870c50366b33cccde6d9a82a92a571cf9f29c752b750a5c07c"
    ),
}
ROOT_COHORT_ROLES = frozenset(
    {
        "design_tomography",
        "coefficient_calibration",
        "untouched_coverage",
        "shadow_one_seat",
        "relabel_selection",
    }
)
COMPLETE_GAME_SEED_ROLES = frozenset({"promotion", "target"})
ROOT_KINDS = frozenset({"prelude_policy_root", "draft_policy_root"})
PUBLIC_ROOT_ID_PREFIX = "cascadiav3.rival_public_root.v1:sha256:"
ACTION_CONTENT_ID_PREFIX = "cascadiav3.rival_action_content.v1:sha256:"
CANDIDATE_OCCURRENCE_ID_PREFIX = "cascadiav3.rival_candidate_action_occurrence.v1:sha256:"
RULES_MENU_HASH_PREFIX = "cascadiav3.rival_rules_menu.v1:sha256:"
INCUMBENT_MENU_HASH_PREFIX = "cascadiav3.rival_incumbent_menu.v1:sha256:"
CANDIDATE_SET_SCHEMA_ID = "cascadiav3.rival_candidate_set.v1"
DEPLOYMENT_DESIGN_SCHEMA_ID = "cascadiav3.rival_deployment_design.v1"


@dataclass(frozen=True)
class PolicyIdentity:
    policy_kind: str
    identity_sha256: str
    ruleset: dict[str, str]
    source_revision: str
    source_digest: str
    executable_sha256: str
    checkpoint_sha256: str
    weights_sha256: str
    compiler_identity: str
    simulator_identity: str
    sampler_identity: str
    candidate_generator_identity: str


_POLICY_OUTER_FIELDS = ("schema_id", "policy_kind", "fields")
_POLICY_FIELDS = (
    "ruleset",
    "source_revision",
    "source_digest",
    "executable_sha256",
    "model_manifest_sha256",
    "checkpoint_sha256",
    "weights_sha256",
    "bridge_protocol",
    "tensor_schema",
    "numerical_mode",
    "precision",
    "gumbel_config_sha256",
    "search_config_sha256",
    "refresh_config_sha256",
    "exact_endgame_config_sha256",
    "action_content_id_version",
    "rules_action_occurrence_id_version",
    "candidate_action_occurrence_id_version",
    "rules_menu_hash_version",
    "incumbent_menu_hash_version",
    "rng_contracts",
    "public_observation_schema",
    "policy_memory_schema",
    "failure_behavior",
    "compiler_identity",
    "simulator_identity",
    "sampler_identity",
    "candidate_generator_identity",
    "forbidden_capabilities",
)
_RULESET_FIELDS = (
    "schema_id",
    "legacy_ruleset_id",
    "rules_semantics_id",
    "game_config_sha256",
)
_RNG_CONTRACT_FIELDS = ("physical", "policy", "redetermination", "search", "tie_break")
_FAILURE_FIELDS = ("timeout", "incomplete_unit", "oom", "fallback")
_FORBIDDEN_FIELDS = (
    "table_total_utility",
    "table_native_q",
    "true_hidden_peeking",
    "model_fallback",
)
_DIGEST_FIELDS = (
    "source_digest",
    "executable_sha256",
    "model_manifest_sha256",
    "checkpoint_sha256",
    "weights_sha256",
    "gumbel_config_sha256",
    "search_config_sha256",
    "refresh_config_sha256",
    "exact_endgame_config_sha256",
)
_TEXT_FIELDS = (
    "source_revision",
    "bridge_protocol",
    "tensor_schema",
    "action_content_id_version",
    "rules_action_occurrence_id_version",
    "candidate_action_occurrence_id_version",
    "rules_menu_hash_version",
    "incumbent_menu_hash_version",
    "public_observation_schema",
    "policy_memory_schema",
    "compiler_identity",
    "simulator_identity",
    "sampler_identity",
    "candidate_generator_identity",
)


def _require_rust_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise RivalSchemaError(f"{field} must use the Rust 'sha256:' wire")
    return "sha256:" + require_sha256(value, field)


def validate_policy_identity(
    record: Mapping[str, Any], *, expected_policy_kind: str | None = None
) -> PolicyIdentity:
    require_schema(record, RIVAL_POLICY_IDENTITY_SCHEMA_ID)
    require_exact_keys(record, required=_POLICY_OUTER_FIELDS, where="policy identity")
    kind = require_nonempty_string(record["policy_kind"], "policy_kind")
    if kind not in POLICY_KINDS:
        raise RivalSchemaError(f"unknown or substitutable policy_kind: {kind!r}")
    if expected_policy_kind is not None and kind != expected_policy_kind:
        raise RivalSchemaError(
            f"policy kind substitution: observed {kind!r}; expected {expected_policy_kind!r}"
        )
    fields = record["fields"]
    if not isinstance(fields, Mapping):
        raise RivalSchemaError("policy identity fields must be an object")
    require_exact_keys(fields, required=_POLICY_FIELDS, where="policy identity fields")

    ruleset_raw = fields["ruleset"]
    if not isinstance(ruleset_raw, Mapping):
        raise RivalSchemaError("policy ruleset must be a structured object")
    require_exact_keys(ruleset_raw, required=_RULESET_FIELDS, where="policy ruleset")
    ruleset = {
        "schema_id": require_nonempty_string(ruleset_raw["schema_id"], "ruleset.schema_id"),
        "legacy_ruleset_id": require_nonempty_string(
            ruleset_raw["legacy_ruleset_id"], "ruleset.legacy_ruleset_id"
        ),
        "rules_semantics_id": require_nonempty_string(
            ruleset_raw["rules_semantics_id"], "ruleset.rules_semantics_id"
        ),
        "game_config_sha256": _require_rust_sha256(
            ruleset_raw["game_config_sha256"], "ruleset.game_config_sha256"
        ),
    }
    if ruleset != CANONICAL_RESEARCH_RULESET:
        raise RivalSchemaError(
            "policy identity does not carry the canonical Rust-authored research ruleset"
        )

    text = {field: require_nonempty_string(fields[field], field) for field in _TEXT_FIELDS}
    digests = {field: _require_rust_sha256(fields[field], field) for field in _DIGEST_FIELDS}
    if fields["numerical_mode"] not in {"deterministic", "tf32_off", "tf32_on"}:
        raise RivalSchemaError("unknown numerical_mode")
    if fields["precision"] not in {"fp32", "bf16", "fp16"}:
        raise RivalSchemaError("unknown precision")

    rng = fields["rng_contracts"]
    if not isinstance(rng, Mapping):
        raise RivalSchemaError("rng_contracts must be an object")
    require_exact_keys(rng, required=_RNG_CONTRACT_FIELDS, where="rng_contracts")
    for field in _RNG_CONTRACT_FIELDS:
        require_nonempty_string(rng[field], f"rng_contracts.{field}")

    failure = fields["failure_behavior"]
    if not isinstance(failure, Mapping):
        raise RivalSchemaError("failure_behavior must be an object")
    require_exact_keys(failure, required=_FAILURE_FIELDS, where="failure_behavior")
    allowed_dispositions = {"record_incomplete_no_label", "reject_launch", "forbidden"}
    for field in _FAILURE_FIELDS:
        if failure[field] not in allowed_dispositions:
            raise RivalSchemaError(f"unknown failure disposition for {field}")
    if failure["fallback"] != "forbidden":
        raise RivalSchemaError("model fallback must be explicitly forbidden")

    forbidden = fields["forbidden_capabilities"]
    if not isinstance(forbidden, Mapping):
        raise RivalSchemaError("forbidden_capabilities must be an object")
    require_exact_keys(forbidden, required=_FORBIDDEN_FIELDS, where="forbidden_capabilities")
    for field in _FORBIDDEN_FIELDS:
        if forbidden[field] is not False:
            raise RivalSchemaError(f"forbidden capability enabled: {field}")

    return PolicyIdentity(
        policy_kind=kind,
        identity_sha256="sha256:" + sha256_hex(record),
        ruleset=ruleset,
        source_revision=text["source_revision"],
        source_digest=digests["source_digest"],
        executable_sha256=digests["executable_sha256"],
        checkpoint_sha256=digests["checkpoint_sha256"],
        weights_sha256=digests["weights_sha256"],
        compiler_identity=text["compiler_identity"],
        simulator_identity=text["simulator_identity"],
        sampler_identity=text["sampler_identity"],
        candidate_generator_identity=text["candidate_generator_identity"],
    )


@dataclass(frozen=True)
class CandidateSelectionEntry:
    candidate_action_occurrence_id: str
    action_content_id: str
    expected_s: int


def _require_namespaced_digest(value: Any, field: str, prefix: str) -> str:
    text = require_nonempty_string(value, field)
    if not text.startswith(prefix):
        raise RivalSchemaError(f"{field} must use namespace {prefix!r}")
    require_sha256(text.removeprefix(prefix), field)
    return text


def candidate_set_identity(entries: Sequence[CandidateSelectionEntry]) -> str:
    """Hash the exact ordered candidate menu without its S allocation."""

    return "sha256:" + sha256_hex(
        {
            "schema_id": CANDIDATE_SET_SCHEMA_ID,
            "ordered_candidates": [
                {
                    "candidate_action_occurrence_id": row.candidate_action_occurrence_id,
                    "action_content_id": row.action_content_id,
                }
                for row in entries
            ],
        }
    )


_DEPLOYMENT_DESIGN_FIELDS = (
    "manifest_id",
    "ruleset_identity",
    "source_revision",
    "root_id",
    "source_game_id",
    "source_game_identity_sha256",
    "root_kind",
    "root_cohort_role",
    "complete_game_seed_role",
    "inference_mode",
    "required_panels",
    "forbidden_panels",
    "panel_identities",
    "multifidelity_claim",
    "incumbent_policy_identity",
    "incumbent_action_id",
    "incumbent_candidate_occurrence_id",
    "rules_menu_hash",
    "incumbent_menu_hash",
    "low_policy_identity",
    "candidate_set_identity",
    "candidate_selection_entries",
    "sampler_identity",
    "policy_rng_factory_identity",
    "terminal_verifier_executable_sha256",
    "terminal_verifier_contract_id",
    "allocation_identity",
    "bound_certificate_identity",
    "error_ledger_identity",
    "expected_s",
    "expected_h",
    "expected_l",
    "practical_margin",
    "preference_weight",
    "selection_rule",
    "low_expectation_id",
    "low_law_h_id",
    "low_law_l_id",
    "max_abs_beta",
    "a_panel_enabled",
    "quantitative_target_enabled",
)


def deployment_design_identity(record: Mapping[str, Any]) -> str:
    """Hash every deployment choice frozen before coefficient calibration."""

    try:
        payload = {field: record[field] for field in _DEPLOYMENT_DESIGN_FIELDS}
    except KeyError as exc:
        raise RivalSchemaError(
            f"deployment design is missing required field {exc.args[0]!r}"
        ) from exc
    return "sha256:" + sha256_hex({"schema_id": DEPLOYMENT_DESIGN_SCHEMA_ID, **payload})


@dataclass(frozen=True)
class RootManifest:
    manifest_id: str
    ruleset_identity: str
    source_revision: str
    root_id: str
    source_game_id: str
    source_game_identity_sha256: str
    root_kind: str
    root_cohort_role: str
    inference_mode: str
    required_panels: tuple[str, ...]
    forbidden_panels: tuple[str, ...]
    panel_identities: tuple[tuple[str, str | None], ...]
    beta_cv: float
    multifidelity_claim: bool
    incumbent_policy_identity: str
    incumbent_action_id: str
    incumbent_candidate_occurrence_id: str
    rules_menu_hash: str
    incumbent_menu_hash: str
    low_policy_identity: str | None
    candidate_set_identity: str
    candidate_selection_entries: tuple[CandidateSelectionEntry, ...]
    sampler_identity: str
    policy_rng_factory_identity: str
    terminal_verifier_executable_sha256: str
    terminal_verifier_contract_id: str
    coefficient_identity: str | None
    deployment_design_sha256: str
    allocation_identity: str
    bound_certificate_identity: str
    error_ledger_identity: str
    expected_s: int
    expected_h: int
    expected_l: int
    practical_margin: float
    preference_weight: float
    selection_rule: str
    low_expectation_id: str | None
    low_law_h_id: str | None
    low_law_l_id: str | None
    max_abs_beta: float | None
    a_panel_enabled: bool
    quantitative_target_enabled: bool
    content_sha256: str
    _validation_capability: object | None = dataclass_field(
        default=None,
        repr=False,
        compare=False,
    )
    _artifact_file_sha256: str | None = dataclass_field(
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
            _require_validation_capability(
                self._validation_capability,
                artifact_kind="RootManifest",
                content_sha256=_root_manifest_runtime_fingerprint(self),
            )
        if self._external_pin_capability is not None:
            require_externally_pinned_root_manifest(self)

    def panel_identity(self, panel: str) -> str | None:
        """Return the preregistered panel identity without exposing mutation."""

        try:
            return dict(self.panel_identities)[panel]
        except KeyError as exc:
            raise RivalSchemaError(f"unknown panel {panel!r}") from exc

    @property
    def validated(self) -> bool:
        try:
            require_validated_root_manifest(self)
        except RivalSchemaError:
            return False
        return True

    @property
    def externally_pinned(self) -> bool:
        try:
            require_externally_pinned_root_manifest(self)
        except RivalSchemaError:
            return False
        return True


def _root_manifest_runtime_fingerprint(manifest: RootManifest) -> str:
    payload = asdict(manifest)
    payload.pop("_validation_capability", None)
    payload.pop("_artifact_file_sha256", None)
    payload.pop("_external_pin_capability", None)
    return sha256_hex(
        {
            "schema_id": "cascadiav3.rival_validated_root_manifest_runtime.v1",
            "fields": payload,
        }
    )


def require_validated_root_manifest(manifest: RootManifest) -> None:
    """Reject objects that did not come from :func:`validate_root_manifest`."""

    if not isinstance(manifest, RootManifest):
        raise RivalSchemaError("root manifest must be a RootManifest")
    _require_validation_capability(
        manifest._validation_capability,
        artifact_kind="RootManifest",
        content_sha256=_root_manifest_runtime_fingerprint(manifest),
    )


def _externally_pinned_manifest_runtime_fingerprint(manifest: RootManifest) -> str:
    return sha256_hex(
        {
            "schema_id": "cascadiav3.rival_externally_pinned_root_manifest_runtime.v1",
            "semantic_runtime_sha256": _root_manifest_runtime_fingerprint(manifest),
            "artifact_file_sha256": manifest._artifact_file_sha256,
            "content_sha256": manifest.content_sha256,
        }
    )


def require_externally_pinned_root_manifest(manifest: RootManifest) -> None:
    """Require a manifest loaded under independent canonical byte/content pins."""

    require_validated_root_manifest(manifest)
    if manifest._artifact_file_sha256 is None:
        raise RivalSchemaError(
            "root manifest must be loaded from an externally byte-pinned artifact"
        )
    require_sha256(manifest._artifact_file_sha256, "root manifest artifact_file_sha256")
    _require_validation_capability(
        manifest._external_pin_capability,
        artifact_kind="ExternallyPinnedRootManifest",
        content_sha256=_externally_pinned_manifest_runtime_fingerprint(manifest),
    )


_ROOT_MANIFEST_FIELDS = (
    "schema_id",
    "manifest_id",
    "ruleset_identity",
    "source_revision",
    "root_id",
    "source_game_id",
    "source_game_identity_sha256",
    "root_kind",
    "root_cohort_role",
    "complete_game_seed_role",
    "inference_mode",
    "required_panels",
    "forbidden_panels",
    "panel_identities",
    "beta_cv",
    "multifidelity_claim",
    "incumbent_policy_identity",
    "incumbent_action_id",
    "incumbent_candidate_occurrence_id",
    "rules_menu_hash",
    "incumbent_menu_hash",
    "low_policy_identity",
    "candidate_set_identity",
    "candidate_selection_entries",
    "sampler_identity",
    "policy_rng_factory_identity",
    "terminal_verifier_executable_sha256",
    "terminal_verifier_contract_id",
    "coefficient_identity",
    "deployment_design_sha256",
    "allocation_identity",
    "bound_certificate_identity",
    "error_ledger_identity",
    "expected_s",
    "expected_h",
    "expected_l",
    "practical_margin",
    "preference_weight",
    "selection_rule",
    "low_expectation_id",
    "low_law_h_id",
    "low_law_l_id",
    "max_abs_beta",
    "a_panel_enabled",
    "quantitative_target_enabled",
    "content_sha256",
)


def validate_root_manifest(record: Mapping[str, Any]) -> RootManifest:
    require_schema(record, RIVAL_ROOT_MANIFEST_SCHEMA_ID)
    require_exact_keys(record, required=_ROOT_MANIFEST_FIELDS, where="root manifest")

    root_kind = require_nonempty_string(record["root_kind"], "root_kind")
    if root_kind not in ROOT_KINDS:
        raise RivalSchemaError(f"unknown root_kind: {root_kind!r}")
    cohort = require_nonempty_string(record["root_cohort_role"], "root_cohort_role")
    if cohort not in ROOT_COHORT_ROLES:
        raise RivalSchemaError(f"unknown root_cohort_role: {cohort!r}")
    if record["complete_game_seed_role"] is not None:
        raise RivalSchemaError(
            "complete-game seed roles are not root cohorts and must be null in a root manifest"
        )
    if record["a_panel_enabled"] is not False:
        raise RivalSchemaError("A panel is structurally disabled in Rival v1")
    if record["quantitative_target_enabled"] is not False:
        raise RivalSchemaError("quantitative targets require a separately registered A panel")

    expected_s = require_positive_int(record["expected_s"], "expected_s")
    expected_h = require_positive_int(record["expected_h"], "expected_h")
    beta_cv = require_finite(record["beta_cv"], "beta_cv")
    practical_margin = require_finite(record["practical_margin"], "practical_margin")
    preference_weight = require_finite(record["preference_weight"], "preference_weight")
    if practical_margin < 0.0:
        raise RivalSchemaError("practical_margin must be non-negative")
    if preference_weight <= 0.0:
        raise RivalSchemaError("preference_weight must be positive")
    selection_rule = require_nonempty_string(record["selection_rule"], "selection_rule")
    if selection_rule != "highest_mean_then_lexicographic_action_id":
        raise RivalSchemaError("unsupported or adaptive selection_rule")

    root_id = _require_namespaced_digest(record["root_id"], "root_id", PUBLIC_ROOT_ID_PREFIX)
    incumbent_action_id = _require_namespaced_digest(
        record["incumbent_action_id"], "incumbent_action_id", ACTION_CONTENT_ID_PREFIX
    )
    incumbent_occurrence_id = _require_namespaced_digest(
        record["incumbent_candidate_occurrence_id"],
        "incumbent_candidate_occurrence_id",
        CANDIDATE_OCCURRENCE_ID_PREFIX,
    )
    rules_menu_hash = _require_namespaced_digest(
        record["rules_menu_hash"], "rules_menu_hash", RULES_MENU_HASH_PREFIX
    )
    incumbent_menu_hash = _require_namespaced_digest(
        record["incumbent_menu_hash"],
        "incumbent_menu_hash",
        INCUMBENT_MENU_HASH_PREFIX,
    )
    candidate_entries_raw = record["candidate_selection_entries"]
    if not isinstance(candidate_entries_raw, list) or len(candidate_entries_raw) < 2:
        raise RivalSchemaError(
            "candidate_selection_entries must contain the incumbent and at least one challenger"
        )
    candidate_entries: list[CandidateSelectionEntry] = []
    for index, raw in enumerate(candidate_entries_raw):
        if not isinstance(raw, Mapping):
            raise RivalSchemaError(f"candidate selection entry {index} must be an object")
        require_exact_keys(
            raw,
            required=(
                "candidate_action_occurrence_id",
                "action_content_id",
                "expected_s",
            ),
            where=f"candidate selection entry {index}",
        )
        occurrence_id = _require_namespaced_digest(
            raw["candidate_action_occurrence_id"],
            f"candidate_selection_entries[{index}].candidate_action_occurrence_id",
            CANDIDATE_OCCURRENCE_ID_PREFIX,
        )
        action_content_id = _require_namespaced_digest(
            raw["action_content_id"],
            f"candidate_selection_entries[{index}].action_content_id",
            ACTION_CONTENT_ID_PREFIX,
        )
        allocation = require_positive_int(
            raw["expected_s"],
            f"candidate_selection_entries[{index}].expected_s",
            allow_zero=True,
        )
        candidate_entries.append(
            CandidateSelectionEntry(occurrence_id, action_content_id, allocation)
        )
    occurrence_ids = [row.candidate_action_occurrence_id for row in candidate_entries]
    action_content_ids = [row.action_content_id for row in candidate_entries]
    if len(set(occurrence_ids)) != len(occurrence_ids):
        raise RivalSchemaError("candidate occurrence IDs must be unique and ordered")
    if len(set(action_content_ids)) != len(action_content_ids):
        raise RivalSchemaError("candidate action content IDs must be unique")
    incumbent_matches = [
        row
        for row in candidate_entries
        if row.candidate_action_occurrence_id == incumbent_occurrence_id
    ]
    if len(incumbent_matches) != 1:
        raise RivalSchemaError("incumbent occurrence is absent from the candidate set")
    if incumbent_matches[0].action_content_id != incumbent_action_id:
        raise RivalSchemaError("incumbent occurrence does not carry incumbent action content")
    if incumbent_matches[0].expected_s != 0:
        raise RivalSchemaError("incumbent must receive zero challenger-selection S units")
    if any(
        row.expected_s <= 0
        for row in candidate_entries
        if row.candidate_action_occurrence_id != incumbent_occurrence_id
    ):
        raise RivalSchemaError("every eligible challenger requires a positive S allocation")
    if sum(row.expected_s for row in candidate_entries) != expected_s:
        raise RivalSchemaError("expected_s does not equal the exact per-candidate S allocation")
    expected_candidate_set_identity = candidate_set_identity(candidate_entries)
    if record["candidate_set_identity"] != expected_candidate_set_identity:
        raise RivalSchemaError("candidate_set_identity does not bind the ordered candidate menu")

    inference_mode = require_nonempty_string(record["inference_mode"], "inference_mode")
    required_panels_raw = record["required_panels"]
    forbidden_panels_raw = record["forbidden_panels"]
    if not isinstance(required_panels_raw, list) or not all(
        isinstance(panel, str) for panel in required_panels_raw
    ):
        raise RivalSchemaError("required_panels must be a string list")
    if not isinstance(forbidden_panels_raw, list) or not all(
        isinstance(panel, str) for panel in forbidden_panels_raw
    ):
        raise RivalSchemaError("forbidden_panels must be a string list")
    required_panels = tuple(required_panels_raw)
    forbidden_panels = tuple(forbidden_panels_raw)
    panel_identities_raw = record["panel_identities"]
    if not isinstance(panel_identities_raw, Mapping):
        raise RivalSchemaError("panel_identities must be an object")
    require_exact_keys(
        panel_identities_raw,
        required=("S", "H", "L", "A"),
        where="panel_identities",
    )
    panel_identities: dict[str, str | None] = {}
    for panel in ("S", "H", "L", "A"):
        value = panel_identities_raw[panel]
        if panel in required_panels:
            if not isinstance(value, str) or not value.startswith("sha256:"):
                raise RivalSchemaError(
                    f"required panel_identities.{panel} must use the 'sha256:' wire"
                )
            require_sha256(value, f"panel_identities.{panel}")
            panel_identities[panel] = value
        else:
            if value is not None:
                raise RivalSchemaError(f"non-required panel_identities.{panel} must be null")
            panel_identities[panel] = None
    active_panel_ids = [value for value in panel_identities.values() if value is not None]
    if len(active_panel_ids) != len(set(active_panel_ids)):
        raise RivalSchemaError("panel identities must be unique within a root")
    if inference_mode == "multifidelity":
        if required_panels != ("S", "H", "L") or forbidden_panels != ("A",):
            raise RivalSchemaError("multifidelity v1 requires S,H,L and forbids A")
        if record["multifidelity_claim"] is not True:
            raise RivalSchemaError("multifidelity mode must declare its claim explicitly")
        expected_l = require_positive_int(record["expected_l"], "expected_l")
        low_policy_identity = require_nonempty_string(
            record["low_policy_identity"], "low_policy_identity"
        )
        coefficient_identity = require_nonempty_string(
            record["coefficient_identity"], "coefficient_identity"
        )
        low_expectation_id = require_nonempty_string(
            record["low_expectation_id"], "low_expectation_id"
        )
        low_law_h_id = require_nonempty_string(record["low_law_h_id"], "low_law_h_id")
        low_law_l_id = require_nonempty_string(record["low_law_l_id"], "low_law_l_id")
        max_abs_beta = require_finite(record["max_abs_beta"], "max_abs_beta")
        if max_abs_beta <= 0.0 or abs(beta_cv) > max_abs_beta:
            raise RivalSchemaError("beta_cv exceeds the positive frozen max_abs_beta")
    elif inference_mode == "high_fidelity_only":
        if required_panels != ("S", "H") or forbidden_panels != ("L", "A"):
            raise RivalSchemaError("high-fidelity-only mode requires S,H and forbids L,A")
        if record["multifidelity_claim"] is not False:
            raise RivalSchemaError("high-fidelity-only mode cannot make a multifidelity claim")
        if beta_cv != 0.0:
            raise RivalSchemaError("high-fidelity-only mode fixes beta_cv = 0")
        expected_l = require_positive_int(record["expected_l"], "expected_l", allow_zero=True)
        if expected_l != 0:
            raise RivalSchemaError("high-fidelity-only mode must allocate zero L rows")
        if record["low_policy_identity"] is not None:
            raise RivalSchemaError("high-fidelity-only mode has no low policy identity")
        low_policy_identity = None
        if record["coefficient_identity"] is not None:
            raise RivalSchemaError("high-fidelity-only mode has no control-variate coefficient")
        coefficient_identity = None
        for field in ("low_expectation_id", "low_law_h_id", "low_law_l_id", "max_abs_beta"):
            if record[field] is not None:
                raise RivalSchemaError(f"high-fidelity-only mode requires {field}=null")
        low_expectation_id = None
        low_law_h_id = None
        low_law_l_id = None
        max_abs_beta = None
    else:
        raise RivalSchemaError(f"unknown inference_mode: {inference_mode!r}")
    text_fields = (
        "manifest_id",
        "ruleset_identity",
        "source_revision",
        "source_game_id",
        "source_game_identity_sha256",
        "incumbent_policy_identity",
        "candidate_set_identity",
        "sampler_identity",
        "policy_rng_factory_identity",
        "terminal_verifier_executable_sha256",
        "terminal_verifier_contract_id",
        "deployment_design_sha256",
        "allocation_identity",
        "bound_certificate_identity",
        "error_ledger_identity",
    )
    values = {field: require_nonempty_string(record[field], field) for field in text_fields}
    digest_fields = (
        "ruleset_identity",
        "incumbent_policy_identity",
        "candidate_set_identity",
        "sampler_identity",
        "source_game_identity_sha256",
        "policy_rng_factory_identity",
        "terminal_verifier_executable_sha256",
        "allocation_identity",
        "bound_certificate_identity",
        "error_ledger_identity",
        "deployment_design_sha256",
    )
    if low_policy_identity is not None:
        digest_fields += ("low_policy_identity",)
    if coefficient_identity is not None:
        digest_fields += ("coefficient_identity",)
    for field in digest_fields:
        value = (
            low_policy_identity
            if field == "low_policy_identity"
            else (coefficient_identity if field == "coefficient_identity" else values[field])
        )
        if not isinstance(value, str) or not value.startswith("sha256:"):
            raise RivalSchemaError(f"{field} must use the 'sha256:' identity wire")
        require_sha256(value, field)
    deployment_identity = deployment_design_identity(record)
    if values["deployment_design_sha256"] != deployment_identity:
        raise RivalSchemaError(
            "deployment_design_sha256 does not bind the full pre-coefficient design"
        )
    content_hash = verify_content_hash(record)
    manifest = RootManifest(
        manifest_id=values["manifest_id"],
        ruleset_identity=values["ruleset_identity"],
        source_revision=values["source_revision"],
        root_id=root_id,
        source_game_id=values["source_game_id"],
        source_game_identity_sha256=values["source_game_identity_sha256"],
        root_kind=root_kind,
        root_cohort_role=cohort,
        inference_mode=inference_mode,
        required_panels=required_panels,
        forbidden_panels=forbidden_panels,
        panel_identities=tuple(panel_identities.items()),
        beta_cv=beta_cv,
        multifidelity_claim=bool(record["multifidelity_claim"]),
        incumbent_policy_identity=values["incumbent_policy_identity"],
        incumbent_action_id=incumbent_action_id,
        incumbent_candidate_occurrence_id=incumbent_occurrence_id,
        rules_menu_hash=rules_menu_hash,
        incumbent_menu_hash=incumbent_menu_hash,
        low_policy_identity=low_policy_identity,
        candidate_set_identity=values["candidate_set_identity"],
        candidate_selection_entries=tuple(candidate_entries),
        sampler_identity=values["sampler_identity"],
        policy_rng_factory_identity=values["policy_rng_factory_identity"],
        terminal_verifier_executable_sha256=values["terminal_verifier_executable_sha256"],
        terminal_verifier_contract_id=values["terminal_verifier_contract_id"],
        coefficient_identity=coefficient_identity,
        deployment_design_sha256=values["deployment_design_sha256"],
        allocation_identity=values["allocation_identity"],
        bound_certificate_identity=values["bound_certificate_identity"],
        error_ledger_identity=values["error_ledger_identity"],
        expected_s=expected_s,
        expected_h=expected_h,
        expected_l=expected_l,
        practical_margin=practical_margin,
        preference_weight=preference_weight,
        selection_rule=selection_rule,
        low_expectation_id=low_expectation_id,
        low_law_h_id=low_law_h_id,
        low_law_l_id=low_law_l_id,
        max_abs_beta=max_abs_beta,
        a_panel_enabled=False,
        quantitative_target_enabled=False,
        content_sha256=content_hash,
    )
    return replace(
        manifest,
        _validation_capability=_issue_validation_capability(
            "RootManifest",
            _root_manifest_runtime_fingerprint(manifest),
        ),
    )


def load_json_object(path: str | Path) -> dict[str, Any]:
    """Read an unpinned manifest-shaped object for fixture construction only.

    Scientific callers must use :func:`load_root_manifest`; a self-carried
    content hash is not an external root of trust.
    """

    return read_strict_json_object(path, field="root manifest")


def load_root_manifest(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_content_sha256: str,
) -> RootManifest:
    """Load one canonical manifest under independent byte and content pins.

    The byte pin protects the exact preregistered file, while the content pin
    protects its semantic identity.  Neither value is learned from the file
    being admitted.
    """

    record = read_pinned_canonical_json_object(
        path,
        expected_file_sha256=expected_file_sha256,
        field="root manifest",
    )
    expected_content = require_sha256(
        expected_content_sha256,
        "expected_content_sha256",
    )
    observed_content = verify_content_hash(record)
    if observed_content != expected_content:
        raise RivalSchemaError(
            "root manifest differs from its externally preregistered content pin"
        )
    manifest = validate_root_manifest(record)
    pinned = replace(
        manifest,
        _artifact_file_sha256=(
            "sha256:" + require_sha256(expected_file_sha256, "expected_file_sha256")
        ),
    )
    return replace(
        pinned,
        _external_pin_capability=_issue_validation_capability(
            "ExternallyPinnedRootManifest",
            _externally_pinned_manifest_runtime_fingerprint(pinned),
        ),
    )
