#!/usr/bin/env python3
"""Create private R2-MAP gate seed domains and bootstrap gate inputs.

This is control-plane bookkeeping only. It does not import or execute game,
model, training, or evaluation code. The Rust container validates every
generated contract before benchmark execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

CAMPAIGN_ID = "r2-map-expert-iteration-v1"
SEED_DOMAIN_SCHEMA = "cascadia.r2-map.protected-seed-domain.v2"
SEED_REGISTRY_SCHEMA = "cascadia.r2-map.protected-seed-registry.v2"
CONTRACT_SCHEMA = "cascadia.r2-map.focal-contract.v4"
FIELD_SCHEMA = "cascadia.r2-map.opponent-field.v4"
IMPLEMENTATION_SCHEMA = "cascadia.r2-map.implementation-binding.v1.1"
CONTRACT_REVISION = "sequential-public-market-v1.1"
QUALIFIED_EXACT_NNUE_CHECKPOINT_ID = (
    "canonical-action-legacy-exact-mlx-v1-k32-r600-lmr-no-paid-prelude"
)
CROSS_ARCH_INFERENCE_SETTINGS_ID = (
    "r2-map-exhaustive-argmax-vs-qualified-exact-nnue-k32-r600-v1"
)
DEFAULT_REGISTRATION_SHA256 = "429f031a1748227caed047d90511318bc39b07f2ae9ba0ba20da333432de1cd5"
DOMAIN_SPECS = (
    ("r2-map-strength-blinded-smoke-20-v2", "strength-blinded-smoke", 20),
    ("r2-map-fixed-development-gate-250-v2", "development", 250),
    ("r2-map-final-domain-1000-v2", "final-strength", 1_000),
)
STAGES = {
    "smoke": ("strength-blinded-smoke", 20),
    "development": ("development", 250),
}
FOCAL_REPORT_SCHEMA = "cascadia.r2-map.focal-report.v4"
SCHEDULER_REPORT_SCHEMA = "cascadia.r2-map.scheduler-provenance.v1"
FOCAL_MAX_RSS_BYTES = 4 * 1024 * 1024 * 1024


class GateControlError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: Any) -> str:
    return _sha256(_canonical(value))


def _write_json(path: Path, value: Mapping[str, Any], *, mode: int) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = _encoded_json(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise GateControlError(f"refusing to replace different artifact: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fchmod(handle.fileno(), mode)
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def _encoded_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, indent=2).encode("utf-8") + b"\n"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _domain_commitment(domain_id: str, purpose: str, seeds: Sequence[str]) -> str:
    return _digest({"domain_id": domain_id, "purpose": purpose, "seeds": list(seeds)})


def create_registry(
    private_directory: Path,
    public_registry: Path,
    *,
    entropy: Callable[[int], bytes] = secrets.token_bytes,
) -> dict[str, Any]:
    if private_directory.parent.resolve() != public_registry.parent.resolve():
        raise GateControlError("private domains and public registry must share one control root")
    public_registry.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(public_registry.parent, 0o700)
    private_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(private_directory, 0o700)
    public_domains = []
    for domain_id, purpose, count in DOMAIN_SPECS:
        seeds = [entropy(32).hex() for _ in range(count)]
        if len(set(seeds)) != count or any(len(seed) != 64 for seed in seeds):
            raise GateControlError("entropy source produced duplicate or malformed seeds")
        commitment = _domain_commitment(domain_id, purpose, seeds)
        private = {
            "schema_id": SEED_DOMAIN_SCHEMA,
            "schema_version": 2,
            "campaign_id": CAMPAIGN_ID,
            "domain_id": domain_id,
            "purpose": purpose,
            "count": count,
            "domain_commitment_sha256": commitment,
            "opened": False,
            "seeds": seeds,
        }
        _write_json(private_directory / f"{domain_id}.json", private, mode=0o600)
        public_domains.append(
            {
                "domain_id": domain_id,
                "purpose": purpose,
                "count": count,
                "domain_commitment_sha256": commitment,
                "opened": False,
                "seed_material_present": False,
            }
        )
    registry = {
        "schema_id": SEED_REGISTRY_SCHEMA,
        "schema_version": 2,
        "campaign_id": CAMPAIGN_ID,
        "supersedes_unmaterialized_v1_domains": True,
        "supersession_reason": (
            "The frozen v1 commitments had no retained seed material and were never opened; "
            "v2 freezes usable values before any candidate result is observed."
        ),
        "domains": public_domains,
    }
    registry["registry_sha256"] = _digest(registry)
    _write_json(public_registry, registry, mode=0o600)
    verify_registry(private_directory, public_registry)
    return registry


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise GateControlError(f"cannot read JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise GateControlError(f"JSON artifact is not an object: {path}")
    return value


def _verify_domain(value: Mapping[str, Any], *, expected: tuple[str, str, int]) -> None:
    domain_id, purpose, count = expected
    required = {
        "schema_id",
        "schema_version",
        "campaign_id",
        "domain_id",
        "purpose",
        "count",
        "domain_commitment_sha256",
        "opened",
        "seeds",
    }
    if set(value) != required:
        raise GateControlError(f"protected domain fields differ: {domain_id}")
    seeds = value["seeds"]
    if (
        value["schema_id"] != SEED_DOMAIN_SCHEMA
        or value["schema_version"] != 2
        or value["campaign_id"] != CAMPAIGN_ID
        or value["domain_id"] != domain_id
        or value["purpose"] != purpose
        or value["count"] != count
        or value["opened"] is not False
        or not isinstance(seeds, list)
        or len(seeds) != count
        or len(set(seeds)) != count
        or any(
            not isinstance(seed, str)
            or len(seed) != 64
            or any(character not in "0123456789abcdef" for character in seed)
            for seed in seeds
        )
    ):
        raise GateControlError(f"protected domain is malformed: {domain_id}")
    expected_commitment = _domain_commitment(domain_id, purpose, seeds)
    if value["domain_commitment_sha256"] != expected_commitment:
        raise GateControlError(f"protected domain commitment differs: {domain_id}")


def verify_registry(private_directory: Path, public_registry: Path) -> dict[str, Any]:
    registry = _read_json(public_registry)
    registry_without_hash = dict(registry)
    observed_registry_hash = registry_without_hash.pop("registry_sha256", None)
    if (
        registry.get("schema_id") != SEED_REGISTRY_SCHEMA
        or registry.get("schema_version") != 2
        or registry.get("campaign_id") != CAMPAIGN_ID
        or observed_registry_hash != _digest(registry_without_hash)
    ):
        raise GateControlError("public protected-domain registry is malformed")
    expected_public = []
    for spec in DOMAIN_SPECS:
        domain = _read_json(private_directory / f"{spec[0]}.json")
        _verify_domain(domain, expected=spec)
        expected_public.append(
            {
                "domain_id": spec[0],
                "purpose": spec[1],
                "count": spec[2],
                "domain_commitment_sha256": domain["domain_commitment_sha256"],
                "opened": False,
                "seed_material_present": False,
            }
        )
    if registry.get("domains") != expected_public:
        raise GateControlError("public registry differs from private protected domains")
    if "seeds" in json.dumps(registry):
        raise GateControlError("public registry exposes protected seed material")
    return registry


def _implementation_binding(reference_manifest: Path, registration_sha256: str) -> dict[str, Any]:
    manifest = _read_json(reference_manifest)
    identity = manifest.get("implementation_identity")
    if (
        manifest.get("schema_id") != "cascadia.r2-map.reference-panel-manifest.v1.1"
        or manifest.get("campaign_id") != CAMPAIGN_ID
        or not isinstance(identity, dict)
        or len(registration_sha256) != 64
    ):
        raise GateControlError("W0 v1.1 reference manifest or registration identity is invalid")
    names = (
        "maximum_width_panel_sha256",
        "replay_pinecone_panel_sha256",
        "source_bundle_sha256",
        "serving_protocol_schema_sha256",
        "market_action_schema_blake3",
        "request_schema_blake3",
        "response_schema_blake3",
        "protocol_fixture_canonical_blake3",
        "protocol_fixture_file_blake3",
        "model_schema_sha256",
        "open_reference_seed_domain_id",
    )
    if any(not isinstance(identity.get(name), str) for name in names):
        raise GateControlError("W0 v1.1 implementation identity is incomplete")

    def decode(value: str) -> list[int]:
        return list(bytes.fromhex(value))

    return {
        "schema_id": IMPLEMENTATION_SCHEMA,
        "contract_revision": CONTRACT_REVISION,
        "w0_registration_sha256": registration_sha256,
        "reference_manifest_sha256": manifest["manifest_sha256"],
        **{name: identity[name] for name in names},
        "protocols": {
            "collector_hash": decode(identity["replay_pinecone_panel_sha256"]),
            "source_hash": decode(identity["source_bundle_sha256"]),
            "serving_protocol_hash": decode(identity["serving_protocol_schema_sha256"]),
        },
    }


def _admit_strength_blinded_smoke(
    campaign: Path,
    *,
    checkpoint_id: str,
    image_digest: str,
    candidate_freeze_receipt_sha256: str,
    exact_weights_sha256: str,
) -> dict[str, Any]:
    contract_path = campaign / "contract.json"
    report_path = campaign / "reports/focal-benchmark.json"
    scheduler_path = campaign / "reports/scheduler-provenance.json"
    contract = _read_json(contract_path)
    report = _read_json(report_path)
    scheduler = _read_json(scheduler_path)
    scheduler_payload = dict(scheduler)
    scheduler_claimed = scheduler_payload.pop("report_sha256", None)
    execution = contract.get("execution_binding")
    result = report.get("result")
    statistics = result.get("statistics") if isinstance(result, dict) else None
    expected_statistics = {
        "schema_version",
        "protocol_id",
        "stage",
        "strength_outputs_blinded",
        "pairs",
        "physical_games",
        "wall_seconds",
        "games_per_second",
        "peak_rss_bytes",
        "maximum_swap_delta_bytes",
        "all_clean_shutdowns",
        "all_pinecone_conservation_checks_passed",
    }
    work_items = report.get("work_items")
    expected_items = {f"pair-{index:04}" for index in range(20)}
    if (
        contract.get("schema_id") != CONTRACT_SCHEMA
        or contract.get("stage") != "strength-blinded-smoke"
        or contract.get("pair_count") != 20
        or contract.get("candidate_checkpoint_id") != checkpoint_id
        or not isinstance(execution, dict)
        or execution.get("image_digest") != image_digest
        or execution.get("candidate_freeze_receipt_sha256")
        != candidate_freeze_receipt_sha256
        or execution.get("exact_weights_sha256") != exact_weights_sha256
        or report.get("schema_id") != FOCAL_REPORT_SCHEMA
        or report.get("contract_sha256") != _sha256_file(contract_path)
        or not isinstance(result, dict)
        or result.get("kind") != "strength-blinded-smoke"
        or not isinstance(statistics, dict)
        or set(statistics) != expected_statistics
        or statistics.get("stage") != "strength-blinded-smoke"
        or statistics.get("strength_outputs_blinded") is not True
        or statistics.get("pairs") != 20
        or statistics.get("physical_games") != 40
        or statistics.get("all_clean_shutdowns") is not True
        or statistics.get("all_pinecone_conservation_checks_passed") is not True
        or not isinstance(statistics.get("peak_rss_bytes"), int)
        or statistics["peak_rss_bytes"] > FOCAL_MAX_RSS_BYTES
        or not isinstance(statistics.get("maximum_swap_delta_bytes"), int)
        or statistics["maximum_swap_delta_bytes"] > 0
        or not isinstance(work_items, list)
        or len(work_items) != 20
    ):
        raise GateControlError("strength-blinded smoke did not pass admission")
    observed_items = set()
    for item in work_items:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("work_item_id"), str)
            or item.get("pairs") != 1
            or item.get("physical_games") != 2
            or item.get("all_clean_shutdowns") is not True
            or item.get("all_pinecone_conservation_checks_passed") is not True
            or not isinstance(item.get("peak_rss_bytes"), int)
            or item["peak_rss_bytes"] > FOCAL_MAX_RSS_BYTES
            or not isinstance(item.get("maximum_swap_delta_bytes"), int)
            or item["maximum_swap_delta_bytes"] > 0
        ):
            raise GateControlError("strength-blinded smoke work item failed admission")
        observed_items.add(item["work_item_id"])
    scheduler_items = scheduler.get("work_items")
    if (
        observed_items != expected_items
        or scheduler.get("schema_id") != SCHEDULER_REPORT_SCHEMA
        or scheduler.get("stage") != "smoke"
        or scheduler.get("image_digest") != image_digest
        or scheduler_claimed != _digest(scheduler_payload)
        or not isinstance(scheduler_items, list)
        or {item.get("item_id") for item in scheduler_items if isinstance(item, dict)}
        != expected_items
        or not isinstance(scheduler.get("retry_count"), int)
        or scheduler["retry_count"] < 0
    ):
        raise GateControlError("strength-blinded smoke scheduler provenance failed admission")
    admission: dict[str, Any] = {
        "schema_id": "cascadia.r2-map.strength-blinded-smoke-admission.v1",
        "campaign_id": CAMPAIGN_ID,
        "candidate_checkpoint_id": checkpoint_id,
        "image_digest": image_digest,
        "smoke_contract_sha256": _sha256_file(contract_path),
        "smoke_report_sha256": _sha256_file(report_path),
        "smoke_scheduler_provenance_sha256": _sha256_file(scheduler_path),
        "pairs": 20,
        "physical_games": 40,
        "strength_outputs_blinded": True,
        "admitted": True,
    }
    admission["receipt_sha256"] = _digest(admission)
    return admission


def prepare_bootstrap_gate(
    *,
    stage: str,
    checkpoint_id: str,
    private_domain: Path,
    output_directory: Path,
    reference_manifest: Path,
    image_digest: str,
    candidate_freeze_receipt: Path,
    exact_weights: Path,
    smoke_campaign_directory: Path | None = None,
    registration_sha256: str = DEFAULT_REGISTRATION_SHA256,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if stage not in STAGES:
        raise GateControlError(f"unsupported bootstrap gate stage: {stage}")
    stage_name, expected_count = STAGES[stage]
    if not checkpoint_id or checkpoint_id == "greedy-v1":
        raise GateControlError("candidate checkpoint identity is empty or aliases greedy")
    if re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", image_digest) is None:
        raise GateControlError("candidate image must be an immutable OCI digest")
    freeze = _read_json(candidate_freeze_receipt)
    if freeze.get("checkpoint_id") != checkpoint_id:
        raise GateControlError("candidate freeze receipt checkpoint differs")
    if not exact_weights.is_file():
        raise GateControlError("qualified exact weights are absent")
    domain = _read_json(private_domain)
    matching_spec = next(
        (spec for spec in DOMAIN_SPECS if spec[0] == domain.get("domain_id")), None
    )
    if matching_spec is None:
        raise GateControlError("protected domain is not registered")
    _verify_domain(domain, expected=matching_spec)
    if domain["purpose"] != stage_name or domain["count"] != expected_count:
        raise GateControlError("protected domain purpose/count differs from gate stage")
    binding = _implementation_binding(reference_manifest, registration_sha256)
    field_id = f"{domain['domain_id']}-all-greedy-field-v1"
    benchmark_id = f"r2-map-bootstrap-{stage_name}-v1"
    assignments = []
    for pair_index, seed in enumerate(domain["seeds"]):
        focal_seat = pair_index % 4
        assignments.append(
            {
                "pair_index": pair_index,
                "game_seed": list(bytes.fromhex(seed)),
                "seed_domain_id": f"{domain['domain_id']}:{pair_index:04d}",
                "focal_seat": focal_seat,
                "opponents": [
                    {"seat": seat, "checkpoint_id": "greedy-v1"}
                    for seat in range(4)
                    if seat != focal_seat
                ],
            }
        )
    field = {
        "schema_version": 4,
        "schema_id": FIELD_SCHEMA,
        "manifest_id": field_id,
        "assignments": assignments,
    }
    execution_binding = {
        "image_digest": image_digest,
        "candidate_freeze_receipt_sha256": _sha256_file(candidate_freeze_receipt),
        "exact_weights_sha256": _sha256_file(exact_weights),
        "opponent_field_sha256": _sha256(_encoded_json(field)),
    }
    if stage == "development":
        if smoke_campaign_directory is None:
            raise GateControlError("development gate requires admitted blinded smoke")
        admission = _admit_strength_blinded_smoke(
            smoke_campaign_directory,
            checkpoint_id=checkpoint_id,
            image_digest=image_digest,
            candidate_freeze_receipt_sha256=execution_binding[
                "candidate_freeze_receipt_sha256"
            ],
            exact_weights_sha256=execution_binding["exact_weights_sha256"],
        )
        _write_json(
            output_directory / "smoke-admission-receipt.json", admission, mode=0o600
        )
        execution_binding["smoke_admission_receipt_sha256"] = _sha256_file(
            output_directory / "smoke-admission-receipt.json"
        )
    elif smoke_campaign_directory is not None:
        raise GateControlError("smoke gate may not consume a prior smoke admission")
    contract = {
        "schema_version": 4,
        "schema_id": CONTRACT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "benchmark_id": benchmark_id,
        "iteration": 0,
        "stage": stage_name,
        "pair_count": expected_count,
        "execution_partition": {"kind": "scheduler-managed-pairs"},
        "candidate_checkpoint_id": checkpoint_id,
        "control_checkpoint_id": QUALIFIED_EXACT_NNUE_CHECKPOINT_ID,
        "opponent_field_manifest_id": field_id,
        "inference_settings_id": CROSS_ARCH_INFERENCE_SETTINGS_ID,
        "implementation_binding": binding,
        "execution_binding": execution_binding,
    }
    _write_json(output_directory / "contract.json", contract, mode=0o600)
    _write_json(output_directory / "opponent-field.json", field, mode=0o600)
    receipt = {
        "schema_id": "cascadia.r2-map.bootstrap-gate-input-receipt.v2",
        "schema_version": 2,
        "campaign_id": CAMPAIGN_ID,
        "stage": stage_name,
        "candidate_checkpoint_id": checkpoint_id,
        "control_checkpoint_id": QUALIFIED_EXACT_NNUE_CHECKPOINT_ID,
        "domain_id": domain["domain_id"],
        "domain_commitment_sha256": domain["domain_commitment_sha256"],
        "contract_sha256": _sha256((output_directory / "contract.json").read_bytes()),
        "opponent_field_sha256": _sha256((output_directory / "opponent-field.json").read_bytes()),
        "scores_opened": False,
    }
    _write_json(output_directory / "input-receipt.json", receipt, mode=0o600)
    return contract, field


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create-registry")
    create.add_argument("--private-directory", type=Path, required=True)
    create.add_argument("--public-registry", type=Path, required=True)
    verify = subparsers.add_parser("verify-registry")
    verify.add_argument("--private-directory", type=Path, required=True)
    verify.add_argument("--public-registry", type=Path, required=True)
    prepare = subparsers.add_parser("prepare-bootstrap")
    prepare.add_argument("--stage", choices=sorted(STAGES), required=True)
    prepare.add_argument("--checkpoint-id", required=True)
    prepare.add_argument("--private-domain", type=Path, required=True)
    prepare.add_argument("--output-directory", type=Path, required=True)
    prepare.add_argument("--reference-manifest", type=Path, required=True)
    prepare.add_argument("--image-digest", required=True)
    prepare.add_argument("--candidate-freeze-receipt", type=Path, required=True)
    prepare.add_argument("--exact-weights", type=Path, required=True)
    prepare.add_argument("--smoke-campaign-directory", type=Path)
    prepare.add_argument("--registration-sha256", default=DEFAULT_REGISTRATION_SHA256)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.command == "create-registry":
        value = create_registry(arguments.private_directory, arguments.public_registry)
    elif arguments.command == "verify-registry":
        value = verify_registry(arguments.private_directory, arguments.public_registry)
    else:
        contract, field = prepare_bootstrap_gate(
            stage=arguments.stage,
            checkpoint_id=arguments.checkpoint_id,
            private_domain=arguments.private_domain,
            output_directory=arguments.output_directory,
            reference_manifest=arguments.reference_manifest,
            image_digest=arguments.image_digest,
            candidate_freeze_receipt=arguments.candidate_freeze_receipt,
            exact_weights=arguments.exact_weights,
            smoke_campaign_directory=arguments.smoke_campaign_directory,
            registration_sha256=arguments.registration_sha256,
        )
        value = {
            "contract": contract["benchmark_id"],
            "pairs": contract["pair_count"],
            "field": field["manifest_id"],
        }
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
