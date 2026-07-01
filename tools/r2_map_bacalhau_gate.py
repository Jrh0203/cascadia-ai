#!/usr/bin/env python3
"""Run the R2-MAP blinded smoke or fixed-250 gate on the Bacalhau fabric.

Every benchmark pair is one independent scheduler-managed work item. The
controller never chooses a physical node. John1 remains authoritative: inputs
are frozen and content addressed, worker outputs are validated by
``cascadia_cluster``, and the final campaign is assembled and aggregated in a
separate container job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

from cascadia_cluster import (
    ClusterClient,
    ContainerInput,
    JobStatus,
    ObjectStoreClient,
    ObjectStoreConfig,
    Resources,
)

CONTRACT_SCHEMA = "cascadia.r2-map.focal-contract.v4"
FIELD_SCHEMA = "cascadia.r2-map.opponent-field.v4"
STAGES = {
    "smoke": ("strength-blinded-smoke", 20),
    "development": ("development", 250),
}
WORK_ITEM_SCRIPT = r"""
set -eu
work_item="$1"
mkdir -p /input /outputs/campaign/receipts/"$work_item" /outputs/campaign/work-item-summaries
tar -xf /inputs/r2-gate/r2-gate.tar -C /input
# These variables belong to the outer artifact wrapper, not the scientific
# policy process. Keep them in the entrypoint parent for manifest publication
# while preventing the exact policy's fail-closed CASCADIA_* audit from
# mistaking transport metadata for an inference setting.
unset CASCADIA_APPLICATION_METADATA_JSON CASCADIA_OUTPUT_ROOT \
  CASCADIA_PROTOCOL_VERSION CASCADIA_RETRYABLE_EXIT_CODES
r2-map-cross-arch-focal \
  --root /input/campaign \
  --work-item "$work_item" \
  --r2-bundle /input/r2-run/bundle.json \
  --r2-backend-parity-receipt /input/r2-backend-parity.json \
  --r2-python /usr/bin/python3 \
  --r2-python-path /opt/cascadia/repo/python \
  --exact-weights /input/exact-weights.bin \
  --exact-rollouts 600
cp /input/campaign/contract.json /outputs/campaign/contract.json
cp /input/campaign/opponent-field.json /outputs/campaign/opponent-field.json
if [ -f /input/campaign/smoke-admission-receipt.json ]; then
  cp /input/campaign/smoke-admission-receipt.json \
    /outputs/campaign/smoke-admission-receipt.json
fi
cp /input/campaign/work-item-summaries/"$work_item".json \
  /outputs/campaign/work-item-summaries/"$work_item".json
cp /input/campaign/receipts/"$work_item"/pair-*.json /outputs/campaign/receipts/"$work_item"/
""".strip()
AGGREGATE_SCRIPT = r"""
set -eu
wall_seconds="$1"
mkdir -p /input /outputs
tar -xf /inputs/r2-campaign/r2-campaign.tar -C /input
r2-map-focal-control aggregate --root /input/campaign --wall-seconds "$wall_seconds"
cp -R /input/campaign /outputs/campaign
""".strip()


class GateFabricError(RuntimeError):
    pass


def expected_work_items(stage: str) -> tuple[str, ...]:
    return tuple(f"pair-{pair_index:04}" for pair_index in range(STAGES[stage][1]))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise GateFabricError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise GateFabricError(f"JSON artifact is not an object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _write_json_once(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    _write_bytes_once(path, encoded)


def _write_bytes_once(path: Path, encoded: bytes) -> None:
    if path.exists():
        if path.read_bytes() != encoded:
            raise GateFabricError(f"refusing to replace different artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_replace(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _same_file(left: Path, right: Path) -> bool:
    return left.stat().st_size == right.stat().st_size and _sha256_file(left) == _sha256_file(right)


def _copy_exact(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or not _same_file(source, destination):
            raise GateFabricError(f"refusing to replace different campaign artifact: {destination}")
        return
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    shutil.copyfile(source, temporary)
    if not _same_file(source, temporary):
        temporary.unlink(missing_ok=True)
        raise GateFabricError(f"campaign artifact changed during copy: {source}")
    os.replace(temporary, destination)


def _validate_gate_inputs(
    gate_directory: Path, stage: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = _read_json(gate_directory / "contract.json")
    field = _read_json(gate_directory / "opponent-field.json")
    expected_stage, expected_pairs = STAGES[stage]
    execution = contract.get("execution_binding")
    expected_execution_keys = {
        "image_digest",
        "candidate_freeze_receipt_sha256",
        "exact_weights_sha256",
        "opponent_field_sha256",
    }
    if stage == "development":
        expected_execution_keys.add("smoke_admission_receipt_sha256")
    if (
        contract.get("schema_id") != CONTRACT_SCHEMA
        or contract.get("stage") != expected_stage
        or contract.get("pair_count") != expected_pairs
        or contract.get("execution_partition") != {"kind": "scheduler-managed-pairs"}
        or field.get("schema_id") != FIELD_SCHEMA
        or field.get("manifest_id") != contract.get("opponent_field_manifest_id")
        or not isinstance(execution, dict)
        or set(execution) != expected_execution_keys
        or re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", str(execution.get("image_digest", "")))
        is None
    ):
        raise GateFabricError("gate inputs differ from the registered topology-free stage")
    assignments = field.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != expected_pairs:
        raise GateFabricError("opponent field has incomplete pair coverage")
    indices = set()
    for assignment in assignments:
        if not isinstance(assignment, dict):
            raise GateFabricError("opponent field assignment is malformed")
        pair_index = assignment.get("pair_index")
        if not isinstance(pair_index, int) or pair_index in indices:
            raise GateFabricError("opponent field repeats or malforms a pair index")
        forbidden = {"executor_shard", "host", "node", "compatible_hosts"}
        if forbidden.intersection(assignment):
            raise GateFabricError("opponent field contains topology-bearing assignment data")
        indices.add(pair_index)
    if indices != set(range(expected_pairs)):
        raise GateFabricError("opponent field pair indices are not contiguous")
    if execution["opponent_field_sha256"] != _sha256_file(gate_directory / "opponent-field.json"):
        raise GateFabricError("opponent field content differs from its execution binding")
    admission = gate_directory / "smoke-admission-receipt.json"
    if stage == "development" and (
        not admission.is_file()
        or execution["smoke_admission_receipt_sha256"] != _sha256_file(admission)
    ):
        raise GateFabricError("development gate lacks its blinded-smoke admission binding")
    if stage == "smoke" and admission.exists():
        raise GateFabricError("smoke gate unexpectedly contains an admission receipt")
    return contract, field


def _add_tree(archive: tarfile.TarFile, source: Path, target: str) -> None:
    paths = [source, *sorted(source.rglob("*"))] if source.is_dir() else [source]
    for path in paths:
        if path.is_symlink():
            raise GateFabricError(f"input bundle may not contain symlinks: {path}")
        relative = Path(target) if path == source else Path(target) / path.relative_to(source)
        info = archive.gettarinfo(str(path), arcname=relative.as_posix())
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        info.mtime = 0
        if info.isfile():
            with path.open("rb") as stream:
                archive.addfile(info, stream)
        elif info.isdir():
            archive.addfile(info)
        else:
            raise GateFabricError(f"input bundle contains unsupported file type: {path}")


def build_gate_archive(
    *,
    gate_directory: Path,
    candidate_freeze: Path,
    exact_weights: Path,
    stage: str,
    destination: Path,
) -> dict[str, Any]:
    contract, _field = _validate_gate_inputs(gate_directory, stage)
    freeze = _read_json(candidate_freeze / "freeze-receipt.json")
    if freeze.get("checkpoint_id") != contract.get("candidate_checkpoint_id"):
        raise GateFabricError("candidate freeze and gate contract checkpoint differ")
    execution = contract["execution_binding"]
    if execution["candidate_freeze_receipt_sha256"] != _sha256_file(
        candidate_freeze / "freeze-receipt.json"
    ) or execution["exact_weights_sha256"] != _sha256_file(exact_weights):
        raise GateFabricError("gate execution binding differs from candidate or exact weights")
    required = (
        candidate_freeze / "r2-run" / "bundle.json",
        candidate_freeze / "r2-backend-parity.json",
        exact_weights,
    )
    if any(not path.is_file() for path in required):
        raise GateFabricError("gate input bundle is incomplete")
    workspace = Path(tempfile.mkdtemp(prefix="r2-gate-input-"))
    try:
        campaign = workspace / "campaign"
        for relative in ("work-item-summaries", "reports", "projections"):
            (campaign / relative).mkdir(parents=True)
        for pair_index in range(contract["pair_count"]):
            (campaign / "receipts" / f"pair-{pair_index:04}").mkdir(parents=True)
        shutil.copyfile(gate_directory / "contract.json", campaign / "contract.json")
        shutil.copyfile(gate_directory / "opponent-field.json", campaign / "opponent-field.json")
        admission = gate_directory / "smoke-admission-receipt.json"
        if admission.is_file():
            shutil.copyfile(admission, campaign / admission.name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(destination, "w", format=tarfile.PAX_FORMAT) as archive:
            _add_tree(archive, campaign, "campaign")
            _add_tree(archive, candidate_freeze / "r2-run", "r2-run")
            _add_tree(
                archive,
                candidate_freeze / "r2-backend-parity.json",
                "r2-backend-parity.json",
            )
            _add_tree(archive, exact_weights, "exact-weights.bin")
        return {
            "schema_id": "cascadia.r2-map.bacalhau-gate-input.v1",
            "stage": stage,
            "image_digest": execution["image_digest"],
            "checkpoint_id": contract["candidate_checkpoint_id"],
            "archive_sha256": _sha256_file(destination),
            "archive_bytes": destination.stat().st_size,
            "contract_sha256": _sha256_file(gate_directory / "contract.json"),
            "field_sha256": _sha256_file(gate_directory / "opponent-field.json"),
            "exact_weights_sha256": _sha256_file(exact_weights),
            "freeze_receipt_sha256": _sha256_file(candidate_freeze / "freeze-receipt.json"),
        }
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def _store() -> ObjectStoreClient:
    try:
        access_key = os.environ["AWS_ACCESS_KEY_ID"]
        secret_key = os.environ["AWS_SECRET_ACCESS_KEY"]
    except KeyError as error:
        raise GateFabricError(f"missing object-store environment: {error.args[0]}") from error
    return ObjectStoreClient(
        ObjectStoreConfig(
            endpoint=os.environ.get("AWS_ENDPOINT_URL_S3", "http://100.110.109.6:9000"),
            access_key=access_key,
            secret_key=secret_key,
        )
    )


def _client(state_directory: Path, artifact_directory: Path) -> ClusterClient:
    store = _store()
    store.ensure_bucket(store.config.input_bucket)
    store.ensure_bucket(store.config.result_bucket)
    return ClusterClient(
        os.environ.get("BACALHAU_ENDPOINT", "http://100.110.109.6:1234"),
        state_directory=state_directory,
        object_store=store,
        artifact_directory=artifact_directory,
    )


def submit_gate(
    *,
    image: str,
    archive: Path,
    input_receipt: dict[str, Any],
    stage: str,
    request_id: str,
    state_directory: Path,
    artifact_directory: Path,
):
    if (
        input_receipt.get("stage") != stage
        or input_receipt.get("archive_sha256") != _sha256_file(archive)
        or input_receipt.get("image_digest") != image
    ):
        raise GateFabricError("gate input receipt differs from the submitted archive")
    store = _store()
    reference = store.stage_file(archive, target="/inputs/r2-gate")
    client = _client(state_directory, artifact_directory)
    _validate_live_gate_fabric(client.api.nodes())
    work_items = expected_work_items(stage)
    jobs = [
        ContainerInput(
            work_item,
            args=(
                "/bin/sh",
                "-eu",
                "-c",
                WORK_ITEM_SCRIPT,
                "r2-map-gate",
                work_item,
            ),
            environment={
                # The qualified exact control is fail-closed: its constructor
                # rejects any environment other than the frozen K32/R600/LMR
                # configuration.  Keep these values on every independently
                # scheduled pair rather than relying on a worker-global env.
                "MCE_LMR": "1",
                "MCE_DIVERSE_PREFILTER": "1",
                "RAYON_NUM_THREADS": "2",
                "OMP_NUM_THREADS": "2",
                "OPENBLAS_NUM_THREADS": "2",
            },
            inputs=(reference,),
            application_metadata={
                "stage": stage,
                "work_item": work_item,
                "pair_index": work_item.removeprefix("pair-"),
                "image_digest": image,
                "gate_archive_sha256": str(input_receipt["archive_sha256"]),
                "contract_sha256": str(input_receipt["contract_sha256"]),
                "opponent_field_sha256": str(input_receipt["field_sha256"]),
                "exact_weights_sha256": str(input_receipt["exact_weights_sha256"]),
                "freeze_receipt_sha256": str(input_receipt["freeze_receipt_sha256"]),
            },
        )
        for work_item in work_items
    ]
    return client.submit_map(
        image=image,
        jobs=jobs,
        # The cross-architecture pair keeps the frozen R2 bundle and exact
        # NNUE K32/R600 rollout state resident together. Live smoke evidence
        # showed that a 2 GiB cgroup limit intermittently kills valid pairs;
        # reserve the preregistered 4 GiB RSS ceiling honestly instead of
        # relying on host overcommit.
        resources=Resources(cpu=2, memory_gib=4, disk_gib=4),
        outputs=("/outputs",),
        timeout_seconds=12 * 60 * 60,
        entrypoint=("/usr/local/bin/cascadia-cluster-job",),
        experiment_id=f"r2-map-{stage}-gate-v1",
        request_id=request_id,
        # Bacalhau 1.9's internal over-subscription queue is finite. Keep one
        # durable logical request, but admit only the aggregate number of jobs
        # that connected scheduler capacity can safely pack. Bacalhau retains
        # exclusive control of placement, retry, and rescheduling.
        scheduler_backpressure=True,
    )


def _validate_live_gate_fabric(nodes: list[dict[str, Any]]) -> None:
    """Fail closed unless the scheduler membership is exactly john1-john3."""

    observed: dict[str, float] = {}
    for node in nodes:
        info = node.get("Info") if isinstance(node, dict) else None
        labels = info.get("Labels") if isinstance(info, dict) else None
        compute = info.get("ComputeNodeInfo") if isinstance(info, dict) else None
        name = labels.get("cascadia_internal_node") if isinstance(labels, dict) else None
        maximum = compute.get("MaxCapacity") if isinstance(compute, dict) else None
        engines = compute.get("ExecutionEngines") if isinstance(compute, dict) else None
        cpu = maximum.get("CPU") if isinstance(maximum, dict) else None
        if (
            name not in {"john1", "john2", "john3"}
            or node.get("Connection") != "CONNECTED"
            or not isinstance(engines, list)
            or "docker" not in engines
            or not isinstance(cpu, (int, float))
            or isinstance(cpu, bool)
        ):
            raise GateFabricError("live Bacalhau membership differs from the active fabric")
        observed[name] = float(cpu)
    if observed != {"john1": 9.0, "john2": 10.0, "john3": 10.0}:
        raise GateFabricError("live Bacalhau capacity differs from registered 9/10/10")


def _merge_work_items(
    *,
    gate_directory: Path,
    result_root: Path,
    request_id: str,
    image: str,
    input_receipt: dict[str, Any],
    destination: Path,
) -> None:
    stage = _stage_from_contract(gate_directory)
    _validate_gate_inputs(gate_directory, stage)
    work_items = expected_work_items(stage)
    destination.mkdir(parents=True, exist_ok=True)
    _copy_exact(gate_directory / "contract.json", destination / "contract.json")
    _copy_exact(gate_directory / "opponent-field.json", destination / "opponent-field.json")
    admission = gate_directory / "smoke-admission-receipt.json"
    if admission.is_file():
        _copy_exact(admission, destination / admission.name)
    for relative in (
        "work-item-summaries",
        "reports",
        "projections",
        "scheduler-provenance",
    ):
        (destination / relative).mkdir(parents=True, exist_ok=True)
    for work_item in work_items:
        (destination / "receipts" / work_item).mkdir(parents=True, exist_ok=True)
        source = result_root / request_id / work_item / "campaign"
        contract_matches = _same_file(source / "contract.json", destination / "contract.json")
        field_matches = _same_file(
            source / "opponent-field.json", destination / "opponent-field.json"
        )
        if not contract_matches or not field_matches:
            raise GateFabricError(f"{work_item} result changed immutable campaign inputs")
        if admission.is_file() and (
            not (source / admission.name).is_file()
            or not _same_file(source / admission.name, admission)
        ):
            raise GateFabricError(f"{work_item} result changed smoke admission")
        summary = source / "work-item-summaries" / f"{work_item}.json"
        receipts = source / "receipts" / work_item
        if not summary.is_file() or not receipts.is_dir():
            raise GateFabricError(f"{work_item} result omitted its summary or receipts")
        _copy_exact(summary, destination / "work-item-summaries" / summary.name)
        for receipt in sorted(receipts.glob("pair-*.json")):
            _copy_exact(receipt, destination / "receipts" / work_item / receipt.name)
        scheduler_receipt = result_root / request_id / ".receipts" / f"{work_item}.json"
        _validate_scheduler_receipt(
            scheduler_receipt,
            request_id=request_id,
            item_id=work_item,
            image=image,
            input_receipt=input_receipt,
        )
        _copy_exact(
            scheduler_receipt,
            destination / "scheduler-provenance" / f"{work_item}.json",
        )


def _validate_scheduler_receipt(
    path: Path,
    *,
    request_id: str,
    item_id: str,
    image: str,
    input_receipt: dict[str, Any],
) -> dict[str, Any]:
    value = _read_json(path)
    payload = dict(value)
    claimed = payload.pop("receipt_sha256", None)
    metadata = value.get("application_metadata")
    expected_metadata = {
        "stage": str(input_receipt["stage"]),
        "work_item": item_id,
        "pair_index": item_id.removeprefix("pair-"),
        "image_digest": image,
        "gate_archive_sha256": str(input_receipt["archive_sha256"]),
        "contract_sha256": str(input_receipt["contract_sha256"]),
        "opponent_field_sha256": str(input_receipt["field_sha256"]),
        "exact_weights_sha256": str(input_receipt["exact_weights_sha256"]),
        "freeze_receipt_sha256": str(input_receipt["freeze_receipt_sha256"]),
    }
    if (
        value.get("schema_id") != "cascadia.cluster.accepted-result.v1"
        or value.get("request_id") != request_id
        or value.get("item_id") != item_id
        or value.get("image_digest") != image
        or not isinstance(value.get("bacalhau_job_id"), str)
        or not isinstance(value.get("accepted_execution_id"), str)
        or not isinstance(value.get("spec_sha256"), str)
        or len(value["spec_sha256"]) != 64
        or not isinstance(value.get("output_manifest_sha256"), str)
        or len(value["output_manifest_sha256"]) != 64
        or metadata != expected_metadata
        or claimed != _canonical_sha256(payload)
    ):
        raise GateFabricError(f"scheduler provenance differs for {item_id}")
    return value


def _stage_from_contract(gate_directory: Path) -> str:
    stage = _read_json(gate_directory / "contract.json").get("stage")
    for name, (registered_stage, _count) in STAGES.items():
        if stage == registered_stage:
            return name
    raise GateFabricError(f"unsupported focal stage: {stage}")


def _campaign_archive(campaign: Path, destination: Path) -> None:
    workspace = Path(tempfile.mkdtemp(prefix="r2-gate-aggregate-input-"))
    try:
        projected = workspace / "campaign"
        for relative in ("work-item-summaries", "receipts", "scheduler-provenance"):
            source = campaign / relative
            if not source.is_dir():
                raise GateFabricError(f"aggregate input omits {relative}")
            shutil.copytree(source, projected / relative)
        root_files = ["contract.json", "opponent-field.json"]
        if (campaign / "smoke-admission-receipt.json").is_file():
            root_files.append("smoke-admission-receipt.json")
        for relative in root_files:
            _copy_exact(campaign / relative, projected / relative)
        # Aggregate outputs are intentionally absent from the immutable input.
        # This keeps the archive stable when a caller reconnects after a prior
        # aggregate already populated reports and projections.
        (projected / "reports").mkdir()
        (projected / "projections").mkdir()
        (projected / "scheduler-provenance" / "aggregate.json").unlink(missing_ok=True)
        with tarfile.open(destination, "w", format=tarfile.PAX_FORMAT) as archive:
            _add_tree(archive, projected, "campaign")
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def _aggregate_request_id(request_id: str, campaign_archive_sha256: str) -> str:
    if not request_id or re.fullmatch(r"[0-9a-f]{64}", campaign_archive_sha256) is None:
        raise GateFabricError("aggregate request identity is malformed")
    return f"{request_id}-aggregate-{campaign_archive_sha256[:12]}"


def _scheduler_wall_seconds(results: Any) -> float:
    """Return one reconnect-stable wall interval for the scheduler campaign."""

    starts = [item.created_unix_ns for item in results]
    finishes = [item.modified_unix_ns for item in results]
    if (
        not starts
        or any(not isinstance(value, int) or value <= 0 for value in starts)
        or any(not isinstance(value, int) or value <= 0 for value in finishes)
        or any(finish < start for start, finish in zip(starts, finishes, strict=True))
    ):
        raise GateFabricError("scheduler timestamps are incomplete or nonmonotonic")
    return max(0.001, (max(finishes) - min(starts)) / 1_000_000_000.0)


def _scheduler_observation(
    *,
    nodes: list[dict[str, Any]],
    statuses: Any,
    observed_unix_ms: int,
) -> dict[str, Any]:
    projected_nodes = []
    for raw in nodes:
        info = raw.get("Info") if isinstance(raw, dict) else None
        labels = info.get("Labels") if isinstance(info, dict) else None
        compute = info.get("ComputeNodeInfo") if isinstance(info, dict) else None
        maximum = compute.get("MaxCapacity") if isinstance(compute, dict) else None
        available = compute.get("AvailableCapacity") if isinstance(compute, dict) else None
        name = labels.get("cascadia_internal_node") if isinstance(labels, dict) else None
        capacity = maximum.get("CPU") if isinstance(maximum, dict) else None
        # Bacalhau v1.9 omits zero-valued resource fields from its JSON
        # projection. A fully allocated node therefore reports an
        # ``AvailableCapacity`` object without ``CPU`` rather than ``CPU: 0``.
        # Preserve strict shape validation for the enclosing object while
        # normalizing that documented protobuf/JSON zero-value elision here.
        free = available.get("CPU", 0) if isinstance(available, dict) else None
        if (
            name not in {"john1", "john2", "john3"}
            or not isinstance(capacity, (int, float))
            or isinstance(capacity, bool)
            or not isinstance(free, (int, float))
            or isinstance(free, bool)
            or capacity < 0
            or free < 0
            or free > capacity
        ):
            raise GateFabricError("Bacalhau scheduler node capacity observation is malformed")
        projected_nodes.append(
            {
                "name": name,
                "cpu_capacity": float(capacity),
                "cpu_allocated": float(capacity - free),
                "running_executions": int(compute.get("RunningExecutions", 0)),
                "connected": raw.get("Connection") == "CONNECTED",
            }
        )
    projected_nodes.sort(key=lambda node: node["name"])
    if {node["name"] for node in projected_nodes} != {"john1", "john2", "john3"}:
        raise GateFabricError("Bacalhau scheduler observation lacks the three-node fabric")
    state_counts: dict[str, int] = {}
    for status in statuses:
        state_counts[status.value] = state_counts.get(status.value, 0) + 1
    capacity = sum(node["cpu_capacity"] for node in projected_nodes)
    allocated = sum(node["cpu_allocated"] for node in projected_nodes)
    return {
        "observed_unix_ms": observed_unix_ms,
        "cpu_capacity": capacity,
        "cpu_allocated": allocated,
        "cpu_utilization": allocated / capacity if capacity > 0 else 0.0,
        "nodes": projected_nodes,
        "work_item_states": dict(sorted(state_counts.items())),
    }


def _scheduler_utilization(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise GateFabricError("scheduler utilization has no observations")
    timestamps = [sample.get("observed_unix_ms") for sample in samples]
    malformed = any(not isinstance(value, int) or value <= 0 for value in timestamps)
    if malformed or timestamps != sorted(timestamps):
        raise GateFabricError("scheduler utilization timestamps are malformed")
    if len(samples) == 1:
        weights = [1.0]
    else:
        weights = [
            max(0.001, (timestamps[index + 1] - timestamps[index]) / 1000.0)
            for index in range(len(samples) - 1)
        ]
        weights.append(weights[-1])
    total_weight = sum(weights)

    def weighted(values: list[float]) -> float:
        numerator = sum(value * weight for value, weight in zip(values, weights, strict=True))
        return numerator / total_weight

    nodes = {}
    for name in ("john1", "john2", "john3"):
        values = []
        capacities = []
        for sample in samples:
            match = next(node for node in sample["nodes"] if node["name"] == name)
            values.append(float(match["cpu_allocated"]))
            capacities.append(float(match["cpu_capacity"]))
        nodes[name] = {
            "cpu_capacity_min": min(capacities),
            "cpu_capacity_max": max(capacities),
            "cpu_allocated_mean": weighted(values),
            "cpu_allocated_peak": max(values),
        }
    allocations = [float(sample["cpu_allocated"]) for sample in samples]
    capacities = [float(sample["cpu_capacity"]) for sample in samples]
    utilizations = [float(sample["cpu_utilization"]) for sample in samples]
    return {
        "sample_count": len(samples),
        "observed_seconds": max(0.0, (timestamps[-1] - timestamps[0]) / 1000.0),
        "cpu_capacity_min": min(capacities),
        "cpu_capacity_max": max(capacities),
        "cpu_allocated_mean": weighted(allocations),
        "cpu_allocated_peak": max(allocations),
        "cpu_utilization_mean": weighted(utilizations),
        "cpu_utilization_peak": max(utilizations),
        "nodes": nodes,
    }


def _monitor_scheduler_request(
    *,
    handle: Any,
    client: ClusterClient,
    state_directory: Path,
    request_id: str,
    poll_seconds: float = 15.0,
) -> tuple[list[dict[str, Any]], Path]:
    path = state_directory / "observations" / f"{request_id}.json"
    if path.exists():
        state = _read_json(path)
        payload = dict(state)
        claimed = payload.pop("state_sha256", None)
        if (
            state.get("schema_id") != "cascadia.cluster.scheduler-observations.v1"
            or state.get("request_id") != request_id
            or not isinstance(state.get("samples"), list)
            or claimed != _canonical_sha256(payload)
        ):
            raise GateFabricError("persisted scheduler observations differ")
        if state.get("terminal") is True:
            return state["samples"], path
    else:
        state = {
            "schema_id": "cascadia.cluster.scheduler-observations.v1",
            "request_id": request_id,
            "terminal": False,
            "samples": [],
        }
    while True:
        statuses = handle.status()
        state["samples"].append(
            _scheduler_observation(
                nodes=client.api.nodes(),
                statuses=statuses,
                observed_unix_ms=time.time_ns() // 1_000_000,
            )
        )
        state["terminal"] = all(status.terminal for status in statuses)
        payload = dict(state)
        payload.pop("state_sha256", None)
        state["state_sha256"] = _canonical_sha256(payload)
        _write_json_replace(path, state)
        if state["terminal"]:
            return state["samples"], path
        time.sleep(poll_seconds)


def _write_completion_artifacts(
    *,
    campaign_directory: Path,
    scheduler_report: dict[str, Any],
) -> None:
    focal_json_path = campaign_directory / "reports/focal-benchmark.json"
    focal_markdown_path = campaign_directory / "reports/focal-benchmark.md"
    focal = _read_json(focal_json_path)
    focal_markdown = focal_markdown_path.read_text()
    utilization = scheduler_report["scheduler_utilization"]
    nodes = utilization["nodes"]
    scheduler_markdown = (
        "\n## Scheduler and immutable provenance\n\n"
        f"- Request: `{scheduler_report['request_id']}`\n"
        f"- Aggregate request: `{scheduler_report['aggregate_request_id']}`\n"
        f"- Immutable image: `{scheduler_report['image_digest']}`\n"
        f"- Pair work items / retries: {len(scheduler_report['work_items'])} / "
        f"{scheduler_report['retry_count']}\n"
        f"- CPU capacity min/max: {utilization['cpu_capacity_min']:.1f} / "
        f"{utilization['cpu_capacity_max']:.1f}\n"
        f"- CPU allocated mean/peak: {utilization['cpu_allocated_mean']:.3f} / "
        f"{utilization['cpu_allocated_peak']:.1f}\n"
        f"- Scheduler utilization mean/peak: "
        f"{100.0 * utilization['cpu_utilization_mean']:.2f}% / "
        f"{100.0 * utilization['cpu_utilization_peak']:.2f}%\n"
        f"- Observations / observed seconds: {utilization['sample_count']} / "
        f"{utilization['observed_seconds']:.3f}\n\n"
        "| Node | Capacity min/max | Allocated mean/peak |\n"
        "|---|---:|---:|\n"
        + "".join(
            f"| {name} | {nodes[name]['cpu_capacity_min']:.1f} / "
            f"{nodes[name]['cpu_capacity_max']:.1f} | "
            f"{nodes[name]['cpu_allocated_mean']:.3f} / "
            f"{nodes[name]['cpu_allocated_peak']:.1f} |\n"
            for name in ("john1", "john2", "john3")
        )
        + "\n"
        f"- Gate input archive SHA-256: "
        f"`{scheduler_report['gate_input']['archive_sha256']}`\n"
        f"- Scheduler observation SHA-256: "
        f"`{scheduler_report['scheduler_observations_sha256']}`\n"
        f"- Scheduler provenance report SHA-256: "
        f"`{scheduler_report['report_sha256']}`\n"
    )
    combined_path = campaign_directory / "reports/focal-benchmark-complete.md"
    combined = focal_markdown.rstrip() + "\n" + scheduler_markdown
    _write_bytes_once(combined_path, combined.encode())
    result = focal.get("result")
    statistics = result.get("statistics") if isinstance(result, dict) else None
    completion: dict[str, Any] = {
        "schema_id": "cascadia.r2-map.campaign-completion.v1",
        "stage": scheduler_report["stage"],
        "pairs": statistics.get("pairs") if isinstance(statistics, dict) else None,
        "physical_games": (
            statistics.get("physical_games") if isinstance(statistics, dict) else None
        ),
        "classification": (
            statistics.get("classification") if isinstance(statistics, dict) else None
        ),
        "image_digest": scheduler_report["image_digest"],
        "request_id": scheduler_report["request_id"],
        "aggregate_request_id": scheduler_report["aggregate_request_id"],
        "retry_count": scheduler_report["retry_count"],
        "scheduler_utilization": utilization,
        "focal_report_sha256": _sha256_file(focal_json_path),
        "focal_markdown_sha256": _sha256_file(focal_markdown_path),
        "scheduler_provenance_sha256": _sha256_file(
            campaign_directory / "reports/scheduler-provenance.json"
        ),
        "combined_markdown_sha256": _sha256_file(combined_path),
    }
    completion["completion_sha256"] = _canonical_sha256(completion)
    _write_json_once(campaign_directory / "reports/campaign-completion.json", completion)


def collect_and_aggregate(
    *,
    image: str,
    input_receipt: dict[str, Any],
    stage: str,
    gate_directory: Path,
    request_id: str,
    state_directory: Path,
    artifact_directory: Path,
    campaign_directory: Path,
) -> str:
    client = _client(state_directory, artifact_directory)
    handle = client.reconnect(request_id)
    scheduler_samples, scheduler_observation_path = _monitor_scheduler_request(
        handle=handle,
        client=client,
        state_directory=state_directory,
        request_id=request_id,
    )
    result = handle.results()
    if result.failure_count or any(
        item.status is not JobStatus.SUCCEEDED for item in result.results
    ):
        details = ", ".join(
            f"{item.item_key}={item.status}:{item.failure_reason or '-'}" for item in result.results
        )
        raise GateFabricError(f"R2 gate work-item request failed: {details}")
    expected = set(expected_work_items(stage))
    if {item.item_key for item in result.results} != expected:
        raise GateFabricError("R2 gate result does not cover every registered pair work item")
    if any(item.image_digest != image for item in result.results):
        raise GateFabricError("R2 gate result image digest differs")
    wall_seconds = _scheduler_wall_seconds(result.results)
    _merge_work_items(
        gate_directory=gate_directory,
        result_root=artifact_directory,
        request_id=request_id,
        image=image,
        input_receipt=input_receipt,
        destination=campaign_directory,
    )
    _copy_exact(
        scheduler_observation_path,
        campaign_directory / "scheduler-provenance/request-observations.json",
    )
    with tempfile.TemporaryDirectory(prefix="r2-gate-aggregate-") as temporary:
        archive = Path(temporary) / "r2-campaign.tar"
        _campaign_archive(campaign_directory, archive)
        campaign_archive_sha256 = _sha256_file(archive)
        # Bind reducer identity to the exact assembled campaign. Besides making
        # retries explicit, this prevents any failed/obsolete reducer request
        # from being reused for different campaign bytes.
        aggregate_request_id = _aggregate_request_id(request_id, campaign_archive_sha256)
        reference = _store().stage_file(archive, target="/inputs/r2-campaign")
        aggregate = client.submit_map(
            image=image,
            jobs=(
                ContainerInput(
                    "aggregate",
                    args=(
                        "/bin/sh",
                        "-eu",
                        "-c",
                        AGGREGATE_SCRIPT,
                        "r2-map-aggregate",
                        f"{wall_seconds:.6f}",
                    ),
                    inputs=(reference,),
                    application_metadata={
                        "stage": stage,
                        "operation": "aggregate",
                        "image_digest": image,
                        "campaign_archive_sha256": campaign_archive_sha256,
                        "gate_archive_sha256": str(input_receipt["archive_sha256"]),
                    },
                ),
            ),
            resources=Resources(cpu=1, memory_gib=2, disk_gib=4),
            outputs=("/outputs",),
            timeout_seconds=30 * 60,
            entrypoint=("/usr/local/bin/cascadia-cluster-job",),
            experiment_id=f"r2-map-{stage}-gate-aggregate-v1",
            request_id=aggregate_request_id,
        )
        aggregate_result = aggregate.wait(timeout_seconds=30 * 60).results()
    if aggregate_result.failure_count:
        raise GateFabricError("R2 gate aggregation job failed")
    final_campaign = artifact_directory / aggregate_request_id / "aggregate" / "campaign"
    if not (final_campaign / "reports/focal-benchmark.json").is_file():
        raise GateFabricError("R2 gate aggregate omitted its final report")
    for path in sorted(final_campaign.rglob("*")):
        if path.is_file():
            _copy_exact(path, campaign_directory / path.relative_to(final_campaign))
    aggregate_receipt_path = (
        artifact_directory / aggregate_request_id / ".receipts" / "aggregate.json"
    )
    aggregate_receipt = _read_json(aggregate_receipt_path)
    aggregate_payload = dict(aggregate_receipt)
    aggregate_claimed = aggregate_payload.pop("receipt_sha256", None)
    aggregate_metadata = aggregate_receipt.get("application_metadata")
    if (
        aggregate_receipt.get("schema_id") != "cascadia.cluster.accepted-result.v1"
        or aggregate_receipt.get("request_id") != aggregate_request_id
        or aggregate_receipt.get("item_id") != "aggregate"
        or aggregate_receipt.get("image_digest") != image
        or aggregate_metadata
        != {
            "stage": stage,
            "operation": "aggregate",
            "image_digest": image,
            "campaign_archive_sha256": campaign_archive_sha256,
            "gate_archive_sha256": str(input_receipt["archive_sha256"]),
        }
        or aggregate_claimed != _canonical_sha256(aggregate_payload)
    ):
        raise GateFabricError("aggregate scheduler provenance differs")
    _copy_exact(
        aggregate_receipt_path,
        campaign_directory / "scheduler-provenance/aggregate.json",
    )
    pair_receipts = [
        _read_json(campaign_directory / "scheduler-provenance" / f"{item}.json")
        for item in expected_work_items(stage)
    ]
    scheduler_report = {
        "schema_id": "cascadia.r2-map.scheduler-provenance.v1",
        "stage": stage,
        "request_id": request_id,
        "aggregate_request_id": aggregate_request_id,
        "image_digest": image,
        "gate_input": input_receipt,
        "scheduler_observations_sha256": _sha256_file(scheduler_observation_path),
        "scheduler_utilization": _scheduler_utilization(scheduler_samples),
        "work_items": [
            {
                key: receipt[key]
                for key in (
                    "item_id",
                    "bacalhau_job_id",
                    "accepted_execution_id",
                    "spec_sha256",
                    "output_manifest_sha256",
                    "attempts",
                    "created_unix_ns",
                    "modified_unix_ns",
                    "receipt_sha256",
                )
            }
            for receipt in pair_receipts
        ],
        "aggregate": {
            key: aggregate_receipt[key]
            for key in (
                "bacalhau_job_id",
                "accepted_execution_id",
                "spec_sha256",
                "output_manifest_sha256",
                "attempts",
                "created_unix_ns",
                "modified_unix_ns",
                "receipt_sha256",
            )
        },
        "retry_count": sum(max(0, int(receipt["attempts"]) - 1) for receipt in pair_receipts),
    }
    scheduler_report["report_sha256"] = _canonical_sha256(scheduler_report)
    _write_json_once(
        campaign_directory / "reports/scheduler-provenance.json",
        scheduler_report,
    )
    _write_completion_artifacts(
        campaign_directory=campaign_directory,
        scheduler_report=scheduler_report,
    )
    return aggregate_request_id


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--image", required=True, help="immutable OCI digest")
    parser.add_argument("--gate-directory", type=Path, required=True)
    parser.add_argument("--candidate-freeze", type=Path, required=True)
    parser.add_argument("--exact-weights", type=Path, required=True)
    parser.add_argument("--state-directory", type=Path, required=True)
    parser.add_argument("--artifact-directory", type=Path, required=True)
    parser.add_argument("--campaign-directory", type=Path, required=True)
    parser.add_argument("--request-id")
    parser.add_argument("--submit-only", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    contract, _ = _validate_gate_inputs(args.gate_directory, args.stage)
    request_id = args.request_id or (
        f"r2-{args.stage}-{str(contract['candidate_checkpoint_id'])[-20:]}"
    )
    with tempfile.TemporaryDirectory(prefix="r2-gate-submit-") as temporary:
        archive = Path(temporary) / "r2-gate.tar"
        receipt = build_gate_archive(
            gate_directory=args.gate_directory,
            candidate_freeze=args.candidate_freeze,
            exact_weights=args.exact_weights,
            stage=args.stage,
            destination=archive,
        )
        submit_gate(
            image=args.image,
            archive=archive,
            input_receipt=receipt,
            stage=args.stage,
            request_id=request_id,
            state_directory=args.state_directory,
            artifact_directory=args.artifact_directory,
        )
    if args.submit_only:
        print(json.dumps({"request_id": request_id, "input": receipt}, sort_keys=True))
        return 0
    aggregate_request_id = collect_and_aggregate(
        image=args.image,
        input_receipt=receipt,
        stage=args.stage,
        gate_directory=args.gate_directory,
        request_id=request_id,
        state_directory=args.state_directory,
        artifact_directory=args.artifact_directory,
        campaign_directory=args.campaign_directory,
    )
    print(
        json.dumps(
            {
                "request_id": request_id,
                "aggregate_request_id": aggregate_request_id,
                "campaign_directory": str(args.campaign_directory),
                "report": str(args.campaign_directory / "reports/focal-benchmark.json"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (GateFabricError, OSError, ValueError) as error:
        raise SystemExit(f"R2-MAP Bacalhau gate refused: {error}") from error
