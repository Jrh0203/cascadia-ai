#!/usr/bin/env python3
"""Durable cluster execution for V3 replay verification and teacher labels."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import uuid
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import blake3
from cascadia_cluster import (
    BacalhauAPIError,
    ClusterClient,
    ContainerInput,
    JobStatus,
    ObjectStoreClient,
    ObjectStoreConfig,
    Resources,
)

IMAGE = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")
READINESS = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_ENDPOINT = "http://100.110.109.6:1234"
DEFAULT_STORE_ENDPOINT = "http://100.110.109.6:9000"


class PipelineError(ValueError):
    """A Phase 2 pipeline artifact or transition is invalid."""


def _write_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _store() -> ObjectStoreClient:
    missing = [
        name
        for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
        if not os.environ.get(name)
    ]
    if missing:
        raise PipelineError(f"missing object-store environment: {missing}")
    store = ObjectStoreClient(
        ObjectStoreConfig(
            endpoint=os.environ.get("AWS_ENDPOINT_URL_S3", DEFAULT_STORE_ENDPOINT),
            access_key=os.environ["AWS_ACCESS_KEY_ID"],
            secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        )
    )
    store.ensure_bucket(store.config.input_bucket)
    store.ensure_bucket(store.config.result_bucket)
    return store


def _client(state_directory: Path, artifact_directory: Path) -> ClusterClient:
    return ClusterClient(
        os.environ.get("BACALHAU_ENDPOINT", DEFAULT_ENDPOINT),
        state_directory=state_directory,
        object_store=_store(),
        artifact_directory=artifact_directory,
    )


def _validate_fabric(nodes: list[dict[str, Any]]) -> None:
    observed: dict[str, float] = {}
    for node in nodes:
        info = node.get("Info") if isinstance(node, dict) else None
        labels = info.get("Labels") if isinstance(info, dict) else None
        compute = info.get("ComputeNodeInfo") if isinstance(info, dict) else None
        name = labels.get("cascadia_internal_node") if isinstance(labels, dict) else None
        capacity = compute.get("MaxCapacity") if isinstance(compute, dict) else None
        engines = compute.get("ExecutionEngines") if isinstance(compute, dict) else None
        cpu = capacity.get("CPU") if isinstance(capacity, dict) else None
        if (
            name not in {"john1", "john2", "john3"}
            or node.get("Connection") != "CONNECTED"
            or not isinstance(engines, list)
            or "docker" not in engines
            or not isinstance(cpu, (int, float))
            or isinstance(cpu, bool)
        ):
            raise PipelineError("live Bacalhau membership differs from the authorized fabric")
        observed[name] = float(cpu)
    if observed != {"john1": 9.0, "john2": 10.0, "john3": 10.0}:
        raise PipelineError(f"live Bacalhau capacity differs from 9/10/10: {observed}")


def _validate_image(image: str) -> None:
    if not IMAGE.fullmatch(image):
        raise PipelineError("worker image must be an immutable registry digest")


def _read_authorized_state(path: Path, expected_phase: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    approved = value.get("approved_readiness_sha256")
    if (
        value.get("schema_id") != "cascadia-v3-campaign-state-v1"
        or value.get("part") != 2
        or value.get("phase2_authorized") is not True
        or value.get("phase") != expected_phase
        or value.get("protected_seed_values_opened") is not False
        or not isinstance(approved, str)
        or not READINESS.fullmatch(approved)
        or value.get("readiness_sha256") != approved
    ):
        raise PipelineError(f"campaign state is not authorized for {expected_phase}")
    return value


def build_verify_jobs(
    shards: list[Path], store: ObjectStoreClient
) -> tuple[list[ContainerInput], dict[str, Path]]:
    if not shards or len({path.name for path in shards}) != len(shards):
        raise PipelineError("verification requires uniquely named replay shards")
    jobs = []
    source_by_key = {}
    for index, path in enumerate(sorted(shards)):
        if path.suffix != ".v3g" or not path.is_file():
            raise PipelineError(f"invalid replay shard: {path}")
        key = f"verify-{index:05d}"
        reference = store.stage_file(path, target="/inputs/shard")
        jobs.append(
            ContainerInput(
                key=key,
                args=(
                    "v3-campaign-worker",
                    "verify-shard",
                    "--input",
                    reference.mounted_path,
                    "--output",
                    "/outputs/verification.json",
                ),
                environment={"RAYON_NUM_THREADS": "1"},
                inputs=(reference,),
                application_metadata={
                    "campaign": "cascadia-v3",
                    "stage": "bootstrap-replay-verification",
                    "source_shard": path.name,
                    "source_sha256": reference.sha256,
                    "source_bytes": str(path.stat().st_size),
                },
            )
        )
        source_by_key[key] = path
    return jobs, source_by_key


def build_label_jobs(
    roots: list[Path],
    store: ObjectStoreClient,
    *,
    campaign_state: Path,
    v1_weights: Path,
    approved_readiness_sha256: str,
    cycle: int | None,
) -> tuple[list[ContainerInput], dict[str, Path]]:
    if not roots or len({path.name for path in roots}) != len(roots):
        raise PipelineError("labeling requires uniquely named root shards")
    state_reference = store.stage_file(campaign_state, target="/inputs/control")
    v1_reference = store.stage_file(v1_weights, target="/inputs/v1")
    jobs = []
    source_by_key = {}
    for index, path in enumerate(sorted(roots)):
        if path.suffix != ".v3r" or not path.is_file():
            raise PipelineError(f"invalid teacher-root shard: {path}")
        key = f"label-{index:05d}"
        root_reference = store.stage_file(path, target="/inputs/root")
        output_name = f"{path.stem}.v3l"
        arguments = [
            "v3-campaign-worker",
            "label-roots",
            "--input",
            root_reference.mounted_path,
            "--output",
            f"/outputs/{output_name}",
            "--v1-weights",
            v1_reference.mounted_path,
            "--rollouts",
            "600",
            "--campaign-state",
            state_reference.mounted_path,
            "--approved-readiness-sha256",
            approved_readiness_sha256,
        ]
        if cycle is not None:
            arguments.extend(("--cycle", str(cycle)))
        jobs.append(
            ContainerInput(
                key=key,
                args=tuple(arguments),
                environment={"RAYON_NUM_THREADS": "1"},
                inputs=(root_reference, state_reference, v1_reference),
                application_metadata={
                    "campaign": "cascadia-v3",
                    "stage": "teacher-labeling",
                    "cycle": str(cycle or 0),
                    "source_roots": path.name,
                    "source_sha256": root_reference.sha256,
                    "source_bytes": str(path.stat().st_size),
                    "campaign_state_sha256": state_reference.sha256,
                    "v1_weights_sha256": v1_reference.sha256,
                    "rollouts_per_root": "600",
                },
            )
        )
        source_by_key[key] = path
    return jobs, source_by_key


def build_validation_cache_jobs(
    labels: list[Path], store: ObjectStoreClient
) -> tuple[list[ContainerInput], dict[str, Path]]:
    if not labels or len({path.name for path in labels}) != len(labels):
        raise PipelineError("validation caching requires uniquely named label shards")
    jobs = []
    source_by_key = {}
    for index, path in enumerate(sorted(labels)):
        if path.suffix != ".v3l" or not path.is_file():
            raise PipelineError(f"invalid validation label shard: {path}")
        source_receipt = path.with_suffix(".receipt.json")
        if not source_receipt.is_file():
            raise PipelineError(f"validation label receipt is missing: {source_receipt}")
        source_value = json.loads(source_receipt.read_text())
        if (
            source_value.get("schema_id") != "cascadia-v3-teacher-label-shard-receipt-v1"
            or source_value.get("passed") is not True
            or source_value.get("roots") != 1_000
            or not isinstance(source_value.get("candidate_estimates"), int)
            or source_value["candidate_estimates"] < source_value["roots"]
        ):
            raise PipelineError(f"validation label receipt is invalid: {source_receipt}")
        key = f"validation-cache-{index:05d}"
        reference = store.stage_file(path, target="/inputs/labels")
        output_name = f"{path.stem}.v3t"
        receipt_name = f"{path.stem}.receipt.json"
        jobs.append(
            ContainerInput(
                key=key,
                args=(
                    "teacher_labels_to_training",
                    "--input",
                    reference.mounted_path,
                    "--output",
                    f"/outputs/{output_name}",
                    "--receipt",
                    f"/outputs/{receipt_name}",
                ),
                environment={"RAYON_NUM_THREADS": "1"},
                inputs=(reference,),
                application_metadata={
                    "campaign": "cascadia-v3",
                    "stage": "bootstrap-validation-cache",
                    "source_labels": path.name,
                    "source_sha256": reference.sha256,
                    "source_bytes": str(path.stat().st_size),
                    "expected_roots": str(source_value["roots"]),
                    "expected_rows": str(source_value["candidate_estimates"]),
                },
            )
        )
        source_by_key[key] = path
    return jobs, source_by_key


def _monitor(
    *,
    client: ClusterClient,
    image: str,
    jobs: list[ContainerInput],
    resources: Resources,
    request_id: str,
    experiment_id: str,
    artifact_directory: Path,
    progress: Path,
    timeout_seconds: int,
    validate: Callable[[Path, ContainerInput], dict[str, int]],
) -> dict[str, Any]:
    handle = client.submit_map(
        image=image,
        jobs=jobs,
        resources=resources,
        outputs=("/outputs",),
        timeout_seconds=timeout_seconds,
        entrypoint=("/usr/local/bin/cascadia-cluster-job",),
        experiment_id=experiment_id,
        request_id=request_id,
        scheduler_backpressure=True,
    )
    started = time.monotonic()
    scheduler_transient_errors = 0
    statuses: tuple[JobStatus, ...] = ()
    snapshot: dict[str, Any] = {}

    def write_snapshot(*, scheduler_status: str, error: Exception | None = None) -> None:
        counts = Counter(status.value for status in statuses)
        value: dict[str, Any] = {
            "schema_id": "cascadia-v3-cluster-stage-progress-v1",
            "request_id": request_id,
            "image_digest": image,
            "work_items": len(jobs),
            "status_counts": dict(sorted(counts.items())),
            "terminal_items": sum(status.terminal for status in statuses),
            "fraction_complete": (
                sum(status.terminal for status in statuses) / len(jobs)
                if statuses
                else 0.0
            ),
            "elapsed_seconds": time.monotonic() - started,
            "updated_unix_ms": time.time_ns() // 1_000_000,
            "scheduler_status": scheduler_status,
            "scheduler_transient_errors": scheduler_transient_errors,
        }
        if error is not None:
            value["last_scheduler_error"] = str(error)
        snapshot.clear()
        snapshot.update(value)
        _write_atomic(progress, value)
        print(json.dumps(value, sort_keys=True), flush=True)

    while True:
        try:
            statuses = handle.status()
        except BacalhauAPIError as error:
            scheduler_transient_errors += 1
            if time.monotonic() - started >= timeout_seconds:
                raise TimeoutError(
                    f"cluster stage {request_id} exceeded its timeout during scheduler recovery"
                ) from error
            write_snapshot(scheduler_status="retrying", error=error)
            time.sleep(15)
            continue
        write_snapshot(scheduler_status="healthy")
        if all(status.terminal for status in statuses):
            break
        if time.monotonic() - started >= timeout_seconds:
            raise TimeoutError(f"cluster stage {request_id} exceeded its timeout")
        time.sleep(15)
    while True:
        try:
            result = handle.results()
        except BacalhauAPIError as error:
            scheduler_transient_errors += 1
            if time.monotonic() - started >= timeout_seconds:
                raise TimeoutError(
                    f"cluster stage {request_id} exceeded its timeout during result recovery"
                ) from error
            write_snapshot(scheduler_status="results-retrying", error=error)
            time.sleep(15)
            continue
        write_snapshot(scheduler_status="healthy")
        break
    if result.failure_count:
        failures = [
            {
                "item": item.item_key,
                "status": item.status.value,
                "exit_code": item.exit_code,
                "reason": item.failure_reason,
            }
            for item in result.results
            if item.status is not JobStatus.SUCCEEDED
        ]
        raise PipelineError(f"cluster stage failed: {failures}")
    totals: Counter[str] = Counter()
    for job in jobs:
        item_directory = artifact_directory / request_id / job.key
        values = validate(item_directory, job)
        totals.update(values)
    completion = {
        "schema_id": "cascadia-v3-cluster-stage-completion-v1",
        "passed": True,
        "request_id": request_id,
        "experiment_id": experiment_id,
        "image_digest": image,
        "work_items": len(jobs),
        "succeeded": len(result.results),
        "elapsed_seconds": result.elapsed_seconds,
        "totals": dict(sorted(totals.items())),
        "artifact_root": str((artifact_directory / request_id).resolve()),
        "inputs": [
            {"item": job.key, **dict(job.application_metadata)}
            for job in jobs
        ],
    }
    return completion


def _validate_verification(
    item_directory: Path,
    job: ContainerInput,
    entries_per_game: int = 80,
) -> dict[str, int]:
    path = item_directory / "verification.json"
    value = json.loads(path.read_text())
    if (
        value.get("schema_id") != "cascadia-v3-compact-shard-verification-v1"
        or value.get("passed") is not True
        or not isinstance(value.get("records"), int)
        or not isinstance(value.get("expanded_training_entries"), int)
        or value["expanded_training_entries"] != value["records"] * entries_per_game
    ):
        raise PipelineError(f"invalid verification artifact for {job.key}")
    return {
        "records": value["records"],
        "expanded_training_entries": value["expanded_training_entries"],
        "bytes": int(value["bytes"]),
    }


def _validate_label(item_directory: Path, job: ContainerInput) -> dict[str, int]:
    receipts = sorted(item_directory.glob("*.receipt.json"))
    labels = sorted(item_directory.glob("*.v3l"))
    if len(receipts) != 1 or len(labels) != 1:
        raise PipelineError(f"label artifact set is incomplete for {job.key}")
    value = json.loads(receipts[0].read_text())
    if (
        value.get("schema_id") != "cascadia-v3-teacher-label-shard-receipt-v1"
        or value.get("passed") is not True
        or value.get("scientific_eligible") is not True
        or value.get("rollouts_per_root") != 600
        or not isinstance(value.get("roots"), int)
        or not isinstance(value.get("candidate_estimates"), int)
        or value["candidate_estimates"] < value["roots"]
        or value.get("output_bytes") != labels[0].stat().st_size
    ):
        raise PipelineError(f"invalid label artifact for {job.key}")
    return {
        "roots": value["roots"],
        "candidate_estimates": value["candidate_estimates"],
        "output_bytes": value["output_bytes"],
        "rollouts": value["roots"] * 600,
    }


def _blake3(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_validation_cache(
    item_directory: Path, job: ContainerInput
) -> dict[str, int]:
    receipts = sorted(item_directory.glob("*.receipt.json"))
    shards = sorted(item_directory.glob("*.v3t"))
    if len(receipts) != 1 or len(shards) != 1:
        raise PipelineError(f"validation-cache artifact set is incomplete for {job.key}")
    value = json.loads(receipts[0].read_text())
    expected_roots = int(job.application_metadata["expected_roots"])
    expected_rows = int(job.application_metadata["expected_rows"])
    if (
        value.get("schema_id") != "cascadia-v3-teacher-training-expansion-v1"
        or value.get("passed") is not True
        or value.get("scientific_eligible") is not True
        or value.get("roots") != expected_roots
        or value.get("rows") != expected_rows
        or not isinstance(value.get("realized_rows"), int)
        or not 0 <= value["realized_rows"] <= expected_roots
        or value.get("counterfactual_rows") != expected_rows - value["realized_rows"]
        or value.get("output_bytes") != shards[0].stat().st_size
        or value.get("output_blake3") != _blake3(shards[0])
    ):
        raise PipelineError(f"validation-cache artifact is invalid for {job.key}")
    return {
        "roots": value["roots"],
        "rows": value["rows"],
        "realized_rows": value["realized_rows"],
        "counterfactual_rows": value["counterfactual_rows"],
        "output_bytes": value["output_bytes"],
    }


def _validation_cache_manifest(
    *,
    jobs: list[ContainerInput],
    sources: dict[str, Path],
    artifact_directory: Path,
    request_id: str,
    image: str,
    completion: dict[str, Any],
) -> dict[str, Any]:
    shards = []
    for job in jobs:
        directory = artifact_directory / request_id / job.key
        receipt_path = next(iter(sorted(directory.glob("*.receipt.json"))))
        shard_path = next(iter(sorted(directory.glob("*.v3t"))))
        receipt = json.loads(receipt_path.read_text())
        source = sources[job.key]
        shards.append(
            {
                "item": job.key,
                "source_path": str(source.resolve()),
                "source_blake3": _blake3(source),
                "source_bytes": source.stat().st_size,
                "path": str(shard_path.resolve()),
                "blake3": _blake3(shard_path),
                "bytes": shard_path.stat().st_size,
                "roots": receipt["roots"],
                "rows": receipt["rows"],
                "receipt": str(receipt_path.resolve()),
                "receipt_blake3": _blake3(receipt_path),
            }
        )
    return {
        "schema_id": "cascadia-v3-validation-cache-v1",
        "passed": True,
        "request_id": request_id,
        "image_digest": image,
        "completion": completion,
        "shards": shards,
        "totals": {
            "shards": len(shards),
            "roots": sum(item["roots"] for item in shards),
            "rows": sum(item["rows"] for item in shards),
            "bytes": sum(item["bytes"] for item in shards),
        },
        "created_unix_ms": time.time_ns() // 1_000_000,
    }


def _reconcile_validation_cache(
    *,
    jobs: list[ContainerInput],
    artifact_directory: Path,
    request_id: str,
    image: str,
) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    for job in jobs:
        totals.update(
            _validate_validation_cache(artifact_directory / request_id / job.key, job)
        )
    return {
        "schema_id": "cascadia-v3-cluster-stage-completion-v1",
        "passed": True,
        "request_id": request_id,
        "experiment_id": "cascadia-v3-bootstrap-validation-cache",
        "image_digest": image,
        "work_items": len(jobs),
        "succeeded": len(jobs),
        "reconciled_existing_results": True,
        "totals": dict(sorted(totals.items())),
        "artifact_root": str((artifact_directory / request_id).resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--state-directory", type=Path, required=True)
    parser.add_argument("--artifact-directory", type=Path, required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--completion", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify-collection")
    verify.add_argument("--shard", type=Path, action="append", required=True)
    verify.add_argument("--entries-per-game", type=int, choices=(20, 80), default=80)
    label = subparsers.add_parser("label-roots")
    label.add_argument("--root-shard", type=Path, action="append", required=True)
    label.add_argument("--campaign-state", type=Path, required=True)
    label.add_argument("--v1-weights", type=Path, required=True)
    label.add_argument("--cycle", type=int)
    cache = subparsers.add_parser("cache-validation")
    cache.add_argument("--label-shard", type=Path, action="append", required=True)
    cache.add_argument("--campaign-state", type=Path, required=True)
    cache.add_argument("--manifest", type=Path, required=True)
    cache.add_argument("--reconcile-existing", action="store_true")
    args = parser.parse_args()
    _validate_image(args.image)
    client = _client(args.state_directory, args.artifact_directory)
    _validate_fabric(client.api.nodes())
    store = client.object_store
    assert store is not None
    if args.command == "verify-collection":
        jobs, _ = build_verify_jobs(args.shard, store)
        completion = _monitor(
            client=client,
            image=args.image,
            jobs=jobs,
            resources=Resources(cpu=1, memory_gib=0.75, disk_gib=1),
            request_id=args.request_id,
            experiment_id="cascadia-v3-bootstrap-replay-verification",
            artifact_directory=args.artifact_directory,
            progress=args.progress,
            timeout_seconds=6 * 60 * 60,
            validate=lambda directory, job: _validate_verification(
                directory, job, args.entries_per_game
            ),
        )
    elif args.command == "label-roots":
        expected_phase = (
            "bootstrap_labeling"
            if args.cycle is None
            else f"cycle-{args.cycle:02d}-labeling"
        )
        state = _read_authorized_state(args.campaign_state, expected_phase)
        jobs, _ = build_label_jobs(
            args.root_shard,
            store,
            campaign_state=args.campaign_state,
            v1_weights=args.v1_weights,
            approved_readiness_sha256=state["approved_readiness_sha256"],
            cycle=args.cycle,
        )
        completion = _monitor(
            client=client,
            image=args.image,
            jobs=jobs,
            resources=Resources(cpu=1, memory_gib=0.75, disk_gib=1),
            request_id=args.request_id,
            experiment_id="cascadia-v3-teacher-labeling",
            artifact_directory=args.artifact_directory,
            progress=args.progress,
            timeout_seconds=12 * 60 * 60,
            validate=_validate_label,
        )
    else:
        _read_authorized_state(args.campaign_state, "bootstrap_training")
        jobs, sources = build_validation_cache_jobs(args.label_shard, store)
        if args.reconcile_existing:
            completion = _reconcile_validation_cache(
                jobs=jobs,
                artifact_directory=args.artifact_directory,
                request_id=args.request_id,
                image=args.image,
            )
        else:
            completion = _monitor(
                client=client,
                image=args.image,
                jobs=jobs,
                resources=Resources(cpu=1, memory_gib=1.0, disk_gib=2),
                request_id=args.request_id,
                experiment_id="cascadia-v3-bootstrap-validation-cache",
                artifact_directory=args.artifact_directory,
                progress=args.progress,
                timeout_seconds=4 * 60 * 60,
                validate=_validate_validation_cache,
            )
        _write_atomic(
            args.manifest,
            _validation_cache_manifest(
                jobs=jobs,
                sources=sources,
                artifact_directory=args.artifact_directory,
                request_id=args.request_id,
                image=args.image,
                completion=completion,
            ),
        )
    _write_atomic(args.completion, completion)
    print(json.dumps(completion, sort_keys=True))


if __name__ == "__main__":
    main()
