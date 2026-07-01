#!/usr/bin/env python3
"""Repair only failed V3 expert-collection shards and reconcile an exact corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import v3_cycle_collection
import v3_phase2_jobs
from cascadia_cluster import ContainerInput
from v3_phase2_pipeline import _client


class CollectionRepairError(ValueError):
    """Original and repair artifacts do not form one exact expert corpus."""


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise CollectionRepairError(f"{path} is not a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _link_tree_exact(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise CollectionRepairError(f"collection item directory is absent: {source}")
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if not path.is_file():
            raise CollectionRepairError(f"collection artifact is not regular: {path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.stat().st_size != path.stat().st_size or _sha256(target) != _sha256(path):
                raise CollectionRepairError(f"reconciled artifact already differs: {target}")
            continue
        os.link(path, target)


def _validate_original_request(
    request: dict[str, Any], jobs: list[ContainerInput], image: str
) -> None:
    items = request.get("items")
    if (
        request.get("schema_id") != "cascadia.cluster.managed-request-state.v2"
        or request.get("image_digest") != image
        or not isinstance(items, list)
        or not jobs
        or len(items) != len(jobs)
    ):
        raise CollectionRepairError("original managed request identity differs")
    by_key = {str(item.get("key")): item for item in items if isinstance(item, dict)}
    if set(by_key) != {job.key for job in jobs}:
        raise CollectionRepairError("original request item domain differs")
    for job in jobs:
        payload = by_key[job.key].get("job_payload")
        meta = payload.get("Meta") if isinstance(payload, dict) else None
        if not isinstance(meta, dict):
            raise CollectionRepairError(f"original request metadata is absent: {job.key}")
        for key, value in job.application_metadata.items():
            if meta.get(f"cascadia.app.{key}") != str(value):
                raise CollectionRepairError(f"original request metadata differs: {job.key}/{key}")


def _match_original_layout(
    *,
    request: dict[str, Any],
    plan: dict[str, Any],
    store: Any,
    campaign_state: Path,
    v1_weights: Path,
    newest_model: Path,
    prior_models: list[Path],
    image: str,
) -> tuple[list[ContainerInput], int | None]:
    """Reconstruct the exact legacy or shard-local request layout."""

    failures = []
    for bundles_per_shard in (1, None):
        jobs, _ = v3_cycle_collection.build_jobs(
            plan=plan,
            store=store,
            campaign_state=campaign_state,
            v1_weights=v1_weights,
            newest_model=newest_model,
            prior_models=prior_models,
            prior_bundles_per_shard=bundles_per_shard,
        )
        try:
            _validate_original_request(request, jobs, image)
        except CollectionRepairError as error:
            failures.append(str(error))
            continue
        return jobs, bundles_per_shard
    raise CollectionRepairError(
        "original request matches neither shard-local nor legacy full-pool layout: "
        + "; ".join(failures)
    )


def _artifact_values(directory: Path, job: ContainerInput) -> dict[str, int]:
    return v3_cycle_collection._validate_item(directory, job)


def rejected_items(
    *, original_root: Path, original_request_id: str, jobs: list[ContainerInput]
) -> list[ContainerInput]:
    rejected = []
    for job in jobs:
        try:
            _artifact_values(original_root / original_request_id / job.key, job)
        except (OSError, ValueError, json.JSONDecodeError):
            rejected.append(job)
    return rejected


def reconcile(
    *,
    cycle: int,
    image: str,
    original_request: dict[str, Any],
    original_root: Path,
    repair_completion: dict[str, Any],
    repair_root: Path,
    reconciled_root: Path,
    jobs: list[ContainerInput],
) -> dict[str, Any]:
    original_request_id = str(original_request.get("request_id", ""))
    invalid = rejected_items(
        original_root=original_root,
        original_request_id=original_request_id,
        jobs=jobs,
    )
    invalid_keys = {job.key for job in invalid}
    repair_inputs = repair_completion.get("inputs")
    repaired_keys = {
        str(item.get("item")) for item in repair_inputs or [] if isinstance(item, dict)
    }
    repair_request_id = str(repair_completion.get("request_id", ""))
    if (
        repair_completion.get("schema_id") != "cascadia-v3-cluster-stage-completion-v1"
        or repair_completion.get("passed") is not True
        or repair_completion.get("repair_mode") is not True
        or repair_completion.get("work_items") != len(invalid_keys)
        or repair_completion.get("succeeded") != len(invalid_keys)
        or repaired_keys != invalid_keys
        or not repair_request_id
    ):
        raise CollectionRepairError("repair completion does not replace exactly rejected shards")

    reconciled_root.mkdir(parents=True, exist_ok=True)
    expected_keys = {job.key for job in jobs}
    existing_keys = {path.name for path in reconciled_root.iterdir()}
    unexpected = sorted(existing_keys - expected_keys)
    if unexpected:
        raise CollectionRepairError(
            f"reconciled root contains artifacts outside the item domain: {unexpected}"
        )
    totals: Counter[str] = Counter()
    lineage: dict[str, dict[str, str]] = {}
    for job in jobs:
        if job.key in invalid_keys:
            source_request = repair_request_id
            source = repair_root / repair_request_id / job.key
        else:
            source_request = original_request_id
            source = original_root / original_request_id / job.key
        values = _artifact_values(source, job)
        totals.update(values)
        destination = reconciled_root / job.key
        _link_tree_exact(source, destination)
        _write_atomic(
            destination / "lineage.json",
            {
                "schema_id": "cascadia-v3-expert-collection-reconciled-item-v1",
                "item": job.key,
                "source_request_id": source_request,
                "source_manifest_sha256": _sha256(source / "manifest.json"),
            },
        )
        lineage[job.key] = {"source_request_id": source_request}

    observed_keys = {path.name for path in reconciled_root.iterdir() if path.is_dir()}
    if observed_keys != expected_keys:
        raise CollectionRepairError("reconciled root does not contain the exact item domain")

    expected_v1, expected_prior = v3_cycle_collection.expected_policy_seats(cycle)
    if (
        len(jobs) != 100
        or totals.get("games") != 10_000
        or totals.get("seat_games") != 40_000
        or totals.get("newest_seat_games") != 10_000
        or totals.get("v1_seat_games") != expected_v1
        or totals.get("prior_v3_seat_games") != expected_prior
    ):
        raise CollectionRepairError(f"reconciled policy-seat accounting differs: {totals}")

    newest_ids = {str(job.application_metadata["newest_model_id"]) for job in jobs}
    if len(newest_ids) != 1:
        raise CollectionRepairError("reconciled newest model identity differs across shards")
    return {
        "schema_id": "cascadia-v3-cluster-stage-completion-v1",
        "passed": True,
        "request_id": f"{original_request_id}-reconciled-v1",
        "experiment_id": f"cascadia-v3-expert-cycle-{cycle:02d}-collection-reconciled",
        "image_digest": image,
        "work_items": 100,
        "succeeded": 100,
        "elapsed_seconds": float(repair_completion.get("elapsed_seconds", 0.0)),
        "totals": dict(sorted(totals.items())),
        "artifact_root": str(reconciled_root.resolve()),
        "inputs": [{"item": job.key, **dict(job.application_metadata)} for job in jobs],
        "cycle": cycle,
        "newest_model_id": newest_ids.pop(),
        "opponent_mix": {
            "v1_seat_games": expected_v1,
            "prior_v3_seat_games": expected_prior,
            "v1_fraction": expected_v1 / 30_000,
        },
        "manual_host_sharding": False,
        "scheduler_owns_placement": True,
        "protected_seed_values_opened": False,
        "repair": {
            "reason": "original-shard-memory-limit-exceeded",
            "original_request_id": original_request_id,
            "repair_request_id": repair_request_id,
            "repaired_items": sorted(invalid_keys),
            "repaired_item_count": len(invalid_keys),
            "memory_gib": repair_completion.get("requested_memory_gib"),
            "lineage": lineage,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    state = v3_phase2_jobs._authorized(args.campaign_state)
    phase = str(state.get("phase"))
    expected_phase = f"cycle-{args.cycle:02d}-collecting"
    if phase != expected_phase or state.get("protected_seed_values_opened") is not False:
        raise CollectionRepairError(f"collection repair requires sealed {expected_phase}")

    client = _client(args.state_directory, args.original_artifact_directory)
    store = client.object_store
    assert store is not None
    original_request_path = args.state_directory / "requests" / f"{args.original_request_id}.json"
    original_request = _read(original_request_path)
    plan = v3_phase2_jobs.build_plan(state, args.image, 100)
    jobs, prior_bundles_per_shard = _match_original_layout(
        request=original_request,
        plan=plan,
        store=store,
        campaign_state=args.campaign_state,
        v1_weights=args.v1_weights,
        newest_model=args.newest_model,
        prior_models=args.prior_model,
        image=args.image,
    )
    invalid = rejected_items(
        original_root=args.original_artifact_directory,
        original_request_id=args.original_request_id,
        jobs=jobs,
    )
    if not invalid:
        raise CollectionRepairError("collection repair found no rejected shards")

    repair_request_id = f"{args.original_request_id}-memory-repair-v1"
    repair_completion_path = args.repair_artifact_directory / "completion.json"
    command = [
        str(args.python),
        str(args.collection_program),
        "--image",
        args.image,
        "--campaign-state",
        str(args.campaign_state),
        "--v1-weights",
        str(args.v1_weights),
        "--newest-model",
        str(args.newest_model),
        "--state-directory",
        str(args.state_directory),
        "--artifact-directory",
        str(args.repair_artifact_directory / "accepted"),
        "--request-id",
        repair_request_id,
        "--progress",
        str(args.repair_artifact_directory / "progress.json"),
        "--completion",
        str(repair_completion_path),
        "--memory-gib",
        str(args.memory_gib),
        "--prior-bundles-per-shard",
        "0" if prior_bundles_per_shard is None else "1",
    ]
    for model in args.prior_model:
        command.extend(("--prior-model", str(model)))
    for job in invalid:
        command.extend(("--item-key", job.key))
    if not repair_completion_path.is_file():
        subprocess.run(command, check=True)
    repair_completion = _read(repair_completion_path)
    final = reconcile(
        cycle=args.cycle,
        image=args.image,
        original_request=original_request,
        original_root=args.original_artifact_directory,
        repair_completion=repair_completion,
        repair_root=args.repair_artifact_directory / "accepted",
        reconciled_root=args.reconciled_root,
        jobs=jobs,
    )
    _write_atomic(args.completion, final)
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--campaign-state", type=Path, required=True)
    parser.add_argument("--v1-weights", type=Path, required=True)
    parser.add_argument("--newest-model", type=Path, required=True)
    parser.add_argument("--prior-model", type=Path, action="append", default=[])
    parser.add_argument("--state-directory", type=Path, required=True)
    parser.add_argument("--original-artifact-directory", type=Path, required=True)
    parser.add_argument("--original-request-id", required=True)
    parser.add_argument("--repair-artifact-directory", type=Path, required=True)
    parser.add_argument("--reconciled-root", type=Path, required=True)
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument("--memory-gib", type=float, default=2.0)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--collection-program", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.cycle <= 10 or args.memory_gib <= 1.0:
        raise SystemExit("repair cycle or memory bound is invalid")
    try:
        result = run(args)
    except (
        CollectionRepairError,
        OSError,
        ValueError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
    ) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
