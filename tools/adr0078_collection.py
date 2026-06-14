"""Distributed collection, validation, and john3 provisioning for ADR 0078."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import time
from typing import Any

import adr0078_cluster_runtime as rt


def wait_for_collections() -> tuple[dict[str, Any], dict[str, Any]]:
    rt.log("waiting for the frozen train and validation manifests")
    state = rt.load_state()
    previous = {
        "train": int(state.get("train_completed", 0)),
        "validation": int(state.get("validation_completed", 0)),
    }
    rt.update_state(
        "collecting",
        train_completed=previous["train"],
        validation_completed=previous["validation"],
        unavailable_host=None,
    )
    progress_at = {label: time.monotonic() for label in previous}
    unavailable_since: dict[str, float] = {}
    while True:
        manifests: dict[str, dict[str, Any]] = {}
        all_complete = True
        for spec in (rt.TRAIN_SPEC, rt.VALIDATION_SPEC):
            try:
                manifest = rt.load_manifest(spec)
            except rt.RemoteHostUnavailable as error:
                all_complete = False
                if spec.label not in unavailable_since:
                    unavailable_since[spec.label] = time.monotonic()
                    rt.log(
                        f"{spec.label} host unavailable; preserving "
                        f"{previous[spec.label]}/{spec.requested_games}: {error}"
                    )
                    rt.update_state(
                        "collecting",
                        train_completed=previous["train"],
                        validation_completed=previous["validation"],
                        unavailable_host=spec.host,
                    )
                if time.monotonic() - unavailable_since[spec.label] > rt.STALE_PROGRESS_SECONDS:
                    raise RuntimeError(
                        f"{spec.label} host remained unreachable for "
                        f"{rt.STALE_PROGRESS_SECONDS // 60} minutes"
                    ) from error
                continue
            if spec.label in unavailable_since:
                rt.log(f"{spec.label} host connectivity recovered")
                unavailable_since.pop(spec.label)
                rt.update_state(
                    "collecting",
                    train_completed=previous["train"],
                    validation_completed=previous["validation"],
                    unavailable_host=None,
                )

            if manifest is None:
                completed = 0
                all_complete = False
            else:
                rt.validate_manifest_contract(manifest, spec, require_complete=False)
                manifests[spec.label] = manifest
                completed = int(manifest["completed_games"])
                all_complete &= completed == spec.requested_games

            if completed < previous[spec.label]:
                raise RuntimeError(
                    f"{spec.label} manifest regressed from "
                    f"{previous[spec.label]} to {completed} completed games"
                )
            if previous[spec.label] != completed:
                rt.log(f"{spec.label} progress: {completed}/{spec.requested_games}")
                previous[spec.label] = completed
                progress_at[spec.label] = time.monotonic()
                rt.update_state(
                    "collecting",
                    train_completed=previous.get("train", 0),
                    validation_completed=previous.get("validation", 0),
                )

            if completed < spec.requested_games:
                try:
                    running = rt.collector_running(spec)
                except rt.RemoteHostUnavailable as error:
                    unavailable_since.setdefault(spec.label, time.monotonic())
                    rt.log(
                        f"{spec.label} process probe unavailable; preserving "
                        f"{completed}/{spec.requested_games}: {error}"
                    )
                    continue
                if not running:
                    raise RuntimeError(
                        f"{spec.label} collector stopped at {completed}/{spec.requested_games}"
                    )
                if (
                    time.monotonic() - progress_at.get(spec.label, time.monotonic())
                    > rt.STALE_PROGRESS_SECONDS
                ):
                    raise RuntimeError(
                        f"{spec.label} collector made no manifest progress for "
                        f"{rt.STALE_PROGRESS_SECONDS // 60} minutes"
                    )

        if all_complete:
            train = manifests["train"]
            validation = manifests["validation"]
            rt.validate_manifest_contract(train, rt.TRAIN_SPEC, require_complete=True)
            rt.validate_manifest_contract(validation, rt.VALIDATION_SPEC, require_complete=True)
            rt.update_state(
                "collection-complete",
                train_completed=128,
                validation_completed=32,
            )
            return train, validation
        time.sleep(rt.POLL_SECONDS)


def verify_binary_identity(*, require_remote: bool = True) -> None:
    if rt.sha256_file(rt.LOCAL_BINARY) != rt.EXPECTED_EXECUTABLE_SHA256:
        raise ValueError("local frozen collector SHA-256 changed")
    local_blake3 = rt.run([str(rt.B3SUM), str(rt.LOCAL_BINARY)]).stdout.split()[0]
    if local_blake3 != rt.EXPECTED_EXECUTABLE_BLAKE3:
        raise ValueError("local frozen collector BLAKE3 changed")
    result = rt.remote_shell(
        "john2",
        ["shasum", "-a", "256", str(rt.JOHN2_ROOT / "target/release/cascadia-v2")],
        check=False,
    )
    if result.returncode == 255 and not require_remote:
        rt.log("john2 binary identity check deferred until connectivity recovers")
        return
    if result.returncode == 255:
        raise rt.RemoteHostUnavailable("john2 is unreachable")
    if result.returncode != 0:
        raise RuntimeError(f"could not verify john2 frozen collector: exit {result.returncode}")
    john2_sha = result.stdout.split()[0]
    if john2_sha != rt.EXPECTED_EXECUTABLE_SHA256:
        raise ValueError("john2 frozen collector changed")


def validate_on_producer_hosts() -> None:
    rt.log("validating complete datasets on their producing hosts")
    rt.run(
        [
            str(rt.LOCAL_BINARY),
            "validate-counterfactual-advantage-dataset",
            "--dataset",
            str(rt.TRAIN_DATASET),
        ]
    )
    rt.remote(
        "john2",
        (
            f"cd {shlex.quote(str(rt.JOHN2_ROOT))} && "
            "./target/release/cascadia-v2 "
            "validate-counterfactual-advantage-dataset "
            "--dataset artifacts/datasets/"
            "r12-counterfactual-advantage-v1-validation-32"
        ),
    )
    rt.update_state("producer-validation-complete")


def sync_validation_to_john1() -> None:
    rt.log("copying validation dataset from john2 to john1")
    incoming = rt.VALIDATION_DATASET.with_name(rt.VALIDATION_DATASET.name + ".incoming")
    if incoming.exists():
        shutil.rmtree(incoming)
    incoming.mkdir(parents=True)
    rt.rsync_from_remote(
        "john2",
        f"{rt.REMOTE_VALIDATION_DATASET}/",
        f"{incoming}/",
        delete=True,
    )
    rt.run(
        [
            str(rt.LOCAL_BINARY),
            "validate-counterfactual-advantage-dataset",
            "--dataset",
            str(incoming),
        ]
    )
    if rt.VALIDATION_DATASET.exists():
        if rt.sha256_file(rt.VALIDATION_DATASET / "dataset.json") != rt.sha256_file(
            incoming / "dataset.json"
        ):
            raise ValueError("existing john1 validation dataset differs from john2")
        shutil.rmtree(incoming)
    else:
        os.replace(incoming, rt.VALIDATION_DATASET)
    rt.run(
        [
            str(rt.LOCAL_BINARY),
            "validate-counterfactual-advantage-dataset",
            "--dataset",
            str(rt.VALIDATION_DATASET),
        ]
    )
    rt.update_state(
        "john1-aggregation-complete",
        train_manifest_sha256=rt.sha256_file(rt.TRAIN_DATASET / "dataset.json"),
        validation_manifest_sha256=rt.sha256_file(rt.VALIDATION_DATASET / "dataset.json"),
    )


def provision_john3_data() -> None:
    rt.log("copying the frozen binary and validated datasets to john3")
    rt.remote(
        "john3",
        (
            f"mkdir -p {shlex.quote(str(rt.JOHN3_ROOT / 'target/release'))} "
            f"{shlex.quote(str(rt.JOHN3_ROOT / 'artifacts/datasets'))} "
            f"{shlex.quote(str(rt.JOHN3_ROOT / 'artifacts/runs'))} "
            f"{shlex.quote(str(rt.JOHN3_ROOT / 'artifacts/logs'))}"
        ),
    )
    rt.rsync_to_remote("john3", str(rt.LOCAL_BINARY), str(rt.JOHN3_BINARY))
    rt.rsync_to_remote(
        "john3",
        f"{rt.TRAIN_DATASET}/",
        f"{rt.JOHN3_TRAIN_DATASET}/",
        delete=True,
    )
    rt.rsync_to_remote(
        "john3",
        f"{rt.VALIDATION_DATASET}/",
        f"{rt.JOHN3_VALIDATION_DATASET}/",
        delete=True,
    )
    john3_sha = rt.remote_shell(
        "john3",
        ["shasum", "-a", "256", str(rt.JOHN3_BINARY)],
    ).stdout.split()[0]
    if john3_sha != rt.EXPECTED_EXECUTABLE_SHA256:
        raise ValueError("john3 frozen validator binary changed in transfer")
    rt.remote(
        "john3",
        (
            f"cd {shlex.quote(str(rt.JOHN3_ROOT))} && "
            "./target/release/cascadia-v2 "
            "validate-counterfactual-advantage-dataset "
            "--dataset artifacts/datasets/r12-counterfactual-advantage-v1-train-128 && "
            "./target/release/cascadia-v2 "
            "validate-counterfactual-advantage-dataset "
            "--dataset artifacts/datasets/"
            "r12-counterfactual-advantage-v1-validation-32"
        ),
    )
    source = rt.remote(
        "john3",
        (
            f"cd {shlex.quote(str(rt.JOHN3_ROOT))} && "
            "PYTHONPATH=python .venv/bin/python -c "
            + shlex.quote(
                "from pathlib import Path; "
                "from cascadia_mlx.run_manifest import source_provenance; "
                "import json; "
                "print(json.dumps(source_provenance(Path('.').resolve())))"
            )
        ),
    )
    provenance = json.loads(source.stdout)
    if provenance.get("git_revision") != rt.EXPECTED_REVISION:
        raise ValueError("john3 training revision changed")
    if provenance.get("v2_source_blake3") != rt.EXPECTED_JOHN3_SOURCE:
        raise ValueError("john3 frozen training source changed")
    device = rt.remote(
        "john3",
        (
            f"cd {shlex.quote(str(rt.JOHN3_ROOT))} && "
            ".venv/bin/python -c "
            + shlex.quote("import mlx.core as mx; print(mx.default_device())")
        ),
    ).stdout.strip()
    if device != "Device(gpu, 0)":
        raise ValueError(f"john3 MLX device changed: {device}")
    rt.update_state(
        "john3-ready",
        john3_source_blake3=provenance["v2_source_blake3"],
        john3_device=device,
    )
