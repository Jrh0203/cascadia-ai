"""Checksum-bound scheduled MLX trainer for one Cascadia V3 bootstrap origin."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from cascadia_mlx.checkpoint import (
    TrainerState,
    load_latest_checkpoint_with_factory,
    prune_checkpoints,
    save_checkpoint,
)
from mlx.utils import tree_flatten

from .contracts import FEATURE_SCALE, V3MlxConfig
from .export import export_quantized_bundle
from .model import (
    ACCUMULATOR_HEADROOM_COEFFICIENT,
    ACCUMULATOR_HEADROOM_LIMIT,
    CsrBatch,
    V3Nnue,
    v3_loss,
)
from .provenance import training_source_identity
from .stream import RustBatchStream

MLX_CACHE_LIMIT_BYTES = 512 * 1024**2
LOSS_PROGRESS_EVERY_STEPS = 25

CAMPAIGN_ID = "cascadia-v3-radius7-stockfish-nnue-v1"
RUN_SCHEMA = "cascadia-v3-bootstrap-origin-run-v2"
MAX_CAMPAIGN_BYTES = 40 * 1024**3
MIN_FREE_BYTES = 50 * 1024**3
COLIMA_BINARY = Path("/opt/homebrew/bin/colima")
COLIMA_HOME = Path("/Users/johnherrick/.local/share/cascadia-r2/colima")
COLIMA_PROFILE = "cascadia-r2"


class CampaignTrainingError(ValueError):
    """The immutable schedule or resumable trainer state is invalid."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _checksum(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _acquire_trainer_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    handle.seek(0)
    handle.truncate()
    handle.write(
        json.dumps(
            {
                "pid": os.getpid(),
                "started_unix_ms": time.time_ns() // 1_000_000,
            },
            sort_keys=True,
        )
        + "\n"
    )
    handle.flush()
    return handle


def _campaign_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _trim_sparse_worker_disk() -> bool:
    """Return deleted worker blocks to APFS without removing live Docker data."""

    if not COLIMA_BINARY.is_file():
        return False
    environment = dict(os.environ)
    environment["COLIMA_HOME"] = str(COLIMA_HOME)
    try:
        completed = subprocess.run(
            [
                str(COLIMA_BINARY),
                "ssh",
                "-p",
                COLIMA_PROFILE,
                "--",
                "sudo",
                "fstrim",
                "-v",
                "/var/lib/docker",
            ],
            check=False,
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def _assert_storage(root: Path, planned_bytes: int = 0) -> None:
    if (
        not isinstance(planned_bytes, int)
        or isinstance(planned_bytes, bool)
        or planned_bytes < 0
    ):
        raise CampaignTrainingError("planned training write must be a nonnegative integer")

    campaign_bytes = _campaign_bytes(root)
    usage = shutil.disk_usage(root)
    campaign_over_limit = campaign_bytes + planned_bytes > MAX_CAMPAIGN_BYTES
    free_space_low = usage.free - planned_bytes < MIN_FREE_BYTES
    if not campaign_over_limit and free_space_low:
        # Bacalhau's terminal containers release ext4 blocks inside Colima, but
        # the sparse disk retains them until the guest advertises the holes to
        # APFS. Reclaim once and then enforce the invariant on fresh readings.
        _trim_sparse_worker_disk()
        campaign_bytes = _campaign_bytes(root)
        usage = shutil.disk_usage(root)
        campaign_over_limit = campaign_bytes + planned_bytes > MAX_CAMPAIGN_BYTES
        free_space_low = usage.free - planned_bytes < MIN_FREE_BYTES
    if campaign_over_limit or free_space_low:
        raise CampaignTrainingError(
            "V3 storage guard refused the next training write: "
            f"campaign_bytes={campaign_bytes}, free_bytes={usage.free}, "
            f"planned_bytes={planned_bytes}"
        )


def _validate_checkpoint_serving(
    *,
    model: V3Nnue,
    args: argparse.Namespace,
    state: TrainerState,
    run_manifest_blake3: str,
) -> None:
    if args.checkpoint_integrity_games == 0:
        return
    receipt = args.run_dir / "checkpoint-integrity" / f"{state.examples_seen:012d}.json"
    if receipt.is_file():
        value = json.loads(receipt.read_text())
        if value.get("passed") is True:
            return
        raise CampaignTrainingError(
            f"checkpoint serving integrity previously failed at {state.examples_seen} exposures"
        )
    serving = args.run_dir / "checkpoint-integrity" / "serving"
    report = (
        args.run_dir
        / "checkpoint-integrity"
        / f"{state.examples_seen:012d}.games.json"
    )
    shutil.rmtree(serving, ignore_errors=True)
    model_manifest = export_quantized_bundle(
        model,
        serving,
        args.feature_manifest,
        training_origin=args.origin,
        checkpoint_id=f"step-{state.global_step:09d}",
        training_run_manifest_blake3=run_manifest_blake3,
    )
    command = [
        str(args.checkpoint_integrity_binary),
        "direct-games",
        "--output",
        str(report),
        "--model-dir",
        str(serving),
        "--games",
        str(args.checkpoint_integrity_games),
        "--first-seed",
        str(args.checkpoint_integrity_first_seed),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    game_report = json.loads(report.read_text()) if report.is_file() else None
    value = {
        "schema_id": "cascadia-v3-checkpoint-serving-integrity-v1",
        "passed": completed.returncode == 0 and report.is_file(),
        "origin": args.origin,
        "examples_seen": state.examples_seen,
        "global_step": state.global_step,
        "games": args.checkpoint_integrity_games,
        "first_seed": args.checkpoint_integrity_first_seed,
        "weights_blake3": model_manifest["weights_blake3"],
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "game_report": str(report.resolve()) if report.is_file() else None,
        "game_report_blake3": _checksum(report) if report.is_file() else None,
        "mean_base_score": (
            game_report.get("mean_base_score") if isinstance(game_report, dict) else None
        ),
        "seat_scores": game_report.get("seat_scores") if isinstance(game_report, dict) else None,
    }
    _write_atomic(receipt, value)
    shutil.rmtree(serving, ignore_errors=True)
    if value["passed"] is not True:
        raise CampaignTrainingError(
            f"checkpoint serving integrity failed at {state.examples_seen} exposures"
        )


def _authorized_state(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    recorded = value.pop("state_sha256", None)
    if recorded != hashlib.sha256(_canonical(value)).hexdigest():
        raise CampaignTrainingError("campaign-state checksum is invalid")
    readiness = value.get("approved_readiness_sha256")
    if (
        value.get("schema_id") != "cascadia-v3-campaign-state-v1"
        or value.get("campaign_id") != CAMPAIGN_ID
        or value.get("part") != 2
        or value.get("phase") != "bootstrap_training"
        or value.get("phase2_authorized") is not True
        or value.get("protected_seed_values_opened") is not False
        or value.get("readiness_sha256") != readiness
    ):
        raise CampaignTrainingError("bootstrap training is not checksum-authorized")
    value["state_sha256"] = recorded
    return value


def _schedule(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    bootstrap = value.get("bootstrap", {})
    blocks = bootstrap.get("blocks")
    if (
        value.get("schema_id") != "cascadia-v3-training-schedule-v2"
        or value.get("architecture_fixed") is not True
        or not isinstance(blocks, list)
        or len(blocks) != 12
        or sum(int(block.get("exposures", 0)) for block in blocks) != 36_000_000
        or bootstrap.get("total_exposures_including_calibration") != 120_000_000
    ):
        raise CampaignTrainingError("bootstrap training schedule is incompatible")
    cursor = 0
    for index, block in enumerate(blocks, start=1):
        mix = block.get("data_mix", {})
        if (
            block.get("block") != index
            or block.get("start_exposure") != cursor
            or block.get("end_exposure") != cursor + block.get("exposures")
            or not math.isclose(sum(float(part) for part in mix.values()), 1.0)
        ):
            raise CampaignTrainingError("bootstrap block cursor or mix drifted")
        cursor += int(block["exposures"])
    return value


def _dataset_identity(paths: list[Path]) -> list[dict[str, object]]:
    return [
        {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "blake3": _checksum(path),
        }
        for path in paths
    ]


def _ordered(paths: list[Path], seed: int, block: int, source: str) -> list[Path]:
    def priority(path: Path) -> bytes:
        digest = blake3.blake3()
        digest.update(b"cascadia-v3-training-shard-order-v1")
        digest.update(seed.to_bytes(8, "little"))
        digest.update(block.to_bytes(2, "little"))
        digest.update(source.encode())
        digest.update(str(path.resolve()).encode())
        return digest.digest()

    return sorted(paths, key=priority)


def _round_robin(
    broad: RustBatchStream | None,
    teacher: RustBatchStream | None,
) -> Iterator[tuple[str, CsrBatch]]:
    streams: list[tuple[str, Iterator[CsrBatch]]] = []
    if broad is not None:
        streams.append(("broad", iter(broad)))
    if teacher is not None:
        streams.append(("teacher", iter(teacher)))
    while streams:
        remaining = []
        for source, stream in streams:
            try:
                yield source, next(stream)
                remaining.append((source, stream))
            except StopIteration:
                pass
        streams = remaining


def _save_swa_snapshot(run_dir: Path, model: V3Nnue, examples_seen: int) -> Path:
    directory = run_dir / "swa"
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / "state.json"
    previous_path = None
    previous_count = 0
    previous: dict[str, mx.array] | None = None
    if state_path.is_file():
        state = json.loads(state_path.read_text())
        previous_count = int(state.get("count", 0))
        previous_name = state.get("average_file")
        if previous_count <= 0 or not isinstance(previous_name, str):
            raise CampaignTrainingError("SWA running-average state is invalid")
        previous_path = directory / previous_name
        if (
            not previous_path.is_file()
            or _checksum(previous_path) != state.get("average_blake3")
            or examples_seen <= int(state.get("last_exposure", -1))
        ):
            raise CampaignTrainingError("SWA running-average state is incomplete or stale")
        previous = mx.load(previous_path)
    current = dict(tree_flatten(model.parameters()))
    if previous is None:
        averaged = {name: value.astype(mx.float32) for name, value in current.items()}
    else:
        if set(previous) != set(current):
            raise CampaignTrainingError("SWA running-average parameter sets differ")
        averaged = {
            name: (
                previous[name].astype(mx.float32) * previous_count
                + current[name].astype(mx.float32)
            )
            / (previous_count + 1)
            for name in current
        }
    mx.eval(averaged)
    count = previous_count + 1
    final = directory / f"average-{count:03d}.safetensors"
    temporary = directory / f".{final.stem}.{uuid.uuid4().hex}.tmp.safetensors"
    mx.save_safetensors(temporary, averaged)
    os.replace(temporary, final)
    value = {
        "schema_id": "cascadia-v3-online-swa-state-v1",
        "count": count,
        "last_exposure": examples_seen,
        "average_file": final.name,
        "average_blake3": _checksum(final),
    }
    _write_atomic(state_path, value)
    if previous_path is not None and previous_path != final:
        previous_path.unlink()
    for orphan in directory.glob("average-*.safetensors"):
        if orphan != final:
            orphan.unlink()
    return final


def _validated_swa_state(run_dir: Path) -> dict[str, Any] | None:
    """Load the single durable SWA generation and verify its content identity."""
    state_path = run_dir / "swa/state.json"
    if not state_path.is_file():
        return None
    state = json.loads(state_path.read_text())
    count = int(state.get("count", 0))
    average_name = state.get("average_file")
    if count <= 0 or not isinstance(average_name, str):
        raise CampaignTrainingError("SWA running-average state is invalid")
    average_path = run_dir / "swa" / average_name
    if (
        not average_path.is_file()
        or _checksum(average_path) != state.get("average_blake3")
    ):
        raise CampaignTrainingError("SWA running-average artifact failed verification")
    return state


def _swa_replay_target(
    *,
    run_dir: Path,
    checkpoint_examples: int,
    swa_start: int,
    swa_interval: int,
    batch_size: int,
    run_manifest_blake3: str,
    checkpoint_id: str,
) -> int | None:
    """Return the last already-averaged event when replaying a newer SWA journal.

    The SWA journal is committed before a boundary checkpoint. A process can
    therefore fail after the journal advances but before the model/optimizer
    checkpoint is admitted. Exact checkpoint continuation deterministically
    reconstructs those same model states. The correct recovery is to retain
    the verified journal and skip only the already represented SWA events.
    """
    state = _validated_swa_state(run_dir)
    if state is None:
        return None
    count = int(state["count"])
    last_exposure = int(state.get("last_exposure", -1))
    event_target = swa_start + (count - 1) * swa_interval
    if not event_target <= last_exposure < event_target + batch_size:
        raise CampaignTrainingError("SWA running-average event sequence is invalid")
    if last_exposure <= checkpoint_examples:
        return None
    receipt = {
        "schema_id": "cascadia-v3-swa-forward-journal-replay-v1",
        "passed": True,
        "checkpoint_id": checkpoint_id,
        "checkpoint_examples": checkpoint_examples,
        "run_manifest_blake3": run_manifest_blake3,
        "swa_count": count,
        "swa_last_exposure": last_exposure,
        "swa_event_target": event_target,
        "swa_average_file": state["average_file"],
        "swa_average_blake3": state["average_blake3"],
        "recovery_rule": "exact-replay-skips-only-events-already-in-forward-journal",
    }
    _write_atomic(run_dir / "swa/replay-recovery.json", receipt)
    return event_target


def _average_swa(run_dir: Path, model: V3Nnue) -> tuple[Path, int]:
    directory = run_dir / "swa"
    state_path = directory / "state.json"
    if not state_path.is_file():
        raise CampaignTrainingError("SWA interval produced no running-average state")
    state = json.loads(state_path.read_text())
    count = int(state.get("count", 0))
    average_name = state.get("average_file")
    if count <= 0 or not isinstance(average_name, str):
        raise CampaignTrainingError("SWA running-average state is invalid")
    average_path = directory / average_name
    if (
        not average_path.is_file()
        or _checksum(average_path) != state.get("average_blake3")
    ):
        raise CampaignTrainingError("SWA running-average artifact failed verification")
    destination = run_dir / "swa-final.safetensors"
    temporary = run_dir / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    shutil.copyfile(average_path, temporary)
    os.replace(temporary, destination)
    model.load_weights(str(destination))
    mx.eval(model.parameters())
    return destination, count


def _bind_manifest(path: Path, value: dict[str, Any], resume: bool) -> str:
    value["canonical_blake3"] = blake3.blake3(_canonical(value)).hexdigest()
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise CampaignTrainingError("resume manifest differs from the immutable origin")
    elif resume:
        raise CampaignTrainingError("resume requested without an origin manifest")
    else:
        _write_atomic(path, value)
    return str(value["canonical_blake3"])


def _bind_manifest_with_source_migration(
    path: Path,
    value: dict[str, Any],
    resume: bool,
    receipt_path: Path,
) -> tuple[str, set[str]]:
    """Admit a checksum-bound operational source migration for one live origin."""
    if not resume or not path.is_file() or not receipt_path.is_file():
        raise CampaignTrainingError("source migration requires a resumable origin and receipt")
    existing = json.loads(path.read_text())
    receipt = json.loads(receipt_path.read_text())
    receipt_blake3 = _checksum(receipt_path)
    if (
        receipt.get("schema_id") != "cascadia-v3-operational-source-migration-v1"
        or receipt.get("passed") is not True
        or receipt.get("to_training_source_blake3")
        != value.get("training_source_identity", {}).get("blake3")
    ):
        raise CampaignTrainingError("source migration receipt does not bind both lineages")
    migration = {
        "schema_id": receipt["schema_id"],
        "receipt": str(receipt_path.resolve()),
        "receipt_blake3": receipt_blake3,
        "from_run_manifest_blake3": receipt["from_run_manifest_blake3"],
        "from_training_source_blake3": receipt["from_training_source_blake3"],
        "to_training_source_blake3": receipt["to_training_source_blake3"],
        "classification": receipt["classification"],
    }
    if existing.get("source_migration") is not None:
        value["source_migration"] = migration
        value["canonical_blake3"] = blake3.blake3(_canonical(value)).hexdigest()
        if existing != value:
            raise CampaignTrainingError("resume manifest differs from the migrated origin")
        return str(value["canonical_blake3"]), {
            str(value["canonical_blake3"]),
            str(receipt["from_run_manifest_blake3"]),
        }
    if (
        receipt.get("from_run_manifest_blake3") != existing.get("canonical_blake3")
        or receipt.get("from_training_source_blake3")
        != existing.get("training_source_identity", {}).get("blake3")
    ):
        raise CampaignTrainingError("source migration receipt does not bind prior lineage")
    old_files = {
        item["path"]: (item["bytes"], item["blake3"])
        for item in existing["training_source_identity"]["files"]
    }
    new_files = {
        item["path"]: (item["bytes"], item["blake3"])
        for item in value["training_source_identity"]["files"]
    }
    changes = [
        {
            "path": name,
            "before_bytes": old_files.get(name, (None, None))[0],
            "before_blake3": old_files.get(name, (None, None))[1],
            "after_bytes": new_files.get(name, (None, None))[0],
            "after_blake3": new_files.get(name, (None, None))[1],
        }
        for name in sorted(set(old_files) | set(new_files))
        if old_files.get(name) != new_files.get(name)
    ]
    if changes != receipt.get("changed_files"):
        raise CampaignTrainingError("source migration changed files differ from the receipt")
    old_contract = dict(existing)
    old_contract.pop("canonical_blake3", None)
    old_contract.pop("training_source_identity", None)
    old_contract.pop("source_migration", None)
    new_contract = dict(value)
    new_contract.pop("training_source_identity", None)
    if old_contract != new_contract:
        raise CampaignTrainingError("source migration changed the scientific run contract")
    value["source_migration"] = migration
    value["canonical_blake3"] = blake3.blake3(_canonical(value)).hexdigest()
    _write_atomic(path, value)
    return str(value["canonical_blake3"]), {
        str(value["canonical_blake3"]),
        str(receipt["from_run_manifest_blake3"]),
    }


def _stream(
    *,
    binary: Path,
    inputs: list[Path],
    config: V3MlxConfig,
    batch_size: int,
    examples: int,
    state: Path,
    block: int,
    teacher_lambda: float | None,
    expansion_threads: int,
) -> RustBatchStream | None:
    if examples == 0:
        return None
    return RustBatchStream(
        binary,
        inputs,
        config,
        batch_size=batch_size,
        epochs=1,
        allow_scientific_data=True,
        d6_cycle=True,
        campaign_state=state,
        teacher_lambda=teacher_lambda,
        max_examples=examples,
        uniform_phase=True,
        d6_offset=block - 1,
        expansion_threads=expansion_threads,
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    # MLX otherwise retains freed graph buffers up to its global memory limit.
    # Sparse V3 batches have variable shapes, so a long run accumulated several
    # GiB of unusable cache generations and forced active model state into swap.
    # The limit affects allocator reuse only; model values and update order are
    # unchanged.
    mx.set_cache_limit(MLX_CACHE_LIMIT_BYTES)
    authorization = _authorized_state(args.campaign_state)
    schedule = _schedule(args.schedule)
    if schedule.get("approved_readiness_sha256") != authorization["approved_readiness_sha256"]:
        raise CampaignTrainingError("training schedule differs from campaign authorization")
    if not args.broad_dataset or not args.teacher_dataset:
        raise CampaignTrainingError("bootstrap training requires broad and teacher datasets")
    for path in [
        args.feature_manifest,
        args.schedule,
        args.batch_stream_binary,
        args.checkpoint_integrity_binary,
        *args.broad_dataset,
        *args.teacher_dataset,
    ]:
        if not path.is_file():
            raise CampaignTrainingError(f"training input is missing: {path}")
    # Refuse an unsavable run before allocating MLX state or consuming the
    # corpus. The same guard remains at every atomic write because free space
    # may change while a long origin is training.
    _assert_storage(args.campaign_root, args.checkpoint_bytes)
    feature = json.loads(args.feature_manifest.read_text())
    config = V3MlxConfig(
        opportunity_feature_rows=feature["opportunity_feature_rows"],
        opportunity_training_factor_rows=feature["opportunity_training_factor_rows"],
    )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_id": RUN_SCHEMA,
        "campaign_state_sha256": authorization["state_sha256"],
        "approved_readiness_sha256": authorization["approved_readiness_sha256"],
        "origin": args.origin,
        "seed": args.seed,
        "base_learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "accumulator_headroom": {
            "limit_float_units": ACCUMULATOR_HEADROOM_LIMIT,
            "limit_integer_units": int(ACCUMULATOR_HEADROOM_LIMIT * FEATURE_SCALE),
            "coefficient": ACCUMULATOR_HEADROOM_COEFFICIENT,
            "scope": "maximum-absolute-own-and-field-accumulator-per-example",
        },
        "batch_size": schedule["batch_size"],
        "mlx_cache_limit_bytes": MLX_CACHE_LIMIT_BYTES,
        "loss_progress_every_steps": LOSS_PROGRESS_EVERY_STEPS,
        "feature_manifest": _dataset_identity([args.feature_manifest])[0],
        "schedule": _dataset_identity([args.schedule])[0],
        "batch_stream_binary": _dataset_identity([args.batch_stream_binary])[0],
        "checkpoint_serving_integrity": {
            "binary": _dataset_identity([args.checkpoint_integrity_binary])[0],
            "games": args.checkpoint_integrity_games,
            "first_seed": args.checkpoint_integrity_first_seed,
        },
        "training_source_identity": training_source_identity(),
        "broad_datasets": _dataset_identity(args.broad_dataset),
        "teacher_datasets": _dataset_identity(args.teacher_dataset),
    }
    if args.source_migration_receipt is not None:
        manifest_hash, accepted_checkpoint_manifests = (
            _bind_manifest_with_source_migration(
                args.run_dir / "run-manifest.json",
                manifest,
                args.resume,
                args.source_migration_receipt,
            )
        )
    else:
        manifest_hash = _bind_manifest(
            args.run_dir / "run-manifest.json", manifest, args.resume
        )
        accepted_checkpoint_manifests = {manifest_hash}
    if args.resume:
        model, optimizer, state, checkpoint = load_latest_checkpoint_with_factory(
            args.run_dir,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            model_factory=lambda values: V3Nnue(V3MlxConfig.from_dict(values)),
        )
        checkpoint_manifest = json.loads((checkpoint / "checkpoint.json").read_text())
        if (
            checkpoint_manifest.get("metadata", {}).get("run_manifest_blake3")
            not in accepted_checkpoint_manifests
        ):
            raise CampaignTrainingError("checkpoint is not bound to this origin manifest")
    else:
        mx.random.seed(args.seed)
        model = V3Nnue(config)
        optimizer = optim.AdamW(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        state = TrainerState()
    if model.config != config:
        raise CampaignTrainingError("checkpoint architecture differs from the feature manifest")

    blocks = schedule["bootstrap"]["blocks"]
    checkpoint_interval = int(schedule["checkpoint_every_exposures"])
    swa = schedule["bootstrap"]["stochastic_weight_averaging"]
    swa_start = int(swa["start_exposure"])
    swa_interval = int(swa["update_interval_exposures"])
    swa_replay_target = (
        _swa_replay_target(
            run_dir=args.run_dir,
            checkpoint_examples=state.examples_seen,
            swa_start=swa_start,
            swa_interval=swa_interval,
            batch_size=int(schedule["batch_size"]),
            run_manifest_blake3=manifest_hash,
            checkpoint_id=checkpoint.name,
        )
        if args.resume
        else None
    )
    next_checkpoint = ((state.examples_seen // checkpoint_interval) + 1) * checkpoint_interval
    next_swa = max(
        swa_start,
        swa_start
        + math.ceil(max(0, state.examples_seen - swa_start) / swa_interval) * swa_interval,
    )
    loss_and_grad = nn.value_and_grad(model, v3_loss)
    loss_path = args.run_dir / "loss.json"
    loss_samples = (
        list(json.loads(loss_path.read_text()).get("samples", []))
        if args.resume and loss_path.is_file()
        else []
    )
    started = time.perf_counter()
    interrupted = False
    planned_stop = False
    if args.planned_stop_after_examples is not None:
        if state.examples_seen > args.planned_stop_after_examples:
            raise CampaignTrainingError("checkpoint is beyond the immutable planned stop")
        planned_stop = state.examples_seen == args.planned_stop_after_examples
    for block_index in range(state.schedule_block, len(blocks)):
        if planned_stop:
            break
        block = blocks[block_index]
        block_number = int(block["block"])
        block_exposures = int(block["exposures"])
        if args.planned_stop_after_examples is not None:
            block_exposures = min(
                block_exposures,
                args.planned_stop_after_examples - int(block["start_exposure"]),
            )
        if block_exposures <= 0:
            planned_stop = True
            break
        mix = block["data_mix"]
        broad_examples = round(block_exposures * float(mix["broad"]))
        teacher_examples = block_exposures - broad_examples
        if broad_examples % 8 or teacher_examples % 8:
            raise CampaignTrainingError("phase-balanced source quotas must be divisible by eight")
        learning_rate = args.learning_rate * float(block["learning_rate_multiplier"])
        optimizer.learning_rate = learning_rate
        teacher_lambda = block["teacher_lambda"]
        active_sources = int(broad_examples > 0) + int(teacher_examples > 0)
        # John1 reserves nine CPU slots during MLX training: one for Python/MLX
        # orchestration and eight for native feature expansion. Broad-only
        # blocks must not leave half of that preprocessing budget idle; mixed
        # blocks split it evenly between their concurrent source streams.
        expansion_threads = 8 // active_sources
        broad = _stream(
            binary=args.batch_stream_binary,
            inputs=_ordered(args.broad_dataset, args.seed, block_number, "broad"),
            config=config,
            batch_size=schedule["batch_size"],
            examples=broad_examples,
            state=args.campaign_state,
            block=block_number,
            teacher_lambda=None,
            expansion_threads=expansion_threads,
        )
        teacher = _stream(
            binary=args.batch_stream_binary,
            inputs=_ordered(args.teacher_dataset, args.seed, block_number, "teacher"),
            config=config,
            batch_size=schedule["batch_size"],
            examples=teacher_examples,
            state=args.campaign_state,
            block=block_number,
            teacher_lambda=teacher_lambda,
            expansion_threads=expansion_threads,
        )
        try:
            for batch_index, (source, batch) in enumerate(_round_robin(broad, teacher)):
                if batch_index < state.batch_in_block:
                    continue
                loss, gradients = loss_and_grad(model, batch)
                optimizer.update(model, gradients)
                mx.eval(loss, model.parameters(), optimizer.state)
                rows = int(batch.targets.shape[0])
                state.global_step += 1
                state.batch_in_block = batch_index + 1
                state.examples_seen += rows
                if source == "broad":
                    state.broad_examples_seen += rows
                else:
                    state.teacher_examples_seen += rows
                state.elapsed_seconds += time.perf_counter() - started
                started = time.perf_counter()
                loss_samples.append(
                    {
                        "step": state.global_step,
                        "examples": state.examples_seen,
                        "block": block_number,
                        "source": source,
                        "loss": float(loss.item()),
                        "learning_rate": learning_rate,
                        "teacher_lambda": teacher_lambda,
                    }
                )
                if len(loss_samples) > 2_000:
                    loss_samples = loss_samples[-2_000:]
                if state.global_step % LOSS_PROGRESS_EVERY_STEPS == 0:
                    _write_atomic(loss_path, {"samples": loss_samples})
                save = False
                if state.examples_seen >= next_checkpoint:
                    save = True
                    while next_checkpoint <= state.examples_seen:
                        next_checkpoint += checkpoint_interval
                while (
                    swa_replay_target is not None
                    and next_swa <= swa_replay_target
                    and state.examples_seen >= next_swa
                ):
                    next_swa += swa_interval
                if state.examples_seen >= next_swa:
                    _assert_storage(args.campaign_root, args.swa_snapshot_bytes)
                    _save_swa_snapshot(args.run_dir, model, state.examples_seen)
                    while next_swa <= state.examples_seen:
                        next_swa += swa_interval
                if save:
                    _assert_storage(args.campaign_root, args.checkpoint_bytes)
                    save_checkpoint(
                        args.run_dir,
                        model,
                        optimizer,
                        state,
                        metadata={
                            "run_manifest_blake3": manifest_hash,
                            "block": block_number,
                            "examples_seen": state.examples_seen,
                            "teacher_lambda": teacher_lambda,
                        },
                    )
                    prune_checkpoints(args.run_dir, keep_recent=2)
                    _write_atomic(args.run_dir / "loss.json", {"samples": loss_samples})
                    _validate_checkpoint_serving(
                        model=model,
                        args=args,
                        state=state,
                        run_manifest_blake3=manifest_hash,
                    )
                if (
                    args.stop_after_examples is not None
                    and state.examples_seen >= args.stop_after_examples
                ):
                    interrupted = True
                    break
                if (
                    args.planned_stop_after_examples is not None
                    and state.examples_seen >= args.planned_stop_after_examples
                ):
                    planned_stop = True
                    break
        finally:
            if broad is not None:
                broad.close()
            if teacher is not None:
                teacher.close()
        if interrupted or planned_stop:
            _assert_storage(args.campaign_root, args.checkpoint_bytes)
            save_checkpoint(
                args.run_dir,
                model,
                optimizer,
                state,
                metadata={
                    "run_manifest_blake3": manifest_hash,
                    "controlled_interruption": interrupted,
                    "planned_stop": planned_stop,
                    "block": block_number,
                    "examples_seen": state.examples_seen,
                    "teacher_lambda": teacher_lambda,
                },
            )
            prune_checkpoints(args.run_dir, keep_recent=2)
            _write_atomic(loss_path, {"samples": loss_samples})
            _validate_checkpoint_serving(
                model=model,
                args=args,
                state=state,
                run_manifest_blake3=manifest_hash,
            )
            break
        state.schedule_block = block_index + 1
        state.batch_in_block = 0
        _assert_storage(args.campaign_root, args.checkpoint_bytes)
        save_checkpoint(
            args.run_dir,
            model,
            optimizer,
            state,
            metadata={
                "run_manifest_blake3": manifest_hash,
                "completed_block": block_number,
                "examples_seen": state.examples_seen,
                "teacher_lambda": teacher_lambda,
            },
        )
        prune_checkpoints(args.run_dir, keep_recent=2)
        _write_atomic(loss_path, {"samples": loss_samples})
        _validate_checkpoint_serving(
            model=model,
            args=args,
            state=state,
            run_manifest_blake3=manifest_hash,
        )

    swa_path = None
    swa_samples = 0
    if not interrupted and not planned_stop and state.schedule_block == len(blocks):
        swa_path, swa_samples = _average_swa(args.run_dir, model)
        _assert_storage(args.campaign_root, args.checkpoint_bytes)
        save_checkpoint(
            args.run_dir,
            model,
            optimizer,
            state,
            metadata={
                "run_manifest_blake3": manifest_hash,
                "training_complete": True,
                "swa_samples": swa_samples,
                "swa_blake3": _checksum(swa_path),
            },
        )
        prune_checkpoints(args.run_dir, keep_recent=2)
        _validate_checkpoint_serving(
            model=model,
            args=args,
            state=state,
            run_manifest_blake3=manifest_hash,
        )
    report = {
        "schema_id": "cascadia-v3-bootstrap-origin-training-report-v1",
        "passed": (
            (not interrupted and state.schedule_block == len(blocks))
            or (
                planned_stop
                and args.planned_stop_after_examples is not None
                and state.examples_seen == args.planned_stop_after_examples
            )
        ),
        "interrupted": interrupted,
        "planned_stop": planned_stop,
        "planned_stop_after_examples": args.planned_stop_after_examples,
        "origin": args.origin,
        "run_manifest_blake3": manifest_hash,
        "examples_seen": state.examples_seen,
        "broad_examples_seen": state.broad_examples_seen,
        "teacher_examples_seen": state.teacher_examples_seen,
        "global_step": state.global_step,
        "completed_blocks": state.schedule_block,
        "swa_samples": swa_samples,
        "swa_path": str(swa_path) if swa_path is not None else None,
        "elapsed_seconds": state.elapsed_seconds,
        "latest_loss": loss_samples[-1] if loss_samples else None,
    }
    _write_atomic(args.run_dir / "training-report.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--campaign-state", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--batch-stream-binary", type=Path, required=True)
    parser.add_argument("--checkpoint-integrity-binary", type=Path, required=True)
    parser.add_argument("--checkpoint-integrity-games", type=int, default=2)
    parser.add_argument("--checkpoint-integrity-first-seed", type=int, default=1_650_000)
    parser.add_argument("--broad-dataset", type=Path, action="append", default=[])
    parser.add_argument("--teacher-dataset", type=Path, action="append", default=[])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--origin", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--source-migration-receipt", type=Path)
    parser.add_argument("--stop-after-examples", type=int)
    parser.add_argument("--planned-stop-after-examples", type=int)
    parser.add_argument("--checkpoint-bytes", type=int, default=1280 * 1024**2)
    parser.add_argument("--swa-snapshot-bytes", type=int, default=512 * 1024**2)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.seed < 0 or args.learning_rate <= 0 or args.weight_decay < 0:
        raise SystemExit("seed and optimizer parameters are invalid")
    if args.checkpoint_integrity_games < 0 or args.checkpoint_integrity_first_seed < 0:
        raise SystemExit("checkpoint integrity game settings are invalid")
    if args.stop_after_examples is not None and args.stop_after_examples <= 0:
        raise SystemExit("stop-after-examples must be positive")
    if (
        args.planned_stop_after_examples is not None
        and args.planned_stop_after_examples <= 0
    ):
        raise SystemExit("planned-stop-after-examples must be positive")
    if args.stop_after_examples is not None and args.planned_stop_after_examples is not None:
        raise SystemExit("controlled interruption and planned stop are mutually exclusive")
    lock = _acquire_trainer_lock(args.run_dir / ".trainer.lock")
    if lock is None:
        raise SystemExit(f"another trainer owns {args.run_dir}")
    try:
        report = run(args)
    except (CampaignTrainingError, OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
