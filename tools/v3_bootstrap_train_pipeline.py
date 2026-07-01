#!/usr/bin/env python3
"""Run the checksum-bound V3 bootstrap calibration and three-origin selection."""

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
from pathlib import Path
from typing import Any

import blake3
from cluster_build_push import DEFAULT_DOCKER_HOST, _workspace_source_identity
from v3_checkpoint_lifecycle import (
    compact_completed_run,
    retire_completed_run,
    retire_rejected_run,
)

REPOSITORY = Path("/Users/johnherrick/cascadia")
ROOT = Path("/Users/johnherrick/cascadia-bench/v3-nnue")
PYTHON = REPOSITORY / ".venv/bin/python"
FEATURE_MANIFEST = ROOT / "models/feature-schema.json"
SCHEDULE = ROOT / "control/bootstrap-training-schedule.json"
CAMPAIGN_STATE = ROOT / "control/campaign-state.json"
BATCH_STREAM = REPOSITORY / "target/release/v3-batch-stream"
GAME_BINARY = REPOSITORY / "target/release/v3-engineering-smoke"
CONTROLLER_LOCK = ROOT / "control/bootstrap-training-controller.lock"
FINAL_IMAGE_RECEIPT = ROOT / "smoke/image-publication-stage2-final-v1.json"
FINAL_IMAGE_HEALTH = ROOT / "smoke/image-health-stage2-final-v1.json"
FINAL_HANDOFF = ROOT / "phase2/bootstrap/training/stage2-worker-handoff.json"
VALIDATION_CACHE_MANIFEST = ROOT / "phase2/bootstrap/training/validation-cache/manifest.json"
CALIBRATION_RATES = (5e-4, 1e-3, 1.5e-3)
CALIBRATION_EXPOSURES = 4_000_000
CALIBRATION_SEED = 83_000
ORIGIN_SEEDS = (83_101, 83_102, 83_103)
OPEN_GAMES = 32
OPEN_FIRST_SEED = 1_700_000
NONREGRESSION_MARGIN = -1.00


class BootstrapTrainingError(ValueError):
    """The bootstrap training or selection contract failed."""


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _digest(path: Path, algorithm: str = "blake3") -> str:
    digest = blake3.blake3() if algorithm == "blake3" else hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _acquire_controller_lock(path: Path):
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


def _run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as output:
        output.write(("\n$ " + " ".join(command) + "\n").encode())
        output.flush()
        subprocess.run(
            command,
            cwd=REPOSITORY,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
        )


def _wait_for_inputs(poll_seconds: int) -> tuple[list[Path], list[Path], list[Path]]:
    corpus_path = ROOT / "phase2/bootstrap/corpus.json"
    labels_path = ROOT / "phase2/bootstrap/teacher-label-corpus.json"
    while True:
        if corpus_path.is_file() and labels_path.is_file() and CAMPAIGN_STATE.is_file():
            state = _read(CAMPAIGN_STATE)
            if state.get("phase") == "bootstrap_training":
                break
        failure = ROOT / "phase2/bootstrap/pipeline-failure.json"
        if failure.is_file():
            raise BootstrapTrainingError(f"bootstrap data pipeline failed: {failure.read_text()}")
        time.sleep(poll_seconds)
    corpus = _read(corpus_path)
    labels = _read(labels_path)
    if corpus.get("passed") is not True or labels.get("passed") is not True:
        raise BootstrapTrainingError("bootstrap corpus or teacher labels are not eligible")
    broad = [Path(item["path"]) for item in corpus["files"]]
    teacher = [Path(item["path"]) for item in labels["files"] if item["split"] == "teacher"]
    validation_records = [item for item in labels["files"] if item["split"] == "validation"]
    validation = [Path(item["path"]) for item in validation_records]
    if VALIDATION_CACHE_MANIFEST.is_file():
        cache = _read(VALIDATION_CACHE_MANIFEST)
        shards = cache.get("shards")
        totals = cache.get("totals")
        if (
            cache.get("schema_id") != "cascadia-v3-validation-cache-v1"
            or cache.get("passed") is not True
            or not isinstance(shards, list)
            or len(shards) != 20
            or totals
            != {
                "shards": 20,
                "roots": 20_000,
                "rows": sum(int(item["candidate_estimates"]) for item in validation_records),
                "bytes": sum(int(item["bytes"]) for item in shards),
            }
        ):
            raise BootstrapTrainingError("validation cache manifest is invalid")
        expected = {
            str(Path(item["path"]).resolve()): (item["blake3"], item["bytes"])
            for item in validation_records
        }
        observed = {
            str(Path(item["source_path"]).resolve()): (
                item["source_blake3"],
                item["source_bytes"],
            )
            for item in shards
        }
        if observed != expected:
            raise BootstrapTrainingError("validation cache source identity differs")
        cached = [Path(item["path"]) for item in shards]
        if any(
            not path.is_file()
            or path.stat().st_size != int(item["bytes"])
            or _digest(path) != item["blake3"]
            for path, item in zip(cached, shards, strict=True)
        ):
            raise BootstrapTrainingError("validation cache shard integrity failed")
        validation = cached
    if len(broad) != 250 or len(teacher) != 100 or len(validation) != 20:
        raise BootstrapTrainingError(
            f"bootstrap training inputs differ: {len(broad)}/{len(teacher)}/{len(validation)}"
        )
    missing = [str(path) for path in [*broad, *teacher, *validation] if not path.is_file()]
    if missing:
        raise BootstrapTrainingError(f"bootstrap training files are missing: {missing[:3]}")
    return broad, teacher, validation


def _training_command(
    *,
    run_dir: Path,
    origin: str,
    seed: int,
    learning_rate: float,
    broad: list[Path],
    teacher: list[Path],
    planned_stop: int | None,
) -> list[str]:
    command = [
        str(PYTHON),
        "-m",
        "cascadia_v3_mlx.campaign_train",
        "--campaign-root",
        str(ROOT),
        "--campaign-state",
        str(CAMPAIGN_STATE),
        "--feature-manifest",
        str(FEATURE_MANIFEST),
        "--schedule",
        str(SCHEDULE),
        "--batch-stream-binary",
        str(BATCH_STREAM),
        "--checkpoint-integrity-binary",
        str(GAME_BINARY),
        "--checkpoint-integrity-games",
        "2",
        "--checkpoint-integrity-first-seed",
        "1650000",
        "--run-dir",
        str(run_dir),
        "--origin",
        origin,
        "--seed",
        str(seed),
        "--learning-rate",
        repr(learning_rate),
    ]
    for path in broad:
        command.extend(("--broad-dataset", str(path)))
    for path in teacher:
        command.extend(("--teacher-dataset", str(path)))
    if planned_stop is not None:
        command.extend(("--planned-stop-after-examples", str(planned_stop)))
    if (run_dir / "latest.json").is_file():
        command.append("--resume")
    migration = run_dir / "source-migration.json"
    if migration.is_file():
        command.extend(("--source-migration-receipt", str(migration)))
    return command


def _train(
    *,
    run_dir: Path,
    origin: str,
    seed: int,
    learning_rate: float,
    broad: list[Path],
    teacher: list[Path],
    planned_stop: int | None,
) -> dict[str, Any]:
    report = run_dir / "training-report.json"
    if report.is_file():
        value = _read(report)
        if value.get("passed") is True:
            expected = planned_stop if planned_stop is not None else 36_000_000
            if value.get("examples_seen") == expected:
                return value
    _run(
        _training_command(
            run_dir=run_dir,
            origin=origin,
            seed=seed,
            learning_rate=learning_rate,
            broad=broad,
            teacher=teacher,
            planned_stop=planned_stop,
        ),
        run_dir / "training.log",
    )
    value = _read(report)
    expected = planned_stop if planned_stop is not None else 36_000_000
    if value.get("passed") is not True or value.get("examples_seen") != expected:
        raise BootstrapTrainingError(f"training did not reach its exact exposure gate: {run_dir}")
    return value


def _evaluate(
    *,
    run_dir: Path,
    origin: str,
    validation: list[Path],
    games: int,
    cycle: int | None = None,
    tolerate_failure: bool = False,
) -> dict[str, Any]:
    output = run_dir / "evaluation.json"
    game_output = run_dir / "evaluation-open-games.json"
    failure_output = run_dir / "evaluation-failure.json"
    if output.is_file() and game_output.is_file():
        value = _read(output)
        if value.get("passed") is True:
            compact_completed_run(run_dir)
            return value
    if tolerate_failure and failure_output.is_file():
        value = _read(failure_output)
        if value.get("passed") is False:
            return value
    export = run_dir / "serving"
    command = [
        str(PYTHON),
        "-m",
        "cascadia_v3_mlx.checkpoint_eval",
        "--run-dir",
        str(run_dir),
        "--feature-manifest",
        str(FEATURE_MANIFEST),
        "--stream-binary",
        str(BATCH_STREAM),
        "--campaign-state",
        str(CAMPAIGN_STATE),
        "--training-origin",
        origin,
        "--export-dir",
        str(export),
        "--game-binary",
        str(GAME_BINARY),
        "--games",
        str(games),
        "--first-seed",
        str(OPEN_FIRST_SEED),
        "--output",
        str(output),
    ]
    for path in validation:
        command.extend(("--dataset", str(path)))
    if cycle is not None:
        command.extend(("--cycle", str(cycle)))
    log = run_dir / "evaluation.log"
    try:
        _run(command, log)
    except subprocess.CalledProcessError as error:
        failure = {
            "schema_id": "cascadia-v3-checkpoint-evaluation-failure-v1",
            "passed": False,
            "run_dir": str(run_dir.resolve()),
            "training_origin": origin,
            "returncode": error.returncode,
            "command": command,
            "log": str(log.resolve()),
            "log_blake3": _digest(log),
            "reason": "checkpoint evaluation or serving-integrity gate failed",
        }
        _write_atomic(failure_output, failure)
        if tolerate_failure:
            return failure
        raise
    value = _read(output)
    if value.get("passed") is not True or not game_output.is_file():
        raise BootstrapTrainingError(f"checkpoint evaluation is incomplete: {run_dir}")
    compact_completed_run(run_dir)
    return value


def paired_nonregression(treatment: list[int], control: list[int]) -> dict[str, object]:
    if len(treatment) != len(control) or len(treatment) < 2:
        raise BootstrapTrainingError("paired open-game score vectors differ")
    deltas = [float(left) - float(right) for left, right in zip(treatment, control, strict=True)]
    mean = sum(deltas) / len(deltas)
    variance = sum((value - mean) ** 2 for value in deltas) / (len(deltas) - 1)
    standard_error = math.sqrt(variance / len(deltas))
    lower = mean - 1.96 * standard_error
    return {
        "pairs": len(deltas),
        "mean_delta": mean,
        "standard_error": standard_error,
        "lower_95": lower,
        "registered_margin": NONREGRESSION_MARGIN,
        "passed": lower >= NONREGRESSION_MARGIN,
    }


def _candidate(
    *,
    label: str,
    rate: float,
    training: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    if evaluation.get("passed") is not True:
        return {
            "label": label,
            "learning_rate": rate,
            "training_report": training,
            "evaluation_path": str(Path(evaluation["run_dir"]) / "evaluation-failure.json"),
            "evaluation_passed": False,
            "evaluation_failure": evaluation,
        }
    games = _read(Path(evaluation["open_games"]))
    return {
        "label": label,
        "learning_rate": rate,
        "training_report": training,
        "evaluation_path": str(Path(evaluation["run_dir"]) / "evaluation.json"),
        "quantized_validation_loss": evaluation["validation"]["quantized_power_loss"],
        "open_game_mean": games["mean_base_score"],
        "open_scores": games["seat_scores"],
        "evaluation_passed": True,
    }


def _select(candidates: list[dict[str, Any]], purpose: str) -> dict[str, Any]:
    valid = [
        candidate for candidate in candidates if candidate.get("evaluation_passed") is not False
    ]
    if not valid:
        raise BootstrapTrainingError(f"no {purpose} candidate passed checkpoint evaluation")
    reference = max(valid, key=lambda candidate: candidate["open_game_mean"])
    for candidate in candidates:
        if candidate.get("evaluation_passed") is False:
            candidate["open_nonregression"] = {
                "passed": False,
                "reason": "checkpoint evaluation failed",
                "reference": reference["label"],
            }
            candidate["eligible"] = False
            continue
        nonregression = paired_nonregression(candidate["open_scores"], reference["open_scores"])
        candidate["open_nonregression"] = {
            **nonregression,
            "reference": reference["label"],
        }
        candidate["eligible"] = nonregression["passed"]
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    if not eligible:
        raise BootstrapTrainingError(f"no {purpose} candidate passed open-game nonregression")
    return min(eligible, key=lambda candidate: candidate["quantized_validation_loss"])


def _retire_candidate(candidate: dict[str, Any], *, reason: str) -> dict[str, Any]:
    run_dir = Path(candidate["run_dir"])
    if candidate.get("evaluation_passed") is False:
        return retire_rejected_run(run_dir, reason=reason)
    return retire_completed_run(run_dir, reason=reason)


def _calibrate(
    broad: list[Path], teacher: list[Path], validation: list[Path]
) -> tuple[float, dict[str, Any]]:
    directory = ROOT / "phase2/bootstrap/training/calibration"
    receipt_path = directory / "selection.json"
    if receipt_path.is_file():
        receipt = _read(receipt_path)
        if receipt.get("passed") is True:
            for candidate in receipt["candidates"]:
                _retire_candidate(candidate, reason="learning-rate-calibration-complete")
            return float(receipt["selected_learning_rate"]), receipt
    candidates = []
    for index, rate in enumerate(CALIBRATION_RATES, start=1):
        label = f"calibration-{index}-lr-{rate:g}"
        run_dir = directory / label
        training = _train(
            run_dir=run_dir,
            origin=label,
            seed=CALIBRATION_SEED,
            learning_rate=rate,
            broad=broad,
            teacher=teacher,
            planned_stop=CALIBRATION_EXPOSURES,
        )
        evaluation = _evaluate(
            run_dir=run_dir,
            origin=label,
            validation=validation,
            games=OPEN_GAMES,
            tolerate_failure=True,
        )
        candidate = _candidate(
            label=label,
            rate=rate,
            training=training,
            evaluation=evaluation,
        )
        candidate["run_dir"] = str(run_dir.resolve())
        candidates.append(candidate)
    selected = _select(candidates, "calibration")
    receipt = {
        "schema_id": "cascadia-v3-bootstrap-learning-rate-selection-v1",
        "passed": True,
        "selection_rule": "minimum-quantized-validation-loss-among-open-nonregressing",
        "open_domain": {
            "games": OPEN_GAMES,
            "first_seed": OPEN_FIRST_SEED,
            "paired_seat_scores": OPEN_GAMES * 4,
            "nonregression_margin": NONREGRESSION_MARGIN,
            "reference": "best-open-mean-among-the-three-calibration-rates",
        },
        "candidates": candidates,
        "selected": selected["label"],
        "selected_learning_rate": selected["learning_rate"],
    }
    _write_atomic(receipt_path, receipt)
    for candidate in candidates:
        _retire_candidate(candidate, reason="learning-rate-calibration-complete")
    return float(selected["learning_rate"]), receipt


def _origins(
    *,
    learning_rate: float,
    broad: list[Path],
    teacher: list[Path],
    validation: list[Path],
) -> tuple[Path, dict[str, Any]]:
    directory = ROOT / "phase2/bootstrap/training/origins"
    receipt_path = directory / "selection.json"
    if receipt_path.is_file():
        receipt = _read(receipt_path)
        if receipt.get("passed") is True:
            for candidate in receipt["candidates"]:
                run_dir = Path(candidate["run_dir"])
                if candidate["label"] == receipt["selected"]:
                    compact_completed_run(run_dir)
                else:
                    _retire_candidate(candidate, reason="bootstrap-origin-not-selected")
            return Path(receipt["selected_run_dir"]), receipt
    candidates = []
    for index, seed in enumerate(ORIGIN_SEEDS, start=1):
        label = f"bootstrap-origin-{index}"
        run_dir = directory / label
        training = _train(
            run_dir=run_dir,
            origin=label,
            seed=seed,
            learning_rate=learning_rate,
            broad=broad,
            teacher=teacher,
            planned_stop=None,
        )
        evaluation = _evaluate(
            run_dir=run_dir,
            origin=label,
            validation=validation,
            games=OPEN_GAMES,
            tolerate_failure=True,
        )
        candidates.append(
            _candidate(
                label=label,
                rate=learning_rate,
                training=training,
                evaluation=evaluation,
            )
        )
        candidates[-1]["run_dir"] = str(run_dir.resolve())
    selected = _select(candidates, "bootstrap origin")
    receipt = {
        "schema_id": "cascadia-v3-bootstrap-origin-selection-v1",
        "passed": True,
        "selection_rule": "minimum-quantized-validation-loss-among-open-nonregressing",
        "learning_rate": learning_rate,
        "candidates": candidates,
        "selected": selected["label"],
        "selected_run_dir": selected["run_dir"],
    }
    _write_atomic(receipt_path, receipt)
    for candidate in candidates:
        run_dir = Path(candidate["run_dir"])
        if candidate["label"] == selected["label"]:
            compact_completed_run(run_dir)
        else:
            _retire_candidate(candidate, reason="bootstrap-origin-not-selected")
    return Path(selected["run_dir"]), receipt


def _freeze(run_dir: Path, calibration: dict[str, Any], origins: dict[str, Any]) -> Path:
    directory = ROOT / "phase2/bootstrap/training/winner-parity"
    report = directory / "parity-report.json"
    if not report.is_file():
        _run(
            [
                str(PYTHON),
                "-m",
                "cascadia_v3_mlx.verify",
                "--feature-manifest",
                str(FEATURE_MANIFEST),
                "--output-dir",
                str(directory),
                "--rust-binary",
                str(GAME_BINARY),
                "--groups",
                "100",
                "--candidates-per-group",
                "64",
                "--checkpoint-run-dir",
                str(run_dir),
            ],
            directory / "parity.log",
        )
    parity = _read(report)
    if (
        parity.get("rust_mlx_quantized_bit_identical") is not True
        or parity.get("rust_scalar_neon_bit_identical") is not True
        or float(parity.get("float_quantized_top32_agreement", 0.0)) < 0.999
    ):
        raise BootstrapTrainingError("bootstrap winner failed quantized parity")
    source = directory / "model"
    destination = ROOT / "models/bootstrap-champion"
    if not destination.is_dir():
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        shutil.copytree(source, temporary)
        os.replace(temporary, destination)
    model = destination / "model.json"
    weights = destination / "weights.v3q"
    evidence = ROOT / "phase2/bootstrap/training/bootstrap-champion.json"
    value = {
        "schema_id": "cascadia-v3-frozen-bootstrap-champion-v1",
        "passed": True,
        "run_dir": str(run_dir.resolve()),
        "calibration_selection": {
            "path": str(ROOT / "phase2/bootstrap/training/calibration/selection.json"),
            "blake3": _digest(ROOT / "phase2/bootstrap/training/calibration/selection.json"),
        },
        "origin_selection": {
            "path": str(ROOT / "phase2/bootstrap/training/origins/selection.json"),
            "blake3": _digest(ROOT / "phase2/bootstrap/training/origins/selection.json"),
        },
        "selected_learning_rate": calibration["selected_learning_rate"],
        "selected_origin": origins["selected"],
        "model_manifest": str(model.resolve()),
        "model_manifest_blake3": _digest(model),
        "weights": str(weights.resolve()),
        "weights_blake3": _digest(weights),
        "parity_report": str(report.resolve()),
        "parity_report_blake3": _digest(report),
        "protected_seed_values_opened": False,
    }
    _write_atomic(evidence, value)
    return evidence


def _publish_final_worker(bootstrap_evidence: Path) -> tuple[str, Path]:
    source_identity = _workspace_source_identity(REPOSITORY)
    publication = _read(FINAL_IMAGE_RECEIPT) if FINAL_IMAGE_RECEIPT.is_file() else None
    if (
        publication is None
        or publication.get("schema_id") != "cascadia.cluster.image-publication.v1"
        or publication.get("source_identity") != source_identity
    ):
        _run(
            [
                str(PYTHON),
                "tools/cluster_build_push.py",
                "--context",
                str(REPOSITORY),
                "--dockerfile",
                str(REPOSITORY / "Dockerfile.v3"),
                "--name",
                "v3-worker",
                "--tag",
                "stage2-final-v1",
                "--receipt",
                str(FINAL_IMAGE_RECEIPT),
            ],
            ROOT / "smoke/image-publication-stage2-final-v1.log",
        )
        publication = _read(FINAL_IMAGE_RECEIPT)
    image = publication.get("image_digest")
    if not isinstance(image, str) or "@sha256:" not in image:
        raise BootstrapTrainingError("final worker publication has no immutable image digest")
    health = _read(FINAL_IMAGE_HEALTH) if FINAL_IMAGE_HEALTH.is_file() else None
    if health is None or health.get("image_digest") != image or health.get("passed") is not True:
        command = [
            "/opt/homebrew/bin/docker",
            "--host",
            DEFAULT_DOCKER_HOST,
            "run",
            "--rm",
            "--entrypoint",
            "/usr/local/bin/v3-campaign-worker",
            image,
            "health",
            "--output",
            "/tmp/health.json",
        ]
        try:
            _run(command, ROOT / "smoke/image-health-stage2-final-v1.log")
        except subprocess.CalledProcessError as error:
            raise BootstrapTrainingError("final worker image health check failed") from error
        health = {
            "schema_id": "cascadia-v3-final-worker-health-v1",
            "passed": True,
            "image_digest": image,
            "publication_receipt": str(FINAL_IMAGE_RECEIPT.resolve()),
            "publication_blake3": _digest(FINAL_IMAGE_RECEIPT),
            "command": command,
        }
        _write_atomic(FINAL_IMAGE_HEALTH, health)
    handoff = {
        "schema_id": "cascadia-v3-stage2-worker-handoff-v1",
        "passed": True,
        "bootstrap_evidence": str(bootstrap_evidence.resolve()),
        "bootstrap_evidence_blake3": _digest(bootstrap_evidence),
        "publication_receipt": str(FINAL_IMAGE_RECEIPT.resolve()),
        "publication_receipt_blake3": _digest(FINAL_IMAGE_RECEIPT),
        "health_receipt": str(FINAL_IMAGE_HEALTH.resolve()),
        "health_receipt_blake3": _digest(FINAL_IMAGE_HEALTH),
        "image_digest": image,
        "source_identity": source_identity,
        "protected_seed_values_opened": False,
    }
    _write_atomic(FINAL_HANDOFF, handoff)
    return image, FINAL_HANDOFF


def _advance(evidence: Path) -> None:
    state = _read(CAMPAIGN_STATE)
    if state.get("phase") == "cycle-01-collecting":
        return
    if state.get("phase") != "bootstrap_training":
        raise BootstrapTrainingError(f"unexpected state before cycle 1: {state.get('phase')}")
    _run(
        [
            str(PYTHON),
            "tools/v3_campaign.py",
            "advance-phase2",
            "--to",
            "cycle-01-collecting",
            "--evidence",
            str(evidence),
            "--evidence-sha256",
            _digest(evidence, "sha256"),
        ],
        ROOT / "phase2/bootstrap/training/advance.log",
    )


def run(poll_seconds: int) -> None:
    for path in (PYTHON, FEATURE_MANIFEST, SCHEDULE, BATCH_STREAM, GAME_BINARY):
        if not path.is_file():
            raise BootstrapTrainingError(f"bootstrap training dependency is missing: {path}")
    broad, teacher, validation = _wait_for_inputs(poll_seconds)
    learning_rate, calibration = _calibrate(broad, teacher, validation)
    winner, origins = _origins(
        learning_rate=learning_rate,
        broad=broad,
        teacher=teacher,
        validation=validation,
    )
    evidence = _freeze(winner, calibration, origins)
    image, handoff = _publish_final_worker(evidence)
    _advance(handoff)
    _run(
        [
            str(PYTHON),
            "tools/v3_expert_campaign.py",
            "--image",
            image,
            "--poll-seconds",
            str(poll_seconds),
        ],
        ROOT / "phase2/expert-campaign.log",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        raise SystemExit("poll-seconds must be positive")
    lock = _acquire_controller_lock(CONTROLLER_LOCK)
    if lock is None:
        print(json.dumps({"status": "already-running", "lock": str(CONTROLLER_LOCK)}))
        return
    try:
        run(args.poll_seconds)
    except (
        BootstrapTrainingError,
        OSError,
        ValueError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
    ) as error:
        failure = ROOT / "phase2/bootstrap/training/pipeline-failure.json"
        _write_atomic(
            failure,
            {
                "schema_id": "cascadia-v3-bootstrap-training-failure-v1",
                "error": str(error),
                "failed_unix_ms": time.time_ns() // 1_000_000,
            },
        )
        raise SystemExit(str(error)) from error
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    main()
