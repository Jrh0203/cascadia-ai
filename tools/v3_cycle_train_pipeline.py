#!/usr/bin/env python3
"""Train, select, verify, and freeze the candidate for one V3 expert cycle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

from v3_bootstrap_train_pipeline import _candidate, _evaluate, _select
from v3_checkpoint_lifecycle import compact_completed_run, retire_completed_run
from v3_model_stage import _digest

ROOT = Path("/Users/johnherrick/cascadia-bench/v3-nnue")
REPOSITORY = Path("/Users/johnherrick/cascadia")
PYTHON = REPOSITORY / ".venv/bin/python"
STATE = ROOT / "control/campaign-state.json"
FEATURE = ROOT / "models/feature-schema.json"
BATCH_STREAM = REPOSITORY / "target/release/v3-batch-stream"
GAME_BINARY = REPOSITORY / "target/release/v3-engineering-smoke"
VALIDATION_CACHE_MANIFEST = ROOT / "phase2/bootstrap/training/validation-cache/manifest.json"


class CycleTrainingPipelineError(ValueError):
    """The cycle training candidate set or freeze gate is invalid."""


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _write_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


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


def _start_parent_benchmark(
    cycle: int, image: str, parent_model: Path
) -> subprocess.Popen:
    directory = ROOT / f"phase2/cycles/cycle-{cycle:02d}/training/parent-benchmark"
    completion = directory / "completion.json"
    if completion.is_file() and _read(completion).get("passed") is True:
        return subprocess.Popen(["true"])
    request_id = f"v3-cycle-{cycle:02d}-training-parent-benchmark-v1"
    command = [
        str(PYTHON),
        "tools/v3_training_benchmark.py",
        "--cycle",
        str(cycle),
        "--image",
        image,
        "--parent-model",
        str(parent_model),
        "--state-directory",
        str(directory / "cluster-state"),
        "--artifact-directory",
        str(directory / "artifacts"),
        "--request-id",
        request_id,
        "--progress",
        str(directory / "progress.json"),
        "--completion",
        str(completion),
    ]
    log = directory / "benchmark.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    output = log.open("ab")
    output.write(("\n$ " + " ".join(command) + "\n").encode())
    output.flush()
    process = subprocess.Popen(
        command,
        cwd=REPOSITORY,
        stdout=output,
        stderr=subprocess.STDOUT,
    )
    output.close()
    return process


def _files(manifest: Path, split: str | None = None) -> list[Path]:
    value = _read(manifest)
    if value.get("passed") is not True:
        raise CycleTrainingPipelineError(f"dataset manifest is not passing: {manifest}")
    return [
        Path(item["path"])
        for item in value["files"]
        if split is None or item.get("split") == split
    ]


def _inputs(cycle: int) -> dict[str, list[Path]]:
    current = ROOT / f"phase2/cycles/cycle-{cycle:02d}"
    recent = []
    for prior in range(max(1, cycle - 3), cycle):
        recent.extend(_files(ROOT / f"phase2/cycles/cycle-{prior:02d}/corpus.json"))
    return {
        "current_broad": _files(current / "corpus.json"),
        "current_teacher": _files(current / "teacher-label-corpus.json", "teacher"),
        "recent": recent,
        "older_broad": _files(ROOT / "phase2/bootstrap/corpus.json"),
        "older_teacher": _files(
            ROOT / "phase2/bootstrap/teacher-label-corpus.json", "teacher"
        ),
    }


def _train_origin(
    *,
    cycle: int,
    origin: int,
    parent: Path,
    inputs: dict[str, list[Path]],
) -> tuple[Path, dict[str, Any]]:
    run_dir = ROOT / f"phase2/cycles/cycle-{cycle:02d}/training/origin-{origin}"
    report = run_dir / "training-report.json"
    if report.is_file() and _read(report).get("passed") is True:
        return run_dir, _read(report)
    command = [
        str(PYTHON),
        "-m",
        "cascadia_v3_mlx.cycle_train",
        "--campaign-root",
        str(ROOT),
        "--campaign-state",
        str(STATE),
        "--feature-manifest",
        str(FEATURE),
        "--batch-stream-binary",
        str(BATCH_STREAM),
        "--parent-run-dir",
        str(parent),
        "--run-dir",
        str(run_dir),
        "--cycle",
        str(cycle),
        "--origin",
        f"cycle-{cycle:02d}-origin-{origin}",
        "--seed",
        str(90_000 + cycle * 10 + origin),
    ]
    for source, paths in inputs.items():
        option = "--" + source.replace("_", "-")
        for path in paths:
            command.extend((option, str(path)))
    if (run_dir / "latest.json").is_file():
        command.append("--resume")
    _run(command, run_dir / "training.log")
    value = _read(report)
    if value.get("passed") is not True or value.get("examples_seen") != 1_200_000:
        raise CycleTrainingPipelineError(f"cycle origin did not complete: {run_dir}")
    return run_dir, value


def _cached_validation(label_manifest: Path, cache_manifest: Path) -> list[Path]:
    labels = _read(label_manifest)
    if labels.get("passed") is not True:
        raise CycleTrainingPipelineError("bootstrap validation manifest is not passing")
    validation_records = [
        item for item in labels.get("files", []) if item.get("split") == "validation"
    ]
    cache = _read(cache_manifest)
    shards = cache.get("shards")
    totals = cache.get("totals")
    if (
        cache.get("schema_id") != "cascadia-v3-validation-cache-v1"
        or cache.get("passed") is not True
        or not isinstance(shards, list)
        or len(shards) != len(validation_records)
        or totals
        != {
            "shards": len(validation_records),
            "roots": sum(int(item["roots"]) for item in shards),
            "rows": sum(int(item["candidate_estimates"]) for item in validation_records),
            "bytes": sum(int(item["bytes"]) for item in shards),
        }
    ):
        raise CycleTrainingPipelineError("validation cache manifest is invalid")
    expected = {
        str(Path(item["path"]).resolve()): (item["blake3"], int(item["bytes"]))
        for item in validation_records
    }
    observed = {
        str(Path(item["source_path"]).resolve()): (
            item["source_blake3"],
            int(item["source_bytes"]),
        )
        for item in shards
    }
    if observed != expected:
        raise CycleTrainingPipelineError("validation cache source identity differs")
    cached = [Path(item["path"]) for item in shards]
    if any(
        not path.is_file()
        or path.stat().st_size != int(item["bytes"])
        or _digest(path) != item["blake3"]
        for path, item in zip(cached, shards, strict=True)
    ):
        raise CycleTrainingPipelineError("validation cache shard integrity failed")
    return cached


def _validation() -> list[Path]:
    validation = _cached_validation(
        ROOT / "phase2/bootstrap/teacher-label-corpus.json",
        VALIDATION_CACHE_MANIFEST,
    )
    if len(validation) != 20:
        raise CycleTrainingPipelineError("validation cache does not contain 20 shards")
    return validation


def _select_origins(
    cycle: int,
    parent: Path,
    inputs: dict[str, list[Path]],
) -> tuple[Path, dict[str, Any]]:
    directory = ROOT / f"phase2/cycles/cycle-{cycle:02d}/training"
    receipt_path = directory / "selection.json"
    if receipt_path.is_file() and _read(receipt_path).get("passed") is True:
        receipt = _read(receipt_path)
        for candidate in receipt["candidates"]:
            run_dir = Path(candidate["run_dir"])
            if candidate["label"] == receipt["selected"]:
                compact_completed_run(run_dir)
            else:
                retire_completed_run(run_dir, reason="expert-cycle-origin-not-selected")
        return Path(receipt["selected_run_dir"]), receipt
    candidates = []
    validation = _validation()
    for origin in (1, 2):
        run_dir, training = _train_origin(
            cycle=cycle,
            origin=origin,
            parent=parent,
            inputs=inputs,
        )
        label = f"cycle-{cycle:02d}-origin-{origin}"
        evaluation = _evaluate(
            run_dir=run_dir,
            origin=label,
            validation=validation,
            games=32,
            cycle=cycle,
        )
        candidate = _candidate(
            label=label,
            rate=3e-5,
            training=training,
            evaluation=evaluation,
        )
        candidate["run_dir"] = str(run_dir.resolve())
        candidates.append(candidate)
    selected = _select(candidates, f"cycle {cycle} origin")
    receipt = {
        "schema_id": "cascadia-v3-expert-cycle-origin-selection-v1",
        "passed": True,
        "cycle": cycle,
        "parent_run_dir": str(parent.resolve()),
        "selection_rule": "minimum-quantized-validation-loss-among-open-nonregressing",
        "candidates": candidates,
        "selected": selected["label"],
        "selected_run_dir": selected["run_dir"],
        "protected_seed_values_opened": False,
    }
    _write_atomic(receipt_path, receipt)
    for candidate in candidates:
        run_dir = Path(candidate["run_dir"])
        if candidate["label"] == selected["label"]:
            compact_completed_run(run_dir)
        else:
            retire_completed_run(run_dir, reason="expert-cycle-origin-not-selected")
    return Path(selected["run_dir"]), receipt


def _verify_and_freeze(cycle: int, run_dir: Path, selection: dict[str, Any]) -> Path:
    directory = ROOT / f"phase2/cycles/cycle-{cycle:02d}/candidate-parity"
    parity_path = directory / "parity-report.json"
    if not parity_path.is_file():
        _run(
            [
                str(PYTHON),
                "-m",
                "cascadia_v3_mlx.verify",
                "--feature-manifest",
                str(FEATURE),
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
    parity = _read(parity_path)
    if (
        parity.get("rust_mlx_quantized_bit_identical") is not True
        or parity.get("rust_scalar_neon_bit_identical") is not True
        or float(parity.get("float_quantized_top32_agreement", 0.0)) < 0.999
    ):
        raise CycleTrainingPipelineError("cycle candidate failed cross-backend parity")
    source = directory / "model"
    destination = ROOT / f"models/cycle-{cycle:02d}-candidate"
    if not destination.is_dir():
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        shutil.copytree(source, temporary)
        os.replace(temporary, destination)
    evidence = ROOT / f"phase2/cycles/cycle-{cycle:02d}/training/candidate.json"
    value = {
        "schema_id": "cascadia-v3-frozen-cycle-candidate-v1",
        "passed": True,
        "cycle": cycle,
        "selected_origin": selection["selected"],
        "run_dir": str(run_dir.resolve()),
        "model_dir": str(destination.resolve()),
        "model_manifest_sha256": _sha256(destination / "model.json"),
        "weights_sha256": _sha256(destination / "weights.v3q"),
        "parity_report": str(parity_path.resolve()),
        "parity_report_sha256": _sha256(parity_path),
        "protected_seed_values_opened": False,
    }
    _write_atomic(evidence, value)
    return evidence


def _advance(cycle: int, evidence: Path) -> None:
    state = _read(STATE)
    destination = f"cycle-{cycle:02d}-promotion"
    if state.get("phase") == destination:
        return
    if state.get("phase") != f"cycle-{cycle:02d}-training":
        raise CycleTrainingPipelineError("campaign is not at the cycle training gate")
    _run(
        [
            str(PYTHON),
            "tools/v3_campaign.py",
            "advance-phase2",
            "--to",
            destination,
            "--evidence",
            str(evidence),
            "--evidence-sha256",
            _sha256(evidence),
        ],
        evidence.with_suffix(".advance.log"),
    )


def run(cycle: int, parent: Path, parent_model: Path, image: str) -> None:
    state = _read(STATE)
    if state.get("phase") not in {
        f"cycle-{cycle:02d}-training",
        f"cycle-{cycle:02d}-promotion",
    }:
        raise CycleTrainingPipelineError("campaign is not at this cycle's training gate")
    benchmark = _start_parent_benchmark(cycle, image, parent_model)
    inputs = _inputs(cycle)
    selected, receipt = _select_origins(cycle, parent, inputs)
    if benchmark.wait() != 0:
        raise CycleTrainingPipelineError("John2/John3 parent benchmark failed")
    evidence = _verify_and_freeze(cycle, selected, receipt)
    _advance(cycle, evidence)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--parent-run-dir", type=Path, required=True)
    parser.add_argument("--parent-model", type=Path, required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    if not 1 <= args.cycle <= 10:
        raise SystemExit("cycle is outside 1..=10")
    try:
        run(args.cycle, args.parent_run_dir, args.parent_model, args.image)
    except (
        CycleTrainingPipelineError,
        OSError,
        ValueError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
    ) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
