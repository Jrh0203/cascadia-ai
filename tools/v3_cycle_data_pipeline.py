#!/usr/bin/env python3
"""Carry one authorized V3 expert cycle from collection through exact labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from v3_colima_reclaim import reclaim_completed_increment, reclaim_remote_workers

ROOT = Path("/Users/johnherrick/cascadia-bench/v3-nnue")
REPOSITORY = Path("/Users/johnherrick/cascadia")
PYTHON = REPOSITORY / ".venv/bin/python"
CAMPAIGN_STATE = ROOT / "control/campaign-state.json"
V1_WEIGHTS = ROOT / "phase2/inputs/v1/qualified-v1.bin"
ROOT_SELECT = REPOSITORY / "target/release/teacher_root_select"
ROOT_SPLIT = REPOSITORY / "target/release/teacher_root_split"


class CycleDataError(ValueError):
    """An expert-cycle collection or labeling gate failed."""


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=REPOSITORY, check=True)


def _advance(destination: str, evidence: Path) -> None:
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
        ]
    )


def _collection(
    cycle: int,
    image: str,
    newest_model: Path,
    prior_models: list[Path],
) -> tuple[Path, Path]:
    directory = ROOT / f"phase2/cycles/cycle-{cycle:02d}/collection"
    completion = directory / "completion.json"
    request = f"v3-cycle-{cycle:02d}-collect"
    if not completion.is_file() or _read(completion).get("passed") is not True:
        command = [
            str(PYTHON),
            "tools/v3_cycle_collection.py",
            "--image",
            image,
            "--campaign-state",
            str(CAMPAIGN_STATE),
            "--v1-weights",
            str(V1_WEIGHTS),
            "--newest-model",
            str(newest_model),
            "--state-directory",
            str(ROOT / "phase2/control/cluster-client"),
            "--artifact-directory",
            str(directory / "accepted"),
            "--request-id",
            request,
            "--progress",
            str(directory / "progress.json"),
            "--completion",
            str(completion),
        ]
        for model in prior_models:
            command.extend(("--prior-model", str(model)))
        try:
            _run(command)
        except subprocess.CalledProcessError:
            repair = [
                str(PYTHON),
                "tools/v3_cycle_collection_repair.py",
                "--cycle",
                str(cycle),
                "--image",
                image,
                "--campaign-state",
                str(CAMPAIGN_STATE),
                "--v1-weights",
                str(V1_WEIGHTS),
                "--newest-model",
                str(newest_model),
                "--state-directory",
                str(ROOT / "phase2/control/cluster-client"),
                "--original-artifact-directory",
                str(directory / "accepted"),
                "--original-request-id",
                request,
                "--repair-artifact-directory",
                str(directory / "repair"),
                "--reconciled-root",
                str(directory / "reconciled"),
                "--completion",
                str(completion),
                "--python",
                str(PYTHON),
                "--collection-program",
                str(REPOSITORY / "tools/v3_cycle_collection.py"),
            ]
            for model in prior_models:
                repair.extend(("--prior-model", str(model)))
            _run(repair)
    accepted = Path(str(_read(completion).get("artifact_root", "")))
    if not accepted.is_dir():
        raise CycleDataError("collection completion artifact root is absent")
    return completion, accepted


def _verification(cycle: int, image: str, shards: list[Path]) -> Path:
    directory = ROOT / f"phase2/cycles/cycle-{cycle:02d}/verification"
    completion = directory / "completion.json"
    if completion.is_file() and _read(completion).get("passed") is True:
        return completion
    command = [
        str(PYTHON),
        "tools/v3_phase2_pipeline.py",
        "--image",
        image,
        "--state-directory",
        str(ROOT / "phase2/control/cluster-client"),
        "--artifact-directory",
        str(directory / "accepted"),
        "--request-id",
        f"v3-cycle-{cycle:02d}-verify",
        "--progress",
        str(directory / "progress.json"),
        "--completion",
        str(completion),
        "verify-collection",
        "--entries-per-game",
        "20",
    ]
    for shard in shards:
        command.extend(("--shard", str(shard)))
    _run(command)
    return completion


def _corpus(
    cycle: int,
    collection: Path,
    verification: Path,
    accepted: Path,
) -> Path:
    output = ROOT / f"phase2/cycles/cycle-{cycle:02d}/corpus.json"
    if output.is_file() and _read(output).get("passed") is True:
        return output
    _run(
        [
            str(PYTHON),
            "tools/v3_cycle_corpus.py",
            "--cycle",
            str(cycle),
            "--collection",
            str(collection),
            "--verification",
            str(verification),
            "--accepted-root",
            str(accepted),
            "--campaign-state",
            str(CAMPAIGN_STATE),
            "--output",
            str(output),
        ]
    )
    return output


def _roots(cycle: int, corpus: Path) -> list[Path]:
    directory = ROOT / f"phase2/cycles/cycle-{cycle:02d}/roots"
    teacher = directory / "teacher.v3r"
    validation = directory / "validation-empty.v3r"
    receipt = directory / "selection.json"
    if not receipt.is_file() or _read(receipt).get("passed") is not True:
        command = [
            str(ROOT_SELECT),
            "--teacher-output",
            str(teacher),
            "--validation-output",
            str(validation),
            "--receipt",
            str(receipt),
            "--teacher-roots",
            "2500",
            "--validation-roots",
            "0",
            "--oversample-permyriad",
            "300",
        ]
        for item in _read(corpus)["files"]:
            command.extend(("--input", item["path"]))
        _run(command)
    split = directory / "split.json"
    shards = directory / "shards"
    if not split.is_file():
        _run(
            [
                str(ROOT_SPLIT),
                "--input",
                str(teacher),
                "--output-dir",
                str(shards),
                "--prefix",
                "teacher",
                "--roots-per-shard",
                "100",
                "--receipt",
                str(split),
            ]
        )
    result = sorted(shards.glob("*.v3r"))
    if len(result) != 25:
        raise CycleDataError(f"cycle {cycle} produced {len(result)} root shards, expected 25")
    return result


def _label(cycle: int, image: str, roots: list[Path]) -> tuple[Path, Path]:
    directory = ROOT / f"phase2/cycles/cycle-{cycle:02d}/labeling"
    completion = directory / "completion.json"
    request = f"v3-cycle-{cycle:02d}-label"
    if not completion.is_file() or _read(completion).get("passed") is not True:
        command = [
            str(PYTHON),
            "tools/v3_phase2_pipeline.py",
            "--image",
            image,
            "--state-directory",
            str(ROOT / "phase2/control/cluster-client"),
            "--artifact-directory",
            str(directory / "accepted"),
            "--request-id",
            request,
            "--progress",
            str(directory / "progress.json"),
            "--completion",
            str(completion),
            "label-roots",
            "--campaign-state",
            str(CAMPAIGN_STATE),
            "--v1-weights",
            str(V1_WEIGHTS),
            "--cycle",
            str(cycle),
        ]
        for root in roots:
            command.extend(("--root-shard", str(root)))
        _run(command)
    return completion, directory / "accepted" / request


def _label_evidence(
    cycle: int,
    image: str,
    completion: Path,
    accepted: Path,
) -> Path:
    output = ROOT / f"phase2/cycles/cycle-{cycle:02d}/teacher-label-corpus.json"
    if output.is_file() and _read(output).get("passed") is True:
        return output
    _run(
        [
            str(PYTHON),
            "tools/v3_teacher_labels.py",
            "--completion",
            str(completion),
            "--accepted-root",
            str(accepted),
            "--root-directory",
            str(ROOT / f"phase2/cycles/cycle-{cycle:02d}/roots/shards"),
            "--campaign-state",
            str(CAMPAIGN_STATE),
            "--image",
            image,
            "--cycle",
            str(cycle),
            "--output",
            str(output),
        ]
    )
    return output


def run(cycle: int, image: str, newest_model: Path, prior_models: list[Path]) -> None:
    state = _read(CAMPAIGN_STATE)
    collecting = f"cycle-{cycle:02d}-collecting"
    labeling = f"cycle-{cycle:02d}-labeling"
    training = f"cycle-{cycle:02d}-training"
    if state.get("phase") not in {collecting, labeling, training}:
        raise CycleDataError(
            f"cycle data expected {collecting}, {labeling}, or {training}; "
            f"observed {state.get('phase')}"
        )
    if state.get("phase") == collecting:
        collection, accepted = _collection(cycle, image, newest_model, prior_models)
        shards = sorted(accepted.glob("*/*.v3g"))
        if len(shards) != 100:
            raise CycleDataError(f"cycle {cycle} imported {len(shards)} collection shards")
        reclaim_completed_increment(
            collection,
            collection.with_name("storage-reclaim.json"),
        )
        reclaim_remote_workers(
            collection,
            collection.with_name("remote-worker-reclaim.json"),
        )
        verification = _verification(cycle, image, shards)
        corpus = _corpus(cycle, collection, verification, accepted)
        _advance(labeling, corpus)
    if _read(CAMPAIGN_STATE).get("phase") == training:
        return
    corpus = ROOT / f"phase2/cycles/cycle-{cycle:02d}/corpus.json"
    roots = _roots(cycle, corpus)
    label_completion, label_accepted = _label(cycle, image, roots)
    reclaim_completed_increment(
        label_completion,
        label_completion.with_name("storage-reclaim.json"),
    )
    reclaim_remote_workers(
        label_completion,
        label_completion.with_name("remote-worker-reclaim.json"),
    )
    evidence = _label_evidence(cycle, image, label_completion, label_accepted)
    _advance(training, evidence)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--newest-model", type=Path, required=True)
    parser.add_argument("--prior-model", type=Path, action="append", default=[])
    args = parser.parse_args()
    if not 1 <= args.cycle <= 10:
        raise SystemExit("cycle is outside 1..=10")
    try:
        run(args.cycle, args.image, args.newest_model, args.prior_model)
    except (
        CycleDataError,
        OSError,
        ValueError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
    ) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
