#!/usr/bin/env python3
"""Idempotently carry the authorized V3 bootstrap from collection to training."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

ROOT = Path("/Users/johnherrick/cascadia-bench/v3-nnue")
REPOSITORY = Path("/Users/johnherrick/cascadia")
READINESS = "b15e44af519060936416af23e9af1795dd37149309a3f65bc86941d5399b3ebe"
COLLECTION_REQUEST = "v3-bootstrap-collection-b15e44af-0aebd736-v1"
LABEL_IMAGE = (
    "100.110.109.6:5000/cascadia/v3-worker@"
    "sha256:ab2d49849931f7a535c5b313f5f1a5fc36663a85e6b6eb3252b67d70d731af3d"
)
VERIFY_REQUEST = "v3-bootstrap-verify-b15e44af-ab2d4984-v1"
LABEL_REQUEST = "v3-bootstrap-label-b15e44af-ab2d4984-v1"


class BootstrapError(ValueError):
    """A bootstrap barrier failed and cannot be crossed."""


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=REPOSITORY, check=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _advance(destination: str, evidence: Path) -> None:
    _run(
        [
            str(REPOSITORY / ".venv/bin/python"),
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


def _wait_for_collection(poll_seconds: int) -> Path:
    completion = ROOT / "phase2/bootstrap/collection/completion-receipt.json"
    while not completion.is_file():
        failure = ROOT / "phase2/bootstrap/collection/monitor-failure.json"
        if failure.is_file():
            raise BootstrapError(f"collection monitor failed: {failure.read_text()}")
        time.sleep(poll_seconds)
    value = _read(completion)
    if (
        value.get("schema_id")
        != "cascadia-v3-bootstrap-collection-completion-v1"
        or value.get("passed") is not True
        or value.get("work_items") != 250
        or value.get("games") != 500_000
        or value.get("failures") != []
        or value.get("request_id") != COLLECTION_REQUEST
        or value.get("approved_readiness_sha256") != READINESS
    ):
        raise BootstrapError("collection completion is not a passing 250-shard receipt")
    return completion


def _verify(shards: list[Path]) -> Path:
    directory = ROOT / "phase2/bootstrap/verification"
    completion = directory / "completion.json"
    if completion.is_file() and _read(completion).get("passed") is True:
        return completion
    command = [
        str(REPOSITORY / ".venv/bin/python"),
        "tools/v3_phase2_pipeline.py",
        "--image",
        LABEL_IMAGE,
        "--state-directory",
        str(ROOT / "phase2/control/cluster-client"),
        "--artifact-directory",
        str(directory / "accepted"),
        "--request-id",
        VERIFY_REQUEST,
        "--progress",
        str(directory / "progress.json"),
        "--completion",
        str(completion),
        "verify-collection",
    ]
    for path in shards:
        command.extend(("--shard", str(path)))
    _run(command)
    return completion


def _corpus(verification: Path) -> Path:
    output = ROOT / "phase2/bootstrap/corpus.json"
    if output.is_file() and _read(output).get("passed") is True:
        return output
    accepted = ROOT / "phase2/bootstrap/collection/accepted" / COLLECTION_REQUEST
    _run(
        [
            str(REPOSITORY / ".venv/bin/python"),
            "tools/v3_bootstrap_corpus.py",
            "--plan",
            str(ROOT / "control/bootstrap-collection-plan.json"),
            "--accepted-root",
            str(accepted),
            "--verification",
            str(verification),
            "--approved-readiness-sha256",
            READINESS,
            "--output",
            str(output),
        ]
    )
    return output


def _select_and_split(corpus: Path) -> list[Path]:
    directory = ROOT / "phase2/bootstrap/roots"
    selection = directory / "selection.json"
    teacher = directory / "teacher.v3r"
    validation = directory / "validation.v3r"
    if not selection.is_file() or _read(selection).get("passed") is not True:
        command = [
            str(REPOSITORY / "target/release/teacher_root_select"),
            "--teacher-output",
            str(teacher),
            "--validation-output",
            str(validation),
            "--receipt",
            str(selection),
            "--teacher-roots",
            "100000",
            "--validation-roots",
            "20000",
        ]
        for item in _read(corpus)["files"]:
            command.extend(("--input", item["path"]))
        _run(command)
    split_directory = directory / "shards"
    teacher_split = directory / "teacher-split.json"
    validation_split = directory / "validation-split.json"
    if not teacher_split.is_file():
        _run(
            [
                str(REPOSITORY / "target/release/teacher_root_split"),
                "--input",
                str(teacher),
                "--output-dir",
                str(split_directory),
                "--prefix",
                "teacher",
                "--roots-per-shard",
                "1000",
                "--receipt",
                str(teacher_split),
            ]
        )
    if not validation_split.is_file():
        _run(
            [
                str(REPOSITORY / "target/release/teacher_root_split"),
                "--input",
                str(validation),
                "--output-dir",
                str(split_directory),
                "--prefix",
                "validation",
                "--roots-per-shard",
                "1000",
                "--receipt",
                str(validation_split),
            ]
        )
    shards = sorted(split_directory.glob("*.v3r"))
    if len(shards) != 120:
        raise BootstrapError(f"expected 120 root shards, found {len(shards)}")
    return shards


def _label(shards: list[Path]) -> Path:
    directory = ROOT / "phase2/bootstrap/labeling"
    completion = directory / "completion.json"
    if completion.is_file() and _read(completion).get("passed") is True:
        return completion
    command = [
        str(REPOSITORY / ".venv/bin/python"),
        "tools/v3_phase2_pipeline.py",
        "--image",
        LABEL_IMAGE,
        "--state-directory",
        str(ROOT / "phase2/control/cluster-client"),
        "--artifact-directory",
        str(directory / "accepted"),
        "--request-id",
        LABEL_REQUEST,
        "--progress",
        str(directory / "progress.json"),
        "--completion",
        str(completion),
        "label-roots",
        "--campaign-state",
        str(ROOT / "control/campaign-state.json"),
        "--v1-weights",
        str(ROOT / "phase2/inputs/v1/qualified-v1.bin"),
    ]
    for path in shards:
        command.extend(("--root-shard", str(path)))
    _run(command)
    return completion


def _label_evidence(completion: Path) -> Path:
    output = ROOT / "phase2/bootstrap/teacher-label-corpus.json"
    if output.is_file() and _read(output).get("passed") is True:
        return output
    _run(
        [
            str(REPOSITORY / ".venv/bin/python"),
            "tools/v3_teacher_labels.py",
            "--completion",
            str(completion),
            "--accepted-root",
            str(ROOT / "phase2/bootstrap/labeling/accepted" / LABEL_REQUEST),
            "--root-directory",
            str(ROOT / "phase2/bootstrap/roots/shards"),
            "--campaign-state",
            str(ROOT / "control/campaign-state.json"),
            "--image",
            LABEL_IMAGE,
            "--output",
            str(output),
        ]
    )
    return output


def run(poll_seconds: int) -> None:
    _wait_for_collection(poll_seconds)
    accepted = ROOT / "phase2/bootstrap/collection/accepted" / COLLECTION_REQUEST
    shards = sorted(accepted.glob("*/*.v3g"))
    if len(shards) != 250:
        raise BootstrapError(f"expected 250 imported replay shards, found {len(shards)}")
    verification = _verify(shards)
    corpus = _corpus(verification)
    state = _read(ROOT / "control/campaign-state.json")
    if state["phase"] == "bootstrap_collecting":
        _advance("bootstrap_labeling", corpus)
    elif state["phase"] != "bootstrap_labeling":
        raise BootstrapError(f"unexpected phase before labeling: {state['phase']}")
    root_shards = _select_and_split(corpus)
    labels = _label(root_shards)
    evidence = _label_evidence(labels)
    state = _read(ROOT / "control/campaign-state.json")
    if state["phase"] == "bootstrap_labeling":
        _advance("bootstrap_training", evidence)
    elif state["phase"] != "bootstrap_training":
        raise BootstrapError(f"unexpected phase after labeling: {state['phase']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        raise SystemExit("poll seconds must be positive")
    try:
        run(args.poll_seconds)
    except (BootstrapError, OSError, json.JSONDecodeError, subprocess.CalledProcessError) as error:
        failure = ROOT / "phase2/bootstrap/pipeline-failure.json"
        failure.write_text(
            json.dumps(
                {
                    "schema_id": "cascadia-v3-bootstrap-pipeline-failure-v1",
                    "error": str(error),
                    "failed_unix_ms": time.time_ns() // 1_000_000,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
