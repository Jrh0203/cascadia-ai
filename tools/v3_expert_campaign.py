#!/usr/bin/env python3
"""Restart-safe controller for all ten Cascadia V3 expert-iteration cycles."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path("/Users/johnherrick/cascadia-bench/v3-nnue")
REPOSITORY = Path("/Users/johnherrick/cascadia")
PYTHON = REPOSITORY / ".venv/bin/python"
STATE = ROOT / "control/campaign-state.json"
CONTROLLER_LOCK = ROOT / "control/expert-campaign-controller.lock"


class ExpertCampaignError(ValueError):
    """The durable campaign state cannot be reconciled with frozen models."""


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=REPOSITORY, check=True)


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
            {"pid": os.getpid(), "started_unix_ms": time.time_ns() // 1_000_000},
            sort_keys=True,
        )
        + "\n"
    )
    handle.flush()
    return handle


def _bootstrap() -> tuple[Path, Path]:
    receipt = _read(ROOT / "phase2/bootstrap/training/bootstrap-champion.json")
    if receipt.get("passed") is not True:
        raise ExpertCampaignError("bootstrap champion receipt is not passing")
    return ROOT / "models/bootstrap-champion", Path(receipt["run_dir"])


def _reconcile() -> tuple[Path, Path, list[Path]]:
    model, run_dir = _bootstrap()
    history = [model]
    for cycle in range(1, 11):
        candidate = ROOT / f"models/cycle-{cycle:02d}-candidate"
        if candidate.is_dir():
            history.append(candidate)
        champion = ROOT / f"phase2/cycles/cycle-{cycle:02d}/promotion/champion.json"
        if champion.is_file():
            value = _read(champion)
            if value.get("passed") is not True:
                raise ExpertCampaignError(f"cycle {cycle} champion receipt is not passing")
            model = Path(value["champion_model_dir"])
            run_dir = Path(value["champion_run_dir"])
    return model, run_dir, history


def _cycle_from_phase(phase: str) -> int | None:
    if not phase.startswith("cycle-"):
        return None
    try:
        cycle = int(phase.split("-")[1])
    except (IndexError, ValueError) as error:
        raise ExpertCampaignError(f"malformed expert phase: {phase}") from error
    return cycle


def run(image: str, poll_seconds: int) -> None:
    while True:
        state = _read(STATE)
        phase = str(state.get("phase"))
        if phase in {
            "bootstrap_collecting",
            "bootstrap_labeling",
            "bootstrap_training",
        }:
            time.sleep(poll_seconds)
            continue
        if phase == "complete":
            return
        if phase in {"final_protected_comparison", "final_all_v3_evaluation"}:
            champion_model, _, _ = _reconcile()
            _run(
                [
                    str(PYTHON),
                    "tools/v3_final_pipeline.py",
                    "--image",
                    image,
                    "--champion-model",
                    str(champion_model),
                ]
            )
            continue
        cycle = _cycle_from_phase(phase)
        if cycle is None or not 1 <= cycle <= 10:
            raise ExpertCampaignError(f"campaign is not in an expert-cycle phase: {phase}")
        champion_model, champion_run, history = _reconcile()
        if phase.endswith("collecting") or phase.endswith("labeling"):
            command = [
                str(PYTHON),
                "tools/v3_cycle_data_pipeline.py",
                "--cycle",
                str(cycle),
                "--image",
                image,
                "--newest-model",
                str(champion_model),
            ]
            if cycle > 1:
                for model in history:
                    command.extend(("--prior-model", str(model)))
            _run(command)
            phase = str(_read(STATE)["phase"])
        if phase.endswith("training"):
            _run(
                [
                    str(PYTHON),
                    "tools/v3_cycle_train_pipeline.py",
                    "--cycle",
                    str(cycle),
                    "--parent-run-dir",
                    str(champion_run),
                    "--parent-model",
                    str(champion_model),
                    "--image",
                    image,
                ]
            )
            phase = str(_read(STATE)["phase"])
        if phase.endswith("promotion"):
            candidate = _read(
                ROOT / f"phase2/cycles/cycle-{cycle:02d}/training/candidate.json"
            )
            _run(
                [
                    str(PYTHON),
                    "tools/v3_cycle_promotion.py",
                    "--cycle",
                    str(cycle),
                    "--image",
                    image,
                    "--treatment-model",
                    candidate["model_dir"],
                    "--treatment-run-dir",
                    candidate["run_dir"],
                    "--control-model",
                    str(champion_model),
                    "--control-run-dir",
                    str(champion_run),
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        raise SystemExit("poll-seconds must be positive")
    lock = _acquire_controller_lock(CONTROLLER_LOCK)
    if lock is None:
        print(json.dumps({"status": "already-running", "lock": str(CONTROLLER_LOCK)}))
        return
    try:
        run(args.image, args.poll_seconds)
    except (
        ExpertCampaignError,
        OSError,
        ValueError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
    ) as error:
        raise SystemExit(str(error)) from error
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    main()
