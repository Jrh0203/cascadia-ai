"""Frozen john3 MLX training, validation evaluation, and retrieval for ADR 0078."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import time
from pathlib import Path
from typing import Any, Literal

import adr0078_cluster_runtime as rt

TrainingAction = Literal["complete", "monitor", "launch", "resume", "fail"]


def training_action(
    *,
    final_report: bool,
    process_running: bool,
    status: int | None,
    run_manifest: bool,
    latest_checkpoint: bool,
) -> TrainingAction:
    if final_report and process_running:
        return "monitor"
    if final_report and status in {None, 0}:
        return "complete"
    if status not in {None, 0}:
        return "fail"
    if process_running:
        return "monitor"
    if latest_checkpoint:
        return "resume"
    if run_manifest:
        return "launch"
    return "launch"


def remote_exists(path: Path) -> bool:
    return rt.remote_path_exists("john3", path)


def remote_status() -> int | None:
    result = rt.remote_shell(
        "john3",
        ["cat", str(rt.JOHN3_TRAIN_STATUS)],
        check=False,
        quiet=True,
    )
    if result.returncode != 0:
        return None
    return int(result.stdout.strip())


def remote_training_running() -> bool:
    result = rt.remote(
        "john3",
        (
            f"test -f {shlex.quote(str(rt.JOHN3_TRAIN_PID))} && "
            f"pid=$(cat {shlex.quote(str(rt.JOHN3_TRAIN_PID))}) && "
            'kill -0 "$pid" 2>/dev/null && '
            'command=$(/bin/ps -p "$pid" -o command=) && '
            f"printf '%s\\n' \"$command\" | /usr/bin/grep -F -- "
            f"{shlex.quote('cascadia_mlx.counterfactual_advantage_train')} "
            "> /dev/null"
        ),
        check=False,
        quiet=True,
    )
    return result.returncode == 0


def remote_training_progress_mtime() -> int:
    paths = (
        rt.JOHN3_TRAIN_LOG,
        rt.JOHN3_RUN_DIR / "run.json",
        rt.JOHN3_RUN_DIR / "latest.json",
        rt.JOHN3_RUN_DIR / "metrics.jsonl",
        rt.JOHN3_RUN_DIR / "final-report.json",
    )
    command = (
        "latest=0; "
        + " ".join(
            (
                f"if test -e {shlex.quote(str(path))}; then "
                f"value=$(/usr/bin/stat -f %m {shlex.quote(str(path))}); "
                'if test "$value" -gt "$latest"; then latest="$value"; fi; fi;'
            )
            for path in paths
        )
        + " printf '%s\\n' \"$latest\""
    )
    result = rt.remote("john3", command, quiet=True)
    return int(result.stdout.strip())


def training_is_stalled(*, progress_mtime: int, now: int) -> bool:
    return progress_mtime > 0 and now - progress_mtime > rt.STALE_TRAINING_SECONDS


def launch_remote_training(*, resume: bool) -> None:
    command = [*rt.TRAIN_COMMAND, *(["--resume"] if resume else [])]
    training = f"cd {shlex.quote(str(rt.JOHN3_ROOT))} && {shlex.join(command)}"
    wrapped = (
        f"{training}; code=$?; "
        f"printf '%s\\n' \"$code\" > {shlex.quote(str(rt.JOHN3_TRAIN_STATUS))}.tmp; "
        f"mv {shlex.quote(str(rt.JOHN3_TRAIN_STATUS))}.tmp "
        f'{shlex.quote(str(rt.JOHN3_TRAIN_STATUS))}; exit "$code"'
    )
    launch = (
        f"rm -f {shlex.quote(str(rt.JOHN3_TRAIN_STATUS))}; "
        f"nohup /bin/zsh -lc {shlex.quote(wrapped)} "
        f">> {shlex.quote(str(rt.JOHN3_TRAIN_LOG))} 2>&1 < /dev/null & "
        "pid=$!; "
        f"printf '%s\\n' \"$pid\" > {shlex.quote(str(rt.JOHN3_TRAIN_PID))}.tmp; "
        f"mv {shlex.quote(str(rt.JOHN3_TRAIN_PID))}.tmp "
        f"{shlex.quote(str(rt.JOHN3_TRAIN_PID))}; printf '%s\\n' \"$pid\""
    )
    result = rt.remote("john3", launch)
    rt.log(f"john3 training launched as PID {result.stdout.strip()} resume={resume}")


def train_on_john3() -> None:
    abrupt_resumes = int(rt.load_state().get("abrupt_resumes", 0))
    while True:
        action = training_action(
            final_report=remote_exists(rt.JOHN3_RUN_DIR / "final-report.json"),
            process_running=remote_training_running(),
            status=remote_status(),
            run_manifest=remote_exists(rt.JOHN3_RUN_DIR / "run.json"),
            latest_checkpoint=remote_exists(rt.JOHN3_RUN_DIR / "latest.json"),
        )
        if action == "complete":
            rt.log("john3 frozen training is complete")
            rt.update_state("training-complete")
            return
        if action == "fail":
            raise RuntimeError(
                "john3 training exited unsuccessfully; no automatic statistical retry"
            )
        if action in {"launch", "resume"}:
            if action == "resume":
                abrupt_resumes += 1
                if abrupt_resumes > rt.MAX_ABRUPT_RESUMES:
                    raise RuntimeError("john3 training exceeded abrupt-resume safety limit")
            launch_remote_training(resume=action == "resume")
            rt.update_state(
                "training",
                abrupt_resumes=abrupt_resumes,
                training_mode=action,
            )
        else:
            progress_mtime = remote_training_progress_mtime()
            if training_is_stalled(progress_mtime=progress_mtime, now=int(time.time())):
                raise RuntimeError(
                    "john3 training produced no log, checkpoint, or report progress for "
                    f"{rt.STALE_TRAINING_SECONDS // 60} minutes"
                )
            tail = rt.remote_shell(
                "john3",
                ["tail", "-n", "1", str(rt.JOHN3_TRAIN_LOG)],
                check=False,
                quiet=True,
            ).stdout.strip()
            if tail:
                rt.log(f"john3 training: {tail[-1_000:]}")
            rt.update_state(
                "training",
                abrupt_resumes=abrupt_resumes,
                training_mode="monitor",
                training_progress_mtime=progress_mtime,
            )
        time.sleep(rt.TRAIN_POLL_SECONDS)


def evaluate_on_john3() -> dict[str, Any]:
    report_path = rt.JOHN3_RUN_DIR / "validation-report.json"
    if remote_exists(report_path):
        rt.log("using the existing frozen validation report on john3")
    else:
        rt.log("running the single frozen validation evaluator on john3")
        rt.remote(
            "john3",
            f"cd {shlex.quote(str(rt.JOHN3_ROOT))} && {shlex.join(rt.EVALUATE_COMMAND)}",
        )
    result = rt.remote_shell("john3", ["cat", str(report_path)])
    report = json.loads(result.stdout)
    if report.get("domain") != "validation":
        raise ValueError("ADR 0078 evaluator opened an unexpected domain")
    if report.get("test_domain_opened") is not False:
        raise ValueError("ADR 0078 evaluator reported test access")
    if report.get("gameplay_domain_opened") is not False:
        raise ValueError("ADR 0078 evaluator reported gameplay access")
    rt.update_state(
        "validation-evaluated",
        validation_passed=bool(report.get("passed")),
        failed_gates=report.get("failed_gates", []),
    )
    return report


def retrieve_run(report: dict[str, Any]) -> None:
    rt.log("copying the integrity-checked run and validation reports to john1")
    incoming = rt.RUN_DIR.with_name(rt.RUN_DIR.name + ".incoming")
    if incoming.exists():
        shutil.rmtree(incoming)
    incoming.mkdir(parents=True)
    rt.rsync_from_remote(
        "john3",
        f"{rt.JOHN3_RUN_DIR}/",
        f"{incoming}/",
        delete=True,
    )
    retrieved = json.loads((incoming / "validation-report.json").read_text())
    if retrieved != report:
        raise ValueError("retrieved validation report differs from john3")
    if rt.RUN_DIR.exists():
        existing_report = rt.RUN_DIR / "validation-report.json"
        if not existing_report.exists() or rt.sha256_file(existing_report) != rt.sha256_file(
            incoming / "validation-report.json"
        ):
            raise ValueError("existing john1 run differs from john3")
        shutil.rmtree(incoming)
    else:
        os.replace(incoming, rt.RUN_DIR)

    rt.atomic_copy(rt.RUN_DIR / "validation-report.json", rt.CANONICAL_JSON_REPORT)
    rt.atomic_copy(rt.RUN_DIR / "validation-report.md", rt.CANONICAL_MARKDOWN_REPORT)
    rt.update_state(
        "validation-complete",
        validation_passed=bool(report.get("passed")),
        failed_gates=report.get("failed_gates", []),
        run_manifest_sha256=rt.sha256_file(rt.RUN_DIR / "run.json"),
        checkpoint_manifest_sha256=rt.sha256_file(
            rt.RUN_DIR / "checkpoints" / report["checkpoint"] / "checkpoint.json"
        ),
        validation_report_sha256=rt.sha256_file(rt.CANONICAL_JSON_REPORT),
    )
