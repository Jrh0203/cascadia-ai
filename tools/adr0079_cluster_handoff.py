"""Conditional sealed-test collection and evaluation for ADR 0079."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import time
from typing import Any

import adr0078_cluster_runtime as rt


def authorize_test_collection(validation_report: dict[str, Any]) -> dict[str, Any]:
    if not validation_report.get("passed") or validation_report.get("failed_gates"):
        raise ValueError("ADR 0079 cannot be authorized without a complete validation pass")
    validation_sha256 = rt.sha256_file(rt.CANONICAL_JSON_REPORT)
    if rt.TEST_AUTHORIZATION.exists():
        authorization = json.loads(rt.TEST_AUTHORIZATION.read_text())
        validate_test_authorization(
            authorization,
            validation_report=validation_report,
            validation_report_sha256=validation_sha256,
        )
        return authorization

    absent = {
        "john1": not rt.TEST_DATASET.exists(),
        "john2": not rt.remote_path_exists("john2", rt.REMOTE_TEST_DATASET),
        "john3": not rt.remote_path_exists("john3", rt.JOHN3_TEST_DATASET),
    }
    if not all(absent.values()):
        raise RuntimeError("sealed test data existed before ADR 0079 authorization")
    authorization = {
        "schema_version": 1,
        "experiment": "r12-counterfactual-advantage-set-ranker-v1-sealed-test-20260613",
        "parent_experiment": "r12-counterfactual-advantage-set-ranker-v1-20260613",
        "authorized_at": rt.timestamp(),
        "authorized_at_unix_seconds": int(time.time()),
        "validation_passed": True,
        "validation_report_sha256": validation_sha256,
        "validation_checkpoint": validation_report["checkpoint"],
        "validation_checkpoint_manifest_blake3": validation_report["checkpoint_manifest_blake3"],
        "test_absent_on_nodes": absent,
        "collector_executable_blake3": rt.EXPECTED_EXECUTABLE_BLAKE3,
        "first_game_index": 71_000,
        "requested_games": 32,
    }
    rt.atomic_json(rt.TEST_AUTHORIZATION, authorization)
    validate_test_authorization(
        authorization,
        validation_report=validation_report,
        validation_report_sha256=validation_sha256,
    )
    rt.update_state(
        "test-authorized",
        test_authorization_sha256=rt.sha256_file(rt.TEST_AUTHORIZATION),
    )
    rt.log("ADR 0079 sealed test authorized after complete validation pass")
    return authorization


def validate_test_authorization(
    authorization: dict[str, Any],
    *,
    validation_report: dict[str, Any],
    validation_report_sha256: str,
) -> None:
    expected = {
        "schema_version": 1,
        "experiment": "r12-counterfactual-advantage-set-ranker-v1-sealed-test-20260613",
        "parent_experiment": "r12-counterfactual-advantage-set-ranker-v1-20260613",
        "validation_passed": True,
        "validation_report_sha256": validation_report_sha256,
        "validation_checkpoint": validation_report["checkpoint"],
        "validation_checkpoint_manifest_blake3": validation_report["checkpoint_manifest_blake3"],
        "collector_executable_blake3": rt.EXPECTED_EXECUTABLE_BLAKE3,
        "first_game_index": 71_000,
        "requested_games": 32,
    }
    for key, value in expected.items():
        if authorization.get(key) != value:
            raise ValueError(
                f"ADR 0079 authorization {key}={authorization.get(key)!r}, expected {value!r}"
            )
    if int(authorization.get("authorized_at_unix_seconds", 0)) <= 0:
        raise ValueError("ADR 0079 authorization has no valid timestamp")
    absent = authorization.get("test_absent_on_nodes")
    if not isinstance(absent, dict) or set(absent) != {"john1", "john2", "john3"}:
        raise ValueError("ADR 0079 authorization has incomplete node-absence evidence")
    if not all(value is True for value in absent.values()):
        raise ValueError("ADR 0079 authorization did not prove the test was sealed")


def remote_test_status() -> int | None:
    result = rt.remote_shell(
        "john2",
        ["cat", str(rt.REMOTE_TEST_STATUS)],
        check=False,
        quiet=True,
    )
    if result.returncode != 0:
        return None
    return int(result.stdout.strip())


def remote_test_collector_running() -> bool:
    result = rt.remote(
        "john2",
        (
            f"test -f {shlex.quote(str(rt.REMOTE_TEST_PID))} && "
            f"pid=$(cat {shlex.quote(str(rt.REMOTE_TEST_PID))}) && "
            'kill -0 "$pid" 2>/dev/null && '
            'command=$(/bin/ps -p "$pid" -o command=) && '
            f"printf '%s\\n' \"$command\" | /usr/bin/grep -F -- "
            f"{shlex.quote('r12-counterfactual-advantage-v1-test-32')} "
            "> /dev/null"
        ),
        check=False,
        quiet=True,
    )
    return result.returncode == 0


def launch_test_collection() -> None:
    collection = f"cd {shlex.quote(str(rt.JOHN2_ROOT))} && {shlex.join(rt.TEST_COLLECT_COMMAND)}"
    wrapped = (
        f"{collection}; code=$?; "
        f"printf '%s\\n' \"$code\" > {shlex.quote(str(rt.REMOTE_TEST_STATUS))}.tmp; "
        f"mv {shlex.quote(str(rt.REMOTE_TEST_STATUS))}.tmp "
        f'{shlex.quote(str(rt.REMOTE_TEST_STATUS))}; exit "$code"'
    )
    launch = (
        f"mkdir -p {shlex.quote(str(rt.REMOTE_TEST_LOG.parent))}; "
        f"rm -f {shlex.quote(str(rt.REMOTE_TEST_STATUS))}; "
        f"nohup /bin/zsh -lc {shlex.quote(wrapped)} "
        f">> {shlex.quote(str(rt.REMOTE_TEST_LOG))} 2>&1 < /dev/null & "
        "pid=$!; "
        f"printf '%s\\n' \"$pid\" > {shlex.quote(str(rt.REMOTE_TEST_PID))}.tmp; "
        f"mv {shlex.quote(str(rt.REMOTE_TEST_PID))}.tmp "
        f"{shlex.quote(str(rt.REMOTE_TEST_PID))}; printf '%s\\n' \"$pid\""
    )
    result = rt.remote("john2", launch)
    rt.log(f"john2 ADR 0079 test collection launched as PID {result.stdout.strip()}")


def wait_for_test_collection() -> dict[str, Any]:
    previous = -1
    state = rt.load_state()
    abrupt_resumes = int(state.get("test_abrupt_resumes", 0))
    started = bool(state.get("test_collection_started", False))
    unavailable_since: float | None = None
    while True:
        try:
            manifest = rt.load_manifest(rt.TEST_SPEC)
        except rt.RemoteHostUnavailable as error:
            if unavailable_since is None:
                unavailable_since = time.monotonic()
                rt.log(
                    "john2 unavailable during ADR 0079 collection; "
                    f"preserving {max(previous, 0)}/{rt.TEST_SPEC.requested_games}: {error}"
                )
            if time.monotonic() - unavailable_since > rt.STALE_PROGRESS_SECONDS:
                raise RuntimeError(
                    "john2 remained unreachable during ADR 0079 collection for "
                    f"{rt.STALE_PROGRESS_SECONDS // 60} minutes"
                ) from error
            time.sleep(rt.POLL_SECONDS)
            continue
        if unavailable_since is not None:
            rt.log("john2 connectivity recovered during ADR 0079 collection")
            unavailable_since = None
        completed = 0
        if manifest is not None:
            rt.validate_manifest_contract(manifest, rt.TEST_SPEC, require_complete=False)
            completed = int(manifest["completed_games"])
            if completed == rt.TEST_SPEC.requested_games:
                rt.validate_manifest_contract(manifest, rt.TEST_SPEC, require_complete=True)
                rt.update_state(
                    "test-collection-complete",
                    test_completed=completed,
                    test_abrupt_resumes=abrupt_resumes,
                )
                return manifest

        if completed != previous:
            rt.log(f"test progress: {completed}/{rt.TEST_SPEC.requested_games}")
            previous = completed
            rt.update_state(
                "test-collecting",
                test_completed=completed,
                test_abrupt_resumes=abrupt_resumes,
                test_collection_started=started,
            )

        running = remote_test_collector_running()
        if running and manifest is not None:
            updated = int(manifest.get("updated_unix_seconds", 0))
            if updated and int(time.time()) - updated > rt.STALE_PROGRESS_SECONDS:
                raise RuntimeError(
                    "ADR 0079 test collector made no manifest progress for "
                    f"{rt.STALE_PROGRESS_SECONDS // 60} minutes"
                )

        if not running:
            status = remote_test_status()
            if status is not None:
                raise RuntimeError(
                    f"john2 ADR 0079 collector exited before completion with status {status}"
                )
            if started:
                abrupt_resumes += 1
                if abrupt_resumes > rt.MAX_TEST_ABRUPT_RESUMES:
                    raise RuntimeError(
                        "ADR 0079 test collection exceeded abrupt-resume safety limit"
                    )
            launch_test_collection()
            started = True
            rt.update_state(
                "test-collecting",
                test_completed=completed,
                test_abrupt_resumes=abrupt_resumes,
                test_collection_started=True,
            )
        time.sleep(rt.POLL_SECONDS)


def validate_and_sync_test_dataset() -> None:
    rt.log("validating ADR 0079 test data on john2 and copying it to john1")
    rt.remote(
        "john2",
        (
            f"cd {shlex.quote(str(rt.JOHN2_ROOT))} && "
            "./target/release/cascadia-v2 "
            "validate-counterfactual-advantage-dataset "
            "--dataset artifacts/datasets/r12-counterfactual-advantage-v1-test-32"
        ),
    )
    incoming = rt.TEST_DATASET.with_name(rt.TEST_DATASET.name + ".incoming")
    if incoming.exists():
        shutil.rmtree(incoming)
    incoming.mkdir(parents=True)
    rt.rsync_from_remote(
        "john2",
        f"{rt.REMOTE_TEST_DATASET}/",
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
    if rt.TEST_DATASET.exists():
        if rt.sha256_file(rt.TEST_DATASET / "dataset.json") != rt.sha256_file(
            incoming / "dataset.json"
        ):
            raise ValueError("existing john1 test dataset differs from john2")
        shutil.rmtree(incoming)
    else:
        os.replace(incoming, rt.TEST_DATASET)

    rt.remote(
        "john3",
        (
            f"mkdir -p {shlex.quote(str(rt.JOHN3_ROOT / 'tools'))} "
            f"{shlex.quote(str(rt.JOHN3_ROOT / 'artifacts/datasets'))}"
        ),
    )
    rt.rsync_to_remote(
        "john3",
        f"{rt.TEST_DATASET}/",
        f"{rt.JOHN3_TEST_DATASET}/",
        delete=True,
    )
    rt.rsync_to_remote(
        "john3",
        str(rt.TEST_EVALUATOR),
        str(rt.JOHN3_TEST_EVALUATOR),
    )
    rt.rsync_to_remote(
        "john3",
        str(rt.TEST_AUTHORIZATION),
        str(rt.JOHN3_TEST_AUTHORIZATION),
    )
    rt.remote(
        "john3",
        (
            f"cd {shlex.quote(str(rt.JOHN3_ROOT))} && "
            "./target/release/cascadia-v2 "
            "validate-counterfactual-advantage-dataset "
            "--dataset artifacts/datasets/r12-counterfactual-advantage-v1-test-32"
        ),
    )
    evaluator_sha = rt.remote_shell(
        "john3",
        ["shasum", "-a", "256", str(rt.JOHN3_TEST_EVALUATOR)],
    ).stdout.split()[0]
    if evaluator_sha != rt.sha256_file(rt.TEST_EVALUATOR):
        raise ValueError("ADR 0079 evaluator changed during transfer")
    remote_validation_sha = rt.remote_shell(
        "john3",
        ["shasum", "-a", "256", str(rt.JOHN3_RUN_DIR / "validation-report.json")],
    ).stdout.split()[0]
    if remote_validation_sha != rt.sha256_file(rt.CANONICAL_JSON_REPORT):
        raise ValueError("john3 validation report differs from the authorized report")
    rt.update_state(
        "test-data-ready-on-john3",
        test_manifest_sha256=rt.sha256_file(rt.TEST_DATASET / "dataset.json"),
        test_evaluator_sha256=rt.sha256_file(rt.TEST_EVALUATOR),
    )


def evaluate_test_on_john3() -> dict[str, Any]:
    report_path = rt.JOHN3_RUN_DIR / "test-report.json"
    if rt.remote_path_exists("john3", report_path):
        rt.log("using the existing ADR 0079 sealed-test report on john3")
    else:
        rt.log("running the single ADR 0079 sealed-test evaluator on john3")
        command = [
            "env",
            "PYTHONPATH=python",
            ".venv/bin/python",
            str(rt.JOHN3_TEST_EVALUATOR.relative_to(rt.JOHN3_ROOT)),
            "--run-dir",
            str(rt.JOHN3_RUN_DIR.relative_to(rt.JOHN3_ROOT)),
            "--dataset",
            str(rt.JOHN3_TEST_DATASET.relative_to(rt.JOHN3_ROOT)),
            "--validation-report",
            str((rt.JOHN3_RUN_DIR / "validation-report.json").relative_to(rt.JOHN3_ROOT)),
            "--authorization",
            str(rt.JOHN3_TEST_AUTHORIZATION.relative_to(rt.JOHN3_ROOT)),
            "--output",
            str(report_path.relative_to(rt.JOHN3_ROOT)),
            "--markdown-output",
            str((rt.JOHN3_RUN_DIR / "test-report.md").relative_to(rt.JOHN3_ROOT)),
            "--group-batch-size",
            "32",
        ]
        rt.remote(
            "john3",
            f"cd {shlex.quote(str(rt.JOHN3_ROOT))} && {shlex.join(command)}",
        )
    result = rt.remote_shell("john3", ["cat", str(report_path)])
    report = json.loads(result.stdout)
    if report.get("domain") != "test" or report.get("test_domain_opened") is not True:
        raise ValueError("ADR 0079 evaluator did not report the sealed test domain")
    if report.get("gameplay_domain_opened") is not False:
        raise ValueError("ADR 0079 evaluator reported gameplay access")
    rt.update_state(
        "test-evaluated",
        test_passed=bool(report.get("passed")),
        test_failed_gates=report.get("failed_gates", []),
    )
    return report


def retrieve_test_report(report: dict[str, Any]) -> None:
    incoming = rt.RUN_DIR.with_name(rt.RUN_DIR.name + ".test-report.incoming")
    if incoming.exists():
        shutil.rmtree(incoming)
    incoming.mkdir(parents=True)
    for name in ("test-report.json", "test-report.md"):
        rt.rsync_from_remote(
            "john3",
            str(rt.JOHN3_RUN_DIR / name),
            str(incoming / name),
        )
    retrieved = json.loads((incoming / "test-report.json").read_text())
    if retrieved != report:
        raise ValueError("retrieved ADR 0079 report differs from john3")
    for name in ("test-report.json", "test-report.md"):
        destination = rt.RUN_DIR / name
        if destination.exists() and rt.sha256_file(destination) != rt.sha256_file(incoming / name):
            raise ValueError(f"existing john1 {name} differs from john3")
        rt.atomic_copy(incoming / name, destination)
    shutil.rmtree(incoming)
    rt.atomic_copy(rt.RUN_DIR / "test-report.json", rt.CANONICAL_TEST_JSON_REPORT)
    rt.atomic_copy(rt.RUN_DIR / "test-report.md", rt.CANONICAL_TEST_MARKDOWN_REPORT)
    rt.update_state(
        "complete",
        validation_passed=True,
        test_authorized=True,
        test_passed=bool(report.get("passed")),
        test_failed_gates=report.get("failed_gates", []),
        test_report_sha256=rt.sha256_file(rt.CANONICAL_TEST_JSON_REPORT),
    )


def complete_conditional_test(validation_report: dict[str, Any]) -> None:
    if not validation_report.get("passed"):
        rt.log("ADR 0078 validation failed; ADR 0079 remains unopened")
        rt.update_state(
            "complete",
            validation_passed=False,
            test_authorized=False,
            test_domain_opened=False,
        )
        return
    authorize_test_collection(validation_report)
    wait_for_test_collection()
    validate_and_sync_test_dataset()
    report = evaluate_test_on_john3()
    retrieve_test_report(report)
    rt.log(
        "ADR 0079 sealed test complete: "
        f"passed={report['passed']} failed_gates={report['failed_gates']}"
    )
