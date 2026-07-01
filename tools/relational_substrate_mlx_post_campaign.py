#!/usr/bin/env python3
"""Build the checksum-bound post-training graph for ADR 0161."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import blake3

EXPERIMENT_ID = "relational-substrate-mlx-tournament-v1"
PROTOCOL_ID = "r5-s3-s5-matched-mlx-v1"
ADR_ID = "0161"
CONTROL_ARM = "c0-exact-r2"
TRAINING_STEPS = 3_000
CAMPAIGN_ROOT = Path("artifacts/experiments") / EXPERIMENT_ID
FROZEN_SOURCE = CAMPAIGN_ROOT / "frozen-source"
CONTROL_RUN = CAMPAIGN_ROOT / "runs" / "c0_exact_r2"
CONTROL_REPORT = CAMPAIGN_ROOT / "reports" / "c0_exact_r2.json"
DEFAULT_OUTPUT = CAMPAIGN_ROOT / "post-training-queue-spec.json"
TRAIN_DATASET = Path("artifacts/datasets/complete-action-graded-oracle-v1-train")
VALIDATION_DATASET = Path(
    "artifacts/datasets/complete-action-graded-oracle-v1-validation"
)
R3_CACHE = Path(
    "artifacts/experiments/r3-action-edit-mlx-comparison-v1/cache/"
    "0de6365fe5dfe57329298e1c3370baeddf14e6edc5909fa930c234d1abc97156"
)
RELATIONAL_CACHE = CAMPAIGN_ROOT / "cache" / (
    "d4f8e2eb83db237b136fd478b73802544938c36adf77db0bf40f2b3276181bef"
)
S1_CACHE = Path(
    "artifacts/experiments/exact-semantic-supply-learned-comparison-v1/cache/"
    "2323ead43b1bff7a506ecef4b8bd4793cebe4d53c6f8940b03404573ca5e6c15"
)
R6_BINARY = Path(
    "tools/relational_feature_census/target/release/"
    "relational-substrate-r6-replay"
)
AUTHORIZATION = CAMPAIGN_ROOT / "control" / "authorization.json"
REMOTE_ROOTS = {
    "john1": Path("/Users/johnherrick/cascadia"),
    "john2": Path("/Users/john2/cascadia-bench"),
    "john3": Path("/Users/john3/cascadia-bench"),
    "john4": Path("/Users/john4/cascadia-bench"),
}
TREATMENT_REPORTS = {
    "q1-r5-quotient-local": CAMPAIGN_ROOT
    / "reports"
    / "q1_r5_quotient_local.json",
    "g2-r5-s3": CAMPAIGN_ROOT / "reports" / "g2_r5_s3.json",
    "d3-r5-s3-s5": CAMPAIGN_ROOT / "reports" / "d3_r5_s3_s5.json",
}


class PostCampaignError(RuntimeError):
    """Raised when the completed C0 evidence cannot seed paired controls."""


@dataclass(frozen=True)
class Replay:
    host: str
    treatment_arm: str

    @property
    def task_id(self) -> str:
        return f"relmlx-c0-replay-{self.host}"

    @property
    def report_relative(self) -> Path:
        slug = self.treatment_arm.replace("-", "_")
        return CAMPAIGN_ROOT / "reports" / f"c0_replay_{slug}_{self.host}.json"


REPLAYS = (
    Replay("john2", "q1-r5-quotient-local"),
    Replay("john3", "g2-r5-s3"),
    Replay("john4", "d3-r5-s3-s5"),
)


def canonical_blake3(value: object) -> str:
    return blake3.blake3(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def file_blake3(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise PostCampaignError(f"{label} is unreadable: {path}") from error
    if not isinstance(value, dict):
        raise PostCampaignError(f"{label} is not a JSON object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _remote(host: str, relative: Path) -> str:
    return str(REMOTE_ROOTS[host] / relative)


def validate_completed_control(repository: Path) -> dict[str, Any]:
    run = repository / CONTROL_RUN
    final_report_path = run / "final-report.json"
    report_path = repository / CONTROL_REPORT
    final_report = _read_json(final_report_path, "C0 final report")
    collected_report = _read_json(report_path, "C0 output report")
    if final_report != collected_report:
        raise PostCampaignError(
            "C0 run final report and top-level output report differ"
        )
    if (
        final_report.get("schema_version") != 1
        or final_report.get("experiment_id") != EXPERIMENT_ID
        or final_report.get("protocol_id") != PROTOCOL_ID
        or final_report.get("adr") != ADR_ID
        or final_report.get("mode") != "production"
        or final_report.get("arm") != CONTROL_ARM
        or final_report.get("host") != "john1"
        or final_report.get("optimization", {}).get("global_step")
        != TRAINING_STEPS
    ):
        raise PostCampaignError("C0 final report is not the completed ADR 0161 control")
    identity = final_report.get("scientific_identity")
    if (
        not isinstance(identity, dict)
        or canonical_blake3(identity) != final_report.get("report_id")
    ):
        raise PostCampaignError("C0 final report identity is malformed")

    latest = _read_json(run / "latest.json", "C0 latest checkpoint pointer")
    checkpoint_name = latest.get("checkpoint")
    if not isinstance(checkpoint_name, str) or not checkpoint_name:
        raise PostCampaignError("C0 latest checkpoint pointer is malformed")
    checkpoint = run / "checkpoints" / checkpoint_name
    if not checkpoint.is_dir():
        raise PostCampaignError(f"C0 latest checkpoint is missing: {checkpoint}")
    checkpoint_identity = final_report.get("checkpoint")
    if not isinstance(checkpoint_identity, dict):
        raise PostCampaignError("C0 final report lacks checkpoint identity")
    manifest = checkpoint / "checkpoint.json"
    model = checkpoint / "model.safetensors"
    if not manifest.is_file() or not model.is_file():
        raise PostCampaignError("C0 latest checkpoint is incomplete")
    if (
        file_blake3(manifest) != checkpoint_identity.get("manifest_blake3")
        or file_blake3(model) != checkpoint_identity.get("model_blake3")
        or Path(str(checkpoint_identity.get("path", ""))).name
        != checkpoint_name
    ):
        raise PostCampaignError("C0 latest checkpoint bytes differ from its report")
    return {
        "report_id": final_report["report_id"],
        "checkpoint_name": checkpoint_name,
        "checkpoint_manifest_blake3": checkpoint_identity["manifest_blake3"],
        "checkpoint_model_blake3": checkpoint_identity["model_blake3"],
    }


def _task(
    *,
    task_id: str,
    title: str,
    decision: str,
    workload_class: str,
    priority: int,
    expected_runtime_seconds: int,
    decision_terminal: bool,
    compatible_hosts: list[str],
    dependencies: list[str],
    command: list[str],
    artifact_path: Path,
    stop_rule: str,
    cpu_cores: int,
    memory_gib: float,
    uses_mlx: bool,
) -> dict[str, Any]:
    return {
        "id": task_id,
        "title": title,
        "experiment_id": EXPERIMENT_ID,
        "decision": decision,
        "workload_class": workload_class,
        "priority": priority,
        "decision_value": 1.0,
        "expected_runtime_seconds": expected_runtime_seconds,
        "critical_path": True,
        "decision_terminal": decision_terminal,
        "compatible_hosts": compatible_hosts,
        "dependencies": dependencies,
        "command": command,
        "artifact_path": str(artifact_path),
        "stop_rule": stop_rule,
        "resources": {
            "cpu_cores": cpu_cores,
            "memory_gib": memory_gib,
            "uses_mlx": uses_mlx,
        },
    }


def _replay_command(replay: Replay) -> list[str]:
    root = REMOTE_ROOTS[replay.host]
    source = root / FROZEN_SOURCE
    return [
        "/usr/bin/env",
        "-C",
        str(source),
        f"PYTHONPATH={source / 'python'}",
        str(root / ".venv/bin/python"),
        "-B",
        "tools/relational_substrate_mlx_control_replay.py",
        "--control-report",
        _remote(replay.host, CONTROL_RUN / "final-report.json"),
        "--authorization",
        _remote(replay.host, AUTHORIZATION),
        "--treatment-arm",
        replay.treatment_arm,
        "--train-dataset",
        _remote(replay.host, TRAIN_DATASET),
        "--validation-dataset",
        _remote(replay.host, VALIDATION_DATASET),
        "--r3-cache",
        _remote(replay.host, R3_CACHE),
        "--relational-cache",
        _remote(replay.host, RELATIONAL_CACHE),
        "--s1-cache",
        _remote(replay.host, S1_CACHE),
        "--r6-binary",
        _remote(replay.host, R6_BINARY),
        "--run-dir",
        _remote(replay.host, CONTROL_RUN),
        "--output",
        _remote(replay.host, replay.report_relative),
        "--warmup-iterations",
        "5",
        "--steady-iterations",
        "30",
    ]


def build_task_specs(repository: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    control_identity = validate_completed_control(repository)
    fanout_id = "relmlx-c0-run-fanout"
    fanout_report = CAMPAIGN_ROOT / "control" / "c0-run-fanout.json"
    fanout_command = [
        ".venv/bin/python",
        "-B",
        "tools/cluster_artifact_fanout.py",
        "--source",
        f"{CONTROL_RUN}/",
        "--local-root",
        str(CONTROL_RUN),
    ]
    for replay in REPLAYS:
        fanout_command.extend(
            [
                "--destination",
                f"{replay.host}:{_remote(replay.host, CONTROL_RUN)}/",
            ]
        )
    fanout_command.extend(
        [
            "--required-file",
            "final-report.json",
            "--required-file",
            "latest.json",
            "--verify-tree",
            "--output",
            str(fanout_report),
        ]
    )
    tasks = [
        _task(
            task_id=fanout_id,
            title="Fan out exact ADR 0161 C0 run",
            decision="Bind all treatment hosts to the exact completed C0 checkpoint",
            workload_class="shared-prerequisite",
            priority=0,
            expected_runtime_seconds=300,
            decision_terminal=False,
            compatible_hosts=["john1"],
            dependencies=[],
            command=fanout_command,
            artifact_path=fanout_report,
            stop_rule="Every regular C0 run file must match on john1 through john4.",
            cpu_cores=1,
            memory_gib=1.0,
            uses_mlx=False,
        )
    ]

    replay_ids = []
    for replay in REPLAYS:
        replay_ids.append(replay.task_id)
        tasks.append(
            _task(
                task_id=replay.task_id,
                title=f"Replay exact C0 beside {replay.treatment_arm} on {replay.host}",
                decision="Measure material efficiency against a same-host exact control",
                workload_class="independent-experiment",
                priority=1,
                expected_runtime_seconds=1_200,
                decision_terminal=False,
                compatible_hosts=[replay.host],
                dependencies=[fanout_id],
                command=_replay_command(replay),
                artifact_path=replay.report_relative,
                stop_rule=(
                    "Measure all 240 decisions and 860,203 actions with exact "
                    "C0 and R6 bytes; any parity or identity failure invalidates "
                    "the replay."
                ),
                cpu_cores=10,
                memory_gib=8.0,
                uses_mlx=True,
            )
        )

    collection = CAMPAIGN_ROOT / "control" / "paired-control-collection.json"
    collect_command = [
        "/usr/bin/env",
        "-C",
        str(REMOTE_ROOTS["john1"]),
        str(REMOTE_ROOTS["john1"] / ".venv/bin/python"),
        "-B",
        "tools/cluster_artifact_collect.py",
    ]
    for replay in REPLAYS:
        collect_command.extend(
            [
                "--artifact",
                f"{replay.host}:{_remote(replay.host, replay.report_relative)}",
                _remote("john1", replay.report_relative),
            ]
        )
    collect_command.extend(["--output", _remote("john1", collection)])
    collect_id = "relmlx-c0-replay-collect"
    tasks.append(
        _task(
            task_id=collect_id,
            title="Collect three host-paired C0 replays",
            decision="Require checksum-bound paired-control evidence from every host",
            workload_class="shared-prerequisite",
            priority=2,
            expected_runtime_seconds=90,
            decision_terminal=False,
            compatible_hosts=["john1"],
            dependencies=replay_ids,
            command=collect_command,
            artifact_path=collection,
            stop_rule="All three replay reports must be present and SHA-256 verified.",
            cpu_cores=1,
            memory_gib=1.0,
            uses_mlx=False,
        )
    )

    forward = CAMPAIGN_ROOT / "aggregate-forward.json"
    reverse = CAMPAIGN_ROOT / "aggregate-reverse.json"
    order_proof = CAMPAIGN_ROOT / "order-proof.json"
    source = REMOTE_ROOTS["john1"] / FROZEN_SOURCE
    classify_command = [
        "/usr/bin/env",
        "-C",
        str(source),
        f"PYTHONPATH={source / 'python'}",
        str(REMOTE_ROOTS["john1"] / ".venv/bin/python"),
        "-B",
        "tools/relational_substrate_mlx_report.py",
    ]
    for report in (CONTROL_REPORT, *TREATMENT_REPORTS.values()):
        classify_command.extend(["--report", _remote("john1", report)])
    for replay in REPLAYS:
        classify_command.extend(
            ["--paired-control", _remote("john1", replay.report_relative)]
        )
    classify_command.extend(
        [
            "--forward-output",
            _remote("john1", forward),
            "--reverse-output",
            _remote("john1", reverse),
            "--order-proof-output",
            _remote("john1", order_proof),
        ]
    )
    tasks.append(
        _task(
            task_id="relmlx-classify",
            title="Classify the ADR 0161 relational substrate tournament",
            decision="Select a compact relational substrate only under every frozen gate",
            workload_class="shared-prerequisite",
            priority=3,
            expected_runtime_seconds=60,
            decision_terminal=True,
            compatible_hosts=["john1"],
            dependencies=[collect_id],
            command=classify_command,
            artifact_path=forward,
            stop_rule=(
                "Structural drift is invalid evidence; quality or serving gate "
                "misses remain valid negative evidence."
            ),
            cpu_cores=1,
            memory_gib=1.0,
            uses_mlx=False,
        )
    )
    return tasks, control_identity


def build_queue_spec(repository: Path) -> dict[str, Any]:
    tasks, control_identity = build_task_specs(repository)
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "control_identity": control_identity,
        "task_count": len(tasks),
        "tasks": tasks,
        "task_spec_blake3": canonical_blake3(tasks),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-spec")
    build.add_argument(
        "--repository",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    build.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = build_queue_spec(args.repository)
    _write_json(args.output, payload)
    print(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "control_report_id": payload["control_identity"]["report_id"],
                "task_count": payload["task_count"],
                "task_spec_blake3": payload["task_spec_blake3"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
