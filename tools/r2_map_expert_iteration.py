#!/usr/bin/env python3
"""Storage-host control-plane CLI for the R2-MAP expert-iteration campaign.

Every command proves it is running on john2 and performs the native-disk
preflight before reading or mutating campaign control state. Remote
orchestrators must invoke it through the strict storage transport/executor and
must never treat the john2 path as a local path. The tool never launches
gameplay or MLX itself.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPOSITORY / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from cascadia_mlx.r2_map_campaign_controller import (  # noqa: E402
    CampaignControllerError,
    ControllerPaths,
    advance_campaign,
    import_benchmark_feed,
    import_receipt,
    phase_barrier,
    reconcile,
    recover_current_phase,
    run_isolated_dry_run,
    write_controller_schemas,
)
from cascadia_mlx.r2_map_contracts import (  # noqa: E402
    CAMPAIGN_ROOT,
    ContractError,
    Phase,
    append_decision,
    initialize_layout,
    new_campaign_state,
    new_storage_supersession_genesis,
    preflight_storage,
    read_decision_log,
    read_state,
    transition_state,
    write_contract_schemas,
    write_state,
    write_storage_supersession_genesis,
)
from cascadia_mlx.r2_map_dashboard_status import (  # noqa: E402
    DashboardStatusInputs,
    build_dashboard_status,
    read_compact_json,
    write_dashboard_status,
)

STATE_PATH = CAMPAIGN_ROOT / "control/campaign-state.json"
DECISION_LOG_PATH = CAMPAIGN_ROOT / "control/decision-log.jsonl"
DASHBOARD_STATUS_PATH = CAMPAIGN_ROOT / "control/dashboard-status.json"
def _json(value: Any) -> None:
    print(json.dumps(value, sort_keys=True, indent=2))


def _configured_paths(values: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ContractError(f"configured path must be NAME=/absolute/path, got {value!r}")
        name, raw_path = value.split("=", 1)
        if not name or name in parsed:
            raise ContractError(f"configured path name is empty or duplicated: {name!r}")
        parsed[name] = Path(raw_path)
    return parsed


def _preflight(arguments: argparse.Namespace) -> dict[str, Any]:
    dashboard_paths = {}
    for name in (
        "campaign_state_path",
        "host_receipts_path",
        "host_safety_path",
        "training_progress_path",
        "benchmark_aggregate_path",
        "model_manifest_path",
        "pool_manifest_path",
        "output",
    ):
        value = getattr(arguments, name, None)
        if value is not None:
            dashboard_paths[f"dashboard_{name}"] = value
    configured = {
        "campaign_state": STATE_PATH,
        "decision_log": DECISION_LOG_PATH,
        **dashboard_paths,
        **_configured_paths(getattr(arguments, "path", [])),
    }
    return preflight_storage(
        configured_paths=configured,
        expected_run_bytes=getattr(arguments, "expected_run_bytes", 0),
        measure_campaign_bytes=not getattr(arguments, "skip_campaign_scan", False),
    )


def command_preflight(arguments: argparse.Namespace) -> None:
    _json(_preflight(arguments))


def command_init(arguments: argparse.Namespace) -> None:
    proof = _preflight(arguments)
    missing_genesis_hashes = [
        name
        for name in (
            "legacy_campaign_state_sha256",
            "legacy_decision_head_sha256",
            "authorization_sha256",
        )
        if getattr(arguments, name) is None
    ]
    if not DECISION_LOG_PATH.exists() and missing_genesis_hashes:
        raise ContractError(
            "canonical initialization requires full storage-genesis hashes: "
            + ", ".join(missing_genesis_hashes)
        )
    if DECISION_LOG_PATH.exists() and not STATE_PATH.exists():
        raise ContractError("decision genesis exists without its canonical campaign state")
    initialize_layout()
    schema_hashes = write_contract_schemas(CAMPAIGN_ROOT)
    schema_hashes.update(write_controller_schemas(ControllerPaths.under(CAMPAIGN_ROOT)))
    if STATE_PATH.exists():
        state = read_state(STATE_PATH)
        created = False
    else:
        state = new_campaign_state()
        write_state(STATE_PATH, state)
        created = True
    if DECISION_LOG_PATH.exists():
        decisions = read_decision_log(DECISION_LOG_PATH)
        if not decisions or decisions[0].get("storage_supersession_genesis") is not True:
            raise ContractError(
                "canonical john2 decision log lacks its storage-supersession genesis"
            )
        genesis = decisions[0]
    else:
        genesis = new_storage_supersession_genesis(
            legacy_campaign_state_sha256=arguments.legacy_campaign_state_sha256,
            legacy_decision_head_sha256=arguments.legacy_decision_head_sha256,
            canonical_state=state,
            authorization_sha256=arguments.authorization_sha256,
        )
        write_storage_supersession_genesis(DECISION_LOG_PATH, genesis)
    if genesis["canonical_campaign_state_sha256"] != state["state_sha256"]:
        raise ContractError("storage-supersession genesis names another canonical state")
    _json(
        {
            "preflight": proof,
            "schema_sha256": schema_hashes,
            "state_created": created,
            "state": state,
            "storage_supersession_genesis": genesis,
        }
    )


def command_show_state(arguments: argparse.Namespace) -> None:
    _preflight(arguments)
    _json(read_state(STATE_PATH))


def command_transition(arguments: argparse.Namespace) -> None:
    _preflight(arguments)
    current = read_state(STATE_PATH)
    proposed = transition_state(
        current,
        arguments.phase,
        reason=arguments.reason,
        generation_manifest_sha256=arguments.generation_manifest_sha256,
        candidate_checkpoint_sha256=arguments.candidate_checkpoint_sha256,
        completed_shard_hosts=arguments.completed_shard_host,
    )
    write_state(STATE_PATH, proposed, expected_current=current)
    _json(proposed)


def command_record_decision(arguments: argparse.Namespace) -> None:
    _preflight(arguments)
    state = read_state(STATE_PATH)
    entry = append_decision(
        DECISION_LOG_PATH,
        actor=arguments.actor,
        triggering_evidence=arguments.evidence,
        alternatives_considered=arguments.alternative,
        chosen_action=arguments.chosen_action,
        affected_artifacts=arguments.artifact,
        rollback_path=arguments.rollback_path,
        state=state,
        decision_kind=arguments.kind,
        authorization_sha256=arguments.authorization_sha256,
    )
    _json(entry)


def command_verify_decisions(arguments: argparse.Namespace) -> None:
    _preflight(arguments)
    entries = read_decision_log(DECISION_LOG_PATH)
    _json(
        {
            "entries": len(entries),
            "head_sha256": None if not entries else entries[-1]["decision_sha256"],
            "valid": True,
        }
    )


def _controller_paths(arguments: argparse.Namespace) -> ControllerPaths:
    if getattr(arguments, "isolated_queue", False):
        return ControllerPaths.under(arguments.campaign_root)
    return ControllerPaths.with_existing_queue_and_ledger(
        arguments.campaign_root,
        queue=arguments.queue,
        ledger=arguments.ledger,
    )


def _controller_preflight(arguments: argparse.Namespace) -> ControllerPaths:
    paths = _controller_paths(arguments)
    preflight_storage(
        configured_paths={
            "controller_root": paths.root,
            "controller_state": paths.state,
            "controller_queue": paths.queue,
            "controller_ledger": paths.ledger,
            "controller_packets": paths.packets,
            "controller_receipts": paths.receipts,
        },
        expected_run_bytes=getattr(arguments, "expected_run_bytes", 0),
    )
    return paths


def _commands_manifest(path: Path) -> dict[str, list[str]]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignControllerError(f"cannot read command manifest: {error}") from error
    if (
        not isinstance(value, dict)
        or not value
        or any(
            not isinstance(operation, str)
            or not operation
            or not isinstance(command, list)
            or not command
            or any(not isinstance(item, str) or not item for item in command)
            for operation, command in value.items()
        )
    ):
        raise CampaignControllerError("command manifest must map operations to argument arrays")
    return value


def command_advance_phase(arguments: argparse.Namespace) -> None:
    paths = _controller_preflight(arguments)
    _json(
        advance_campaign(
            paths,
            commands=_commands_manifest(arguments.commands_manifest),
            artifact_root=arguments.artifact_root,
            reason=arguments.reason,
            now=arguments.at,
            now_ms=time.time_ns() // 1_000_000,
        )
    )


def command_import_work_receipt(arguments: argparse.Namespace) -> None:
    paths = _controller_preflight(arguments)
    _json(import_receipt(paths, source=arguments.receipt))


def command_import_benchmark_feed(arguments: argparse.Namespace) -> None:
    paths = _controller_preflight(arguments)
    _json(
        import_benchmark_feed(
            paths,
            feed_path=arguments.feed,
            aggregate_task_id=arguments.aggregate_task_id,
            expected_state_sha256=arguments.expected_state_sha256,
        )
    )


def command_reconcile_controller(arguments: argparse.Namespace) -> None:
    paths = _controller_preflight(arguments)
    _json(reconcile(paths, now_ms=time.time_ns() // 1_000_000))


def command_recover_phase(arguments: argparse.Namespace) -> None:
    paths = _controller_preflight(arguments)
    _json(
        recover_current_phase(
            paths,
            commands=_commands_manifest(arguments.commands_manifest),
            artifact_root=arguments.artifact_root,
            now_ms=time.time_ns() // 1_000_000,
        )
    )


def command_phase_barrier(arguments: argparse.Namespace) -> None:
    paths = _controller_preflight(arguments)
    _json(phase_barrier(paths))


def command_w6_dry_run(arguments: argparse.Namespace) -> None:
    paths = ControllerPaths.under(arguments.campaign_root)
    preflight_storage(
        configured_paths={"w6_dry_run_root": paths.root},
        expected_run_bytes=arguments.expected_run_bytes,
    )
    _json(run_isolated_dry_run(paths))


def _optional_compact_json(path: Path | None, *, label: str) -> dict[str, Any] | None:
    return None if path is None else read_compact_json(path, label=label)


def _publish_dashboard_status(arguments: argparse.Namespace) -> dict[str, Any]:
    _preflight(arguments)
    state = read_state(arguments.campaign_state_path)
    status = build_dashboard_status(
        DashboardStatusInputs(
            campaign_state=state,
            host_receipts=_optional_compact_json(
                arguments.host_receipts_path, label="host receipts"
            ),
            host_safety=(
                None
                if arguments.host_safety_path is None or not arguments.host_safety_path.exists()
                else read_compact_json(arguments.host_safety_path, label="host safety")
            ),
            training_progress=_optional_compact_json(
                arguments.training_progress_path, label="training progress"
            ),
            benchmark_aggregate=_optional_compact_json(
                arguments.benchmark_aggregate_path, label="benchmark aggregate"
            ),
            model_manifest=_optional_compact_json(
                arguments.model_manifest_path, label="model manifest"
            ),
            pool_manifest=_optional_compact_json(
                arguments.pool_manifest_path, label="pool manifest"
            ),
            stale_after_seconds=arguments.stale_after_seconds,
        ),
        updated_unix_ms=time.time_ns() // 1_000_000,
    )
    written_bytes = write_dashboard_status(arguments.output, status)
    return {
        "path": str(arguments.output),
        "bytes": written_bytes,
        "status": status,
    }


def command_publish_dashboard_status(arguments: argparse.Namespace) -> None:
    if not arguments.watch:
        _json(_publish_dashboard_status(arguments))
        return
    while True:
        published = _publish_dashboard_status(arguments)
        print(
            json.dumps(
                {
                    "published_unix_ms": published["status"]["updated_unix_ms"],
                    "canonical_path": published["path"],
                    "bytes": published["bytes"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        time.sleep(arguments.interval_seconds)


def _add_storage_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--expected-run-bytes",
        type=int,
        default=0,
        help="planned additional bytes; rejected above the 40 GiB run gate or free-space floor",
    )
    parser.add_argument(
        "--path",
        action="append",
        default=[],
        metavar="NAME=/ABSOLUTE/PATH",
        help="additional campaign output path to validate (repeatable)",
    )


def _add_controller_arguments(parser: argparse.ArgumentParser) -> None:
    _add_storage_arguments(parser)
    parser.add_argument("--campaign-root", type=Path, default=CAMPAIGN_ROOT)
    parser.add_argument(
        "--queue",
        type=Path,
        default=CAMPAIGN_ROOT / "control/research-queue-v1.json",
        help="authoritative john2 research queue",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=CAMPAIGN_ROOT / "control/research-experiments-v1.json",
    )
    parser.add_argument(
        "--isolated-queue",
        action="store_true",
        help="tests only: keep queue and ledger inside campaign-root",
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="prove the john2 disk contract")
    _add_storage_arguments(preflight)
    preflight.set_defaults(function=command_preflight)

    initialize = subparsers.add_parser(
        "init", help="create the registered layout, schemas, and initial durable state"
    )
    _add_storage_arguments(initialize)
    initialize.add_argument("--legacy-campaign-state-sha256")
    initialize.add_argument("--legacy-decision-head-sha256")
    initialize.add_argument("--authorization-sha256")
    initialize.set_defaults(function=command_init)

    show_state = subparsers.add_parser("show-state", help="validate and print durable state")
    _add_storage_arguments(show_state)
    show_state.set_defaults(function=command_show_state)

    transition = subparsers.add_parser(
        "transition", help="perform one legal compare-and-swap phase transition"
    )
    _add_storage_arguments(transition)
    transition.add_argument("phase", choices=[phase.value for phase in Phase])
    transition.add_argument("--reason", required=True)
    transition.add_argument("--generation-manifest-sha256")
    transition.add_argument("--candidate-checkpoint-sha256")
    transition.add_argument(
        "--completed-shard-host",
        action="append",
        choices=["john1", "john2", "john3"],
        default=None,
    )
    transition.set_defaults(function=command_transition)

    decision = subparsers.add_parser(
        "record-decision", help="append and fsync one operational decision"
    )
    _add_storage_arguments(decision)
    decision.add_argument("--actor", required=True)
    decision.add_argument("--evidence", action="append", required=True)
    decision.add_argument("--alternative", action="append", required=True)
    decision.add_argument("--chosen-action", required=True)
    decision.add_argument("--artifact", action="append", required=True)
    decision.add_argument("--rollback-path", required=True)
    decision.add_argument(
        "--kind",
        choices=["bounded-operational-adaptation", "scientific-contract-amendment"],
        default="bounded-operational-adaptation",
    )
    decision.add_argument("--authorization-sha256")
    decision.set_defaults(function=command_record_decision)

    verify = subparsers.add_parser(
        "verify-decisions", help="verify the entire append-only decision hash chain"
    )
    _add_storage_arguments(verify)
    verify.set_defaults(function=command_verify_decisions)

    advance = subparsers.add_parser(
        "advance-phase",
        help="cross one proven barrier and install typed tasks in the existing queue",
    )
    _add_controller_arguments(advance)
    advance.add_argument("--commands-manifest", type=Path, required=True)
    advance.add_argument(
        "--artifact-root",
        required=True,
        metavar="REMOTE_RELATIVE_PATH",
        help="canonical path relative to the authoritative john2 campaign root",
    )
    advance.add_argument("--reason", required=True)
    advance.add_argument("--at", required=True)
    advance.set_defaults(function=command_advance_phase)

    import_work = subparsers.add_parser(
        "import-work-receipt", help="validate and centralize one host-local packet receipt"
    )
    _add_controller_arguments(import_work)
    import_work.add_argument("--receipt", type=Path, required=True)
    import_work.set_defaults(function=command_import_work_receipt)

    import_benchmark = subparsers.add_parser(
        "import-benchmark-feed",
        help="stamp and upsert one receipt-bound deterministic benchmark ledger feed",
    )
    _add_controller_arguments(import_benchmark)
    import_benchmark.add_argument("--feed", type=Path, required=True)
    import_benchmark.add_argument("--aggregate-task-id", required=True)
    import_benchmark.add_argument("--expected-state-sha256", required=True)
    import_benchmark.set_defaults(function=command_import_benchmark_feed)

    reconcile_controller = subparsers.add_parser(
        "reconcile-controller",
        help="reconcile packet, queue, ledger, receipt, and dashboard projections",
    )
    _add_controller_arguments(reconcile_controller)
    reconcile_controller.set_defaults(function=command_reconcile_controller)

    recover = subparsers.add_parser(
        "recover-phase",
        help="repair an interrupted state-CAS, packet, queue, or ledger projection",
    )
    _add_controller_arguments(recover)
    recover.add_argument("--commands-manifest", type=Path, required=True)
    recover.add_argument(
        "--artifact-root",
        required=True,
        metavar="REMOTE_RELATIVE_PATH",
        help="canonical path relative to the authoritative john2 campaign root",
    )
    recover.set_defaults(function=command_recover_phase)

    barrier = subparsers.add_parser(
        "phase-barrier", help="prove every task and receipt required by the current phase"
    )
    _add_controller_arguments(barrier)
    barrier.set_defaults(function=command_phase_barrier)

    dry_run = subparsers.add_parser(
        "w6-dry-run",
        help="exercise every W6 transition shape with synthetic receipts on john2 storage",
    )
    _add_storage_arguments(dry_run)
    dry_run.add_argument("--campaign-root", type=Path, required=True)
    dry_run.set_defaults(function=command_w6_dry_run)

    dashboard = subparsers.add_parser(
        "publish-dashboard-status",
        help="atomically publish one compact R2-MAP dashboard mirror from explicit inputs",
    )
    _add_storage_arguments(dashboard)
    dashboard.add_argument("--campaign-state-path", type=Path, default=STATE_PATH)
    dashboard.add_argument("--host-receipts-path", type=Path)
    dashboard.add_argument(
        "--host-safety-path",
        type=Path,
        default=CAMPAIGN_ROOT / "control/host-safety.json",
    )
    dashboard.add_argument("--training-progress-path", type=Path)
    dashboard.add_argument("--benchmark-aggregate-path", type=Path)
    dashboard.add_argument("--model-manifest-path", type=Path)
    dashboard.add_argument("--pool-manifest-path", type=Path)
    dashboard.add_argument("--output", type=Path, default=DASHBOARD_STATUS_PATH)
    dashboard.add_argument("--stale-after-seconds", type=int, default=30)
    dashboard.add_argument("--watch", action="store_true")
    dashboard.add_argument("--interval-seconds", type=int, default=10, choices=range(5, 31))
    dashboard.set_defaults(function=command_publish_dashboard_status, skip_campaign_scan=True)
    return result


def main() -> int:
    arguments = parser().parse_args()
    try:
        arguments.function(arguments)
    except ContractError as error:
        print(f"r2-map controller refused operation: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
