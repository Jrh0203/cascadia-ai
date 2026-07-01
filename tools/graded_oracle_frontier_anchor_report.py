#!/usr/bin/env python3
"""Close ADR 0089 from frozen frontier-anchored replica evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import blake3

EXPERIMENT_ID = "complete-action-frontier-anchored-set-ranker-v1"
EXPECTED_ARCHITECTURE = "complete-action-graded-residual-v1"
EXPECTED_KIND = "graded-oracle-frontier-anchored-ranking"
REPLICAS = (
    {
        "host": "john1",
        "seed": 2026061601,
        "cross_host": "john3",
    },
    {
        "host": "john2",
        "seed": 2026061602,
        "cross_host": "john4",
    },
    {
        "host": "john3",
        "seed": 2026061603,
        "cross_host": "john1",
    },
    {
        "host": "john4",
        "seed": 2026061604,
        "cross_host": "john2",
    },
)
HOST_ALIASES = {
    "Johns-Mac-mini": "john1",
    "john1": "john1",
    "john2": "john2",
    "john3": "john3",
    "john4": "john4",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def blake3_file(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path, inputs: dict[str, str], root: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read frozen artifact {path}: {error}") from error
    inputs[str(path.relative_to(root))] = sha256_file(path)
    return value


def load_protocol_manifest(
    path: Path,
    inputs: dict[str, str],
    root: Path,
) -> dict[str, Any]:
    """Load a manifest while hashing only its immutable scientific protocol."""
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read frozen artifact {path}: {error}") from error
    protocol = json.loads(json.dumps(manifest))
    protocol.pop("status", None)
    protocol.pop("closure", None)
    if isinstance(protocol.get("training"), dict):
        protocol["training"].pop("status", None)
    payload = json.dumps(
        protocol,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    relative = str(path.relative_to(root))
    inputs[f"{relative}#immutable-protocol"] = hashlib.sha256(payload).hexdigest()
    return manifest


def normalize_host(host: str) -> str:
    try:
        return HOST_ALIASES[host]
    except KeyError as error:
        raise ValueError(f"unknown Cascadia cluster host: {host}") from error


def select_replica(replicas: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply ADR 0089's exact frozen checkpoint-selection order."""
    if not replicas:
        raise ValueError("no frontier-anchored replicas were supplied")
    return min(
        replicas,
        key=lambda item: (
            item["selection_loss"],
            -item["top64_confidence_set_coverage_95"],
            item["mean_top64_retained_r4800_regret"],
            item["seed"],
        ),
    )


def validate_report_pair(
    origin: dict[str, Any],
    cross: dict[str, Any],
    *,
    origin_host: str,
    cross_host: str,
) -> None:
    if normalize_host(str(origin["host"])) != origin_host:
        raise ValueError(f"{origin_host} origin report host drifted")
    if normalize_host(str(cross["host"])) != cross_host:
        raise ValueError(f"{origin_host} cross report host drifted")
    if origin["scientific_blake3"] != cross["scientific_blake3"]:
        raise ValueError(f"{origin_host} cross-host scientific digest drifted")
    if origin["scientific"] != cross["scientific"]:
        raise ValueError(f"{origin_host} cross-host scientific payload drifted")
    for label, report in (("origin", origin), ("cross", cross)):
        if report["scientific"]["test_split_opened"]:
            raise ValueError(f"{origin_host} {label} report opened the sealed split")
        if "passed" not in report["performance"]:
            raise ValueError(f"{origin_host} {label} performance result is incomplete")


def _replica_paths(
    experiment_root: Path,
    spec: dict[str, Any],
) -> tuple[Path, Path]:
    run_name = f"{spec['host']}-seed-{spec['seed']}"
    return (
        experiment_root / "runs" / run_name,
        experiment_root / "cross-host" / run_name,
    )


def validate_replica(
    experiment_root: Path,
    spec: dict[str, Any],
    inputs: dict[str, str],
) -> dict[str, Any]:
    run_dir, cross_dir = _replica_paths(experiment_root, spec)
    run_path = run_dir / "run.json"
    run = load_json(run_path, inputs, experiment_root)
    best = load_json(run_dir / "best.json", inputs, experiment_root)
    final = load_json(run_dir / "final-report.json", inputs, experiment_root)
    origin = load_json(
        run_dir / f"validation-report-{spec['host']}.json",
        inputs,
        experiment_root,
    )
    cross = load_json(
        cross_dir / f"validation-report-{spec['cross_host']}.json",
        inputs,
        experiment_root,
    )
    validate_report_pair(
        origin,
        cross,
        origin_host=spec["host"],
        cross_host=spec["cross_host"],
    )

    seed = int(run["training"]["seed"])
    if seed != spec["seed"]:
        raise ValueError(f"{spec['host']} training seed drifted: {seed}")
    if run["kind"] != EXPECTED_KIND:
        raise ValueError(f"{spec['host']} run kind drifted")
    if run["training"]["model"]["architecture"] != EXPECTED_ARCHITECTURE:
        raise ValueError(f"{spec['host']} architecture drifted")

    scientific = origin["scientific"]
    if scientific["experiment_id"] != EXPERIMENT_ID:
        raise ValueError(f"{spec['host']} report experiment identity drifted")
    if best["checkpoint"] != scientific["checkpoint"]:
        raise ValueError(f"{spec['host']} selected checkpoint drifted")
    if best["validation"] != scientific["metrics"]:
        raise ValueError(f"{spec['host']} selected metrics drifted")
    if final["best_ranking_loss"] != best["selection_loss"]:
        raise ValueError(f"{spec['host']} best selection loss drifted")
    if scientific["source_run_manifest_blake3"] != blake3_file(run_path):
        raise ValueError(f"{spec['host']} source run manifest drifted")
    if (
        scientific["dataset"]["manifest_blake3"]
        != run["datasets"]["validation_manifest_blake3"]
    ):
        raise ValueError(f"{spec['host']} validation dataset drifted")

    metrics = scientific["metrics"]
    return {
        "host": spec["host"],
        "seed": seed,
        "cross_host": spec["cross_host"],
        "checkpoint": best["checkpoint"],
        "selection_loss": best["selection_loss"],
        "top64_r4800_winner_recall": metrics["top64_r4800_winner_recall"],
        "top64_confidence_set_coverage_95": metrics[
            "top64_confidence_set_coverage_95"
        ],
        "top64_distinguishable_winner_recall": metrics[
            "top64_distinguishable_winner_recall"
        ],
        "mean_top64_retained_r4800_regret": metrics[
            "mean_top64_retained_r4800_regret"
        ],
        "model_blake3": scientific["model_blake3"],
        "checkpoint_manifest_blake3": scientific["checkpoint_manifest_blake3"],
        "source_run_manifest_blake3": scientific["source_run_manifest_blake3"],
        "scientific_blake3": origin["scientific_blake3"],
        "metrics": metrics,
        "quality_gates": {
            key: value
            for key, value in origin["gates"].items()
            if not key.endswith("_performance_passed")
        },
        "origin_performance": origin["performance"],
        "cross_performance": cross["performance"],
        "epochs": final["epochs"],
        "global_step": final["global_step"],
        "training_elapsed_seconds": final["elapsed_seconds"],
        "stopped_reason": final["stopped_reason"],
        "source_v2_blake3": run["source"]["v2_source_blake3"],
        "train_manifest_blake3": run["datasets"]["train_manifest_blake3"],
        "validation_manifest_blake3": run["datasets"][
            "validation_manifest_blake3"
        ],
    }


def verify_sealed_state(
    experiment_root: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    authorization_paths = list(experiment_root.rglob("test-authorization.json"))
    forbidden_outputs: list[Path] = []
    for pattern in ("test-report*.json", "sealed-test*.json", "gameplay-report*.json"):
        forbidden_outputs.extend(experiment_root.rglob(pattern))
    passed = (
        manifest["datasets"]["test"]["model_access"] == "sealed"
        and not manifest["datasets"]["test"]["opened"]
        and not authorization_paths
        and not forbidden_outputs
    )
    return {
        "manifest_model_access": manifest["datasets"]["test"]["model_access"],
        "manifest_opened": manifest["datasets"]["test"]["opened"],
        "test_authorization_exists": bool(authorization_paths),
        "test_or_gameplay_output_exists": bool(forbidden_outputs),
        "test_groups_read_by_reporter": False,
        "passed": passed,
    }


def load_jsonl(
    path: Path,
    inputs: dict[str, str],
    root: Path,
) -> list[dict[str, Any]]:
    inputs[str(path.relative_to(root))] = sha256_file(path)
    events = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
        if not isinstance(event, dict):
            raise ValueError(f"non-object JSONL event at {path}:{line_number}")
        events.append(event)
    return events


def summarize_execution(
    experiment_root: Path,
    manifest: dict[str, Any],
    inputs: dict[str, str],
) -> dict[str, Any]:
    event_paths = sorted(experiment_root.glob("events-*.jsonl"))
    events = [
        event
        for path in event_paths
        for event in load_jsonl(path, inputs, experiment_root)
    ]
    by_host: dict[str, dict[str, Any]] = {}
    for host in ("john1", "john2", "john3", "john4"):
        host_events = [
            event
            for event in events
            if normalize_host(str(event.get("host", ""))) == host
        ]
        starts = [event for event in host_events if event.get("event") == "started"]
        finishes = [event for event in host_events if event.get("event") == "finished"]
        launch_failures = [
            event for event in host_events if event.get("event") == "launch-failed"
        ]
        completed = [event for event in finishes if int(event["return_code"]) == 0]
        failed = [event for event in finishes if int(event["return_code"]) != 0]
        if not starts or not finishes:
            raise ValueError(f"{host} execution event evidence is incomplete")
        assigned_start = min(float(event["queued_unix_seconds"]) for event in starts)
        assigned_end = max(float(event["ended_unix_seconds"]) for event in finishes)
        by_host[host] = {
            "assigned_wall_seconds": assigned_end - assigned_start,
            "productive_wall_seconds": sum(
                float(event["elapsed_seconds"]) for event in completed
            ),
            "idle_with_work_queued_seconds": sum(
                float(event.get("queued_seconds", 0.0)) for event in starts
            ),
            "jobs_started": len(starts),
            "jobs_completed": len(completed),
            "jobs_failed": len(failed) + len(launch_failures),
            "retries": max(0, len(starts) - len({str(event["name"]) for event in starts})),
            "work_completed": sorted(str(event["name"]) for event in completed),
        }
    aggregate_productive = sum(
        float(values["productive_wall_seconds"]) for values in by_host.values()
    )
    aggregate_assigned = sum(
        float(values["assigned_wall_seconds"]) for values in by_host.values()
    )
    return {
        "training_start_unix_seconds": manifest["training"][
            "launched_at_unix_seconds"
        ],
        "training_launch_skew_seconds": manifest["training"]["launch_skew_seconds"],
        "all_four_hosts_started_concurrently": (
            float(manifest["training"]["launch_skew_seconds"]) < 300.0
        ),
        "by_host": by_host,
        "aggregate_assigned_wall_seconds": aggregate_assigned,
        "aggregate_productive_wall_seconds": aggregate_productive,
        "aggregate_productive_fraction": (
            aggregate_productive / aggregate_assigned
            if aggregate_assigned > 0.0
            else 0.0
        ),
        "total_jobs_completed": sum(
            int(values["jobs_completed"]) for values in by_host.values()
        ),
        "total_jobs_failed": sum(
            int(values["jobs_failed"]) for values in by_host.values()
        ),
        "total_retries": sum(int(values["retries"]) for values in by_host.values()),
        "intentional_idle_reasons": {
            "campaign": (
                "ADR 0089 preregistered four training replicas before the cluster "
                "policy changed to single-host MLX pilots plus independent work."
            ),
            "john1": (
                "After its origin replay, john1 waited only for john3's frozen "
                "checkpoint before running the assigned cross replay."
            ),
            "john2": (
                "john2 remained occupied by the longest training replica, then "
                "immediately ran its origin replay and john4 cross replay."
            ),
            "john3": (
                "After training, john3 immediately ran its origin replay and the "
                "assigned john1 cross replay."
            ),
            "john4": (
                "After its early-stopped origin replay, john4 had no compatible "
                "frozen cross job until john2 training completed."
            ),
        },
    }


def build_report(experiment_root: Path) -> dict[str, Any]:
    experiment_root = experiment_root.resolve()
    inputs: dict[str, str] = {}
    manifest = load_protocol_manifest(
        experiment_root / "manifest.json",
        inputs,
        experiment_root,
    )
    if manifest["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("experiment manifest identity drifted")

    replicas = [
        validate_replica(experiment_root, spec, inputs) for spec in REPLICAS
    ]
    train_hashes = {item["train_manifest_blake3"] for item in replicas}
    validation_hashes = {item["validation_manifest_blake3"] for item in replicas}
    if len(train_hashes) != 1 or len(validation_hashes) != 1:
        raise ValueError("replica dataset identities differ")
    source_identities = {
        host: load_json(
            experiment_root / "source-identity" / f"{host}.json",
            inputs,
            experiment_root,
        )
        for host in ("john1", "john2", "john3", "john4")
    }
    mlx_source_bundles = {
        identity["bundle_sha256"] for identity in source_identities.values()
    }
    mlx_source_entries = {
        json.dumps(identity["entries"], sort_keys=True)
        for identity in source_identities.values()
    }
    if len(mlx_source_bundles) != 1 or len(mlx_source_entries) != 1:
        raise ValueError("replica MLX runtime source identities differ")
    for host, identity in source_identities.items():
        if normalize_host(str(identity["host"])) != host:
            raise ValueError(f"{host} MLX source identity host drifted")
        if identity["identity_kind"] != "complete-mlx-runtime-source-v1":
            raise ValueError(f"{host} MLX source identity kind drifted")

    selected = select_replica(replicas)
    failed_quality_gates = sorted(
        key for key, value in selected["quality_gates"].items() if not value
    )
    quality_passed = not failed_quality_gates
    performance_by_replay = {
        f"{item['host']}-origin": item["origin_performance"] for item in replicas
    }
    performance_by_replay.update(
        {
            f"{item['host']}-cross-on-{item['cross_host']}": item[
                "cross_performance"
            ]
            for item in replicas
        }
    )
    performance_passed = all(
        replay["passed"] for replay in performance_by_replay.values()
    )
    sealed = verify_sealed_state(experiment_root, manifest)
    execution = summarize_execution(experiment_root, manifest, inputs)
    validation_passed = quality_passed and performance_passed and sealed["passed"]
    status = "validation_passed" if validation_passed else "rejected_on_validation"
    initial = load_json(
        experiment_root
        / "runs"
        / f"{selected['host']}-seed-{selected['seed']}"
        / "initial-validation.json",
        inputs,
        experiment_root,
    )["validation"]
    metrics = selected["metrics"]

    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "protocol_status": {
            "validation": "passed" if validation_passed else "failed",
            "sealed_test": "authorized_by_separate_adr" if validation_passed else "closed_unopened",
            "gameplay": "closed_unopened",
        },
        "replica_selection": {
            "objective": [
                "lowest top-64 R4800-winner miss rate",
                "higher top-64 R4800 confidence-set coverage",
                "lower retained mean top-64 R4800 regret",
                "lower training seed",
            ],
            "selected_host": selected["host"],
            "selected_seed": selected["seed"],
            "selected_checkpoint": selected["checkpoint"],
            "selected_model_blake3": selected["model_blake3"],
            "replicas": [
                {
                    key: item[key]
                    for key in (
                        "host",
                        "seed",
                        "cross_host",
                        "checkpoint",
                        "selection_loss",
                        "top64_r4800_winner_recall",
                        "top64_confidence_set_coverage_95",
                        "top64_distinguishable_winner_recall",
                        "mean_top64_retained_r4800_regret",
                        "model_blake3",
                        "checkpoint_manifest_blake3",
                        "scientific_blake3",
                        "epochs",
                        "global_step",
                        "training_elapsed_seconds",
                        "stopped_reason",
                    )
                }
                for item in replicas
            ],
        },
        "selected_validation": {
            "metrics": metrics,
            "quality_gates": selected["quality_gates"],
            "failed_quality_gates": failed_quality_gates,
            "quality_passed": quality_passed,
        },
        "cross_host_portability": {
            "all_origin_cross_scientific_digests_identical": True,
            "all_origin_cross_scientific_payloads_identical": True,
            "performance_by_replay": performance_by_replay,
            "performance_passed": performance_passed,
        },
        "integrity": {
            "all_replica_mlx_runtime_sources_identical": True,
            "mlx_runtime_source_bundle_sha256": next(iter(mlx_source_bundles)),
            "mlx_runtime_source_files": source_identities["john1"]["files"],
            "broad_repository_source_blake3_by_host": {
                item["host"]: item["source_v2_blake3"] for item in replicas
            },
            "broad_repository_digest_scope_note": (
                "The legacy aggregate also covers unrelated Rust and web paths "
                "that are intentionally absent from worker-only MLX directories."
            ),
            "all_replica_train_datasets_identical": True,
            "all_replica_validation_datasets_identical": True,
            "all_actions_scored_once": metrics["all_candidates_scored_once"],
            "all_groups_scored_once": metrics["all_groups_scored_once"],
            "all_scores_finite": metrics["all_scores_finite"],
            "proposal_width": metrics["proposal_width"],
            "sealed_test_opened": False,
        },
        "execution": execution,
        "sealed_test": sealed,
        "diagnosis": {
            "initial_top64_r4800_winner_recall": initial[
                "top64_r4800_winner_recall"
            ],
            "selected_top64_r4800_winner_recall": metrics[
                "top64_r4800_winner_recall"
            ],
            "winner_recall_change_percentage_points": 100.0
            * (
                metrics["top64_r4800_winner_recall"]
                - initial["top64_r4800_winner_recall"]
            ),
            "initial_confidence_set_coverage_95": initial[
                "top64_confidence_set_coverage_95"
            ],
            "selected_confidence_set_coverage_95": metrics[
                "top64_confidence_set_coverage_95"
            ],
            "initial_mean_top64_retained_r4800_regret": initial[
                "mean_top64_retained_r4800_regret"
            ],
            "selected_mean_top64_retained_r4800_regret": metrics[
                "mean_top64_retained_r4800_regret"
            ],
            "selected_target_positive_recall": metrics["target_positive_recall"],
            "selected_target_set_exact_fraction": metrics[
                "target_set_exact_fraction"
            ],
            "conclusion": (
                "Hard frontier retention is mechanically sound, but the unchanged "
                "observable ranker learned only a small fraction of the required "
                "nonfrontier R1200 fill. The next treatment must change the "
                "observable learning representation or optimization mechanism, "
                "not repeat four duplicate replicas of this objective."
            ),
        },
        "passed": validation_passed,
        "input_sha256": dict(sorted(inputs.items())),
    }


def render_markdown(report: dict[str, Any]) -> str:
    selected = report["replica_selection"]
    validation = report["selected_validation"]
    metrics = validation["metrics"]
    passed = report["passed"]
    verdict = (
        "passed validation; sealed test requires a separate frozen ADR"
        if passed
        else "rejected on validation; sealed test and gameplay closed unopened"
    )
    lines = [
        "# Complete-Action Frontier-Anchored Set Ranker V1",
        "",
        f"Status: **{verdict}**",
        "",
        "## Verdict",
        "",
        (
            "All four preregistered replicas and all origin/cross-host replays "
            "completed with bit-identical scientific payloads. "
            + (
                "Every frozen validation and performance gate passed."
                if passed
                else (
                    f"The selected replica failed "
                    f"{len(validation['failed_quality_gates'])} frozen quality gates."
                )
            )
        ),
        "",
        "## Replica Selection",
        "",
        "| Train host | Seed | Cross host | Epochs | Winner recall | Coverage | Regret |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for replica in selected["replicas"]:
        lines.append(
            "| {host} | {seed} | {cross_host} | {epochs} | "
            "{top64_r4800_winner_recall:.2%} | "
            "{top64_confidence_set_coverage_95:.2%} | "
            "{mean_top64_retained_r4800_regret:.6f} |".format(**replica)
        )
    lines.extend(
        [
            "",
            "Selected checkpoint: `{}` from {}.".format(
                selected["selected_checkpoint"],
                selected["selected_host"],
            ),
            "",
            "## Frozen Gates",
            "",
            "| Metric | Required | Observed |",
            "|---|---:|---:|",
            "| Exact winner recall | >98% | {:.2%} |".format(
                metrics["top64_r4800_winner_recall"]
            ),
            "| Confidence-set coverage | >=99% | {:.2%} |".format(
                metrics["top64_confidence_set_coverage_95"]
            ),
            "| Distinguishable-winner recall | >=98% | {:.2%} |".format(
                metrics["top64_distinguishable_winner_recall"]
            ),
            "| Mean retained regret | <0.15 | {:.6f} |".format(
                metrics["mean_top64_retained_r4800_regret"]
            ),
            "| Target-positive recall | diagnostic | {:.2%} |".format(
                metrics["target_positive_recall"]
            ),
            "| Exact target-set recovery | diagnostic | {:.2%} |".format(
                metrics["target_set_exact_fraction"]
            ),
            "",
            "Failed gates:",
        ]
    )
    if validation["failed_quality_gates"]:
        lines.extend(f"- `{gate}`" for gate in validation["failed_quality_gates"])
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Performance",
            "",
            "| Replay | Action scores/s | P99 decision ms | Peak RSS MiB | Swap delta |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for replay, performance in report["cross_host_portability"][
        "performance_by_replay"
    ].items():
        lines.append(
            "| {} | {:,.0f} | {:.2f} | {:.1f} | {} |".format(
                replay,
                performance["action_scores_per_second"],
                performance["p99_decision_milliseconds"],
                performance["peak_process_rss_bytes"] / (1024 * 1024),
                performance["system_swap_delta_bytes"],
            )
        )
    lines.extend(
        [
            "",
            "## Cluster Execution",
            "",
            "| Host | Assigned s | Productive s | Queued idle s | Completed | Failed |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for host, execution in report["execution"]["by_host"].items():
        lines.append(
            "| {} | {:.1f} | {:.1f} | {:.3f} | {} | {} |".format(
                host,
                execution["assigned_wall_seconds"],
                execution["productive_wall_seconds"],
                execution["idle_with_work_queued_seconds"],
                execution["jobs_completed"],
                execution["jobs_failed"],
            )
        )
    lines.extend(
        [
            "",
            "The four training replicas were frozen before the cluster policy moved",
            "to single-host MLX pilots plus independent experiments. John 4's longer",
            "assigned idle interval was a dependency wait for John 2's frozen cross",
            "checkpoint, not unqueued compatible work.",
            "",
            "## Diagnosis",
            "",
            "- Hard frontier retention improved exact recall by {:.2f} percentage points.".format(
                report["diagnosis"]["winner_recall_change_percentage_points"]
            ),
            (
                "- The selected model recovered only {:.2%} of target-positive "
                "nonfrontier slots and no complete target sets."
            ).format(report["diagnosis"]["selected_target_positive_recall"]),
            "- The treatment was portable and fast; the failure is scientific.",
            "- Future discovery should train one MLX pilot while the other Macs run",
            "  independent representation or optimization hypotheses.",
            "",
            "## Protocol Boundary",
            "",
            "- Origin/cross scientific payloads: bit-identical.",
            "- Test authorization file: absent.",
            "- Sealed-test or gameplay output: absent.",
            "- Test groups read by this reporter: no.",
            "",
            "Machine-readable evidence is in the adjacent JSON report.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args.experiment_root)
    write_atomic(args.output_json, json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_atomic(args.output_markdown, render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
