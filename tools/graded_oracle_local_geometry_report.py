#!/usr/bin/env python3
"""Close ADR 0088 from frozen local-geometry replica evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

EXPERIMENT_ID = "complete-action-local-geometry-ranker-v1"
EXPECTED_ARCHITECTURE = "complete-action-graded-local-geometry-v1"
EXPECTED_KIND = "graded-oracle-local-geometry-ranking"
REPLICAS = (
    {
        "host": "john1",
        "seed": 2026061601,
        "run": "runs/john1-seed-2026061601",
        "origin_report": "runs/john1-seed-2026061601/validation-report-john1.json",
        "cross_host": "john3",
        "cross_report": (
            "cross-host/john1-seed-2026061601/validation-report-john3.json"
        ),
    },
    {
        "host": "john2",
        "seed": 2026061602,
        "run": "runs/john2-seed-2026061602",
        "origin_report": "runs/john2-seed-2026061602/validation-report-john2.json",
        "cross_host": "john3",
        "cross_report": (
            "cross-host/john2-seed-2026061602/validation-report-john3.json"
        ),
    },
    {
        "host": "john3",
        "seed": 2026061603,
        "run": "runs/john3-seed-2026061603",
        "origin_report": "runs/john3-seed-2026061603/validation-report-john3.json",
        "cross_host": "john1",
        "cross_report": (
            "cross-host/john3-seed-2026061603/validation-report-john1.json"
        ),
    },
)
HOST_ALIASES = {
    "Johns-Mac-mini": "john1",
    "john1": "john1",
    "john2": "john2",
    "john3": "john3",
}
TRAINING_EVENT_PATHS = (
    "events-john1.jsonl",
    "events-john2.jsonl",
    "events-john3.jsonl",
    "events-resume-john2.jsonl",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def blake3_file(path: Path) -> str:
    try:
        import blake3
    except ImportError as error:
        raise ValueError("blake3 is required to verify artifact identity") from error
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


def normalize_host(host: str) -> str:
    try:
        return HOST_ALIASES[host]
    except KeyError as error:
        raise ValueError(f"unknown Cascadia cluster host: {host}") from error


def select_replica(replicas: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply ADR 0088's exact frozen checkpoint-selection order."""
    if not replicas:
        raise ValueError("no local-geometry replicas were supplied")
    return min(
        replicas,
        key=lambda item: (
            item["selection_loss"],
            -item["top64_r4800_winner_recall"],
            item["r4800_residual_mae"],
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
    if not origin["performance"]["passed"] or not cross["performance"]["passed"]:
        raise ValueError(f"{origin_host} performance replay failed")
    if origin["scientific"]["test_split_opened"]:
        raise ValueError(f"{origin_host} origin report opened the sealed split")
    if cross["scientific"]["test_split_opened"]:
        raise ValueError(f"{origin_host} cross report opened the sealed split")


def validate_replica(
    experiment_root: Path,
    spec: dict[str, Any],
    inputs: dict[str, str],
) -> dict[str, Any]:
    run_dir = experiment_root / spec["run"]
    run_path = run_dir / "run.json"
    run = load_json(run_path, inputs, experiment_root)
    best = load_json(run_dir / "best.json", inputs, experiment_root)
    final = load_json(run_dir / "final-report.json", inputs, experiment_root)
    origin = load_json(
        experiment_root / spec["origin_report"],
        inputs,
        experiment_root,
    )
    cross = load_json(
        experiment_root / spec["cross_report"],
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
    model = run["training"]["model"]
    if model["architecture"] != EXPECTED_ARCHITECTURE:
        raise ValueError(f"{spec['host']} architecture drifted")
    if model["relation_schema"] != "active-board-local-13-v1":
        raise ValueError(f"{spec['host']} relation schema drifted")
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
    confidence = scientific["confidence"]
    return {
        "host": spec["host"],
        "seed": seed,
        "checkpoint": best["checkpoint"],
        "selection_loss": best["selection_loss"],
        "top64_r4800_winner_recall": metrics["top64_r4800_winner_recall"],
        "r4800_residual_mae": metrics["r4800_residual_mae"],
        "model_blake3": scientific["model_blake3"],
        "checkpoint_manifest_blake3": scientific["checkpoint_manifest_blake3"],
        "source_run_manifest_blake3": scientific["source_run_manifest_blake3"],
        "scientific_blake3": origin["scientific_blake3"],
        "cross_host": spec["cross_host"],
        "origin_performance": origin["performance"],
        "cross_performance": cross["performance"],
        "metrics": metrics,
        "confidence": confidence,
        "gates": origin["gates"],
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


def load_jsonl(path: Path, inputs: dict[str, str], root: Path) -> list[dict[str, Any]]:
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
    replicas: list[dict[str, Any]],
    inputs: dict[str, str],
) -> dict[str, Any]:
    events = []
    for relative in TRAINING_EVENT_PATHS:
        events.extend(load_jsonl(experiment_root / relative, inputs, experiment_root))

    initial_starts = {}
    queue_seconds = []
    completed = []
    incomplete = []
    for event in events:
        name = str(event.get("name", ""))
        if event.get("event") == "started":
            queue_seconds.append(float(event.get("queued_seconds", 0.0)))
            if name.startswith("adr0088-train-"):
                initial_starts[normalize_host(str(event["host"]))] = float(
                    event["started_unix_seconds"]
                )
        elif event.get("event") == "finished":
            item = {
                "host": normalize_host(str(event["host"])),
                "name": name,
                "elapsed_seconds": float(event["elapsed_seconds"]),
                "return_code": int(event["return_code"]),
            }
            completed.append(item)

    for event in events:
        if event.get("event") != "started":
            continue
        key = (normalize_host(str(event["host"])), str(event.get("name", "")))
        if not any((item["host"], item["name"]) == key for item in completed):
            incomplete.append({"host": key[0], "name": key[1]})

    if set(initial_starts) != {"john1", "john2", "john3"}:
        raise ValueError("initial three-host training launch evidence is incomplete")
    launch_skew = max(initial_starts.values()) - min(initial_starts.values())
    training = {
        item["host"]: {
            "seed": item["seed"],
            "epochs": item["epochs"],
            "global_step": item["global_step"],
            "elapsed_seconds": item["training_elapsed_seconds"],
            "stopped_reason": item["stopped_reason"],
        }
        for item in replicas
    }
    return {
        "initial_training_start_unix_seconds": initial_starts,
        "initial_training_launch_skew_seconds": launch_skew,
        "maximum_lock_queue_seconds": max(queue_seconds, default=0.0),
        "training_by_host": training,
        "completed_wrapper_jobs": completed,
        "incomplete_wrapper_jobs": incomplete,
        "infrastructure_retries": [
            {
                "host": "john2",
                "reason": (
                    "the initial SSH wrapper disconnected after checkpoints were "
                    "written; the frozen run resumed from its latest checkpoint"
                ),
                "scientific_contract_changed": False,
            }
        ],
        "all_three_hosts_started_concurrently": launch_skew < 300.0,
    }


def verify_sealed_state(
    experiment_root: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    authorization_paths = list(experiment_root.rglob("test-authorization.json"))
    forbidden_outputs = []
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


def build_report(experiment_root: Path) -> dict[str, Any]:
    experiment_root = experiment_root.resolve()
    inputs: dict[str, str] = {}
    manifest = load_json(experiment_root / "manifest.json", inputs, experiment_root)
    if manifest["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("experiment manifest identity drifted")

    replicas = [
        validate_replica(experiment_root, spec, inputs) for spec in REPLICAS
    ]
    source_hashes = {item["source_v2_blake3"] for item in replicas}
    train_hashes = {item["train_manifest_blake3"] for item in replicas}
    validation_hashes = {item["validation_manifest_blake3"] for item in replicas}
    if len(source_hashes) != 1:
        raise ValueError("replica source identities differ")
    if len(train_hashes) != 1 or len(validation_hashes) != 1:
        raise ValueError("replica dataset identities differ")

    selected = select_replica(replicas)
    quality_gates = {
        key: value
        for key, value in selected["gates"].items()
        if not key.endswith("_performance_passed")
    }
    quality_passed = all(quality_gates.values())
    failed_quality_gates = sorted(
        key for key, value in quality_gates.items() if not value
    )
    all_performance = {
        f"{item['host']}-origin": item["origin_performance"] for item in replicas
    }
    all_performance.update(
        {
            f"{item['host']}-cross-on-{item['cross_host']}": item[
                "cross_performance"
            ]
            for item in replicas
        }
    )
    performance_passed = all(item["passed"] for item in all_performance.values())
    sealed = verify_sealed_state(experiment_root, manifest)
    execution = summarize_execution(experiment_root, replicas, inputs)

    metrics = selected["metrics"]
    top64 = selected["confidence"]["overall"]["ranking"]["model"]["top64"]
    screen_regret = float(metrics["screen_mean_top64_retained_r4800_regret"])
    learned_regret = float(metrics["mean_top64_retained_r4800_regret"])
    screen_recall = float(metrics["screen_top64_r4800_winner_recall"])
    learned_recall = float(metrics["top64_r4800_winner_recall"])
    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "rejected_on_validation",
        "protocol_status": {
            "validation": "failed",
            "sealed_test": "closed_unopened",
            "gameplay": "closed_unopened",
        },
        "replica_selection": {
            "objective": [
                "lowest validation retained mean R4800 regret",
                "higher top-64 R4800-winner recall",
                "lower R4800 residual MAE",
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
                        "checkpoint",
                        "selection_loss",
                        "top64_r4800_winner_recall",
                        "r4800_residual_mae",
                        "model_blake3",
                        "checkpoint_manifest_blake3",
                        "scientific_blake3",
                        "cross_host",
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
            "confidence_top64": top64,
            "phase_top64": {
                phase: selected["confidence"]["phases"][phase]["ranking"]["model"][
                    "top64"
                ]
                for phase in ("early", "middle", "late")
            },
            "subset_top64": {
                subset: selected["confidence"]["subsets"][subset]["ranking"]["model"][
                    "top64"
                ]
                for subset in (
                    "nature_token_available",
                    "independent_draft_winner",
                )
            },
            "quality_gates": quality_gates,
            "failed_quality_gates": failed_quality_gates,
            "quality_passed": quality_passed,
        },
        "cross_host_portability": {
            "all_origin_cross_scientific_digests_identical": True,
            "all_origin_cross_scientific_payloads_identical": True,
            "performance_by_replay": all_performance,
            "performance_passed": performance_passed,
        },
        "integrity": {
            "all_replica_sources_identical": len(source_hashes) == 1,
            "all_replica_train_datasets_identical": len(train_hashes) == 1,
            "all_replica_validation_datasets_identical": len(validation_hashes) == 1,
            "all_actions_scored_once": metrics["all_candidates_scored_once"],
            "all_groups_scored_once": metrics["all_groups_scored_once"],
            "all_scores_finite": metrics["all_scores_finite"],
            "relation_schema": "active-board-local-13-v1",
            "sealed_test_opened": False,
        },
        "execution": execution,
        "sealed_test": sealed,
        "diagnosis": {
            "screen_top64_r4800_winner_recall": screen_recall,
            "learned_top64_r4800_winner_recall": learned_recall,
            "winner_recall_change_percentage_points": 100.0
            * (learned_recall - screen_recall),
            "screen_mean_top64_retained_r4800_regret": screen_regret,
            "learned_mean_top64_retained_r4800_regret": learned_regret,
            "retained_regret_reduction_points": screen_regret - learned_regret,
            "retained_regret_reduction_fraction": (
                (screen_regret - learned_regret) / screen_regret
            ),
            "conclusion": (
                "Explicit rotation-canonical local geometry reduced retained regret "
                "but did not materially solve exact complete-action winner recovery. "
                "The representation treatment is rejected before sealed test or "
                "gameplay."
            ),
        },
        "passed": (
            quality_passed
            and performance_passed
            and sealed["passed"]
            and execution["all_three_hosts_started_concurrently"]
        ),
        "input_sha256": dict(sorted(inputs.items())),
    }
    if report["passed"]:
        raise ValueError("rejection reporter unexpectedly observed a validation pass")
    return report


def render_markdown(report: dict[str, Any]) -> str:
    selected = report["replica_selection"]
    validation = report["selected_validation"]
    metrics = validation["metrics"]
    top64 = validation["confidence_top64"]
    diagnosis = report["diagnosis"]
    lines = [
        "# Complete-Action Local-Geometry Ranker V1 Rejection",
        "",
        "Status: **rejected on validation; sealed test and gameplay closed unopened**",
        "",
        "## Verdict",
        "",
        "All three preregistered MLX replicas completed, each selected checkpoint was",
        "replayed on another Mac, and all origin/cross scientific payloads were",
        "bit-identical. The john2 replica won the frozen selection order and passed",
        "every integrity, regret, throughput, latency, memory, and swap gate. It",
        "nevertheless missed every overall winner-recovery gate, every phase recall",
        "and coverage gate, and both subset recall gates. ADR 0088 is rejected without",
        "opening sealed test data or gameplay.",
        "",
        "## Replica Selection",
        "",
        "| Train host | Seed | Cross host | Epochs | Retained regret | Top-64 recall | R4800 MAE |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for replica in selected["replicas"]:
        lines.append(
            "| {host} | {seed} | {cross_host} | {epochs} | {selection_loss:.6f} | "
            "{top64_r4800_winner_recall:.2%} | {r4800_residual_mae:.6f} |".format(
                **replica
            )
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
            "| Gate | Required | Observed | Result |",
            "|---|---:|---:|---|",
            "| Overall exact winner recall | >98% | {:.2%} | Fail |".format(
                metrics["top64_r4800_winner_recall"]
            ),
            "| Overall confidence-set coverage | >=99% | {:.2%} | Fail |".format(
                top64["confidence_set_coverage_95"]
            ),
            "| Distinguishable-winner recall | >=98% | {:.2%} | Fail |".format(
                top64["distinguishable_winner_recall"]
            ),
            "| Mean retained R4800 regret | <0.15 | {:.6f} | Pass |".format(
                metrics["mean_top64_retained_r4800_regret"]
            ),
        ]
    )
    for phase in ("early", "middle", "late"):
        phase_metrics = validation["phase_top64"][phase]
        lines.append(
            "| {} exact recall | >=97% | {:.2%} | Fail |".format(
                phase.capitalize(),
                phase_metrics["exact_winner_recall"],
            )
        )
        lines.append(
            "| {} confidence coverage | >=98% | {:.2%} | Fail |".format(
                phase.capitalize(),
                phase_metrics["confidence_set_coverage_95"],
            )
        )
    for label, subset in (
        ("Nature-token", "nature_token_available"),
        ("Independent-draft", "independent_draft_winner"),
    ):
        subset_metrics = validation["subset_top64"][subset]
        lines.append(
            "| {} exact recall | >=95% | {:.2%} | Fail |".format(
                label,
                subset_metrics["exact_winner_recall"],
            )
        )
    lines.extend(
        [
            "",
            "Every phase and subset retained-regret gate passed.",
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
            "## Diagnosis",
            "",
            "- Retained regret improved from {:.6f} to {:.6f}, a {:.1%} reduction.".format(
                diagnosis["screen_mean_top64_retained_r4800_regret"],
                diagnosis["learned_mean_top64_retained_r4800_regret"],
                diagnosis["retained_regret_reduction_fraction"],
            ),
            "- Exact-winner recall improved only from {:.2%} to {:.2%}.".format(
                diagnosis["screen_top64_r4800_winner_recall"],
                diagnosis["learned_top64_r4800_winner_recall"],
            ),
            "- The treatment is portable and fast; the failure is scientific, not an",
            "  execution artifact.",
            "- Local geometry alone is not the missing mechanism. The observed R4800",
            "  exceptions motivate hard retention of public champion/frontier anchors",
            "  while learning only the nonfrontier fill.",
            "",
            "## Execution",
            "",
            "- Initial three-host training launch skew: {:.3f} seconds.".format(
                report["execution"]["initial_training_launch_skew_seconds"]
            ),
            "- Maximum host-lock queue: {:.6f} seconds.".format(
                report["execution"]["maximum_lock_queue_seconds"]
            ),
            "- john2 resumed once after its SSH wrapper disconnected; checkpoint and",
            "  scientific contracts were unchanged.",
            "- All six origin/cross performance replays passed.",
            "",
            "## Protocol Closure",
            "",
            "- Test authorization file: absent.",
            "- Sealed-test or gameplay output: absent.",
            "- Test groups read by this reporter: no.",
            "- New teacher compute: not used.",
            "- K2048: not opened.",
            "",
            "Machine-readable evidence is in",
            "`docs/v2/reports/complete-action-local-geometry-ranker-v1-rejection.json`.",
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
