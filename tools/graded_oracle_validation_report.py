#!/usr/bin/env python3
"""Close ADR 0081 from frozen replica and cross-host validation artifacts."""

# ruff: noqa: UP045 - cluster reporting must run under macOS system Python 3.9.

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional

EXPERIMENT_ID = "complete-action-graded-oracle-ranker-v1"
REPLICAS = (
    {
        "host": "john1",
        "seed": 2026061601,
        "metadata_run": "runs/john1-seed-2026061601",
        "report_run": "cross-host/john1-seed-2026061601",
        "cross_host": "john2",
    },
    {
        "host": "john2",
        "seed": 2026061602,
        "metadata_run": "cross-host/john2-seed-2026061602",
        "report_run": "cross-host/john2-seed-2026061602",
        "cross_host": "john3",
    },
    {
        "host": "john3",
        "seed": 2026061603,
        "metadata_run": "cross-host/john3-seed-2026061603",
        "report_run": "cross-host/john3-seed-2026061603",
        "cross_host": "john1",
    },
)
HOST_ALIASES = {
    "john1": "john1",
    "Johns-Mac-mini": "john1",
    "john2": "john2",
    "john3": "john3",
}
VALID_JOB_PREFIXES = (
    "graded-oracle-convert-",
    "graded-oracle-merge-",
    "graded-oracle-max-width-smoke",
    "graded-oracle-train-schema2-",
    "graded-oracle-cross-eval-",
    "graded-oracle-selected-performance-",
)
CONCURRENT_PHASE_PREFIXES = (
    "graded-oracle-convert-",
    "graded-oracle-max-width-smoke",
    "graded-oracle-train-schema2-",
    "graded-oracle-cross-eval-",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_host(host: str) -> str:
    try:
        return HOST_ALIASES[host]
    except KeyError as error:
        raise ValueError(f"unknown Cascadia cluster host: {host}") from error


def load_json(path: Path, inputs: dict[str, str], root: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read frozen artifact {path}: {error}") from error
    inputs[str(path.relative_to(root))] = sha256_file(path)
    return value


def select_replica(replicas: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply ADR 0081's exact replica-selection order."""
    if not replicas:
        raise ValueError("no graded-oracle replicas were supplied")
    return min(
        replicas,
        key=lambda item: (
            item["selection_loss"],
            -item["top64_r4800_winner_recall"],
            item["r4800_residual_mae"],
            item["seed"],
        ),
    )


def find_host_report(
    report_run: Path,
    expected_host: str,
    inputs: dict[str, str],
    root: Path,
) -> dict[str, Any]:
    matches = []
    for path in sorted(report_run.glob("validation-report-*.json")):
        report = load_json(path, inputs, root)
        if normalize_host(str(report["host"])) == expected_host:
            matches.append(report)
    if len(matches) != 1:
        raise ValueError(
            f"expected one {expected_host} report in {report_run}, found {len(matches)}"
        )
    return matches[0]


def validate_replica(
    experiment_root: Path,
    spec: dict[str, Any],
    inputs: dict[str, str],
) -> dict[str, Any]:
    metadata_run = experiment_root / spec["metadata_run"]
    report_run = experiment_root / spec["report_run"]
    run = load_json(metadata_run / "run.json", inputs, experiment_root)
    best = load_json(metadata_run / "best.json", inputs, experiment_root)
    final_report = load_json(metadata_run / "final-report.json", inputs, experiment_root)
    cross_report = find_host_report(
        report_run,
        spec["cross_host"],
        inputs,
        experiment_root,
    )

    seed = int(run["training"]["seed"])
    if seed != spec["seed"]:
        raise ValueError(f"{spec['host']} training seed drifted: {seed}")
    if best["checkpoint"] != cross_report["checkpoint"]:
        raise ValueError(f"{spec['host']} cross-host checkpoint identity drifted")
    if best["validation"] != cross_report["metrics"]:
        raise ValueError(f"{spec['host']} cross-host metrics are not bit-identical")
    if final_report["best_ranking_loss"] != best["selection_loss"]:
        raise ValueError(f"{spec['host']} best selection loss drifted")
    if run["training"]["model"]["schema_version"] != 2:
        raise ValueError(f"{spec['host']} did not use corrected model schema 2")
    if run["training"]["model"]["prior_feature_schema"] != "observable-screen-priors-v1":
        raise ValueError(f"{spec['host']} did not use observable-only priors")
    if cross_report["dataset_manifest_blake3"] != run["datasets"]["validation_manifest_blake3"]:
        raise ValueError(f"{spec['host']} validation dataset identity drifted")
    if cross_report["source_run_manifest_blake3"] != _blake3_file(metadata_run / "run.json"):
        raise ValueError(f"{spec['host']} source run manifest identity drifted")
    if not cross_report["performance"]["passed"]:
        raise ValueError(f"{spec['host']} cross-host performance gates failed")

    metrics = best["validation"]
    return {
        "host": spec["host"],
        "seed": seed,
        "checkpoint": best["checkpoint"],
        "selection_loss": best["selection_loss"],
        "top64_r4800_winner_recall": metrics["top64_r4800_winner_recall"],
        "r4800_residual_mae": metrics["r4800_residual_mae"],
        "model_blake3": cross_report["model_blake3"],
        "checkpoint_manifest_blake3": cross_report["checkpoint_manifest_blake3"],
        "source_v2_blake3": run["source"]["v2_source_blake3"],
        "train_manifest_blake3": run["datasets"]["train_manifest_blake3"],
        "validation_manifest_blake3": run["datasets"]["validation_manifest_blake3"],
        "cross_evaluation_host": spec["cross_host"],
        "cross_evaluation_passed": cross_report["passed"],
        "cross_evaluation_performance": cross_report["performance"],
        "metrics": metrics,
        "gates": cross_report["gates"],
        "metadata_run": spec["metadata_run"],
        "report_run": spec["report_run"],
    }


def _blake3_file(path: Path) -> str:
    try:
        import blake3
    except ImportError as error:
        raise ValueError("blake3 is required to verify evaluator identities") from error
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_cluster_events(
    experiment_root: Path,
    inputs: dict[str, str],
) -> dict[str, Any]:
    paths = [
        experiment_root / "cluster" / f"events-{host}.jsonl"
        for host in ("john1", "john2", "john3")
    ]
    paths.extend(
        [
            experiment_root / "launch" / "cluster-events.jsonl",
            experiment_root / "launch" / "cluster-events-john2.jsonl",
            experiment_root / "launch" / "cluster-events-john3.jsonl",
        ]
    )
    events = []
    seen = set()
    for path in paths:
        inputs[str(path.relative_to(experiment_root))] = sha256_file(path)
        for line in path.read_text().splitlines():
            event = json.loads(line)
            name = str(event.get("name", ""))
            if not name.startswith(VALID_JOB_PREFIXES):
                continue
            key = (
                event.get("event"),
                event.get("host"),
                event.get("pid"),
                event.get("started_unix_seconds"),
                name,
            )
            if key in seen:
                continue
            seen.add(key)
            events.append(event)

    hosts = {
        host: {
            "assigned_wall_seconds": 0.0,
            "productive_wall_seconds": 0.0,
            "queued_idle_seconds": 0.0,
            "failures_or_retries": 0,
            "jobs_completed": 0,
            "work_completed": [],
        }
        for host in ("john1", "john2", "john3")
    }
    starts_by_phase: dict[str, dict[str, float]] = {}
    for event in events:
        host = normalize_host(str(event["host"]))
        name = str(event["name"])
        if event["event"] == "started":
            hosts[host]["queued_idle_seconds"] += float(event.get("queued_seconds", 0.0))
            for prefix in CONCURRENT_PHASE_PREFIXES:
                if name.startswith(prefix):
                    starts_by_phase.setdefault(prefix, {})[host] = float(
                        event["started_unix_seconds"]
                    )
                    break
            continue
        if event["event"] != "finished":
            continue
        elapsed = float(event["elapsed_seconds"])
        hosts[host]["assigned_wall_seconds"] += elapsed
        if int(event["return_code"]) == 0:
            hosts[host]["productive_wall_seconds"] += elapsed
            hosts[host]["jobs_completed"] += 1
            hosts[host]["work_completed"].append(name)
        else:
            hosts[host]["failures_or_retries"] += 1

    phase_skews = {}
    for phase, starts in starts_by_phase.items():
        if len(starts) < 2:
            continue
        phase_skews[phase] = max(starts.values()) - min(starts.values())
    max_phase_skew = max(phase_skews.values(), default=0.0)
    max_queue = max(
        (float(event.get("queued_seconds", 0.0)) for event in events),
        default=0.0,
    )
    return {
        "hosts": hosts,
        "phase_start_skew_seconds": phase_skews,
        "maximum_phase_start_skew_seconds": max_phase_skew,
        "maximum_lock_queue_seconds": max_queue,
        "valid_failures_or_retries": sum(
            int(item["failures_or_retries"]) for item in hosts.values()
        ),
        "no_healthy_host_idle_over_five_minutes_with_compatible_work_queued": (
            max(max_phase_skew, max_queue) < 300.0
        ),
        "invalid_diagnostics_excluded": [
            "serde parser launch contradiction",
            "teacher-provenance prior leakage runs terminated with exit 143",
        ],
    }


def verify_sealed_state(
    experiment_root: Path,
    manifest: dict[str, Any],
    readiness: dict[str, Any],
) -> dict[str, Any]:
    authorization_paths = list(experiment_root.rglob("test-authorization.json"))
    forbidden_outputs = []
    for pattern in ("test-report*.json", "sealed-test*.json", "gameplay-report*.json"):
        forbidden_outputs.extend(experiment_root.rglob(pattern))
    return {
        "manifest_status": manifest["test"]["status"],
        "manifest_model_access": manifest["datasets"]["test"]["model_access"],
        "launch_readiness_model_access": readiness["datasets"]["sealed_test_opened_by_model"],
        "test_authorization_exists": bool(authorization_paths),
        "test_evaluation_output_exists": bool(forbidden_outputs),
        "test_groups_read_by_reporter": False,
        "passed": (
            manifest["test"]["status"]
            in {"sealed", "closed_unopened_validation_failed"}
            and manifest["datasets"]["test"]["model_access"] == "sealed"
            and not readiness["datasets"]["sealed_test_opened_by_model"]
            and not authorization_paths
            and not forbidden_outputs
        ),
    }


def build_report(experiment_root: Path) -> dict[str, Any]:
    experiment_root = experiment_root.resolve()
    inputs: dict[str, str] = {}
    manifest = load_json(experiment_root / "manifest.json", inputs, experiment_root)
    readiness = load_json(
        experiment_root / "launch" / "launch-readiness.json",
        inputs,
        experiment_root,
    )
    if manifest["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("experiment manifest identity drifted")

    replicas = [
        validate_replica(experiment_root, spec, inputs)
        for spec in REPLICAS
    ]
    source_hashes = {item["source_v2_blake3"] for item in replicas}
    train_hashes = {item["train_manifest_blake3"] for item in replicas}
    validation_hashes = {item["validation_manifest_blake3"] for item in replicas}
    if len(source_hashes) != 1 or len(train_hashes) != 1 or len(validation_hashes) != 1:
        raise ValueError("replica source or dataset identities differ")

    selected = select_replica(replicas)
    selected_run = experiment_root / selected["report_run"]
    selected_performance = {}
    selected_reports = {}
    for host in ("john1", "john2", "john3"):
        report = find_host_report(selected_run, host, inputs, experiment_root)
        if report["model_blake3"] != selected["model_blake3"]:
            raise ValueError(f"selected model identity drifted on {host}")
        if report["metrics"] != selected["metrics"]:
            raise ValueError(f"selected model metrics drifted on {host}")
        if not report["performance"]["passed"]:
            raise ValueError(f"selected model performance failed on {host}")
        selected_performance[host] = report["performance"]
        selected_reports[host] = {
            "checkpoint_manifest_blake3": report["checkpoint_manifest_blake3"],
            "model_blake3": report["model_blake3"],
            "source_run_manifest_blake3": report["source_run_manifest_blake3"],
        }

    quality_gates = {
        key: value
        for key, value in selected["gates"].items()
        if not key.endswith("_performance_passed")
    }
    failed_quality_gates = sorted(key for key, value in quality_gates.items() if not value)
    quality_passed = all(quality_gates.values())
    performance_passed = all(item["passed"] for item in selected_performance.values())
    sealed_test = verify_sealed_state(experiment_root, manifest, readiness)
    cluster_utilization = summarize_cluster_events(experiment_root, inputs)
    metrics = selected["metrics"]
    screen_regret = float(metrics["screen_mean_top64_retained_r4800_regret"])
    learned_regret = float(metrics["mean_top64_retained_r4800_regret"])
    screen_recall = float(metrics["screen_top64_r4800_winner_recall"])
    learned_recall = float(metrics["top64_r4800_winner_recall"])

    compact_replicas = []
    for item in replicas:
        compact_replicas.append(
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
                    "cross_evaluation_host",
                    "cross_evaluation_passed",
                )
            }
        )
    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "rejected_before_test",
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
            "replicas": compact_replicas,
            "selected_host": selected["host"],
            "selected_seed": selected["seed"],
            "selected_checkpoint": selected["checkpoint"],
            "selected_model_blake3": selected["model_blake3"],
        },
        "selected_validation": {
            "metrics": metrics,
            "quality_gates": quality_gates,
            "failed_quality_gates": failed_quality_gates,
            "quality_passed": quality_passed,
        },
        "selected_performance_by_host": selected_performance,
        "selected_identity_by_host": selected_reports,
        "integrity": {
            "all_replica_sources_identical": len(source_hashes) == 1,
            "all_replica_train_datasets_identical": len(train_hashes) == 1,
            "all_replica_validation_datasets_identical": len(validation_hashes) == 1,
            "cross_host_metrics_bit_identical_to_selection": True,
            "all_actions_scored_once": metrics["all_candidates_scored_once"],
            "all_groups_scored_once": metrics["all_groups_scored_once"],
            "all_scores_finite": metrics["all_scores_finite"],
            "observable_only_prior_schema": "observable-screen-priors-v1",
            "teacher_provenance_used_as_model_input": False,
        },
        "performance_passed_on_all_hosts": performance_passed,
        "sealed_test": sealed_test,
        "cluster_utilization": cluster_utilization,
        "diagnosis": {
            "screen_top64_r4800_winner_recall": screen_recall,
            "learned_top64_r4800_winner_recall": learned_recall,
            "winner_recall_change_percentage_points": 100.0 * (learned_recall - screen_recall),
            "screen_mean_top64_retained_r4800_regret": screen_regret,
            "learned_mean_top64_retained_r4800_regret": learned_regret,
            "retained_regret_reduction_points": screen_regret - learned_regret,
            "retained_regret_reduction_fraction": (
                (screen_regret - learned_regret) / screen_regret
            ),
            "conclusion": (
                "The observable residual learner reduced retained regret but did not "
                "recover stable exact R4800 winner identity. The frozen recall gates "
                "therefore reject this target design before sealed test or gameplay."
            ),
        },
        "passed": (
            quality_passed
            and performance_passed
            and sealed_test["passed"]
            and cluster_utilization[
                "no_healthy_host_idle_over_five_minutes_with_compatible_work_queued"
            ]
        ),
        "input_sha256": dict(sorted(inputs.items())),
    }
    if report["passed"]:
        raise ValueError("rejection reporter unexpectedly observed a complete validation pass")
    return report


def render_markdown(report: dict[str, Any]) -> str:
    selected = report["replica_selection"]
    metrics = report["selected_validation"]["metrics"]
    diagnosis = report["diagnosis"]
    lines = [
        "# Complete-Action Graded Oracle Ranker V1 Rejection",
        "",
        "Status: **rejected on validation; sealed test and gameplay closed unopened**",
        "",
        "## Verdict",
        "",
        "The corrected observable-only MLX experiment completed all three frozen replicas,",
        "the preregistered cross-host validation matrix, and selected-model performance",
        "checks on john1, john2, and john3. The john2 replica won the frozen selection",
        "objective, but it failed every overall, phase, and subset winner-recall gate.",
        "ADR 0082 was therefore not authorized and no test group or gameplay seed was opened.",
        "",
        "## Replica Selection",
        "",
        "| Train host | Seed | Cross host | Retained regret | Top-64 recall | R4800 MAE |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for replica in selected["replicas"]:
        lines.append(
            "| {host} | {seed} | {cross_evaluation_host} | {selection_loss:.6f} | "
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
            "## Validation",
            "",
            "| Gate | Required | Observed | Result |",
            "|---|---:|---:|---|",
            "| Overall top-64 R4800 winner recall | >98% | {:.2%} | Fail |".format(
                metrics["top64_r4800_winner_recall"]
            ),
            "| Mean retained R4800 regret | <0.15 | {:.6f} | Pass |".format(
                metrics["mean_top64_retained_r4800_regret"]
            ),
            "| Early recall | >=97% | {:.2%} | Fail |".format(
                metrics["phase"]["early"]["top64_r4800_winner_recall"]
            ),
            "| Middle recall | >=97% | {:.2%} | Fail |".format(
                metrics["phase"]["middle"]["top64_r4800_winner_recall"]
            ),
            "| Late recall | >=97% | {:.2%} | Fail |".format(
                metrics["phase"]["late"]["top64_r4800_winner_recall"]
            ),
            "| Nature-token subset recall | >=95% | {:.2%} | Fail |".format(
                metrics["subsets"]["nature_token_available"][
                    "top64_r4800_winner_recall"
                ]
            ),
            "| Independent-draft subset recall | >=95% | {:.2%} | Fail |".format(
                metrics["subsets"]["independent_draft_winner"][
                    "top64_r4800_winner_recall"
                ]
            ),
            "",
            "Every regret, finite-score, complete-group, latency, memory, and swap gate passed.",
            "",
            "## Selected-Model Performance",
            "",
            "| Host | Action scores/s | P99 decision ms | Peak RSS MiB | Swap delta |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for host, performance in report["selected_performance_by_host"].items():
        lines.append(
            "| {} | {:,.0f} | {:.2f} | {:.1f} | {} |".format(
                host,
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
            "- The learned screen reduced retained regret from {:.6f} to {:.6f}, a "
            "{:.1%} reduction.".format(
                diagnosis["screen_mean_top64_retained_r4800_regret"],
                diagnosis["learned_mean_top64_retained_r4800_regret"],
                diagnosis["retained_regret_reduction_fraction"],
            ),
            "- Exact-winner recall moved only from {:.2%} to {:.2%}.".format(
                diagnosis["screen_top64_r4800_winner_recall"],
                diagnosis["learned_top64_r4800_winner_recall"],
            ),
            "- Cross-host metrics were bit-identical and selected-model inference passed on",
            "  every Mac, so this is a target/learning result rather than an execution or",
            "  portability failure.",
            "- The frozen gates are not weakened post hoc. The next experiment must revise",
            "  the oracle or target design; K2048 and a large self-play launch remain closed.",
            "",
            "## Protocol Closure",
            "",
            "- Test authorization file: absent.",
            "- Sealed-test evaluation output: absent.",
            "- Test groups read by this reporter: no.",
            "- ADR 0082: closed unopened.",
            "- ADR 0083: closed unopened.",
            "",
            "Machine-readable evidence is in",
            "`docs/v2/reports/complete-action-graded-oracle-ranker-v1-rejection.json`.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    report = build_report(args.experiment_root)
    write_atomic(
        args.output_json,
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    write_atomic(args.output_markdown, render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
