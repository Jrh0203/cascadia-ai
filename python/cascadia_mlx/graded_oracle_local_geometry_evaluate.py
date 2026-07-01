"""Cross-host validation and confidence evaluation for ADR 0088."""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import numpy as np

from cascadia_mlx.checkpoint import load_checkpoint_pointer_with_factory
from cascadia_mlx.graded_oracle_dataset import (
    GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
    GRADED_ORACLE_PACKED_ACTION_LIMIT,
    GradedOracleDataset,
)
from cascadia_mlx.graded_oracle_identifiability import (
    PHASE_NAMES,
    AuditAccumulator,
    analyze_decision,
)
from cascadia_mlx.graded_oracle_local_geometry_model import (
    LocalGeometryModelConfig,
    LocalGeometryRanker,
)
from cascadia_mlx.graded_oracle_metrics import (
    benchmark_graded_oracle,
    evaluate_graded_oracle,
    graded_oracle_validation_gates,
)
from cascadia_mlx.graded_oracle_model import predict_graded_oracle_batch

EXPERIMENT_ID = "complete-action-local-geometry-ranker-v1"
HOST_ALIASES = {
    "Johns-Mac-mini": "john1",
    "john1": "john1",
    "john2": "john2",
    "john3": "john3",
    "john4": "john4",
}


def evaluate_confidence(
    model: LocalGeometryRanker,
    dataset: GradedOracleDataset,
) -> dict[str, Any]:
    """Measure R4800 confidence-set and distinguishable-winner recall."""
    if dataset.split != "validation":
        raise ValueError("ADR 0088 confidence evaluation accepts only validation")
    model.eval()
    overall = AuditAccumulator()
    phases = {name: AuditAccumulator() for name in PHASE_NAMES.values()}
    subsets = {
        "nature_token_available": AuditAccumulator(),
        "independent_draft_winner": AuditAccumulator(),
    }
    groups_seen = 0
    candidates_seen = 0
    nonfinite_model_scores = 0

    for batch in dataset.batches(
        64,
        maximum_actions_per_batch=GRADED_ORACLE_PACKED_ACTION_LIMIT,
        maximum_group_actions=GRADED_ORACLE_MAXIMUM_GROUP_ACTIONS,
    ):
        prediction = predict_graded_oracle_batch(model, batch)
        mx.eval(prediction.scores)
        scores = np.asarray(prediction.scores)
        masks = np.asarray(batch.candidate_mask)
        screen = np.asarray(batch.screen_value)
        selected = np.asarray(batch.selected_index)
        hashes = np.asarray(batch.action_hash)
        r1200 = np.asarray(batch.r1200_mean)
        r1200_stddev = np.asarray(batch.r1200_stddev)
        r1200_samples = np.asarray(batch.r1200_samples)
        r1200_mask = np.asarray(batch.r1200_mask)
        r4800 = np.asarray(batch.r4800_mean)
        r4800_stddev = np.asarray(batch.r4800_stddev)
        r4800_samples = np.asarray(batch.r4800_samples)
        r4800_mask = np.asarray(batch.r4800_mask)
        phase_values = np.asarray(batch.phase)
        tokens = np.asarray(batch.active_nature_tokens)
        draft_kind = np.asarray(batch.draft_kind)
        raw_seed = np.asarray(batch.game_index)

        for group_index, mask in enumerate(masks):
            count = int(np.sum(mask))
            model_scores = scores[group_index, :count]
            nonfinite_model_scores += int(np.sum(~np.isfinite(model_scores)))
            winner = int(selected[group_index])
            observation = analyze_decision(
                model_scores=model_scores,
                screen_scores=screen[group_index, :count],
                action_hashes=hashes[group_index, :count],
                selected_index=winner,
                r1200_mean=r1200[group_index, :count],
                r1200_stddev=r1200_stddev[group_index, :count],
                r1200_samples=r1200_samples[group_index, :count],
                r1200_mask=r1200_mask[group_index, :count],
                r4800_mean=r4800[group_index, :count],
                r4800_stddev=r4800_stddev[group_index, :count],
                r4800_samples=r4800_samples[group_index, :count],
                r4800_mask=r4800_mask[group_index, :count],
                phase=int(phase_values[group_index]),
                nature_token_available=int(tokens[group_index]) > 0,
                independent_draft_winner=int(draft_kind[group_index, winner]) == 1,
                raw_seed=int(raw_seed[group_index]),
            )
            overall.add(observation)
            phases[observation.phase].add(observation)
            if observation.nature_token_available:
                subsets["nature_token_available"].add(observation)
            if observation.independent_draft_winner:
                subsets["independent_draft_winner"].add(observation)
            groups_seen += 1
            candidates_seen += count

    return {
        "overall": overall.report(),
        "phases": {
            name: accumulator.report()
            for name, accumulator in phases.items()
        },
        "subsets": {
            name: accumulator.report()
            for name, accumulator in subsets.items()
            if accumulator.groups
        },
        "integrity": {
            "split_is_validation": dataset.split == "validation",
            "groups_seen": groups_seen,
            "expected_groups": dataset.group_count,
            "all_groups_seen_once": groups_seen == dataset.group_count,
            "candidates_seen": candidates_seen,
            "expected_candidates": dataset.candidate_count,
            "all_candidates_seen_once": candidates_seen == dataset.candidate_count,
            "nonfinite_model_scores": nonfinite_model_scores,
            "all_model_scores_finite": nonfinite_model_scores == 0,
            "test_split_opened": False,
        },
    }


def local_geometry_validation_gates(
    metrics: dict[str, Any],
    confidence: dict[str, Any],
    performance_by_host: dict[str, dict[str, Any]] | None = None,
) -> dict[str, bool]:
    """Apply every frozen ADR 0088 validation threshold."""
    gates = graded_oracle_validation_gates(metrics, performance_by_host)
    top64 = confidence["overall"]["ranking"]["model"]["top64"]
    gates["top64_confidence_set_coverage_at_least_0_99"] = (
        float(top64["confidence_set_coverage_95"]) >= 0.99
    )
    distinguishable_recall = top64["distinguishable_winner_recall"]
    gates["top64_distinguishable_winner_recall_at_least_0_98"] = (
        distinguishable_recall is not None
        and float(distinguishable_recall) >= 0.98
    )
    for phase, values in confidence["phases"].items():
        coverage = values["ranking"]["model"]["top64"]["confidence_set_coverage_95"]
        gates[f"{phase}_confidence_set_coverage_at_least_0_98"] = (
            float(coverage) >= 0.98
        )
    integrity = confidence["integrity"]
    gates["confidence_all_groups_scored_once"] = bool(
        integrity["all_groups_seen_once"]
    )
    gates["confidence_all_candidates_scored_once"] = bool(
        integrity["all_candidates_seen_once"]
    )
    gates["confidence_all_scores_finite"] = bool(
        integrity["all_model_scores_finite"]
    )
    gates["sealed_test_unopened"] = not bool(integrity["test_split_opened"])
    return gates


def evaluate_validation(
    run_dir: str | Path,
    validation_dataset: str | Path,
) -> dict[str, object]:
    """Evaluate one selected ADR 0088 replica without opening sealed test."""
    started = time.perf_counter()
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run.json").read_text())
    _validate_run_kind(run)
    training = run["training"]
    dataset = GradedOracleDataset(validation_dataset)
    if dataset.split != "validation":
        raise ValueError("ADR 0088 evaluation accepts only validation")
    manifest_hash = _checksum(dataset.root / "dataset.json")
    if manifest_hash != run["datasets"]["validation_manifest_blake3"]:
        raise ValueError("validation manifest does not match the training run")

    model, _optimizer, _state, checkpoint = load_checkpoint_pointer_with_factory(
        run_dir,
        pointer="best",
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        model_factory=lambda values: LocalGeometryRanker(
            LocalGeometryModelConfig.from_dict(values)
        ),
    )
    metrics = evaluate_graded_oracle(model, dataset, group_batch_size=64)
    confidence = evaluate_confidence(model, dataset)
    performance = benchmark_graded_oracle(model, dataset)
    host_name = socket.gethostname().split(".")[0]
    host = HOST_ALIASES.get(host_name, host_name)
    gates = local_geometry_validation_gates(
        metrics,
        confidence,
        performance_by_host={host: performance},
    )
    scientific = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "checkpoint": checkpoint.name,
        "checkpoint_manifest_blake3": _checksum(checkpoint / "checkpoint.json"),
        "model_blake3": _checksum(checkpoint / "model.safetensors"),
        "source_run_manifest_blake3": _checksum(run_dir / "run.json"),
        "dataset": _dataset_identity(dataset, manifest_hash),
        "metrics": metrics,
        "confidence": confidence,
        "test_split_opened": False,
    }
    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "evaluation_kind": "validation-cross-host",
        "host": host,
        "scientific": scientific,
        "scientific_blake3": _canonical_digest(scientific),
        "performance": performance,
        "gates": gates,
        "passed": all(gates.values()),
        "execution": {
            "elapsed_seconds": time.perf_counter() - started,
            "test_split_opened": False,
        },
    }
    _write_json_atomic(run_dir / f"validation-report-{host}.json", report)
    return report


def _validate_run_kind(run: dict[str, Any]) -> None:
    if run.get("kind") != "graded-oracle-local-geometry-ranking":
        raise ValueError("run is not an ADR 0088 local-geometry replica")


def _dataset_identity(
    dataset: GradedOracleDataset,
    manifest_hash: str,
) -> dict[str, Any]:
    return {
        "dataset_id": dataset.manifest["dataset_id"],
        "split": dataset.split,
        "games": dataset.manifest["completed_games"],
        "seeds": dataset.manifest["seeds"],
        "groups": dataset.group_count,
        "candidates": dataset.candidate_count,
        "manifest_blake3": manifest_hash,
    }


def _checksum(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return blake3.blake3(payload).hexdigest()


def _write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_validation(args.run_dir, args.validation_dataset)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
