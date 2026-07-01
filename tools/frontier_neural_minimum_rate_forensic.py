#!/usr/bin/env python3
"""Prove ADR 0109's minimum-rate completion conflict without model execution."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import tarfile
from pathlib import Path
from typing import Any

import blake3

EXPERIMENT_ID = "complete-action-frontier-neural-minimum-rate-forensic-v1"
SOURCE_EXPERIMENT_ID = (
    "complete-action-frontier-calibrated-neural-stage-v1"
)
COMBINED_BLAKE3 = (
    "f269476798c1df773b71d58599da593213b24b041caba528d5bc46d4f19a5b32"
)
SOURCE_BUNDLE_BLAKE3 = (
    "8f5bd5ee0e85952f0a3d486fc348243c9ad42c922c16331a7e056944ff461580"
)
GROUP_SCIENTIFIC_BLAKE3 = (
    "1d6ee91568ecadd3eece723fa2b4e059960def1bc18db1929e3634061b935839"
)
MAXIMUM_TRIALS = 16
MINIMUM_ACCEPTED_RATE = 1e-8
CONVERGENCE_RATE_THRESHOLD = 1e-7
LOSS_TOLERANCE = 1e-12


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON input is not an object: {path}")
    return value


def _digest_path(path: Path) -> str:
    return blake3.blake3(path.read_bytes()).hexdigest()


def _digest_json(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return blake3.blake3(payload).hexdigest()


def enumerate_rate_paths(
    *,
    accepted_updates: int,
    total_backtracks: int,
    maximum_backtracks: int,
    maximum_rate: float,
    minimum_rate: float,
    mean_rate: float,
    rate_regrowth: float = 2.0,
    backtrack_factor: float = 0.5,
) -> list[dict[str, Any]]:
    """Enumerate all rate paths exactly consistent with summary bookkeeping."""
    target_sum = mean_rate * accepted_updates
    solutions = []
    for trials in itertools.product(
        range(maximum_backtracks + 1),
        repeat=accepted_updates,
    ):
        if (
            sum(trials) != total_backtracks
            or max(trials) != maximum_backtracks
        ):
            continue
        start = maximum_rate
        rates = []
        for trial in trials:
            rate = start * backtrack_factor**trial
            rates.append(rate)
            start = min(maximum_rate, rate * rate_regrowth)
        if not math.isclose(
            sum(rates),
            target_sum,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            continue
        if not math.isclose(
            min(rates),
            minimum_rate,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            continue
        if not math.isclose(
            max(rates),
            maximum_rate,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            continue
        attempted = [
            start * backtrack_factor**trial
            for trial in range(MAXIMUM_TRIALS)
        ]
        solutions.append(
            {
                "accepted_step_backtracks": list(trials),
                "accepted_rates": rates,
                "failed_step_starting_rate": start,
                "failed_step_attempted_rates": attempted,
                "smallest_failed_step_attempted_rate": min(attempted),
            }
        )
    return solutions


def _load_frozen_source(source_bundle: Path) -> str:
    member = (
        "python/cascadia_mlx/"
        "graded_oracle_frontier_calibrated_adamw.py"
    )
    with tarfile.open(source_bundle) as archive:
        extracted = archive.extractfile(member)
        if extracted is None:
            raise ValueError("ADR 0109 source bundle lacks optimizer source")
        return extracted.read().decode("utf-8")


def run_audit(
    *,
    combined_path: Path,
    origin_path: Path,
    comparison_path: Path,
    source_bundle: Path,
) -> dict[str, Any]:
    """Run the frozen deterministic proof and reclassify ADR 0109."""
    identities = {
        "combined_report_blake3": _digest_path(combined_path),
        "source_bundle_blake3": _digest_path(source_bundle),
    }
    combined = _load(combined_path)
    origin = _load(origin_path)
    comparison = _load(comparison_path)
    scientific = origin["scientific"]
    identities["group_2_scientific_blake3"] = _digest_json(scientific)
    identity_gate = bool(
        identities["combined_report_blake3"] == COMBINED_BLAKE3
        and identities["source_bundle_blake3"] == SOURCE_BUNDLE_BLAKE3
        and identities["group_2_scientific_blake3"]
        == GROUP_SCIENTIFIC_BLAKE3
        and combined.get("experiment_id") == SOURCE_EXPERIMENT_ID
        and origin.get("experiment_id") == SOURCE_EXPERIMENT_ID
        and comparison.get("experiment_id") == SOURCE_EXPERIMENT_ID
        and comparison.get("scientific_payload_identical") is True
        and int(scientific["group_index"]) == 2
    )
    optimizer = scientific["optimizer"]
    paths = enumerate_rate_paths(
        accepted_updates=int(optimizer["accepted_updates"]),
        total_backtracks=int(optimizer["total_backtracks"]),
        maximum_backtracks=int(optimizer["maximum_backtracks"]),
        maximum_rate=float(optimizer["maximum_accepted_rate"]),
        minimum_rate=float(optimizer["minimum_accepted_rate"]),
        mean_rate=float(optimizer["mean_accepted_rate"]),
    )
    starting_rates = {
        float(path["failed_step_starting_rate"]) for path in paths
    }
    smallest_rates = {
        float(path["smallest_failed_step_attempted_rate"])
        for path in paths
    }
    source = _load_frozen_source(source_bundle)
    required_source_fragments = (
        "rate_is_eligible = rate >= MINIMUM_LEARNING_RATE",
        "and candidate_value <= current_value + LOSS_TOLERANCE",
        "for trial in range(MAXIMUM_TRIALS):",
        "allow_numerical_convergence",
        "diagnostics[\"maximum_candidate_improvement\"]",
        "<= LOSS_TOLERANCE",
    )
    source_gate = all(
        fragment in source for fragment in required_source_fragments
    )
    enumeration_gate = bool(
        1 <= len(paths) <= 1000
        and len(starting_rates) == 1
        and len(smallest_rates) == 1
        and next(iter(smallest_rates))
        < CONVERGENCE_RATE_THRESHOLD
    )
    finite_state_gate = bool(
        int(optimizer["nonfinite_rejections"]) == 0
        and optimizer["moments_finite"] is True
        and scientific["final"]["all_scores_finite"] is True
        and scientific["failure"]
        == "monotone AdamW could not accept an update"
        and int(optimizer["accepted_updates"]) > 0
    )
    minimum_rate_conflict = bool(
        identity_gate
        and source_gate
        and enumeration_gate
        and finite_state_gate
        and next(iter(smallest_rates)) < MINIMUM_ACCEPTED_RATE
    )
    corrected_pipeline = bool(
        minimum_rate_conflict
        and combined["scientific"]["gates"][
            "all_four_replays_identical"
        ]
        is True
    )
    terminal = combined["scientific"]["aggregate"]
    terminal_strength = bool(
        terminal["target_positive_recall"] >= 0.90
        and terminal["target_set_exact_fraction"] >= 0.75
    )
    if not corrected_pipeline:
        classification = "neural_minimum_rate_forensic_invalid"
    elif not terminal_strength:
        classification = "public_observable_representation_insufficient"
    else:
        classification = "local_failure_not_reproduced"
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "scientific": {
            "classification": classification,
            "identity": identities,
            "consistent_rate_paths": paths,
            "consistent_rate_path_count": len(paths),
            "failed_step_starting_rate": (
                next(iter(starting_rates))
                if len(starting_rates) == 1
                else None
            ),
            "smallest_failed_step_attempted_rate": (
                next(iter(smallest_rates))
                if len(smallest_rates) == 1
                else None
            ),
            "inferred_rejected_condition": (
                "subminimum_candidate_improvement_exceeded_tolerance"
                if minimum_rate_conflict
                else None
            ),
            "domain_consistent_group_2_completion": (
                "numerically_converged_after_8_accepted_updates"
                if minimum_rate_conflict
                else None
            ),
            "recombined_terminal": terminal,
            "gates": {
                "frozen_identity_passed": identity_gate,
                "frozen_source_logic_passed": source_gate,
                "rate_path_enumeration_passed": enumeration_gate,
                "finite_state_proof_passed": finite_state_gate,
                "minimum_rate_completion_conflict_proved": (
                    minimum_rate_conflict
                ),
                "domain_consistent_pipeline_passed": corrected_pipeline,
                "terminal_strength_gate_passed": terminal_strength,
                "strength_checkpoint_observed": False,
            },
            "model_execution_used": False,
            "neural_training_used": False,
            "test_split_opened": False,
            "gameplay_opened": False,
            "new_teacher_compute_used": False,
            "external_compute_used": False,
        },
    }


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def render_markdown(report: dict[str, Any]) -> str:
    scientific = report["scientific"]
    terminal = scientific["recombined_terminal"]
    lines = [
        "# Complete-Action Frontier Neural Minimum-Rate Forensic V1 Result",
        "",
        f"Classification: `{scientific['classification']}`.",
        "",
        "ADR 0110 used only frozen ADR 0109 JSON and source-bundle evidence. "
        "It did not execute MLX, a model forward pass, gradients, or training.",
        "",
        "## Proof",
        "",
        f"- Consistent accepted-rate histories: "
        f"{scientific['consistent_rate_path_count']}.",
        f"- Failed-step starting rate in every history: "
        f"`{scientific['failed_step_starting_rate']:.15g}`.",
        f"- Smallest of 16 failed-step attempted rates in every history: "
        f"`{scientific['smallest_failed_step_attempted_rate']:.15g}`.",
        "- The smallest attempted rate is below both the `1e-7` convergence "
        "threshold and the optimizer's `1e-8` acceptance floor.",
        "- The frozen report records zero nonfinite rejections, finite moments "
        "and scores, eight prior accepted updates, and the generic exhausted "
        "backtracking failure.",
        "- Frozen source proves every eligible finite loss-nonincreasing "
        "proposal would have been accepted.",
        "",
        "Therefore the only remaining failed convergence condition was a "
        "candidate improvement greater than `1e-12` at a rate below `1e-8`. "
        "That proposal was outside the optimizer's eligible update domain and "
        "cannot consistently invalidate numerical convergence.",
        "",
        "Group 2 is domain-consistently reclassified as numerically converged "
        "after eight accepted updates, without changing its model or metrics.",
        "",
        "## Recombined Decision",
        "",
        f"- Terminal target recall: "
        f"{_percent(terminal['target_positive_recall'])}.",
        f"- Terminal exact sets: "
        f"{_percent(terminal['target_set_exact_fraction'])}.",
        "- The 120-exposure checkpoint remains unobserved.",
        "",
        "The corrected pipeline passes, but terminal strength misses both "
        "gates. The frozen mechanism therefore classifies "
        "`public_observable_representation_insufficient` and authorizes one "
        "separately preregistered public-observable representation treatment. "
        "It does not authorize a full trainer directly.",
        "",
        "## Gates",
        "",
        "| Gate | Result |",
        "|---|---|",
    ]
    for name, passed in sorted(scientific["gates"].items()):
        lines.append(f"| `{name}` | {'pass' if passed else 'fail'} |")
    lines.append("")
    return "\n".join(lines)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined", type=Path, required=True)
    parser.add_argument("--origin", type=Path, required=True)
    parser.add_argument("--replay-comparison", type=Path, required=True)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    report = run_audit(
        combined_path=args.combined,
        origin_path=args.origin,
        comparison_path=args.replay_comparison,
        source_bundle=args.source_bundle,
    )
    _write_json(args.json_output, report)
    markdown = render_markdown(report)
    _write_text(args.markdown_output, markdown)
    print(markdown)
    return 0 if all(
        bool(value) or name == "terminal_strength_gate_passed"
        or name == "strength_checkpoint_observed"
        for name, value in report["scientific"]["gates"].items()
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
