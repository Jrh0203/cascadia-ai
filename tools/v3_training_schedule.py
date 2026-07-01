#!/usr/bin/env python3
"""Materialize the immutable V3 bootstrap and ten-cycle training contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

EXPLORATION = [0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.035, 0.03, 0.02]
CALIBRATION_EXPOSURES_PER_RATE = 4_000_000
BOOTSTRAP_EXPOSURES_PER_ORIGIN = 36_000_000


def bootstrap_blocks() -> list[dict[str, object]]:
    """Twelve bounded blocks totaling 36M exposures for one origin.

    "Pass" in the research spec denotes one schedule block, not a complete
    traversal of all 40M afterstates. A literal twelve full-corpus traversal
    would silently inflate the approved 120M-exposure projection to 1.44B.
    """

    blocks: list[dict[str, object]] = []
    for _ in range(4):
        blocks.append(
            {
                "kind": "broad",
                "exposures": 4_500_000,
                "data_mix": {"broad": 1.0, "teacher": 0.0},
                "learning_rate_multiplier": 1.0,
                "teacher_lambda": None,
            }
        )
    teacher_block = 0
    for kind, count, multiplier in (
        ("broad-teacher-50-50", 6, 1.0),
        ("low-rate-consolidation", 2, 0.2),
    ):
        for _ in range(count):
            teacher_lambda = round(1.0 - 0.25 * teacher_block / 7, 8)
            blocks.append(
                {
                    "kind": kind,
                    "exposures": 2_250_000,
                    "data_mix": {"broad": 0.5, "teacher": 0.5},
                    "learning_rate_multiplier": multiplier,
                    "teacher_lambda": teacher_lambda,
                }
            )
            teacher_block += 1
    cursor = 0
    for index, block in enumerate(blocks, start=1):
        block["block"] = index
        block["start_exposure"] = cursor
        cursor += int(block["exposures"])
        block["end_exposure"] = cursor
        block["checkpoint_at_boundary"] = True
    if cursor != BOOTSTRAP_EXPOSURES_PER_ORIGIN:
        raise AssertionError("bootstrap exposure schedule drifted")
    return blocks


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def authorized_state(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    recorded = value.pop("state_sha256", None)
    if (
        recorded != hashlib.sha256(_canonical(value)).hexdigest()
        or value.get("schema_id") != "cascadia-v3-campaign-state-v1"
        or value.get("part") != 2
        or value.get("phase2_authorized") is not True
        or value.get("approved_readiness_sha256") is None
    ):
        raise ValueError("training schedule requires checksum-bound Phase 2 authorization")
    return value


def build(batch_size: int) -> dict[str, object]:
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    return {
        "schema_id": "cascadia-v3-training-schedule-v2",
        "architecture_fixed": True,
        "batch_size": batch_size,
        "checkpoint_every_exposures": 5_000_000,
        "bootstrap": {
            "learning_rate_calibration": {
                "corpus_fraction": 0.10,
                "rates": [5e-4, 1e-3, 1.5e-3],
                "exposures_per_rate": CALIBRATION_EXPOSURES_PER_RATE,
                "total_exposures": 3 * CALIBRATION_EXPOSURES_PER_RATE,
                "selection": "quantized-validation-loss-with-open-game-nonregression",
            },
            "origins": 3,
            "pass_semantics": "bounded-exposure-block-not-full-corpus-epoch",
            "exposures_per_origin": BOOTSTRAP_EXPOSURES_PER_ORIGIN,
            "origin_exposures": 3 * BOOTSTRAP_EXPOSURES_PER_ORIGIN,
            "total_exposures_including_calibration": 120_000_000,
            "blocks": bootstrap_blocks(),
            "teacher_lambda": {"start": 1.0, "end": 0.75},
            "stochastic_weight_averaging": {
                "final_fraction": 0.20,
                "start_exposure": 28_800_000,
                "update_interval_exposures": 900_000,
            },
            "d6_online": True,
            "uniform_phase_sampling": True,
        },
        "cycles": [
            {
                "cycle": cycle,
                "origins": 2,
                "passes": [
                    {"learning_rate": 3e-5, "exposures": 400_000},
                    {"learning_rate": 3e-5, "exposures": 400_000},
                    {"learning_rate": 1e-5, "exposures": 400_000},
                ],
                "data_mix": {
                    "current_cycle": 0.50,
                    "preceding_three_cycles": 0.30,
                    "bootstrap_and_older": 0.20,
                },
                "equalize_phase": True,
                "equalize_score_quantile_within_phase": True,
                "source_quotas_per_pass": (
                    {
                        "current_cycle_replay": 120_000,
                        "current_cycle_teacher": 80_000,
                        "preceding_three_cycles": 0,
                        "bootstrap_replay": 100_000,
                        "bootstrap_teacher": 100_000,
                    }
                    if cycle == 1
                    else {
                        "current_cycle_replay": 120_000,
                        "current_cycle_teacher": 80_000,
                        "preceding_three_cycles": 120_000,
                        "bootstrap_replay": 40_000,
                        "bootstrap_teacher": 40_000,
                    }
                ),
                "total_exposures_per_origin": 1_200_000,
                "exploration_epsilon": EXPLORATION[cycle - 1],
            }
            for cycle in range(1, 11)
        ],
        "topology": {
            "trainer": "john1-native-mlx",
            "during_training": {
                "john1": "mlx-training",
                "john2": "previous-checkpoint-benchmark",
                "john3": "previous-checkpoint-benchmark",
                "john4": "dashboard-only",
            },
            "next_cycle_generation_during_training": False,
        },
    }


def _write_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-state", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8_192)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state = authorized_state(args.campaign_state)
    result = build(args.batch_size)
    result["approved_readiness_sha256"] = state["approved_readiness_sha256"]
    _write_atomic(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
