#!/usr/bin/env python3
"""Project V3 Part 2 wall time and storage from measured Part 1 rates."""

from __future__ import annotations

import argparse
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def project(
    gameplay: dict[str, Any],
    direct: dict[str, Any],
    r600: dict[str, Any],
    mlx: dict[str, Any],
    compact: dict[str, Any],
) -> dict[str, object]:
    compact_shard = compact.get("compact_shard", compact)
    if not isinstance(compact_shard, dict):
        raise ValueError("compact replay measurement is missing")
    decision_seconds = 1.0 / float(gameplay["decisions_per_second"])
    r600_decision_seconds = float(r600["r600_seconds_per_game"]) / 20.0
    compact_bytes_per_game = (
        int(compact["bytes_per_game"])
        if "bytes_per_game" in compact
        else math.ceil(int(compact_shard["bytes"]) / int(compact_shard["games"]))
    )
    direct_seconds_per_game = float(direct["seconds_per_game"])

    # CPU collections run one game per scheduler allocation and consume all
    # cores on that host during V3 ranking. Expert games have exactly 20 newest
    # model decisions; the other three seats are frozen V1/prior policies.
    collection_seconds_per_game = (
        float(compact["collection_seconds_per_game"])
        if "collection_seconds_per_game" in compact
        else float(compact["elapsed_seconds"]) / int(compact_shard["games"])
    )
    bootstrap_collection = collection_seconds_per_game * 500_000 / 3
    # One seat is always the newest V3. Cycle 1 uses three V1 opponents; later
    # cycles keep 80% of opponent seats on qualified V1 and sample prior V3 on
    # the remaining 20%, avoiding a progressively V3-heavy collection cost.
    expected_v3_seats = 1.0 + 9 * (1.0 + 3 * 0.20)
    expert_collection = (
        10_000 * direct_seconds_per_game * expected_v3_seats / 4 / 3
    )
    teacher_roots = 100_000 + 20_000 + 10 * 2_500
    teacher_search = teacher_roots * r600_decision_seconds / 3
    # Maximum sequential promotion traffic: 500 pairs, two physical focal
    # games, ten cycles, split over John2 and John3 during John1 training.
    promotion_r600 = 10 * 500 * 2 * float(r600["r600_seconds_per_game"]) / 2
    promotion_other_tiers = promotion_r600 * 0.35
    final_evaluation = (
        250 * 2 * float(r600["r600_seconds_per_game"]) / 2
        + 4_000 * direct_seconds_per_game / 3
        + 4 * 3600
    )
    mlx_training = float(mlx["projected_part2_seconds"])
    measured_sum = (
        bootstrap_collection
        + expert_collection
        + teacher_search
        + promotion_r600
        + promotion_other_tiers
        + final_evaluation
        + mlx_training
    )
    recovery_margin = measured_sum * 0.20
    active_wall_seconds = measured_sum + recovery_margin

    total_games = 500_000 + 10 * 10_000 + 4_000 + 500
    replay_bytes = compact_bytes_per_game * total_games
    teacher_bytes = teacher_roots * 32 * 96
    frozen_serving_bundles = 14 * 120 * 1024**2
    rolling_training_checkpoints = 4 * int(compact.get("checkpoint_bytes", 1_150_000_000))
    engineering_and_reports = 2 * 1024**3
    projected_campaign_bytes = (
        replay_bytes
        + teacher_bytes
        + frozen_serving_bundles
        + rolling_training_checkpoints
        + engineering_and_reports
    )
    return {
        "schema_id": "cascadia-v3-part2-capacity-projection-v1",
        "protected_seed_values_opened": False,
        "scientific_training_started": False,
        "active_wall_seconds": active_wall_seconds,
        "active_wall_days": active_wall_seconds / 86400,
        "measured_components_seconds": {
            "bootstrap_collection": bootstrap_collection,
            "expert_collection": expert_collection,
            "teacher_search": teacher_search,
            "promotion_r600": promotion_r600,
            "promotion_other_tiers": promotion_other_tiers,
            "final_evaluation": final_evaluation,
            "mlx_training": mlx_training,
            "recovery_margin": recovery_margin,
        },
        "projected_campaign_bytes": projected_campaign_bytes,
        "projected_campaign_gib": projected_campaign_bytes / 1024**3,
        "storage_components_bytes": {
            "compact_replays": replay_bytes,
            "teacher_labels": teacher_bytes,
            "frozen_serving_bundles": frozen_serving_bundles,
            "rolling_training_checkpoints": rolling_training_checkpoints,
            "engineering_and_reports": engineering_and_reports,
        },
        "rates": {
            "direct_decision_seconds": decision_seconds,
            "direct_all_v3_seconds_per_game": direct_seconds_per_game,
            "r600_decision_seconds": r600_decision_seconds,
            "mlx_examples_per_second": mlx["examples_per_second"],
            "compact_bytes_per_game": compact_bytes_per_game,
        },
        "assumptions": {
            "cpu_hosts": 3,
            "benchmark_hosts_during_training": 2,
            "expert_cycles": 10,
            "expert_games_per_cycle": 10_000,
            "newest_model_decisions_per_expert_game": 20,
            "qualified_v1_opponent_fraction": 0.80,
            "expected_v3_seat_equivalents_across_cycles": expected_v3_seats,
            "teacher_roots": teacher_roots,
            "recovery_margin_fraction": 0.20,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gameplay-profile", type=Path, required=True)
    parser.add_argument("--direct-smoke", type=Path, required=True)
    parser.add_argument("--r600-smoke", type=Path, required=True)
    parser.add_argument("--mlx-profile", type=Path, required=True)
    parser.add_argument("--compact-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = project(
        _read(args.gameplay_profile),
        _read(args.direct_smoke),
        _read(args.r600_smoke),
        _read(args.mlx_profile),
        _read(args.compact_receipt),
    )
    _write_atomic(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
