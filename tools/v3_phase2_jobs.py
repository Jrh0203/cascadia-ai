#!/usr/bin/env python3
"""Materialize topology-free Bacalhau work items for an authorized V3 phase."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from cascadia_cluster import ContainerInput, ContainerSpec, Resources

IMAGE = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")
EXPLORATION = [0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.035, 0.03, 0.02]


def _authorized(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    recorded = value.pop("state_sha256", None)
    observed = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        recorded != observed
        or value.get("schema_id") != "cascadia-v3-campaign-state-v1"
        or value.get("part") != 2
        or value.get("phase2_authorized") is not True
    ):
        raise ValueError("V3 Phase 2 state is not checksum-authorized")
    return value


def _items(
    *,
    component: str,
    games: int,
    first_game_index: int,
    games_per_item: int,
    approved_readiness_sha256: str,
    cycle: int | None = None,
    epsilon: float = 0.0,
) -> list[ContainerInput]:
    if games <= 0 or games_per_item <= 0 or games % games_per_item:
        raise ValueError("game count must divide evenly into positive scheduler items")
    result = []
    for offset in range(0, games, games_per_item):
        key = f"{component}-{cycle or 0:02d}-{offset // games_per_item:06d}"
        args = [
            "v3-campaign-worker",
            "collect",
            "--output",
            f"/outputs/{key}.v3g",
            "--games",
            str(games_per_item),
            "--first-game-index",
            str(first_game_index + offset),
            "--component",
            component,
            "--epsilon",
            str(epsilon),
            "--campaign-state",
            "/inputs/control/campaign-state.json",
            "--approved-readiness-sha256",
            approved_readiness_sha256,
        ]
        if cycle is not None:
            args.extend(("--cycle", str(cycle)))
        if component in {"v1-direct", "mixed-frozen", "expert-iteration"}:
            args.extend(("--v1-weights", "/inputs/v1/qualified-v1.bin"))
        if component == "expert-iteration":
            args.extend(("--v3-model-dir", "/inputs/v3/newest"))
            for prior in range(1, cycle or 1):
                args.extend(("--v3-model-dir", f"/inputs/v3/cycle-{prior:02d}"))
        result.append(
            ContainerInput(
                key=key,
                args=tuple(args),
                environment={"RAYON_NUM_THREADS": "1"},
                application_metadata={
                    "campaign": "cascadia-v3",
                    "component": component,
                    "cycle": str(cycle or 0),
                    "first_game_index": str(first_game_index + offset),
                    "games": str(games_per_item),
                },
            )
        )
    return result


def build_plan(state: dict[str, Any], image: str, games_per_item: int) -> dict[str, object]:
    if not IMAGE.fullmatch(image):
        raise ValueError("V3 worker image must be an immutable registry digest")
    approved_readiness_sha256 = state.get("approved_readiness_sha256")
    if not isinstance(approved_readiness_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", approved_readiness_sha256
    ):
        raise ValueError("V3 collection requires the checksum-bound readiness approval")
    phase = str(state["phase"])
    items: list[ContainerInput] = []
    if phase == "bootstrap_collecting":
        components = [
            ("greedy", 100_000, 1_000_000_000),
            ("v1-direct", 200_000, 1_100_000_000),
            ("mixed-frozen", 100_000, 1_300_000_000),
            ("rare-softmax", 100_000, 1_400_000_000),
        ]
        for component, games, first in components:
            items.extend(
                _items(
                    component=component,
                    games=games,
                    first_game_index=first,
                    games_per_item=games_per_item,
                    approved_readiness_sha256=approved_readiness_sha256,
                )
            )
    else:
        match = re.fullmatch(r"cycle-(\d{2})-collecting", phase)
        if match is None:
            raise ValueError(f"phase {phase} does not schedule collection")
        cycle = int(match.group(1))
        items = _items(
            component="expert-iteration",
            games=10_000,
            first_game_index=2_000_000_000 + cycle * 10_000,
            games_per_item=games_per_item,
            cycle=cycle,
            epsilon=EXPLORATION[cycle - 1],
            approved_readiness_sha256=approved_readiness_sha256,
        )
    container = ContainerSpec(image=image)
    resources = Resources(cpu=1, memory_gib=1.5, disk_gib=1)
    return {
        "schema_id": "cascadia-v3-bacalhau-collection-plan-v1",
        "phase": phase,
        "image": container.image,
        "resources_per_item": resources.bacalhau(),
        "scheduler_owns_placement": True,
        "manual_host_sharding": False,
        "required_artifacts": {
            "campaign_state": "/inputs/control/campaign-state.json",
            "approved_readiness_sha256": approved_readiness_sha256,
            "qualified_v1": "/inputs/v1/qualified-v1.bin",
            "newest_v3": "/inputs/v3/newest" if phase.startswith("cycle-") else None,
            "prior_v3_pool": (
                [f"/inputs/v3/cycle-{cycle:02d}" for cycle in range(1, int(phase[6:8]))]
                if phase.startswith("cycle-")
                else []
            ),
            "transport": "content-addressed-object-store",
        },
        "items": [
            {
                "key": item.key,
                "args": list(item.args),
                "environment": dict(item.environment),
                "application_metadata": dict(item.application_metadata),
            }
            for item in items
        ],
        "games": sum(int(item.application_metadata["games"]) for item in items),
        "work_items": len(items),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-state", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--games-per-item", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_plan(_authorized(args.campaign_state), args.image, args.games_per_item)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"work_items": result["work_items"], "games": result["games"]}))


if __name__ == "__main__":
    main()
