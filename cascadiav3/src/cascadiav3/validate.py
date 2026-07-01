"""CPU-only validation entry point for the pre-GPU scaffold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .fixtures import (
    radius6_census,
    tiny_replay_manifest,
    tiny_replay_records,
    tiny_search_root_record,
)
from .hex import RADIUS6_CELL_COUNT, RADIUS6_COORDS, cell_index, coord_for_index, coord_ref
from .model_smoke import mock_forward, validate_mock_output
from .replay import read_replay_jsonl, replay_manifest_for_records, write_replay_jsonl
from .schema import validate_replay_manifest, validate_search_root_record


def _cgab_demo_edges(root: dict[str, Any]) -> list[dict[str, Any]]:
    edges = []
    for idx, action in enumerate(root["legal_actions"]):
        edges.append(
            {
                "template_id": "action_draft_slot",
                "source": f"action:{action['action_id']}",
                "target": f"market_slot:{action['draft_slot']}",
                "action_index": idx,
            }
        )
        edges.append(
            {
                "template_id": "action_target_tile_coordinate",
                "source": f"action:{action['action_id']}",
                "target": action["target_coord_ref"],
                "action_index": idx,
            }
        )
    return edges


def run_validation() -> dict[str, Any]:
    indexes = [cell_index(coord.q, coord.r) for coord in RADIUS6_COORDS]
    if sorted(indexes) != list(range(RADIUS6_CELL_COUNT)):
        raise AssertionError("radius-6 indexes are not contiguous 0..126")
    for idx in range(RADIUS6_CELL_COUNT):
        coord = coord_for_index(idx)
        if cell_index(coord.q, coord.r) != idx:
            raise AssertionError("radius-6 index roundtrip failed")

    overflow = coord_ref(7, 0, owner_seat=0, placement_id=999)
    if overflow["kind"] != "overflow" or overflow["s"] != -7:
        raise AssertionError("overflow coordinate is not exact")

    root = tiny_search_root_record()
    validate_search_root_record(root)
    manifest = tiny_replay_manifest(root)
    validate_replay_manifest(manifest)
    replay_records = tiny_replay_records()

    cgab_edges = _cgab_demo_edges(root)
    state_tokens = [
        {"token_kind": "GameToken", "turn_index": 0, "active_seat": 0},
        {"token_kind": "PlayerToken", "seat": 0},
        {"token_kind": "PlayerToken", "seat": 1},
        {"token_kind": "PlayerToken", "seat": 2},
        {"token_kind": "PlayerToken", "seat": 3},
    ]
    model_output = mock_forward(
        state_tokens=state_tokens,
        action_tokens=root["legal_actions"],
        cgab_edges=cgab_edges,
    )
    validate_mock_output(model_output, action_count=len(root["legal_actions"]))

    census = radius6_census(
        [
            {"q": 0, "r": 0},
            {"q": 1, "r": 0},
            {"q": -2, "r": 1},
            {"q": 6, "r": 0},
            {"q": 7, "r": 0},
        ]
    )

    return {
        "status": "pass",
        "radius6_cell_count": RADIUS6_CELL_COUNT,
        "root_action_count": len(root["legal_actions"]),
        "replay_record_count": len(replay_records),
        "replay_action_counts": [len(record["legal_actions"]) for record in replay_records],
        "cgab_edge_count": len(cgab_edges),
        "mock_model": {
            "legal_action_logits": len(model_output["legal_action_logits"]),
            "value_vector": len(model_output["value_vector"]),
            "rank_logits": [
                len(model_output["rank_logits"]),
                len(model_output["rank_logits"][0]),
            ],
            "score_decomposition_categories": sorted(model_output["score_decomposition"]),
        },
        "radius6_census": census,
        "manifest": manifest,
    }


def write_artifacts(root: Path) -> None:
    fixture_dir = root / "fixtures"
    report_dir = root / "reports"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    root_record = tiny_search_root_record()
    replay_records = tiny_replay_records()
    (fixture_dir / "tiny_search_root.json").write_text(
        json.dumps(root_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (fixture_dir / "tiny_replay_manifest.json").write_text(
        json.dumps(tiny_replay_manifest(root_record), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    replay_path = fixture_dir / "tiny_replay.jsonl"
    write_replay_jsonl(replay_path, replay_records)
    roundtrip_records = read_replay_jsonl(replay_path)
    (fixture_dir / "tiny_replay_shard_manifest.json").write_text(
        json.dumps(
            replay_manifest_for_records(
                roundtrip_records,
                source_generator="cascadiav3.fixtures.tiny_replay_records",
                seed_domain="fixed-demo-replay-seed",
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (report_dir / "pre_gpu_validation.json").write_text(
        json.dumps(run_validation(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-artifacts",
        action="store_true",
        help="write tiny fixtures and validation report under cascadiav3/",
    )
    parser.add_argument(
        "--root",
        default="cascadiav3",
        help="cascadiav3 root for artifact output",
    )
    args = parser.parse_args()

    result = run_validation()
    if args.write_artifacts:
        write_artifacts(Path(args.root))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
