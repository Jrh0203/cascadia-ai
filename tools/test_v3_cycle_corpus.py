from __future__ import annotations

import json
from pathlib import Path

import blake3
import v3_cycle_corpus as corpus


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value))


def test_exact_cycle_corpus_is_accepted(tmp_path: Path) -> None:
    cycle = 3
    accepted = tmp_path / "accepted"
    accepted.mkdir()
    first = 2_000_000_000 + cycle * 10_000
    for index in range(100):
        directory = accepted / f"item-{index:03d}"
        directory.mkdir()
        shard = directory / f"cycle-{index:03d}.v3g"
        shard.write_bytes(f"shard-{index}".encode())
        digest = blake3.blake3(shard.read_bytes()).hexdigest()
        _write(
            directory / f"cycle-{index:03d}.receipt.json",
            {
                "component": "expert-iteration",
                "cycle": cycle,
                "first_game_index": first + index * 100,
                "games": 100,
                "records": 100,
                "newest_model_seats_per_expert_game": 1,
                "bytes": shard.stat().st_size,
                "blake3": digest,
            },
        )
    collection = tmp_path / "collection.json"
    verification = tmp_path / "verification.json"
    state = tmp_path / "state.json"
    _write(
        collection,
        {"passed": True, "cycle": cycle, "work_items": 100, "totals": {"games": 10_000}},
    )
    _write(
        verification,
        {
            "passed": True,
            "work_items": 100,
            "totals": {"records": 10_000, "expanded_training_entries": 200_000},
        },
    )
    _write(
        state,
        {"phase": "cycle-03-collecting", "protected_seed_values_opened": False},
    )
    value = corpus.aggregate(
        cycle=cycle,
        collection=collection,
        verification=verification,
        accepted_root=accepted,
        campaign_state=state,
    )
    assert value["passed"] is True
    assert value["games"] == 10_000
    assert value["training_entries"] == 200_000
