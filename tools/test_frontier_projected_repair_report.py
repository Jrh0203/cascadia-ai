from __future__ import annotations

import json
from pathlib import Path

import pytest
from frontier_projected_repair_report import (
    summarize_campaign_events,
    validate_replay_comparisons,
    validate_source_identities,
)


def test_source_identity_requires_four_matching_hosts(tmp_path: Path) -> None:
    paths = []
    for host in ("john1", "john2", "john3", "john4"):
        path = tmp_path / f"{host}.json"
        path.write_text(
            json.dumps(
                {
                    "host": host,
                    "files": 110,
                    "bundle_sha256": "a" * 64,
                }
            )
        )
        paths.append(path)
    assert validate_source_identities(paths)["files"] == 110
    paths[-1].write_text(
        json.dumps(
            {
                "host": "john4",
                "files": 110,
                "bundle_sha256": "b" * 64,
            }
        )
    )
    with pytest.raises(ValueError, match="not identical"):
        validate_source_identities(paths)


def test_replay_validation_requires_all_four_shards(tmp_path: Path) -> None:
    paths = []
    for shard_index in range(4):
        path = tmp_path / f"replay-{shard_index}.json"
        path.write_text(
            json.dumps(
                {
                    "shard_index": shard_index,
                    "scientific_payload_identical": True,
                }
            )
        )
        paths.append(path)
    assert len(validate_replay_comparisons(paths)["reports"]) == 4
    paths[-1].write_text(
        json.dumps(
            {
                "shard_index": 3,
                "scientific_payload_identical": False,
            }
        )
    )
    with pytest.raises(ValueError, match="payloads differ"):
        validate_replay_comparisons(paths)


def test_campaign_summary_tracks_four_origins_and_replays(tmp_path: Path) -> None:
    paths = []
    for index in range(8):
        replay = index >= 4
        name = f"job-{index}{'-replay' if replay else ''}"
        path = tmp_path / f"{name}.jsonl"
        start = 100.0 + index
        path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "event": "started",
                            "name": name,
                            "host": f"john{index % 4 + 1}",
                            "started_unix_seconds": start,
                        }
                    ),
                    json.dumps(
                        {
                            "event": "finished",
                            "name": name,
                            "ended_unix_seconds": start + 10.0,
                            "elapsed_seconds": 10.0,
                            "return_code": 0,
                        }
                    ),
                ]
            )
            + "\n"
        )
        paths.append(path)
    summary = summarize_campaign_events(paths)
    assert summary["origin_makespan_seconds"] == 13.0
    assert summary["end_to_end_makespan_seconds"] == 17.0
    assert summary["total_job_seconds"] == 80.0
    assert summary["confirmation_compute_fraction"] == 0.5
