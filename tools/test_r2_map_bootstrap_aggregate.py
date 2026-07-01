from __future__ import annotations

import hashlib
import json
from pathlib import Path

import blake3
import pytest

import r2_map_bootstrap_aggregate as subject
from cascadia_mlx.r2_map_collector_audit import audit


def _packet(path: Path) -> Path:
    digest = "01" * 32
    workers = {
        host: [{"worker": 0, "first_game_index": index, "games": 1, "last_game_index": index}]
        for index, host in enumerate(("john1", "john2", "john3"))
    }
    value = {
        "campaign_id": "r2-map-expert-iteration-v1",
        "run_id": "fixture",
        "iteration": 0,
        "games_total": 3,
        "immutable_identity": {
            "collector_binary_blake3": digest,
            "source_sha256": digest,
            "serving_protocol_blake3": digest,
        },
        "hosts": {host: {"workers": values} for host, values in workers.items()},
    }
    path.write_text(json.dumps(value))
    return path


def _workers(tmp_path: Path) -> dict[str, Path]:
    digest = "01" * 32
    roots = {host: tmp_path / host for host in ("john1", "john2", "john3")}
    for index, (host, root) in enumerate(roots.items()):
        worker = root / "worker-0"
        worker.mkdir(parents=True)
        shard = worker / "shard-00000.r2sh"
        shard.write_bytes(f"shard-{host}".encode())
        manifest = {
            "schema_version": 1,
            "trajectory_schema_version": 2,
            "dataset_id": host,
            "game": {"player_count": 4, "mode": "Standard", "scoring_cards": {animal: "A" for animal in ("bear", "elk", "salmon", "hawk", "fox")}, "habitat_bonuses": False},
            "collection_kind": "bootstrap",
            "lease": {"campaign_id": "r2-map-expert-iteration-v1", "iteration": 0, "purpose": "bootstrap", "host_id": host, "first_game_index": index, "game_count": 1},
            "policy": {"kind": "bootstrap-greedy"},
            "protocols": {name: list(bytes.fromhex(digest)) for name in ("collector_hash", "source_hash", "serving_protocol_hash")},
            "requested_games": 1,
            "completed_games": 1,
            "primary_example_count": 80,
            "shards": [{"file": shard.name, "first_game_index": index, "game_count": 1, "primary_example_count": 80, "byte_count": shard.stat().st_size, "blake3": blake3.blake3(shard.read_bytes()).hexdigest()}],
        }
        (worker / "dataset.json").write_text(json.dumps(manifest))
        (worker / ".collector.lock").write_bytes(b"")
        payload = root / f"worker-{index}-collector-payload.json"
        payload.write_text(json.dumps(manifest))
        audit(
            dataset=worker,
            validation_payload=payload,
            validation_manifest=worker / "validation.json",
            receipt_path=worker / "completion-audit.json",
            semantic_validation_proof="collector-zero-exit",
        )
        payload.unlink()
        checksums = []
        for name in (
            ".collector.lock",
            "completion-audit.json",
            "dataset.json",
            shard.name,
            "validation.json",
        ):
            digest_value = hashlib.sha256((worker / name).read_bytes()).hexdigest()
            checksums.append(f"{digest_value}  ./{name}")
        (worker / "copied-files.sha256").write_text("\n".join(checksums) + "\n")
    return roots


def test_aggregate_validates_and_hardlinks(tmp_path: Path) -> None:
    roots = _workers(tmp_path)
    receipt = subject.aggregate(packet_path=_packet(tmp_path / "packet.json"), host_roots=roots, output_path=tmp_path / "receipt.json", flat_shard_root=tmp_path / "flat", require_worker_validation=True)
    assert receipt["games"] == 3
    assert receipt["primary_example_count"] == 240
    assert receipt["worker_datasets"] == 3
    assert len(list((tmp_path / "flat").iterdir())) == 3
    assert all(path.stat().st_nlink == 2 for path in (tmp_path / "flat").iterdir())


def test_aggregate_rejects_shard_tamper(tmp_path: Path) -> None:
    roots = _workers(tmp_path)
    (roots["john2"] / "worker-0/shard-00000.r2sh").write_bytes(b"tampered")
    with pytest.raises(subject.AggregateError, match="shard contract"):
        subject.aggregate(packet_path=_packet(tmp_path / "packet.json"), host_roots=roots, output_path=tmp_path / "receipt.json", flat_shard_root=tmp_path / "flat", require_worker_validation=True)
