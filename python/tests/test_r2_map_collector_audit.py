from __future__ import annotations

import hashlib
import json
from pathlib import Path

import blake3
import pytest

from cascadia_mlx.r2_map_collector_audit import CollectorAuditError, audit


def _dataset(root: Path) -> tuple[Path, Path]:
    root.mkdir()
    shard = root / "shard-00000.r2sh"
    shard.write_bytes(b"complete replay shard")
    digest = "01" * 32
    manifest = {
        "schema_version": 1,
        "trajectory_schema_version": 2,
        "dataset_id": "fixture",
        "collection_kind": "bootstrap",
        "lease": {
            "purpose": "bootstrap",
            "host_id": "john1",
            "first_game_index": 17,
            "game_count": 1,
        },
        "shard_games": 256,
        "requested_games": 1,
        "completed_games": 1,
        "primary_example_count": 80,
        "shards": [
            {
                "file": shard.name,
                "first_game_index": 17,
                "game_count": 1,
                "primary_example_count": 80,
                "byte_count": shard.stat().st_size,
                "blake3": blake3.blake3(shard.read_bytes()).hexdigest(),
            }
        ],
        "protocols": {"source_hash": list(bytes.fromhex(digest))},
    }
    manifest_path = root / "dataset.json"
    manifest_path.write_text(json.dumps(manifest))
    (root / ".collector.lock").write_text("pid=1\n")
    payload = root.parent / "collector.json"
    payload.write_text(json.dumps(manifest))
    return root, payload


def test_audit_binds_zero_exit_payload_and_shards(tmp_path: Path) -> None:
    dataset, payload = _dataset(tmp_path / "worker-0")
    receipt = audit(
        dataset=dataset,
        validation_payload=payload,
        validation_manifest=dataset / "validation.json",
        receipt_path=dataset / "completion-audit.json",
        semantic_validation_proof="collector-zero-exit",
        copy_manifest=dataset / "copied-files.sha256",
    )
    assert receipt["result"] == "pass"
    assert receipt["game_count"] == 1
    assert json.loads((dataset / "validation.json").read_text()) == json.loads(
        (dataset / "dataset.json").read_text()
    )
    assert "./completion-audit.json" in (dataset / "copied-files.sha256").read_text()
    unsigned = dict(receipt)
    claimed = unsigned.pop("receipt_sha256")
    assert claimed == hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_audit_rejects_payload_or_shard_drift(tmp_path: Path) -> None:
    dataset, payload = _dataset(tmp_path / "worker-0")
    value = json.loads(payload.read_text())
    value["completed_games"] = 0
    payload.write_text(json.dumps(value))
    with pytest.raises(CollectorAuditError, match="payload differs"):
        audit(
            dataset=dataset,
            validation_payload=payload,
            validation_manifest=dataset / "validation.json",
            receipt_path=dataset / "completion-audit.json",
            semantic_validation_proof="collector-zero-exit",
        )

    payload.write_text((dataset / "dataset.json").read_text())
    (dataset / "shard-00000.r2sh").write_bytes(b"tampered")
    with pytest.raises(CollectorAuditError, match="shard hash differs"):
        audit(
            dataset=dataset,
            validation_payload=payload,
            validation_manifest=dataset / "validation.json",
            receipt_path=dataset / "completion-audit.json",
            semantic_validation_proof="collector-zero-exit",
        )
