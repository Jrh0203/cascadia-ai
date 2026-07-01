from __future__ import annotations

import json
from pathlib import Path

import pytest
from v3_cycle_train_pipeline import CycleTrainingPipelineError, _cached_validation
from v3_model_stage import _digest


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value))


def test_cached_validation_is_bound_to_every_source_and_cache_shard(tmp_path: Path) -> None:
    labels = []
    shards = []
    for index in range(2):
        source = tmp_path / f"source-{index}.v3l"
        cached = tmp_path / f"cache-{index}.v3t"
        source.write_bytes(f"source-{index}".encode())
        cached.write_bytes(f"cache-{index}".encode())
        labels.append(
            {
                "split": "validation",
                "path": str(source),
                "bytes": source.stat().st_size,
                "blake3": _digest(source),
                "candidate_estimates": 10 + index,
            }
        )
        shards.append(
            {
                "path": str(cached),
                "bytes": cached.stat().st_size,
                "blake3": _digest(cached),
                "source_path": str(source),
                "source_bytes": source.stat().st_size,
                "source_blake3": _digest(source),
                "roots": 5,
            }
        )
    label_manifest = tmp_path / "labels.json"
    cache_manifest = tmp_path / "cache.json"
    _write(label_manifest, {"passed": True, "files": labels})
    _write(
        cache_manifest,
        {
            "schema_id": "cascadia-v3-validation-cache-v1",
            "passed": True,
            "shards": shards,
            "totals": {
                "shards": 2,
                "roots": 10,
                "rows": 21,
                "bytes": sum(item["bytes"] for item in shards),
            },
        },
    )

    assert _cached_validation(label_manifest, cache_manifest) == [
        Path(item["path"]) for item in shards
    ]

    Path(shards[1]["path"]).write_bytes(b"tampered")
    with pytest.raises(CycleTrainingPipelineError, match="integrity"):
        _cached_validation(label_manifest, cache_manifest)
