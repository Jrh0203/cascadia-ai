"""JSONL replay shard utilities for validated search-root records."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .schema import SCHEMA_ID, validate_replay_manifest, validate_search_root_record


def canonical_record_line(record: dict[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"


def records_checksum(records: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(canonical_record_line(record).encode("utf-8"))
    return digest.hexdigest()


def write_replay_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            validate_search_root_record(record)
            handle.write(canonical_record_line(record))


def read_replay_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            try:
                validate_search_root_record(record)
            except Exception as exc:  # pragma: no cover - preserves source line context.
                raise ValueError(f"{path}:{line_no}: invalid search root record") from exc
            records.append(record)
    if not records:
        raise ValueError(f"{path}: replay shard has no records")
    return records


def replay_manifest_for_records(
    records: list[dict[str, Any]],
    *,
    source_generator: str,
    seed_domain: str,
    scientific_eligibility: str = "dry_run",
    format: str = "jsonl",
) -> dict[str, Any]:
    schema_id = records[0].get("schema_id", SCHEMA_ID) if records else SCHEMA_ID
    manifest = {
        "schema_id": schema_id,
        "source_generator": source_generator,
        "seed_domain": seed_domain,
        "record_count": len(records),
        "checksum": records_checksum(records),
        "scientific_eligibility": scientific_eligibility,
        "created_at_utc": datetime(2026, 6, 29, tzinfo=UTC).isoformat(),
        "format": format,
        "notes": "CPU/GPU smoke replay shard; not training evidence.",
    }
    validate_replay_manifest(manifest)
    return manifest
