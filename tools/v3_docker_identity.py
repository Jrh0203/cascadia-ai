#!/usr/bin/env python3
"""Certify one John1-built V3 image executed unchanged on John2 and John3."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def certify(publication: dict[str, Any], retry: dict[str, Any]) -> dict[str, object]:
    image = publication.get("image_digest")
    placements = retry.get("initial_placements")
    nodes = sorted(set(placements.values())) if isinstance(placements, dict) else []
    passed = (
        publication.get("schema_id") == "cascadia.cluster.image-publication.v1"
        and publication.get("build_host") == "john1"
        and isinstance(image, str)
        and retry.get("passed") is True
        and retry.get("image_digest") == image
        and nodes == ["john2", "john3"]
    )
    return {
        "schema_id": "cascadia-v3-docker-identity-v1",
        "passed": passed,
        "build_host": publication.get("build_host"),
        "image_digest": image,
        "source_commit": publication.get("source_commit"),
        "source_identity": publication.get("source_identity"),
        "execution_nodes": nodes,
        "worker_retry_passed": retry.get("passed") is True,
    }


def _write_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication", type=Path, required=True)
    parser.add_argument("--retry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = certify(_read(args.publication), _read(args.retry))
    _write_atomic(args.output, result)
    print(json.dumps(result, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
