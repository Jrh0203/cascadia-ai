#!/usr/bin/env python3
"""Aggregate exact teacher labels without materializing sparse training rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import blake3


class LabelError(ValueError):
    """Teacher labels are incomplete or differ from their immutable roots."""


def _digest(path: Path, algorithm: str) -> str:
    digest = blake3.blake3() if algorithm == "blake3" else hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def aggregate(
    *,
    completion_path: Path,
    accepted_root: Path,
    root_directory: Path,
    campaign_state: Path,
    image_digest: str,
    cycle: int | None = None,
) -> dict[str, Any]:
    completion = json.loads(completion_path.read_text())
    state = json.loads(campaign_state.read_text())
    totals = completion.get("totals", {})
    if cycle is not None and not 1 <= cycle <= 10:
        raise LabelError("teacher-label cycle is outside 1..=10")
    expected_items = 120 if cycle is None else 25
    expected_roots = 120_000 if cycle is None else 2_500
    expected_rollouts = expected_roots * 600
    expected_phase = "bootstrap_labeling" if cycle is None else f"cycle-{cycle:02d}-labeling"
    if (
        completion.get("schema_id") != "cascadia-v3-cluster-stage-completion-v1"
        or completion.get("passed") is not True
        or completion.get("image_digest") != image_digest
        or completion.get("work_items") != expected_items
        or completion.get("succeeded") != expected_items
        or totals.get("roots") != expected_roots
        or totals.get("rollouts") != expected_rollouts
        or state.get("phase") != expected_phase
        or state.get("protected_seed_values_opened") is not False
    ):
        raise LabelError("teacher-label completion or campaign phase is invalid")
    root_paths = sorted(root_directory.glob("*.v3r"))
    if len(root_paths) != expected_items:
        raise LabelError(f"root shard directory does not contain {expected_items} shards")
    expected_inputs = {
        (
            path.name,
            _digest(path, "sha256"),
            path.stat().st_size,
        )
        for path in root_paths
    }
    try:
        recorded_inputs = {
            (
                item["source_roots"],
                item["source_sha256"],
                int(item["source_bytes"]),
            )
            for item in completion["inputs"]
        }
    except (KeyError, TypeError, ValueError) as error:
        raise LabelError("label completion input identities are malformed") from error
    if recorded_inputs != expected_inputs:
        raise LabelError("labeled root inputs differ from the split corpus")
    item_directories = [
        path
        for path in accepted_root.iterdir()
        if path.is_dir() and path.name != ".receipts"
    ]
    if len(item_directories) != expected_items:
        raise LabelError("accepted label artifact count differs")
    files = []
    split_roots: Counter[str] = Counter()
    candidate_estimates = 0
    bridge_totals: Counter[str] = Counter()
    for directory in sorted(item_directories):
        labels = sorted(directory.glob("*.v3l"))
        receipts = sorted(directory.glob("*.receipt.json"))
        if len(labels) != 1 or len(receipts) != 1:
            raise LabelError(f"label artifact set is incomplete for {directory.name}")
        label = labels[0]
        receipt = json.loads(receipts[0].read_text())
        source = Path(str(receipt.get("input", ""))).name
        split = "teacher" if source.startswith("teacher-") else "validation"
        if (
            source not in {path.name for path in root_paths}
            or receipt.get("schema_id") != "cascadia-v3-teacher-label-shard-receipt-v1"
            or receipt.get("cycle") != cycle
            or receipt.get("passed") is not True
            or receipt.get("scientific_eligible") is not True
            or receipt.get("rollouts_per_root") != 600
            or receipt.get("approved_readiness_sha256")
            != state.get("approved_readiness_sha256")
            or receipt.get("campaign_state_sha256") != state.get("state_sha256")
            or receipt.get("output_bytes") != label.stat().st_size
            or receipt.get("output_blake3") != _digest(label, "blake3")
        ):
            raise LabelError(f"label receipt differs for {directory.name}")
        split_roots[split] += int(receipt["roots"])
        candidate_estimates += int(receipt["candidate_estimates"])
        diagnostics = receipt.get("bridge_diagnostics", {})
        if isinstance(diagnostics, dict):
            for key, value in diagnostics.items():
                if isinstance(value, int) and not isinstance(value, bool):
                    bridge_totals[key] += value
        files.append(
            {
                "item": directory.name,
                "split": split,
                "source_root_shard": source,
                "roots": receipt["roots"],
                "candidate_estimates": receipt["candidate_estimates"],
                "path": str(label.resolve()),
                "bytes": label.stat().st_size,
                "blake3": receipt["output_blake3"],
                "sha256": _digest(label, "sha256"),
            }
        )
    expected_splits = (
        Counter({"teacher": 100_000, "validation": 20_000})
        if cycle is None
        else Counter({"teacher": 2_500})
    )
    if split_roots != expected_splits:
        raise LabelError(f"teacher/validation root totals differ: {split_roots}")
    result = {
        "schema_id": (
            "cascadia-v3-bootstrap-teacher-label-corpus-v1"
            if cycle is None
            else "cascadia-v3-expert-cycle-teacher-label-corpus-v1"
        ),
        "passed": True,
        "scientific_eligible": True,
        "approved_readiness_sha256": state["approved_readiness_sha256"],
        "campaign_state_sha256": state["state_sha256"],
        "image_digest": image_digest,
        "teacher_id": "qualified-v1-direct-top32-terminal-r600-sequential-halving-v1",
        "cycle": cycle,
        "roots": expected_roots,
        "teacher_roots": 100_000 if cycle is None else 2_500,
        "validation_roots": 20_000 if cycle is None else 0,
        "rollouts": expected_rollouts,
        "candidate_estimates": candidate_estimates,
        "bridge_diagnostics": dict(sorted(bridge_totals.items())),
        "compact_streaming": True,
        "materialized_sparse_training_rows": False,
        "files": files,
        "total_bytes": sum(item["bytes"] for item in files),
        "completion": {
            "path": str(completion_path.resolve()),
            "sha256": _digest(completion_path, "sha256"),
        },
        "protected_seed_values_opened": False,
    }
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["canonical_sha256"] = hashlib.sha256(canonical).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument("--accepted-root", type=Path, required=True)
    parser.add_argument("--root-directory", type=Path, required=True)
    parser.add_argument("--campaign-state", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--cycle", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = aggregate(
            completion_path=args.completion,
            accepted_root=args.accepted_root,
            root_directory=args.root_directory,
            campaign_state=args.campaign_state,
            image_digest=args.image,
            cycle=args.cycle,
        )
    except (LabelError, OSError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    _write_atomic(args.output, result)
    print(
        json.dumps(
            {"passed": True, "roots": result["roots"], "bytes": result["total_bytes"]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
