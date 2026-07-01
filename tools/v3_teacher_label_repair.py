#!/usr/bin/env python3
"""Reconcile complete original V3 teacher labels with bounded repair shards.

The tool never accepts partial artifacts.  It validates every root, receipt,
label checksum, campaign-state identity, request/image lineage, and aggregate
total before creating a hard-linked, immutable training corpus.
"""

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


class RepairError(ValueError):
    """Original and repair artifacts do not form one exact eligible corpus."""


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RepairError(f"{path} is not a JSON object")
    return value


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


def _request_inputs(request: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in request.get("items", []):
        payload = item.get("job_payload", {})
        metadata = payload.get("Meta", {})
        source = metadata.get("cascadia.app.source_roots")
        if not isinstance(source, str) or source in result:
            raise RepairError("request contains malformed or duplicate root identities")
        result[source] = {
            "item": item.get("key"),
            "source_sha256": metadata.get("cascadia.app.source_sha256"),
            "source_bytes": metadata.get("cascadia.app.source_bytes"),
        }
    return result


def _completion_inputs(completion: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in completion.get("inputs", []):
        source = item.get("source_roots")
        if not isinstance(source, str) or source in result:
            raise RepairError("repair completion contains malformed root identities")
        result[source] = item
    return result


def _artifact(
    *,
    item_directory: Path,
    root: Path,
    state: dict[str, Any],
    image_digest: str,
    request_id: str,
) -> dict[str, Any]:
    if (item_directory / "application-failure.json").exists():
        raise RepairError(f"application failure cannot enter the corpus: {item_directory}")
    labels = sorted(item_directory.glob("*.v3l"))
    receipts = sorted(item_directory.glob("*.receipt.json"))
    if len(labels) != 1 or len(receipts) != 1:
        raise RepairError(f"label artifact set is incomplete: {item_directory}")
    label = labels[0]
    receipt_path = receipts[0]
    receipt = _read(receipt_path)
    if (
        receipt.get("schema_id") != "cascadia-v3-teacher-label-shard-receipt-v1"
        or receipt.get("passed") is not True
        or receipt.get("scientific_eligible") is not True
        or receipt.get("cycle") is not None
        or receipt.get("teacher_id")
        != "qualified-v1-direct-top32-terminal-r600-sequential-halving-v1"
        or receipt.get("roots") != 1_000
        or receipt.get("rollouts_per_root") != 600
        or not isinstance(receipt.get("candidate_estimates"), int)
        or receipt["candidate_estimates"] < 1_000
        or Path(str(receipt.get("input", ""))).name != root.name
        or receipt.get("input_bytes") != root.stat().st_size
        or receipt.get("input_blake3") != _digest(root, "blake3")
        or receipt.get("output_bytes") != label.stat().st_size
        or receipt.get("output_blake3") != _digest(label, "blake3")
        or receipt.get("approved_readiness_sha256") != state.get("approved_readiness_sha256")
        or receipt.get("campaign_state_sha256") != state.get("state_sha256")
    ):
        raise RepairError(f"label receipt differs from its root or campaign: {item_directory}")
    return {
        "source_root_shard": root.name,
        "roots": 1_000,
        "candidate_estimates": receipt["candidate_estimates"],
        "label": label,
        "receipt": receipt_path,
        "bridge_diagnostics": receipt.get("bridge_diagnostics", {}),
        "image_digest": image_digest,
        "request_id": request_id,
    }


def _link_exact(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size or _digest(
            destination, "sha256"
        ) != _digest(source, "sha256"):
            raise RepairError(f"existing reconciled artifact differs: {destination}")
        return
    os.link(source, destination)


def reconcile(
    *,
    original_request_path: Path,
    original_accepted_root: Path,
    repair_completion_path: Path,
    repair_accepted_root: Path,
    root_directory: Path,
    campaign_state_path: Path,
    reconciled_root: Path,
    completion_output: Path,
    corpus_output: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    state = _read(campaign_state_path)
    if (
        state.get("schema_id") != "cascadia-v3-campaign-state-v1"
        or state.get("phase") != "bootstrap_labeling"
        or state.get("phase2_authorized") is not True
        or state.get("protected_seed_values_opened") is not False
    ):
        raise RepairError("campaign is not at the sealed bootstrap-labeling gate")
    roots = {path.name: path for path in sorted(root_directory.glob("*.v3r"))}
    if len(roots) != 120:
        raise RepairError(f"expected 120 immutable root shards, found {len(roots)}")

    original_request = _read(original_request_path)
    original_image = original_request.get("image_digest")
    original_request_id = original_request.get("request_id")
    original_inputs = _request_inputs(original_request)
    if (
        not isinstance(original_image, str)
        or not isinstance(original_request_id, str)
        or set(original_inputs) != set(roots)
    ):
        raise RepairError("original request does not bind the complete root corpus")

    original: dict[str, dict[str, Any]] = {}
    rejected: set[str] = set()
    for source, metadata in original_inputs.items():
        root = roots[source]
        try:
            recorded_bytes = int(metadata["source_bytes"])
        except (TypeError, ValueError) as error:
            raise RepairError("original request source bytes are malformed") from error
        if (
            metadata["source_sha256"] != _digest(root, "sha256")
            or recorded_bytes != root.stat().st_size
            or not isinstance(metadata["item"], str)
        ):
            raise RepairError(f"original request input differs for {source}")
        item_directory = original_accepted_root / metadata["item"]
        try:
            original[source] = _artifact(
                item_directory=item_directory,
                root=root,
                state=state,
                image_digest=original_image,
                request_id=original_request_id,
            )
        except RepairError:
            rejected.add(source)

    repair_completion = _read(repair_completion_path)
    repair_image = repair_completion.get("image_digest")
    repair_request_id = repair_completion.get("request_id")
    repair_inputs = _completion_inputs(repair_completion)
    if (
        repair_completion.get("schema_id") != "cascadia-v3-cluster-stage-completion-v1"
        or repair_completion.get("passed") is not True
        or repair_completion.get("work_items") != len(rejected)
        or repair_completion.get("succeeded") != len(rejected)
        or not isinstance(repair_image, str)
        or not isinstance(repair_request_id, str)
        or set(repair_inputs) != rejected
    ):
        raise RepairError("repair completion does not replace exactly the rejected roots")

    repaired: dict[str, dict[str, Any]] = {}
    for source, metadata in repair_inputs.items():
        root = roots[source]
        try:
            source_bytes = int(metadata["source_bytes"])
        except (TypeError, ValueError) as error:
            raise RepairError("repair source bytes are malformed") from error
        if (
            metadata.get("source_sha256") != _digest(root, "sha256")
            or source_bytes != root.stat().st_size
            or not isinstance(metadata.get("item"), str)
        ):
            raise RepairError(f"repair request input differs for {source}")
        repaired[source] = _artifact(
            item_directory=repair_accepted_root / metadata["item"],
            root=root,
            state=state,
            image_digest=repair_image,
            request_id=repair_request_id,
        )

    combined = {**original, **repaired}
    if len(original) != 111 or len(repaired) != 9 or set(combined) != set(roots):
        raise RepairError(
            "expected 111 original plus 9 repair shards, "
            f"found {len(original)} plus {len(repaired)}"
        )

    files: list[dict[str, Any]] = []
    split_roots: Counter[str] = Counter()
    bridge_totals: Counter[str] = Counter()
    candidate_estimates = 0
    for source in sorted(combined):
        artifact = combined[source]
        split = "teacher" if source.startswith("teacher-") else "validation"
        split_roots[split] += artifact["roots"]
        candidate_estimates += artifact["candidate_estimates"]
        diagnostics = artifact["bridge_diagnostics"]
        if isinstance(diagnostics, dict):
            for key, value in diagnostics.items():
                if isinstance(value, int) and not isinstance(value, bool):
                    bridge_totals[key] += value
        destination = reconciled_root / Path(source).stem
        label_path = destination / f"{Path(source).stem}.v3l"
        receipt_path = destination / f"{Path(source).stem}.receipt.json"
        _link_exact(artifact["label"], label_path)
        _link_exact(artifact["receipt"], receipt_path)
        _write_atomic(
            destination / "lineage.json",
            {
                "schema_id": "cascadia-v3-teacher-label-reconciled-item-v1",
                "source_root_shard": source,
                "source_request_id": artifact["request_id"],
                "source_image_digest": artifact["image_digest"],
                "label_sha256": _digest(label_path, "sha256"),
                "receipt_sha256": _digest(receipt_path, "sha256"),
            },
        )
        files.append(
            {
                "item": Path(source).stem,
                "split": split,
                "source_root_shard": source,
                "roots": artifact["roots"],
                "candidate_estimates": artifact["candidate_estimates"],
                "path": str(label_path.resolve()),
                "bytes": label_path.stat().st_size,
                "blake3": _digest(label_path, "blake3"),
                "sha256": _digest(label_path, "sha256"),
                "source_request_id": artifact["request_id"],
                "source_image_digest": artifact["image_digest"],
            }
        )

    if split_roots != Counter({"teacher": 100_000, "validation": 20_000}):
        raise RepairError(f"reconciled split totals differ: {split_roots}")
    completion = {
        "schema_id": "cascadia-v3-reconciled-label-completion-v1",
        "passed": True,
        "scientific_eligible": True,
        "work_items": 120,
        "succeeded": 120,
        "roots": 120_000,
        "rollouts": 72_000_000,
        "original": {
            "request_id": original_request_id,
            "image_digest": original_image,
            "accepted_shards": len(original),
        },
        "repair": {
            "request_id": repair_request_id,
            "image_digest": repair_image,
            "accepted_shards": len(repaired),
            "completion_path": str(repair_completion_path.resolve()),
            "completion_sha256": _digest(repair_completion_path, "sha256"),
        },
        "rejected_partial_shards": sorted(rejected),
        "protected_seed_values_opened": False,
    }
    _write_atomic(completion_output, completion)
    corpus = {
        "schema_id": "cascadia-v3-bootstrap-teacher-label-corpus-v1",
        "passed": True,
        "scientific_eligible": True,
        "approved_readiness_sha256": state["approved_readiness_sha256"],
        "campaign_state_sha256": state["state_sha256"],
        "image_digests": [original_image, repair_image],
        "image_lineage": {
            original_image: len(original),
            repair_image: len(repaired),
        },
        "teacher_id": "qualified-v1-direct-top32-terminal-r600-sequential-halving-v1",
        "cycle": None,
        "roots": 120_000,
        "teacher_roots": 100_000,
        "validation_roots": 20_000,
        "rollouts": 72_000_000,
        "candidate_estimates": candidate_estimates,
        "bridge_diagnostics": dict(sorted(bridge_totals.items())),
        "compact_streaming": True,
        "materialized_sparse_training_rows": False,
        "files": files,
        "total_bytes": sum(item["bytes"] for item in files),
        "completion": {
            "path": str(completion_output.resolve()),
            "sha256": _digest(completion_output, "sha256"),
        },
        "protected_seed_values_opened": False,
    }
    canonical = json.dumps(corpus, sort_keys=True, separators=(",", ":")).encode()
    corpus["canonical_sha256"] = hashlib.sha256(canonical).hexdigest()
    _write_atomic(corpus_output, corpus)
    return completion, corpus


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-request", type=Path, required=True)
    parser.add_argument("--original-accepted-root", type=Path, required=True)
    parser.add_argument("--repair-completion", type=Path, required=True)
    parser.add_argument("--repair-accepted-root", type=Path, required=True)
    parser.add_argument("--root-directory", type=Path, required=True)
    parser.add_argument("--campaign-state", type=Path, required=True)
    parser.add_argument("--reconciled-root", type=Path, required=True)
    parser.add_argument("--completion-output", type=Path, required=True)
    parser.add_argument("--corpus-output", type=Path, required=True)
    args = parser.parse_args()
    try:
        completion, corpus = reconcile(
            original_request_path=args.original_request,
            original_accepted_root=args.original_accepted_root,
            repair_completion_path=args.repair_completion,
            repair_accepted_root=args.repair_accepted_root,
            root_directory=args.root_directory,
            campaign_state_path=args.campaign_state,
            reconciled_root=args.reconciled_root,
            completion_output=args.completion_output,
            corpus_output=args.corpus_output,
        )
    except (RepairError, OSError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    print(
        json.dumps(
            {
                "passed": True,
                "original_shards": completion["original"]["accepted_shards"],
                "repair_shards": completion["repair"]["accepted_shards"],
                "roots": corpus["roots"],
                "rollouts": corpus["rollouts"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
