from __future__ import annotations

import hashlib
import json
from pathlib import Path

import blake3
import pytest
import v3_teacher_label_repair as repair

OLD_IMAGE = "registry/v3@sha256:" + "1" * 64
NEW_IMAGE = "registry/v3@sha256:" + "2" * 64
OLD_REQUEST = "original-label-request"
NEW_REQUEST = "coordinate-frame-repair-request"
READINESS = "3" * 64
STATE_HASH = "4" * 64
REPAIRED_INDICES = {38, 40, 42, 43, 44, 50, 54, 68, 110}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _root_name(index: int) -> str:
    return f"teacher-{index:05d}.v3r" if index < 100 else f"validation-{index - 100:05d}.v3r"


def _artifact(directory: Path, root: Path) -> None:
    directory.mkdir(parents=True)
    stem = root.stem
    label = directory / f"{stem}.v3l"
    label.write_bytes(f"label:{root.name}".encode())
    _write_json(
        directory / f"{stem}.receipt.json",
        {
            "schema_id": "cascadia-v3-teacher-label-shard-receipt-v1",
            "passed": True,
            "scientific_eligible": True,
            "teacher_id": "qualified-v1-direct-top32-terminal-r600-sequential-halving-v1",
            "cycle": None,
            "roots": 1_000,
            "candidate_estimates": 32_000,
            "rollouts_per_root": 600,
            "input": str(root),
            "input_bytes": root.stat().st_size,
            "input_blake3": blake3.blake3(root.read_bytes()).hexdigest(),
            "output_bytes": label.stat().st_size,
            "output_blake3": blake3.blake3(label.read_bytes()).hexdigest(),
            "approved_readiness_sha256": READINESS,
            "campaign_state_sha256": STATE_HASH,
            "bridge_diagnostics": {"states_translated": 1_000},
        },
    )


def _fixture(tmp_path: Path) -> dict[str, Path]:
    roots = tmp_path / "roots"
    original = tmp_path / "original"
    repaired = tmp_path / "repair"
    roots.mkdir()
    original.mkdir()
    repaired.mkdir()
    request_items = []
    repair_inputs = []
    for index in range(120):
        root = roots / _root_name(index)
        root.write_bytes(f"root:{index}".encode())
        source_sha256 = hashlib.sha256(root.read_bytes()).hexdigest()
        request_items.append(
            {
                "key": f"label-{index:05d}",
                "job_payload": {
                    "Meta": {
                        "cascadia.app.source_roots": root.name,
                        "cascadia.app.source_sha256": source_sha256,
                        "cascadia.app.source_bytes": str(root.stat().st_size),
                    }
                },
            }
        )
        original_item = original / f"label-{index:05d}"
        if index in REPAIRED_INDICES:
            original_item.mkdir()
            _write_json(original_item / "application-failure.json", {"exit_code": 1})
            repair_item = f"label-{len(repair_inputs):05d}"
            repair_inputs.append(
                {
                    "item": repair_item,
                    "source_roots": root.name,
                    "source_sha256": source_sha256,
                    "source_bytes": str(root.stat().st_size),
                }
            )
            _artifact(repaired / repair_item, root)
        else:
            _artifact(original_item, root)
    paths = {
        "roots": roots,
        "original": original,
        "repaired": repaired,
        "request": tmp_path / "request.json",
        "repair_completion": tmp_path / "repair-completion.json",
        "state": tmp_path / "state.json",
        "reconciled": tmp_path / "reconciled",
        "completion": tmp_path / "completion.json",
        "corpus": tmp_path / "corpus.json",
    }
    _write_json(
        paths["request"],
        {
            "request_id": OLD_REQUEST,
            "image_digest": OLD_IMAGE,
            "items": request_items,
        },
    )
    _write_json(
        paths["repair_completion"],
        {
            "schema_id": "cascadia-v3-cluster-stage-completion-v1",
            "passed": True,
            "request_id": NEW_REQUEST,
            "image_digest": NEW_IMAGE,
            "work_items": 9,
            "succeeded": 9,
            "inputs": repair_inputs,
        },
    )
    _write_json(
        paths["state"],
        {
            "schema_id": "cascadia-v3-campaign-state-v1",
            "phase": "bootstrap_labeling",
            "phase2_authorized": True,
            "protected_seed_values_opened": False,
            "approved_readiness_sha256": READINESS,
            "state_sha256": STATE_HASH,
        },
    )
    return paths


def _reconcile(paths: dict[str, Path]) -> tuple[dict, dict]:
    return repair.reconcile(
        original_request_path=paths["request"],
        original_accepted_root=paths["original"],
        repair_completion_path=paths["repair_completion"],
        repair_accepted_root=paths["repaired"],
        root_directory=paths["roots"],
        campaign_state_path=paths["state"],
        reconciled_root=paths["reconciled"],
        completion_output=paths["completion"],
        corpus_output=paths["corpus"],
    )


def test_reconcile_accepts_exactly_111_original_and_nine_repair_shards(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    completion, corpus = _reconcile(paths)
    assert completion["passed"] is True
    assert completion["original"]["accepted_shards"] == 111
    assert completion["repair"]["accepted_shards"] == 9
    assert len(completion["rejected_partial_shards"]) == 9
    assert corpus["roots"] == 120_000
    assert corpus["rollouts"] == 72_000_000
    assert corpus["image_lineage"] == {OLD_IMAGE: 111, NEW_IMAGE: 9}
    assert len(corpus["files"]) == 120
    source = paths["repaired"] / "label-00000" / "teacher-00038.v3l"
    linked = paths["reconciled"] / "teacher-00038" / "teacher-00038.v3l"
    assert source.stat().st_ino == linked.stat().st_ino


def test_reconcile_rejects_a_missing_repair_root(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    completion = json.loads(paths["repair_completion"].read_text())
    completion["inputs"].pop()
    completion["work_items"] = 8
    completion["succeeded"] = 8
    _write_json(paths["repair_completion"], completion)
    with pytest.raises(repair.RepairError, match="exactly the rejected roots"):
        _reconcile(paths)
