from __future__ import annotations

import hashlib
import json
from pathlib import Path

import blake3
import v3_teacher_labels as labels


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value))


def test_cycle_label_corpus_accepts_exact_2500_roots(tmp_path: Path) -> None:
    cycle = 4
    image = "registry/v3@sha256:" + "a" * 64
    readiness = "b" * 64
    roots = tmp_path / "roots"
    accepted = tmp_path / "accepted"
    roots.mkdir()
    accepted.mkdir()
    inputs = []
    for index in range(25):
        root = roots / f"teacher-{index:05d}.v3r"
        root.write_bytes(f"root-{index}".encode())
        inputs.append(
            {
                "source_roots": root.name,
                "source_sha256": hashlib.sha256(root.read_bytes()).hexdigest(),
                "source_bytes": root.stat().st_size,
            }
        )
        directory = accepted / f"label-{index:05d}"
        directory.mkdir()
        label = directory / f"teacher-{index:05d}.v3l"
        label.write_bytes(f"label-{index}".encode())
        _write(
            directory / f"teacher-{index:05d}.receipt.json",
            {
                "schema_id": "cascadia-v3-teacher-label-shard-receipt-v1",
                "passed": True,
                "scientific_eligible": True,
                "cycle": cycle,
                "rollouts_per_root": 600,
                "approved_readiness_sha256": readiness,
                "campaign_state_sha256": "state-hash",
                "input": str(root),
                "roots": 100,
                "candidate_estimates": 3_200,
                "output_bytes": label.stat().st_size,
                "output_blake3": blake3.blake3(label.read_bytes()).hexdigest(),
            },
        )
    completion = tmp_path / "completion.json"
    state = tmp_path / "state.json"
    _write(
        completion,
        {
            "schema_id": "cascadia-v3-cluster-stage-completion-v1",
            "passed": True,
            "image_digest": image,
            "work_items": 25,
            "succeeded": 25,
            "totals": {"roots": 2_500, "rollouts": 1_500_000},
            "inputs": inputs,
        },
    )
    _write(
        state,
        {
            "phase": "cycle-04-labeling",
            "protected_seed_values_opened": False,
            "approved_readiness_sha256": readiness,
            "state_sha256": "state-hash",
        },
    )
    result = labels.aggregate(
        completion_path=completion,
        accepted_root=accepted,
        root_directory=roots,
        campaign_state=state,
        image_digest=image,
        cycle=cycle,
    )
    assert result["passed"] is True
    assert result["teacher_roots"] == 2_500
    assert result["validation_roots"] == 0
    assert result["rollouts"] == 1_500_000
