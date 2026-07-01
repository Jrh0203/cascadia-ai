from __future__ import annotations

import json
from pathlib import Path

import pytest
from cascadia_mlx.run_manifest import (
    RunManifestError,
    source_provenance,
    validate_resume_manifest,
)


def test_resume_manifest_allows_only_epoch_budget_and_resume_flag_changes(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    original_training = {
        "train_dataset": "/data/train",
        "validation_dataset": "/data/validation",
        "run_dir": str(run),
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "seed": 7,
        "checkpoint_steps": 50,
        "resume": False,
        "model": {"architecture": "test"},
    }
    datasets = {"train_manifest_blake3": "train", "validation_manifest_blake3": "validation"}
    runtime = {"mlx_version": "test", "device": "gpu"}
    source = {"v2_source_blake3": "source"}
    (run / "run.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "training": original_training,
                "datasets": datasets,
                "runtime": runtime,
                "source": source,
            }
        )
    )

    resumed_training = {**original_training, "epochs": 20, "resume": True}
    validate_resume_manifest(
        run,
        training=resumed_training,
        datasets=datasets,
        runtime=runtime,
        source=source,
    )

    with pytest.raises(RunManifestError, match="training configuration"):
        validate_resume_manifest(
            run,
            training={**resumed_training, "learning_rate": 2e-4},
            datasets=datasets,
            runtime=runtime,
            source=source,
        )
    with pytest.raises(RunManifestError, match="dataset identity"):
        validate_resume_manifest(
            run,
            training=resumed_training,
            datasets={**datasets, "train_manifest_blake3": "changed"},
            runtime=runtime,
            source=source,
        )
    with pytest.raises(RunManifestError, match="source digest"):
        validate_resume_manifest(
            run,
            training=resumed_training,
            datasets=datasets,
            runtime=runtime,
            source={"v2_source_blake3": "changed"},
        )


def test_source_provenance_digest_tracks_v2_source_content(tmp_path: Path) -> None:
    package = tmp_path / "python" / "cascadia_mlx"
    package.mkdir(parents=True)
    source_file = package / "model.py"
    source_file.write_text("VALUE = 1\n")
    (tmp_path / "Cargo.toml").write_text("[workspace]\n")

    before = source_provenance(tmp_path)["v2_source_blake3"]
    source_file.write_text("VALUE = 2\n")
    after = source_provenance(tmp_path)["v2_source_blake3"]

    assert before != after


def test_source_provenance_covers_linked_legacy_teacher_crates(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "legacy" / "crates" / "cascadia-ai" / "src"
    legacy.mkdir(parents=True)
    source_file = legacy / "eval.rs"
    source_file.write_text("pub const VALUE: u8 = 1;\n")

    before = source_provenance(tmp_path)["v2_source_blake3"]
    source_file.write_text("pub const VALUE: u8 = 2;\n")
    after = source_provenance(tmp_path)["v2_source_blake3"]

    assert before != after
