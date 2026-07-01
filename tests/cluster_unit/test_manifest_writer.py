from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from cascadia_cluster.manifest_writer import write_manifest
from cascadia_cluster.results import validate_output_directory

ENTRYPOINT = Path(__file__).parents[2] / "infra/bacalhau/cluster-job-entrypoint.sh"


def test_manifest_writer_is_deterministic_and_importer_compatible(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested/result.json").write_text('{"score":95}\n')
    expected = write_manifest(
        tmp_path,
        command=["simulate", "--seed", "1"],
        application_metadata={"seed": 1},
        protocol_version="unit-v1",
    )
    observed = validate_output_directory(tmp_path)
    assert observed.protocol_version == "unit-v1"
    assert observed.application_metadata["seed"] == 1
    assert expected["files"][0]["path"] == "nested/result.json"


def test_manifest_writer_rejects_symlink_outputs(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_text("value")
    (tmp_path / "link").symlink_to(target)
    with pytest.raises(ValueError, match="symlinks"):
        write_manifest(
            tmp_path,
            command=["worker"],
            application_metadata={},
            protocol_version="unit-v1",
        )


def test_job_entrypoint_records_deterministic_failure_without_scheduler_retry(
    tmp_path: Path,
) -> None:
    environment = dict(os.environ)
    environment.update(
        {
            "CASCADIA_OUTPUT_ROOT": str(tmp_path),
            "CASCADIA_APPLICATION_METADATA_JSON": '{"item":"one"}',
            "CASCADIA_RETRYABLE_EXIT_CODES": "137",
            "PYTHONPATH": str(Path(__file__).parents[2] / "python"),
        }
    )
    result = subprocess.run(
        [str(ENTRYPOINT), "/bin/sh", "-c", "exit 2"],
        env=environment,
        check=False,
    )
    assert result.returncode == 0
    manifest = validate_output_directory(tmp_path)
    assert manifest.application_metadata["cascadia_application_status"] == "failed"
    assert manifest.application_metadata["cascadia_exit_code"] == 2


def test_job_entrypoint_preserves_retryable_exit_for_bacalhau(tmp_path: Path) -> None:
    environment = dict(os.environ)
    environment.update(
        {
            "CASCADIA_OUTPUT_ROOT": str(tmp_path),
            "CASCADIA_RETRYABLE_EXIT_CODES": "137",
            "PYTHONPATH": str(Path(__file__).parents[2] / "python"),
        }
    )
    result = subprocess.run(
        [str(ENTRYPOINT), "/bin/sh", "-c", "exit 137"],
        env=environment,
        check=False,
    )
    assert result.returncode == 137
    assert not (tmp_path / "manifest.json").exists()
