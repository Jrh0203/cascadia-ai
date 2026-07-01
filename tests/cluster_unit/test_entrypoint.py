from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
ENTRYPOINT = REPOSITORY / "infra/bacalhau/cluster-job-entrypoint.sh"


def test_entrypoint_accepts_empty_input_checksum_map_and_writes_manifest(
    tmp_path: Path,
) -> None:
    output = tmp_path / "outputs"
    environment = dict(os.environ)
    environment.update(
        {
            "CASCADIA_APPLICATION_METADATA_JSON": "{}",
            "CASCADIA_INPUT_SHA256_JSON": "{}",
            "CASCADIA_OUTPUT_ROOT": str(output),
            "CASCADIA_PROTOCOL_VERSION": "cascadia-cluster-map-v1",
            "PYTHONPATH": str(REPOSITORY / "python"),
        }
    )
    subprocess.run(
        [
            str(ENTRYPOINT),
            "/bin/sh",
            "-c",
            'printf "ok\\n" > "$CASCADIA_OUTPUT_ROOT/result.txt"',
        ],
        check=True,
        env=environment,
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["files"][0]["path"] == "result.txt"


def test_entrypoint_exposes_and_removes_transient_scratch_root(tmp_path: Path) -> None:
    output = tmp_path / "outputs"
    environment = dict(os.environ)
    environment.update(
        {
            "CASCADIA_APPLICATION_METADATA_JSON": "{}",
            "CASCADIA_INPUT_SHA256_JSON": "{}",
            "CASCADIA_OUTPUT_ROOT": str(output),
            "CASCADIA_PROTOCOL_VERSION": "cascadia-cluster-map-v1",
            "PYTHONPATH": str(REPOSITORY / "python"),
        }
    )
    subprocess.run(
        [
            str(ENTRYPOINT),
            "/bin/sh",
            "-c",
            (
                'printf "%s\\n" "$CASCADIA_SCRATCH_ROOT" > '
                '"$CASCADIA_OUTPUT_ROOT/scratch-path.txt" && '
                'printf "scratch\\n" > "$CASCADIA_SCRATCH_ROOT/work.txt"'
            ),
        ],
        check=True,
        env=environment,
    )
    scratch = Path((output / "scratch-path.txt").read_text().strip())
    assert scratch.is_absolute()
    assert not scratch.exists()
    manifest = json.loads((output / "manifest.json").read_text())
    assert {file["path"] for file in manifest["files"]} == {"scratch-path.txt"}
