#!/usr/bin/env python3
"""Deterministic host-local conversion for the ADR 0081 cluster corpus."""

# ruff: noqa: UP045 - cluster tools must run under the macOS system Python 3.9.

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

EXPERIMENT_ID = "complete-action-graded-oracle-ranker-v1"
HOST_SPLITS = {
    "john1": {
        "train": (61000, 61001, 61002),
        "validation": (61003,),
        "test": (61004,),
    },
    "john2": {
        "train": (61005, 61006),
        "validation": (61007,),
        "test": (61008,),
    },
    "john3": {
        "train": (61009, 61010),
        "validation": (61011,),
        "test": (61012,),
    },
}


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], log_path: Path) -> None:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout + completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with exit {completed.returncode}: {' '.join(command)}"
        )


def convert_host(
    host: str,
    binary: Path,
    source_root: Path,
    output_root: Path,
    *,
    resume: bool,
) -> dict[str, Any]:
    if host not in HOST_SPLITS:
        raise ValueError("host must be john1, john2, or john3")
    actual_host = socket.gethostname().split(".")[0].lower()
    if host in {"john2", "john3"} and actual_host != host:
        raise ValueError(f"conversion assignment {host} is running on {actual_host}")
    if not binary.is_file():
        raise ValueError(f"graded-oracle exporter is missing: {binary}")

    started = time.time()
    split_reports = {}
    for split, seeds in HOST_SPLITS[host].items():
        sources = [source_root / host / f"seed-{seed}.json" for seed in seeds]
        missing = [str(path) for path in sources if not path.is_file()]
        if missing:
            raise ValueError(f"graded-oracle source files are missing: {missing}")
        output = output_root / "partials" / host / split
        command = [str(binary), "export-graded-oracle"]
        for source in sources:
            command.extend(["--input", str(source)])
        command.extend(["--output", str(output), "--split", split])
        if resume:
            command.append("--resume")
        run(command, output_root / "logs" / host / f"convert-{split}.log")
        run(
            [str(binary), "validate-graded-oracle", "--dataset", str(output)],
            output_root / "logs" / host / f"validate-{split}.log",
        )
        manifest_path = output / "dataset.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest["seeds"] != list(seeds) or manifest["split"] != split:
            raise ValueError(f"{host} {split} manifest identity drifted")
        split_reports[split] = {
            "dataset": str(output.resolve()),
            "manifest_sha256": checksum(manifest_path),
            "seeds": list(seeds),
            "groups": manifest["total_groups"],
            "candidates": manifest["total_records"],
            "source_sha256": {
                str(path): checksum(path)
                for path in sources
            },
            "test_data_opened_by_model": False,
        }

    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "stage": "host-conversion",
        "host": host,
        "actual_hostname": socket.gethostname(),
        "binary": str(binary.resolve()),
        "binary_sha256": checksum(binary),
        "started_unix_seconds": started,
        "ended_unix_seconds": time.time(),
        "splits": split_reports,
    }
    report_path = output_root / "reports" / f"conversion-{host}.json"
    write_json_atomic(report_path, report)
    return report


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=sorted(HOST_SPLITS), required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    report = convert_host(
        args.host,
        args.binary,
        args.source_root,
        args.output_root,
        resume=args.resume,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
