"""Smoke-check hidden redetermination invariants exposed by the Rust game crate."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


def _parse_seed_range(raw: str) -> list[int]:
    if ":" in raw:
        start, end = raw.split(":", 1)
        return list(range(int(start), int(end)))
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def validate_with_rust(binary: Path, seeds: list[int]) -> dict[str, Any]:
    if not binary.exists():
        return {
            "status": "skipped",
            "reason": f"Rust exporter binary not built at {binary}",
            "seeds_requested": len(seeds),
        }
    with TemporaryDirectory() as tmp:
        out = Path(tmp) / "hidden_redetermination.json"
        command = [
            str(binary),
            "--validate-hidden-redetermination",
            "--first-seed",
            str(seeds[0]),
            "--seed-count",
            str(len(seeds)),
            "--out",
            str(out),
        ]
        completed = subprocess.run(command, check=False, text=True, capture_output=True)
        if completed.returncode != 0:
            return {
                "status": "fail",
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        return json.loads(out.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", required=True, help="Half-open start:end range or comma-separated seeds")
    parser.add_argument(
        "--binary",
        default="cascadiav3/real-root-exporter/target/release/cascadiav3-real-root-exporter",
    )
    args = parser.parse_args()

    seeds = _parse_seed_range(args.seeds)
    if not seeds:
        raise ValueError("--seeds produced no seed values")
    report = validate_with_rust(Path(args.binary), seeds)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] in {"pass", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
