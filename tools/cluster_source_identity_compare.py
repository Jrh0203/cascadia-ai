#!/usr/bin/env python3
"""Compare complete MLX source identities and persist the proof."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

IDENTITY_KIND = "complete-mlx-runtime-source-v1"


class SourceIdentityError(RuntimeError):
    """Raised when source bundles are missing, malformed, or different."""


def compare_identities(
    paths: list[Path],
    *,
    expected_bundle_sha256: str | None = None,
) -> dict[str, Any]:
    if len(paths) < 2:
        raise SourceIdentityError("at least two source identities are required")
    reports = [json.loads(path.read_text()) for path in paths]
    for path, report in zip(paths, reports, strict=True):
        if report.get("identity_kind") != IDENTITY_KIND:
            raise SourceIdentityError(f"unsupported source identity: {path}")
        bundle = report.get("bundle_sha256")
        if not isinstance(bundle, str) or len(bundle) != 64:
            raise SourceIdentityError(f"invalid source bundle digest: {path}")
    bundles = {report["bundle_sha256"] for report in reports}
    if len(bundles) != 1:
        raise SourceIdentityError("source bundle identities differ")
    bundle = next(iter(bundles))
    if expected_bundle_sha256 is not None and bundle != expected_bundle_sha256:
        raise SourceIdentityError("source bundle differs from preregistration")
    return {
        "schema_version": 1,
        "identity_kind": IDENTITY_KIND,
        "bundle_sha256": bundle,
        "expected_bundle_sha256": expected_bundle_sha256,
        "matches_expected": (
            expected_bundle_sha256 is None
            or bundle == expected_bundle_sha256
        ),
        "hosts": [str(report.get("host", "unknown")) for report in reports],
        "inputs": [str(path) for path in paths],
        "all_identities_match": True,
    }


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", action="append", type=Path, required=True)
    parser.add_argument("--expected-bundle-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare_identities(
        args.identity,
        expected_bundle_sha256=args.expected_bundle_sha256,
    )
    write_json_atomic(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
