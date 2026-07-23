#!/usr/bin/env python3
"""Merge independently frozen exact certificates into an AAAAA catalog.

Run this only after the catalog writer has exited. Each certificate is fully
revalidated, including deterministic reproduction of its relaxed-superset
enumeration, before it can replace an incomplete catalog row.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from tools import aaaaa_wildlife_catalog as catalog
from tools import aaaaa_wildlife_motif_certificate as motif


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_motif_certificate(
    certificate_path: Path,
    row: dict[str, Any],
    *,
    reproduce: bool = True,
) -> dict[str, Any]:
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    if certificate.get("schema") != "aaaaa-motif-incompatibility-certificate-v1":
        raise ValueError(f"unsupported certificate schema: {certificate_path}")
    if tuple(certificate.get("counts", ())) != motif.COUNTS:
        raise ValueError("motif certificate count vector mismatch")
    if (
        certificate.get("excluded_score") != motif.EXCLUDED_SCORE
        or certificate.get("certified_upper_bound") != motif.CERTIFIED_UPPER_BOUND
        or not certificate.get("proof_complete")
    ):
        raise ValueError("motif certificate conclusion mismatch")
    if certificate.get("source_sha256") != sha256(Path(motif.__file__).resolve()):
        raise ValueError("motif certificate source hash mismatch")
    if reproduce and certificate.get("enumeration") != motif.enumerate_relaxed_superset():
        raise ValueError("motif certificate enumeration did not reproduce")

    counts = tuple(int(value) for value in row["counts"])
    if counts != motif.COUNTS:
        raise ValueError("catalog row does not match motif certificate")
    tokens, breakdown = catalog.validate_witness(counts, certificate["incumbent"]["tokens"])
    if sum(breakdown) != motif.CERTIFIED_UPPER_BOUND:
        raise ValueError("motif certificate incumbent score mismatch")

    merged = copy.deepcopy(row)
    merged.update(
        {
            "optimum": motif.CERTIFIED_UPPER_BOUND,
            "score_breakdown": breakdown,
            "tokens": tokens,
            "proof_method": certificate["proof_method"],
            "proof_complete": True,
            "external_certificate": {
                "path": str(certificate_path),
                "sha256": sha256(certificate_path),
                "schema": certificate["schema"],
                "source_sha256": certificate["source_sha256"],
                "elapsed_seconds": certificate["elapsed_seconds"],
                "excluded_score": certificate["excluded_score"],
                "relaxation": certificate["relaxation"],
                "enumeration": certificate["enumeration"],
            },
        }
    )
    return merged


def merge(
    payload: dict[str, Any],
    certificate_paths: list[Path],
    *,
    reproduce: bool = True,
) -> dict[str, Any]:
    if payload.get("schema") != catalog.SCHEMA:
        raise ValueError("unsupported AAAAA catalog schema")
    results = {
        tuple(int(value) for value in row["counts"]): copy.deepcopy(row)
        for row in payload["results"]
    }
    records = list(payload.get("external_certificates", []))
    for certificate_path in certificate_paths:
        certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
        counts = tuple(int(value) for value in certificate.get("counts", ()))
        if counts not in results:
            raise ValueError(f"catalog has no row for certificate counts {counts}")
        results[counts] = validate_motif_certificate(
            certificate_path, results[counts], reproduce=reproduce
        )
        record = results[counts]["external_certificate"]
        records = [existing for existing in records if existing.get("path") != record["path"]]
        records.append(record)

    ordered = [
        results[counts]
        for counts, _ in catalog.count_vectors()
        if counts in results
    ]
    merged_payload = copy.deepcopy(payload)
    merged_payload["results"] = ordered
    merged_payload["completed_count"] = sum(row.get("proof_complete", False) for row in ordered)
    merged_payload["proof_complete"] = (
        len(ordered) == int(payload["allocation_count"])
        and merged_payload["completed_count"] == int(payload["allocation_count"])
    )
    merged_payload["external_certificates"] = records
    merged_payload["certificate_merge_source_sha256"] = sha256(Path(__file__).resolve())
    return merged_payload


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--certificate", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    args = parser.parse_args()
    payload = json.loads(args.catalog.read_text(encoding="utf-8"))
    merged = merge(payload, args.certificate)
    atomic_write(args.output, json.dumps(merged, indent=2) + "\n")
    atomic_write(args.markdown, catalog.render_markdown(merged) + "\n")
    print(f"merged exact certificates: {merged['completed_count']}/{merged['allocation_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
