#!/usr/bin/env python3
"""Exercise live Part 1 resource, shutdown, dashboard, and retry gates."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import v3_campaign as campaign
from cascadia_v3_mlx.contracts import V3MlxConfig
from cascadia_v3_mlx.stream import RustBatchStream


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _dashboard_hosts(url: str) -> list[str]:
    with urllib.request.urlopen(url, timeout=10) as response:
        value = json.load(response)
    nodes = value.get("nodes")
    if isinstance(nodes, list):
        node_ids = [node.get("id") for node in nodes if isinstance(node, dict)]
        if node_ids and all(isinstance(node_id, str) for node_id in node_ids):
            return sorted(node_ids)
    if isinstance(value.get("hosts"), dict):
        return sorted(value["hosts"])
    status = value.get("status")
    if isinstance(status, dict) and isinstance(status.get("hosts"), dict):
        return sorted(status["hosts"])
    canonical = value.get("canonical_payload")
    if isinstance(canonical, str):
        parsed = json.loads(canonical)
        if isinstance(parsed.get("hosts"), dict):
            return sorted(parsed["hosts"])
    raise ValueError("dashboard response does not contain host status")


def exercise(
    *,
    root: Path,
    feature_manifest: Path,
    mlx_profile: Path,
    recovery_receipt: Path,
    worker_retry_receipt: Path,
    dashboard_url: str,
    batch_stream_binary: Path,
    dataset: Path,
    output: Path,
) -> dict[str, object]:
    feature = _read(feature_manifest)
    profile = _read(mlx_profile)
    recovery = _read(recovery_receipt)
    retry = _read(worker_retry_receipt)
    storage = campaign._storage(root)

    disk_refusal = False
    try:
        campaign.assert_capacity_for_write(root, campaign.MAX_BYTES)
    except campaign.CampaignError:
        disk_refusal = True

    config = V3MlxConfig(
        opportunity_feature_rows=feature["opportunity_feature_rows"],
        opportunity_training_factor_rows=feature["opportunity_training_factor_rows"],
    )
    stream = RustBatchStream(
        batch_stream_binary,
        [dataset],
        config,
        batch_size=8_000,
        epochs=1,
        allow_scientific_data=False,
    )
    rows = int(next(stream).targets.shape[0])
    exhausted = False
    try:
        next(stream)
    except StopIteration:
        exhausted = True
    stream.close()
    clean_shutdown = (
        rows == 8_000
        and exhausted
        and stream.returncode == 0
        and not stream.producer_alive
    )
    unauthorized = subprocess.run(
        [
            str(batch_stream_binary),
            "--input",
            str(dataset),
            "--batch-size",
            "1",
            "--epochs",
            "1",
            "--allow-scientific-data",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    scientific_data_guard = (
        unauthorized.returncode != 0
        and "campaign-state" in unauthorized.stderr
    )

    hosts = _dashboard_hosts(dashboard_url)
    memory_limit = int(profile["peak_memory_bytes"]) <= int(
        profile["physical_memory_bytes"]
    ) * 0.70
    receipt = {
        "schema_id": "cascadia-v3-part1-infrastructure-receipt-v1",
        "passed": all(
            (
                disk_refusal,
                memory_limit,
                clean_shutdown,
                scientific_data_guard,
                recovery.get("checkpoint_exact_continuation") is True,
                retry.get("passed") is True,
                hosts == ["john1", "john2", "john3", "john4"],
                storage["within_campaign_limit"],
                storage["free_space_preserved"],
            )
        ),
        "disk_limit": disk_refusal,
        "memory_limit": memory_limit,
        "clean_shutdown": clean_shutdown,
        "scientific_data_guard": scientific_data_guard,
        "john1_trainer_restart": recovery.get("checkpoint_exact_continuation") is True,
        "bacalhau_worker_retry": retry.get("passed") is True,
        "dashboard_hosts": hosts,
        "stream_rows_before_clean_exit": rows,
        "stream_return_code": stream.returncode,
        "storage": storage,
        "peak_memory_bytes": profile["peak_memory_bytes"],
        "physical_memory_bytes": profile["physical_memory_bytes"],
    }
    _write_atomic(output, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=campaign.DEFAULT_ROOT)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--mlx-profile", type=Path, required=True)
    parser.add_argument("--recovery-receipt", type=Path, required=True)
    parser.add_argument("--worker-retry-receipt", type=Path, required=True)
    parser.add_argument("--dashboard-url", required=True)
    parser.add_argument("--batch-stream-binary", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = exercise(
        root=args.root,
        feature_manifest=args.feature_manifest,
        mlx_profile=args.mlx_profile,
        recovery_receipt=args.recovery_receipt,
        worker_retry_receipt=args.worker_retry_receipt,
        dashboard_url=args.dashboard_url,
        batch_stream_binary=args.batch_stream_binary,
        dataset=args.dataset,
        output=args.output,
    )
    print(json.dumps(result, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
