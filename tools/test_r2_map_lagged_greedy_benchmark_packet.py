from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path("/Users/johnherrick/cascadia-bench/r2-map-v1/control/work-packets")
SCRIPT = ROOT / "run-r2-map-lagged-greedy-baseline-v1.zsh"
PACKET = ROOT / "r2-map-lagged-greedy-baseline-v1.json"


def test_packet_binds_authoritative_image_and_disjoint_seed_leases() -> None:
    value = json.loads(PACKET.read_text())
    john2 = value["hosts"]["john2"]
    john3 = value["hosts"]["john3"]
    assert value["image_id"] == (
        "sha256:584c6edccfc203c0cc78a636ac2d364132e36ed3dfb6e782438f63d2c089523b"
    )
    assert john2["last_seed"] < john3["first_seed"]
    assert john2["games"] + john3["games"] == value["total_games"] == 5_000
    assert value["total_seat_games"] == 20_000
    assert hashlib.sha256(SCRIPT.read_bytes()).hexdigest() == value["script_sha256"]


def test_script_enforces_container_boundaries_and_truthful_metric_coverage() -> None:
    script = SCRIPT.read_text()
    for required in (
        "--network none",
        "--read-only",
        "--user 10001:10001",
        "--cap-drop ALL",
        "--security-opt no-new-privileges",
        "--memory 4g",
        "--memory-swap 4g",
        "--pids-limit 256",
        'type=volume,src=$VOLUME_NAME,dst=/output',
        'docker cp "$CONTAINER_NAME:/output/."',
        '"metric_coverage"',
        '"not-emitted-by-authoritative-v2-basic-benchmark"',
    ):
        assert required in script
    assert "john4" not in script
    assert "rm -rf" not in script
    assert "type=bind" not in script
