from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from r2_d0.canonical import (
    INSTALL_OPERATIONS_BY_HOST,
    JOHN2_ONLY_CAPABILITIES,
    REJECTED_HELPER_ARCHIVE_SHA256,
    D0Error,
    canonical_json,
    load_canonical_json,
    render_document,
    safe_relative,
    validate_work_packet,
)
from r2_d0_test_support import work_spec

SOURCE_OWNERSHIP_PATH = Path(__file__).parent / "r2_d0" / "SOURCE_OWNERSHIP.json"


def test_source_ownership_names_the_live_single_writer_and_read_only_auditors() -> None:
    ownership = json.loads(SOURCE_OWNERSHIP_PATH.read_bytes())
    assert ownership == {
        "campaign_id": "r2-map-expert-iteration-v1",
        "concurrency_limit": 4,
        "designated_writer": "/root/john1_d0_owner",
        "effective_date": "2026-06-18",
        "independent_auditor": "/root/john3_d0_auditor",
        "lock_path": "/Users/johnherrick/cascadia/tools/r2_d0/.source-owner.lock",
        "policy": (
            "The root agent is control-plane only. /root/john1_d0_owner is the sole source "
            "writer and /root/john3_d0_auditor is the independent read-only auditor. No "
            "overlapping source edits are permitted."
        ),
        "prior_invalid_succession_recorded": True,
        "root_orchestrator": "/root",
        "run_id": "d0-runtime-bootstrap-20260618-v1",
        "schema_id": "cascadia.r2-map.d0-source-ownership.v3",
        "schema_version": 3,
        "status": "active-freeze-pending-final-audit",
    }


def test_canonical_json_is_exact_and_rejects_noncanonical_input() -> None:
    value = {"z": [2, 1], "a": "ascii"}
    assert canonical_json(value) == b'{"a":"ascii","z":[2,1]}'
    assert load_canonical_json(canonical_json(value), maximum=100, label="fixture") == value
    with pytest.raises(D0Error, match="canonical"):
        load_canonical_json(b'{"z":[2,1],"a":"ascii"}', maximum=100, label="fixture")
    with pytest.raises(D0Error, match="represented"):
        canonical_json({"number": float("nan")})


@pytest.mark.parametrize("path", ["", "/absolute", "../escape", "a/../b", "a//b", "a\\b"])
def test_safe_relative_rejects_escapes(path: str) -> None:
    with pytest.raises(D0Error):
        safe_relative(path, "path")


@pytest.mark.parametrize(
    ("host", "expected_bottles"),
    [("john1", 3), ("john2", 4), ("john3", 3)],
)
def test_work_packet_roles_have_exact_bottle_sets(host: str, expected_bottles: int) -> None:
    phase = "preflight"
    encoded = render_document(work_spec(host, phase), kind="work")
    decoded = json.loads(encoded)
    assert len(decoded["artifacts"]["bottles"]) == expected_bottles
    assert validate_work_packet(decoded) == decoded


@pytest.mark.parametrize("host", ["john1", "john3"])
@pytest.mark.parametrize("capability", sorted(JOHN2_ONLY_CAPABILITIES))
def test_execution_only_hosts_reject_every_john2_only_capability(
    host: str, capability: str
) -> None:
    specification = work_spec(host, "verify", operations=[capability])
    with pytest.raises(D0Error, match="John2-only"):
        render_document(specification, kind="work")


def test_john1_packet_accepts_the_full_positive_runtime_chain() -> None:
    for phase in ("preflight", "start", "verify", "rollback", "postflight"):
        render_document(work_spec("john1", phase), kind="work")
    for operation in INSTALL_OPERATIONS_BY_HOST["john1"]:
        render_document(
            work_spec("john1", "install", operations=[operation]),
            kind="work",
        )


def test_only_john2_acquisition_packet_may_defer_derived_smoke_archive() -> None:
    acquisition = work_spec(
        "john2",
        "install",
        operations=["acquire-smoke"],
    )
    acquisition["artifacts"]["smoke_oci"] = None
    assert json.loads(render_document(acquisition, kind="work"))["artifacts"]["smoke_oci"] is None
    worker = work_spec("john3", "install", operations=["install-runtime"])
    worker["artifacts"]["smoke_oci"] = None
    with pytest.raises(D0Error, match="production boundary"):
        render_document(worker, kind="work")


def test_packet_hash_binds_every_semantic_field() -> None:
    encoded = render_document(work_spec("john2", "preflight"), kind="work")
    packet = json.loads(encoded)
    changed = copy.deepcopy(packet)
    changed["expires_unix_ms"] -= 1
    with pytest.raises(D0Error, match="SHA-256"):
        validate_work_packet(changed)


def test_obsolete_source_freeze_helper_is_explicitly_rejected() -> None:
    specification = work_spec(
        "john2",
        "preflight",
        helper_sha256=REJECTED_HELPER_ARCHIVE_SHA256,
    )
    with pytest.raises(D0Error, match="rejected obsolete helper"):
        render_document(specification, kind="work")


def test_packet_rejects_retired_ssd_and_wrong_campaign_root() -> None:
    ssd = work_spec("john2", "preflight")
    ssd["paths"]["output_root"] = "/Volumes/John_1/r2-map"
    with pytest.raises(D0Error, match="retired SSD"):
        render_document(ssd, kind="work")
    wrong_root = work_spec("john2", "preflight")
    wrong_root["paths"]["campaign_root"] = "/tmp/r2-map"
    with pytest.raises(D0Error, match="storage contract"):
        render_document(wrong_root, kind="work")


def test_final_live_cycle_requires_qualification_barrier_and_current_dependencies() -> None:
    specification = work_spec(
        "john2",
        "install",
        cycle_id="final-live",
        operations=["render-probe-context"],
    )
    packet = json.loads(render_document(specification, kind="work"))
    assert packet["cycle_id"] == "final-live"
    qualification = [
        item
        for item in packet["predecessors"]
        if item["host"] == "john2" and item["cycle_id"] == "qualification"
    ]
    assert qualification[-1]["phase"] == "postflight"
    assert qualification[-1]["status"] == "pass"

    no_barrier = copy.deepcopy(specification)
    no_barrier["predecessors"] = [
        item for item in no_barrier["predecessors"] if item["cycle_id"] == "final-live"
    ]
    with pytest.raises(D0Error, match="qualification postflight barrier"):
        render_document(no_barrier, kind="work")

    cross_host = copy.deepcopy(specification)
    cross_host["predecessors"][0]["host"] = "john3"
    with pytest.raises(D0Error, match="crosses a host boundary"):
        render_document(cross_host, kind="work")


@pytest.mark.parametrize("host", ["john1", "john2", "john3"])
def test_final_live_preflight_binds_completed_qualification(host: str) -> None:
    specification = work_spec(host, "preflight", cycle_id="final-live")
    packet = json.loads(render_document(specification, kind="work"))
    qualification = [item for item in packet["predecessors"] if item["cycle_id"] == "qualification"]
    assert qualification
    assert qualification[-1]["host"] == host
    assert qualification[-1]["phase"] == "postflight"
    assert qualification[-1]["status"] == "pass"
    assert not [item for item in packet["predecessors"] if item["cycle_id"] == "final-live"]


def test_both_cycles_render_the_complete_positive_host_phase_graph() -> None:
    for cycle_id in ("qualification", "final-live"):
        for host in ("john1", "john2", "john3"):
            for operation in INSTALL_OPERATIONS_BY_HOST[host]:
                render_document(
                    work_spec(host, "install", cycle_id=cycle_id, operations=[operation]),
                    kind="work",
                )
            for phase in ("preflight", "start", "verify", "rollback", "postflight"):
                render_document(work_spec(host, phase, cycle_id=cycle_id), kind="work")


def test_qualification_cycle_rejects_final_live_predecessor() -> None:
    specification = work_spec("john2", "start")
    specification["predecessors"][-1]["cycle_id"] = "final-live"
    with pytest.raises(D0Error, match="future cycle"):
        render_document(specification, kind="work")


def test_failed_rollback_can_only_transition_to_rollback_retry() -> None:
    retry = work_spec("john2", "rollback")
    predecessor = copy.deepcopy(retry["predecessors"][-1])
    predecessor.update(
        {
            "phase": "rollback",
            "operation": "rollback-runtime",
            "status": "fail",
            "packet_sha256": "a" * 64,
            "report_sha256": "b" * 64,
            "receipt_relative": f"receipts/{'b' * 64}",
            "finished_unix_ms": predecessor["finished_unix_ms"] + 500,
        }
    )
    retry["predecessors"].append(predecessor)
    assert json.loads(render_document(retry, kind="work"))["phase"] == "rollback"

    postflight = work_spec("john2", "postflight")
    postflight["predecessors"][-1]["status"] = "fail"
    with pytest.raises(D0Error, match="rollback predecessor"):
        render_document(postflight, kind="work")
