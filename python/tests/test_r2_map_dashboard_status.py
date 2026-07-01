from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import blake3
import pytest
from cascadia_mlx.r2_map_apfs_lifecycle import build_host_safety_receipt
from cascadia_mlx.r2_map_contracts import (
    ALLOWED_HOSTS,
    Phase,
    new_campaign_state,
    transition_state,
)
from cascadia_mlx.r2_map_dashboard_status import (
    MAX_COMPACT_JSON_BYTES,
    MAX_HOST_DETAIL_BYTES,
    DashboardStatusError,
    DashboardStatusInputs,
    build_dashboard_status,
    read_compact_json,
    validate_dashboard_status,
    write_dashboard_status,
    write_serving_projection,
)

REPOSITORY = Path(__file__).resolve().parents[2]
FIXTURES = REPOSITORY / "tests/fixtures/r2_map"
CONTRACTS_READY_FIXTURE = FIXTURES / "dashboard-status-v1-contracts-ready.json"
TRAINING_FIXTURE = FIXTURES / "dashboard-status-v1-training.json"
HASH_A = "a" * 64
HASH_B = "b" * 64


def _fixture(path: Path) -> dict:
    return json.loads(path.read_text())


def _training_state() -> dict:
    state = new_campaign_state(now="2026-06-18T00:00:00.000Z")
    state = transition_state(
        state,
        Phase.BOOTSTRAP_GENERATING,
        reason="contracts frozen",
        now="2026-06-18T00:00:01.000Z",
    )
    state = transition_state(
        state,
        Phase.BOOTSTRAP_VALIDATED,
        reason="bootstrap shards validated",
        generation_manifest_sha256=HASH_A,
        completed_shard_hosts=ALLOWED_HOSTS,
        now="2026-06-18T00:00:02.000Z",
    )
    state = transition_state(
        state,
        Phase.BOOTSTRAP_TRAINING,
        reason="train bootstrap",
        now="2026-06-18T00:00:03.000Z",
    )
    state = transition_state(
        state,
        Phase.BOOTSTRAP_CANDIDATE_GATE,
        reason="candidate verified",
        candidate_checkpoint_sha256=HASH_B,
        now="2026-06-18T00:00:04.000Z",
    )
    state = transition_state(
        state,
        Phase.INCUMBENT_PROMOTED,
        reason="candidate promoted",
        now="2026-06-18T00:00:05.000Z",
    )
    state = transition_state(
        state,
        Phase.ROUND_ALLOCATED,
        reason="allocate round zero",
        now="2026-06-18T00:00:06.000Z",
    )
    state = transition_state(
        state,
        Phase.GENERATING,
        reason="start generation",
        now="2026-06-18T00:00:07.000Z",
    )
    state = transition_state(
        state,
        Phase.LOCAL_SHARDS_COMPLETE,
        reason="all local shards complete",
        completed_shard_hosts=ALLOWED_HOSTS,
        now="2026-06-18T00:00:08.000Z",
    )
    state = transition_state(
        state,
        Phase.COLLECTED_AND_VALIDATED,
        reason="merged corpus validated",
        generation_manifest_sha256=HASH_A,
        now="2026-06-18T00:00:09.000Z",
    )
    return transition_state(
        state,
        Phase.TRAINING_AND_BENCHMARKING,
        reason="train and benchmark",
        now="2026-06-18T00:00:10.000Z",
    )


def test_contracts_ready_builder_matches_shared_rust_golden_fixture() -> None:
    expected = _fixture(CONTRACTS_READY_FIXTURE)
    actual = build_dashboard_status(
        DashboardStatusInputs(campaign_state=new_campaign_state()),
        updated_unix_ms=expected["updated_unix_ms"],
    )
    assert actual == expected


def test_populated_builder_matches_shared_rust_golden_fixture() -> None:
    expected = _fixture(TRAINING_FIXTURE)
    models = expected["models"]
    actual = build_dashboard_status(
        DashboardStatusInputs(
            campaign_state=_training_state(),
            host_receipts=expected["hosts"],
            training_progress=expected["training"],
            benchmark_aggregate=expected["benchmark"],
            model_manifest={
                "incumbent": models["incumbent"],
                "candidate": models["candidate"],
            },
            pool_manifest={"opponent_pool": models["opponent_pool"]},
        ),
        updated_unix_ms=expected["updated_unix_ms"],
    )
    assert actual == expected


@pytest.mark.parametrize("path", [CONTRACTS_READY_FIXTURE, TRAINING_FIXTURE])
def test_shared_golden_fixtures_are_strictly_valid(path: Path) -> None:
    assert validate_dashboard_status(_fixture(path)) == _fixture(path)


def test_atomic_writer_replaces_one_compact_file_and_cleans_temporary(tmp_path: Path) -> None:
    path = tmp_path / "control/dashboard-status.json"
    initial = _fixture(CONTRACTS_READY_FIXTURE)
    written = write_dashboard_status(path, initial)
    assert written == path.stat().st_size
    assert json.loads(path.read_text()) == initial

    replacement = dict(initial)
    replacement["updated_unix_ms"] += 1
    write_dashboard_status(path, replacement)
    assert json.loads(path.read_text()) == replacement
    assert not list(path.parent.glob(".dashboard-status.json.*.tmp"))


def test_serving_projection_is_bounded_read_only_hash_bound_and_disposable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    modes: list[int] = []
    original_fchmod = os.fchmod

    def record_fchmod(descriptor: int, mode: int) -> None:
        modes.append(mode)
        original_fchmod(descriptor, mode)

    monkeypatch.setattr(os, "fchmod", record_fchmod)
    canonical = tmp_path / "ssd/dashboard-status.json"
    serving = tmp_path / "internal/r2-map-dashboard-serving-projection-v1.json"
    status = _fixture(CONTRACTS_READY_FIXTURE)
    write_dashboard_status(canonical, status)
    metadata = write_serving_projection(serving, canonical_path=canonical, status=status)
    projection = json.loads(serving.read_text())
    canonical_payload = canonical.read_bytes()
    assert serving.stat().st_size <= 64 * 1024
    assert modes[-1] == 0o444
    assert stat.S_IMODE(serving.stat().st_mode) in {0o444, 0o700}
    assert projection["canonical_payload"].encode() == canonical_payload
    assert projection["canonical_blake3"] == blake3.blake3(canonical_payload).hexdigest()
    assert projection["canonical_updated_unix_ms"] == status["updated_unix_ms"]
    assert metadata["canonical_blake3"] == projection["canonical_blake3"]

    original = serving.read_bytes()
    serving.unlink()
    write_serving_projection(serving, canonical_path=canonical, status=status)
    assert serving.read_bytes() == original

    oversized = _fixture(CONTRACTS_READY_FIXTURE)
    oversized["hosts"]["john1"]["detail"] = "x" * (MAX_HOST_DETAIL_BYTES + 1)
    with pytest.raises(DashboardStatusError, match=r"detail exceeds.*512-byte"):
        write_serving_projection(serving, canonical_path=canonical, status=oversized)


def test_host_detail_limit_counts_utf8_bytes_not_codepoints() -> None:
    accepted = _fixture(CONTRACTS_READY_FIXTURE)
    accepted["hosts"]["john1"]["detail"] = "é" * (MAX_HOST_DETAIL_BYTES // 2)
    validate_dashboard_status(accepted)
    rejected = _fixture(CONTRACTS_READY_FIXTURE)
    rejected["hosts"]["john1"]["detail"] = "é" * (MAX_HOST_DETAIL_BYTES // 2 + 1)
    with pytest.raises(DashboardStatusError, match="512-byte UTF-8"):
        validate_dashboard_status(rejected)


def test_publisher_rejects_unknown_fields_bad_receipts_and_john4(tmp_path: Path) -> None:
    status = _fixture(CONTRACTS_READY_FIXTURE)
    status["unversioned_extension"] = True
    with pytest.raises(DashboardStatusError, match="extra"):
        validate_dashboard_status(status)

    receipt = dict(_fixture(CONTRACTS_READY_FIXTURE)["hosts"]["john1"])
    receipt["intent"] = "generate"
    with pytest.raises(DashboardStatusError, match="intent disagrees"):
        build_dashboard_status(
            DashboardStatusInputs(
                campaign_state=new_campaign_state(), host_receipts={"john1": receipt}
            ),
            updated_unix_ms=1,
        )

    with pytest.raises(DashboardStatusError, match="unexpected hosts"):
        build_dashboard_status(
            DashboardStatusInputs(
                campaign_state=new_campaign_state(), host_receipts={"john4": receipt}
            ),
            updated_unix_ms=1,
        )

    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b" " * (MAX_COMPACT_JSON_BYTES + 1))
    with pytest.raises(DashboardStatusError, match="maximum"):
        read_compact_json(oversized, label="oversized input")


def test_host_safety_overlays_blocker_without_changing_campaign_phase_or_intent() -> None:
    state = new_campaign_state()
    safety = build_host_safety_receipt(
        status="blocked-host-recovery",
        observed_unix_ms=1,
        syspolicyd_rss_bytes=4_461_944_832,
        system_swap_baseline_bytes=2_000_000_000,
        system_swap_observed_bytes=2_000_000_000,
        quiet_window_passed=False,
        detail="host stop remains active",
    )
    status = build_dashboard_status(
        DashboardStatusInputs(campaign_state=state, host_safety=safety),
        updated_unix_ms=2,
    )
    assert status["phase"] == Phase.CONTRACTS_READY.value
    assert status["hosts"]["john1"]["intent"] == "control"
    assert status["hosts"]["john1"]["detail"].startswith("blocked-host-recovery")
    assert status["hosts"]["john1"]["rss_bytes"] == 4_461_944_832
    assert status["hosts"]["john1"]["swap_delta_bytes"] == 0


def test_model_and_pool_manifests_must_match_state_and_remain_unique() -> None:
    state = _training_state()
    with pytest.raises(DashboardStatusError, match="incumbent model id disagrees"):
        build_dashboard_status(
            DashboardStatusInputs(
                campaign_state=state,
                model_manifest={
                    "incumbent": {"id": "wrong", "blake3": HASH_A},
                    "candidate": {"id": "T[0]", "blake3": HASH_B},
                },
            ),
            updated_unix_ms=1,
        )

    with pytest.raises(DashboardStatusError, match="duplicated"):
        build_dashboard_status(
            DashboardStatusInputs(
                campaign_state=state,
                model_manifest={
                    "incumbent": {"id": "C[0]", "blake3": HASH_A},
                    "candidate": {"id": "T[0]", "blake3": HASH_B},
                },
                pool_manifest={"opponent_pool": [{"id": "C[0]", "blake3": "c" * 64}]},
            ),
            updated_unix_ms=1,
        )
