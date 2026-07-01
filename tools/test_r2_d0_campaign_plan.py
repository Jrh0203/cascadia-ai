from __future__ import annotations

import time

from r2_d0.canonical import render_document
from r2_d0_campaign_plan import (
    ACTIVE_HOSTS,
    bootstrap_spec,
    ordered_transactions,
    preflight_spec,
)


def test_execution_plan_is_exact_topological_and_uses_direct_john1_edges() -> None:
    transactions = ordered_transactions()
    assert len(transactions) == 46
    positions = {item["key"]: item["sequence"] for item in transactions}
    assert len(positions) == 46
    assert all(
        positions[dependency] < item["sequence"]
        for item in transactions
        for dependency in item["dependencies"]
    )
    john3_supply = next(
        item
        for item in transactions
        if item["key"] == "qualification/john3/install/materialize-runtime-supply"
    )
    assert "qualification/john1/install/materialize-runtime-supply" in john3_supply[
        "dependencies"
    ]
    assert not any("worker-channel" in item["key"] for item in transactions)


def test_currently_ready_bootstrap_and_preflight_specs_render_for_all_hosts() -> None:
    now = time.time_ns() // 1_000_000
    public_key = (
        b"ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIDAYfd7JtG91R6n8eMlKzUjjnknXCLBGdzD43Pmp57vk "
        b"fixture\n"
    )
    policy = {
        "goal_sha256": "1" * 64,
        "plan_sha256": "2" * 64,
        "runbook_sha256": "3" * 64,
    }
    for host in ACTIVE_HOSTS:
        render_document(
            bootstrap_spec(
                host,
                helper_sha256="4" * 64,
                helper_size=1024,
                public_key=public_key,
                issued_unix_ms=now,
            ),
            kind="bootstrap",
        )
        render_document(
            preflight_spec(
                host,
                helper_sha256="4" * 64,
                public_key=public_key,
                policy=policy,
                issued_unix_ms=now,
            ),
            kind="work",
        )
