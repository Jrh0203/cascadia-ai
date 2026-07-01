from __future__ import annotations

import json
import time
from argparse import Namespace
from pathlib import Path
from typing import Optional

import pytest
from r2_d0.canonical import D0Error, canonical_json, document_sha256
from r2_d0_qualification_retry import render

CAMPAIGN = Path("/Users/johnherrick/cascadia-bench/r2-map-v1/control/d0-v8-campaign")
MIGRATION = CAMPAIGN / "migration-v14-docker-network-lifecycle"
FAILURE = CAMPAIGN / "qualification-ready/john2/verify-runtime-v13-managed-link-retry-02"


def _arguments(output: Path, *, lineage: Optional[Path] = None) -> Namespace:  # noqa: UP045
    return Namespace(
        failed_packet=FAILURE / "work-packet.json",
        failure_completion=FAILURE / "control-completion.json",
        lineage=lineage or MIGRATION / "accepted-lineage.json",
        helper_transition=MIGRATION / "transitions/v13-to-v14.json",
        helper_transition_signature=(MIGRATION / "transitions/v13-to-v14-signature.json"),
        execution_plan=CAMPAIGN / "plan/execution-plan.json",
        public_key=CAMPAIGN / "plan/campaign-public-key",
        private_key=CAMPAIGN / "private/campaign-ed25519",
        output_root=output,
        issued_unix_ms=time.time_ns() // 1_000_000,
    )


def test_retry_renderer_emits_fresh_unclaimed_v42_bound_packet(tmp_path: Path) -> None:
    output = tmp_path / "retry"
    manifest = render(_arguments(output))
    packet = json.loads((output / "work-packet.json").read_bytes())

    assert manifest["status"] == "prepared-not-dispatched"
    assert manifest["operation_identity"]["failed_packet_replay_allowed"] is False
    assert manifest["operation_identity"]["control_claim_state"] == ("unclaimed-pre-dispatch")
    assert packet["packet_sha256"] != manifest["operation_identity"]["failed_packet_sha256"]
    assert packet["helper_sha256"] == (
        "dfd721a2e557c84d2b3bc9337cca8c9625a2b5c2df6d9cc104a2b1342fe4de03"
    )
    assert len(packet["helper_transitions"]) == 27
    assert manifest["retry_lineage"]["post_failure_read_only_evidence_sha256"] == (
        "ae490410796cca00214fbf435531a16b9dbc6f9b81dcf8a71b9e3d607f2dfc79"
    )
    assert not (output / "control-claim.json").exists()


def test_retry_renderer_rejects_reclassified_v42_lineage(tmp_path: Path) -> None:
    lineage = json.loads((MIGRATION / "accepted-lineage.json").read_bytes())
    lineage["retry_lineage"]["post_failure_read_only_proof"]["network_change_classification"] = (
        "unclassified"
    )
    lineage["lineage_sha256"] = document_sha256(lineage, "lineage_sha256")
    lineage_path = tmp_path / "lineage.json"
    lineage_path.write_bytes(canonical_json(lineage))

    with pytest.raises(D0Error, match="qualification retry lineage binding differs"):
        render(_arguments(tmp_path / "rejected", lineage=lineage_path))
