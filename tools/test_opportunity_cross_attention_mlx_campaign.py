from __future__ import annotations

import json

import pytest
from opportunity_cross_attention_mlx_campaign import (
    HOSTS,
    CampaignError,
    canonical_blake3,
    validate_smoke_proof,
)
from opportunity_cross_attention_mlx_smoke_compare import PASS


def _proof() -> dict:
    identity = {
        "experiment_id": "opportunity-cross-attention-mlx-tournament-v1",
        "protocol_id": "exact-r2-opportunity-query-factorial-v1",
        "adr": "0166",
        "arm": "c0-parent-conditioned",
        "steps": 3,
        "hosts": list(HOSTS),
        "r3_cache_id": "1" * 64,
        "relational_cache_id": "2" * 64,
        "s1_cache_id": "3" * 64,
        "r6_binary_blake3": "4" * 64,
        "source_blake3": "5" * 64,
        "warm_start_id": "6" * 64,
        "checks": {host: True for host in HOSTS},
    }
    return {
        "schema_version": 1,
        "experiment_id": identity["experiment_id"],
        "protocol_id": identity["protocol_id"],
        "adr": identity["adr"],
        "classification": PASS,
        "proof_id": canonical_blake3(identity),
        "scientific_identity": identity,
    }


def test_smoke_proof_requires_every_bound_identity(tmp_path) -> None:
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps(_proof()))

    observed = validate_smoke_proof(
        path,
        source_blake3="5" * 64,
        warm_start_id="6" * 64,
        r3_cache_id="1" * 64,
        relational_cache_id="2" * 64,
        s1_cache_id="3" * 64,
        r6_binary_blake3="4" * 64,
    )

    assert observed["classification"] == PASS


def test_smoke_proof_rejects_source_drift(tmp_path) -> None:
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps(_proof()))

    with pytest.raises(CampaignError, match="smoke proof"):
        validate_smoke_proof(
            path,
            source_blake3="f" * 64,
            warm_start_id="6" * 64,
            r3_cache_id="1" * 64,
            relational_cache_id="2" * 64,
            s1_cache_id="3" * 64,
            r6_binary_blake3="4" * 64,
        )
