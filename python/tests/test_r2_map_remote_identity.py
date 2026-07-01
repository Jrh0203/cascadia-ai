from __future__ import annotations

import copy
import hashlib

import pytest
from cascadia_mlx import r2_map_remote_worker as worker
from cascadia_mlx.r2_map_remote_identity import (
    BOOTSTRAP_GENERATION_MANIFEST_SCHEMA,
    BOOTSTRAP_PHASE_BARRIER_SCHEMA,
    CAMPAIGN_ID,
    MAXIMUM_WIDTH_CANDIDATES,
    REFERENCE_MANIFEST_SCHEMA,
    SOURCE_MANIFEST_SCHEMA,
    R2MapRemoteIdentityError,
    VerifiedRemoteJson,
    _source_path_prefix_collision,
    john1_attestation_publication_receipt_relative,
    require_open_transaction_object,
    validate_bootstrap_aggregate_generation_binding,
    validate_bootstrap_generation_manifest_value,
    validate_bootstrap_phase_barrier_value,
    validate_john1_attestation_publication_receipt,
    validate_source_identity,
)
from cascadia_mlx.r2_map_remote_storage import (
    REMOTE_IDENTITY_SHA256,
    REMOTE_ROOT,
    canonical_json,
    content_sha256,
    document_sha256,
)
from cascadia_mlx.r2_map_remote_training import RemoteObjectEvidence


def _document(
    relative: str,
    value: dict,
    *,
    mode: int = 0o400,
    trailing_newline: bool = False,
) -> VerifiedRemoteJson:
    payload = canonical_json(value) + (b"\n" if trailing_newline else b"")
    token = {
        "schema_version": 1,
        "schema_id": worker.OBJECT_TOKEN_SCHEMA,
        "relative": relative,
        "sha256": content_sha256(payload),
        "size": len(payload),
        "device": 1,
        "inode": 2,
        "mtime_ns": 3,
        "ctime_ns": 4,
        "mode": mode,
    }
    token["token_sha256"] = document_sha256(token, "token_sha256")
    return VerifiedRemoteJson(
        value=value,
        payload_sha256=token["sha256"],
        payload_blake3="a" * 64,
        evidence=RemoteObjectEvidence(
            relative=relative,
            object_token=token,
            open_receipt={
                "storage_receipt_relative": "control/receipts/req-open.json",
                "storage_receipt_sha256": "b" * 64,
            },
            range_receipts=(),
        ),
    )


def _source_documents() -> tuple[VerifiedRemoteJson, ...]:
    target = "source/w0-test"
    reference_relative = f"{target}/docs/v2/reports/r2-map-w0-reference-panel-manifest-v1.1.json"
    panel = {
        "panel_id": "maximum-width-service",
        "definition": {
            "reference_candidate_count": MAXIMUM_WIDTH_CANDIDATES,
            "expected_action_evaluations": MAXIMUM_WIDTH_CANDIDATES,
            "complete_cardinality_required": True,
            "truncation_allowed": False,
        },
        "source_bindings": [],
    }
    panel["panel_sha256"] = content_sha256(canonical_json(panel))
    reference = {
        "schema_id": REFERENCE_MANIFEST_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "contract_revision": "sequential-public-market-v1.1",
        "implementation_identity": {"maximum_width_panel_sha256": panel["panel_sha256"]},
        "panels": [panel],
    }
    reference["manifest_sha256"] = document_sha256(reference, "manifest_sha256")
    reference_document = _document(reference_relative, reference)
    file_payloads = {
        "docs/v2/reports/r2-map-w0-reference-panel-manifest-v1.1.json": (
            canonical_json(reference),
            "0400",
        ),
        "tools/r2_map_source_archive.py": (b"archive verifier\n", "0500"),
        "tools/r2_map_rust_w4_target_gate.mk": (b"target gate\n", "0400"),
        "tools/r2_map_rust_p1_gate.mk": (b"p1 gate\n", "0400"),
        "tools/r2_map_rust_release_gate.mk": (b"release gate\n", "0400"),
        "tools/r2_map_python_boundary_gate.mk": (b"python gate\n", "0400"),
        "tools/r2_map_rust_compile_gate.mk": (b"compile gate\n", "0400"),
        "tools/r2_map_python_fixture_gate.mk": (b"fixture gate\n", "0400"),
    }
    files = [
        {
            "relative": relative,
            "sha256": content_sha256(payload),
            "size": len(payload),
            "mode": mode,
        }
        for relative, (payload, mode) in sorted(file_payloads.items())
    ]
    source = {
        "schema_id": SOURCE_MANIFEST_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "protected_seed_values_opened": False,
        "w0_reference_manifest_sha256": reference_document.payload_sha256,
        "file_count": len(files),
        "total_bytes": sum(item["size"] for item in files),
        "files": files,
    }
    source["document_sha256"] = document_sha256(source, "document_sha256")
    source_document = _document(f"{target}/source-manifest.json", source)
    raw_member_bytes = sum(
        512 + ((item["size"] + 511) // 512) * 512 for item in files
    )
    archive_bytes = ((raw_member_bytes + 1_024 + 10_239) // 10_240) * 10_240
    archive_sha256 = "d" * 64
    archive_verification = {
        "schema_id": "cascadia.r2-map.source-archive-verification.v1",
        "status": "valid",
        "document_sha256": source["document_sha256"],
        "archive_sha256": archive_sha256,
        "archive_bytes": archive_bytes,
        "member_count": len(files),
        "member_names_sha256": hashlib.sha256(
            ("\n".join(item["relative"] for item in files) + "\n").encode("ascii")
        ).hexdigest(),
        "content_bytes": source["total_bytes"],
        "terminal_zero_bytes": archive_bytes - raw_member_bytes,
        "regular_only": True,
        "pax_or_extended_headers": False,
        "metadata_normalized": True,
    }
    archive_verification_document = _document(
        f"{target}/source-archive-verification.json",
        archive_verification,
        trailing_newline=True,
    )
    file_entries = {item["relative"]: item for item in files}
    alias_sources = {
        "target.mk": "tools/r2_map_rust_w4_target_gate.mk",
        "p1.mk": "tools/r2_map_rust_p1_gate.mk",
        "release.mk": "tools/r2_map_rust_release_gate.mk",
        "python.mk": "tools/r2_map_python_boundary_gate.mk",
        "compile.mk": "tools/r2_map_rust_compile_gate.mk",
        "fixture.mk": "tools/r2_map_python_fixture_gate.mk",
        "archive-verify.py": "tools/r2_map_source_archive.py",
    }
    objects = [
        {
            "relative": item["relative"],
            "sha256": item["sha256"],
            "size": item["size"],
            "mode": item["mode"],
        }
        for item in files
    ]
    objects.extend(
        {
            "relative": alias,
            "sha256": file_entries[source_relative]["sha256"],
            "size": file_entries[source_relative]["size"],
            "mode": "0500" if alias == "archive-verify.py" else "0400",
        }
        for alias, source_relative in alias_sources.items()
    )
    objects.extend(
        (
            {
                "relative": "source-manifest.json",
                "sha256": source_document.payload_sha256,
                "size": source_document.evidence.object_token["size"],
                "mode": "0400",
            },
            {
                "relative": "source.tar",
                "sha256": archive_sha256,
                "size": archive_bytes,
                "mode": "0400",
            },
            {
                "relative": "source-archive-verification.json",
                "sha256": archive_verification_document.payload_sha256,
                "size": archive_verification_document.evidence.object_token["size"],
                "mode": "0400",
            },
        )
    )
    transaction = {
        "schema_version": 1,
        "schema_id": worker.TRANSACTION_SCHEMA,
        "transaction_id": "w0-test",
        "target_relative": target,
        "objects": sorted(objects, key=lambda item: item["relative"]),
    }
    transaction["manifest_sha256"] = document_sha256(transaction, "manifest_sha256")
    transaction_document = _document(f"{target}/.r2-map-transaction.json", transaction)
    receipt = {
        "schema_version": 1,
        "schema_id": worker.RECEIPT_SCHEMA,
        "request_id": "req-commit",
        "command_sha256": "c" * 64,
        "status": "ok",
        "operation": "transaction-commit",
        "host": "john2",
        "host_identity_sha256": REMOTE_IDENTITY_SHA256,
        "root": str(REMOTE_ROOT),
        "completed_unix_ms": 1,
        "result": {
            "transaction_id": transaction["transaction_id"],
            "target_relative": target,
            "manifest_sha256": transaction["manifest_sha256"],
            "object_count": len(transaction["objects"]),
            "committed": True,
            "payload_size": 0,
            "payload_sha256": hashlib.sha256(b"").hexdigest(),
        },
    }
    receipt["receipt_sha256"] = document_sha256(receipt, "receipt_sha256")
    receipt_document = _document("control/receipts/req-commit.json", receipt)
    return (
        source_document,
        reference_document,
        archive_verification_document,
        transaction_document,
        receipt_document,
    )


def _rebind_transaction(
    transaction_document: VerifiedRemoteJson,
    receipt_document: VerifiedRemoteJson,
    transaction: dict,
) -> tuple[VerifiedRemoteJson, VerifiedRemoteJson]:
    transaction = copy.deepcopy(transaction)
    transaction["objects"] = sorted(
        transaction["objects"], key=lambda item: item["relative"]
    )
    transaction["manifest_sha256"] = document_sha256(
        transaction, "manifest_sha256"
    )
    rebound_transaction = _document(
        transaction_document.evidence.relative,
        transaction,
    )
    receipt = copy.deepcopy(receipt_document.value)
    receipt["result"]["manifest_sha256"] = transaction["manifest_sha256"]
    receipt["result"]["object_count"] = len(transaction["objects"])
    receipt["receipt_sha256"] = document_sha256(receipt, "receipt_sha256")
    rebound_receipt = _document(receipt_document.evidence.relative, receipt)
    return rebound_transaction, rebound_receipt


def test_source_identity_is_derived_only_from_receipt_bound_objects() -> None:
    source, reference, archive_verification, transaction, receipt = _source_documents()
    identity = validate_source_identity(
        source_manifest=source,
        reference_manifest=reference,
        source_archive_verification=archive_verification,
        transaction_manifest=transaction,
        transaction_commit_receipt=receipt,
    )
    assert len(identity["source_blake3"]) == 64
    assert identity["maximum_width_candidates"] == MAXIMUM_WIDTH_CANDIDATES
    assert identity["source_archive"]["relative"] == "source/w0-test/source.tar"
    assert identity["source_archive_verifier"]["mode"] == "0500"
    assert set(identity["source_gate_aliases"]) == {
        "target.mk",
        "p1.mk",
        "release.mk",
        "python.mk",
        "compile.mk",
        "fixture.mk",
    }
    assert (
        identity["maximum_width_panel_sha256"]
        == reference.value["implementation_identity"]["maximum_width_panel_sha256"]
    )


def test_source_path_prefix_collisions_fail_closed() -> None:
    assert _source_path_prefix_collision({"a", "a/b"}) is True
    assert _source_path_prefix_collision({"a/b", "a/c", "b"}) is False


def test_source_identity_rejects_caller_like_panel_drift() -> None:
    source, reference, archive_verification, transaction, receipt = _source_documents()
    drifted = copy.deepcopy(reference)
    drifted.value["panels"][0]["definition"]["reference_candidate_count"] -= 1
    with pytest.raises(R2MapRemoteIdentityError, match="reference manifest identity"):
        validate_source_identity(
            source_manifest=source,
            reference_manifest=drifted,
            source_archive_verification=archive_verification,
            transaction_manifest=transaction,
            transaction_commit_receipt=receipt,
        )


@pytest.mark.parametrize("missing", ("source.tar", "archive-verify.py", "target.mk"))
def test_source_identity_rejects_missing_archive_closure_object(missing: str) -> None:
    source, reference, archive_verification, transaction, receipt = _source_documents()
    value = copy.deepcopy(transaction.value)
    value["objects"] = [
        item for item in value["objects"] if item["relative"] != missing
    ]
    transaction, receipt = _rebind_transaction(transaction, receipt, value)
    with pytest.raises(R2MapRemoteIdentityError, match="exact archive closure"):
        validate_source_identity(
            source_manifest=source,
            reference_manifest=reference,
            source_archive_verification=archive_verification,
            transaction_manifest=transaction,
            transaction_commit_receipt=receipt,
        )


def test_source_identity_rejects_extra_archive_closure_object() -> None:
    source, reference, archive_verification, transaction, receipt = _source_documents()
    value = copy.deepcopy(transaction.value)
    value["objects"].append(
        {"relative": "unregistered", "sha256": "a" * 64, "size": 1, "mode": "0400"}
    )
    transaction, receipt = _rebind_transaction(transaction, receipt, value)
    with pytest.raises(R2MapRemoteIdentityError, match="exact archive closure"):
        validate_source_identity(
            source_manifest=source,
            reference_manifest=reference,
            source_archive_verification=archive_verification,
            transaction_manifest=transaction,
            transaction_commit_receipt=receipt,
        )


@pytest.mark.parametrize("alias", ("archive-verify.py", "target.mk", "python.mk"))
def test_source_identity_rejects_alias_byte_drift(alias: str) -> None:
    source, reference, archive_verification, transaction, receipt = _source_documents()
    value = copy.deepcopy(transaction.value)
    descriptor = next(item for item in value["objects"] if item["relative"] == alias)
    descriptor["sha256"] = "f" * 64
    transaction, receipt = _rebind_transaction(transaction, receipt, value)
    expected = "archive verifier" if alias == "archive-verify.py" else "gate alias"
    with pytest.raises(R2MapRemoteIdentityError, match=expected):
        validate_source_identity(
            source_manifest=source,
            reference_manifest=reference,
            source_archive_verification=archive_verification,
            transaction_manifest=transaction,
            transaction_commit_receipt=receipt,
        )


def test_source_identity_rejects_receipt_bound_archive_report_drift() -> None:
    source, reference, archive_verification, transaction, receipt = _source_documents()
    drifted_value = copy.deepcopy(archive_verification.value)
    drifted_value["archive_sha256"] = "e" * 64
    drifted_report = _document(
        archive_verification.evidence.relative,
        drifted_value,
        trailing_newline=True,
    )
    transaction_value = copy.deepcopy(transaction.value)
    descriptor = next(
        item
        for item in transaction_value["objects"]
        if item["relative"] == "source-archive-verification.json"
    )
    descriptor["sha256"] = drifted_report.payload_sha256
    descriptor["size"] = drifted_report.evidence.object_token["size"]
    transaction, receipt = _rebind_transaction(
        transaction, receipt, transaction_value
    )
    with pytest.raises(R2MapRemoteIdentityError, match="archive verification differs"):
        validate_source_identity(
            source_manifest=source,
            reference_manifest=reference,
            source_archive_verification=drifted_report,
            transaction_manifest=transaction,
            transaction_commit_receipt=receipt,
        )


def test_live_transaction_object_must_match_the_committed_descriptor() -> None:
    _, _, _, transaction_document, _ = _source_documents()
    transaction = transaction_document.value
    relative = (
        "source/w0-test/docs/v2/reports/"
        "r2-map-w0-reference-panel-manifest-v1.1.json"
    )
    descriptor = next(
        item
        for item in transaction["objects"]
        if item["relative"]
        == "docs/v2/reports/r2-map-w0-reference-panel-manifest-v1.1.json"
    )
    token = {
        "schema_version": 1,
        "schema_id": worker.OBJECT_TOKEN_SCHEMA,
        "relative": relative,
        "sha256": descriptor["sha256"],
        "size": descriptor["size"],
        "device": 1,
        "inode": 2,
        "mtime_ns": 3,
        "ctime_ns": 4,
        "mode": 0o400,
    }
    token["token_sha256"] = document_sha256(token, "token_sha256")

    class Client:
        def open_object_with_receipt(self, requested: str) -> dict:
            assert requested == relative
            return {
                "object_token": copy.deepcopy(token),
                "storage_receipt_relative": "control/receipts/req-open-live.json",
                "storage_receipt_sha256": "d" * 64,
            }

    assert require_open_transaction_object(Client(), transaction, relative)[
        "object_token"
    ] == token

    drifted = copy.deepcopy(token)
    drifted["sha256"] = "e" * 64
    drifted["token_sha256"] = document_sha256(drifted, "token_sha256")

    class DriftedClient:
        def open_object_with_receipt(self, requested: str) -> dict:
            assert requested == relative
            return {
                "object_token": copy.deepcopy(drifted),
                "storage_receipt_relative": "control/receipts/req-open-live.json",
                "storage_receipt_sha256": "d" * 64,
            }

    with pytest.raises(R2MapRemoteIdentityError, match="live transaction object"):
        require_open_transaction_object(DriftedClient(), transaction, relative)


def _john1_attestation_publication_documents() -> tuple[
    VerifiedRemoteJson, VerifiedRemoteJson
]:
    attestation = {
        "schema_version": 1,
        "schema_id": "cascadia.r2-map.john1-local-write-attestation.v1",
        "run_id": "sweep-test",
    }
    attestation["attestation_sha256"] = document_sha256(
        attestation, "attestation_sha256"
    )
    attestation_document = _document(
        "reports/w2-w3/sweep-test/local-write-attestation.json",
        attestation,
    )
    receipt_relative = john1_attestation_publication_receipt_relative(
        attestation["attestation_sha256"]
    )
    receipt = {
        "schema_version": 1,
        "schema_id": worker.RECEIPT_SCHEMA,
        "request_id": receipt_relative.rsplit("/", 1)[-1].removesuffix(".json"),
        "command_sha256": "c" * 64,
        "status": "ok",
        "operation": "put-file",
        "host": "john2",
        "host_identity_sha256": REMOTE_IDENTITY_SHA256,
        "root": str(REMOTE_ROOT),
        "completed_unix_ms": 1,
        "result": {
            "relative": attestation_document.evidence.relative,
            "sha256": attestation_document.payload_sha256,
            "size": attestation_document.evidence.object_token["size"],
            "mode": "0o400",
            "previous_sha256": None,
            "payload_size": 0,
            "payload_sha256": hashlib.sha256(b"").hexdigest(),
        },
    }
    receipt["receipt_sha256"] = document_sha256(receipt, "receipt_sha256")
    return attestation_document, _document(receipt_relative, receipt)


def test_john1_attestation_direct_put_receipt_is_deterministically_bound() -> None:
    attestation, publication = _john1_attestation_publication_documents()
    binding = validate_john1_attestation_publication_receipt(
        attestation_document=attestation,
        publication_document=publication,
    )
    assert binding == {
        "relative": publication.evidence.relative,
        "object_sha256": publication.payload_sha256,
        "object_token_sha256": publication.evidence.object_token["token_sha256"],
        "receipt_sha256": publication.value["receipt_sha256"],
    }


@pytest.mark.parametrize(
    ("target", "field", "value", "message"),
    (
        ("attestation", "attestation_sha256", "f" * 64, "attestation object"),
        ("result", "previous_sha256", "e" * 64, "immutable publication"),
        ("result", "mode", "0o600", "immutable publication"),
    ),
)
def test_john1_attestation_publication_chain_rejects_tampering(
    target: str, field: str, value: object, message: str
) -> None:
    attestation, publication = _john1_attestation_publication_documents()
    if target == "attestation":
        attestation.value[field] = value
    else:
        publication.value["result"][field] = value
        publication.value["receipt_sha256"] = document_sha256(
            publication.value, "receipt_sha256"
        )
        publication = _document(publication.evidence.relative, publication.value)
    with pytest.raises(R2MapRemoteIdentityError, match=message):
        validate_john1_attestation_publication_receipt(
            attestation_document=attestation,
            publication_document=publication,
        )


def test_john1_attestation_publication_chain_rejects_token_tampering() -> None:
    attestation, publication = _john1_attestation_publication_documents()
    publication.evidence.object_token["token_sha256"] = "f" * 64
    with pytest.raises(R2MapRemoteIdentityError, match="attestation object"):
        validate_john1_attestation_publication_receipt(
            attestation_document=attestation,
            publication_document=publication,
        )


def _bootstrap_barrier() -> dict:
    target = "datasets/bootstrap-final"
    task_contract = (
        ("bootstrap-generate-john1", "generate", "john1"),
        ("bootstrap-generate-john2", "generate", "john2"),
        ("bootstrap-generate-john3", "generate", "john3"),
        ("bootstrap-generation-aggregate", "aggregate", "john1"),
    )
    phase_receipts = []
    for operation, kind, host in task_contract:
        task_id = f"r2map-r0001-{operation}"
        phase_receipts.append(
            {
                "task_id": task_id,
                "task_kind": kind,
                "host": host,
                "packet_sha256": "1" * 64,
                "receipt_relative": f"control/receipts/{task_id}.json",
                "receipt_sha256": "2" * 64,
            }
        )
    value = {
        "schema_version": 1,
        "schema_id": BOOTSTRAP_PHASE_BARRIER_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "phase": "bootstrap-generating",
        "controller_state_sha256": "3" * 64,
        "aggregate_task_id": "bootstrap-generation-aggregate",
        "phase_receipts": phase_receipts,
        "receipt_count": 4,
        "dataset_transaction": {
            "target_relative": target,
            "manifest_relative": f"{target}/.r2-map-transaction.json",
            "manifest_sha256": "4" * 64,
            "commit_receipt_relative": "control/receipts/req-dataset-commit.json",
            "commit_receipt_sha256": "5" * 64,
        },
        "compact_index": {
            "relative": f"{target}/index.json",
            "payload_sha256": "6" * 64,
            "index_blake3": "7" * 64,
            "protocol_id": "r2-map-compact-index-v3",
            "game_count": 100_000,
            "collection_kind": "bootstrap",
            "dataset_blake3": "8" * 64,
            "shard_root_relative": f"{target}/shards",
            "shard_count": 3,
        },
        "generation_manifest": {
            "relative": f"{target}.generation-manifest.json",
            "bytes": 123,
            "payload_sha256": "9" * 64,
            "identity_sha256": "a" * 64,
            "publication_receipt_relative": (
                "control/receipts/req-generation-manifest.json"
            ),
            "publication_receipt_sha256": "b" * 64,
        },
    }
    value["identity_sha256"] = hashlib.sha256(canonical_json(value)).hexdigest()
    value["publication_receipt_relative"] = (
        "control/receipts/req-bootstrap-barrier-"
        f"{value['identity_sha256'][:32]}.json"
    )
    value["barrier_sha256"] = document_sha256(value, "barrier_sha256")
    return value


def _bootstrap_generation_manifest() -> dict:
    target = "datasets/bootstrap-final"
    counts = (33_334, 33_333, 33_333)
    receipts = []
    bindings = []
    next_index = 0
    for ordinal, (host, count) in enumerate(
        zip(("john1", "john2", "john3"), counts, strict=True)
    ):
        task_id = f"r2map-r0001-bootstrap-generate-{host}"
        source_path = f"runs/bootstrap/{host}/source-{ordinal}.r2sh"
        target_sha = f"{ordinal + 1}" * 64
        file_name = f"bootstrap-{ordinal:03d}.r2sh"
        artifact = {
            "label": "generate-artifact",
            "path": source_path,
            "bytes": 100 + ordinal,
            "sha256": target_sha,
            "storage_receipt_relative": f"control/receipts/req-source-{ordinal}.json",
            "storage_receipt_sha256": f"{ordinal + 4}" * 64,
        }
        receipts.append(
            {
                "task_id": task_id,
                "host": host,
                "packet_sha256": f"{ordinal + 7}" * 64,
                "receipt_relative": f"control/receipts/{task_id}.json",
                "receipt_sha256": "a" * 64,
                "used_seed_prefix": {
                    "lease_sha256": "b" * 64,
                    "used_count": count,
                    "unused_count": 1_000_000 - count,
                    "last_index": count - 1,
                },
                "artifacts": [artifact],
            }
        )
        bindings.append(
            {
                "source_task_id": task_id,
                "source_artifact_path": source_path,
                "source_artifact_sha256": target_sha,
                "source_artifact_bytes": 100 + ordinal,
                "target_relative": f"{target}/shards/{file_name}",
                "target_sha256": target_sha,
                "target_blake3": "c" * 64,
                "bytes": 100 + ordinal,
                "file_name": file_name,
                "first_game_index": next_index,
                "next_game_index": next_index + count,
                "game_count": count,
            }
        )
        next_index += count
    value = {
        "schema_version": 1,
        "schema_id": BOOTSTRAP_GENERATION_MANIFEST_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "phase": "bootstrap-generating",
        "controller_state_sha256": "d" * 64,
        "aggregate_task_id": "bootstrap-generation-aggregate",
        "generation_receipts": receipts,
        "dataset_transaction": {
            "target_relative": target,
            "manifest_relative": f"{target}/.r2-map-transaction.json",
            "manifest_sha256": "e" * 64,
            "commit_receipt_relative": "control/receipts/req-dataset-commit.json",
            "commit_receipt_sha256": "f" * 64,
        },
        "compact_index": {
            "relative": f"{target}/index.json",
            "bytes": 1_000,
            "sha256": "0" * 64,
            "index_blake3": "1" * 64,
            "protocol_id": "r2-map-compact-index-v3",
            "collection_kind": "bootstrap",
            "game_count": 100_000,
            "dataset_blake3": "2" * 64,
            "shard_root_relative": f"{target}/shards",
            "shard_count": 3,
        },
        "shard_bindings": bindings,
    }
    value["identity_sha256"] = document_sha256(value, "identity_sha256")
    return value


def test_bootstrap_phase_barrier_derives_its_publication_locator_without_a_cycle() -> None:
    barrier = _bootstrap_barrier()
    assert (
        validate_bootstrap_phase_barrier_value(
            barrier,
            dataset_target_relative="datasets/bootstrap-final",
        )
        == barrier
    )


def test_bootstrap_generation_manifest_bijectively_binds_three_receipts_to_shards() -> None:
    manifest = _bootstrap_generation_manifest()
    assert (
        validate_bootstrap_generation_manifest_value(
            manifest,
            dataset_target_relative="datasets/bootstrap-final",
        )
        == manifest
    )


def test_bootstrap_generation_manifest_rejects_unrelated_shard_pairing() -> None:
    manifest = _bootstrap_generation_manifest()
    manifest["shard_bindings"][0]["source_artifact_sha256"] = "9" * 64
    manifest["identity_sha256"] = document_sha256(manifest, "identity_sha256")
    with pytest.raises(R2MapRemoteIdentityError, match="binding"):
        validate_bootstrap_generation_manifest_value(
            manifest,
            dataset_target_relative="datasets/bootstrap-final",
        )


def test_bootstrap_generation_manifest_rejects_per_task_game_mismatch() -> None:
    manifest = _bootstrap_generation_manifest()
    manifest["shard_bindings"][0]["game_count"] -= 1
    manifest["shard_bindings"][0]["next_game_index"] -= 1
    manifest["shard_bindings"][1]["game_count"] += 1
    manifest["shard_bindings"][1]["next_game_index"] += 1
    manifest["identity_sha256"] = document_sha256(manifest, "identity_sha256")
    with pytest.raises(R2MapRemoteIdentityError, match="per-task"):
        validate_bootstrap_generation_manifest_value(
            manifest,
            dataset_target_relative="datasets/bootstrap-final",
        )


def _aggregate_binding_fixture() -> tuple[dict, dict, dict, dict]:
    barrier = _bootstrap_barrier()
    generation = _bootstrap_generation_manifest()
    generation_section = barrier["generation_manifest"]
    generation_section["relative"] = "datasets/bootstrap-final.generation-manifest.json"
    generation_section["bytes"] = 123
    generation_section["payload_sha256"] = "9" * 64
    generation_section["identity_sha256"] = generation["identity_sha256"]
    generation_section["publication_receipt_relative"] = (
        "control/receipts/req-generation-manifest.json"
    )
    generation_section["publication_receipt_sha256"] = "8" * 64
    packets = {}
    receipts = {}
    for mirrored in generation["generation_receipts"]:
        task_id = mirrored["task_id"]
        packets[task_id] = {"packet_sha256": mirrored["packet_sha256"]}
        receipts[task_id] = {
            "host": mirrored["host"],
            "receipt_sha256": mirrored["receipt_sha256"],
            "used_seed_prefix": copy.deepcopy(mirrored["used_seed_prefix"]),
            "artifacts": copy.deepcopy(mirrored["artifacts"]),
        }
    aggregate_task_id = barrier["phase_receipts"][-1]["task_id"]
    generation_task_ids = [entry["task_id"] for entry in barrier["phase_receipts"][:-1]]
    packets[aggregate_task_id] = {
        "dependencies": generation_task_ids,
        "aggregate_kind": "generation",
    }
    receipts[aggregate_task_id] = {
        "artifacts": [
            {
                "label": "generation-manifest",
                "path": generation_section["relative"],
                "bytes": generation_section["bytes"],
                "sha256": generation_section["payload_sha256"],
                "storage_receipt_relative": generation_section[
                    "publication_receipt_relative"
                ],
                "storage_receipt_sha256": generation_section[
                    "publication_receipt_sha256"
                ],
            }
        ]
    }
    return barrier, generation, packets, receipts


def test_aggregate_receipt_singularly_binds_the_generation_manifest() -> None:
    barrier, generation, packets, receipts = _aggregate_binding_fixture()
    assert validate_bootstrap_aggregate_generation_binding(
        barrier=barrier,
        generation_manifest=generation,
        packets_by_task=packets,
        receipts_by_task=receipts,
    ) == tuple(entry["task_id"] for entry in barrier["phase_receipts"][:-1])


def test_aggregate_receipt_rejects_an_unrelated_generation_manifest() -> None:
    barrier, generation, packets, receipts = _aggregate_binding_fixture()
    aggregate_task_id = barrier["phase_receipts"][-1]["task_id"]
    receipts[aggregate_task_id]["artifacts"][0]["sha256"] = "7" * 64
    with pytest.raises(R2MapRemoteIdentityError, match="generation-manifest binding"):
        validate_bootstrap_aggregate_generation_binding(
            barrier=barrier,
            generation_manifest=generation,
            packets_by_task=packets,
            receipts_by_task=receipts,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("publication_receipt_relative", "control/receipts/req-caller-chosen.json"),
        ("receipt_count", 3),
    ),
)
def test_bootstrap_phase_barrier_rejects_identity_drift(field: str, value: object) -> None:
    barrier = _bootstrap_barrier()
    barrier[field] = value
    barrier["barrier_sha256"] = document_sha256(barrier, "barrier_sha256")
    with pytest.raises(R2MapRemoteIdentityError, match="barrier"):
        validate_bootstrap_phase_barrier_value(
            barrier,
            dataset_target_relative="datasets/bootstrap-final",
        )
