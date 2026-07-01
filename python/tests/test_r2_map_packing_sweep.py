from __future__ import annotations

import copy
import hashlib
import json

import blake3
import pytest
from cascadia_mlx.r2_map_local_write_guard import (
    JOHN1_MLX_INTERPRETER,
    SANDBOX_PROFILE,
    SANDBOX_PROFILE_SHA256,
)
from cascadia_mlx.r2_map_packing_sweep import (
    MAXIMUM_CANDIDATES_PER_BATCH,
    MAXIMUM_WIDTH_CANDIDATES,
    PRODUCTION_MEASUREMENT_PROTOCOL,
    QUALIFYING_CAPS,
    QUALIFYING_EPOCHS,
    QUALIFYING_GAMES,
    QUALIFYING_SWEEP_SCHEMA,
    REPRESENTATIVE_MEASUREMENT_PROTOCOL,
    SELECTOR_ID,
    R2MapPackingSweepError,
    report_sha256,
    selected_training_contract,
    validate_qualifying_packing_report,
    validate_sweep_local_write_attestation,
)
from cascadia_mlx.r2_map_remote_storage import document_sha256


def _measurement(label: str, protocol: str, *, remote: bool = False) -> dict:
    if label == "synthetic-maximum-width":
        groups = 1
        widths = [MAXIMUM_WIDTH_CANDIDATES]
    elif label.startswith("g"):
        groups = int(label.split("-", 1)[0][1:])
        representative_width = (
            100 if label.endswith("imitation-max") else 10 if label.endswith("imitation-p50") else 1
        )
        widths = [representative_width, *([1] * (groups - 1))]
    else:
        raise AssertionError(label)
    candidates = sum(widths)
    padded_candidates = max(widths) * groups
    return {
        "label": label,
        "measurement_protocol": protocol,
        "warmup_steps": 1,
        "warmup_synchronized": True,
        "timed_steps": 5,
        "elapsed_ns": 50,
        "step_durations_ns": [10, 10, 10, 10, 10],
        "p50_step_duration_ns": 10,
        "steps_per_second": 5 * 1_000_000_000 / 50,
        "draft_groups_per_second": groups * 5 * 1_000_000_000 / 50,
        "draft_candidates_per_second": candidates * 5 * 1_000_000_000 / 50,
        "training_counters": {
            "draft_groups": groups * 5,
            "draft_candidates": candidates * 5,
            "padded_draft_candidates": padded_candidates * 5,
            "draft_policy_targets": sum(width > 1 for width in widths) * 5,
            "market_groups": 0,
            "market_actions": 0,
            "market_policy_targets": 0,
        },
        "resource_receipt": {
            "maximum_rss_bytes": 1,
            "process_swaps": 0,
            "system_swap_baseline_bytes": 0,
            "maximum_system_swap_bytes": 0,
            "system_swap_delta_bytes": 0,
            "sample_count": 1,
        },
        "mlx_memory": {"active_bytes": 1, "cache_bytes": 1, "peak_active_bytes": 1},
        "expected_group_count_per_step": [groups] * 5,
        "observed_group_count_per_step": [groups] * 5,
        "decode_and_padding_inside_timed_step": True,
        "mlx_allocation_inside_timed_step": True,
        "remote_window_acquisition_inside_timed_interval": remote,
        "remote_windows_acquired": 1 if remote else 0,
        "remote_window_durations_ns": [2] if remote else [],
        "remote_window_duration_ns_per_step": [2, 0, 0, 0, 0] if remote else [0, 0, 0, 0, 0],
        "candidate_widths": [] if remote else widths,
        "frame_indices": (
            [] if remote or label == "synthetic-maximum-width" else list(range(groups))
        ),
    }


def _remote_object_evidence(relative: str, payload_sha256: str, *, size: int = 7) -> dict:
    token = {
        "schema_version": 1,
        "schema_id": "cascadia.r2-map.remote-object-token.v1",
        "relative": relative,
        "sha256": payload_sha256,
        "size": size,
        "device": 1,
        "inode": 2,
        "mtime_ns": 3,
        "ctime_ns": 4,
        "mode": 0o400,
    }
    token["token_sha256"] = document_sha256(token, "token_sha256")
    return {
        "relative": relative,
        "object_token": token,
        "open_receipt": {
            "storage_receipt_relative": "control/receipts/req-open.json",
            "storage_receipt_sha256": "a" * 64,
        },
        "range_receipts": [
            {
                "payload_sha256": "b" * 64,
                "object_token_sha256": token["token_sha256"],
                "offset": 0,
                "length": size,
                "storage_receipt_relative": f"control/receipts/req-range-{size}.json",
                "storage_receipt_sha256": "c" * 64,
            }
        ],
    }


def _remote_json_identity(relative: str, payload_sha256: str) -> dict:
    return {
        "payload_sha256": payload_sha256,
        "payload_blake3": "d" * 64,
        "evidence": _remote_object_evidence(relative, payload_sha256),
    }


def _window_evidence(source: str) -> dict:
    run_id = f"r2win-{'a' * 32}"
    manifest = _remote_object_evidence(f"build/run-{run_id}/window.json", "1" * 64)
    dataset = _remote_object_evidence(f"build/run-{run_id}/window.r2map", "2" * 64)
    value = {
        "schema_version": 1,
        "schema_id": "cascadia.r2-map.window-read-evidence.v1",
        "run_id": run_id,
        "source": source,
        "mode": "train",
        "epoch": 0,
        "sampler_seed": 20260618,
        "run_receipt": {
            "run_id": run_id,
            "cwd_relative": "datasets/bootstrap-final/shards",
            "argv_sha256": "3" * 64,
            "output_relative": f"logs/window-exports/{run_id}",
            "exit_code": 0,
            "timed_out": False,
            "duration_ms": 1,
            "run_bytes": 100,
            "max_run_bytes": 40 * (1 << 30),
            "campaign_bytes_before": 1_000,
            "campaign_bytes_after": 1_100,
            "campaign_bytes_delta": 100,
            "free_bytes_after": 200 * (1 << 30),
            "temporary_cleaned": True,
            "controller_mode": False,
            "stdout_sha256": hashlib.sha256(b"").hexdigest(),
            "stdout_size": 0,
            "stderr_sha256": hashlib.sha256(b"").hexdigest(),
            "stderr_size": 0,
            "run_receipt_sha256": "4" * 64,
            "storage_receipt_relative": "control/receipts/req-run.json",
            "storage_receipt_sha256": "4" * 64,
        },
        "manifest": manifest,
        "dataset": dataset,
        "cleanup_prepare_receipt": {
            "storage_receipt_relative": "control/receipts/req-cleanup-prepare.json",
            "storage_receipt_sha256": "5" * 64,
        },
        "cleanup_commit_receipt": {
            "run_id": run_id,
            "cleanup_token_sha256": "6" * 64,
            "manifest_object_token_sha256": manifest["object_token"]["token_sha256"],
            "dataset_object_token_sha256": dataset["object_token"]["token_sha256"],
            "removed_bytes": 100,
            "build_already_removed": False,
            "cache_already_removed": False,
            "build_removed": True,
            "cache_removed": True,
            "storage_receipt_relative": "control/receipts/req-cleanup-commit.json",
            "storage_receipt_sha256": "7" * 64,
        },
    }
    value["evidence_sha256"] = document_sha256(value, "evidence_sha256")
    return value


def _sandbox_argv() -> list[str]:
    return [
        "/usr/bin/sandbox-exec",
        "-p",
        SANDBOX_PROFILE,
        JOHN1_MLX_INTERPRETER,
        "-B",
        "/Users/johnherrick/cascadia/tools/r2_map_john1_packing_sweep.py",
        "--source-transaction-manifest-relative",
        "source/frozen/.r2-map-transaction.json",
        "--source-transaction-commit-receipt-relative",
        "control/receipts/req-source-commit.json",
        "--dataset-transaction-manifest-relative",
        "datasets/bootstrap-final/.r2-map-transaction.json",
        "--run-id",
        "packing-test",
    ]


def _attestation_authority_kwargs() -> dict[str, str | int]:
    return {
        "source_transaction_manifest_relative": (
            "source/frozen/.r2-map-transaction.json"
        ),
        "source_transaction_commit_receipt_relative": (
            "control/receipts/req-source-commit.json"
        ),
        "dataset_transaction_manifest_relative": (
            "datasets/bootstrap-final/.r2-map-transaction.json"
        ),
        "maximum_window_bytes": 1 << 30,
        "warmup_steps": 1,
        "timed_steps": 5,
        "seed": 20260618,
    }


def test_sandbox_argv_uses_reviewed_john1_authority_without_ambient_resolution() -> None:
    sandbox_argv = _sandbox_argv()
    assert sandbox_argv[3] == JOHN1_MLX_INTERPRETER
    assert sandbox_argv[5] == (
        "/Users/johnherrick/cascadia/tools/r2_map_john1_packing_sweep.py"
    )


def _report() -> dict:
    plans = []
    projections = []
    for cap in QUALIFYING_CAPS:
        epochs = [
            {
                "epoch": epoch,
                "steps": 10,
                "draft_groups": 100,
                "selected_only_groups": 90,
                "draft_policy_targets": 10,
                "draft_candidates": 200,
                "padded_draft_candidates": 300,
                "maximum_batch_groups": cap,
                "minimum_batch_groups": 1,
            }
            for epoch in range(QUALIFYING_EPOCHS)
        ]
        plan = {
            "schema_version": 1,
            "schema_id": "r2-map-compact-packing-plan-v1",
            "dataset_blake3": "1" * 64,
            "seed": 20260618,
            "epochs": QUALIFYING_EPOCHS,
            "group_batch_size": cap,
            "maximum_candidates_per_batch": MAXIMUM_CANDIDATES_PER_BATCH,
            "epoch_plans": epochs,
            "totals": {
                "steps": 120,
                "draft_groups": 1200,
                "selected_only_groups": 1080,
                "draft_policy_targets": 120,
                "draft_candidates": 2400,
                "padded_draft_candidates": 3600,
            },
            "maximum_candidate_width": 100,
            "maximum_batch_groups": cap,
            "minimum_batch_groups": 1,
        }
        plan["plan_blake3"] = blake3.blake3(
            json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        ).hexdigest()
        plans.append(plan)
        compute_rates = {
            "optimistic": 8 / 1_000_000_000,
            "central": 10 / 1_000_000_000,
            "conservative": 10 / 1_000_000_000,
        }
        remote_rates = {
            "optimistic": 2 / 1_000_000_000,
            "central": 2 / 1_000_000_000,
            "conservative": 2 / 1_000_000_000,
        }
        central_epoch = 10 * compute_rates["central"] + remote_rates["central"]
        optimistic_epoch = 10 * compute_rates["optimistic"] + remote_rates["optimistic"]
        conservative_epoch = 10 * compute_rates["conservative"] + remote_rates["conservative"]
        projections.append(
            {
                "group_batch_size": cap,
                "method": "exact-plan-compute-plus-all-source-remote-window-rate-v4",
                "steps_per_epoch": [10] * QUALIFYING_EPOCHS,
                "remote_windows_per_epoch": 1,
                "central_seconds_per_epoch": [central_epoch] * QUALIFYING_EPOCHS,
                "central_12_epoch_wall_seconds": central_epoch * QUALIFYING_EPOCHS,
                "optimistic_12_epoch_wall_seconds": optimistic_epoch * QUALIFYING_EPOCHS,
                "conservative_12_epoch_wall_seconds": conservative_epoch * QUALIFYING_EPOCHS,
                "compute_seconds_per_step": compute_rates,
                "remote_seconds_per_window": remote_rates,
                "includes_remote_window_acquisition": True,
            }
        )
    representatives = [
        _measurement(f"g{cap}-{suffix}", REPRESENTATIVE_MEASUREMENT_PROTOCOL)
        for cap in QUALIFYING_CAPS
        for suffix in ("selected", "imitation-p50", "imitation-max")
    ]
    production = [
        _measurement(f"g{cap}-production", PRODUCTION_MEASUREMENT_PROTOCOL, remote=True)
        for cap in QUALIFYING_CAPS
    ]
    catalog_window = _window_evidence("bootstrap-000.r2sh")
    report = {
        "schema_version": 3,
        "schema_id": QUALIFYING_SWEEP_SCHEMA,
        "qualification_status": "qualifying-exact-bootstrap",
        "run_id": "packing-test",
        "source_identity": {
            "source_blake3": "3" * 64,
            "source_manifest": {},
            "reference_manifest": {},
            "source_archive": {
                "relative": "source/frozen/source.tar",
                "sha256": "a" * 64,
                "size": 10_240,
                "mode": "0400",
            },
            "source_archive_verification": {},
            "source_archive_verification_descriptor": {
                "relative": "source/frozen/source-archive-verification.json",
                "sha256": "b" * 64,
                "size": 512,
                "mode": "0400",
            },
            "source_archive_verifier": {
                "relative": "source/frozen/archive-verify.py",
                "sha256": "c" * 64,
                "size": 1_024,
                "mode": "0500",
            },
            "source_gate_aliases": {
                alias: {
                    "relative": f"source/frozen/{alias}",
                    "sha256": digest * 64,
                    "size": 256,
                    "mode": "0400",
                }
                for alias, digest in (
                    ("target.mk", "d"),
                    ("p1.mk", "e"),
                    ("release.mk", "f"),
                    ("python.mk", "0"),
                    ("compile.mk", "1"),
                    ("fixture.mk", "2"),
                )
            },
            "transaction_manifest": {},
            "transaction_manifest_sha256": "7" * 64,
            "transaction_commit_receipt": {},
            "transaction_commit_receipt_sha256": "8" * 64,
            "maximum_width_panel_sha256": "4" * 64,
            "maximum_width_candidates": MAXIMUM_WIDTH_CANDIDATES,
        },
        "dataset_identity": {
            "dataset_blake3": "1" * 64,
            "game_count": QUALIFYING_GAMES,
            "collection_kind": "bootstrap",
            "shard_root_relative": "datasets/bootstrap-final/shards",
            "exporter_relative": "source/frozen/cascadia-cli-v2",
            "compact_index": {},
            "transaction_manifest": {},
            "transaction_manifest_sha256": "5" * 64,
            "transaction_commit_receipt": {},
            "transaction_commit_receipt_sha256": "6" * 64,
            "bootstrap_phase_barrier": {
                "barrier_relative": (
                    "datasets/bootstrap-final.bootstrap-phase-barrier.json"
                ),
                "identity_sha256": "e" * 64,
                "barrier_sha256": "f" * 64,
                "controller_state_sha256": "0" * 64,
                "phase_receipt_count": 4,
                "generation_manifest_relative": (
                    "datasets/bootstrap-final.generation-manifest.json"
                ),
                "generation_manifest_payload_sha256": "1" * 64,
                "generation_manifest_identity_sha256": "2" * 64,
                "generation_manifest_publication_receipt_relative": (
                    "control/receipts/req-generation-manifest.json"
                ),
                "generation_manifest_publication_receipt_sha256": "3" * 64,
                "dataset_target_relative": "datasets/bootstrap-final",
                "dataset_transaction_manifest_relative": (
                    "datasets/bootstrap-final/.r2-map-transaction.json"
                ),
                "dataset_transaction_commit_receipt_relative": (
                    "control/receipts/req-dataset-commit.json"
                ),
                "compact_index_relative": "datasets/bootstrap-final/index.json",
                "shard_root_relative": "datasets/bootstrap-final/shards",
                "barrier_document": _remote_json_identity(
                    "datasets/bootstrap-final.bootstrap-phase-barrier.json", "a" * 64
                ),
                "publication_receipt": _remote_json_identity(
                    f"control/receipts/req-bootstrap-barrier-{'e' * 32}.json",
                    "b" * 64,
                ),
                "publication_receipt_sha256": "c" * 64,
                "generation_manifest_document": _remote_json_identity(
                    "datasets/bootstrap-final.generation-manifest.json", "1" * 64
                ),
                "generation_manifest_publication_receipt": _remote_json_identity(
                    "control/receipts/req-generation-manifest.json", "4" * 64
                ),
            },
        },
        "packing_contract": {
            "group_batch_sizes": list(QUALIFYING_CAPS),
            "maximum_candidates_per_batch": MAXIMUM_CANDIDATES_PER_BATCH,
            "maximum_window_bytes": 1 << 30,
            "games": QUALIFYING_GAMES,
            "epochs": QUALIFYING_EPOCHS,
            "warmup_steps": 1,
            "timed_steps": 5,
            "seed": 20260618,
            "production_measurement_protocol": PRODUCTION_MEASUREMENT_PROTOCOL,
            "representative_measurement_protocol": REPRESENTATIVE_MEASUREMENT_PROTOCOL,
            "coverage": [
                "selected-only",
                "imitation-p50",
                "imitation-maximum",
                "registered-maximum-width",
            ],
        },
        "registered_maximum_width": {
            "candidate_count": MAXIMUM_WIDTH_CANDIDATES,
            "panel_sha256": "4" * 64,
            "synthetic_resource_gate_only": True,
            "measurement": _measurement(
                "synthetic-maximum-width", REPRESENTATIVE_MEASUREMENT_PROTOCOL
            ),
        },
        "width_census": {
            "draft_groups": 100,
            "selected_only_groups": 80,
            "imitation_groups": 20,
            "imitation_minimum": 2,
            "imitation_median": 10,
            "imitation_maximum": 100,
        },
        "packing_plans": plans,
        "representative_measurements": representatives,
        "production_path_measurements": production,
        "source_window_timings": [
            {
                "source": "bootstrap-000.r2sh",
                "duration_ns": 2,
                "window_run_id": catalog_window["run_id"],
                "window_evidence_sha256": catalog_window["evidence_sha256"],
            }
        ],
        "wall_projections": projections,
        "selection": {
            "selector": SELECTOR_ID,
            "selected_group_batch_size": 16,
            "selected_schedule_steps": 120,
            "selected_epochs": QUALIFYING_EPOCHS,
            "selected_conservative_12_epoch_wall_seconds": projections[0][
                "conservative_12_epoch_wall_seconds"
            ],
            "candidates": [
                {
                    "group_batch_size": cap,
                    "resource_pass": True,
                    "candidate_budget_pass": True,
                    "conservative_12_epoch_wall_seconds": projections[index][
                        "conservative_12_epoch_wall_seconds"
                    ],
                }
                for index, cap in enumerate(QUALIFYING_CAPS)
            ],
            "rationale": "frozen",
        },
        "sweep_resource_receipt": {
            "maximum_rss_bytes": 1,
            "process_swaps": 0,
            "system_swap_baseline_bytes": 0,
            "maximum_system_swap_bytes": 0,
            "system_swap_delta_bytes": 0,
            "sample_count": 1,
        },
        "window_evidence_publications": [catalog_window],
        "ssh_transport": {
            "alias": "john2",
            "compression": "no",
            "hostname": "100.100.43.38",
            "user": "john2",
            "identityfile": "/Users/johnherrick/.ssh/john2_codex",
            "controlmaster": "no",
            "controlpath": "none",
            "updatehostkeys": "no",
        },
        "storage_preflight_receipt": {
            "storage_receipt_relative": "control/receipts/req-test.json",
            "storage_receipt_sha256": "9" * 64,
        },
        "local_write_guard": {
            "schema_version": 1,
            "schema_id": "cascadia.r2-map.john1-local-write-sandbox.v1",
            "profile_sha256": SANDBOX_PROFILE_SHA256,
            "probe": "/Users/johnherrick/cascadia/tools/r2_map_john1_packing_sweep.py",
            "probe_errno": 1,
            "all_local_file_writes_denied": True,
            "allowed_write_path": "/dev/null",
            "attestation_relative": ("reports/w2-w3/packing-test/local-write-attestation.json"),
        },
    }
    report["report_sha256"] = report_sha256(report)
    return report


def test_qualifying_report_freezes_exact_corpus_caps_and_schedule() -> None:
    report = _report()
    assert validate_qualifying_packing_report(report) == report
    assert selected_training_contract(report) == {
        "group_batch_size": 16,
        "maximum_candidates_per_batch": MAXIMUM_CANDIDATES_PER_BATCH,
        "schedule_steps": 120,
        "epochs": QUALIFYING_EPOCHS,
        "report_sha256": report["report_sha256"],
    }


def test_zero_write_attestation_binds_the_immutable_report_publication() -> None:
    report = _report()
    report_relative = "reports/w2-w3/packing-test/packing-sweep.json"
    report_object_sha256 = "a" * 64
    snapshot = [
        {
            "path": path,
            "state": "absent",
            "entries": 0,
            "sha256": "b" * 64,
        }
        for path in sorted(
            {
                "/Users/johnherrick/.python_history",
                "/Users/johnherrick/.ssh",
                "/Users/johnherrick/.mlx",
                "/Users/johnherrick/.cache/mlx",
                "/Users/johnherrick/cascadia",
                "/Users/johnherrick/Library/Caches/mlx",
                "/Users/johnherrick/Library/Caches/com.apple.Metal",
                "/Users/johnherrick/Library/Logs/r2-map-packing-test-packing-sweep",
                "/private/tmp/r2-map-packing-test",
                "/private/var/empty/r2-map-packing-test",
                "/private/var/empty/r2-map-sweep-packing-test",
            }
        )
    ]
    sandbox_argv = _sandbox_argv()
    attestation = {
        "schema_version": 1,
        "schema_id": "cascadia.r2-map.john1-local-write-attestation.v1",
        "run_id": "packing-test",
        "tool": "packing-sweep",
        "profile_sha256": SANDBOX_PROFILE_SHA256,
        "sandbox_argv": sandbox_argv,
        "sandbox_argv_sha256": hashlib.sha256(
            json.dumps(
                sandbox_argv,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        ).hexdigest(),
        "main_receipt": {
            "schema_version": 1,
            "schema_id": "cascadia.r2-map.john1-packing-sweep-publication.v1",
            "report_relative": report_relative,
            "report_sha256": report["report_sha256"],
            "report_object_sha256": report_object_sha256,
            "report_publication_receipt_relative": ("control/receipts/req-report.json"),
            "report_publication_receipt_sha256": "d" * 64,
            "local_write_attestation_relative": (
                "reports/w2-w3/packing-test/local-write-attestation.json"
            ),
        },
        "main_stdout_bytes": 1,
        "main_stderr_bytes": 0,
        "snapshot_contract": "lstat-tree-metadata-no-follow-v1",
        "snapshot_scope_excludes_legacy_ssd": True,
        "before": snapshot,
        "after": copy.deepcopy(snapshot),
        "unchanged": True,
        "started_unix_ns": 1,
        "completed_unix_ns": 2,
    }
    attestation["attestation_sha256"] = document_sha256(attestation, "attestation_sha256")
    assert (
        validate_sweep_local_write_attestation(
            attestation,
            report_relative=report_relative,
            report_sha256_value=report["report_sha256"],
            report_object_sha256=report_object_sha256,
            **_attestation_authority_kwargs(),
        )["unchanged"]
        is True
    )
    drifted_interpreter = copy.deepcopy(attestation)
    drifted_interpreter["sandbox_argv"][3] = (
        "/Users/john2/cascadia-bench/r2-map-v1/build/ambient/python"
    )
    drifted_interpreter["sandbox_argv_sha256"] = hashlib.sha256(
        json.dumps(
            drifted_interpreter["sandbox_argv"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    drifted_interpreter["attestation_sha256"] = document_sha256(
        drifted_interpreter, "attestation_sha256"
    )
    with pytest.raises(R2MapPackingSweepError, match="attestation"):
        validate_sweep_local_write_attestation(
            drifted_interpreter,
            report_relative=report_relative,
            report_sha256_value=report["report_sha256"],
            report_object_sha256=report_object_sha256,
            **_attestation_authority_kwargs(),
        )
    drifted_authority = copy.deepcopy(attestation)
    authority_index = drifted_authority["sandbox_argv"].index(
        "--dataset-transaction-manifest-relative"
    )
    drifted_authority["sandbox_argv"][authority_index + 1] = (
        "datasets/unrelated/.r2-map-transaction.json"
    )
    drifted_authority["sandbox_argv_sha256"] = hashlib.sha256(
        json.dumps(
            drifted_authority["sandbox_argv"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    drifted_authority["attestation_sha256"] = document_sha256(
        drifted_authority, "attestation_sha256"
    )
    with pytest.raises(R2MapPackingSweepError, match="attestation"):
        validate_sweep_local_write_attestation(
            drifted_authority,
            report_relative=report_relative,
            report_sha256_value=report["report_sha256"],
            report_object_sha256=report_object_sha256,
            **_attestation_authority_kwargs(),
        )
    drifted_timing = copy.deepcopy(attestation)
    drifted_timing["sandbox_argv"].extend(["--timed-steps", "6"])
    drifted_timing["sandbox_argv_sha256"] = hashlib.sha256(
        json.dumps(
            drifted_timing["sandbox_argv"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    drifted_timing["attestation_sha256"] = document_sha256(
        drifted_timing, "attestation_sha256"
    )
    with pytest.raises(R2MapPackingSweepError, match="attestation"):
        validate_sweep_local_write_attestation(
            drifted_timing,
            report_relative=report_relative,
            report_sha256_value=report["report_sha256"],
            report_object_sha256=report_object_sha256,
            **_attestation_authority_kwargs(),
        )
    attestation["after"] = []
    attestation["attestation_sha256"] = document_sha256(attestation, "attestation_sha256")
    with pytest.raises(R2MapPackingSweepError, match="attestation"):
        validate_sweep_local_write_attestation(
            attestation,
            report_relative=report_relative,
            report_sha256_value=report["report_sha256"],
            report_object_sha256=report_object_sha256,
            **_attestation_authority_kwargs(),
        )


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("packing_contract", "group_batch_sizes"), [16, 64, 128]),
        (("dataset_identity", "game_count"), 99_999),
        (("packing_contract", "epochs"), 11),
        (("source_identity", "source_archive", "mode"), "0500"),
        (
            ("source_identity", "source_gate_aliases", "target.mk", "sha256"),
            "not-a-digest",
        ),
        (
            (
                "production_path_measurements",
                0,
                "remote_window_acquisition_inside_timed_interval",
            ),
            False,
        ),
        (("representative_measurements", 0, "expected_group_count_per_step"), [15] * 5),
        (("wall_projections", 0, "conservative_12_epoch_wall_seconds"), -1.0),
        (("wall_projections", 0, "central_seconds_per_epoch", 0), 999.0),
        (("selection", "candidates", 0, "resource_pass"), False),
        (("source_window_timings", 0, "duration_ns"), 99),
        (("window_evidence_publications", 0, "manifest", "object_token", "size"), 8),
    ),
)
def test_qualifying_report_rejects_contract_drift(path: tuple[object, ...], value: object) -> None:
    report = copy.deepcopy(_report())
    target = report
    for key in path[:-1]:
        target = target[key]  # type: ignore[index,assignment]
    target[path[-1]] = value  # type: ignore[index]
    report["report_sha256"] = report_sha256(report)
    with pytest.raises(R2MapPackingSweepError):
        validate_qualifying_packing_report(report)


def test_qualifying_report_rejects_duplicate_representative_labels() -> None:
    report = _report()
    report["representative_measurements"][-1] = copy.deepcopy(
        report["representative_measurements"][0]
    )
    report["report_sha256"] = report_sha256(report)
    with pytest.raises(R2MapPackingSweepError, match="coverage"):
        validate_qualifying_packing_report(report)


def test_zero_write_attestation_rejects_incomplete_snapshot_scope() -> None:
    report = _report()
    report_relative = "reports/w2-w3/packing-test/packing-sweep.json"
    snapshot = [
        {
            "path": "/Users/johnherrick/cascadia",
            "state": "present",
            "entries": 1,
            "apparent_bytes": 0,
            "sha256": "b" * 64,
        }
    ]
    sandbox_argv = _sandbox_argv()
    attestation = {
        "schema_version": 1,
        "schema_id": "cascadia.r2-map.john1-local-write-attestation.v1",
        "run_id": "packing-test",
        "tool": "packing-sweep",
        "profile_sha256": SANDBOX_PROFILE_SHA256,
        "sandbox_argv": sandbox_argv,
        "sandbox_argv_sha256": hashlib.sha256(
            json.dumps(
                sandbox_argv,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        ).hexdigest(),
        "main_receipt": {
            "schema_version": 1,
            "schema_id": "cascadia.r2-map.john1-packing-sweep-publication.v1",
            "report_relative": report_relative,
            "report_sha256": report["report_sha256"],
            "report_object_sha256": "a" * 64,
            "report_publication_receipt_relative": "control/receipts/req-report.json",
            "report_publication_receipt_sha256": "d" * 64,
            "local_write_attestation_relative": (
                "reports/w2-w3/packing-test/local-write-attestation.json"
            ),
        },
        "main_stdout_bytes": 1,
        "main_stderr_bytes": 0,
        "snapshot_contract": "lstat-tree-metadata-no-follow-v1",
        "snapshot_scope_excludes_legacy_ssd": True,
        "before": snapshot,
        "after": copy.deepcopy(snapshot),
        "unchanged": True,
        "started_unix_ns": 1,
        "completed_unix_ns": 2,
    }
    attestation["attestation_sha256"] = document_sha256(attestation, "attestation_sha256")
    with pytest.raises(R2MapPackingSweepError, match="attestation"):
        validate_sweep_local_write_attestation(
            attestation,
            report_relative=report_relative,
            report_sha256_value=report["report_sha256"],
            report_object_sha256="a" * 64,
            **_attestation_authority_kwargs(),
        )
