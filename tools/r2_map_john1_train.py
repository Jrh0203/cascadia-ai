#!/usr/bin/env python3
"""Run filesystem-free R2-MAP MLX training on John1 over frozen John2 I/O."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
REPOSITORY = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPOSITORY / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from cascadia_mlx.checkpoint import verify_r2_map_checkpoint_bundle  # noqa: E402
from cascadia_mlx.r2_map_contracts import local_campaign_host_id  # noqa: E402
from cascadia_mlx.r2_map_dataset import (  # noqa: E402
    R2MapCompactDatasetAdapter,
    compact_storage_projection,
    validate_compact_index_value,
)
from cascadia_mlx.r2_map_local_write_guard import (  # noqa: E402
    john1_attestation_publication_receipt_relative,
    require_no_local_write_sandbox,
)
from cascadia_mlx.r2_map_packing_sweep import (  # noqa: E402
    QUALIFYING_EPOCHS,
    QUALIFYING_GAMES,
    selected_training_contract,
    validate_qualifying_packing_report,
    validate_sweep_local_write_attestation,
)
from cascadia_mlx.r2_map_remote_identity import (  # noqa: E402
    load_verified_bootstrap_phase_barrier,
    load_verified_remote_json,
    require_transaction_object,
    transaction_object_descriptor,
    validate_immutable_publication_receipt,
    validate_john1_attestation_publication_receipt,
    validate_source_identity,
    validate_transaction_commit,
    validate_transaction_manifest,
)
from cascadia_mlx.r2_map_remote_storage import (  # noqa: E402
    RemoteOperationError,
    RemoteStorageClient,
    SshTransport,
    canonical_json,
)
from cascadia_mlx.r2_map_remote_training import (  # noqa: E402
    John2RemoteCheckpointStore,
    John2RemoteWindowLoader,
)
from cascadia_mlx.r2_map_train import (  # noqa: E402
    PRIMARY_VALIDATION_METRIC,
    R2MapTrainer,
    R2MapTrainerConfig,
    select_best_validation_checkpoint_bundle,
)
from cascadia_mlx.r2_map_training_resources import (  # noqa: E402
    R2MapTrainingResourceMonitor,
    validate_training_resource_receipt,
)
from cascadia_mlx.r2_map_verify import (  # noqa: E402
    verify_r2_map_checkpoint_bundle_in_memory,
)

RESULT_SCHEMA = "cascadia.r2-map.john1-training-result.v3"
WINDOW_EVIDENCE_SCHEMA = "cascadia.r2-map.window-read-evidence.v1"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--packing-report-relative", required=True)
    result.add_argument("--run-id", required=True)
    result.add_argument("--branch-id", default="main")
    result.add_argument("--checkpoint-every", type=int, default=1_000)
    result.add_argument("--checkpoint-seconds", type=int, default=300)
    result.add_argument("--loss-event-every", type=int, choices=range(10, 26), default=20)
    result.add_argument("--fixed-panel-games", type=int, default=1)
    result.add_argument("--seed", type=int, default=20260618)
    result.add_argument("--learning-rate", type=float, default=3e-5)
    result.add_argument("--minimum-learning-rate", type=float, default=3e-6)
    result.add_argument("--warmup-steps", type=int, default=10)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--resume-pointer", default="last_verified")
    return result


def _arguments() -> argparse.Namespace:
    arguments = parser().parse_args()
    if local_campaign_host_id() != "john1":
        raise SystemExit("R2-MAP MLX training is authorized only on John1")
    if (
        arguments.checkpoint_every <= 0
        or not 1 <= arguments.checkpoint_seconds <= 300
        or arguments.fixed_panel_games <= 0
        or arguments.seed < 0
    ):
        raise SystemExit("training/checkpoint bounds are invalid")
    return arguments


def _reported_relative(value: dict[str, Any], label: str) -> str:
    try:
        relative = value["evidence"]["relative"]
    except (KeyError, TypeError) as error:
        raise RuntimeError(f"packing report omits {label} evidence") from error
    if (
        not isinstance(relative, str)
        or not relative
        or relative.startswith("/")
        or ".." in Path(relative).parts
    ):
        raise RuntimeError(f"packing report {label} relative path is unsafe")
    return relative


def _require_same_object(reported: dict[str, Any], observed: Any, label: str) -> None:
    try:
        expected = reported["evidence"]["object_token"]["token_sha256"]
    except (KeyError, TypeError) as error:
        raise RuntimeError(f"packing report omits {label} object token") from error
    if observed.evidence.object_token.get("token_sha256") != expected:
        raise RuntimeError(f"receipt-bound {label} object changed after the sweep")


def _require_same_reported_remote_identity(
    reported: dict[str, Any], observed: dict[str, Any], label: str
) -> None:
    try:
        reported_token = reported["evidence"]["object_token"]["token_sha256"]
        observed_token = observed["evidence"]["object_token"]["token_sha256"]
        reported_relative = reported["evidence"]["relative"]
        observed_relative = observed["evidence"]["relative"]
    except (KeyError, TypeError) as error:
        raise RuntimeError(f"packing report omits {label} object identity") from error
    if (
        reported.get("payload_sha256") != observed.get("payload_sha256")
        or reported.get("payload_blake3") != observed.get("payload_blake3")
        or reported_token != observed_token
        or reported_relative != observed_relative
    ):
        raise RuntimeError(f"receipt-bound {label} object changed after the sweep")


def _load_packing_contract(
    client: RemoteStorageClient, report_relative: str
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    report_document = load_verified_remote_json(client, report_relative, maximum_bytes=64 << 20)
    report = validate_qualifying_packing_report(report_document.value)
    if report_document.evidence.relative != report_relative:
        raise RuntimeError("packing report object path differs")
    attestation_relative = report["local_write_guard"]["attestation_relative"]
    attestation_document = load_verified_remote_json(
        client, attestation_relative, maximum_bytes=1 << 20
    )
    attestation = validate_sweep_local_write_attestation(
        attestation_document.value,
        report_relative=report_relative,
        report_sha256_value=report["report_sha256"],
        report_object_sha256=report_document.payload_sha256,
        source_transaction_manifest_relative=_reported_relative(
            report["source_identity"]["transaction_manifest"],
            "source transaction manifest",
        ),
        source_transaction_commit_receipt_relative=_reported_relative(
            report["source_identity"]["transaction_commit_receipt"],
            "source transaction commit receipt",
        ),
        dataset_transaction_manifest_relative=_reported_relative(
            report["dataset_identity"]["transaction_manifest"],
            "dataset transaction manifest",
        ),
        maximum_window_bytes=report["packing_contract"]["maximum_window_bytes"],
        warmup_steps=report["packing_contract"]["warmup_steps"],
        timed_steps=report["packing_contract"]["timed_steps"],
        seed=report["packing_contract"]["seed"],
    )
    attestation_publication_relative = john1_attestation_publication_receipt_relative(
        attestation["attestation_sha256"]
    )
    attestation_publication_document = load_verified_remote_json(
        client,
        attestation_publication_relative,
        maximum_bytes=2 << 20,
    )
    attestation_publication = validate_john1_attestation_publication_receipt(
        attestation_document=attestation_document,
        publication_document=attestation_publication_document,
    )
    main_receipt = attestation["main_receipt"]
    publication_receipt_document = load_verified_remote_json(
        client,
        main_receipt["report_publication_receipt_relative"],
        maximum_bytes=2 << 20,
    )
    publication_receipt = validate_immutable_publication_receipt(
        publication_receipt_document,
        object_relative=report_relative,
        object_sha256=report_document.payload_sha256,
        object_size=report_document.evidence.object_token["size"],
    )
    if publication_receipt["receipt_sha256"] != main_receipt["report_publication_receipt_sha256"]:
        raise RuntimeError("packing report publication receipt locator differs")

    reported_source = report["source_identity"]
    source_manifest = load_verified_remote_json(
        client, _reported_relative(reported_source["source_manifest"], "source manifest")
    )
    reference_manifest = load_verified_remote_json(
        client,
        _reported_relative(reported_source["reference_manifest"], "reference manifest"),
    )
    source_archive_verification = load_verified_remote_json(
        client,
        _reported_relative(
            reported_source["source_archive_verification"],
            "source archive verification",
        ),
    )
    source_transaction_document = load_verified_remote_json(
        client,
        _reported_relative(reported_source["transaction_manifest"], "source transaction manifest"),
    )
    source_commit = load_verified_remote_json(
        client,
        _reported_relative(
            reported_source["transaction_commit_receipt"],
            "source transaction commit receipt",
        ),
        maximum_bytes=2 << 20,
    )
    source_identity = validate_source_identity(
        source_manifest=source_manifest,
        reference_manifest=reference_manifest,
        source_archive_verification=source_archive_verification,
        transaction_manifest=source_transaction_document,
        transaction_commit_receipt=source_commit,
    )
    for name, document in (
        ("source_manifest", source_manifest),
        ("reference_manifest", reference_manifest),
        ("source_archive_verification", source_archive_verification),
        ("transaction_manifest", source_transaction_document),
        ("transaction_commit_receipt", source_commit),
    ):
        _require_same_object(reported_source[name], document, name)
    for name in (
        "source_blake3",
        "transaction_manifest_sha256",
        "transaction_commit_receipt_sha256",
        "maximum_width_panel_sha256",
        "maximum_width_candidates",
        "source_archive",
        "source_archive_verification_descriptor",
        "source_archive_verifier",
        "source_gate_aliases",
    ):
        if source_identity[name] != reported_source.get(name):
            raise RuntimeError(f"independent source identity differs: {name}")
    source_transaction = validate_transaction_manifest(source_transaction_document)

    reported_dataset = report["dataset_identity"]
    dataset_transaction_hint = load_verified_remote_json(
        client,
        _reported_relative(
            reported_dataset["transaction_manifest"], "dataset transaction manifest"
        ),
    )
    (
        phase_barrier_identity,
        index_document,
        dataset_transaction_document,
        dataset_commit,
    ) = load_verified_bootstrap_phase_barrier(
        client,
        dataset_transaction_hint=dataset_transaction_hint,
    )
    reported_phase_barrier = reported_dataset["bootstrap_phase_barrier"]
    for name in (
        "barrier_relative",
        "identity_sha256",
        "barrier_sha256",
        "controller_state_sha256",
        "phase_receipt_count",
        "generation_manifest_relative",
        "generation_manifest_payload_sha256",
        "generation_manifest_identity_sha256",
        "generation_manifest_publication_receipt_relative",
        "generation_manifest_publication_receipt_sha256",
        "dataset_target_relative",
        "dataset_transaction_manifest_relative",
        "dataset_transaction_commit_receipt_relative",
        "compact_index_relative",
        "shard_root_relative",
        "publication_receipt_sha256",
    ):
        if phase_barrier_identity[name] != reported_phase_barrier.get(name):
            raise RuntimeError(f"independent bootstrap phase barrier differs: {name}")
    for name in (
        "barrier_document",
        "publication_receipt",
        "generation_manifest_document",
        "generation_manifest_publication_receipt",
    ):
        _require_same_reported_remote_identity(
            reported_phase_barrier[name], phase_barrier_identity[name], name
        )
    index = validate_compact_index_value(index_document.value)
    dataset_transaction = validate_transaction_manifest(dataset_transaction_document)
    if dataset_transaction["target_relative"].split("/", 1)[0] != "datasets":
        raise RuntimeError("packing-bound bootstrap transaction is outside datasets/")
    require_transaction_object(dataset_transaction, index_document)
    validate_transaction_commit(dataset_commit, dataset_transaction)
    for name, document in (
        ("compact_index", index_document),
        ("transaction_manifest", dataset_transaction_document),
        ("transaction_commit_receipt", dataset_commit),
    ):
        _require_same_object(reported_dataset[name], document, name)
    manifest = index["dataset_manifest"]
    if (
        manifest["dataset_blake3"] != reported_dataset["dataset_blake3"]
        or manifest["game_count"] != QUALIFYING_GAMES
        or manifest["round"]["collection_kind"] != "bootstrap"
        or reported_dataset["transaction_manifest_sha256"] != dataset_transaction["manifest_sha256"]
        or reported_dataset["transaction_commit_receipt_sha256"] != dataset_commit.payload_sha256
    ):
        raise RuntimeError("independent bootstrap dataset identity differs")
    exporter_descriptor = transaction_object_descriptor(
        source_transaction, reported_dataset["exporter_relative"]
    )
    if exporter_descriptor.get("mode") != "0500":
        raise RuntimeError("packing-bound compact exporter is not executable")
    for source in manifest["sources"]:
        descriptor = transaction_object_descriptor(
            dataset_transaction,
            f"{reported_dataset['shard_root_relative']}/{source['file_name']}",
        )
        if descriptor.get("size") != source["bytes"]:
            raise RuntimeError("packing-bound compact shard size differs")

    selected = selected_training_contract(report)
    phase_barrier = report["dataset_identity"]["bootstrap_phase_barrier"]
    binding = {
        "report_relative": report_relative,
        "report_sha256": report["report_sha256"],
        "report_object_sha256": report_document.payload_sha256,
        "report_object_token_sha256": report_document.evidence.object_token["token_sha256"],
        "publication_receipt_relative": publication_receipt_document.evidence.relative,
        "publication_receipt_object_sha256": publication_receipt_document.payload_sha256,
        "publication_receipt_sha256": publication_receipt["receipt_sha256"],
        "local_write_attestation_relative": attestation_relative,
        "local_write_attestation_object_sha256": attestation_document.payload_sha256,
        "local_write_attestation_object_token_sha256": attestation_document.evidence.object_token[
            "token_sha256"
        ],
        "local_write_attestation_sha256": attestation["attestation_sha256"],
        "local_write_attestation_publication_receipt_relative": attestation_publication[
            "relative"
        ],
        "local_write_attestation_publication_receipt_object_sha256": (
            attestation_publication["object_sha256"]
        ),
        "local_write_attestation_publication_receipt_object_token_sha256": (
            attestation_publication["object_token_sha256"]
        ),
        "local_write_attestation_publication_receipt_sha256": attestation_publication[
            "receipt_sha256"
        ],
        "bootstrap_phase_barrier_identity_sha256": phase_barrier["identity_sha256"],
        "bootstrap_phase_barrier_sha256": phase_barrier["barrier_sha256"],
        "bootstrap_phase_barrier_publication_receipt_sha256": phase_barrier[
            "publication_receipt_sha256"
        ],
        "bootstrap_controller_state_sha256": phase_barrier["controller_state_sha256"],
        "bootstrap_generation_manifest_payload_sha256": phase_barrier[
            "generation_manifest_payload_sha256"
        ],
        "bootstrap_generation_manifest_identity_sha256": phase_barrier[
            "generation_manifest_identity_sha256"
        ],
        "bootstrap_generation_manifest_publication_receipt_sha256": phase_barrier[
            "generation_manifest_publication_receipt_sha256"
        ],
        "selected_group_batch_size": selected["group_batch_size"],
        "maximum_candidates_per_batch": selected["maximum_candidates_per_batch"],
        "schedule_steps": selected["schedule_steps"],
        "epochs": selected["epochs"],
    }
    return report, index, source_identity, binding


def main() -> int:
    arguments = _arguments()
    local_write_guard = require_no_local_write_sandbox(Path(__file__))
    transport = SshTransport(compression=False)
    ssh_configuration = transport.verify_local_configuration()
    client = RemoteStorageClient(transport)
    preflight = client.preflight()
    packing_report, index, source_identity, packing_binding = _load_packing_contract(
        client, arguments.packing_report_relative
    )
    manifest = index["dataset_manifest"]
    game_count = manifest["game_count"]
    collection_kind = manifest["round"]["collection_kind"]
    if collection_kind != "bootstrap" or game_count != QUALIFYING_GAMES:
        raise SystemExit("production R2-MAP training requires the exact bootstrap corpus")
    maximum_window_bytes = packing_report["packing_contract"]["maximum_window_bytes"]
    storage_projection = compact_storage_projection(
        index,
        target_games=QUALIFYING_GAMES,
        maximum_window_bytes=maximum_window_bytes,
        maximum_prefetch_windows=0,
    )
    if not storage_projection.compact_fits_run_budget:
        raise SystemExit("projected compact R2-MAP corpus exceeds the 40-GiB run budget")
    if collection_kind == "bootstrap" and storage_projection.expanded_fits_run_budget:
        raise SystemExit("bootstrap expanded-corpus storage gate did not fail closed")
    store = John2RemoteCheckpointStore(client, run_id=arguments.run_id)
    window_evidence: list[dict[str, Any]] = []
    window_evidence_bytes = 0

    def record_window_evidence(evidence: Any) -> None:
        nonlocal window_evidence_bytes
        value: dict[str, Any] = {
            "schema_version": 1,
            "schema_id": WINDOW_EVIDENCE_SCHEMA,
            **evidence.to_dict(),
        }
        value["evidence_sha256"] = hashlib.sha256(canonical_json(value)).hexdigest()
        window_evidence_bytes += len(canonical_json(value))
        if window_evidence_bytes > 64 << 20:
            raise RuntimeError("training window evidence exceeds its in-memory bound")
        window_evidence.append(value)

    loader = John2RemoteWindowLoader(
        client,
        exporter_relative=packing_report["dataset_identity"]["exporter_relative"],
        shard_root_relative=packing_report["dataset_identity"]["shard_root_relative"],
        maximum_window_bytes=maximum_window_bytes,
        evidence_sink=record_window_evidence,
    )
    forbidden_local_run = Path(f"/private/var/empty/r2-map-{arguments.run_id}")
    if forbidden_local_run.exists():
        raise SystemExit("forbidden local training path unexpectedly exists")
    monitor = R2MapTrainingResourceMonitor.start()
    monitor.sample()

    with R2MapCompactDatasetAdapter(
        index=index,
        window_loader=loader,
        group_batch_size=packing_binding["selected_group_batch_size"],
        maximum_candidates_per_batch=packing_binding["maximum_candidates_per_batch"],
        maximum_window_bytes=maximum_window_bytes,
        maximum_prefetch_windows=0,
        fixed_panel_games=arguments.fixed_panel_games,
    ) as adapter:
        config = R2MapTrainerConfig(
            run_dir=forbidden_local_run,
            run_id=arguments.run_id,
            branch_id=arguments.branch_id,
            source_blake3=source_identity["source_blake3"],
            dataset_blake3=adapter.dataset_blake3,
            adapter_protocol_id=adapter.protocol_id,
            group_batch_size=packing_binding["selected_group_batch_size"],
            maximum_candidates_per_batch=packing_binding["maximum_candidates_per_batch"],
            packing_report_binding=packing_binding,
            learning_rate=arguments.learning_rate,
            minimum_learning_rate=arguments.minimum_learning_rate,
            warmup_steps=arguments.warmup_steps,
            schedule_steps=packing_binding["schedule_steps"],
            loss_event_interval_steps=arguments.loss_event_every,
            seed=arguments.seed,
        )
        existing_best = None
        if arguments.resume:
            resume = store.load_checkpoint(arguments.resume_pointer)
            trainer = R2MapTrainer.resume_from_bundle(
                config,
                adapter,
                bundle=resume.bundle,
                loss_content=resume.loss_content,
            )
            del resume
            for pointer in {"latest_complete", "last_verified"} - {arguments.resume_pointer}:
                with suppress(RemoteOperationError):
                    store.load_checkpoint(pointer)
            try:
                existing_best = store.load_checkpoint("best_validation")
            except RemoteOperationError:
                existing_best = None
        else:
            trainer = R2MapTrainer(config, adapter, in_memory=True)
            store.publish_loss_stream(b"")

        start_step = trainer.global_step
        target_step = packing_binding["schedule_steps"]
        if start_step >= target_step:
            raise RuntimeError("packing-bound training schedule is already complete or exceeded")
        last_checkpoint_monotonic = time.monotonic()
        best_candidate: tuple[Any, dict[str, Any]] | None = None
        publications = []
        if existing_best is not None:
            assert existing_best.verification_receipt is not None
            best_candidate = (
                existing_best.bundle,
                existing_best.verification_receipt,
            )
            existing_best = None

        while trainer.global_step < target_step:
            trainer.step()
            monitor.sample()
            due = (
                trainer.global_step % arguments.checkpoint_every == 0
                or time.monotonic() - last_checkpoint_monotonic >= arguments.checkpoint_seconds
                or trainer.global_step == target_step
            )
            if not due:
                continue
            validation = trainer.validation_metrics()
            monitor.sample()
            if not math.isfinite(validation[PRIMARY_VALIDATION_METRIC]):
                raise RuntimeError("R2-MAP validation loss is non-finite")
            bundle = trainer.checkpoint_bundle(validation=validation)
            monitor.sample()
            verification = verify_r2_map_checkpoint_bundle_in_memory(
                bundle,
                loss_content=trainer.loss_content,
                adapter=adapter,
            )
            publication = store.publish_checkpoint(
                bundle,
                loss_content=trainer.loss_content,
                verification_receipt=verification,
            )
            monitor.sample()
            candidate = (bundle, verification)
            if best_candidate is None:
                best_candidate = candidate
            else:
                selected = select_best_validation_checkpoint_bundle((best_candidate, candidate))
                if selected.checkpoint_id == bundle.checkpoint_id:
                    best_candidate = candidate
                del selected
            publications.append(
                {
                    **publication.to_dict(),
                    "work_artifact": publication.work_artifact(bundle),
                    "validation": validation,
                    "verification": verification,
                }
            )
            if best_candidate[0] is not bundle:
                del bundle
            del candidate
            last_checkpoint_monotonic = time.monotonic()

        if best_candidate is None:
            raise RuntimeError("R2-MAP training produced no verified checkpoint")
        if (
            trainer.global_step != target_step
            or trainer.epoch != QUALIFYING_EPOCHS
            or trainer.cursor
            != {
                "epoch": QUALIFYING_EPOCHS,
                "source_offset": 0,
                "game_offset": 0,
                "turn_offset": 0,
            }
        ):
            raise RuntimeError(
                "bootstrap training did not stop on the packing-bound exact 12-epoch schedule"
            )
        best = best_candidate[0]
        monitor.sample()
        _, best_state, _ = verify_r2_map_checkpoint_bundle(best)
        assert best_state.validation is not None
        best_pointer = store.publish_pointer(
            "best_validation",
            best,
            metadata={
                PRIMARY_VALIDATION_METRIC: best_state.validation[PRIMARY_VALIDATION_METRIC],
                "global_step": best_state.global_step,
                "checkpoint_manifest_blake3": best.manifest_blake3,
                "selection_tiebreak": "global-step-then-checkpoint-manifest-blake3",
            },
        )
        resource_receipt = validate_training_resource_receipt(monitor.receipt())
        result: dict[str, Any] = {
            "schema_version": 1,
            "schema_id": RESULT_SCHEMA,
            "run_id": arguments.run_id,
            "branch_id": arguments.branch_id,
            "source_blake3": source_identity["source_blake3"],
            "dataset_blake3": adapter.dataset_blake3,
            "dataset_contract": adapter.dataset_contract,
            "packing_report_binding": packing_binding,
            "source_identity": source_identity,
            "dataset_identity": packing_report["dataset_identity"],
            "storage_projection": asdict(storage_projection),
            "ssh_transport": ssh_configuration,
            "start_step": start_step,
            "final_step": trainer.global_step,
            "target_step": target_step,
            "completed_epochs": trainer.epoch,
            "examples_seen": trainer.examples_seen,
            "training_counters": dict(trainer.training_counters),
            "next_batch_identity": trainer.peek_next_batch_identity(),
            "checkpoint_publications": publications,
            "best_validation_checkpoint": best.checkpoint_id,
            "best_validation_pointer": best_pointer,
            "window_evidence": window_evidence,
            "resource_receipt": resource_receipt,
            "storage_preflight_receipt": {
                key: preflight[key]
                for key in ("storage_receipt_relative", "storage_receipt_sha256")
            },
            "local_write_guard": local_write_guard,
        }
        result["result_sha256"] = hashlib.sha256(canonical_json(result)).hexdigest()
        result_relative = f"runs/{arguments.run_id}/training-result.json"
        result_publication = store.publish_immutable_json(result_relative, result)
        if forbidden_local_run.exists():
            raise RuntimeError("filesystem-free training created a local run tree")
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "schema_id": "cascadia.r2-map.john1-training-publication.v1",
                    "result_relative": result_relative,
                    "result_sha256": result["result_sha256"],
                    "result_object_sha256": result_publication["sha256"],
                    "result_publication_receipt_relative": result_publication[
                        "storage_receipt_relative"
                    ],
                    "result_publication_receipt_sha256": result_publication[
                        "storage_receipt_sha256"
                    ],
                    "local_write_attestation_relative": local_write_guard["attestation_relative"],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
