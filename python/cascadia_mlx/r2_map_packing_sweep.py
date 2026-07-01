"""Frozen qualifying packing-sweep contract shared by John1 sweep and trainer."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

import blake3

from cascadia_mlx.r2_map_local_write_guard import (
    JOHN1_MLX_INTERPRETER,
    SANDBOX_PROFILE,
    SANDBOX_PROFILE_SHA256,
)
from cascadia_mlx.r2_map_remote_storage import document_sha256

QUALIFYING_SWEEP_SCHEMA = "cascadia.r2-map.john1-packing-sweep.v3"
QUALIFYING_CAPS = (16, 32, 64, 128)
MAXIMUM_CANDIDATES_PER_BATCH = 16_384
QUALIFYING_GAMES = 100_000
QUALIFYING_EPOCHS = 12
MINIMUM_TIMED_STEPS = 5
MAXIMUM_WIDTH_CANDIDATES = 6_372
SELECTOR_ID = "minimum-conservative-12-epoch-production-wall-then-lower-cap-v3"
PRODUCTION_MEASUREMENT_PROTOCOL = "r2-map-compact-production-path-measurement-v1"
REPRESENTATIVE_MEASUREMENT_PROTOCOL = "r2-map-reader-decode-measurement-v2"
TRAINING_COUNTER_NAMES = {
    "draft_groups",
    "draft_candidates",
    "padded_draft_candidates",
    "draft_policy_targets",
    "market_groups",
    "market_actions",
    "market_policy_targets",
}
RESOURCE_RECEIPT_FIELDS = {
    "maximum_rss_bytes",
    "process_swaps",
    "system_swap_baseline_bytes",
    "maximum_system_swap_bytes",
    "system_swap_delta_bytes",
    "sample_count",
}


class R2MapPackingSweepError(ValueError):
    """A sweep report cannot authorize production training."""


def report_sha256(value: Mapping[str, Any]) -> str:
    return document_sha256(value, "report_sha256")


def validate_sweep_local_write_attestation(
    value: object,
    *,
    report_relative: str,
    report_sha256_value: str,
    report_object_sha256: str,
    source_transaction_manifest_relative: str,
    source_transaction_commit_receipt_relative: str,
    dataset_transaction_manifest_relative: str,
    maximum_window_bytes: int,
    warmup_steps: int,
    timed_steps: int,
    seed: int,
) -> dict[str, Any]:
    attestation = _exact_keys(
        value,
        {
            "schema_version",
            "schema_id",
            "run_id",
            "tool",
            "profile_sha256",
            "sandbox_argv",
            "sandbox_argv_sha256",
            "main_receipt",
            "main_stdout_bytes",
            "main_stderr_bytes",
            "snapshot_contract",
            "snapshot_scope_excludes_legacy_ssd",
            "before",
            "after",
            "unchanged",
            "started_unix_ns",
            "completed_unix_ns",
            "attestation_sha256",
        },
        "local-write attestation",
    )
    expected_digest = document_sha256(attestation, "attestation_sha256")
    main_receipt = _exact_keys(
        attestation["main_receipt"],
        {
            "schema_version",
            "schema_id",
            "report_relative",
            "report_sha256",
            "report_object_sha256",
            "report_publication_receipt_relative",
            "report_publication_receipt_sha256",
            "local_write_attestation_relative",
        },
        "packing sweep publication receipt",
    )
    expected_attestation_relative = (
        f"reports/w2-w3/{attestation['run_id']}/local-write-attestation.json"
    )
    expected_report_relative = f"reports/w2-w3/{attestation['run_id']}/packing-sweep.json"
    snapshots = attestation["before"]
    required_snapshot_paths = {
        "/Users/johnherrick/cascadia",
        "/Users/johnherrick/.ssh",
        "/Users/johnherrick/.mlx",
        "/Users/johnherrick/.cache/mlx",
        "/Users/johnherrick/.python_history",
        "/Users/johnherrick/Library/Caches/mlx",
        "/Users/johnherrick/Library/Caches/com.apple.Metal",
        f"/Users/johnherrick/Library/Logs/r2-map-{attestation['run_id']}-packing-sweep",
        f"/private/tmp/r2-map-{attestation['run_id']}",
        f"/private/var/empty/r2-map-{attestation['run_id']}",
        f"/private/var/empty/r2-map-sweep-{attestation['run_id']}",
    }
    sandbox_argv = attestation["sandbox_argv"]
    expected_sandbox_prefix = [
        "/usr/bin/sandbox-exec",
        "-p",
        SANDBOX_PROFILE,
    ]
    sandbox_python = (
        sandbox_argv[3]
        if isinstance(sandbox_argv, list) and len(sandbox_argv) > 3
        else None
    )
    expected_authorities = {
        "--source-transaction-manifest-relative": source_transaction_manifest_relative,
        "--source-transaction-commit-receipt-relative": (
            source_transaction_commit_receipt_relative
        ),
        "--dataset-transaction-manifest-relative": dataset_transaction_manifest_relative,
    }
    expected_effective_arguments = {
        **expected_authorities,
        "--run-id": attestation["run_id"],
        "--maximum-window-bytes": str(maximum_window_bytes),
        "--warmup-steps": str(warmup_steps),
        "--timed-steps": str(timed_steps),
        "--seed": str(seed),
    }
    if (
        attestation["schema_version"] != 1
        or attestation["schema_id"] != "cascadia.r2-map.john1-local-write-attestation.v1"
        or attestation["tool"] != "packing-sweep"
        or attestation["profile_sha256"] != SANDBOX_PROFILE_SHA256
        or attestation["snapshot_contract"] != "lstat-tree-metadata-no-follow-v1"
        or attestation["snapshot_scope_excludes_legacy_ssd"] is not True
        or attestation["unchanged"] is not True
        or attestation["before"] != attestation["after"]
        or not isinstance(snapshots, list)
        or not snapshots
        or not _identifier_or_false(attestation["run_id"])
        or report_relative != expected_report_relative
        or not isinstance(sandbox_argv, list)
        or sandbox_argv[: len(expected_sandbox_prefix)] != expected_sandbox_prefix
        or sandbox_python != JOHN1_MLX_INTERPRETER
        or sandbox_argv[4:6]
        != ["-B", "/Users/johnherrick/cascadia/tools/r2_map_john1_packing_sweep.py"]
        or any(not isinstance(item, str) or not item for item in sandbox_argv)
        or not _sweep_sandbox_arguments_match(
            sandbox_argv,
            expected_effective_arguments=expected_effective_arguments,
        )
        or any(item == "/Volumes" or item.startswith("/Volumes/") for item in sandbox_argv)
        or attestation["sandbox_argv_sha256"]
        != hashlib.sha256(
            json.dumps(
                sandbox_argv,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        ).hexdigest()
        or not _digest_or_false(attestation["sandbox_argv_sha256"])
        or not _validate_snapshot_set(snapshots, required_snapshot_paths)
        or not isinstance(attestation["started_unix_ns"], int)
        or isinstance(attestation["started_unix_ns"], bool)
        or not isinstance(attestation["completed_unix_ns"], int)
        or isinstance(attestation["completed_unix_ns"], bool)
        or attestation["completed_unix_ns"] < attestation["started_unix_ns"]
        or attestation["attestation_sha256"] != expected_digest
        or not isinstance(attestation["main_stdout_bytes"], int)
        or isinstance(attestation["main_stdout_bytes"], bool)
        or not 0 < attestation["main_stdout_bytes"] <= 8 << 20
        or not isinstance(attestation["main_stderr_bytes"], int)
        or isinstance(attestation["main_stderr_bytes"], bool)
        or not 0 <= attestation["main_stderr_bytes"] <= 256 << 10
        or main_receipt["schema_id"] != "cascadia.r2-map.john1-packing-sweep-publication.v1"
        or main_receipt["schema_version"] != 1
        or main_receipt["report_relative"] != report_relative
        or main_receipt["report_relative"] != expected_report_relative
        or main_receipt["report_sha256"] != report_sha256_value
        or main_receipt["report_object_sha256"] != report_object_sha256
        or not _digest_or_false(main_receipt["report_sha256"])
        or not _digest_or_false(main_receipt["report_object_sha256"])
        or not _receipt_locator_or_false(
            main_receipt["report_publication_receipt_relative"]
        )
        or not _digest_or_false(main_receipt["report_publication_receipt_sha256"])
        or main_receipt["local_write_attestation_relative"] != expected_attestation_relative
    ):
        raise R2MapPackingSweepError("local-write attestation does not bind the sweep")
    return json.loads(json.dumps(attestation, allow_nan=False))


def _exact_keys(value: object, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise R2MapPackingSweepError(f"{label} schema differs")
    return value


def _positive_finite(value: object, label: str) -> float:
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise R2MapPackingSweepError(f"{label} must be finite and positive")
    return float(value)


def _digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise R2MapPackingSweepError(f"{label} is not a lowercase digest")
    return value


def _digest_or_false(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _receipt_locator_or_false(value: object) -> bool:
    if not isinstance(value, str):
        return False
    path = PurePosixPath(value)
    return bool(
        not path.is_absolute()
        and path.as_posix() == value
        and len(path.parts) == 3
        and path.parts[:2] == ("control", "receipts")
        and path.name.startswith("req-")
        and path.suffix == ".json"
        and all(part not in {"", ".", ".."} for part in path.parts)
    )


def _safe_relative_or_false(value: object) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return bool(
        not path.is_absolute()
        and path.as_posix() == value
        and all(part not in {"", ".", ".."} for part in path.parts)
    )


def _identifier_or_false(value: object) -> bool:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    return bool(
        isinstance(value, str)
        and 1 <= len(value) <= 128
        and value[0].isalnum()
        and value[-1].isalnum()
        and all(character in allowed for character in value)
    )


def _sweep_sandbox_arguments_match(
    sandbox_argv: object,
    *,
    expected_effective_arguments: Mapping[str, str],
) -> bool:
    if not isinstance(sandbox_argv, list):
        return False
    tail = sandbox_argv[6:]
    if len(tail) % 2:
        return False
    allowed = set(expected_effective_arguments)
    required = {
        "--source-transaction-manifest-relative",
        "--source-transaction-commit-receipt-relative",
        "--dataset-transaction-manifest-relative",
        "--run-id",
    }
    defaults = {
        "--maximum-window-bytes": str(1 << 30),
        "--warmup-steps": "1",
        "--timed-steps": str(MINIMUM_TIMED_STEPS),
        "--seed": "20260618",
    }
    observed: dict[str, str] = {}
    for index in range(0, len(tail), 2):
        name, value = tail[index : index + 2]
        if (
            name not in allowed
            or name in observed
            or not isinstance(value, str)
            or not value
            or value.startswith("-")
        ):
            return False
        observed[name] = value
    if not required.issubset(observed):
        return False
    effective = {**defaults, **observed}
    if set(effective) != allowed:
        return False
    for name, expected in expected_effective_arguments.items():
        value = effective[name]
        if name in defaults:
            try:
                if int(value) != int(expected):
                    return False
            except ValueError:
                return False
        elif value != expected:
            return False
    return True


def _same_number(observed: object, expected: int | float) -> bool:
    return bool(
        isinstance(observed, int | float)
        and not isinstance(observed, bool)
        and math.isfinite(float(observed))
        and math.isclose(float(observed), float(expected), rel_tol=1e-12, abs_tol=1e-12)
    )


def _validate_snapshot_set(
    snapshots: object, required_paths: set[str]
) -> bool:
    if not isinstance(snapshots, list):
        return False
    paths: list[str] = []
    for snapshot in snapshots:
        if not isinstance(snapshot, dict):
            return False
        state = snapshot.get("state")
        expected_keys = (
            {"path", "state", "entries", "sha256"}
            if state == "absent"
            else {"path", "state", "entries", "apparent_bytes", "sha256"}
        )
        path = snapshot.get("path")
        if (
            set(snapshot) != expected_keys
            or state not in {"absent", "present"}
            or not isinstance(path, str)
            or not PurePosixPath(path).is_absolute()
            or any(part in {".", ".."} for part in PurePosixPath(path).parts)
            or path == "/Volumes"
            or path.startswith("/Volumes/")
            or not isinstance(snapshot.get("entries"), int)
            or isinstance(snapshot.get("entries"), bool)
            or snapshot["entries"] < 0
            or not _digest_or_false(snapshot.get("sha256"))
        ):
            return False
        if state == "absent" and snapshot["entries"] != 0:
            return False
        if state == "present" and (
            snapshot["entries"] < 1
            or not isinstance(snapshot.get("apparent_bytes"), int)
            or isinstance(snapshot.get("apparent_bytes"), bool)
            or snapshot["apparent_bytes"] < 0
        ):
            return False
        paths.append(path)
    return paths == sorted(set(paths)) and required_paths.issubset(paths)


def _validate_resource_receipt(value: object, label: str) -> dict[str, Any]:
    receipt = _exact_keys(value, RESOURCE_RECEIPT_FIELDS, label)
    if (
        any(
            not isinstance(item, int) or isinstance(item, bool) or item < 0
            for item in receipt.values()
        )
        or receipt["maximum_rss_bytes"] > 4 * (1 << 30)
        or receipt["process_swaps"] != 0
        or receipt["system_swap_delta_bytes"] != 0
        or receipt["maximum_system_swap_bytes"] > receipt["system_swap_baseline_bytes"]
        or receipt["sample_count"] < 1
    ):
        raise R2MapPackingSweepError(f"{label} violates the resource gates")
    return receipt


def _validate_receipt_binding(value: object, label: str) -> dict[str, str]:
    receipt = _exact_keys(
        value,
        {"storage_receipt_relative", "storage_receipt_sha256"},
        label,
    )
    if not _receipt_locator_or_false(receipt["storage_receipt_relative"]):
        raise R2MapPackingSweepError(f"{label} locator differs")
    _digest(receipt["storage_receipt_sha256"], f"{label} digest")
    return receipt


def _validate_remote_object_evidence(
    value: object,
    *,
    label: str,
    expected_relative: str,
) -> dict[str, Any]:
    evidence = _exact_keys(
        value,
        {"relative", "object_token", "open_receipt", "range_receipts"},
        label,
    )
    token = _exact_keys(
        evidence["object_token"],
        {
            "schema_version",
            "schema_id",
            "relative",
            "sha256",
            "size",
            "device",
            "inode",
            "mtime_ns",
            "ctime_ns",
            "mode",
            "token_sha256",
        },
        f"{label} object token",
    )
    numeric_names = ("size", "device", "inode", "mtime_ns", "ctime_ns", "mode")
    if (
        evidence["relative"] != expected_relative
        or token["schema_version"] != 1
        or token["schema_id"] != "cascadia.r2-map.remote-object-token.v1"
        or token["relative"] != expected_relative
        or any(
            not isinstance(token[name], int)
            or isinstance(token[name], bool)
            or token[name] < 0
            for name in numeric_names
        )
        or token["size"] <= 0
        or not 0 <= token["mode"] <= 0o777
        or token["token_sha256"] != document_sha256(token, "token_sha256")
    ):
        raise R2MapPackingSweepError(f"{label} object token differs")
    _digest(token["sha256"], f"{label} object SHA-256")
    _validate_receipt_binding(evidence["open_receipt"], f"{label} open receipt")
    ranges = evidence["range_receipts"]
    if not isinstance(ranges, list) or not ranges:
        raise R2MapPackingSweepError(f"{label} range receipts are empty")
    expected_offset = 0
    locators: list[str] = []
    for item in ranges:
        receipt = _exact_keys(
            item,
            {
                "payload_sha256",
                "object_token_sha256",
                "offset",
                "length",
                "storage_receipt_relative",
                "storage_receipt_sha256",
            },
            f"{label} range receipt",
        )
        if (
            receipt["object_token_sha256"] != token["token_sha256"]
            or receipt["offset"] != expected_offset
            or not isinstance(receipt["length"], int)
            or isinstance(receipt["length"], bool)
            or not 1 <= receipt["length"] <= 64 << 20
            or not _receipt_locator_or_false(receipt["storage_receipt_relative"])
        ):
            raise R2MapPackingSweepError(f"{label} range sequence differs")
        _digest(receipt["payload_sha256"], f"{label} range payload")
        _digest(receipt["storage_receipt_sha256"], f"{label} range receipt")
        expected_offset += receipt["length"]
        locators.append(receipt["storage_receipt_relative"])
    if expected_offset != token["size"] or len(locators) != len(set(locators)):
        raise R2MapPackingSweepError(f"{label} range coverage differs")
    return evidence


def _validate_remote_json_identity(
    value: object,
    *,
    label: str,
    expected_relative: str,
) -> dict[str, Any]:
    identity = _exact_keys(
        value,
        {"payload_sha256", "payload_blake3", "evidence"},
        label,
    )
    _digest(identity["payload_sha256"], f"{label} payload SHA-256")
    _digest(identity["payload_blake3"], f"{label} payload BLAKE3")
    evidence = _validate_remote_object_evidence(
        identity["evidence"],
        label=f"{label} evidence",
        expected_relative=expected_relative,
    )
    if evidence["object_token"]["sha256"] != identity["payload_sha256"]:
        raise R2MapPackingSweepError(f"{label} payload/object token differs")
    return identity


def _validate_window_evidence(value: object) -> dict[str, Any]:
    evidence = _exact_keys(
        value,
        {
            "schema_version",
            "schema_id",
            "run_id",
            "source",
            "mode",
            "epoch",
            "sampler_seed",
            "run_receipt",
            "manifest",
            "dataset",
            "cleanup_prepare_receipt",
            "cleanup_commit_receipt",
            "evidence_sha256",
        },
        "remote window evidence",
    )
    run_id = evidence["run_id"]
    source = evidence["source"]
    if (
        evidence["schema_version"] != 1
        or evidence["schema_id"] != "cascadia.r2-map.window-read-evidence.v1"
        or not isinstance(run_id, str)
        or not run_id.startswith("r2win-")
        or len(run_id) != len("r2win-") + 32
        or any(character not in "0123456789abcdef" for character in run_id[6:])
        or not isinstance(source, str)
        or PurePosixPath(source).name != source
        or source in {"", ".", ".."}
        or evidence["mode"] not in {"train", "validation"}
        or any(
            not isinstance(evidence[name], int)
            or isinstance(evidence[name], bool)
            or evidence[name] < 0
            for name in ("epoch", "sampler_seed")
        )
        or evidence["evidence_sha256"] != document_sha256(evidence, "evidence_sha256")
    ):
        raise R2MapPackingSweepError("remote window identity differs")
    manifest = _validate_remote_object_evidence(
        evidence["manifest"],
        label="window manifest",
        expected_relative=f"build/run-{run_id}/window.json",
    )
    dataset = _validate_remote_object_evidence(
        evidence["dataset"],
        label="window dataset",
        expected_relative=f"build/run-{run_id}/window.r2map",
    )
    run_receipt = _exact_keys(
        evidence["run_receipt"],
        {
            "run_id",
            "cwd_relative",
            "argv_sha256",
            "output_relative",
            "exit_code",
            "timed_out",
            "duration_ms",
            "run_bytes",
            "max_run_bytes",
            "campaign_bytes_before",
            "campaign_bytes_after",
            "campaign_bytes_delta",
            "free_bytes_after",
            "temporary_cleaned",
            "controller_mode",
            "stdout_sha256",
            "stdout_size",
            "stderr_sha256",
            "stderr_size",
            "run_receipt_sha256",
            "storage_receipt_relative",
            "storage_receipt_sha256",
        },
        "window run receipt",
    )
    unsigned_numeric = (
        "duration_ms",
        "run_bytes",
        "max_run_bytes",
        "campaign_bytes_before",
        "campaign_bytes_after",
        "free_bytes_after",
        "stdout_size",
        "stderr_size",
    )
    if (
        run_receipt["run_id"] != run_id
        or not _safe_relative_or_false(run_receipt["cwd_relative"])
        or run_receipt["output_relative"] != f"logs/window-exports/{run_id}"
        or run_receipt["exit_code"] != 0
        or run_receipt["timed_out"] is not False
        or run_receipt["temporary_cleaned"] is not True
        or run_receipt["controller_mode"] is not False
        or any(
            not isinstance(run_receipt[name], int)
            or isinstance(run_receipt[name], bool)
            or run_receipt[name] < 0
            for name in unsigned_numeric
        )
        or not isinstance(run_receipt["campaign_bytes_delta"], int)
        or isinstance(run_receipt["campaign_bytes_delta"], bool)
        or run_receipt["campaign_bytes_delta"]
        != run_receipt["campaign_bytes_after"] - run_receipt["campaign_bytes_before"]
        or not _receipt_locator_or_false(run_receipt["storage_receipt_relative"])
        or run_receipt["run_receipt_sha256"] != run_receipt["storage_receipt_sha256"]
    ):
        raise R2MapPackingSweepError("window run receipt differs")
    for name in (
        "argv_sha256",
        "stdout_sha256",
        "stderr_sha256",
        "run_receipt_sha256",
        "storage_receipt_sha256",
    ):
        _digest(run_receipt[name], f"window run receipt {name}")
    _validate_receipt_binding(
        evidence["cleanup_prepare_receipt"], "window cleanup prepare receipt"
    )
    cleanup = _exact_keys(
        evidence["cleanup_commit_receipt"],
        {
            "run_id",
            "cleanup_token_sha256",
            "manifest_object_token_sha256",
            "dataset_object_token_sha256",
            "removed_bytes",
            "build_already_removed",
            "cache_already_removed",
            "build_removed",
            "cache_removed",
            "storage_receipt_relative",
            "storage_receipt_sha256",
        },
        "window cleanup commit receipt",
    )
    if (
        cleanup["run_id"] != run_id
        or cleanup["manifest_object_token_sha256"]
        != manifest["object_token"]["token_sha256"]
        or cleanup["dataset_object_token_sha256"]
        != dataset["object_token"]["token_sha256"]
        or not isinstance(cleanup["removed_bytes"], int)
        or isinstance(cleanup["removed_bytes"], bool)
        or cleanup["removed_bytes"]
        < manifest["object_token"]["size"] + dataset["object_token"]["size"]
        or cleanup["build_already_removed"] is not False
        or cleanup["cache_already_removed"] is not False
        or cleanup["build_removed"] is not True
        or cleanup["cache_removed"] is not True
        or not _receipt_locator_or_false(cleanup["storage_receipt_relative"])
    ):
        raise R2MapPackingSweepError("window cleanup commit differs")
    _digest(cleanup["cleanup_token_sha256"], "window cleanup token")
    _digest(cleanup["storage_receipt_sha256"], "window cleanup receipt")
    return evidence


def _validate_measurement_common(
    measurement: object,
    *,
    label: str,
    protocol: str,
    warmup_steps: int,
    timed_steps: int,
) -> dict[str, Any]:
    value = _exact_keys(
        measurement,
        {
            "label",
            "measurement_protocol",
            "warmup_steps",
            "warmup_synchronized",
            "timed_steps",
            "elapsed_ns",
            "step_durations_ns",
            "p50_step_duration_ns",
            "steps_per_second",
            "draft_groups_per_second",
            "draft_candidates_per_second",
            "training_counters",
            "resource_receipt",
            "mlx_memory",
            "expected_group_count_per_step",
            "observed_group_count_per_step",
            "decode_and_padding_inside_timed_step",
            "mlx_allocation_inside_timed_step",
            "remote_window_acquisition_inside_timed_interval",
            "remote_windows_acquired",
            "remote_window_durations_ns",
            "remote_window_duration_ns_per_step",
            "candidate_widths",
            "frame_indices",
        },
        label,
    )
    if (
        value["label"] != label
        or value["measurement_protocol"] != protocol
        or value["warmup_steps"] != warmup_steps
        or value["warmup_synchronized"] is not True
        or value["timed_steps"] != timed_steps
        or not isinstance(value["step_durations_ns"], list)
        or len(value["step_durations_ns"]) != timed_steps
        or value["elapsed_ns"] != sum(value["step_durations_ns"])
        or value["decode_and_padding_inside_timed_step"] is not True
        or value["mlx_allocation_inside_timed_step"] is not True
        or not isinstance(value["remote_window_durations_ns"], list)
        or not isinstance(value["remote_window_duration_ns_per_step"], list)
        or len(value["remote_window_duration_ns_per_step"]) != timed_steps
        or value["remote_windows_acquired"] != len(value["remote_window_durations_ns"])
        or sum(value["remote_window_durations_ns"])
        != sum(value["remote_window_duration_ns_per_step"])
    ):
        raise R2MapPackingSweepError(f"{label} timing contract differs")
    if any(
        not isinstance(item, int) or isinstance(item, bool) or item <= 0
        for item in value["step_durations_ns"]
    ) or any(
        not isinstance(item, int) or isinstance(item, bool) or item <= 0
        for item in value["remote_window_durations_ns"]
    ):
        raise R2MapPackingSweepError(f"{label} duration samples differ")
    if any(
        not isinstance(item, int) or isinstance(item, bool) or item < 0
        for item in value["remote_window_duration_ns_per_step"]
    ):
        raise R2MapPackingSweepError(f"{label} remote window step timing differs")
    if not _same_number(
        value["p50_step_duration_ns"], statistics.median(value["step_durations_ns"])
    ):
        raise R2MapPackingSweepError(f"{label} median duration differs")
    expected_groups = value["expected_group_count_per_step"]
    observed_groups = value["observed_group_count_per_step"]
    if (
        not isinstance(expected_groups, list)
        or not isinstance(observed_groups, list)
        or len(expected_groups) != timed_steps
        or expected_groups != observed_groups
        or any(
            not isinstance(item, int) or isinstance(item, bool) or item <= 0
            for item in expected_groups
        )
    ):
        raise R2MapPackingSweepError(f"{label} group cardinality differs")
    counters = value["training_counters"]
    if (
        not isinstance(counters, dict)
        or set(counters) != TRAINING_COUNTER_NAMES
        or any(
            not isinstance(item, int) or isinstance(item, bool) or item < 0
            for item in counters.values()
        )
        or counters.get("draft_groups") != sum(observed_groups)
        or counters.get("padded_draft_candidates", 0) < counters.get("draft_candidates", 0)
        or counters.get("draft_policy_targets", 0) > counters.get("draft_groups", 0)
        or counters.get("market_policy_targets", 0) > counters.get("market_groups", 0)
    ):
        raise R2MapPackingSweepError(f"{label} training counters differ")
    expected_rates = {
        "steps_per_second": timed_steps * 1_000_000_000 / value["elapsed_ns"],
        "draft_groups_per_second": counters["draft_groups"]
        * 1_000_000_000
        / value["elapsed_ns"],
        "draft_candidates_per_second": counters["draft_candidates"]
        * 1_000_000_000
        / value["elapsed_ns"],
    }
    if any(not _same_number(value[name], expected) for name, expected in expected_rates.items()):
        raise R2MapPackingSweepError(f"{label} throughput accounting differs")
    _validate_resource_receipt(value["resource_receipt"], label)
    memory = value["mlx_memory"]
    if (
        not isinstance(memory, dict)
        or set(memory) != {"active_bytes", "cache_bytes", "peak_active_bytes"}
        or any(
            not isinstance(item, int) or isinstance(item, bool) or item < 0
            for item in memory.values()
        )
        or memory.get("cache_bytes", MAXIMUM_CANDIDATES_PER_BATCH) > 1 << 30
    ):
        raise R2MapPackingSweepError(f"{label} resource gate failed")
    for sequence, sequence_label, allow_empty in (
        (value["candidate_widths"], "candidate widths", True),
        (value["frame_indices"], "frame indices", True),
    ):
        if not isinstance(sequence, list) or any(
            not isinstance(item, int)
            or isinstance(item, bool)
            or item < (1 if sequence_label == "candidate widths" else 0)
            for item in sequence
        ):
            raise R2MapPackingSweepError(f"{label} {sequence_label} differ")
        if not allow_empty and not sequence:
            raise R2MapPackingSweepError(f"{label} {sequence_label} are empty")
    return value


def validate_qualifying_packing_report(value: object) -> dict[str, Any]:
    report = _exact_keys(
        value,
        {
            "schema_version",
            "schema_id",
            "qualification_status",
            "run_id",
            "source_identity",
            "dataset_identity",
            "packing_contract",
            "registered_maximum_width",
            "width_census",
            "packing_plans",
            "representative_measurements",
            "production_path_measurements",
            "source_window_timings",
            "wall_projections",
            "selection",
            "sweep_resource_receipt",
            "window_evidence_publications",
            "ssh_transport",
            "storage_preflight_receipt",
            "local_write_guard",
            "report_sha256",
        },
        "qualifying packing report",
    )
    if (
        report["schema_version"] != 3
        or report["schema_id"] != QUALIFYING_SWEEP_SCHEMA
        or report["qualification_status"] != "qualifying-exact-bootstrap"
        or report["report_sha256"] != report_sha256(report)
        or not _identifier_or_false(report["run_id"])
    ):
        raise R2MapPackingSweepError("qualifying report identity differs")
    source = _exact_keys(
        report["source_identity"],
        {
            "source_blake3",
            "source_manifest",
            "reference_manifest",
            "source_archive",
            "source_archive_verification",
            "source_archive_verification_descriptor",
            "source_archive_verifier",
            "source_gate_aliases",
            "transaction_manifest",
            "transaction_manifest_sha256",
            "transaction_commit_receipt",
            "transaction_commit_receipt_sha256",
            "maximum_width_panel_sha256",
            "maximum_width_candidates",
        },
        "source identity",
    )
    for name in (
        "source_blake3",
        "transaction_manifest_sha256",
        "transaction_commit_receipt_sha256",
        "maximum_width_panel_sha256",
    ):
        _digest(source[name], f"source identity {name}")
    if source["maximum_width_candidates"] != MAXIMUM_WIDTH_CANDIDATES or any(
        not isinstance(source[name], dict)
        for name in (
            "source_manifest",
            "reference_manifest",
            "source_archive_verification",
            "transaction_manifest",
            "transaction_commit_receipt",
        )
    ):
        raise R2MapPackingSweepError("source identity evidence differs")
    source_descriptor_fields = {"relative", "sha256", "size", "mode"}
    for label, descriptor, mode in (
        ("archive", source["source_archive"], "0400"),
        (
            "archive verification",
            source["source_archive_verification_descriptor"],
            "0400",
        ),
        ("archive verifier", source["source_archive_verifier"], "0500"),
    ):
        value = _exact_keys(
            descriptor,
            source_descriptor_fields,
            f"source {label} descriptor",
        )
        if (
            not _safe_relative_or_false(value["relative"])
            or not _digest_or_false(value["sha256"])
            or not isinstance(value["size"], int)
            or isinstance(value["size"], bool)
            or value["size"] < 0
            or value["mode"] != mode
        ):
            raise R2MapPackingSweepError(f"source {label} descriptor differs")
    gate_aliases = _exact_keys(
        source["source_gate_aliases"],
        {"target.mk", "p1.mk", "release.mk", "python.mk", "compile.mk", "fixture.mk"},
        "source gate aliases",
    )
    for alias, descriptor in gate_aliases.items():
        value = _exact_keys(
            descriptor,
            source_descriptor_fields,
            f"source gate alias {alias}",
        )
        if (
            not _safe_relative_or_false(value["relative"])
            or not _digest_or_false(value["sha256"])
            or not isinstance(value["size"], int)
            or isinstance(value["size"], bool)
            or value["size"] < 0
            or value["mode"] != "0400"
        ):
            raise R2MapPackingSweepError(f"source gate alias differs: {alias}")
    contract = _exact_keys(
        report["packing_contract"],
        {
            "group_batch_sizes",
            "maximum_candidates_per_batch",
            "maximum_window_bytes",
            "games",
            "epochs",
            "warmup_steps",
            "timed_steps",
            "seed",
            "production_measurement_protocol",
            "representative_measurement_protocol",
            "coverage",
        },
        "packing contract",
    )
    timed_steps = contract["timed_steps"]
    if (
        contract["group_batch_sizes"] != list(QUALIFYING_CAPS)
        or contract["maximum_candidates_per_batch"] != MAXIMUM_CANDIDATES_PER_BATCH
        or contract["games"] != QUALIFYING_GAMES
        or contract["epochs"] != QUALIFYING_EPOCHS
        or not isinstance(contract["maximum_window_bytes"], int)
        or isinstance(contract["maximum_window_bytes"], bool)
        or not 1 <= contract["maximum_window_bytes"] <= 1 << 30
        or not isinstance(contract["warmup_steps"], int)
        or isinstance(contract["warmup_steps"], bool)
        or contract["warmup_steps"] < 1
        or not isinstance(timed_steps, int)
        or isinstance(timed_steps, bool)
        or timed_steps < MINIMUM_TIMED_STEPS
        or not isinstance(contract["seed"], int)
        or isinstance(contract["seed"], bool)
        or contract["seed"] < 0
        or contract["production_measurement_protocol"] != PRODUCTION_MEASUREMENT_PROTOCOL
        or contract["representative_measurement_protocol"] != REPRESENTATIVE_MEASUREMENT_PROTOCOL
        or contract["coverage"]
        != ["selected-only", "imitation-p50", "imitation-maximum", "registered-maximum-width"]
    ):
        raise R2MapPackingSweepError("qualifying packing contract differs")
    dataset = _exact_keys(
        report["dataset_identity"],
        {
            "dataset_blake3",
            "game_count",
            "collection_kind",
            "shard_root_relative",
            "exporter_relative",
            "compact_index",
            "transaction_manifest",
            "transaction_manifest_sha256",
            "transaction_commit_receipt",
            "transaction_commit_receipt_sha256",
            "bootstrap_phase_barrier",
        },
        "dataset identity",
    )
    if dataset["game_count"] != QUALIFYING_GAMES or dataset["collection_kind"] != "bootstrap":
        raise R2MapPackingSweepError("report is not bound to the exact bootstrap corpus")
    for name in (
        "dataset_blake3",
        "transaction_manifest_sha256",
        "transaction_commit_receipt_sha256",
    ):
        _digest(dataset[name], f"dataset identity {name}")
    if any(
        not isinstance(dataset[name], dict)
        for name in (
            "compact_index",
            "transaction_manifest",
            "transaction_commit_receipt",
        )
    ):
        raise R2MapPackingSweepError("dataset identity evidence differs")
    phase_barrier = _exact_keys(
        dataset["bootstrap_phase_barrier"],
        {
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
            "barrier_document",
            "publication_receipt",
            "publication_receipt_sha256",
            "generation_manifest_document",
            "generation_manifest_publication_receipt",
        },
        "bootstrap phase barrier identity",
    )
    for name in (
        "identity_sha256",
        "barrier_sha256",
        "controller_state_sha256",
        "publication_receipt_sha256",
        "generation_manifest_payload_sha256",
        "generation_manifest_identity_sha256",
        "generation_manifest_publication_receipt_sha256",
    ):
        _digest(phase_barrier[name], f"bootstrap phase barrier {name}")
    if (
        phase_barrier["phase_receipt_count"] != 4
        or not all(
            _safe_relative_or_false(phase_barrier[name])
            for name in (
                "barrier_relative",
                "generation_manifest_relative",
                "generation_manifest_publication_receipt_relative",
                "dataset_target_relative",
                "dataset_transaction_manifest_relative",
                "dataset_transaction_commit_receipt_relative",
                "compact_index_relative",
                "shard_root_relative",
            )
        )
        or phase_barrier["barrier_relative"]
        != f"{phase_barrier['dataset_target_relative']}.bootstrap-phase-barrier.json"
        or phase_barrier["generation_manifest_relative"]
        != f"{phase_barrier['dataset_target_relative']}.generation-manifest.json"
        or phase_barrier["dataset_transaction_manifest_relative"]
        != f"{phase_barrier['dataset_target_relative']}/.r2-map-transaction.json"
        or phase_barrier["shard_root_relative"] != dataset["shard_root_relative"]
        or not _receipt_locator_or_false(
            phase_barrier["generation_manifest_publication_receipt_relative"]
        )
        or not _receipt_locator_or_false(
            phase_barrier["dataset_transaction_commit_receipt_relative"]
        )
    ):
        raise R2MapPackingSweepError("bootstrap phase barrier binding differs")
    expected_barrier_publication_relative = (
        "control/receipts/req-bootstrap-barrier-"
        f"{phase_barrier['identity_sha256'][:32]}.json"
    )
    barrier_document_identity = _validate_remote_json_identity(
        phase_barrier["barrier_document"],
        label="bootstrap phase barrier document",
        expected_relative=phase_barrier["barrier_relative"],
    )
    barrier_publication_identity = _validate_remote_json_identity(
        phase_barrier["publication_receipt"],
        label="bootstrap phase barrier publication receipt",
        expected_relative=expected_barrier_publication_relative,
    )
    generation_document_identity = _validate_remote_json_identity(
        phase_barrier["generation_manifest_document"],
        label="bootstrap generation manifest document",
        expected_relative=phase_barrier["generation_manifest_relative"],
    )
    generation_publication_identity = _validate_remote_json_identity(
        phase_barrier["generation_manifest_publication_receipt"],
        label="bootstrap generation manifest publication receipt",
        expected_relative=phase_barrier[
            "generation_manifest_publication_receipt_relative"
        ],
    )
    if (
        generation_document_identity["payload_sha256"]
        != phase_barrier["generation_manifest_payload_sha256"]
        or any(
            identity["evidence"]["object_token"]["mode"] != 0o400
            for identity in (
                barrier_document_identity,
                barrier_publication_identity,
                generation_document_identity,
                generation_publication_identity,
            )
        )
    ):
        raise R2MapPackingSweepError("bootstrap phase immutable object mode/identity differs")
    for name in ("shard_root_relative", "exporter_relative"):
        relative = dataset[name]
        if (
            not isinstance(relative, str)
            or not relative
            or relative.startswith("/")
            or ".." in relative.split("/")
        ):
            raise R2MapPackingSweepError("report contains an unsafe remote source path")
    maximum = _exact_keys(
        report["registered_maximum_width"],
        {"candidate_count", "panel_sha256", "synthetic_resource_gate_only", "measurement"},
        "registered maximum width",
    )
    if (
        maximum["candidate_count"] != MAXIMUM_WIDTH_CANDIDATES
        or maximum["synthetic_resource_gate_only"] is not True
        or source["maximum_width_panel_sha256"] != maximum["panel_sha256"]
    ):
        raise R2MapPackingSweepError("registered maximum-width binding differs")
    plans = report["packing_plans"]
    if not isinstance(plans, list) or [plan.get("group_batch_size") for plan in plans] != list(
        QUALIFYING_CAPS
    ):
        raise R2MapPackingSweepError("packing plan cap set differs")
    plan_steps: dict[int, list[int]] = {}
    for plan in plans:
        if set(plan) != {
            "schema_version",
            "schema_id",
            "dataset_blake3",
            "seed",
            "epochs",
            "group_batch_size",
            "maximum_candidates_per_batch",
            "epoch_plans",
            "totals",
            "maximum_candidate_width",
            "maximum_batch_groups",
            "minimum_batch_groups",
            "plan_blake3",
        }:
            raise R2MapPackingSweepError("packing plan schema differs")
        identity = dict(plan)
        claimed_plan_blake3 = identity.pop("plan_blake3")
        observed_plan_blake3 = blake3.blake3(
            json.dumps(
                identity,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode()
        ).hexdigest()
        epochs = plan.get("epoch_plans")
        if (
            claimed_plan_blake3 != observed_plan_blake3
            or plan["schema_version"] != 1
            or plan["schema_id"] != "r2-map-compact-packing-plan-v1"
            or plan["dataset_blake3"] != dataset["dataset_blake3"]
            or plan["seed"] != contract["seed"]
            or plan["epochs"] != QUALIFYING_EPOCHS
            or plan["maximum_candidates_per_batch"] != MAXIMUM_CANDIDATES_PER_BATCH
            or not isinstance(epochs, list)
            or len(epochs) != QUALIFYING_EPOCHS
            or [epoch.get("epoch") for epoch in epochs] != list(range(QUALIFYING_EPOCHS))
            or any(
                not isinstance(epoch.get("steps"), int) or epoch["steps"] <= 0 for epoch in epochs
            )
            or plan.get("totals", {}).get("steps") != sum(epoch["steps"] for epoch in epochs)
        ):
            raise R2MapPackingSweepError("packing plan does not contain exact 12-epoch steps")
        epoch_fields = {
            "epoch",
            "steps",
            "draft_groups",
            "selected_only_groups",
            "draft_policy_targets",
            "draft_candidates",
            "padded_draft_candidates",
            "maximum_batch_groups",
            "minimum_batch_groups",
        }
        if any(
            set(epoch) != epoch_fields
            or any(
                not isinstance(item, int) or isinstance(item, bool) or item < 0
                for item in epoch.values()
            )
            or epoch["draft_groups"]
            != epoch["selected_only_groups"] + epoch["draft_policy_targets"]
            or epoch["padded_draft_candidates"] < epoch["draft_candidates"]
            or not 1
            <= epoch["minimum_batch_groups"]
            <= epoch["maximum_batch_groups"]
            <= plan["group_batch_size"]
            for epoch in epochs
        ):
            raise R2MapPackingSweepError("packing epoch accounting differs")
        total_fields = {
            "steps",
            "draft_groups",
            "selected_only_groups",
            "draft_policy_targets",
            "draft_candidates",
            "padded_draft_candidates",
        }
        if set(plan["totals"]) != total_fields or any(
            plan["totals"][name] != sum(epoch[name] for epoch in epochs) for name in total_fields
        ):
            raise R2MapPackingSweepError("packing plan totals differ")
        if (
            not isinstance(plan["maximum_candidate_width"], int)
            or not 1 <= plan["maximum_candidate_width"] <= MAXIMUM_CANDIDATES_PER_BATCH
            or plan["maximum_batch_groups"]
            != max(epoch["maximum_batch_groups"] for epoch in epochs)
            or plan["minimum_batch_groups"]
            != min(epoch["minimum_batch_groups"] for epoch in epochs)
        ):
            raise R2MapPackingSweepError("packing plan extrema differ")
        plan_steps[plan["group_batch_size"]] = [epoch["steps"] for epoch in epochs]
    representatives = report["representative_measurements"]
    expected_labels = {
        f"g{cap}-{suffix}"
        for cap in QUALIFYING_CAPS
        for suffix in ("selected", "imitation-p50", "imitation-max")
    }
    if (
        not isinstance(representatives, list)
        or len(representatives) != len(expected_labels)
        or {item.get("label") for item in representatives} != expected_labels
    ):
        raise R2MapPackingSweepError("representative measurement coverage differs")
    for measurement in representatives:
        checked = _validate_measurement_common(
            measurement,
            label=measurement["label"],
            protocol=REPRESENTATIVE_MEASUREMENT_PROTOCOL,
            warmup_steps=contract["warmup_steps"],
            timed_steps=timed_steps,
        )
        if (
            checked["remote_window_acquisition_inside_timed_interval"] is not False
            or checked["remote_windows_acquired"] != 0
            or checked["remote_window_durations_ns"]
            or any(checked["remote_window_duration_ns_per_step"])
        ):
            raise R2MapPackingSweepError("representative microbench claims remote window work")
    production = report["production_path_measurements"]
    if not isinstance(production, list) or [item.get("label") for item in production] != [
        f"g{cap}-production" for cap in QUALIFYING_CAPS
    ]:
        raise R2MapPackingSweepError("production measurement cap set differs")
    for measurement in production:
        checked = _validate_measurement_common(
            measurement,
            label=measurement["label"],
            protocol=PRODUCTION_MEASUREMENT_PROTOCOL,
            warmup_steps=contract["warmup_steps"],
            timed_steps=timed_steps,
        )
        if (
            checked["remote_window_acquisition_inside_timed_interval"] is not True
            or checked["remote_windows_acquired"] < 1
            or checked["candidate_widths"]
            or checked["frame_indices"]
        ):
            raise R2MapPackingSweepError("production measurement omits remote window acquisition")
    synthetic = _validate_measurement_common(
        maximum["measurement"],
        label="synthetic-maximum-width",
        protocol=REPRESENTATIVE_MEASUREMENT_PROTOCOL,
        warmup_steps=contract["warmup_steps"],
        timed_steps=timed_steps,
    )
    if (
        synthetic["candidate_widths"] != [MAXIMUM_WIDTH_CANDIDATES]
        or synthetic["frame_indices"]
        or synthetic["remote_window_acquisition_inside_timed_interval"] is not False
        or synthetic["remote_windows_acquired"] != 0
        or synthetic["expected_group_count_per_step"] != [1] * timed_steps
        or synthetic["training_counters"]["draft_candidates"]
        != MAXIMUM_WIDTH_CANDIDATES * timed_steps
        or synthetic["training_counters"]["padded_draft_candidates"]
        != MAXIMUM_WIDTH_CANDIDATES * timed_steps
        or synthetic["training_counters"]["draft_policy_targets"] != timed_steps
    ):
        raise R2MapPackingSweepError("maximum-width resource shape differs")
    projections = report["wall_projections"]
    if not isinstance(projections, list) or [
        item.get("group_batch_size") for item in projections
    ] != list(QUALIFYING_CAPS):
        raise R2MapPackingSweepError("wall projection cap set differs")
    window_evidence = report["window_evidence_publications"]
    if not isinstance(window_evidence, list) or not window_evidence:
        raise R2MapPackingSweepError("qualifying sweep has invalid remote window evidence")
    for item in window_evidence:
        _validate_window_evidence(item)
    if len({item["run_id"] for item in window_evidence}) != len(window_evidence):
        raise R2MapPackingSweepError("qualifying sweep repeats remote window evidence")
    train_sources = {
        item["source"] for item in window_evidence if item["mode"] == "train" and item["epoch"] == 0
    }
    if not train_sources:
        raise R2MapPackingSweepError("qualifying sweep has no epoch-zero train windows")
    source_timings = report["source_window_timings"]
    window_by_run_id = {item["run_id"]: item for item in window_evidence}
    if (
        not isinstance(source_timings, list)
        or not source_timings
        or any(
            not isinstance(item, dict)
            or set(item)
            != {
                "source",
                "duration_ns",
                "window_run_id",
                "window_evidence_sha256",
            }
            or not isinstance(item["source"], str)
            or PurePosixPath(item["source"]).name != item["source"]
            or not isinstance(item["duration_ns"], int)
            or isinstance(item["duration_ns"], bool)
            or item["duration_ns"] <= 0
            or not isinstance(item["window_run_id"], str)
            or item["window_run_id"] not in window_by_run_id
            or not _digest_or_false(item["window_evidence_sha256"])
            or item["window_evidence_sha256"]
            != window_by_run_id[item["window_run_id"]]["evidence_sha256"]
            or item["source"] != window_by_run_id[item["window_run_id"]]["source"]
            or window_by_run_id[item["window_run_id"]]["mode"] != "train"
            or window_by_run_id[item["window_run_id"]]["epoch"] != 0
            for item in source_timings
        )
        or [item["source"] for item in source_timings]
        != sorted({item["source"] for item in source_timings})
        or {item["source"] for item in source_timings} != train_sources
    ):
        raise R2MapPackingSweepError("all-source window timing coverage differs")
    remote_duration_seconds = [item["duration_ns"] / 1_000_000_000 for item in source_timings]
    expected_remote_rates = {
        "optimistic": min(remote_duration_seconds),
        "central": statistics.median(remote_duration_seconds),
        "conservative": max(remote_duration_seconds),
    }
    production_by_label = {measurement["label"]: measurement for measurement in production}
    representative_by_label = {
        measurement["label"]: measurement for measurement in representatives
    }
    for projection in projections:
        cap = projection["group_batch_size"]
        if set(projection) != {
            "group_batch_size",
            "method",
            "steps_per_epoch",
            "remote_windows_per_epoch",
            "central_seconds_per_epoch",
            "central_12_epoch_wall_seconds",
            "optimistic_12_epoch_wall_seconds",
            "conservative_12_epoch_wall_seconds",
            "compute_seconds_per_step",
            "remote_seconds_per_window",
            "includes_remote_window_acquisition",
        }:
            raise R2MapPackingSweepError("wall projection schema differs")
        if projection.get("steps_per_epoch") != plan_steps[cap]:
            raise R2MapPackingSweepError("wall projection schedule differs from packing plan")
        central_epochs = projection["central_seconds_per_epoch"]
        compute_rates = projection["compute_seconds_per_step"]
        remote_rates = projection["remote_seconds_per_window"]
        production_measurement = production_by_label[f"g{cap}-production"]
        production_compute_seconds = [
            (duration - remote_duration) / 1_000_000_000
            for duration, remote_duration in zip(
                production_measurement["step_durations_ns"],
                production_measurement["remote_window_duration_ns_per_step"],
                strict=True,
            )
        ]
        if any(duration <= 0 for duration in production_compute_seconds):
            raise R2MapPackingSweepError("production compute duration is non-positive")
        representative_compute_seconds = [
            duration / 1_000_000_000
            for suffix in ("selected", "imitation-p50", "imitation-max")
            for duration in representative_by_label[f"g{cap}-{suffix}"]["step_durations_ns"]
        ]
        expected_compute_rates = {
            "optimistic": min(production_compute_seconds),
            "central": statistics.median(production_compute_seconds),
            "conservative": max(
                *production_compute_seconds,
                *representative_compute_seconds,
            ),
        }
        if (
            projection["method"]
            != "exact-plan-compute-plus-all-source-remote-window-rate-v4"
            or projection["includes_remote_window_acquisition"] is not True
            or not isinstance(projection["remote_windows_per_epoch"], int)
            or isinstance(projection["remote_windows_per_epoch"], bool)
            or projection["remote_windows_per_epoch"] != len(source_timings)
            or not isinstance(central_epochs, list)
            or len(central_epochs) != QUALIFYING_EPOCHS
            or not isinstance(compute_rates, dict)
            or set(compute_rates) != {"optimistic", "central", "conservative"}
            or not isinstance(remote_rates, dict)
            or set(remote_rates) != {"optimistic", "central", "conservative"}
        ):
            raise R2MapPackingSweepError("wall projection contract differs")
        for item in central_epochs:
            _positive_finite(item, "central epoch wall")
        for name, rate in compute_rates.items():
            _positive_finite(rate, f"{name} production step rate")
            if not _same_number(rate, expected_compute_rates[name]):
                raise R2MapPackingSweepError("wall projection compute rate differs")
        for name, rate in remote_rates.items():
            _positive_finite(rate, f"{name} remote window rate")
            if not _same_number(rate, expected_remote_rates[name]):
                raise R2MapPackingSweepError("wall projection remote rate differs")
        expected_epochs = {
            name: [
                steps * expected_compute_rates[name]
                + len(source_timings) * expected_remote_rates[name]
                for steps in plan_steps[cap]
            ]
            for name in ("optimistic", "central", "conservative")
        }
        if any(
            not _same_number(observed, expected)
            for observed, expected in zip(
                central_epochs, expected_epochs["central"], strict=True
            )
        ):
            raise R2MapPackingSweepError("central epoch wall projection differs")
        expected_totals = {
            name: sum(values) for name, values in expected_epochs.items()
        }
        for name, field in (
            ("central", "central_12_epoch_wall_seconds"),
            ("optimistic", "optimistic_12_epoch_wall_seconds"),
            ("conservative", "conservative_12_epoch_wall_seconds"),
        ):
            if not _same_number(projection[field], expected_totals[name]):
                raise R2MapPackingSweepError("12-epoch wall projection total differs")
        for field in (
            "central_12_epoch_wall_seconds",
            "optimistic_12_epoch_wall_seconds",
            "conservative_12_epoch_wall_seconds",
        ):
            _positive_finite(projection.get(field), f"cap {cap} {field}")
    selection = report["selection"]
    if (
        not isinstance(selection, dict)
        or set(selection)
        != {
            "selector",
            "selected_group_batch_size",
            "selected_schedule_steps",
            "selected_epochs",
            "selected_conservative_12_epoch_wall_seconds",
            "candidates",
            "rationale",
        }
        or selection.get("selector") != SELECTOR_ID
        or selection.get("selected_group_batch_size") not in QUALIFYING_CAPS
        or selection.get("selected_schedule_steps")
        != sum(plan_steps[selection["selected_group_batch_size"]])
        or selection.get("selected_epochs") != QUALIFYING_EPOCHS
        or not _same_number(
            selection.get("selected_conservative_12_epoch_wall_seconds"),
            next(
                projection["conservative_12_epoch_wall_seconds"]
                for projection in projections
                if projection["group_batch_size"]
                == selection.get("selected_group_batch_size")
            ),
        )
        or not isinstance(selection.get("candidates"), list)
        or [item.get("group_batch_size") for item in selection["candidates"]]
        != list(QUALIFYING_CAPS)
        or not isinstance(selection.get("rationale"), str)
        or not selection["rationale"]
    ):
        raise R2MapPackingSweepError("packing selector result differs")
    projection_by_cap = {projection["group_batch_size"]: projection for projection in projections}
    plan_by_cap = {plan["group_batch_size"]: plan for plan in plans}
    measurement_by_label = {
        measurement["label"]: measurement for measurement in [*representatives, *production]
    }
    for candidate in selection["candidates"]:
        cap = candidate.get("group_batch_size") if isinstance(candidate, dict) else None
        if cap not in QUALIFYING_CAPS:
            raise R2MapPackingSweepError("packing selector candidate cap differs")
        shape_measurements = [
            measurement_by_label[f"g{cap}-{suffix}"]
            for suffix in ("selected", "imitation-p50", "imitation-max", "production")
        ] + [synthetic]
        expected_resource_pass = all(
            measurement["resource_receipt"]["maximum_rss_bytes"] <= 4 * (1 << 30)
            and measurement["resource_receipt"]["process_swaps"] == 0
            and measurement["resource_receipt"]["system_swap_delta_bytes"] == 0
            and measurement["mlx_memory"]["cache_bytes"] <= 1 << 30
            for measurement in shape_measurements
        )
        expected_candidate_budget_pass = (
            plan_by_cap[cap]["maximum_candidate_width"]
            <= contract["maximum_candidates_per_batch"]
            and all(
                measurement["training_counters"]["padded_draft_candidates"]
                <= contract["maximum_candidates_per_batch"] * measurement["timed_steps"]
                for measurement in shape_measurements
            )
        )
        if (
            not isinstance(candidate, dict)
            or set(candidate)
            != {
                "group_batch_size",
                "resource_pass",
                "candidate_budget_pass",
                "conservative_12_epoch_wall_seconds",
            }
            or not isinstance(candidate["resource_pass"], bool)
            or not isinstance(candidate["candidate_budget_pass"], bool)
            or candidate["resource_pass"] is not expected_resource_pass
            or candidate["candidate_budget_pass"] is not expected_candidate_budget_pass
            or candidate["conservative_12_epoch_wall_seconds"]
            != projection_by_cap[cap][
                "conservative_12_epoch_wall_seconds"
            ]
        ):
            raise R2MapPackingSweepError("packing selector candidate evidence differs")
    eligible = [
        candidate
        for candidate in selection["candidates"]
        if candidate["resource_pass"] and candidate["candidate_budget_pass"]
    ]
    if not eligible:
        raise R2MapPackingSweepError("packing selector has no passing cap")
    independently_selected = min(
        eligible,
        key=lambda candidate: (
            candidate["conservative_12_epoch_wall_seconds"],
            candidate["group_batch_size"],
        ),
    )
    if (
        independently_selected["group_batch_size"] != selection["selected_group_batch_size"]
        or independently_selected["conservative_12_epoch_wall_seconds"]
        != selection["selected_conservative_12_epoch_wall_seconds"]
    ):
        raise R2MapPackingSweepError("packing selector result is not independently reproducible")
    width_census = _exact_keys(
        report["width_census"],
        {
            "draft_groups",
            "selected_only_groups",
            "imitation_groups",
            "imitation_minimum",
            "imitation_median",
            "imitation_maximum",
        },
        "width census",
    )
    if (
        any(
            not isinstance(item, int) or isinstance(item, bool) or item < 0
            for item in width_census.values()
        )
        or width_census["draft_groups"]
        != width_census["selected_only_groups"] + width_census["imitation_groups"]
        or width_census["selected_only_groups"] <= 0
        or width_census["imitation_groups"] <= 0
        or not 1
        < width_census["imitation_minimum"]
        <= width_census["imitation_median"]
        <= width_census["imitation_maximum"]
        <= MAXIMUM_CANDIDATES_PER_BATCH
    ):
        raise R2MapPackingSweepError("width census accounting differs")
    representative_by_label = {measurement["label"]: measurement for measurement in representatives}
    for cap in QUALIFYING_CAPS:
        for suffix, expected_width in (
            ("selected", 1),
            ("imitation-p50", width_census["imitation_median"]),
            ("imitation-max", width_census["imitation_maximum"]),
        ):
            measurement = representative_by_label[f"g{cap}-{suffix}"]
            expected_groups = (
                cap
                if expected_width == 1
                else min(cap, MAXIMUM_CANDIDATES_PER_BATCH // expected_width)
            )
            widths = measurement["candidate_widths"]
            indices = measurement["frame_indices"]
            if (
                measurement["expected_group_count_per_step"] != [expected_groups] * timed_steps
                or not isinstance(widths, list)
                or len(widths) != expected_groups
                or max(widths) != expected_width
                or (expected_width == 1 and any(width != 1 for width in widths))
                or (
                    expected_width > 1
                    and (widths[0] != expected_width or any(width != 1 for width in widths[1:]))
                )
                or not isinstance(indices, list)
                or len(indices) != expected_groups
                or len(set(indices)) != expected_groups
                or measurement["training_counters"]["draft_candidates"] != sum(widths) * timed_steps
                or measurement["training_counters"]["padded_draft_candidates"]
                != max(widths) * expected_groups * timed_steps
                or measurement["training_counters"]["draft_policy_targets"]
                != sum(width > 1 for width in widths) * timed_steps
            ):
                raise R2MapPackingSweepError(
                    "representative shape is underfilled or has the wrong width"
                )
    _validate_resource_receipt(report["sweep_resource_receipt"], "sweep resource receipt")
    ssh = _exact_keys(
        report["ssh_transport"],
        {
            "alias",
            "compression",
            "hostname",
            "user",
            "identityfile",
            "controlmaster",
            "controlpath",
            "updatehostkeys",
        },
        "SSH transport",
    )
    if (
        ssh["alias"] != "john2"
        or ssh["compression"] != "no"
        or ssh["controlmaster"] != "no"
        or ssh["controlpath"] != "none"
        or ssh["updatehostkeys"] != "no"
    ):
        raise R2MapPackingSweepError("SSH no-persistence contract differs")
    preflight = _exact_keys(
        report["storage_preflight_receipt"],
        {"storage_receipt_relative", "storage_receipt_sha256"},
        "storage preflight receipt",
    )
    _digest(preflight["storage_receipt_sha256"], "storage preflight receipt")
    local_guard = report["local_write_guard"]
    expected_attestation_relative = (
        f"reports/w2-w3/{report['run_id']}/local-write-attestation.json"
    )
    if (
        not isinstance(local_guard, dict)
        or set(local_guard)
        != {
            "schema_version",
            "schema_id",
            "profile_sha256",
            "probe",
            "probe_errno",
            "all_local_file_writes_denied",
            "allowed_write_path",
            "attestation_relative",
        }
        or local_guard.get("profile_sha256") != SANDBOX_PROFILE_SHA256
        or local_guard.get("schema_version") != 1
        or local_guard.get("schema_id")
        != "cascadia.r2-map.john1-local-write-sandbox.v1"
        or local_guard.get("probe")
        != "/Users/johnherrick/cascadia/tools/r2_map_john1_packing_sweep.py"
        or local_guard.get("probe_errno") not in {1, 13}
        or isinstance(local_guard.get("probe_errno"), bool)
        or local_guard.get("all_local_file_writes_denied") is not True
        or local_guard.get("allowed_write_path") != "/dev/null"
        or local_guard.get("attestation_relative")
        != expected_attestation_relative
    ):
        raise R2MapPackingSweepError("local-write guard binding differs")
    # JSON round-trip rejects non-JSON subclasses and preserves a detached value.
    return json.loads(json.dumps(report, allow_nan=False))


def selected_training_contract(report: Mapping[str, Any]) -> dict[str, Any]:
    validated = validate_qualifying_packing_report(report)
    selection = validated["selection"]
    return {
        "group_batch_size": selection["selected_group_batch_size"],
        "maximum_candidates_per_batch": validated["packing_contract"][
            "maximum_candidates_per_batch"
        ],
        "schedule_steps": selection["selected_schedule_steps"],
        "epochs": selection["selected_epochs"],
        "report_sha256": validated["report_sha256"],
    }
