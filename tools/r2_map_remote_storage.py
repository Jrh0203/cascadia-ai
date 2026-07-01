#!/usr/bin/env python3
"""Operate the authoritative john2 R2-MAP storage root without mounting it."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPOSITORY / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from cascadia_mlx.r2_map_remote_storage import (  # noqa: E402
    RemoteStorageClient,
    TransactionObject,
    build_john1_runtime_manifest,
    build_transaction_manifest,
    bytes_chunks,
    canonical_json,
    deploy_dashboard_api,
    execute_john1_runtime,
)


def _json_stdout(value: Any) -> None:
    sys.stdout.buffer.write(canonical_json(value) + b"\n")


def _source(path: str):
    if path == "-":
        return sys.stdin.buffer, False
    return Path(path).open("rb"), True


def _load_json(path: str) -> dict[str, Any]:
    source, should_close = _source(path)
    try:
        value = json.load(source)
    finally:
        if should_close:
            source.close()
    if not isinstance(value, dict):
        raise ValueError("JSON input must be an object")
    return value


def _command_install(client: RemoteStorageClient, _arguments: argparse.Namespace) -> None:
    _json_stdout(client.install_worker())


def _command_provision(client: RemoteStorageClient, _arguments: argparse.Namespace) -> None:
    client.install_worker()
    _json_stdout(client.provision())


def _command_preflight(client: RemoteStorageClient, _arguments: argparse.Namespace) -> None:
    _json_stdout(client.preflight())


def _command_token(client: RemoteStorageClient, arguments: argparse.Namespace) -> None:
    _json_stdout(client.open_object_with_receipt(arguments.relative))


def _command_fetch(client: RemoteStorageClient, arguments: argparse.Namespace) -> None:
    token = client.open_object(arguments.relative)
    if arguments.expected_sha256 is not None and token["sha256"] != arguments.expected_sha256:
        raise ValueError("remote object differs from --expected-sha256")
    for chunk in client.iter_object(token, window_bytes=arguments.window_bytes):
        sys.stdout.buffer.write(chunk)
    sys.stdout.buffer.flush()


def _command_deploy_dashboard_api(
    client: RemoteStorageClient, arguments: argparse.Namespace
) -> None:
    _json_stdout(
        deploy_dashboard_api(
            client,
            bundle_relative=arguments.relative,
            expected_sha256=arguments.expected_sha256,
        )
    )


def _command_put(client: RemoteStorageClient, arguments: argparse.Namespace) -> None:
    if arguments.source == "-" and (arguments.size is None or arguments.sha256 is None):
        raise ValueError("stdin put requires --size and --sha256")
    source, should_close = _source(arguments.source)
    try:
        if arguments.source == "-":
            size = arguments.size
            sha256 = arguments.sha256
        else:
            details = os.fstat(source.fileno())
            size = details.st_size
            import hashlib

            hasher = hashlib.sha256()
            while chunk := source.read(1 << 20):
                hasher.update(chunk)
            sha256 = hasher.hexdigest()
            source.seek(0)
            if arguments.size is not None and arguments.size != size:
                raise ValueError("source size differs from --size")
            if arguments.sha256 is not None and arguments.sha256 != sha256:
                raise ValueError("source SHA-256 differs from --sha256")
        _json_stdout(
            client.put_stream(
                arguments.relative,
                bytes_chunks(source),
                size=size,
                sha256=sha256,
                expected_current=arguments.expected_current,
                mutable=arguments.mutable,
            )
        )
    finally:
        if should_close:
            source.close()


def _command_put_stream(client: RemoteStorageClient, arguments: argparse.Namespace) -> None:
    source, should_close = _source(arguments.source)
    try:
        _json_stdout(
            client.put_unknown_stream(
                arguments.relative,
                bytes_chunks(source),
                max_bytes=arguments.max_bytes,
                expected_current=arguments.expected_current,
            )
        )
    finally:
        if should_close:
            source.close()


def _command_publish_status(client: RemoteStorageClient, arguments: argparse.Namespace) -> None:
    source, should_close = _source(arguments.source)
    try:
        payload = source.read(65537)
        if source.read(1):
            raise ValueError("status source exceeds 64 KiB")
    finally:
        if should_close:
            source.close()
    _json_stdout(client.publish_status(payload, expected_current=arguments.expected_current))


def _command_lock(client: RemoteStorageClient, arguments: argparse.Namespace) -> None:
    if arguments.lock_action == "acquire":
        result = client.acquire_lock(
            arguments.name,
            arguments.owner,
            arguments.lease_seconds,
            lease_epoch=arguments.lease_epoch,
        )
    elif arguments.lock_action == "renew":
        result = client.renew_lock(
            arguments.name,
            arguments.owner,
            arguments.token,
            arguments.lease_seconds,
            lease_epoch=arguments.lease_epoch,
        )
    else:
        result = client.release_lock(
            arguments.name,
            arguments.owner,
            arguments.token,
            arguments.lease_seconds,
            lease_epoch=arguments.lease_epoch,
        )
    _json_stdout(result)


def _command_manifest(_client: RemoteStorageClient, arguments: argparse.Namespace) -> None:
    descriptors = []
    for encoded in arguments.object:
        fields = encoded.split(":")
        if len(fields) not in {3, 4}:
            raise ValueError("transaction object must be RELATIVE:SIZE:SHA256[:MODE]")
        relative, size, sha256 = fields[:3]
        mode = 0o400 if len(fields) == 3 else int(fields[3], 8)
        descriptors.append(TransactionObject(relative, int(size), sha256, mode=mode))
    _json_stdout(
        build_transaction_manifest(arguments.transaction_id, arguments.target_relative, descriptors)
    )


def _command_transaction_begin(client: RemoteStorageClient, arguments: argparse.Namespace) -> None:
    _json_stdout(client.begin_transaction(_load_json(arguments.manifest)))


def _command_transaction_put(client: RemoteStorageClient, arguments: argparse.Namespace) -> None:
    source, should_close = _source(arguments.source)
    try:
        descriptor = TransactionObject(arguments.relative, arguments.size, arguments.sha256)
        _json_stdout(
            client.put_transaction_object(
                arguments.transaction_id, descriptor, bytes_chunks(source)
            )
        )
    finally:
        if should_close:
            source.close()


def _command_transaction_import(client: RemoteStorageClient, arguments: argparse.Namespace) -> None:
    descriptor = TransactionObject(
        arguments.relative,
        arguments.size,
        arguments.sha256,
        mode=int(arguments.mode, 8),
    )
    _json_stdout(
        client.import_transaction_object(
            arguments.transaction_id,
            descriptor,
            source_relative=arguments.source_relative,
        )
    )


def _command_transaction_finish(client: RemoteStorageClient, arguments: argparse.Namespace) -> None:
    if arguments.transaction_action == "commit":
        result = client.commit_transaction(arguments.transaction_id, arguments.manifest_sha256)
    else:
        result = client.abort_transaction(arguments.transaction_id, arguments.manifest_sha256)
    _json_stdout(result)


def _command_run(client: RemoteStorageClient, arguments: argparse.Namespace) -> None:
    specification = _load_json(arguments.specification)
    required = {"run_id", "cwd_relative", "argv", "output_relative", "timeout_seconds"}
    if not required.issubset(specification):
        raise ValueError("run specification omits a required field")
    _json_stdout(
        client.run_remote(
            run_id=specification["run_id"],
            cwd_relative=specification["cwd_relative"],
            argv=specification["argv"],
            output_relative=specification["output_relative"],
            timeout_seconds=specification["timeout_seconds"],
            environment=specification.get("environment", {}),
            python_path_relatives=specification.get("python_path_relatives", []),
        )
    )


def _command_controller_run(client: RemoteStorageClient, arguments: argparse.Namespace) -> None:
    specification = _load_json(arguments.specification)
    required = {
        "run_id",
        "source_manifest_sha256",
        "cwd_relative",
        "executable_relative",
        "arguments",
        "output_relative",
    }
    if not required.issubset(specification):
        raise ValueError("controller run specification omits a required field")
    _json_stdout(
        client.run_controller(
            run_id=specification["run_id"],
            source_manifest_sha256=specification["source_manifest_sha256"],
            cwd_relative=specification["cwd_relative"],
            executable_relative=specification["executable_relative"],
            arguments=specification["arguments"],
            output_relative=specification["output_relative"],
            timeout_seconds=specification.get("timeout_seconds", 600),
            python_path_relatives=specification.get("python_path_relatives", []),
        )
    )


def _command_run_cleanup_prepare(
    client: RemoteStorageClient, arguments: argparse.Namespace
) -> None:
    specification = _load_json(arguments.specification)
    required = {"run_id", "manifest_object_token", "dataset_object_token"}
    if not required.issubset(specification):
        raise ValueError("run cleanup specification omits a required field")
    _json_stdout(
        client.prepare_run_cleanup(
            run_id=specification["run_id"],
            manifest_object_token=specification["manifest_object_token"],
            dataset_object_token=specification["dataset_object_token"],
        )
    )


def _command_run_cleanup_commit(client: RemoteStorageClient, arguments: argparse.Namespace) -> None:
    specification = _load_json(arguments.token)
    cleanup_token = specification.get("cleanup_token", specification)
    if not isinstance(cleanup_token, dict):
        raise ValueError("run cleanup token input is not an object")
    _json_stdout(client.commit_run_cleanup(cleanup_token))


def _command_failed_run_cleanup_prepare(
    client: RemoteStorageClient, arguments: argparse.Namespace
) -> None:
    _json_stdout(client.prepare_failed_run_cleanup(run_id=arguments.run_id))


def _command_failed_run_cleanup_commit(
    client: RemoteStorageClient, arguments: argparse.Namespace
) -> None:
    specification = _load_json(arguments.token)
    cleanup_token = specification.get("cleanup_token", specification)
    if not isinstance(cleanup_token, dict):
        raise ValueError("failed-run cleanup token input is not an object")
    _json_stdout(client.commit_failed_run_cleanup(cleanup_token))


def _command_inspect_executable(client: RemoteStorageClient, arguments: argparse.Namespace) -> None:
    _json_stdout(
        client.inspect_remote_executable(
            arguments.relative,
            size=arguments.size,
            sha256=arguments.sha256,
        )
    )


def _command_runtime_manifest(_client: RemoteStorageClient, arguments: argparse.Namespace) -> None:
    inspection = _load_json(arguments.inspection)
    source_freeze = _load_json(arguments.source_freeze)
    _json_stdout(
        build_john1_runtime_manifest(
            packet_id=arguments.packet_id,
            executable_relative=arguments.executable_relative,
            inspection=inspection,
            source_freeze=source_freeze,
            build_receipt_relative=arguments.build_receipt_relative,
            build_receipt_sha256=arguments.build_receipt_sha256,
            output_prefix_relative=arguments.output_prefix_relative,
            stdout_max_bytes=arguments.stdout_max_bytes,
            stderr_max_bytes=arguments.stderr_max_bytes,
            created_unix_ms=arguments.created_unix_ms,
        )
    )


def _command_execute_john1(client: RemoteStorageClient, arguments: argparse.Namespace) -> None:
    _json_stdout(
        execute_john1_runtime(
            client,
            manifest_relative=arguments.manifest_relative,
            run_id=arguments.run_id,
            arguments=arguments.runtime_argument,
            timeout_seconds=arguments.timeout_seconds,
        )
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    for name, handler in (
        ("install-worker", _command_install),
        ("provision", _command_provision),
        ("preflight", _command_preflight),
    ):
        command = subparsers.add_parser(name)
        command.set_defaults(handler=handler)

    token = subparsers.add_parser("object-token")
    token.add_argument("--relative", required=True)
    token.set_defaults(handler=_command_token)

    fetch = subparsers.add_parser("fetch")
    fetch.add_argument("--relative", required=True)
    fetch.add_argument("--expected-sha256")
    fetch.add_argument("--window-bytes", type=int, default=64 * (1 << 20))
    fetch.set_defaults(handler=_command_fetch)

    deploy_dashboard = subparsers.add_parser("deploy-dashboard-api")
    deploy_dashboard.add_argument("--relative", required=True)
    deploy_dashboard.add_argument("--expected-sha256", required=True)
    deploy_dashboard.set_defaults(handler=_command_deploy_dashboard_api)

    put = subparsers.add_parser("put")
    put.add_argument("--relative", required=True)
    put.add_argument("--source", required=True)
    put.add_argument("--size", type=int)
    put.add_argument("--sha256")
    put.add_argument("--expected-current", default="absent")
    put.add_argument("--mutable", action="store_true")
    put.set_defaults(handler=_command_put)

    stream = subparsers.add_parser("put-stream")
    stream.add_argument("--relative", required=True)
    stream.add_argument("--source", default="-")
    stream.add_argument("--max-bytes", type=int, required=True)
    stream.add_argument("--expected-current", default="absent")
    stream.set_defaults(handler=_command_put_stream)

    status = subparsers.add_parser("publish-status")
    status.add_argument("--source", default="-")
    status.add_argument("--expected-current", default="absent")
    status.set_defaults(handler=_command_publish_status)

    lock = subparsers.add_parser("lock")
    lock.add_argument("lock_action", choices=("acquire", "renew", "release"))
    lock.add_argument("--name", required=True)
    lock.add_argument("--owner", required=True)
    lock.add_argument("--token")
    lock.add_argument("--lease-seconds", type=int, default=300)
    lock.add_argument("--lease-epoch", required=True)
    lock.set_defaults(handler=_command_lock)

    manifest = subparsers.add_parser("transaction-manifest")
    manifest.add_argument("--transaction-id", required=True)
    manifest.add_argument("--target-relative", required=True)
    manifest.add_argument(
        "--object", action="append", required=True, metavar="RELATIVE:SIZE:SHA256[:MODE]"
    )
    manifest.set_defaults(handler=_command_manifest)

    begin = subparsers.add_parser("transaction-begin")
    begin.add_argument("--manifest", default="-")
    begin.set_defaults(handler=_command_transaction_begin)

    transaction_put = subparsers.add_parser("transaction-put")
    transaction_put.add_argument("--transaction-id", required=True)
    transaction_put.add_argument("--relative", required=True)
    transaction_put.add_argument("--source", default="-")
    transaction_put.add_argument("--size", type=int, required=True)
    transaction_put.add_argument("--sha256", required=True)
    transaction_put.set_defaults(handler=_command_transaction_put)

    transaction_import = subparsers.add_parser("transaction-import")
    transaction_import.add_argument("--transaction-id", required=True)
    transaction_import.add_argument("--relative", required=True)
    transaction_import.add_argument("--source-relative", required=True)
    transaction_import.add_argument("--size", type=int, required=True)
    transaction_import.add_argument("--sha256", required=True)
    transaction_import.add_argument("--mode", choices=("0400", "0500"), default="0400")
    transaction_import.set_defaults(handler=_command_transaction_import)

    finish = subparsers.add_parser("transaction-finish")
    finish.add_argument("transaction_action", choices=("commit", "abort"))
    finish.add_argument("--transaction-id", required=True)
    finish.add_argument("--manifest-sha256", required=True)
    finish.set_defaults(handler=_command_transaction_finish)

    run = subparsers.add_parser("run")
    run.add_argument("--specification", required=True)
    run.set_defaults(handler=_command_run)

    controller_run = subparsers.add_parser("controller-run")
    controller_run.add_argument("--specification", required=True)
    controller_run.set_defaults(handler=_command_controller_run)

    cleanup_prepare = subparsers.add_parser("run-cleanup-prepare")
    cleanup_prepare.add_argument("--specification", required=True)
    cleanup_prepare.set_defaults(handler=_command_run_cleanup_prepare)

    cleanup_commit = subparsers.add_parser("run-cleanup-commit")
    cleanup_commit.add_argument("--token", required=True)
    cleanup_commit.set_defaults(handler=_command_run_cleanup_commit)

    failed_cleanup_prepare = subparsers.add_parser("failed-run-cleanup-prepare")
    failed_cleanup_prepare.add_argument("--run-id", required=True)
    failed_cleanup_prepare.set_defaults(handler=_command_failed_run_cleanup_prepare)

    failed_cleanup_commit = subparsers.add_parser("failed-run-cleanup-commit")
    failed_cleanup_commit.add_argument("--token", required=True)
    failed_cleanup_commit.set_defaults(handler=_command_failed_run_cleanup_commit)

    inspect_executable = subparsers.add_parser("inspect-executable")
    inspect_executable.add_argument("--relative", required=True)
    inspect_executable.add_argument("--size", type=int, required=True)
    inspect_executable.add_argument("--sha256", required=True)
    inspect_executable.set_defaults(handler=_command_inspect_executable)

    runtime_manifest = subparsers.add_parser("runtime-manifest")
    runtime_manifest.add_argument("--packet-id", required=True)
    runtime_manifest.add_argument("--executable-relative", required=True)
    runtime_manifest.add_argument("--inspection", required=True)
    runtime_manifest.add_argument("--source-freeze", required=True)
    runtime_manifest.add_argument("--build-receipt-relative", required=True)
    runtime_manifest.add_argument("--build-receipt-sha256", required=True)
    runtime_manifest.add_argument("--output-prefix-relative", required=True)
    runtime_manifest.add_argument("--stdout-max-bytes", type=int, required=True)
    runtime_manifest.add_argument("--stderr-max-bytes", type=int, required=True)
    runtime_manifest.add_argument("--created-unix-ms", type=int, required=True)
    runtime_manifest.set_defaults(handler=_command_runtime_manifest)

    execute_john1 = subparsers.add_parser("execute-john1-runtime")
    execute_john1.add_argument("--manifest-relative", required=True)
    execute_john1.add_argument("--run-id", required=True)
    execute_john1.add_argument("--runtime-argument", action="append", default=[])
    execute_john1.add_argument("--timeout-seconds", type=int, default=3600)
    execute_john1.set_defaults(handler=_command_execute_john1)
    return result


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    client = RemoteStorageClient()
    cancelling = False

    def cancel_owned_ssh(sent_signal: int, _frame: object) -> None:
        nonlocal cancelling
        if cancelling:
            raise SystemExit(128 + sent_signal)
        cancelling = True
        client.transport.cancel_active()
        raise SystemExit(128 + sent_signal)

    previous = {
        sent_signal: signal.signal(sent_signal, cancel_owned_ssh)
        for sent_signal in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        arguments.handler(client, arguments)
        return 0
    finally:
        client.transport.cancel_active()
        for sent_signal, handler in previous.items():
            signal.signal(sent_signal, handler)


if __name__ == "__main__":
    raise SystemExit(main())
