from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest
from cascadia_mlx.r2_map_remote_training import (
    John2RemoteWindowLoader,
    R2MapRemoteTrainingError,
    read_remote_object,
)


def _binding(name: str) -> dict[str, str]:
    return {
        "storage_receipt_relative": f"control/receipts/req-{name}.json",
        "storage_receipt_sha256": hashlib.sha256(name.encode()).hexdigest(),
    }


class MemoryRemoteClient:
    def __init__(self, objects: dict[str, bytes] | None = None):
        self.objects = dict(objects or {})
        self.cleanup_prepared = 0
        self.cleanup_committed = 0
        self.failed_cleanup_prepared = 0
        self.failed_cleanup_committed = 0
        self.run_exit_code = 0
        self.normal_cleanup_fails = False

    def open_object_with_receipt(self, relative: str) -> dict[str, Any]:
        payload = self.objects[relative]
        digest = hashlib.sha256(payload).hexdigest()
        return {
            "object_token": {
                "schema_version": 1,
                "schema_id": "cascadia.r2-map.remote-object-token.v1",
                "relative": relative,
                "size": len(payload),
                "sha256": digest,
                "token_sha256": hashlib.sha256((relative + digest).encode()).hexdigest(),
            },
            **_binding(f"open-{len(relative)}"),
        }

    def iter_object_with_receipts(
        self, token: dict[str, Any], *, window_bytes: int
    ):
        payload = self.objects[token["relative"]]
        for offset in range(0, len(payload), window_bytes):
            chunk = payload[offset : offset + window_bytes]
            yield {
                "payload": chunk,
                "payload_sha256": hashlib.sha256(chunk).hexdigest(),
                "object_token_sha256": token["token_sha256"],
                "offset": offset,
                "length": len(chunk),
                **_binding(f"read-{offset}"),
            }

    def run_remote(self, **arguments: Any) -> dict[str, Any]:
        run_id = arguments["run_id"]
        manifest = f"build/run-{run_id}/window.json"
        dataset = f"build/run-{run_id}/window.r2map"
        self.objects[manifest] = json.dumps(
            {"sources": [{"file_name": "part.r2sh"}]}
        ).encode()
        self.objects[dataset] = b"bounded-window"
        return {
            "exit_code": self.run_exit_code,
            "timed_out": False,
            "temporary_cleaned": True,
            **_binding("run"),
        }

    def open_ephemeral_run_outputs(
        self, *, run_id: str, manifest_relative: str, dataset_relative: str
    ) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "manifest": self.open_object_with_receipt(manifest_relative),
            "dataset": self.open_object_with_receipt(dataset_relative),
        }

    def prepare_run_cleanup(self, **arguments: Any) -> dict[str, Any]:
        self.cleanup_prepared += 1
        if self.normal_cleanup_fails:
            raise RuntimeError("injected token cleanup failure")
        return {
            "cleanup_token": {
                "cleanup_token_sha256": "a" * 64,
                "run_id": arguments["run_id"],
            },
            **_binding("cleanup-prepare"),
        }

    def commit_run_cleanup(self, cleanup_token: dict[str, Any]) -> dict[str, Any]:
        self.cleanup_committed += 1
        return {
            "cleanup_token_sha256": cleanup_token["cleanup_token_sha256"],
            "removed": True,
            **_binding("cleanup-commit"),
        }

    def prepare_failed_run_cleanup(self, *, run_id: str) -> dict[str, Any]:
        self.failed_cleanup_prepared += 1
        return {
            "cleanup_token": {
                "cleanup_token_sha256": "b" * 64,
                "run_id": run_id,
            },
            **_binding("failed-prepare"),
        }

    def commit_failed_run_cleanup(self, cleanup_token: dict[str, Any]) -> dict[str, Any]:
        self.failed_cleanup_committed += 1
        return {
            "cleanup_token_sha256": cleanup_token["cleanup_token_sha256"],
            **_binding("failed-commit"),
        }


def test_receipt_preserving_object_read_is_contiguous_and_bounded() -> None:
    client = MemoryRemoteClient({"datasets/index.json": b"abcdefghij"})
    value = read_remote_object(
        client,  # type: ignore[arg-type]
        "datasets/index.json",
        maximum_bytes=10,
        as_bytearray=True,
        window_bytes=4,
    )
    assert value.payload == bytearray(b"abcdefghij")
    assert len(value.evidence.range_receipts) == 3
    assert value.evidence.open_receipt["storage_receipt_relative"].startswith(
        "control/receipts/req-"
    )

    with pytest.raises(R2MapRemoteTrainingError, match="memory bound"):
        read_remote_object(
            client,  # type: ignore[arg-type]
            "datasets/index.json",
            maximum_bytes=9,
        )


def test_remote_window_loader_cleans_before_return_and_retains_every_receipt() -> None:
    client = MemoryRemoteClient()
    loader = John2RemoteWindowLoader(
        client,  # type: ignore[arg-type]
        exporter_relative="bundles/exporter/cascadia-v2",
        shard_root_relative="datasets/bootstrap/iteration-0",
        maximum_window_bytes=64,
    )
    manifest, stream = loader("part.r2sh", "train", 2, 7, 0, (7,))
    assert manifest["sources"][0]["file_name"] == "part.r2sh"
    assert stream == bytearray(b"bounded-window")
    assert client.cleanup_prepared == client.cleanup_committed == 1
    assert client.failed_cleanup_prepared == client.failed_cleanup_committed == 0
    assert len(loader.evidence) == 1
    evidence = loader.evidence[0]
    assert evidence.mode == "train"
    assert evidence.dataset.range_receipts
    assert evidence.cleanup_commit_receipt["removed"] is True


def test_remote_window_loader_uses_failed_run_cleanup_on_nonzero_exit() -> None:
    client = MemoryRemoteClient()
    client.run_exit_code = 2
    loader = John2RemoteWindowLoader(
        client,  # type: ignore[arg-type]
        exporter_relative="bundles/exporter/cascadia-v2",
        shard_root_relative="datasets/bootstrap/iteration-0",
    )
    with pytest.raises(R2MapRemoteTrainingError, match="did not finish"):
        loader("part.r2sh", "validation", 0, 0, 0, (7,))
    assert client.failed_cleanup_prepared == client.failed_cleanup_committed == 1
    assert client.cleanup_prepared == client.cleanup_committed == 0


def test_remote_window_loader_falls_back_to_inventory_cleanup() -> None:
    client = MemoryRemoteClient()
    client.normal_cleanup_fails = True
    loader = John2RemoteWindowLoader(
        client,  # type: ignore[arg-type]
        exporter_relative="bundles/exporter/cascadia-v2",
        shard_root_relative="datasets/bootstrap/iteration-0",
        maximum_window_bytes=1,
    )
    with pytest.raises(R2MapRemoteTrainingError, match="memory ceiling"):
        loader("part.r2sh", "train", 0, 0, 0, (7,))
    assert client.cleanup_prepared == 1
    assert client.cleanup_committed == 0
    assert client.failed_cleanup_prepared == client.failed_cleanup_committed == 1


@pytest.mark.parametrize("source", ("../part.r2sh", "nested/part.r2sh", "."))
def test_remote_window_loader_rejects_nonbasename_source(source: str) -> None:
    loader = John2RemoteWindowLoader(
        MemoryRemoteClient(),  # type: ignore[arg-type]
        exporter_relative="bundles/exporter/cascadia-v2",
        shard_root_relative="datasets/bootstrap/iteration-0",
    )
    with pytest.raises(R2MapRemoteTrainingError, match="one safe file name"):
        loader(source, "train", 0, 0, 0, (7,))
