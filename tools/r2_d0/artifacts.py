"""Pinned D0 artifact acquisition and deterministic OCI/context construction."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import shutil
import stat
import tarfile
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .canonical import (
    CAMPAIGN_ID,
    CORE_IMAGE,
    FROZEN_HOMEBREW_TAP_HEAD,
    FROZEN_RUNTIME,
    PROBE_DOCKERFILE,
    PROBE_DOCKERFILE_SHA256,
    PROBE_PAYLOAD,
    PROBE_PAYLOAD_SHA256,
    SCANNER_IMAGE,
    SMOKE_IMAGE,
    D0Error,
    canonical_json,
    document_sha256,
    load_canonical_json,
    sha256_bytes,
)
from .inventory import secure_owner_directory
from .storage import verify_canonical_commit_boundary

ARTIFACT_RECEIPT_SCHEMA = "cascadia.r2-map.d0-artifact-receipt.v1"
OCI_RECEIPT_SCHEMA = "cascadia.r2-map.d0-smoke-oci-receipt.v1"
PROBE_RECEIPT_SCHEMA = "cascadia.r2-map.d0-probe-context-receipt.v1"
HOMEBREW_CLOSURE_SCHEMA = "cascadia.r2-map.d0-homebrew-closure.v1"
RUNTIME_SUPPLY_SCHEMA = "cascadia.r2-map.d0-worker-runtime-supply.v1"
RECORD_SIZE = 10 * 1024
MAX_HTTP_BYTES = 2 * 1024**3
REGISTRY = "https://registry-1.docker.io"
AUTH = "https://auth.docker.io/token"
ALPINE_ACCEPT_INDEX = (
    "application/vnd.oci.image.index.v1+json,"
    "application/vnd.docker.distribution.manifest.list.v2+json"
)
ALPINE_ACCEPT_MANIFEST = "application/vnd.oci.image.manifest.v1+json"


def _tar_bytes(files: Mapping[str, tuple[bytes, int]]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w|", format=tarfile.USTAR_FORMAT) as archive:
        for name in sorted(files):
            payload, mode = files[name]
            if name.startswith("/") or ".." in name.split("/") or len(name.encode("ascii")) > 100:
                raise D0Error("deterministic tar member path is unsafe")
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = mode
            info.uid = 0
            info.gid = 0
            info.mtime = 0
            info.uname = ""
            info.gname = ""
            archive.addfile(info, io.BytesIO(payload))
    value = output.getvalue()
    if len(value) % RECORD_SIZE:
        raise D0Error("deterministic tar record padding differs")
    return value


def probe_context() -> tuple[bytes, dict[str, Any]]:
    if sha256_bytes(PROBE_DOCKERFILE) != PROBE_DOCKERFILE_SHA256:
        raise D0Error("frozen probe Dockerfile hash differs")
    if sha256_bytes(PROBE_PAYLOAD) != PROBE_PAYLOAD_SHA256:
        raise D0Error("frozen probe payload hash differs")
    archive = _tar_bytes(
        {
            "Dockerfile": (PROBE_DOCKERFILE, 0o444),
            "probe.txt": (PROBE_PAYLOAD, 0o444),
        }
    )
    report: dict[str, Any] = {
        "schema_id": PROBE_RECEIPT_SCHEMA,
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "dockerfile_sha256": PROBE_DOCKERFILE_SHA256,
        "payload_sha256": PROBE_PAYLOAD_SHA256,
        "archive_bytes": len(archive),
        "archive_sha256": sha256_bytes(archive),
        "member_count": 2,
        "project_code_present": False,
        "protected_seed_values_opened": False,
    }
    report["receipt_sha256"] = document_sha256(report, "receipt_sha256")
    return archive, report


def _json_object(value: bytes, label: str) -> dict[str, Any]:
    try:
        decoded = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D0Error(f"{label} is not JSON") from error
    if not isinstance(decoded, dict):
        raise D0Error(f"{label} is not an object")
    return decoded


def validate_alpine_objects(
    source_index: bytes,
    manifest: bytes,
    config: bytes,
    layer: bytes,
) -> None:
    frozen = SMOKE_IMAGE
    if f"sha256:{sha256_bytes(source_index)}" != frozen["index_digest"]:
        raise D0Error("Alpine source index digest differs")
    if f"sha256:{sha256_bytes(manifest)}" != frozen["manifest_digest"]:
        raise D0Error("Alpine arm64 manifest digest differs")
    if f"sha256:{sha256_bytes(config)}" != frozen["config_digest"]:
        raise D0Error("Alpine config digest differs")
    if f"sha256:{sha256_bytes(layer)}" != frozen["layer_digest"]:
        raise D0Error("Alpine layer digest differs")
    if (
        len(manifest) != frozen["manifest_size"]
        or len(config) != frozen["config_size"]
        or len(layer) != frozen["layer_size"]
    ):
        raise D0Error("Alpine object size differs")
    index_value = _json_object(source_index, "Alpine source index")
    descriptors = index_value.get("manifests")
    if not isinstance(descriptors, list):
        raise D0Error("Alpine source index descriptors are absent")
    arm = [
        descriptor
        for descriptor in descriptors
        if isinstance(descriptor, dict)
        and descriptor.get("digest") == frozen["manifest_digest"]
        and descriptor.get("platform") == {"architecture": "arm64", "os": "linux", "variant": "v8"}
    ]
    if len(arm) != 1:
        raise D0Error("Alpine arm64 descriptor differs")
    manifest_value = _json_object(manifest, "Alpine manifest")
    if manifest_value.get("config") != {
        "mediaType": "application/vnd.oci.image.config.v1+json",
        "digest": frozen["config_digest"],
        "size": frozen["config_size"],
    }:
        raise D0Error("Alpine config descriptor differs")
    if manifest_value.get("layers") != [
        {
            "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
            "digest": frozen["layer_digest"],
            "size": frozen["layer_size"],
        }
    ]:
        raise D0Error("Alpine layer descriptor differs")


def smoke_oci_archive(
    source_index: bytes,
    manifest: bytes,
    config: bytes,
    layer: bytes,
) -> tuple[bytes, dict[str, Any]]:
    validate_alpine_objects(source_index, manifest, config, layer)
    frozen = SMOKE_IMAGE
    local_index = canonical_json(
        {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.index.v1+json",
            "manifests": [
                {
                    "mediaType": "application/vnd.oci.image.manifest.v1+json",
                    "digest": frozen["manifest_digest"],
                    "size": frozen["manifest_size"],
                    "platform": {"architecture": "arm64", "os": "linux", "variant": "v8"},
                    "annotations": {"org.opencontainers.image.ref.name": "alpine:3.22.1-d0"},
                }
            ],
        }
    )
    oci_layout = canonical_json({"imageLayoutVersion": "1.0.0"})
    files = {
        "oci-layout": (oci_layout, 0o444),
        "index.json": (local_index, 0o444),
        f"blobs/sha256/{frozen['manifest_digest'].split(':', 1)[1]}": (manifest, 0o444),
        f"blobs/sha256/{frozen['config_digest'].split(':', 1)[1]}": (config, 0o444),
        f"blobs/sha256/{frozen['layer_digest'].split(':', 1)[1]}": (layer, 0o444),
    }
    archive = _tar_bytes(files)
    receipt: dict[str, Any] = {
        "schema_id": OCI_RECEIPT_SCHEMA,
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "repository": frozen["repository"],
        "tag": frozen["tag"],
        "source_index_digest": frozen["index_digest"],
        "manifest_digest": frozen["manifest_digest"],
        "config_digest": frozen["config_digest"],
        "layer_digests": [frozen["layer_digest"]],
        "local_index_sha256": sha256_bytes(local_index),
        "archive_bytes": len(archive),
        "archive_sha256": sha256_bytes(archive),
        "platform": "linux/arm64/v8",
    }
    receipt["receipt_sha256"] = document_sha256(receipt, "receipt_sha256")
    return archive, receipt


def verify_smoke_oci_archive(archive: bytes) -> dict[str, Any]:
    """Reparse and authenticate the full frozen OCI graph immediately before load."""

    if not archive or len(archive) > 64 * 1024 * 1024:
        raise D0Error("Alpine OCI archive size differs")
    files: dict[str, tuple[bytes, int]] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as source:
            for item in source:
                if (
                    not item.isfile()
                    or item.name.startswith("/")
                    or ".." in item.name.split("/")
                    or item.name in files
                    or item.uid != 0
                    or item.gid != 0
                    or item.mtime != 0
                    or item.mode != 0o444
                ):
                    raise D0Error("Alpine OCI archive member metadata differs")
                stream = source.extractfile(item)
                if stream is None:
                    raise D0Error("Alpine OCI archive member is unreadable")
                payload = stream.read(64 * 1024 * 1024 + 1)
                if len(payload) != item.size:
                    raise D0Error("Alpine OCI archive member size differs")
                files[item.name] = (payload, 0o444)
    except (OSError, tarfile.TarError) as error:
        raise D0Error("Alpine OCI archive is invalid") from error
    frozen = SMOKE_IMAGE
    manifest_name = f"blobs/sha256/{frozen['manifest_digest'].split(':', 1)[1]}"
    config_name = f"blobs/sha256/{frozen['config_digest'].split(':', 1)[1]}"
    layer_name = f"blobs/sha256/{frozen['layer_digest'].split(':', 1)[1]}"
    expected_names = {"oci-layout", "index.json", manifest_name, config_name, layer_name}
    if set(files) != expected_names:
        raise D0Error("Alpine OCI archive graph differs")
    layout = canonical_json({"imageLayoutVersion": "1.0.0"})
    local_index = canonical_json(
        {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.index.v1+json",
            "manifests": [
                {
                    "mediaType": "application/vnd.oci.image.manifest.v1+json",
                    "digest": frozen["manifest_digest"],
                    "size": frozen["manifest_size"],
                    "platform": {"architecture": "arm64", "os": "linux", "variant": "v8"},
                    "annotations": {"org.opencontainers.image.ref.name": "alpine:3.22.1-d0"},
                }
            ],
        }
    )
    manifest = files[manifest_name][0]
    config = files[config_name][0]
    layer = files[layer_name][0]
    if files["oci-layout"][0] != layout or files["index.json"][0] != local_index:
        raise D0Error("Alpine OCI layout or local index differs")
    if (
        len(manifest) != frozen["manifest_size"]
        or len(config) != frozen["config_size"]
        or len(layer) != frozen["layer_size"]
        or f"sha256:{sha256_bytes(manifest)}" != frozen["manifest_digest"]
        or f"sha256:{sha256_bytes(config)}" != frozen["config_digest"]
        or f"sha256:{sha256_bytes(layer)}" != frozen["layer_digest"]
    ):
        raise D0Error("Alpine OCI content identity differs")
    manifest_value = _json_object(manifest, "Alpine OCI manifest")
    if manifest_value.get("config", {}).get("digest") != frozen["config_digest"] or [
        item.get("digest") for item in manifest_value.get("layers", [])
    ] != [frozen["layer_digest"]]:
        raise D0Error("Alpine OCI descriptor graph differs")
    config_value = _json_object(config, "Alpine OCI config")
    try:
        uncompressed = gzip.decompress(layer)
    except (OSError, EOFError) as error:
        raise D0Error("Alpine OCI layer cannot be decompressed") from error
    diff_id = f"sha256:{sha256_bytes(uncompressed)}"
    layer_payload = _layer_payload_projection(
        uncompressed,
        label="Alpine OCI layer",
        required_regular="bin/busybox",
    )
    if (
        config_value.get("architecture") != "arm64"
        or config_value.get("os") != "linux"
        or config_value.get("rootfs", {}).get("type") != "layers"
        or config_value.get("rootfs", {}).get("diff_ids") != [diff_id]
    ):
        raise D0Error("Alpine OCI config/rootfs identity differs")
    if _tar_bytes(files) != archive:
        raise D0Error("Alpine OCI archive is not canonical deterministic USTAR")
    return {
        "archive_size": len(archive),
        "archive_sha256": sha256_bytes(archive),
        "manifest_digest": frozen["manifest_digest"],
        "config_digest": frozen["config_digest"],
        "layer_digest": frozen["layer_digest"],
        "diff_id": diff_id,
        "layer_payload": layer_payload,
        "local_index_sha256": sha256_bytes(local_index),
        "status": "pass",
    }


def _layer_payload_projection(
    payload: bytes,
    *,
    label: str,
    required_regular: str,
) -> dict[str, Any]:
    if not payload or len(payload) > 512 * 1024 * 1024:
        raise D0Error(f"{label} uncompressed size differs")
    names: set[str] = set()
    regular: set[str] = set()
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
            for member in archive:
                name = member.name.removeprefix("./")
                if (
                    not name
                    or name.startswith("/")
                    or ".." in name.split("/")
                    or name in names
                    or any(marker in name.lower() for marker in ("cascadia", "r2-map", "r2_map"))
                    or not (
                        member.isfile()
                        or member.isdir()
                        or member.issym()
                        or member.islnk()
                    )
                ):
                    raise D0Error(f"{label} payload member differs")
                names.add(name)
                if member.isfile():
                    regular.add(name)
    except (OSError, tarfile.TarError) as error:
        raise D0Error(f"{label} payload is not a safe tar archive") from error
    if required_regular not in regular:
        raise D0Error(f"{label} required executable is absent")
    projection = {
        "entry_count": len(names),
        "regular_file_count": len(regular),
        "required_regular": required_regular,
        "project_paths": [],
        "uncompressed_size": len(payload),
        "uncompressed_sha256": sha256_bytes(payload),
    }
    projection["projection_sha256"] = document_sha256(projection, "projection_sha256")
    return projection


def _scanner_local_index() -> bytes:
    frozen = SCANNER_IMAGE
    return canonical_json(
        {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.index.v1+json",
            "manifests": [
                {
                    "mediaType": "application/vnd.oci.image.manifest.v1+json",
                    "digest": frozen["manifest_digest"],
                    "size": frozen["manifest_size"],
                    "platform": {"architecture": "arm64", "os": "linux"},
                    "annotations": {
                        "org.opencontainers.image.ref.name": (
                            "docker/buildkit-syft-scanner:stable-1"
                        )
                    },
                },
                {
                    "mediaType": "application/vnd.oci.image.manifest.v1+json",
                    "digest": frozen["attestation_manifest_digest"],
                    "size": frozen["attestation_manifest_size"],
                    "platform": {"architecture": "unknown", "os": "unknown"},
                    "annotations": {
                        "vnd.docker.reference.digest": frozen["manifest_digest"],
                        "vnd.docker.reference.type": "attestation-manifest",
                    },
                },
            ],
        }
    )


def validate_scanner_objects(
    source_index: bytes,
    manifest: bytes,
    config: bytes,
    layer: bytes,
    attestation_manifest: bytes,
    spdx: bytes,
    provenance: bytes,
    *,
    source_index_identity: bool = True,
) -> None:
    frozen = SCANNER_IMAGE
    objects = [
        (manifest, frozen["manifest_digest"], frozen["manifest_size"], "manifest"),
        (config, frozen["config_digest"], frozen["config_size"], "config"),
        (layer, frozen["layer_digest"], frozen["layer_size"], "layer"),
        (
            attestation_manifest,
            frozen["attestation_manifest_digest"],
            frozen["attestation_manifest_size"],
            "attestation manifest",
        ),
        (spdx, frozen["spdx_digest"], frozen["spdx_size"], "SPDX attestation"),
        (
            provenance,
            frozen["provenance_digest"],
            frozen["provenance_size"],
            "provenance attestation",
        ),
    ]
    if source_index_identity:
        objects.insert(0, (source_index, frozen["index_digest"], frozen["index_size"], "index"))
    for payload, digest, size, label in objects:
        if len(payload) != size or f"sha256:{sha256_bytes(payload)}" != digest:
            raise D0Error(f"BuildKit scanner {label} identity differs")
    source = _json_object(source_index, "BuildKit scanner source index")
    descriptors = source.get("manifests")
    if not isinstance(descriptors, list):
        raise D0Error("BuildKit scanner index descriptors are absent")
    expected_descriptors = {
        frozen["manifest_digest"]: {"architecture": "arm64", "os": "linux"},
        frozen["attestation_manifest_digest"]: {
            "architecture": "unknown",
            "os": "unknown",
        },
    }
    for digest, platform in expected_descriptors.items():
        matches = [
            item
            for item in descriptors
            if isinstance(item, dict)
            and item.get("digest") == digest
            and item.get("platform") == platform
        ]
        if len(matches) != 1:
            raise D0Error("BuildKit scanner source index graph differs")
    manifest_value = _json_object(manifest, "BuildKit scanner manifest")
    if manifest_value.get("config") != {
        "mediaType": "application/vnd.oci.image.config.v1+json",
        "digest": frozen["config_digest"],
        "size": frozen["config_size"],
    } or manifest_value.get("layers") != [
        {
            "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
            "digest": frozen["layer_digest"],
            "size": frozen["layer_size"],
        }
    ]:
        raise D0Error("BuildKit scanner image descriptor graph differs")
    config_value = _json_object(config, "BuildKit scanner config")
    try:
        uncompressed = gzip.decompress(layer)
    except (OSError, EOFError) as error:
        raise D0Error("BuildKit scanner layer cannot be decompressed") from error
    if (
        config_value.get("architecture") != "arm64"
        or config_value.get("os") != "linux"
        or config_value.get("config", {}).get("Entrypoint") != ["/bin/syft-scanner"]
        or config_value.get("rootfs") != {"type": "layers", "diff_ids": [frozen["diff_id"]]}
        or f"sha256:{sha256_bytes(uncompressed)}" != frozen["diff_id"]
    ):
        raise D0Error("BuildKit scanner runtime config or diff ID differs")
    attestation = _json_object(attestation_manifest, "BuildKit scanner attestation manifest")
    if (
        attestation.get("artifactType") != "application/vnd.docker.attestation.manifest.v1+json"
        or attestation.get("subject")
        != {
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "digest": frozen["manifest_digest"],
            "size": frozen["manifest_size"],
        }
        or attestation.get("config", {}).get("digest") != frozen["attestation_config_digest"]
        or [item.get("digest") for item in attestation.get("layers", [])]
        != [frozen["spdx_digest"], frozen["provenance_digest"]]
    ):
        raise D0Error("BuildKit scanner attestation descriptor graph differs")
    statements = (
        (_json_object(spdx, "BuildKit scanner SPDX"), "https://spdx.dev/Document"),
        (
            _json_object(provenance, "BuildKit scanner provenance"),
            "https://slsa.dev/provenance/v1",
        ),
    )
    for statement, predicate in statements:
        subjects = statement.get("subject")
        if (
            statement.get("predicateType") != predicate
            or not isinstance(subjects, list)
            or not any(
                isinstance(subject, dict)
                and subject.get("digest", {}).get("sha256")
                == frozen["manifest_digest"].split(":", 1)[1]
                for subject in subjects
            )
        ):
            raise D0Error("BuildKit scanner attestation subject or predicate differs")
    resolved = (
        statements[1][0]
        .get("predicate", {})
        .get("buildDefinition", {})
        .get("resolvedDependencies", [])
    )
    if not any(
        isinstance(item, dict)
        and item.get("uri")
        == "https://github.com/docker/buildkit-syft-scanner.git#refs/tags/v1.11.0"
        and item.get("digest", {}).get("sha1") == frozen["source_revision"]
        for item in resolved
    ):
        raise D0Error("BuildKit scanner provenance source revision differs")


def scanner_oci_archive(
    source_index: bytes,
    manifest: bytes,
    config: bytes,
    layer: bytes,
    attestation_manifest: bytes,
    spdx: bytes,
    provenance: bytes,
) -> tuple[bytes, dict[str, Any]]:
    validate_scanner_objects(
        source_index,
        manifest,
        config,
        layer,
        attestation_manifest,
        spdx,
        provenance,
    )
    frozen = SCANNER_IMAGE
    layout = canonical_json({"imageLayoutVersion": "1.0.0"})
    local_index = _scanner_local_index()
    payloads = {
        frozen["manifest_digest"]: manifest,
        frozen["config_digest"]: config,
        frozen["layer_digest"]: layer,
        frozen["attestation_manifest_digest"]: attestation_manifest,
        frozen["attestation_config_digest"]: b"{}",
        frozen["spdx_digest"]: spdx,
        frozen["provenance_digest"]: provenance,
    }
    files = {
        "oci-layout": (layout, 0o444),
        "index.json": (local_index, 0o444),
        **{
            f"blobs/sha256/{digest.split(':', 1)[1]}": (payload, 0o444)
            for digest, payload in payloads.items()
        },
    }
    archive = _tar_bytes(files)
    receipt = {
        "schema_id": "cascadia.r2-map.d0-scanner-oci-receipt.v1",
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "source_index_digest": frozen["index_digest"],
        "manifest_digest": frozen["manifest_digest"],
        "attestation_manifest_digest": frozen["attestation_manifest_digest"],
        "config_digest": frozen["config_digest"],
        "layer_digest": frozen["layer_digest"],
        "spdx_digest": frozen["spdx_digest"],
        "provenance_digest": frozen["provenance_digest"],
        "source_revision": frozen["source_revision"],
        "license": frozen["license"],
        "archive_size": len(archive),
        "archive_sha256": sha256_bytes(archive),
        "local_index_sha256": sha256_bytes(local_index),
        "status": "pass",
    }
    return archive, receipt


def verify_scanner_oci_archive(archive: bytes) -> dict[str, Any]:
    if not archive or len(archive) > 128 * 1024 * 1024:
        raise D0Error("BuildKit scanner OCI archive size differs")
    files: dict[str, tuple[bytes, int]] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as source:
            for item in source:
                if (
                    not item.isfile()
                    or item.name.startswith("/")
                    or ".." in item.name.split("/")
                    or item.name in files
                    or item.uid != 0
                    or item.gid != 0
                    or item.mtime != 0
                    or item.mode != 0o444
                ):
                    raise D0Error("BuildKit scanner OCI member metadata differs")
                stream = source.extractfile(item)
                if stream is None:
                    raise D0Error("BuildKit scanner OCI member is unreadable")
                payload = stream.read(128 * 1024 * 1024 + 1)
                if len(payload) != item.size:
                    raise D0Error("BuildKit scanner OCI member size differs")
                files[item.name] = (payload, 0o444)
    except (OSError, tarfile.TarError) as error:
        raise D0Error("BuildKit scanner OCI archive is invalid") from error
    frozen = SCANNER_IMAGE
    expected_blobs = {
        frozen["manifest_digest"],
        frozen["config_digest"],
        frozen["layer_digest"],
        frozen["attestation_manifest_digest"],
        frozen["attestation_config_digest"],
        frozen["spdx_digest"],
        frozen["provenance_digest"],
    }
    expected_names = {"oci-layout", "index.json"} | {
        f"blobs/sha256/{digest.split(':', 1)[1]}" for digest in expected_blobs
    }
    if set(files) != expected_names:
        raise D0Error("BuildKit scanner OCI graph differs")

    def blob(digest: str) -> bytes:
        return files[f"blobs/sha256/{digest.split(':', 1)[1]}"][0]

    if (
        files["oci-layout"][0] != canonical_json({"imageLayoutVersion": "1.0.0"})
        or files["index.json"][0] != _scanner_local_index()
        or blob(frozen["attestation_config_digest"]) != b"{}"
    ):
        raise D0Error("BuildKit scanner OCI layout or index differs")
    for payload, digest, size in (
        (blob(frozen["manifest_digest"]), frozen["manifest_digest"], frozen["manifest_size"]),
        (blob(frozen["config_digest"]), frozen["config_digest"], frozen["config_size"]),
        (blob(frozen["layer_digest"]), frozen["layer_digest"], frozen["layer_size"]),
        (
            blob(frozen["attestation_manifest_digest"]),
            frozen["attestation_manifest_digest"],
            frozen["attestation_manifest_size"],
        ),
        (blob(frozen["spdx_digest"]), frozen["spdx_digest"], frozen["spdx_size"]),
        (
            blob(frozen["provenance_digest"]),
            frozen["provenance_digest"],
            frozen["provenance_size"],
        ),
    ):
        if len(payload) != size or f"sha256:{sha256_bytes(payload)}" != digest:
            raise D0Error("BuildKit scanner OCI blob identity differs")
    # Reuse the graph/config/attestation checks with the frozen source-index
    # digest bypassed by substituting its descriptor checks below.
    manifest = blob(frozen["manifest_digest"])
    config = blob(frozen["config_digest"])
    layer = blob(frozen["layer_digest"])
    attestation_manifest = blob(frozen["attestation_manifest_digest"])
    spdx = blob(frozen["spdx_digest"])
    provenance = blob(frozen["provenance_digest"])
    local = _json_object(files["index.json"][0], "BuildKit scanner local index")
    local_source = canonical_json(
        {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.index.v1+json",
            "manifests": local["manifests"],
        }
    )
    validate_scanner_objects(
        local_source,
        manifest,
        config,
        layer,
        attestation_manifest,
        spdx,
        provenance,
        source_index_identity=False,
    )
    try:
        uncompressed_layer = gzip.decompress(layer)
    except (OSError, EOFError) as error:
        raise D0Error("BuildKit scanner layer cannot be decompressed") from error
    layer_payload = _layer_payload_projection(
        uncompressed_layer,
        label="BuildKit scanner layer",
        required_regular="bin/syft-scanner",
    )
    if _tar_bytes(files) != archive:
        raise D0Error("BuildKit scanner OCI archive is not canonical deterministic USTAR")
    return {
        "archive_size": len(archive),
        "archive_sha256": sha256_bytes(archive),
        "manifest_digest": frozen["manifest_digest"],
        "manifest_size": frozen["manifest_size"],
        "config_digest": frozen["config_digest"],
        "config_size": frozen["config_size"],
        "layer_digest": frozen["layer_digest"],
        "layer_size": frozen["layer_size"],
        "diff_id": frozen["diff_id"],
        "layer_payload": layer_payload,
        "attestation_config_digest": frozen["attestation_config_digest"],
        "attestation_manifest_digest": frozen["attestation_manifest_digest"],
        "spdx_digest": frozen["spdx_digest"],
        "provenance_digest": frozen["provenance_digest"],
        "reference": "docker/buildkit-syft-scanner:stable-1",
    }


class RegistryClient:
    """Minimal digest-pinned OCI registry client used only for Alpine smoke bytes."""

    def __init__(self, opener: Callable[[urllib.request.Request], bytes] | None = None):
        self._opener = opener or self._open

    @staticmethod
    def _open(request: urllib.request.Request) -> bytes:
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                length = response.headers.get("Content-Length")
                if length is not None and int(length) > MAX_HTTP_BYTES:
                    raise D0Error("registry object exceeds its byte bound")
                value = response.read(MAX_HTTP_BYTES + 1)
        except OSError as error:
            raise D0Error("registry request failed") from error
        if len(value) > MAX_HTTP_BYTES:
            raise D0Error("registry object exceeds its byte bound")
        return value

    def token(self) -> str:
        query = urllib.parse.urlencode(
            {"service": "registry.docker.io", "scope": "repository:library/alpine:pull"}
        )
        value = self._opener(urllib.request.Request(f"{AUTH}?{query}"))
        token = _json_object(value, "registry token").get("token")
        if not isinstance(token, str) or not token or len(token) > 8192:
            raise D0Error("registry token differs")
        return token

    def get(self, relative: str, *, accept: str, token: str) -> bytes:
        if not relative.startswith("/v2/library/alpine/") or ".." in relative:
            raise D0Error("registry path is outside the frozen repository")
        request = urllib.request.Request(
            f"{REGISTRY}{relative}",
            headers={"Authorization": f"Bearer {token}", "Accept": accept},
        )
        return self._opener(request)

    def acquire(self) -> tuple[bytes, bytes, bytes, bytes]:
        token = self.token()
        frozen = SMOKE_IMAGE
        source_index = self.get(
            f"/v2/library/alpine/manifests/{frozen['index_digest']}",
            accept=ALPINE_ACCEPT_INDEX,
            token=token,
        )
        manifest = self.get(
            f"/v2/library/alpine/manifests/{frozen['manifest_digest']}",
            accept=ALPINE_ACCEPT_MANIFEST,
            token=token,
        )
        config = self.get(
            f"/v2/library/alpine/blobs/{frozen['config_digest']}",
            accept="application/octet-stream",
            token=token,
        )
        layer = self.get(
            f"/v2/library/alpine/blobs/{frozen['layer_digest']}",
            accept="application/octet-stream",
            token=token,
        )
        validate_alpine_objects(source_index, manifest, config, layer)
        return source_index, manifest, config, layer


class ScannerRegistryClient:
    """Digest-only client for the reviewed BuildKit Syft scanner graph."""

    def __init__(self, opener: Callable[[urllib.request.Request], bytes] | None = None):
        self._opener = opener or RegistryClient._open

    def token(self) -> str:
        query = urllib.parse.urlencode(
            {
                "service": "registry.docker.io",
                "scope": "repository:docker/buildkit-syft-scanner:pull",
            }
        )
        value = self._opener(urllib.request.Request(f"{AUTH}?{query}"))
        token = _json_object(value, "scanner registry token").get("token")
        if not isinstance(token, str) or not token or len(token) > 8192:
            raise D0Error("scanner registry token differs")
        return token

    def get(self, kind: str, digest: str, *, accept: str, token: str) -> bytes:
        if kind not in {"manifests", "blobs"} or not digest.startswith("sha256:"):
            raise D0Error("scanner registry object path differs")
        request = urllib.request.Request(
            f"{REGISTRY}/v2/docker/buildkit-syft-scanner/{kind}/{digest}",
            headers={"Authorization": f"Bearer {token}", "Accept": accept},
        )
        return self._opener(request)

    def acquire(self) -> tuple[bytes, bytes, bytes, bytes, bytes, bytes, bytes]:
        token = self.token()
        frozen = SCANNER_IMAGE
        source_index = self.get(
            "manifests",
            frozen["index_digest"],
            accept=ALPINE_ACCEPT_INDEX,
            token=token,
        )
        manifest = self.get(
            "manifests",
            frozen["manifest_digest"],
            accept=ALPINE_ACCEPT_MANIFEST,
            token=token,
        )
        attestation = self.get(
            "manifests",
            frozen["attestation_manifest_digest"],
            accept=ALPINE_ACCEPT_MANIFEST,
            token=token,
        )
        blobs = [
            self.get("blobs", frozen[key], accept="application/octet-stream", token=token)
            for key in (
                "config_digest",
                "layer_digest",
                "spdx_digest",
                "provenance_digest",
            )
        ]
        objects = (source_index, manifest, blobs[0], blobs[1], attestation, blobs[2], blobs[3])
        validate_scanner_objects(*objects)
        return objects


def _safe_parent(destination: Path) -> None:
    parent = destination.parent
    try:
        observed = parent.lstat()
    except OSError as error:
        raise D0Error("artifact destination parent is absent") from error
    if not stat.S_ISDIR(observed.st_mode) or observed.st_uid != os.getuid():
        raise D0Error("artifact destination parent is unsafe")
    current = Path("/")
    for part in parent.parts[1:]:
        current /= part
        value = current.lstat()
        if stat.S_ISLNK(value.st_mode):
            raise D0Error("artifact destination ancestor is a symlink")


def atomic_install_bytes(destination: Path, payload: bytes, *, mode: int = 0o400) -> dict[str, Any]:
    _safe_parent(destination)
    if destination.exists() or destination.is_symlink():
        try:
            observed = destination.lstat()
            descriptor = os.open(destination, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            try:
                existing = os.read(descriptor, len(payload) + 1)
            finally:
                os.close(descriptor)
        except OSError as error:
            raise D0Error("artifact destination cannot be recovered") from error
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_uid != os.getuid()
            or observed.st_nlink != 1
            or stat.S_IMODE(observed.st_mode) != mode
            or existing != payload
        ):
            raise D0Error("artifact destination already exists with different bytes or metadata")
        return {
            "path": str(destination),
            "size": len(payload),
            "sha256": sha256_bytes(payload),
            "status": "already-installed",
        }
    temporary = destination.with_name(f".{destination.name}.partial-{os.getpid()}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, mode)
    try:
        position = 0
        while position < len(payload):
            position += os.write(descriptor, payload[position:])
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor)
    try:
        storage_commit = verify_canonical_commit_boundary(destination)
        os.replace(temporary, destination)
        installed = destination.lstat()
        if (
            not stat.S_ISREG(installed.st_mode)
            or installed.st_uid != os.getuid()
            or installed.st_nlink != 1
            or stat.S_IMODE(installed.st_mode) != mode
        ):
            raise D0Error("installed artifact metadata differs")
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        # A post-rename fsync error is safely recoverable: retain only exact
        # committed bytes, otherwise remove the transaction target.
        temporary.unlink(missing_ok=True)
        try:
            if destination.read_bytes() != payload:
                destination.unlink(missing_ok=True)
        except OSError:
            destination.unlink(missing_ok=True)
        raise
    return {
        "path": str(destination),
        "size": len(payload),
        "sha256": sha256_bytes(payload),
        "physical_storage_commit": storage_commit,
        "status": "installed",
    }


def validate_scanner_source_supply(source_archive: bytes, license_bytes: bytes) -> dict[str, Any]:
    frozen = SCANNER_IMAGE
    if (
        len(source_archive) != frozen["source_archive_size"]
        or sha256_bytes(source_archive) != frozen["source_archive_sha256"]
        or len(license_bytes) != frozen["license_size"]
        or sha256_bytes(license_bytes) != frozen["license_sha256"]
    ):
        raise D0Error("BuildKit scanner source or license identity differs")
    required: dict[str, bytes] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(source_archive), mode="r:gz") as source:
            for item in source:
                if item.name.startswith("/") or ".." in item.name.split("/"):
                    raise D0Error("BuildKit scanner source archive path is unsafe")
                if (
                    item.isfile()
                    and item.name.count("/") == 1
                    and item.name.rsplit("/", 1)[-1] in {"LICENSE", "Dockerfile"}
                ):
                    stream = source.extractfile(item)
                    if stream is None:
                        raise D0Error("BuildKit scanner source member is unreadable")
                    required[item.name.rsplit("/", 1)[-1]] = stream.read(1024 * 1024 + 1)
    except (OSError, tarfile.TarError) as error:
        raise D0Error("BuildKit scanner source archive is invalid") from error
    dockerfile = required.get("Dockerfile")
    if (
        required.get("LICENSE") != license_bytes
        or not isinstance(dockerfile, bytes)
        or b"/bin/syft-scanner" not in dockerfile
        or b"Apache License, Version 2.0" not in dockerfile
    ):
        raise D0Error("BuildKit scanner source archive contract differs")
    return {
        "source_revision": frozen["source_revision"],
        "source_archive_size": len(source_archive),
        "source_archive_sha256": sha256_bytes(source_archive),
        "license": frozen["license"],
        "license_size": len(license_bytes),
        "license_sha256": sha256_bytes(license_bytes),
        "dockerfile_sha256": sha256_bytes(dockerfile),
        "status": "pass",
    }


def acquire_scanner_artifacts(
    *,
    oci_destination: Path,
    source_destination: Path,
    license_destination: Path,
    opener: Callable[[urllib.request.Request], bytes] | None = None,
) -> dict[str, Any]:
    fetch = opener or RegistryClient._open
    objects = ScannerRegistryClient(fetch).acquire()
    archive, oci_receipt = scanner_oci_archive(*objects)
    source_archive = fetch(urllib.request.Request(SCANNER_IMAGE["source_archive_url"]))
    license_bytes = fetch(urllib.request.Request(SCANNER_IMAGE["license_url"]))
    source_receipt = validate_scanner_source_supply(source_archive, license_bytes)
    for destination in (oci_destination, source_destination, license_destination):
        secure_owner_directory(destination.parent)
    return {
        "schema_id": "cascadia.r2-map.d0-scanner-supply-receipt.v1",
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "oci": oci_receipt,
        "source": source_receipt,
        "installed": {
            "oci": atomic_install_bytes(oci_destination, archive),
            "source": atomic_install_bytes(source_destination, source_archive),
            "license": atomic_install_bytes(license_destination, license_bytes),
        },
        "status": "pass",
    }


def homebrew_metadata_path(cache_root: Path, formula: str) -> Path:
    if formula not in FROZEN_RUNTIME:
        raise D0Error("formula is outside the frozen runtime")
    return cache_root / "metadata" / f"{formula}.json"


def homebrew_bottle_path(cache_root: Path, formula: str) -> Path:
    if formula not in FROZEN_RUNTIME:
        raise D0Error("formula is outside the frozen runtime")
    version = FROZEN_RUNTIME[formula]["version"]
    return cache_root / "bottles" / f"{formula}-{version}-arm64_tahoe.tar.gz"


def frozen_homebrew_formula_projection(formula: str) -> dict[str, Any]:
    if formula not in FROZEN_RUNTIME:
        raise D0Error("formula is outside the frozen runtime")
    frozen = FROZEN_RUNTIME[formula]
    bottle_url = (
        f"https://ghcr.io/v2/homebrew/core/{formula}/blobs/sha256:{frozen['bottle_sha256']}"
    )
    return {
        "schema_id": "cascadia.r2-map.d0-homebrew-formula-projection.v1",
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "formula": formula,
        "tap": "homebrew/core",
        "reviewed_tap_git_head": FROZEN_HOMEBREW_TAP_HEAD,
        "formula_path": frozen["formula_path"],
        "ruby_source_sha256": frozen["ruby_source_sha256"],
        "version": frozen["version"],
        "revision": frozen["revision"],
        "license": frozen["license"],
        "dependencies": list(frozen["dependencies"]),
        "source": {
            "url": frozen["source_url"],
            "tag": frozen["source_tag"],
            "revision": frozen["source_revision"],
            "checksum_sha256": frozen["source_checksum"],
        },
        "bottle": {
            "tag": frozen["bottle_tag"],
            "url": bottle_url,
            "size": frozen["bottle_size"],
            "sha256": frozen["bottle_sha256"],
        },
    }


def normalize_homebrew_formula_metadata(payload: bytes, formula: str) -> dict[str, Any]:
    """Project rolling official API JSON onto the immutable reviewed semantics."""

    expected = frozen_homebrew_formula_projection(formula)
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D0Error(f"official Homebrew metadata is invalid for {formula}") from error
    frozen = FROZEN_RUNTIME[formula]
    stable_source = value.get("urls", {}).get("stable", {})
    bottle = (
        value.get("bottle", {}).get("stable", {}).get("files", {}).get(frozen["bottle_tag"], {})
    )
    if (
        value.get("name") != formula
        or value.get("tap") != "homebrew/core"
        or value.get("versions", {}).get("stable") != frozen["version"]
        or value.get("revision") != frozen["revision"]
        or value.get("license") != frozen["license"]
        or tuple(value.get("dependencies", ())) != frozen["dependencies"]
        or value.get("ruby_source_path") != frozen["formula_path"]
        or value.get("ruby_source_checksum", {}).get("sha256") != frozen["ruby_source_sha256"]
        or stable_source.get("url") != frozen["source_url"]
        or stable_source.get("tag") != frozen["source_tag"]
        or stable_source.get("revision") != frozen["source_revision"]
        or stable_source.get("checksum") != frozen["source_checksum"]
        or bottle.get("url") != expected["bottle"]["url"]
        or bottle.get("sha256") != frozen["bottle_sha256"]
    ):
        raise D0Error(f"official Homebrew metadata semantics drifted for {formula}")
    return expected


def validate_homebrew_formula_projection(payload: bytes, formula: str) -> dict[str, Any]:
    expected = frozen_homebrew_formula_projection(formula)
    projection = load_canonical_json(
        payload,
        maximum=64 * 1024,
        label=f"Homebrew formula projection for {formula}",
    )
    if projection != expected:
        raise D0Error(f"Homebrew formula projection drifted for {formula}")
    return projection


def _validate_formula_metadata(payload: bytes, formula: str) -> dict[str, Any]:
    return validate_homebrew_formula_projection(payload, formula)


def acquire_homebrew_artifacts(
    cache_root: Path,
    formulas: tuple[str, ...],
    *,
    opener: Callable[[urllib.request.Request], bytes] | None = None,
) -> dict[str, Any]:
    """Acquire the complete pinned runtime closure without Homebrew resolution."""

    fetch = opener or RegistryClient._open
    metadata_receipts: list[dict[str, Any]] = []
    bottle_receipts: list[dict[str, Any]] = []
    for child in (cache_root / "metadata", cache_root / "bottles"):
        secure_owner_directory(child)
    for formula in formulas:
        frozen = FROZEN_RUNTIME[formula]
        projection = canonical_json(frozen_homebrew_formula_projection(formula))
        metadata_install = atomic_install_bytes(
            homebrew_metadata_path(cache_root, formula), projection
        )
        metadata_receipts.append(
            {
                "formula": formula,
                "source": "embedded-reviewed-semantic-projection",
                "reviewed_tap_git_head": FROZEN_HOMEBREW_TAP_HEAD,
                **metadata_install,
            }
        )

        query = urllib.parse.urlencode(
            {"service": "ghcr.io", "scope": f"repository:homebrew/core/{formula}:pull"}
        )
        token_bytes = fetch(urllib.request.Request(f"https://ghcr.io/token?{query}"))
        token = _json_object(token_bytes, "GHCR token").get("token")
        if not isinstance(token, str) or not token or len(token) > 8192:
            raise D0Error("GHCR token differs")
        digest = frozen["bottle_sha256"]
        bottle_url = f"https://ghcr.io/v2/homebrew/core/{formula}/blobs/sha256:{digest}"
        bottle = fetch(
            urllib.request.Request(
                bottle_url,
                headers={"Authorization": f"Bearer {token}", "Accept": "application/octet-stream"},
            )
        )
        if len(bottle) != frozen["bottle_size"] or sha256_bytes(bottle) != digest:
            raise D0Error(f"Homebrew bottle bytes drifted for {formula}")
        bottle_install = atomic_install_bytes(homebrew_bottle_path(cache_root, formula), bottle)
        bottle_receipts.append({"formula": formula, "url": bottle_url, **bottle_install})
    return {
        "schema_id": "cascadia.r2-map.d0-homebrew-artifact-receipt.v1",
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "formulae": list(formulas),
        "metadata": metadata_receipts,
        "bottles": bottle_receipts,
        "dependency_closure": {
            name: list(FROZEN_RUNTIME[name]["dependencies"]) for name in formulas
        },
        "status": "pass",
    }


def homebrew_closure_archive(
    cache_root: Path, formulas: tuple[str, ...]
) -> tuple[bytes, dict[str, Any]]:
    """Render the exact John3 formula-metadata and bottle closure as deterministic USTAR."""

    files: dict[str, tuple[bytes, int]] = {}
    identities: list[dict[str, Any]] = []
    for formula in formulas:
        metadata = homebrew_metadata_path(cache_root, formula).read_bytes()
        _validate_formula_metadata(metadata, formula)
        bottle = homebrew_bottle_path(cache_root, formula).read_bytes()
        frozen = FROZEN_RUNTIME[formula]
        if len(bottle) != frozen["bottle_size"] or sha256_bytes(bottle) != frozen["bottle_sha256"]:
            raise D0Error(f"Homebrew closure bottle differs for {formula}")
        for kind, payload, relative in (
            ("metadata", metadata, f"metadata/{formula}.json"),
            (
                "bottle",
                bottle,
                f"bottles/{formula}-{frozen['version']}-arm64_tahoe.tar.gz",
            ),
        ):
            files[relative] = (payload, 0o400)
            identities.append(
                {
                    "formula": formula,
                    "kind": kind,
                    "path": relative,
                    "size": len(payload),
                    "sha256": sha256_bytes(payload),
                    "mode": "0400",
                }
            )
    manifest: dict[str, Any] = {
        "schema_id": HOMEBREW_CLOSURE_SCHEMA,
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "platform": "arm64_tahoe",
        "formulae": list(formulas),
        "files": sorted(identities, key=lambda item: item["path"]),
    }
    manifest["manifest_sha256"] = document_sha256(manifest, "manifest_sha256")
    files["closure-manifest.json"] = (canonical_json(manifest), 0o400)
    archive = _tar_bytes(files)
    receipt = {
        "schema_id": HOMEBREW_CLOSURE_SCHEMA,
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "manifest_sha256": manifest["manifest_sha256"],
        "archive_size": len(archive),
        "archive_sha256": sha256_bytes(archive),
        "formulae": list(formulas),
        "status": "pass",
    }
    return archive, receipt


def verify_homebrew_closure(
    archive: bytes,
    formulas: tuple[str, ...],
) -> tuple[dict[str, bytes], dict[str, Any]]:
    if not archive or len(archive) > MAX_HTTP_BYTES:
        raise D0Error("Homebrew closure archive size differs")
    members: dict[str, bytes] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as source:
            for item in source:
                if (
                    not item.isfile()
                    or item.name.startswith("/")
                    or ".." in item.name.split("/")
                    or item.name in members
                    or item.uid != 0
                    or item.gid != 0
                    or item.mtime != 0
                    or item.mode != 0o400
                ):
                    raise D0Error("Homebrew closure member metadata differs")
                stream = source.extractfile(item)
                if stream is None:
                    raise D0Error("Homebrew closure member is unreadable")
                payload = stream.read(MAX_HTTP_BYTES + 1)
                if len(payload) != item.size:
                    raise D0Error("Homebrew closure member size differs")
                members[item.name] = payload
    except (OSError, tarfile.TarError) as error:
        raise D0Error("Homebrew closure archive is invalid") from error
    manifest_bytes = members.pop("closure-manifest.json", None)
    if manifest_bytes is None:
        raise D0Error("Homebrew closure manifest is absent")
    manifest = load_canonical_json(
        manifest_bytes, maximum=4 * 1024 * 1024, label="Homebrew closure manifest"
    )
    if (
        set(manifest)
        != {
            "schema_id",
            "schema_version",
            "campaign_id",
            "platform",
            "formulae",
            "files",
            "manifest_sha256",
        }
        or manifest["schema_id"] != HOMEBREW_CLOSURE_SCHEMA
        or manifest["schema_version"] != 1
        or manifest["campaign_id"] != CAMPAIGN_ID
        or manifest["platform"] != "arm64_tahoe"
        or manifest["formulae"] != list(formulas)
        or manifest["manifest_sha256"] != document_sha256(manifest, "manifest_sha256")
    ):
        raise D0Error("Homebrew closure manifest identity differs")
    expected_paths: set[str] = set()
    expected_contract = {
        (formula, "metadata", f"metadata/{formula}.json") for formula in formulas
    } | {
        (
            formula,
            "bottle",
            f"bottles/{formula}-{FROZEN_RUNTIME[formula]['version']}-arm64_tahoe.tar.gz",
        )
        for formula in formulas
    }
    observed_contract: set[tuple[str, str, str]] = set()
    for identity in manifest["files"]:
        if not isinstance(identity, dict) or set(identity) != {
            "formula",
            "kind",
            "path",
            "size",
            "sha256",
            "mode",
        }:
            raise D0Error("Homebrew closure file identity differs")
        formula = identity["formula"]
        payload = members.get(identity["path"])
        if (
            formula not in formulas
            or identity["kind"] not in {"metadata", "bottle"}
            or identity["mode"] != "0400"
            or payload is None
            or identity["size"] != len(payload)
            or identity["sha256"] != sha256_bytes(payload)
        ):
            raise D0Error("Homebrew closure payload differs")
        observed_contract.add((formula, identity["kind"], identity["path"]))
        if identity["kind"] == "metadata":
            _validate_formula_metadata(payload, formula)
        else:
            frozen = FROZEN_RUNTIME[formula]
            if (
                len(payload) != frozen["bottle_size"]
                or sha256_bytes(payload) != frozen["bottle_sha256"]
            ):
                raise D0Error("Homebrew closure bottle identity differs")
        expected_paths.add(identity["path"])
    if observed_contract != expected_contract or expected_paths != set(members):
        raise D0Error("Homebrew closure contains unmanifested or missing files")
    canonical = _tar_bytes(
        {
            **{name: (payload, 0o400) for name, payload in members.items()},
            "closure-manifest.json": (manifest_bytes, 0o400),
        }
    )
    if canonical != archive:
        raise D0Error("Homebrew closure is not canonical deterministic USTAR")
    return members, {
        "archive_size": len(archive),
        "archive_sha256": sha256_bytes(archive),
        "manifest_sha256": manifest["manifest_sha256"],
        "formulae": list(formulas),
        "status": "pass",
    }


def install_homebrew_closure(
    archive: bytes,
    cache_root: Path,
    formulas: tuple[str, ...],
) -> dict[str, Any]:
    members, receipt = verify_homebrew_closure(archive, formulas)
    expected = {cache_root / relative: payload for relative, payload in members.items()}
    if cache_root.exists() or cache_root.is_symlink():
        if cache_root.is_symlink() or not cache_root.is_dir():
            raise D0Error("Homebrew closure destination root is unsafe")
        observed = {
            path: path.read_bytes()
            for path in cache_root.rglob("*")
            if path.is_file() and not path.is_symlink()
        }
        if set(observed) != set(expected) or any(
            observed[path] != payload
            or stat.S_IMODE(path.lstat().st_mode) != 0o400
            or path.lstat().st_uid != os.getuid()
            or path.lstat().st_nlink != 1
            for path, payload in expected.items()
        ):
            raise D0Error("Homebrew closure destination differs from the verified archive")
        return {
            "verification": receipt,
            "installed": [
                {
                    "path": str(path),
                    "size": len(payload),
                    "sha256": sha256_bytes(payload),
                    "status": "already-installed",
                }
                for path, payload in sorted(expected.items())
            ],
            "transaction": "replayed-exact",
            "status": "pass",
        }
    secure_owner_directory(cache_root.parent)
    staging = cache_root.with_name(f".{cache_root.name}.partial-{os.getpid()}")
    if staging.exists() or staging.is_symlink():
        raise D0Error("Homebrew closure transaction staging path already exists")
    staging.mkdir(mode=0o700)
    installed: list[dict[str, Any]] = []
    committed = False
    try:
        for relative, payload in sorted(members.items()):
            destination = staging / relative
            secure_owner_directory(destination.parent)
            installed.append(atomic_install_bytes(destination, payload))
        for directory in sorted(
            (path for path in staging.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            descriptor = os.open(
                directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            )
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        descriptor = os.open(staging, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(staging, cache_root)
        committed = True
        descriptor = os.open(
            cache_root.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if not committed:
            shutil.rmtree(staging, ignore_errors=True)
    return {
        "verification": receipt,
        "installed": [
            {**item, "path": str(cache_root / Path(item["path"]).relative_to(staging))}
            for item in installed
        ],
        "transaction": "atomically-committed",
        "status": "pass",
    }


def runtime_supply_archive(
    core_image: bytes,
    smoke_oci: bytes,
    homebrew_closure: bytes,
    formulas: tuple[str, ...],
) -> tuple[bytes, dict[str, Any]]:
    """Render the one deterministic, worker-role runtime supply rendezvous object."""

    if (
        len(core_image) != CORE_IMAGE["size"]
        or sha256_bytes(core_image) != CORE_IMAGE["sha256"]
    ):
        raise D0Error("runtime supply Colima core identity differs")
    smoke = verify_smoke_oci_archive(smoke_oci)
    _closure_members, closure = verify_homebrew_closure(homebrew_closure, formulas)
    components = [
        {
            "kind": "colima-core",
            "path": "components/colima-core.raw.gz",
            "target_key": "core_image",
            "size": len(core_image),
            "sha256": sha256_bytes(core_image),
            "mode": "0400",
        },
        {
            "kind": "smoke-oci",
            "path": "components/alpine.oci.tar",
            "target_key": "smoke_oci",
            "size": len(smoke_oci),
            "sha256": sha256_bytes(smoke_oci),
            "mode": "0400",
        },
        {
            "kind": "homebrew-closure",
            "path": "components/homebrew-closure.tar",
            "target_key": "homebrew_closure",
            "size": len(homebrew_closure),
            "sha256": sha256_bytes(homebrew_closure),
            "mode": "0400",
        },
    ]
    manifest: dict[str, Any] = {
        "schema_id": RUNTIME_SUPPLY_SCHEMA,
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "target_role": "worker",
        "platform": "darwin-arm64-tahoe",
        "formulae": list(formulas),
        "components": components,
        "smoke_verification_sha256": sha256_bytes(canonical_json(smoke)),
        "homebrew_closure_manifest_sha256": closure["manifest_sha256"],
        "project_code_present": False,
        "protected_seed_values_opened": False,
    }
    manifest["manifest_sha256"] = document_sha256(manifest, "manifest_sha256")
    manifest_bytes = canonical_json(manifest)
    archive = _tar_bytes(
        {
            "components/colima-core.raw.gz": (core_image, 0o400),
            "components/alpine.oci.tar": (smoke_oci, 0o400),
            "components/homebrew-closure.tar": (homebrew_closure, 0o400),
            "supply-manifest.json": (manifest_bytes, 0o400),
        }
    )
    return archive, {
        "schema_id": RUNTIME_SUPPLY_SCHEMA,
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "archive_size": len(archive),
        "archive_sha256": sha256_bytes(archive),
        "manifest_sha256": manifest["manifest_sha256"],
        "formulae": list(formulas),
        "status": "pass",
    }


def verify_runtime_supply_archive(
    archive: bytes,
    formulas: tuple[str, ...],
) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Authenticate every byte and semantic edge before a worker write occurs."""

    if not archive or len(archive) > MAX_HTTP_BYTES:
        raise D0Error("runtime supply archive size differs")
    members: dict[str, bytes] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as source:
            for item in source:
                if (
                    not item.isfile()
                    or item.name.startswith("/")
                    or ".." in item.name.split("/")
                    or item.name in members
                    or item.uid != 0
                    or item.gid != 0
                    or item.mtime != 0
                    or item.mode != 0o400
                ):
                    raise D0Error("runtime supply member metadata differs")
                stream = source.extractfile(item)
                if stream is None:
                    raise D0Error("runtime supply member is unreadable")
                payload = stream.read(MAX_HTTP_BYTES + 1)
                if len(payload) != item.size:
                    raise D0Error("runtime supply member size differs")
                members[item.name] = payload
    except (OSError, tarfile.TarError) as error:
        raise D0Error("runtime supply archive is invalid") from error
    expected_names = {
        "components/colima-core.raw.gz",
        "components/alpine.oci.tar",
        "components/homebrew-closure.tar",
        "supply-manifest.json",
    }
    if set(members) != expected_names:
        raise D0Error("runtime supply archive member set differs")
    manifest_bytes = members["supply-manifest.json"]
    manifest = load_canonical_json(
        manifest_bytes,
        maximum=1024 * 1024,
        label="runtime supply manifest",
    )
    expected_fields = {
        "schema_id",
        "schema_version",
        "campaign_id",
        "target_role",
        "platform",
        "formulae",
        "components",
        "smoke_verification_sha256",
        "homebrew_closure_manifest_sha256",
        "project_code_present",
        "protected_seed_values_opened",
        "manifest_sha256",
    }
    if (
        not isinstance(manifest, dict)
        or set(manifest) != expected_fields
        or manifest["schema_id"] != RUNTIME_SUPPLY_SCHEMA
        or manifest["schema_version"] != 1
        or manifest["campaign_id"] != CAMPAIGN_ID
        or manifest["target_role"] != "worker"
        or manifest["platform"] != "darwin-arm64-tahoe"
        or manifest["formulae"] != list(formulas)
        or manifest["project_code_present"] is not False
        or manifest["protected_seed_values_opened"] is not False
        or manifest["manifest_sha256"] != document_sha256(manifest, "manifest_sha256")
    ):
        raise D0Error("runtime supply manifest identity differs")
    core = members["components/colima-core.raw.gz"]
    smoke_oci = members["components/alpine.oci.tar"]
    closure_archive = members["components/homebrew-closure.tar"]
    if len(core) != CORE_IMAGE["size"] or sha256_bytes(core) != CORE_IMAGE["sha256"]:
        raise D0Error("runtime supply Colima core identity differs")
    smoke = verify_smoke_oci_archive(smoke_oci)
    _closure_members, closure = verify_homebrew_closure(closure_archive, formulas)
    expected_components = [
        {
            "kind": "colima-core",
            "path": "components/colima-core.raw.gz",
            "target_key": "core_image",
            "size": len(core),
            "sha256": sha256_bytes(core),
            "mode": "0400",
        },
        {
            "kind": "smoke-oci",
            "path": "components/alpine.oci.tar",
            "target_key": "smoke_oci",
            "size": len(smoke_oci),
            "sha256": sha256_bytes(smoke_oci),
            "mode": "0400",
        },
        {
            "kind": "homebrew-closure",
            "path": "components/homebrew-closure.tar",
            "target_key": "homebrew_closure",
            "size": len(closure_archive),
            "sha256": sha256_bytes(closure_archive),
            "mode": "0400",
        },
    ]
    if (
        manifest["components"] != expected_components
        or manifest["smoke_verification_sha256"] != sha256_bytes(canonical_json(smoke))
        or manifest["homebrew_closure_manifest_sha256"] != closure["manifest_sha256"]
    ):
        raise D0Error("runtime supply component graph differs")
    canonical = _tar_bytes({name: (payload, 0o400) for name, payload in members.items()})
    if canonical != archive:
        raise D0Error("runtime supply archive is not canonical deterministic USTAR")
    return members, {
        "schema_id": RUNTIME_SUPPLY_SCHEMA,
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "archive_size": len(archive),
        "archive_sha256": sha256_bytes(archive),
        "manifest_sha256": manifest["manifest_sha256"],
        "components": expected_components,
        "smoke": smoke,
        "homebrew_closure": closure,
        "status": "pass",
    }


def install_runtime_supply_archive(
    archive: bytes,
    *,
    runtime_supply_path: Path,
    core_path: Path,
    smoke_path: Path,
    homebrew_closure_path: Path,
    formulas: tuple[str, ...],
) -> dict[str, Any]:
    """Materialize a verified supply atomically into one worker's isolated roots."""

    members, verification = verify_runtime_supply_archive(archive, formulas)
    targets = {
        runtime_supply_path: archive,
        core_path: members["components/colima-core.raw.gz"],
        smoke_path: members["components/alpine.oci.tar"],
        homebrew_closure_path: members["components/homebrew-closure.tar"],
    }
    parents = {path.parent for path in targets}
    if len(parents) != 1:
        raise D0Error("runtime supply targets do not share one atomic staging root")
    destination_root = parents.pop()
    if destination_root.exists() or destination_root.is_symlink():
        if destination_root.is_symlink() or not destination_root.is_dir():
            raise D0Error("runtime supply destination root is unsafe")
        observed = {
            path: path.read_bytes()
            for path in destination_root.iterdir()
            if path.is_file() and not path.is_symlink()
        }
        if set(observed) != set(targets) or any(
            observed[path] != payload
            or stat.S_IMODE(path.lstat().st_mode) != 0o400
            or path.lstat().st_uid != os.getuid()
            or path.lstat().st_nlink != 1
            for path, payload in targets.items()
        ):
            raise D0Error("runtime supply destination differs from the verified archive")
        return {
            "verification": verification,
            "installed": [
                {
                    "path": str(path),
                    "size": len(payload),
                    "sha256": sha256_bytes(payload),
                    "status": "already-installed",
                }
                for path, payload in sorted(targets.items())
            ],
            "transaction": "replayed-exact",
            "status": "pass",
        }
    secure_owner_directory(destination_root.parent)
    staging = destination_root.with_name(f".{destination_root.name}.partial-{os.getpid()}")
    if staging.exists() or staging.is_symlink():
        raise D0Error("runtime supply transaction staging path already exists")
    staging.mkdir(mode=0o700)
    committed = False
    installed: list[dict[str, Any]] = []
    try:
        for destination, payload in sorted(targets.items()):
            staged = staging / destination.name
            installed.append(atomic_install_bytes(staged, payload))
        descriptor = os.open(staging, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(staging, destination_root)
        committed = True
        descriptor = os.open(
            destination_root.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if not committed:
            shutil.rmtree(staging, ignore_errors=True)
    return {
        "verification": verification,
        "installed": [
            {**item, "path": str(destination_root / Path(item["path"]).name)}
            for item in installed
        ],
        "transaction": "atomically-committed",
        "status": "pass",
    }


def acquire_core(
    destination: Path, *, opener: Callable[[urllib.request.Request], bytes] | None = None
) -> dict[str, Any]:
    fetch = opener or RegistryClient._open
    request = urllib.request.Request(CORE_IMAGE["url"])
    payload = fetch(request)
    if (
        len(payload) != CORE_IMAGE["size"]
        or sha256_bytes(payload) != CORE_IMAGE["sha256"]
        or hashlib.sha512(payload).hexdigest() != CORE_IMAGE["sha512"]
    ):
        raise D0Error("Colima core image identity differs")
    installed = atomic_install_bytes(destination, payload)
    receipt: dict[str, Any] = {
        "schema_id": ARTIFACT_RECEIPT_SCHEMA,
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "kind": "colima-core",
        "url": CORE_IMAGE["url"],
        "size": len(payload),
        "sha256": CORE_IMAGE["sha256"],
        "sha512": CORE_IMAGE["sha512"],
        "destination": installed["path"],
        "acquired_unix_ms": time.time_ns() // 1_000_000,
    }
    receipt["receipt_sha256"] = document_sha256(receipt, "receipt_sha256")
    return receipt
