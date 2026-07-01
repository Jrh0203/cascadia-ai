#!/usr/bin/env python3
"""Build once on john1, push once, and return an immutable research image digest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import blake3

REGISTRY = "100.110.109.6:5000"
DEFAULT_DOCKER_HOST = (
    "unix:///Users/johnherrick/.local/share/cascadia-r2/colima/cascadia-r2/docker.sock"
)
DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
NON_SOURCE_PATHS = {Path("STATE.md")}


def _run(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def _run_bytes(*args: str, cwd: Path | None = None) -> bytes:
    return subprocess.check_output(args, cwd=cwd)


def _workspace_source_identity(context: Path) -> dict[str, object]:
    raw = _run_bytes(
        "git",
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "-z",
        cwd=context,
    )
    relative_paths = sorted(
        {
            path
            for value in raw.split(b"\0")
            if value
            for path in (Path(value.decode()),)
            if path not in NON_SOURCE_PATHS
        }
    )
    hasher = blake3.blake3(b"cascadia-cluster-build-workspace-v1")
    files = 0
    total_bytes = 0
    for relative in relative_paths:
        path = context / relative
        if path.is_symlink():
            kind = b"symlink"
            payload = os.readlink(path).encode()
        elif path.is_file():
            kind = b"file"
            payload = path.read_bytes()
        else:
            continue
        encoded_path = relative.as_posix().encode()
        hasher.update(len(encoded_path).to_bytes(8, "little"))
        hasher.update(encoded_path)
        hasher.update(len(kind).to_bytes(8, "little"))
        hasher.update(kind)
        hasher.update(len(payload).to_bytes(8, "little"))
        hasher.update(payload)
        files += 1
        total_bytes += len(payload)
    status = _run_bytes(
        "git",
        "status",
        "--porcelain=v1",
        "-z",
        "--",
        ".",
        ":(exclude)STATE.md",
        cwd=context,
    )
    return {
        "schema_id": "cascadia.cluster.build-workspace-identity.v1",
        "git_revision": _run("git", "rev-parse", "HEAD", cwd=context),
        "git_dirty": bool(status),
        "git_status_blake3": blake3.blake3(status).hexdigest(),
        "workspace_blake3": hasher.hexdigest(),
        "files": files,
        "bytes": total_bytes,
    }


def _write_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _manifest_push_digest(output: str) -> str:
    matches = [line.strip() for line in output.splitlines() if DIGEST.fullmatch(line.strip())]
    if len(matches) != 1:
        raise SystemExit(f"manifest-list digest could not be resolved uniquely: {matches}")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=Path, default=Path.cwd())
    parser.add_argument("--dockerfile", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--build-arg", action="append", default=[])
    parser.add_argument(
        "--docker-host", default=os.environ.get("DOCKER_HOST", DEFAULT_DOCKER_HOST)
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", args.name):
        raise SystemExit("image name must be a portable lowercase registry component")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.tag):
        raise SystemExit("image tag is not portable")
    tailscale_ips = _run("/opt/homebrew/bin/tailscale", "ip", "-4").splitlines()
    if "100.110.109.6" not in tailscale_ips:
        raise SystemExit("research images may only be built and pushed from john1")
    context = args.context.resolve(strict=True)
    dockerfile = args.dockerfile.resolve(strict=True)
    os.environ["DOCKER_HOST"] = args.docker_host
    source_identity = _workspace_source_identity(context)
    reference = f"{REGISTRY}/cascadia/{args.name}:{args.tag}"
    command = [
        "/opt/homebrew/bin/docker",
        "build",
        "--platform",
        "linux/arm64",
        "--pull",
        "--file",
        str(dockerfile),
        "--tag",
        reference,
    ]
    build_args = list(args.build_arg)
    automatic_build_args = {
        "SOURCE_REVISION": str(source_identity["git_revision"]),
        "SOURCE_BLAKE3": str(source_identity["workspace_blake3"]),
        "IMAGE_TAG": args.tag,
    }
    explicit_names = {value.partition("=")[0] for value in build_args}
    reserved = explicit_names.intersection(automatic_build_args)
    if reserved:
        raise SystemExit(f"source identity build arguments are reserved: {sorted(reserved)}")
    build_args.extend(f"{name}={value}" for name, value in automatic_build_args.items())
    for value in build_args:
        command.extend(("--build-arg", value))
    command.append(str(context))
    subprocess.run(command, check=True)
    if _workspace_source_identity(context) != source_identity:
        raise SystemExit("John1 workspace changed during the canonical image build")
    labels = json.loads(
        _run(
            "/opt/homebrew/bin/docker",
            "image",
            "inspect",
            reference,
            "--format",
            "{{json .Config.Labels}}",
        )
    )
    expected_labels = {
        "org.opencontainers.image.revision": source_identity["git_revision"],
        "org.opencontainers.image.source-blake3": source_identity["workspace_blake3"],
        "org.opencontainers.image.version": args.tag,
    }
    if any(labels.get(name) != value for name, value in expected_labels.items()):
        raise SystemExit("built image labels do not match the frozen source identity")
    subprocess.run(["/opt/homebrew/bin/docker", "push", reference], check=True)
    repo_digests = json.loads(
        _run(
            "/opt/homebrew/bin/docker",
            "image",
            "inspect",
            reference,
            "--format",
            "{{json .RepoDigests}}",
        )
    )
    matches = [value for value in repo_digests if value.startswith(f"{REGISTRY}/cascadia/")]
    if len(matches) != 1:
        raise SystemExit(f"pushed digest could not be resolved uniquely: {matches}")
    member = matches[0]
    member_digest = member.rsplit("@", 1)[-1]
    if not DIGEST.fullmatch(member_digest):
        raise SystemExit("registry returned a malformed digest")
    # Bacalhau resolves schedulable platforms from an image index. A bare
    # single-platform manifest has its architecture only in the config blob,
    # which v1.9 reports as an empty platform set to cold workers. Publish a
    # one-member index and use the index digest as the scheduler identity.
    subprocess.run(
        [
            "/opt/homebrew/bin/docker",
            "manifest",
            "create",
            "--insecure",
            reference,
            member,
        ],
        check=True,
    )
    subprocess.run(
        [
            "/opt/homebrew/bin/docker",
            "manifest",
            "annotate",
            reference,
            member,
            "--os",
            "linux",
            "--arch",
            "arm64",
        ],
        check=True,
    )
    index_digest = _manifest_push_digest(
        _run(
            "/opt/homebrew/bin/docker",
            "manifest",
            "push",
            "--insecure",
            "--purge",
            reference,
        )
    )
    index = json.loads(
        _run(
            "/opt/homebrew/bin/docker",
            "manifest",
            "inspect",
            "--insecure",
            reference,
        )
    )
    manifests = index.get("manifests")
    if (
        not isinstance(manifests, list)
        or len(manifests) != 1
        or manifests[0].get("digest") != member_digest
        or manifests[0].get("platform") != {"architecture": "arm64", "os": "linux"}
    ):
        raise SystemExit("published image index does not bind exactly linux/arm64")
    immutable = f"{REGISTRY}/cascadia/{args.name}@{index_digest}"
    receipt = {
        "schema_id": "cascadia.cluster.image-publication.v1",
        "built_unix_ms": time.time_ns() // 1_000_000,
        "build_host": "john1",
        "hostname": socket.gethostname(),
        "source_commit": source_identity["git_revision"],
        "source_dirty": source_identity["git_dirty"],
        "source_identity": source_identity,
        "dockerfile": str(dockerfile),
        "dockerfile_sha256": hashlib.sha256(dockerfile.read_bytes()).hexdigest(),
        "build_context": str(context),
        "docker_host": args.docker_host,
        "build_args": build_args,
        "verified_labels": expected_labels,
        "human_reference": reference,
        "image_member_digest": member,
        "image_digest": immutable,
    }
    _write_atomic(args.receipt, receipt)
    print(immutable)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
