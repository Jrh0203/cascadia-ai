#!/usr/bin/env python3
"""Build and push a Cascadia worker image from any configured Docker host."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

REGISTRY = "100.110.109.6:5000"
DEFAULT_DOCKER_HOST = (
    "unix:///Users/johnherrick/.local/share/cascadia-r2/colima/cascadia-r2/docker.sock"
)
DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


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
    parser.add_argument("--build-arg", action="append", default=[])
    parser.add_argument("--docker-host", default=os.environ.get("DOCKER_HOST", DEFAULT_DOCKER_HOST))
    args = parser.parse_args()

    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", args.name):
        raise SystemExit("image name must be a portable lowercase registry component")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.tag):
        raise SystemExit("image tag is not portable")

    context = args.context.resolve(strict=True)
    dockerfile = args.dockerfile.resolve(strict=True)
    reference = f"{REGISTRY}/cascadia/{args.name}:{args.tag}"
    environment = dict(os.environ, DOCKER_HOST=args.docker_host)
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
    for value in args.build_arg:
        command.extend(("--build-arg", value))
    command.append(str(context))
    subprocess.run(command, check=True, env=environment)
    subprocess.run(
        ["/opt/homebrew/bin/docker", "push", reference],
        check=True,
        env=environment,
    )

    raw_members = subprocess.check_output(
        [
            "/opt/homebrew/bin/docker",
            "image",
            "inspect",
            reference,
            "--format",
            "{{json .RepoDigests}}",
        ],
        text=True,
        env=environment,
    )
    members = [
        value
        for value in json.loads(raw_members)
        if value.startswith(f"{REGISTRY}/cascadia/")
    ]
    if not members:
        raise SystemExit("registry did not return an image member")
    member = members[0]

    # Bacalhau needs a platform-bearing index. The digest is protocol output,
    # not a policy requirement; callers may use the mutable tag.
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
        env=environment,
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
        env=environment,
    )
    index_digest = _manifest_push_digest(
        subprocess.check_output(
            [
                "/opt/homebrew/bin/docker",
                "manifest",
                "push",
                "--insecure",
                "--purge",
                reference,
            ],
            text=True,
            env=environment,
        )
    )
    print(reference)
    print(f"{REGISTRY}/cascadia/{args.name}@{index_digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
