#!/usr/bin/env python3
"""Reconnect to a map request and atomically import every validated result."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from cascadia_cluster import ClusterClient, ObjectStoreClient, ObjectStoreConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("request_id")
    parser.add_argument("--state-directory", type=Path, required=True)
    parser.add_argument("--artifact-directory", type=Path, required=True)
    args = parser.parse_args()
    required = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise SystemExit(f"missing object-store environment: {missing}")
    store = ObjectStoreClient(
        ObjectStoreConfig(
            endpoint=os.environ.get("AWS_ENDPOINT_URL_S3", "http://100.110.109.6:9000"),
            access_key=os.environ["AWS_ACCESS_KEY_ID"],
            secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        )
    )
    client = ClusterClient(
        "http://100.110.109.6:1234",
        state_directory=args.state_directory,
        object_store=store,
        artifact_directory=args.artifact_directory,
    )
    result = client.reconnect(args.request_id).results()
    for item in result.results:
        print(f"{item.item_key}\t{item.status}\t{item.accepted_execution_id or '-'}")
    return 1 if result.failure_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
