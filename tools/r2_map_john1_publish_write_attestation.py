#!/usr/bin/env python3
"""Publish one bounded John1 zero-write attestation directly to John2."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPOSITORY = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPOSITORY / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from cascadia_mlx.r2_map_local_write_guard import (  # noqa: E402
    john1_attestation_publication_receipt_relative,
    require_no_local_write_sandbox,
)
from cascadia_mlx.r2_map_remote_storage import (  # noqa: E402
    RemoteStorageClient,
    SshTransport,
    canonical_json,
    content_sha256,
)

MAX_ATTESTATION_BYTES = 1 << 20


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--relative", required=True)
    arguments = parser.parse_args()
    sandbox = require_no_local_write_sandbox(Path(__file__))
    payload = sys.stdin.buffer.read(MAX_ATTESTATION_BYTES + 1)
    if len(payload) > MAX_ATTESTATION_BYTES:
        raise SystemExit("local-write attestation exceeds its in-memory bound")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit("local-write attestation is invalid JSON") from error
    expected = dict(value)
    claimed = expected.pop("attestation_sha256", None)
    if (
        value.get("schema_id") != "cascadia.r2-map.john1-local-write-attestation.v1"
        or value.get("run_id") != arguments.run_id
        or claimed != hashlib.sha256(canonical_json(expected)).hexdigest()
        or sandbox["attestation_relative"] != arguments.relative
    ):
        raise SystemExit("local-write attestation identity is invalid")
    encoded = canonical_json(value) + b"\n"
    receipt_relative = john1_attestation_publication_receipt_relative(claimed)
    request_id = Path(receipt_relative).stem
    publication = RemoteStorageClient(SshTransport()).put_bytes(
        arguments.relative,
        encoded,
        expected_current="absent",
        request_id=request_id,
    )
    if (
        publication.get("sha256") != content_sha256(encoded)
        or publication.get("mode") != "0o400"
        or publication.get("previous_sha256") is not None
        or publication.get("storage_receipt_relative") != receipt_relative
    ):
        raise SystemExit("local-write attestation publication differs")
    sys.stdout.buffer.write(
        canonical_json(
            {
                "relative": arguments.relative,
                "attestation_sha256": claimed,
                "object_sha256": publication["sha256"],
                "storage_receipt_relative": publication["storage_receipt_relative"],
                "storage_receipt_sha256": publication["storage_receipt_sha256"],
                "publisher_sandbox": sandbox,
            }
        )
        + b"\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
