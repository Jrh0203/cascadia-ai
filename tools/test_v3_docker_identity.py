from __future__ import annotations

import v3_docker_identity as identity


def test_identity_requires_same_digest_on_both_cpu_workers() -> None:
    image = "registry/v3@sha256:" + "a" * 64
    value = identity.certify(
        {
            "schema_id": "cascadia.cluster.image-publication.v1",
            "build_host": "john1",
            "image_digest": image,
        },
        {
            "passed": True,
            "image_digest": image,
            "initial_placements": {"job-a": "john2", "job-b": "john3"},
        },
    )
    assert value["passed"] is True
