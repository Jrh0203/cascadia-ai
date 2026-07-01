from __future__ import annotations

from dataclasses import replace

import pytest
from cascadia_cluster import (
    ContainerInput,
    ContainerSpec,
    InputReference,
    RequestTimeouts,
    Resources,
    RetryPolicy,
    ValidationError,
)
from cascadia_cluster.models import canonical_sha256, item_specification, reject_topology_fields

IMAGE = "registry.cascadia/worker@sha256:" + "a" * 64


def test_public_models_reject_topology_and_mutable_images() -> None:
    with pytest.raises(ValidationError, match="immutable"):
        ContainerSpec("registry.cascadia/worker:latest")
    with pytest.raises(ValidationError, match="topology"):
        reject_topology_fields({"jobs": [{"compatible_hosts": ["john2"]}]})
    with pytest.raises(ValidationError, match="secrets"):
        ContainerInput("item", environment={"ACCESS_TOKEN": "not-allowed"})


def test_specification_hash_is_stable_and_sensitive_to_scientific_inputs() -> None:
    container = ContainerSpec(
        IMAGE,
        entrypoint=("/worker",),
        environment={"MODE": "benchmark"},
        working_directory="/work",
    )
    item = ContainerInput(
        "shard-0",
        args=("simulate", "--seed-start", "0"),
        inputs=(InputReference("cascadia-inputs", "sha256/abc", "b" * 64, "/inputs/data"),),
    )
    resources = Resources(cpu=10, memory_gib=8, disk_gib=2)
    timeouts = RequestTimeouts(10, 80, 10, 100)
    left = item_specification(
        container=container,
        item=item,
        resources=resources,
        outputs=("/outputs",),
        timeouts=timeouts,
        protocol_version="v1",
    )
    right = item_specification(
        container=container,
        item=item,
        resources=resources,
        outputs=("/outputs",),
        timeouts=timeouts,
        protocol_version="v1",
    )
    changed = item_specification(
        container=container,
        item=replace(item, args=("simulate", "--seed-start", "1")),
        resources=resources,
        outputs=("/outputs",),
        timeouts=timeouts,
        protocol_version="v1",
    )
    assert canonical_sha256(left) == canonical_sha256(right)
    assert canonical_sha256(left) != canonical_sha256(changed)


def test_resources_and_timeouts_fail_closed() -> None:
    assert Resources(2.5, 4, 1).bacalhau() == {
        "CPU": "2.5",
        "Memory": "4Gi",
        "Disk": "1Gi",
    }
    with pytest.raises(ValidationError, match="total timeout"):
        RequestTimeouts(10, 20, 10, 39)
    with pytest.raises(ValidationError, match="output paths"):
        item_specification(
            container=ContainerSpec(IMAGE),
            item=ContainerInput("x"),
            resources=Resources(1, 1, 1),
            outputs=(),
            timeouts=RequestTimeouts(1, 2, 1, 4),
            protocol_version="v1",
        )


def test_retry_policy_is_bounded_and_does_not_retry_deterministic_errors() -> None:
    policy = RetryPolicy(maximum_attempts=3, retryable_exit_codes=(137,))
    with pytest.raises(ValidationError, match="Cascadia's contract"):
        RetryPolicy(maximum_attempts=2)
    assert not policy.should_retry(attempt=1, exit_code=2)
    assert policy.should_retry(attempt=1, exit_code=137)
    assert policy.should_retry(attempt=2, exit_code=None, infrastructure_failure=True)
    assert not policy.should_retry(attempt=3, exit_code=137)
