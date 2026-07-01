"""Stable public request and result contracts for Cascadia cluster work."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any

from .errors import ValidationError

_DIGEST_IMAGE = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_FORBIDDEN_TOPOLOGY_FIELDS = frozenset(
    {
        "host",
        "hosts",
        "node",
        "nodes",
        "compatible_hosts",
        "remote_root",
        "remote_roots",
        "ssh",
        "ssh_command",
        "rsync",
    }
)
_SECRET_FIELD = re.compile(r"(?:secret|password|passwd|token|credential|private[_-]?key)", re.I)


def _freeze_mapping(values: Mapping[str, str] | None) -> Mapping[str, str]:
    data = dict(values or {})
    if any(not isinstance(key, str) or not key for key in data):
        raise ValidationError("environment keys must be nonempty strings")
    if any(not isinstance(value, str) for value in data.values()):
        raise ValidationError("environment values must be strings")
    forbidden = sorted(key for key in data if _SECRET_FIELD.search(key))
    if forbidden:
        raise ValidationError(f"secrets must not be embedded in environment: {forbidden}")
    return MappingProxyType(dict(sorted(data.items())))


def _absolute_container_path(value: str, label: str) -> str:
    path = PurePosixPath(value)
    if not value.startswith("/") or ".." in path.parts:
        raise ValidationError(f"{label} must be an absolute normalized container path")
    return str(path)


def reject_topology_fields(value: Any, *, path: str = "request") -> None:
    """Reject topology-bearing dictionaries at the public API boundary."""

    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).lower() in _FORBIDDEN_TOPOLOGY_FIELDS:
                raise ValidationError(f"topology field is forbidden at {path}.{key}")
            reject_topology_fields(nested, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            reject_topology_fields(nested, path=f"{path}[{index}]")


@dataclass(frozen=True)
class InputReference:
    """Content-addressed object staged into one container mount directory."""

    bucket: str
    key: str
    sha256: str
    target: str
    region: str = "us-east-1"
    endpoint: str | None = None

    def __post_init__(self) -> None:
        if not self.bucket or not self.key or not _SHA256.fullmatch(self.sha256):
            raise ValidationError("input reference requires bucket, key, and lowercase SHA-256")
        object.__setattr__(
            self,
            "target",
            _absolute_container_path(self.target, "input target directory"),
        )
        if self.endpoint is not None and not self.endpoint.startswith(("http://", "https://")):
            raise ValidationError("input endpoint must be HTTP(S)")

    @property
    def mounted_path(self) -> str:
        """Path of this single S3 object inside Bacalhau's directory mount."""

        return str(PurePosixPath(self.target) / PurePosixPath(self.key).name)


@dataclass(frozen=True)
class ContainerSpec:
    image: str
    entrypoint: tuple[str, ...] = ()
    environment: Mapping[str, str] = field(default_factory=dict)
    working_directory: str | None = None

    def __post_init__(self) -> None:
        if not _DIGEST_IMAGE.fullmatch(self.image):
            raise ValidationError("image must be an immutable sha256 registry digest")
        if any(not isinstance(part, str) or not part for part in self.entrypoint):
            raise ValidationError("entrypoint values must be nonempty strings")
        object.__setattr__(self, "entrypoint", tuple(self.entrypoint))
        object.__setattr__(self, "environment", _freeze_mapping(self.environment))
        if self.working_directory is not None:
            object.__setattr__(
                self,
                "working_directory",
                _absolute_container_path(self.working_directory, "working directory"),
            )


@dataclass(frozen=True)
class ContainerInput:
    key: str
    args: tuple[str, ...] = ()
    environment: Mapping[str, str] = field(default_factory=dict)
    inputs: tuple[InputReference, ...] = ()
    application_metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _KEY.fullmatch(self.key):
            raise ValidationError("item key must be portable and at most 128 characters")
        if any(not isinstance(part, str) or not part for part in self.args):
            raise ValidationError("item arguments must be nonempty strings")
        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(self, "environment", _freeze_mapping(self.environment))
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "application_metadata", _freeze_mapping(self.application_metadata))
        targets = [reference.target for reference in self.inputs]
        if len(targets) != len(set(targets)):
            raise ValidationError("input mount targets must be unique")


@dataclass(frozen=True)
class Resources:
    cpu: float
    memory_gib: float
    disk_gib: float
    gpu: int = 0

    def __post_init__(self) -> None:
        for name in ("cpu", "memory_gib", "disk_gib"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValidationError(f"{name} must be numeric")
            if not math.isfinite(float(value)) or value <= 0:
                raise ValidationError(f"{name} must be finite and positive")
        if not isinstance(self.gpu, int) or isinstance(self.gpu, bool) or self.gpu < 0:
            raise ValidationError("gpu must be a nonnegative integer")

    def bacalhau(self) -> dict[str, str]:
        return {
            # Match Bacalhau 1.9's canonical `docker run --cpu` output.  Large
            # millicore strings such as `10000m` are accepted and persisted by
            # the REST API, but are not debited from node capacity; whole-core
            # decimal strings are.  Fractional cores remain exact (for example
            # 0.25) without relying on the broken large-millicore path.
            "CPU": f"{float(self.cpu):g}",
            "Memory": f"{self.memory_gib:g}Gi",
            "Disk": f"{self.disk_gib:g}Gi",
            **({"GPU": str(self.gpu)} if self.gpu else {}),
        }


@dataclass(frozen=True)
class RetryPolicy:
    # Cascadia records this desired application-attempt contract as metadata.
    # Bacalhau v1.9 has no per-job execution-attempt knob; its evaluation-broker
    # delivery bound is a separate fabric concern and must not be conflated here.
    maximum_attempts: int = 3
    retryable_exit_codes: tuple[int, ...] = (125, 126, 127, 137, 143)
    initial_backoff_seconds: float = 1.0
    maximum_backoff_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.maximum_attempts != 3:
            raise ValidationError("maximum_attempts must match Cascadia's contract of 3")
        if any(
            not isinstance(code, int) or not 0 <= code <= 255 for code in self.retryable_exit_codes
        ):
            raise ValidationError("retryable exit codes must be bytes")
        if not 0 < self.initial_backoff_seconds <= self.maximum_backoff_seconds:
            raise ValidationError("retry backoff bounds are invalid")
        object.__setattr__(self, "retryable_exit_codes", tuple(self.retryable_exit_codes))

    def should_retry(
        self,
        *,
        attempt: int,
        exit_code: int | None,
        infrastructure_failure: bool = False,
    ) -> bool:
        """Classify one failed attempt without retrying deterministic application errors."""

        if not 1 <= attempt <= self.maximum_attempts:
            raise ValidationError("attempt is outside the retry policy")
        if attempt >= self.maximum_attempts:
            return False
        return infrastructure_failure or exit_code in self.retryable_exit_codes


@dataclass(frozen=True)
class RequestTimeouts:
    queue_seconds: int
    execution_seconds: int
    publication_seconds: int
    total_seconds: int

    def __post_init__(self) -> None:
        values = (
            self.queue_seconds,
            self.execution_seconds,
            self.publication_seconds,
            self.total_seconds,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in values
        ):
            raise ValidationError("timeouts must be positive integer seconds")
        if (
            self.queue_seconds + self.execution_seconds + self.publication_seconds
            > self.total_seconds
        ):
            raise ValidationError("total timeout must cover queue, execution, and publication")

    @classmethod
    def from_total(cls, total_seconds: int) -> RequestTimeouts:
        if total_seconds < 4:
            raise ValidationError("total timeout must be at least four seconds")
        queue = max(1, total_seconds // 10)
        publication = max(1, total_seconds // 10)
        execution = total_seconds - queue - publication
        return cls(queue, execution, publication, total_seconds)


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNSCHEDULABLE = "unschedulable"
    UNKNOWN = "unknown"

    @property
    def terminal(self) -> bool:
        return self in {
            self.SUCCEEDED,
            self.FAILED,
            self.CANCELLED,
            self.UNSCHEDULABLE,
        }

    @property
    def is_failure(self) -> bool:
        return self in {self.FAILED, self.CANCELLED, self.UNSCHEDULABLE, self.UNKNOWN}


@dataclass(frozen=True)
class ArtifactFile:
    path: str
    bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if not self.path or self.path.startswith("/") or ".." in PurePosixPath(self.path).parts:
            raise ValidationError("artifact paths must be relative and normalized")
        if self.bytes < 0 or not _SHA256.fullmatch(self.sha256):
            raise ValidationError("artifact descriptor has invalid size or checksum")


@dataclass(frozen=True)
class ArtifactManifest:
    protocol_version: str
    command: tuple[str, ...]
    files: tuple[ArtifactFile, ...]
    application_metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.protocol_version:
            raise ValidationError("artifact protocol version must be nonempty")
        if not self.command or any(not isinstance(value, str) for value in self.command):
            raise ValidationError("artifact command must contain strings")
        object.__setattr__(self, "command", tuple(self.command))
        object.__setattr__(self, "files", tuple(self.files))
        object.__setattr__(
            self,
            "application_metadata",
            MappingProxyType(dict(self.application_metadata)),
        )


@dataclass(frozen=True)
class JobResult:
    item_key: str
    request_id: str
    bacalhau_job_id: str | None
    accepted_execution_id: str | None
    image_digest: str
    spec_sha256: str
    status: JobStatus
    exit_code: int | None = None
    created_unix_ns: int | None = None
    modified_unix_ns: int | None = None
    queue_seconds: float | None = None
    execution_seconds: float | None = None
    logs_reference: str | None = None
    artifact_manifest: ArtifactManifest | None = None
    application_metadata: Mapping[str, Any] = field(default_factory=dict)
    failure_reason: str | None = None
    attempts: int = 0


@dataclass(frozen=True)
class MapResult:
    request_id: str
    results: tuple[JobResult, ...]
    elapsed_seconds: float

    @property
    def failure_count(self) -> int:
        return sum(result.status.is_failure for result in self.results)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def item_specification(
    *,
    container: ContainerSpec,
    item: ContainerInput,
    resources: Resources,
    outputs: Sequence[str],
    timeouts: RequestTimeouts,
    protocol_version: str,
) -> dict[str, Any]:
    normalized_outputs = tuple(_absolute_container_path(path, "output path") for path in outputs)
    if not normalized_outputs or len(normalized_outputs) != len(set(normalized_outputs)):
        raise ValidationError("output paths must be nonempty and unique")
    value = {
        "image": container.image,
        "item_key": item.key,
        "entrypoint": list(container.entrypoint),
        "args": list(item.args),
        "environment": dict(container.environment) | dict(item.environment),
        "working_directory": container.working_directory,
        "inputs": [asdict(reference) for reference in item.inputs],
        "resources": asdict(resources),
        "timeouts": asdict(timeouts),
        "outputs": list(normalized_outputs),
        "application_protocol_version": protocol_version,
    }
    reject_topology_fields(value)
    return value
