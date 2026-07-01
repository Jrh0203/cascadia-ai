"""Structured failures for the topology-free cluster client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import JobResult


class ClusterError(RuntimeError):
    """Base class for cluster API failures."""


class ValidationError(ClusterError):
    """A caller supplied a non-portable or malformed request."""


class BacalhauAPIError(ClusterError):
    """The Bacalhau REST API returned an invalid or unsuccessful response."""


class RequestConflictError(ClusterError):
    """An item key was previously submitted under a different specification."""


class ArtifactValidationError(ClusterError):
    """A published execution artifact failed manifest or checksum validation."""


@dataclass
class MapError(ClusterError):
    """One or more map items failed while preserving every item result."""

    request_id: str
    results: tuple[JobResult, ...]

    def __str__(self) -> str:
        failed = sum(result.status.is_failure for result in self.results)
        return f"cluster map {self.request_id} has {failed} failed item(s)"
