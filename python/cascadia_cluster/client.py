"""Topology-free one-to-many Bacalhau client with durable reconnect state."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from .bacalhau_api import BacalhauAPI
from .errors import ArtifactValidationError, MapError, RequestConflictError, ValidationError
from .models import (
    ContainerInput,
    ContainerSpec,
    JobResult,
    JobStatus,
    MapResult,
    RequestTimeouts,
    Resources,
    RetryPolicy,
    canonical_sha256,
    item_specification,
)
from .object_store import ObjectStoreClient, ObjectStoreError
from .results import import_execution_result

PROTOCOL_VERSION = "cascadia-cluster-map-v1"
_TERMINAL = {"Completed", "Failed", "Stopped"}
_ATTEMPT_TERMINAL = {"Completed", "Failed", "Cancelled"}


def _state_type(job: Mapping[str, Any]) -> str:
    state = job.get("State")
    return str(state.get("StateType", "Unknown")) if isinstance(state, Mapping) else "Unknown"


def _status(
    job: Mapping[str, Any], *, pending_unschedulable_is_terminal: bool = True
) -> JobStatus:
    state = _state_type(job)
    message = str((job.get("State") or {}).get("Message", "")).lower()
    if state in {"Pending", "Queued"}:
        if pending_unschedulable_is_terminal and (
            "not enough nodes" in message or "unschedulable" in message
        ):
            return JobStatus.UNSCHEDULABLE
        return JobStatus.QUEUED
    if state in {"Running", "Starting"}:
        return JobStatus.RUNNING
    if state == "Completed":
        return JobStatus.SUCCEEDED
    if state == "Stopped":
        return JobStatus.CANCELLED
    if state == "Failed":
        return JobStatus.FAILED
    return JobStatus.UNKNOWN


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")[:40] or "item"


def _attempt_count(executions: Sequence[Mapping[str, Any]]) -> int:
    """Count real container attempts, excluding scheduler bid records."""

    return sum(
        str((execution.get("ComputeState") or {}).get("StateType"))
        in _ATTEMPT_TERMINAL
        for execution in executions
    )


def _write_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class _SubmittedItem:
    key: str
    index: int
    spec_sha256: str
    bacalhau_job_id: str


@dataclass(frozen=True)
class _ManagedItem:
    """One durable logical map item, submitted only when admission permits."""

    key: str
    index: int
    spec_sha256: str
    job_payload: Mapping[str, Any]
    bacalhau_job_id: str | None = None


class MapHandle:
    def __init__(
        self,
        *,
        client: ClusterClient,
        request_id: str,
        image_digest: str,
        items: Sequence[_SubmittedItem],
        started_monotonic: float | None = None,
        managed_items: Sequence[_ManagedItem] | None = None,
        maximum_outstanding: int | None = None,
        experiment_id: str | None = None,
        admission_closed: bool = False,
        recover_next_missing_item: bool = False,
    ) -> None:
        self._client = client
        self.request_id = request_id
        self.image_digest = image_digest
        self._items = tuple(sorted(items, key=lambda item: item.index))
        self._managed_items = (
            list(sorted(managed_items, key=lambda item: item.index))
            if managed_items is not None
            else None
        )
        self._maximum_outstanding = maximum_outstanding
        self._experiment_id = experiment_id
        self._admission_closed = admission_closed
        # A managed request persists immediately before its first scheduler
        # submission and after every accepted job ID.  Consequently, after a
        # reconnect only the first item missing an ID can be an orphan created
        # between Bacalhau accepting the job and the following durable write.
        # Recover that one item by labels/spec; all later admissions are known
        # fresh and can use Bacalhau's idempotent submission token directly.
        self._recover_next_missing_item = recover_next_missing_item
        self._started_monotonic = (
            time.monotonic() if started_monotonic is None else started_monotonic
        )

    def status(self) -> tuple[JobStatus, ...]:
        if self._managed_items is not None:
            self._advance_admission()
            return tuple(
                JobStatus.QUEUED
                if item.bacalhau_job_id is None
                else _status(
                    self._client.api.get_job(item.bacalhau_job_id)["Job"],
                    pending_unschedulable_is_terminal=False,
                )
                for item in self._managed_items
            )
        return tuple(
            _status(self._client.api.get_job(item.bacalhau_job_id)["Job"]) for item in self._items
        )

    def wait(self, *, poll_seconds: float = 1.0, timeout_seconds: float | None = None) -> MapHandle:
        started = time.monotonic()
        while True:
            statuses = self.status()
            if all(status.terminal for status in statuses):
                return self
            if timeout_seconds is not None and time.monotonic() - started >= timeout_seconds:
                raise TimeoutError(
                    f"cluster request {self.request_id} did not reach terminal state"
                )
            time.sleep(poll_seconds)

    def cancel(self, reason: str = "cancelled by Cascadia caller") -> None:
        if self._managed_items is not None:
            self._admission_closed = True
            self._persist_managed()
            submitted = [item for item in self._managed_items if item.bacalhau_job_id is not None]
            for item in submitted:
                assert item.bacalhau_job_id is not None
                job = self._client.api.get_job(item.bacalhau_job_id)["Job"]
                if _state_type(job) not in _TERMINAL:
                    self._client.api.stop(item.bacalhau_job_id, reason=reason)
            return
        for item, status in zip(self._items, self.status(), strict=True):
            if not status.terminal:
                self._client.api.stop(item.bacalhau_job_id, reason=reason)

    def results(self) -> MapResult:
        if self._managed_items is not None:
            missing = [item.key for item in self._managed_items if item.bacalhau_job_id is None]
            if missing:
                raise ValidationError(
                    f"cluster request {self.request_id} still has {len(missing)} items awaiting "
                    "scheduler admission"
                )
            self._items = tuple(
                _SubmittedItem(
                    item.key,
                    item.index,
                    item.spec_sha256,
                    str(item.bacalhau_job_id),
                )
                for item in self._managed_items
            )
        results = tuple(self._result(item) for item in self._items)
        return MapResult(
            request_id=self.request_id,
            results=results,
            elapsed_seconds=max(0.0, time.monotonic() - self._started_monotonic),
        )

    def _result(self, item: _SubmittedItem) -> JobResult:
        response = self._client.api.get_job(item.bacalhau_job_id)
        job = response["Job"]
        status = _status(job)
        executions = self._client.api.executions(item.bacalhau_job_id)
        completed = sorted(
            (
                execution
                for execution in executions
                if str((execution.get("ComputeState") or {}).get("StateType")) == "Completed"
                and isinstance(execution.get("RunOutput"), Mapping)
                and execution["RunOutput"].get("ExitCode") == 0
            ),
            key=lambda execution: (
                int(execution.get("CreateTime", 0)),
                str(execution.get("ID", "")),
            ),
        )
        terminal_attempts = sorted(
            (
                execution
                for execution in executions
                if str((execution.get("ComputeState") or {}).get("StateType"))
                in _ATTEMPT_TERMINAL
            ),
            key=lambda execution: (
                int(execution.get("ModifyTime", execution.get("CreateTime", 0))),
                str(execution.get("ID", "")),
            ),
        )
        accepted = completed[0] if completed else None
        artifact_manifest = None
        artifact_failure = None
        if self._client.object_store is not None and self._client.artifact_directory is not None:
            accepted = None
            failures = []
            for execution in completed:
                execution_id = str(execution.get("ID", ""))
                if not execution_id:
                    continue
                try:
                    artifact_manifest = import_execution_result(
                        object_store=self._client.object_store,
                        job_id=item.bacalhau_job_id,
                        execution_id=execution_id,
                        output_name="output-0",
                        destination=(self._client.artifact_directory / self.request_id / item.key),
                    )
                except (ArtifactValidationError, ObjectStoreError) as error:
                    failures.append(f"{execution_id}: {error}")
                    continue
                accepted = execution
                break
            if completed and accepted is None:
                artifact_failure = "no completed execution had a valid artifact: " + "; ".join(
                    failures
                )
        latest = accepted or (executions[0] if executions else {})
        if accepted is None and terminal_attempts:
            latest = terminal_attempts[-1]
        run_output = latest.get("RunOutput") if isinstance(latest, Mapping) else None
        exit_code = run_output.get("ExitCode") if isinstance(run_output, Mapping) else None
        state = job.get("State") if isinstance(job.get("State"), Mapping) else {}
        application_failure = (
            artifact_manifest is not None
            and artifact_manifest.application_metadata.get("cascadia_application_status")
            == "failed"
        )
        if artifact_failure:
            status = JobStatus.FAILED
        if application_failure:
            status = JobStatus.FAILED
            recorded_exit = artifact_manifest.application_metadata.get("cascadia_exit_code")
            if isinstance(recorded_exit, int):
                exit_code = recorded_exit
        if accepted is None and isinstance(exit_code, int) and exit_code != 0:
            status = JobStatus.FAILED
        result = JobResult(
            item_key=item.key,
            request_id=self.request_id,
            bacalhau_job_id=item.bacalhau_job_id,
            accepted_execution_id=(str(accepted.get("ID")) if accepted else None),
            image_digest=self.image_digest,
            spec_sha256=item.spec_sha256,
            status=status,
            exit_code=exit_code if isinstance(exit_code, int) else None,
            created_unix_ns=job.get("CreateTime")
            if isinstance(job.get("CreateTime"), int)
            else None,
            modified_unix_ns=job.get("ModifyTime")
            if isinstance(job.get("ModifyTime"), int)
            else None,
            logs_reference=f"/api/v1/orchestrator/jobs/{item.bacalhau_job_id}/logs",
            artifact_manifest=artifact_manifest,
            application_metadata=(
                artifact_manifest.application_metadata if artifact_manifest else {}
            ),
            failure_reason=(
                artifact_failure
                if artifact_failure
                else (
                    f"deterministic application exit {exit_code}"
                    if application_failure
                    else (
                        f"worker execution exit {exit_code}"
                        if isinstance(exit_code, int) and exit_code != 0
                        else (str(state.get("Message")) if status.is_failure else None)
                    )
                )
            ),
            attempts=_attempt_count(executions),
        )
        self._client._persist_result_receipt(result)
        return result

    def _advance_admission(self) -> None:
        """Keep a bounded set of jobs in Bacalhau without choosing their nodes."""

        assert self._managed_items is not None
        assert self._maximum_outstanding is not None
        if self._admission_closed:
            return
        active = 0
        for item in self._managed_items:
            if item.bacalhau_job_id is None:
                continue
            status = _status(
                self._client.api.get_job(item.bacalhau_job_id)["Job"],
                pending_unschedulable_is_terminal=False,
            )
            if not status.terminal:
                active += 1
        changed = False
        for position, item in enumerate(self._managed_items):
            if active >= self._maximum_outstanding:
                break
            if item.bacalhau_job_id is not None:
                continue
            recovering = self._recover_next_missing_item
            job_id = self._client._submit_or_recover_item(
                request_id=self.request_id,
                item=item,
                recover_existing=recovering,
            )
            self._managed_items[position] = replace(item, bacalhau_job_id=job_id)
            # Persist after every accepted submission. If the process dies between
            # Bacalhau and this write, label/spec recovery makes the next pass exact.
            self._persist_managed()
            if recovering:
                self._recover_next_missing_item = False
            active += 1
            changed = True
        if changed:
            self._items = tuple(
                _SubmittedItem(
                    item.key,
                    item.index,
                    item.spec_sha256,
                    str(item.bacalhau_job_id),
                )
                for item in self._managed_items
                if item.bacalhau_job_id is not None
            )

    def _persist_managed(self) -> None:
        assert self._managed_items is not None
        assert self._maximum_outstanding is not None
        assert self._experiment_id is not None
        self._client._persist_managed_request(
            request_id=self.request_id,
            image_digest=self.image_digest,
            experiment_id=self._experiment_id,
            items=self._managed_items,
            maximum_outstanding=self._maximum_outstanding,
            admission_closed=self._admission_closed,
            replace_existing=True,
        )


class ClusterClient:
    def __init__(
        self,
        endpoint: str,
        *,
        state_directory: Path,
        object_store: ObjectStoreClient | None = None,
        artifact_directory: Path | None = None,
    ) -> None:
        self.api = BacalhauAPI(endpoint)
        self.state_directory = state_directory
        self.object_store = object_store
        self.artifact_directory = artifact_directory
        if (object_store is None) != (artifact_directory is None):
            raise ValidationError("object_store and artifact_directory must be configured together")

    def _persist_result_receipt(self, result: JobResult) -> None:
        """Bind an accepted artifact to its immutable scheduler provenance."""

        manifest = result.artifact_manifest
        if (
            self.artifact_directory is None
            or result.status is not JobStatus.SUCCEEDED
            or manifest is None
            or result.bacalhau_job_id is None
            or result.accepted_execution_id is None
        ):
            return
        output_manifest = {
            "protocol_version": manifest.protocol_version,
            "command": list(manifest.command),
            "files": [asdict(file) for file in manifest.files],
            "application_metadata": dict(manifest.application_metadata),
        }
        receipt: dict[str, Any] = {
            "schema_id": "cascadia.cluster.accepted-result.v1",
            "request_id": result.request_id,
            "item_id": result.item_key,
            "bacalhau_job_id": result.bacalhau_job_id,
            "accepted_execution_id": result.accepted_execution_id,
            "image_digest": result.image_digest,
            "spec_sha256": result.spec_sha256,
            "output_manifest_sha256": canonical_sha256(output_manifest),
            "application_metadata": dict(manifest.application_metadata),
            "attempts": result.attempts,
            "created_unix_ns": result.created_unix_ns,
            "modified_unix_ns": result.modified_unix_ns,
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        path = (
            self.artifact_directory
            / result.request_id
            / ".receipts"
            / f"{result.item_key}.json"
        )
        if path.exists():
            try:
                existing = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as error:
                raise ArtifactValidationError(
                    f"cannot read accepted result receipt: {path}: {error}"
                ) from error
            if existing != receipt:
                raise ArtifactValidationError(
                    f"accepted result receipt already differs: {path}"
                )
            return
        _write_atomic(path, receipt)

    def map(
        self,
        image: str,
        jobs: Sequence[ContainerInput],
        resources: Resources,
        outputs: Sequence[str],
        timeout_seconds: int,
        *,
        entrypoint: Sequence[str] = (),
        environment: Mapping[str, str] | None = None,
        working_directory: str | None = None,
        retry_policy: RetryPolicy | None = None,
        experiment_id: str = "unassigned",
        request_id: str | None = None,
        scheduler_backpressure: bool = False,
    ) -> list[JobResult]:
        handle = self.submit_map(
            image=image,
            jobs=jobs,
            resources=resources,
            outputs=outputs,
            timeout_seconds=timeout_seconds,
            entrypoint=entrypoint,
            environment=environment,
            working_directory=working_directory,
            retry_policy=retry_policy,
            experiment_id=experiment_id,
            request_id=request_id,
            scheduler_backpressure=scheduler_backpressure,
        )
        result = handle.wait(timeout_seconds=timeout_seconds).results()
        if result.failure_count:
            raise MapError(result.request_id, result.results)
        return list(result.results)

    def submit_map(
        self,
        *,
        image: str,
        jobs: Sequence[ContainerInput],
        resources: Resources,
        outputs: Sequence[str],
        timeout_seconds: int,
        entrypoint: Sequence[str] = (),
        environment: Mapping[str, str] | None = None,
        working_directory: str | None = None,
        retry_policy: RetryPolicy | None = None,
        experiment_id: str = "unassigned",
        request_id: str | None = None,
        scheduler_backpressure: bool = False,
    ) -> MapHandle:
        if not jobs or len({item.key for item in jobs}) != len(jobs):
            raise ValidationError("map inputs must be nonempty with unique item keys")
        if not experiment_id:
            raise ValidationError("experiment_id must be nonempty")
        request_id = request_id or f"req-{uuid.uuid4()}"
        container = ContainerSpec(
            image=image,
            entrypoint=tuple(entrypoint),
            environment=environment or {},
            working_directory=working_directory,
        )
        timeouts = RequestTimeouts.from_total(timeout_seconds)
        retry_policy = retry_policy or RetryPolicy()
        if scheduler_backpressure:
            managed_items = []
            for index, item in enumerate(jobs):
                specification = item_specification(
                    container=container,
                    item=item,
                    resources=resources,
                    outputs=outputs,
                    timeouts=timeouts,
                    protocol_version=PROTOCOL_VERSION,
                )
                spec_sha256 = canonical_sha256(specification)
                managed_items.append(
                    _ManagedItem(
                        key=item.key,
                        index=index,
                        spec_sha256=spec_sha256,
                        job_payload=self._job_payload(
                            request_id=request_id,
                            experiment_id=experiment_id,
                            index=index,
                            item=item,
                            container=container,
                            resources=resources,
                            outputs=outputs,
                            timeouts=timeouts,
                            retry_policy=retry_policy,
                            spec_sha256=spec_sha256,
                        ),
                    )
                )
            maximum_outstanding = self._scheduler_packing_capacity(resources)
            managed_items, recover_next_missing_item = self._initialize_managed_request(
                request_id=request_id,
                image_digest=image,
                experiment_id=experiment_id,
                items=managed_items,
                maximum_outstanding=maximum_outstanding,
            )
            handle = MapHandle(
                client=self,
                request_id=request_id,
                image_digest=image,
                items=(),
                managed_items=managed_items,
                maximum_outstanding=maximum_outstanding,
                experiment_id=experiment_id,
                recover_next_missing_item=recover_next_missing_item,
            )
            handle._advance_admission()
            return handle
        submitted = []
        for index, item in enumerate(jobs):
            specification = item_specification(
                container=container,
                item=item,
                resources=resources,
                outputs=outputs,
                timeouts=timeouts,
                protocol_version=PROTOCOL_VERSION,
            )
            spec_sha256 = canonical_sha256(specification)
            existing = self.api.list_jobs(
                labels={"cascadia.request_id": request_id, "cascadia.item_id": item.key}
            )
            matching = [
                job
                for job in existing
                if job.get("Labels", {}).get("cascadia.spec_sha256") == spec_sha256
            ]
            conflicting = [job for job in existing if job not in matching]
            if conflicting or len(matching) > 1:
                raise RequestConflictError(
                    f"request {request_id} item {item.key} conflicts with existing Bacalhau jobs"
                )
            if matching:
                job_id = str(matching[0]["ID"])
            else:
                payload = self._job_payload(
                    request_id=request_id,
                    experiment_id=experiment_id,
                    index=index,
                    item=item,
                    container=container,
                    resources=resources,
                    outputs=outputs,
                    timeouts=timeouts,
                    retry_policy=retry_policy,
                    spec_sha256=spec_sha256,
                )
                response = self.api.submit(
                    payload,
                    idempotency_token=canonical_sha256(
                        {"request_id": request_id, "item_key": item.key, "spec": spec_sha256}
                    ),
                )
                job_id = str(response.get("JobID", ""))
                if not job_id:
                    raise ValidationError("Bacalhau submission omitted JobID")
            submitted.append(_SubmittedItem(item.key, index, spec_sha256, job_id))
        self._persist_request(request_id, container.image, experiment_id, submitted)
        return MapHandle(client=self, request_id=request_id, image_digest=image, items=submitted)

    def reconnect(self, request_id: str) -> MapHandle:
        path = self.state_directory / "requests" / f"{request_id}.json"
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValidationError(f"cannot reconnect request {request_id}: {error}") from error
        schema_id = value.get("schema_id")
        if schema_id == "cascadia.cluster.managed-request-state.v2":
            self._validate_state_checksum(value, request_id)
            managed_items = tuple(
                _ManagedItem(
                    key=item["key"],
                    index=item["index"],
                    spec_sha256=item["spec_sha256"],
                    job_payload=item["job_payload"],
                    bacalhau_job_id=item.get("bacalhau_job_id"),
                )
                for item in value["items"]
            )
            submitted = tuple(
                _SubmittedItem(
                    item.key,
                    item.index,
                    item.spec_sha256,
                    str(item.bacalhau_job_id),
                )
                for item in managed_items
                if item.bacalhau_job_id is not None
            )
            return MapHandle(
                client=self,
                request_id=request_id,
                image_digest=value["image_digest"],
                items=submitted,
                managed_items=managed_items,
                maximum_outstanding=value["admission"]["maximum_outstanding"],
                experiment_id=value["experiment_id"],
                admission_closed=value["admission"]["closed"],
                recover_next_missing_item=True,
            )
        if schema_id != "cascadia.cluster.request-state.v1":
            raise ValidationError(f"cannot reconnect request {request_id}: unsupported schema")
        self._validate_state_checksum(value, request_id)
        items = tuple(_SubmittedItem(**item) for item in value["items"])
        return MapHandle(
            client=self,
            request_id=request_id,
            image_digest=value["image_digest"],
            items=items,
        )

    @staticmethod
    def _validate_state_checksum(value: Mapping[str, Any], request_id: str) -> None:
        payload = dict(value)
        claimed = payload.pop("state_sha256", None)
        if claimed != canonical_sha256(payload):
            raise ValidationError(f"cannot reconnect request {request_id}: checksum differs")

    def _scheduler_packing_capacity(self, resources: Resources) -> int:
        """Derive a bounded scheduler admission window; never bind a node.

        Bacalhau v1.9 assigns a job to a compute node before that node has a
        free execution slot. With a window equal to exact aggregate capacity,
        placement skew can therefore leave cores idle on one node while jobs
        wait in another node's local queue. On a multi-node fleet, admit one
        additional largest-node window. Bacalhau still owns placement and
        enforces every node resource limit; the extra jobs are only bounded
        queue slack that keeps aggregate capacity work-conserving.
        """

        slot_capacities: list[int] = []
        for node in self.api.nodes():
            if str(node.get("Connection", "")).upper() != "CONNECTED":
                continue
            info = node.get("Info")
            compute = info.get("ComputeNodeInfo") if isinstance(info, Mapping) else None
            maximum = compute.get("MaxCapacity") if isinstance(compute, Mapping) else None
            engines = compute.get("ExecutionEngines") if isinstance(compute, Mapping) else None
            if not isinstance(maximum, Mapping) or not isinstance(engines, Sequence):
                continue
            if "docker" not in engines:
                continue
            try:
                slots = [
                    math.floor(float(maximum["CPU"]) / float(resources.cpu)),
                    math.floor(
                        float(maximum["Memory"])
                        / (float(resources.memory_gib) * 1024**3)
                    ),
                    math.floor(
                        float(maximum["Disk"])
                        / (float(resources.disk_gib) * 1024**3)
                    ),
                ]
                if resources.gpu:
                    slots.append(math.floor(float(maximum.get("GPU", 0)) / resources.gpu))
            except (KeyError, TypeError, ValueError, OverflowError) as error:
                raise ValidationError("scheduler reported malformed node capacity") from error
            slot_capacities.append(max(0, min(slots)))
        capacity = sum(slot_capacities)
        if capacity < 1:
            raise ValidationError("scheduler has no connected capacity for this map item")
        placement_slack = max(slot_capacities) if len(slot_capacities) > 1 else 0
        return capacity + placement_slack

    def _initialize_managed_request(
        self,
        *,
        request_id: str,
        image_digest: str,
        experiment_id: str,
        items: Sequence[_ManagedItem],
        maximum_outstanding: int,
    ) -> tuple[list[_ManagedItem], bool]:
        path = self.state_directory / "requests" / f"{request_id}.json"
        if not path.exists():
            self._persist_managed_request(
                request_id=request_id,
                image_digest=image_digest,
                experiment_id=experiment_id,
                items=items,
                maximum_outstanding=maximum_outstanding,
                admission_closed=False,
                replace_existing=False,
            )
            return list(items), False
        try:
            existing = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise RequestConflictError(
                f"cannot recover durable request {request_id}: {error}"
            ) from error
        if existing.get("schema_id") != "cascadia.cluster.managed-request-state.v2":
            raise RequestConflictError(f"durable request schema differs: {request_id}")
        self._validate_state_checksum(existing, request_id)
        if existing.get("admission", {}).get("closed") is True:
            raise RequestConflictError(f"durable managed request is closed: {request_id}")
        expected_definition = self._managed_request_value(
            request_id=request_id,
            image_digest=image_digest,
            experiment_id=experiment_id,
            items=items,
            maximum_outstanding=maximum_outstanding,
            admission_closed=False,
        )
        observed_definition = dict(existing)
        expected_definition.pop("state_sha256", None)
        observed_definition.pop("state_sha256", None)
        observed_definition["admission"] = dict(observed_definition["admission"])
        expected_definition["admission"] = dict(expected_definition["admission"])
        observed_definition["admission"]["closed"] = False
        # Capacity is live operational flow control, not request identity. A
        # reconnect may safely tighten or widen the window without changing a
        # planned job or its scientific specification.
        observed_definition["admission"]["maximum_outstanding"] = maximum_outstanding
        observed_definition["items"] = [
            {**item, "bacalhau_job_id": None} for item in observed_definition["items"]
        ]
        if observed_definition != expected_definition:
            raise RequestConflictError(f"durable managed request definition differs: {request_id}")
        recovered = [
            _ManagedItem(
                key=item["key"],
                index=item["index"],
                spec_sha256=item["spec_sha256"],
                job_payload=item["job_payload"],
                bacalhau_job_id=item.get("bacalhau_job_id"),
            )
            for item in existing["items"]
        ]
        if existing["admission"]["maximum_outstanding"] != maximum_outstanding:
            self._persist_managed_request(
                request_id=request_id,
                image_digest=image_digest,
                experiment_id=experiment_id,
                items=recovered,
                maximum_outstanding=maximum_outstanding,
                admission_closed=False,
                replace_existing=True,
            )
        return recovered, True

    def _submit_or_recover_item(
        self,
        *,
        request_id: str,
        item: _ManagedItem,
        recover_existing: bool,
    ) -> str:
        if recover_existing:
            existing = self.api.list_jobs(
                labels={"cascadia.request_id": request_id, "cascadia.item_id": item.key}
            )
            matching = [
                job
                for job in existing
                if job.get("Labels", {}).get("cascadia.spec_sha256") == item.spec_sha256
            ]
            conflicting = [job for job in existing if job not in matching]
            if conflicting or len(matching) > 1:
                raise RequestConflictError(
                    f"request {request_id} item {item.key} conflicts with existing Bacalhau jobs"
                )
            if matching:
                return str(matching[0]["ID"])
        response = self.api.submit(
            item.job_payload,
            idempotency_token=canonical_sha256(
                {"request_id": request_id, "item_key": item.key, "spec": item.spec_sha256}
            ),
        )
        job_id = str(response.get("JobID", ""))
        if not job_id:
            raise ValidationError("Bacalhau submission omitted JobID")
        return job_id

    def _persist_request(
        self,
        request_id: str,
        image_digest: str,
        experiment_id: str,
        items: Sequence[_SubmittedItem],
    ) -> None:
        value = {
            "schema_id": "cascadia.cluster.request-state.v1",
            "request_id": request_id,
            "image_digest": image_digest,
            "experiment_id": experiment_id,
            "items": [asdict(item) for item in items],
        }
        value["state_sha256"] = canonical_sha256(value)
        path = self.state_directory / "requests" / f"{request_id}.json"
        if path.exists():
            existing = json.loads(path.read_text())
            if existing != value:
                raise RequestConflictError(f"durable request state differs: {request_id}")
            return
        _write_atomic(path, value)

    @staticmethod
    def _managed_request_value(
        *,
        request_id: str,
        image_digest: str,
        experiment_id: str,
        items: Sequence[_ManagedItem],
        maximum_outstanding: int,
        admission_closed: bool,
    ) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_id": "cascadia.cluster.managed-request-state.v2",
            "request_id": request_id,
            "image_digest": image_digest,
            "experiment_id": experiment_id,
            "admission": {
                "kind": "scheduler-capacity-backpressure",
                "maximum_outstanding": maximum_outstanding,
                "closed": admission_closed,
            },
            "items": [
                {
                    "key": item.key,
                    "index": item.index,
                    "spec_sha256": item.spec_sha256,
                    "job_payload": dict(item.job_payload),
                    "bacalhau_job_id": item.bacalhau_job_id,
                }
                for item in items
            ],
        }
        value["state_sha256"] = canonical_sha256(value)
        return value

    def _persist_managed_request(
        self,
        *,
        request_id: str,
        image_digest: str,
        experiment_id: str,
        items: Sequence[_ManagedItem],
        maximum_outstanding: int,
        admission_closed: bool,
        replace_existing: bool,
    ) -> None:
        value = self._managed_request_value(
            request_id=request_id,
            image_digest=image_digest,
            experiment_id=experiment_id,
            items=items,
            maximum_outstanding=maximum_outstanding,
            admission_closed=admission_closed,
        )
        path = self.state_directory / "requests" / f"{request_id}.json"
        if path.exists() and not replace_existing:
            existing = json.loads(path.read_text())
            if existing != value:
                raise RequestConflictError(f"durable managed request state differs: {request_id}")
            return
        _write_atomic(path, value)

    @staticmethod
    def _job_payload(
        *,
        request_id: str,
        experiment_id: str,
        index: int,
        item: ContainerInput,
        container: ContainerSpec,
        resources: Resources,
        outputs: Sequence[str],
        timeouts: RequestTimeouts,
        retry_policy: RetryPolicy,
        spec_sha256: str,
    ) -> dict[str, Any]:
        environment = dict(container.environment) | dict(item.environment)
        environment |= {
            "CASCADIA_APPLICATION_METADATA_JSON": json.dumps(
                dict(item.application_metadata), sort_keys=True, separators=(",", ":")
            ),
            "CASCADIA_OUTPUT_ROOT": str(outputs[0]),
            "CASCADIA_PROTOCOL_VERSION": PROTOCOL_VERSION,
            "CASCADIA_RETRYABLE_EXIT_CODES": ",".join(
                str(code) for code in retry_policy.retryable_exit_codes
            ),
            "CASCADIA_INPUT_SHA256_JSON": json.dumps(
                {reference.mounted_path: reference.sha256 for reference in item.inputs},
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
        engine_params: dict[str, Any] = {
            "Image": container.image,
            "Parameters": list(item.args),
        }
        if container.entrypoint:
            engine_params["Entrypoint"] = list(container.entrypoint)
        if container.working_directory:
            engine_params["WorkingDirectory"] = container.working_directory
        input_sources = []
        for reference in item.inputs:
            params: dict[str, Any] = {
                "Bucket": reference.bucket,
                "Key": reference.key,
                "Region": reference.region,
            }
            if reference.endpoint:
                params["Endpoint"] = reference.endpoint
            input_sources.append(
                {
                    "Alias": f"input-{len(input_sources)}",
                    "Target": reference.target,
                    "Source": {"Type": "s3", "Params": params},
                }
            )
        result_paths = [
            {"Name": f"output-{index}", "Path": output} for index, output in enumerate(outputs)
        ]
        # Bacalhau treats Name as an update key. A suffix of the request ID is
        # not unique (for example, smoke and development can both end in
        # ``v8-aggregate``), which can mutate an older job in place and inherit
        # its timeout clock. Bind the name to the entire logical request.
        request_name_hash = canonical_sha256({"request_id": request_id})[:16]
        return {
            "Name": f"cascadia-{request_name_hash}-{index:04d}-{_safe_name(item.key)}",
            "Namespace": "cascadia",
            "Type": "batch",
            "Count": 1,
            "Labels": {
                "cascadia.request_id": request_id,
                "cascadia.item_id": item.key,
                "cascadia.spec_sha256": spec_sha256,
                "cascadia.experiment_id": experiment_id,
                "cascadia.protocol": PROTOCOL_VERSION,
            },
            "Meta": {
                "cascadia.retry.maximum_attempts": str(retry_policy.maximum_attempts),
                **{
                    f"cascadia.app.{key}": value for key, value in item.application_metadata.items()
                },
            },
            "Tasks": [
                {
                    "Name": "main",
                    "Engine": {"Type": "docker", "Params": engine_params},
                    "Env": environment,
                    "InputSources": input_sources,
                    "Publisher": {"Type": "s3managed"},
                    "ResultPaths": result_paths,
                    "Resources": resources.bacalhau(),
                    "Timeouts": {
                        "QueueTimeout": timeouts.queue_seconds,
                        "ExecutionTimeout": timeouts.execution_seconds,
                        "TotalTimeout": timeouts.total_seconds,
                    },
                }
            ],
        }
