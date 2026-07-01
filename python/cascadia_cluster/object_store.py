"""Minimal AWS Signature V4 client for the private MinIO object store."""

from __future__ import annotations

import hashlib
import hmac
import http.client
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .errors import ArtifactValidationError, ClusterError
from .models import InputReference


class ObjectStoreError(ClusterError):
    """The private object store rejected or failed an operation."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ObjectStoreConfig:
    endpoint: str
    access_key: str
    secret_key: str
    region: str = "us-east-1"
    input_bucket: str = "cascadia-inputs"
    result_bucket: str = "cascadia-results"

    def __post_init__(self) -> None:
        if not self.endpoint.startswith(("http://", "https://")):
            raise ObjectStoreError("object-store endpoint must be HTTP(S)")
        if not self.access_key or not self.secret_key or not self.region:
            raise ObjectStoreError("object-store credentials and region are required")


class ObjectStoreClient:
    """Path-style S3 operations sufficient for staging and accepting artifacts."""

    def __init__(self, config: ObjectStoreConfig, *, timeout_seconds: float = 120.0) -> None:
        self.config = config
        self.timeout_seconds = timeout_seconds

    def _signed_request_parts(
        self,
        method: str,
        bucket: str,
        key: str,
        payload_hash: str,
    ) -> tuple[urllib.parse.SplitResult, str, dict[str, str]]:
        """Return a SigV4 URL path and headers without materializing a payload."""

        if not bucket or bucket.startswith("/") or ".." in bucket:
            raise ObjectStoreError("invalid object-store bucket")
        normalized = "/".join(
            urllib.parse.quote(part, safe="-_.~") for part in key.split("/") if part
        )
        canonical_uri = f"/{urllib.parse.quote(bucket, safe='-_.~')}"
        if normalized:
            canonical_uri += f"/{normalized}"
        endpoint = urllib.parse.urlsplit(self.config.endpoint.rstrip("/"))
        if not endpoint.hostname:
            raise ObjectStoreError("object-store endpoint has no host")
        host = endpoint.netloc
        now = datetime.now(UTC)
        timestamp = now.strftime("%Y%m%dT%H%M%SZ")
        date = now.strftime("%Y%m%d")
        canonical_headers = (
            f"host:{host}\nx-amz-content-sha256:{payload_hash}\nx-amz-date:{timestamp}\n"
        )
        signed_headers = "host;x-amz-content-sha256;x-amz-date"
        canonical_request = "\n".join(
            [method, canonical_uri, "", canonical_headers, signed_headers, payload_hash]
        )
        scope = f"{date}/{self.config.region}/s3/aws4_request"
        string_to_sign = "\n".join(
            ["AWS4-HMAC-SHA256", timestamp, scope, _sha256_bytes(canonical_request.encode())]
        )

        def sign(key_bytes: bytes, value: str) -> bytes:
            return hmac.new(key_bytes, value.encode(), hashlib.sha256).digest()

        date_key = sign(("AWS4" + self.config.secret_key).encode(), date)
        region_key = sign(date_key, self.config.region)
        service_key = sign(region_key, "s3")
        signing_key = sign(service_key, "aws4_request")
        signature = hmac.new(signing_key, string_to_sign.encode(), hashlib.sha256).hexdigest()
        authorization = (
            f"AWS4-HMAC-SHA256 Credential={self.config.access_key}/{scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        )
        return endpoint, canonical_uri, {
            "Authorization": authorization,
            "Host": host,
            "X-Amz-Content-Sha256": payload_hash,
            "X-Amz-Date": timestamp,
        }

    def _signed_request(
        self,
        method: str,
        bucket: str,
        key: str = "",
        *,
        body: bytes | None = None,
        expected: tuple[int, ...] = (200,),
    ) -> bytes:
        payload = body or b""
        payload_hash = _sha256_bytes(payload)
        endpoint, canonical_uri, headers = self._signed_request_parts(
            method, bucket, key, payload_hash
        )
        request = urllib.request.Request(
            urllib.parse.urlunsplit((endpoint.scheme, endpoint.netloc, canonical_uri, "", "")),
            data=body if body is not None else None,
            method=method,
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                value = response.read()
                status = response.status
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")
            if error.code not in expected:
                raise ObjectStoreError(
                    f"object store {method} {bucket}/{key} failed: HTTP {error.code}; {detail}"
                ) from error
            return error.read()
        except (urllib.error.URLError, TimeoutError) as error:
            raise ObjectStoreError(
                f"object store {method} {bucket}/{key} failed: {error}"
            ) from error
        if status not in expected:
            raise ObjectStoreError(
                f"object store {method} {bucket}/{key} returned unexpected HTTP {status}"
            )
        return value

    def ensure_bucket(self, bucket: str) -> None:
        try:
            self._signed_request("HEAD", bucket, expected=(200,))
        except ObjectStoreError:
            self._signed_request("PUT", bucket, body=b"", expected=(200, 204))

    def put_bytes(self, bucket: str, key: str, value: bytes) -> str:
        digest = _sha256_bytes(value)
        self._signed_request("PUT", bucket, key, body=value, expected=(200,))
        return digest

    def put_file(self, bucket: str, key: str, path: Path) -> str:
        digest = sha256_file(path)
        endpoint, canonical_uri, headers = self._signed_request_parts("PUT", bucket, key, digest)
        headers["Content-Length"] = str(path.stat().st_size)
        connection_type = (
            http.client.HTTPSConnection
            if endpoint.scheme == "https"
            else http.client.HTTPConnection
        )
        connection = connection_type(
            endpoint.hostname,
            endpoint.port,
            timeout=self.timeout_seconds,
        )
        try:
            with path.open("rb") as stream:
                connection.request("PUT", canonical_uri, body=stream, headers=headers)
                response = connection.getresponse()
                detail = response.read()
            if response.status != 200:
                raise ObjectStoreError(
                    f"object store PUT {bucket}/{key} failed: HTTP {response.status}; "
                    f"{detail.decode(errors='replace')}"
                )
        except (OSError, http.client.HTTPException) as error:
            raise ObjectStoreError(f"object store PUT {bucket}/{key} failed: {error}") from error
        finally:
            connection.close()
        return digest

    def get_bytes(self, bucket: str, key: str) -> bytes:
        return self._signed_request("GET", bucket, key, expected=(200,))

    def download(self, bucket: str, key: str, destination: Path) -> str:
        destination.parent.mkdir(parents=True, exist_ok=True)
        endpoint, canonical_uri, headers = self._signed_request_parts(
            "GET", bucket, key, _sha256_bytes(b"")
        )
        request = urllib.request.Request(
            urllib.parse.urlunsplit((endpoint.scheme, endpoint.netloc, canonical_uri, "", "")),
            method="GET",
            headers=headers,
        )
        descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
        temporary = Path(name)
        digest = hashlib.sha256()
        try:
            with os.fdopen(descriptor, "wb") as output:
                try:
                    with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                        while chunk := response.read(1024 * 1024):
                            output.write(chunk)
                            digest.update(chunk)
                except urllib.error.HTTPError as error:
                    detail = error.read().decode(errors="replace")
                    raise ObjectStoreError(
                        f"object store GET {bucket}/{key} failed: HTTP {error.code}; {detail}"
                    ) from error
                except (urllib.error.URLError, TimeoutError) as error:
                    raise ObjectStoreError(
                        f"object store GET {bucket}/{key} failed: {error}"
                    ) from error
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, destination)
            return digest.hexdigest()
        finally:
            temporary.unlink(missing_ok=True)

    def stage_file(self, path: Path, *, target: str) -> InputReference:
        digest = sha256_file(path)
        key = f"sha256/{digest[:2]}/{digest}/{path.name}"
        self.ensure_bucket(self.config.input_bucket)
        self.put_file(self.config.input_bucket, key, path)
        if sha256_file(path) != digest:
            raise ArtifactValidationError("input changed while it was staged")
        return InputReference(
            bucket=self.config.input_bucket,
            key=key,
            sha256=digest,
            target=target,
            region=self.config.region,
            endpoint=self.config.endpoint,
        )

    def result_key(self, job_id: str, execution_id: str) -> str:
        return f"executions/{job_id}/{execution_id}.tar.gz"
