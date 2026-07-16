"""Fail-closed device guard for explicitly CPU-only test sessions.

``CUDA_VISIBLE_DEVICES=""`` does not disable Apple's MPS backend and does not
prevent code from querying accelerator availability. Test sessions that set
``CASCADIA_CPU_ONLY_TESTS=1`` therefore need a guard which runs *before* a
device library is imported. This module deliberately depends only on the
standard library so callers can establish that boundary first.

The guard is test-session infrastructure, not a production device policy. It
is inert unless the environment variable is explicitly enabled.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import NoReturn

CPU_ONLY_TESTS_ENV = "CASCADIA_CPU_ONLY_TESTS"
CASCADIA_DEVICE_ENV = "CASCADIA_DEVICE"
CUDA_VISIBLE_DEVICES_ENV = "CUDA_VISIBLE_DEVICES"


class CpuOnlyTestViolation(RuntimeError):
    """Raised before an accelerator can be requested in a CPU-only session."""


def cpu_only_tests_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Return whether the strict CPU-only test boundary is enabled.

    Only ``0``/unset and ``1`` are accepted. Treating misspellings such as
    ``true`` as false would silently remove the safety boundary, so malformed
    values fail closed.
    """

    source = os.environ if environ is None else environ
    raw = source.get(CPU_ONLY_TESTS_ENV, "").strip()
    if raw in {"", "0"}:
        return False
    if raw == "1":
        return True
    raise ValueError(f"{CPU_ONLY_TESTS_ENV} must be unset, 0, or 1; got {raw!r}")


def assert_cpu_only_test_environment(
    environ: Mapping[str, str] | None = None,
) -> None:
    """Validate the complete CPU-only subprocess contract without probing hardware.

    This is intentionally stricter than :func:`require_cpu_test_device`: it is
    the entry guard for validation scripts, so all three environment controls
    must be explicitly present. Merely hiding CUDA is insufficient on hosts
    with another accelerator backend such as MPS.
    """

    source = os.environ if environ is None else environ
    if not cpu_only_tests_enabled(source):
        raise CpuOnlyTestViolation(f"{CPU_ONLY_TESTS_ENV}=1 is required")
    configured_device = source.get(CASCADIA_DEVICE_ENV)
    if configured_device != "cpu":
        raise CpuOnlyTestViolation(
            f"{CASCADIA_DEVICE_ENV} must be exactly 'cpu'; got {configured_device!r}"
        )
    cuda_visibility = source.get(CUDA_VISIBLE_DEVICES_ENV)
    if cuda_visibility != "":
        raise CpuOnlyTestViolation(
            f"{CUDA_VISIBLE_DEVICES_ENV} must be explicitly empty; got {cuda_visibility!r}"
        )


def require_cpu_test_device(
    device_name: str,
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Reject non-CPU devices before any Torch import or availability query.

    ``auto`` is intentionally rejected: resolving it would itself require
    querying accelerator state and would make the test boundary host-dependent.
    """

    if not cpu_only_tests_enabled(environ):
        # Preserve the pre-existing production path byte-for-byte when the
        # test-only boundary is disabled; validation remains the caller's job.
        return device_name
    if not isinstance(device_name, str) or not device_name.strip():
        raise ValueError("device_name must be a non-empty string")
    normalized = device_name.strip().lower()
    if normalized != "cpu":
        raise CpuOnlyTestViolation(
            f"{CPU_ONLY_TESTS_ENV}=1 forbids device {device_name!r}; "
            "request device='cpu' explicitly"
        )
    return normalized


def skip_accelerator_test(test_case: object, device_name: str) -> None:
    """Skip an intentional accelerator test before it imports a device stack."""

    if not cpu_only_tests_enabled():
        return
    skip = getattr(test_case, "skipTest", None)
    if skip is None:
        _missing_skip_test(test_case)
    skip(
        f"{CPU_ONLY_TESTS_ENV}=1: intentional {device_name.upper()} test is outside "
        "the authorized CPU-only session"
    )


def _missing_skip_test(test_case: object) -> NoReturn:
    raise TypeError(f"test_case does not provide skipTest(): {type(test_case).__name__}")
