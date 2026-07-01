from __future__ import annotations

import inspect
from typing import Any

import pytest
from r2_d0 import cli, runtime


def test_full_policy_wrapper_enables_scoped_egress_trace(monkeypatch: Any) -> None:
    observed: dict[str, Any] = {}

    def fake_probe(packet: Any, runner: Any, **options: Any) -> dict[str, Any]:
        observed.update({"packet": packet, "runner": runner, "options": options})
        return {
            "egress_guard": {"network_after_sha256": "a", "network_before_sha256": "a"},
            "network_lifecycle": {
                "bridge_projection_sha256": "bridge",
                "initial_mode": "warm",
                "status": "exact-restoration",
            },
            "status": "pass",
        }

    monkeypatch.setattr(runtime, "buildkit_probe", fake_probe)
    packet = {"identity": "packet"}
    runner = object()

    result = runtime.full_policy_buildkit_probe(packet, runner)
    assert result["status"] == "pass"
    assert result["network_lifecycle"]["full_policy_probe_count"] == 1
    assert observed == {
        "packet": packet,
        "runner": runner,
        "options": {"allow_docker_lazy_init": True, "egress_trace": True},
    }


def test_full_policy_cold_transition_requires_strict_warm_second_probe(
    monkeypatch: Any,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_probe(_packet: Any, _runner: Any, **options: Any) -> dict[str, Any]:
        calls.append(options)
        if len(calls) == 1:
            return {
                "egress_guard": {
                    "network_before_sha256": "cold",
                    "network_after_sha256": "warm",
                },
                "network_lifecycle": {
                    "bridge_projection_sha256": "bridge",
                    "initial_mode": "cold",
                    "status": "validated-lazy-initialization",
                    "transition": "cold-to-warm",
                },
                "status": "pass",
            }
        return {
            "egress_guard": {
                "network_before_sha256": "warm",
                "network_after_sha256": "warm",
            },
            "network_lifecycle": {
                "bridge_projection_sha256": "bridge",
                "initial_mode": "warm",
                "status": "exact-restoration",
            },
            "status": "pass",
        }

    monkeypatch.setattr(runtime, "buildkit_probe", fake_probe)
    result = runtime.full_policy_buildkit_probe({}, object())
    assert calls == [
        {"allow_docker_lazy_init": True, "egress_trace": True},
        {"egress_trace": True},
    ]
    assert result["network_lifecycle"]["full_policy_probe_count"] == 2
    assert result["cold_initialization_probe"]["network_lifecycle"]["initial_mode"] == "cold"


def test_full_policy_rejects_cold_to_warm_continuity_break(monkeypatch: Any) -> None:
    results = iter(
        [
            {
                "egress_guard": {
                    "network_before_sha256": "cold",
                    "network_after_sha256": "warm-a",
                },
                "network_lifecycle": {
                    "bridge_projection_sha256": "bridge",
                    "initial_mode": "cold",
                    "status": "validated-lazy-initialization",
                    "transition": "cold-to-warm",
                },
            },
            {
                "egress_guard": {
                    "network_before_sha256": "warm-b",
                    "network_after_sha256": "warm-b",
                },
                "network_lifecycle": {
                    "bridge_projection_sha256": "bridge",
                    "initial_mode": "warm",
                    "status": "exact-restoration",
                },
            },
        ]
    )
    monkeypatch.setattr(runtime, "buildkit_probe", lambda *_args, **_kwargs: next(results))
    with pytest.raises(runtime.D0Error, match="continuity differs"):
        runtime.full_policy_buildkit_probe({}, object())


def test_full_policy_does_not_run_second_probe_after_first_failure(monkeypatch: Any) -> None:
    calls = 0

    def fail_first(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        raise runtime.D0Error("first probe failed")

    monkeypatch.setattr(runtime, "buildkit_probe", fail_first)
    with pytest.raises(runtime.D0Error, match="first probe failed"):
        runtime.full_policy_buildkit_probe({}, object())
    assert calls == 1


def test_diagnostic_full_policy_dispatches_shared_wrapper(monkeypatch: Any) -> None:
    observed: list[tuple[Any, Any]] = []

    def fake_full_policy(packet: Any, runner: Any) -> dict[str, Any]:
        observed.append((packet, runner))
        return {"mode": "shared-full-policy"}

    def forbidden_generic(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("full-policy diagnostic bypassed the shared entry point")

    monkeypatch.setattr(cli, "full_policy_buildkit_probe", fake_full_policy)
    monkeypatch.setattr(cli, "buildkit_probe", forbidden_generic)
    packet = {"identity": "packet"}
    runner = object()

    assert cli._run_live_buildkit_probe(
        "live-buildkit-policy-egress-trace-probe",
        packet,
        runner,
    ) == {"mode": "shared-full-policy"}
    assert observed == [(packet, runner)]


def test_canonical_verification_uses_shared_full_policy_entry_point() -> None:
    source = inspect.getsource(runtime.verify_positive_runtime)

    assert "full_policy_buildkit_probe(packet, runner)" in source
    assert '"feature_probe": buildkit_probe(packet, runner)' not in source


def test_nonpolicy_diagnostic_modes_remain_explicit(monkeypatch: Any) -> None:
    observed: dict[str, Any] = {}

    def fake_generic(packet: Any, runner: Any, **options: Any) -> dict[str, Any]:
        observed.update(options)
        return {"status": "pass"}

    monkeypatch.setattr(cli, "buildkit_probe", fake_generic)
    monkeypatch.setattr(
        cli,
        "full_policy_buildkit_probe",
        lambda *_args: (_ for _ in ()).throw(AssertionError("wrong shared mode")),
    )

    assert cli._run_live_buildkit_probe(
        "live-buildkit-policy-output-inventory-probe",
        {},
        object(),
    ) == {"status": "pass"}
    assert observed == {
        "resolver_tls_only": False,
        "egress_trace": True,
        "output_inventory_only": True,
        "attestation_inventory_only": False,
    }
