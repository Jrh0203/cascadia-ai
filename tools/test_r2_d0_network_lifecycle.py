from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from typing import Optional

import pytest
from r2_d0 import runtime
from r2_d0.canonical import D0Error, canonical_json, sha256_bytes
from r2_d0.runtime import (
    _docker_default_bridge_inventory,
    _network_boundary_snapshot,
    _validate_docker_network_lifecycle,
)

MAC = "a2:70:b2:1d:d6:ed"
LINK_LOCAL = "fe80::a070:b2ff:fe1d:d6ed"


def _snapshot(*, warm: bool) -> dict:
    docker_addresses = [
        {
            "broadcast": "172.17.255.255",
            "family": "inet",
            "label": "docker0",
            "local": "172.17.0.1",
            "prefixlen": 16,
            "scope": "global",
        }
    ]
    routes = [{"dev": "eth0", "dst": "default", "gateway": "192.168.5.2"}]
    nftables = [{"table": {"family": "inet", "name": "baseline"}}]
    timers = [
        {
            "family": "inet",
            "ifindex": 2,
            "ifname": "eth0",
            "local": "192.168.5.1",
            "prefixlen": 24,
            "preferred_life_time": 3600,
            "valid_life_time": 3600,
        },
        {
            "family": "inet",
            "ifindex": 3,
            "ifname": "docker0",
            "local": "172.17.0.1",
            "prefixlen": 16,
            "preferred_life_time": 0xFFFFFFFF,
            "valid_life_time": 0xFFFFFFFF,
        },
    ]
    if warm:
        docker_addresses.append(
            {
                "family": "inet6",
                "local": LINK_LOCAL,
                "prefixlen": 64,
                "scope": "link",
            }
        )
        routes.extend(
            [
                {
                    "dev": "docker0",
                    "dst": "fe80::/64",
                    "flags": ["linkdown"],
                    "metric": 256,
                    "pref": "medium",
                    "protocol": "kernel",
                },
                {
                    "dev": "docker0",
                    "dst": LINK_LOCAL,
                    "flags": [],
                    "metric": 0,
                    "pref": "medium",
                    "protocol": "kernel",
                    "table": "local",
                    "type": "local",
                },
                {
                    "dev": "docker0",
                    "dst": "ff00::/8",
                    "flags": ["linkdown"],
                    "metric": 256,
                    "pref": "medium",
                    "protocol": "kernel",
                    "table": "local",
                    "type": "multicast",
                },
            ]
        )
        nftables.extend(
            [
                {"table": {"family": "ip", "name": "raw"}},
                {
                    "chain": {
                        "family": "ip",
                        "hook": "prerouting",
                        "name": "PREROUTING",
                        "policy": "accept",
                        "prio": -300,
                        "table": "raw",
                        "type": "filter",
                    }
                },
            ]
        )
        timers.append(
            {
                "family": "inet6",
                "ifindex": 3,
                "ifname": "docker0",
                "local": LINK_LOCAL,
                "prefixlen": 64,
                "preferred_life_time": 0xFFFFFFFF,
                "valid_life_time": 0xFFFFFFFF,
            }
        )
    state = {
        "addresses": [
            {"addr_info": [], "address": "00:00:00:00:00:00", "ifname": "lo"},
            {"addr_info": [], "address": "52:55:55:e2:c0:2f", "ifname": "eth0"},
            {"addr_info": docker_addresses, "address": MAC, "ifname": "docker0"},
        ],
        "routes": routes,
        "ruleset": {"nftables": nftables},
    }
    return {
        "captured_monotonic_ns": 1_000_000_000,
        "lease_timers": timers,
        "state": state,
        "state_sha256": sha256_bytes(canonical_json(state)),
    }


def _bridge(identity: str = "a" * 64) -> dict:
    return {"id": identity, "projection_sha256": sha256_bytes(identity.encode())}


def _after(snapshot: dict) -> dict:
    value = copy.deepcopy(snapshot)
    value["captured_monotonic_ns"] += 2_000_000_000
    value["lease_timers"][0]["preferred_life_time"] -= 2
    value["lease_timers"][0]["valid_life_time"] -= 2
    return value


def test_exact_cold_to_warm_transition_is_accepted_once() -> None:
    before = _snapshot(warm=False)
    after = _after(_snapshot(warm=True))
    lifecycle, lease = _validate_docker_network_lifecycle(
        before,
        after,
        bridge_before=_bridge(),
        bridge_after=_bridge(),
        allow_cold_transition=True,
    )
    assert lifecycle["status"] == "validated-lazy-initialization"
    assert lifecycle["transition"] == "cold-to-warm"
    assert lifecycle["docker0_link_local"]["local"] == LINK_LOCAL
    assert lease["status"] == "pass"


def test_warm_state_requires_exact_restoration() -> None:
    before = _snapshot(warm=True)
    after = _after(before)
    lifecycle, _lease = _validate_docker_network_lifecycle(
        before,
        after,
        bridge_before=_bridge(),
        bridge_after=_bridge(),
        allow_cold_transition=False,
    )
    assert lifecycle["initial_mode"] == "warm"
    assert lifecycle["status"] == "exact-restoration"


def test_cold_state_must_initialize_when_authorized() -> None:
    before = _snapshot(warm=False)
    with pytest.raises(D0Error, match="did not reach"):
        _validate_docker_network_lifecycle(
            before,
            _after(before),
            bridge_before=_bridge(),
            bridge_after=_bridge(),
            allow_cold_transition=True,
        )


@pytest.mark.parametrize("mutation", ("extra-route", "raw-rule", "finite-lease"))
def test_cold_transition_rejects_extra_or_incomplete_state(mutation: str) -> None:
    before = _snapshot(warm=False)
    after = _after(_snapshot(warm=True))
    if mutation == "extra-route":
        after["state"]["routes"].append({"dev": "docker0", "dst": "2001:db8::/64"})
    elif mutation == "raw-rule":
        after["state"]["ruleset"]["nftables"].append(
            {"rule": {"family": "ip", "table": "raw", "chain": "PREROUTING"}}
        )
    else:
        after["lease_timers"][-1]["valid_life_time"] = 60
    with pytest.raises(D0Error):
        _validate_docker_network_lifecycle(
            before,
            after,
            bridge_before=_bridge(),
            bridge_after=_bridge(),
            allow_cold_transition=True,
        )


def test_cold_transition_rejects_non_eui64_link_local() -> None:
    before = _snapshot(warm=False)
    after = _after(_snapshot(warm=True))
    after["state"]["addresses"][2]["addr_info"][1]["local"] = "fe80::1"
    with pytest.raises(D0Error, match="IPv6 address schema"):
        _validate_docker_network_lifecycle(
            before,
            after,
            bridge_before=_bridge(),
            bridge_after=_bridge(),
            allow_cold_transition=True,
        )


def test_bridge_identity_drift_is_rejected() -> None:
    with pytest.raises(D0Error, match="bridge identity drifted"):
        _validate_docker_network_lifecycle(
            _snapshot(warm=False),
            _after(_snapshot(warm=True)),
            bridge_before=_bridge("a" * 64),
            bridge_after=_bridge("b" * 64),
            allow_cold_transition=True,
        )


def _bridge_inspect(*, containers: Optional[dict] = None) -> bytes:  # noqa: UP045
    return json.dumps(
        [
            {
                "Attachable": False,
                "ConfigOnly": False,
                "Containers": {} if containers is None else containers,
                "Driver": "bridge",
                "EnableIPv4": True,
                "EnableIPv6": False,
                "IPAM": {
                    "Config": [{"Gateway": "172.17.0.1", "Subnet": "172.17.0.0/16"}],
                    "Driver": "default",
                    "Options": None,
                },
                "Id": "a" * 64,
                "Ingress": False,
                "Internal": False,
                "Name": "bridge",
                "Options": runtime.DOCKER_DEFAULT_BRIDGE_OPTIONS,
                "Scope": "local",
            }
        ]
    ).encode()


class _Runner:
    def __init__(self, stdout: bytes):
        self.stdout = stdout

    def run(self, argv: list[str], **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(argv=argv, stdout=self.stdout)


def test_default_bridge_inventory_binds_exact_empty_bridge() -> None:
    result = _docker_default_bridge_inventory(_Runner(_bridge_inspect()))
    assert result["id"] == "a" * 64
    assert result["containers"] == {}
    assert result["options"]["com.docker.network.bridge.name"] == "docker0"


def test_default_bridge_inventory_rejects_attached_container() -> None:
    with pytest.raises(D0Error, match="ownership or schema"):
        _docker_default_bridge_inventory(
            _Runner(_bridge_inspect(containers={"container": {"Name": "unexpected"}}))
        )


def test_boundary_snapshot_rejects_inspection_side_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshots = iter([_snapshot(warm=False), _snapshot(warm=True)])
    monkeypatch.setattr(runtime, "_guest_network_snapshot", lambda _runner: next(snapshots))
    monkeypatch.setattr(
        runtime,
        "_docker_default_bridge_inventory",
        lambda _runner: _bridge(),
    )
    with pytest.raises(D0Error, match="inspection mutated"):
        _network_boundary_snapshot(object())


def test_warm_host_rejects_any_additional_network_change() -> None:
    before = _snapshot(warm=True)
    after = _after(before)
    after["state"]["routes"].append({"dev": "docker0", "dst": "2001:db8::/64"})
    with pytest.raises(D0Error, match="unauthorized"):
        _validate_docker_network_lifecycle(
            before,
            after,
            bridge_before=_bridge(),
            bridge_after=_bridge(),
            allow_cold_transition=False,
        )
