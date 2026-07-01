from __future__ import annotations

import subprocess

import pytest
import r2_d0_network_inventory as inventory


def test_classifies_docker_default_bridge_and_raw_prerouting() -> None:
    snapshot = {
        "parsed": {
            "addresses": [
                {
                    "ifname": "docker0",
                    "addr_info": [
                        {
                            "family": "inet6",
                            "local": "fe80::1",
                            "prefixlen": 64,
                            "scope": "link",
                        }
                    ],
                }
            ],
            "docker_network_inspect": [{"Name": "bridge", "Driver": "bridge"}],
            "nftables": {
                "nftables": [
                    {"table": {"family": "ip", "name": "raw"}},
                    {
                        "chain": {
                            "family": "ip",
                            "table": "raw",
                            "name": "PREROUTING",
                            "hook": "prerouting",
                        }
                    },
                ]
            },
        }
    }
    result = inventory.classify_network_ownership(snapshot)
    assert result["docker0"]["owner"] == "docker-daemon-libnetwork"
    assert result["docker0"]["link_local_ipv6"][0]["local"] == "fe80::1"
    assert result["ip_raw_prerouting"]["present"] is True
    assert result["ip_raw_prerouting"]["owner"] == "docker-daemon-firewall-backend"


def test_does_not_assign_docker_ownership_without_bridge_evidence() -> None:
    snapshot = {
        "parsed": {
            "addresses": [{"ifname": "docker0", "addr_info": []}],
            "docker_network_inspect": [],
            "nftables": {"nftables": [{"table": {"family": "ip", "name": "raw"}}]},
        }
    }
    result = inventory.classify_network_ownership(snapshot)
    assert result["docker0"]["owner"] == "unclassified"
    assert result["ip_raw_prerouting"]["owner"] == "unclassified"


def test_failed_command_preserves_bounded_stdout_and_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        inventory.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 23, stdout=b"partial-output", stderr=b"exact-root-cause"
        ),
    )
    with pytest.raises(inventory.InventoryCommandError) as captured:
        inventory._completed(["/absolute/tool", "arg"])
    assert captured.value.evidence["returncode"] == 23
    assert captured.value.evidence["stdout"] == "partial-output"
    assert captured.value.evidence["stderr"] == "exact-root-cause"
    assert captured.value.evidence["stderr_sha256"] == inventory.sha256_bytes(b"exact-root-cause")
