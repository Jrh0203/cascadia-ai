"""Pure SSH and rsync command construction for the local Cascadia cluster."""

from __future__ import annotations

import shlex
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

SSH = Path("/usr/bin/ssh")
RSYNC = Path("/usr/bin/rsync")
CONNECT_OPTIONS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=8")


@dataclass(frozen=True)
class Endpoint:
    target: str
    options: tuple[str, ...] = ()

    def ssh_command(self, command: str) -> list[str]:
        return [str(SSH), *CONNECT_OPTIONS, *self.options, self.target, command]

    def rsync_shell(self) -> str:
        return shlex.join([str(SSH), *CONNECT_OPTIONS, *self.options])


ENDPOINTS = {
    "john2": (
        Endpoint("john2"),
        Endpoint(
            "john2@192.168.1.238",
            (
                "-i",
                str(Path.home() / ".ssh/john2_codex"),
                "-o",
                "IdentitiesOnly=yes",
                "-o",
                "StrictHostKeyChecking=yes",
                "-o",
                "HostKeyAlias=100.100.43.38",
            ),
        ),
    ),
    "john3": (Endpoint("john3"),),
}


def endpoints(host: str) -> tuple[Endpoint, ...]:
    return ENDPOINTS.get(host, (Endpoint(host),))


def ssh_commands(host: str, command: str) -> list[list[str]]:
    return [endpoint.ssh_command(command) for endpoint in endpoints(host)]


def rsync_commands(
    host: str,
    local_path: str,
    remote_path: str,
    *,
    upload: bool,
    delete: bool = False,
) -> list[list[str]]:
    commands = []
    for endpoint in endpoints(host):
        remote = f"{endpoint.target}:{remote_path}"
        source, destination = (local_path, remote) if upload else (remote, local_path)
        command = [str(RSYNC), "-a"]
        if delete:
            command.append("--delete")
        command.extend(["-e", endpoint.rsync_shell(), source, destination])
        commands.append(command)
    return commands


def run_candidates[Result](
    commands: list[list[str]],
    runner: Callable[[list[str]], Result],
    returncode: Callable[[Result], int],
    on_fallback: Callable[[], None],
) -> Result:
    result: Result | None = None
    for index, command in enumerate(commands):
        result = runner(command)
        if returncode(result) != 255:
            return result
        if index + 1 < len(commands):
            on_fallback()
    assert result is not None
    return result
