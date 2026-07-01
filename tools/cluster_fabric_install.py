#!/usr/bin/env python3
"""Install the pinned Bacalhau fabric without creating a workload dispatch path."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import secrets
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
BACALHAU_SHA256 = "adb62f07b9e0ef2122f11714ba9bc233c8a4e36d61b4044603c7dbea638bd7c7"
JOHN1_ROOT = Path("/Users/johnherrick/cascadia-bench/orchestrator")
JOHN1_BINARY = JOHN1_ROOT / "bacalhau/v1.9.0/bin/bacalhau"
NODES = {
    "john2": ("100.100.43.38", "/Users/john2/cascadia-cluster"),
    "john3": ("100.71.97.55", "/Users/john3/cascadia-cluster"),
    "john4": ("100.118.7.103", "/Users/john4/cascadia-cluster"),
}


def _run(*args: str, input_bytes: bytes | None = None) -> None:
    subprocess.run(args, input=input_bytes, check=True)


def _verify_binary(path: Path) -> None:
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != BACALHAU_SHA256:
        raise SystemExit(f"Bacalhau binary checksum differs: {observed}")


def _secrets() -> str:
    return "\n".join(
        [
            f"BACALHAU_COMPUTE_TOKEN={secrets.token_hex(32)}",
            f"MINIO_ROOT_USER=cascadia{secrets.token_hex(8)}",
            f"MINIO_ROOT_PASSWORD={secrets.token_hex(32)}",
            "AWS_REGION=us-east-1",
            "AWS_DEFAULT_REGION=us-east-1",
            "AWS_ENDPOINT_URL_S3=http://100.110.109.6:9000",
            "AWS_ACCESS_KEY_ID=$MINIO_ROOT_USER",
            "AWS_SECRET_ACCESS_KEY=$MINIO_ROOT_PASSWORD",
            "",
        ]
    )


def _render(template: Path, replacements: dict[str, str]) -> bytes:
    value = template.read_text()
    for source, destination in replacements.items():
        value = value.replace(source, destination)
    if "__" in value:
        unresolved = sorted(part for part in value.split() if "__" in part)
        raise SystemExit(f"unresolved template fields in {template}: {unresolved}")
    return value.encode()


def _write_private(path: Path, value: bytes, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}")
    temporary.write_bytes(value)
    temporary.chmod(mode)
    os.replace(temporary, path)


def _install_john1(*, start: bool) -> None:
    root = JOHN1_ROOT
    for directory in ("bin", "config", "logs", "state/bacalhau", "state/registry", "state/minio"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    _verify_binary(JOHN1_BINARY)
    shutil.copy2(JOHN1_BINARY, root / "bin/bacalhau")
    shutil.copy2(REPOSITORY / "infra/bacalhau/run-node.zsh", root / "bin/run-node.zsh")
    shutil.copy2(REPOSITORY / "infra/bacalhau/launchd-entry.zsh", root / "bin/launchd-entry.zsh")
    shutil.copy2(REPOSITORY / "infra/bacalhau/run-forever.zsh", root / "bin/run-forever.zsh")
    shutil.copy2(REPOSITORY / "infra/minio/run-storage.zsh", root / "bin/run-storage.zsh")
    (root / "bin/run-node.zsh").chmod(0o755)
    (root / "bin/launchd-entry.zsh").chmod(0o755)
    (root / "bin/run-forever.zsh").chmod(0o755)
    (root / "bin/run-storage.zsh").chmod(0o755)
    shutil.copy2(REPOSITORY / "infra/bacalhau/orchestrator.yaml", root / "config/bacalhau.yaml")
    shutil.copy2(REPOSITORY / "infra/registry/config.yml", root / "config/registry.yml")
    secret_path = root / "config/secrets.env"
    if not secret_path.exists():
        _write_private(secret_path, _secrets().encode())
    plist = _render(
        REPOSITORY / "infra/bacalhau/com.cascadia.bacalhau.plist.in",
        {
            "__ROOT__": str(root),
            "__HOME__": str(Path.home()),
            "__ROLE__": "orchestrator",
            "__DOCKER_HOST__": (
                "unix:///Users/johnherrick/.local/share/cascadia-r2/colima/cascadia-r2/docker.sock"
            ),
        },
    )
    launch_agent = Path.home() / "Library/LaunchAgents/com.cascadia.bacalhau.plist"
    _write_private(launch_agent, plist, 0o644)
    if start:
        _stop_local_supervisor(root)
        domain = f"gui/{os.getuid()}"
        subprocess.run(
            ["launchctl", "bootout", f"{domain}/com.cascadia.bacalhau"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        bootstrap = subprocess.run(
            ["launchctl", "bootstrap", domain, str(launch_agent)],
            check=False,
        )
        if bootstrap.returncode:
            _start_local_supervisor(root)


def _stop_local_supervisor(root: Path) -> None:
    pid_file = root / "state/bacalhau-supervisor.pid"
    if not pid_file.is_file():
        return
    try:
        pid = int(pid_file.read_text().strip())
    except ValueError:
        pid_file.unlink(missing_ok=True)
        return
    children = subprocess.run(
        ["pgrep", "-P", str(pid)],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.split()
    for child in children:
        with contextlib.suppress(OSError, ValueError):
            os.kill(int(child), 15)
    with contextlib.suppress(OSError):
        os.kill(pid, 15)
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except OSError:
            break
        time.sleep(0.1)
    pid_file.unlink(missing_ok=True)


def _start_local_supervisor(root: Path) -> None:
    """Run the same persistent supervisor when no GUI launchd domain exists."""

    pid_file = root / "state/bacalhau-supervisor.pid"
    if pid_file.is_file():
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)
            return
        except (OSError, ValueError):
            pid_file.unlink(missing_ok=True)
    environment = dict(os.environ)
    environment.update(
        {
            "HOME": str(Path.home()),
            "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
            "CASCADIA_CLUSTER_ROOT": str(root),
            "CASCADIA_BACALHAU_ROLE": "orchestrator",
            "DOCKER_HOST": (
                "unix:///Users/johnherrick/.local/share/cascadia-r2/colima/"
                "cascadia-r2/docker.sock"
            ),
        }
    )
    with (root / "logs/bacalhau-supervisor.stdout.log").open("ab") as stdout, (
        root / "logs/bacalhau-supervisor.stderr.log"
    ).open("ab") as stderr:
        subprocess.Popen(
            [str(root / "bin/run-forever.zsh")],
            cwd=root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )


def _remote_write(host: str, path: str, value: bytes, mode: str = "600") -> None:
    parent = str(Path(path).parent)
    _run("ssh", host, "mkdir", "-p", parent)
    with tempfile.NamedTemporaryFile() as stream:
        stream.write(value)
        stream.flush()
        _run("scp", "-q", stream.name, f"{host}:{path}")
    _run("ssh", host, "chmod", mode, path)


def _remote_zsh(host: str, command: str) -> None:
    """Preserve one complete command as zsh's `-c` argument across SSH."""

    _run("ssh", host, "/bin/zsh", "-lc", shlex.quote(command))


def _install_compute(host: str, ip: str, root: str, secrets_value: bytes, *, start: bool) -> None:
    home = str(Path(root).parent)
    _run(
        "ssh",
        host,
        "mkdir",
        "-p",
        f"{root}/bin",
        f"{root}/config",
        f"{root}/logs",
        f"{root}/state/bacalhau",
        f"{home}/Library/LaunchAgents",
    )
    _run("scp", "-q", str(JOHN1_BINARY), f"{host}:{root}/bin/bacalhau")
    _remote_write(
        host,
        f"{root}/bin/run-node.zsh",
        (REPOSITORY / "infra/bacalhau/run-node.zsh").read_bytes(),
        "755",
    )
    for script in ("launchd-entry.zsh", "run-forever.zsh"):
        _remote_write(
            host,
            f"{root}/bin/{script}",
            (REPOSITORY / f"infra/bacalhau/{script}").read_bytes(),
            "755",
        )
    config = _render(
        REPOSITORY / "infra/bacalhau/compute.yaml.in",
        {"__TAILSCALE_IP__": ip, "__NODE_NAME__": host},
    )
    _remote_write(host, f"{root}/config/bacalhau.yaml", config)
    _remote_write(host, f"{root}/config/secrets.env", secrets_value)
    plist = _render(
        REPOSITORY / "infra/bacalhau/com.cascadia.bacalhau.plist.in",
        {
            "__ROOT__": root,
            "__HOME__": home,
            "__ROLE__": "compute",
            "__DOCKER_HOST__": f"unix://{home}/.colima/default/docker.sock",
        },
    )
    launch_agent = f"{home}/Library/LaunchAgents/com.cascadia.bacalhau.plist"
    _remote_write(host, launch_agent, plist, "644")
    _run("ssh", host, "shasum", "-a", "256", f"{root}/bin/bacalhau")
    if start:
        remote_uid = subprocess.check_output(["ssh", host, "id", "-u"], text=True).strip()
        domain = f"gui/{remote_uid}"
        _remote_zsh(
            host,
            (
                f"pid_file={shlex.quote(root + '/state/bacalhau-supervisor.pid')}; "
                "if [[ -s $pid_file ]]; then "
                "pid=$(<$pid_file); "
                "if kill -0 \"$pid\" 2>/dev/null; then "
                "pkill -TERM -P \"$pid\" 2>/dev/null || true; "
                "kill -TERM \"$pid\" 2>/dev/null || true; "
                "for attempt in {1..50}; do "
                "kill -0 \"$pid\" 2>/dev/null || break; sleep 0.1; done; "
                "fi; rm -f $pid_file; fi"
            ),
        )
        subprocess.run(
            ["ssh", host, "launchctl", "bootout", f"{domain}/com.cascadia.bacalhau"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        bootstrap = subprocess.run(
            ["ssh", host, "launchctl", "bootstrap", domain, launch_agent],
            check=False,
        )
        if bootstrap.returncode:
            # Headless macOS users have a user bootstrap domain but no Aqua domain.
            # The same staged LaunchAgent takes over automatically at next login.
            docker_host = f"unix://{home}/.colima/default/docker.sock"
            command = " ".join(
                shlex.quote(part)
                for part in (
                    "env",
                    f"HOME={home}",
                    "PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
                    f"CASCADIA_CLUSTER_ROOT={root}",
                    "CASCADIA_BACALHAU_ROLE=compute",
                    f"DOCKER_HOST={docker_host}",
                    f"{root}/bin/run-forever.zsh",
                )
            )
            remote = (
                f"test ! -s {shlex.quote(root + '/state/bacalhau-supervisor.pid')} || "
                f"! kill -0 $(cat {shlex.quote(root + '/state/bacalhau-supervisor.pid')}) "
                f"2>/dev/null || exit 0; nohup {command} "
                f">{shlex.quote(root + '/logs/bacalhau-supervisor.stdout.log')} "
                f"2>{shlex.quote(root + '/logs/bacalhau-supervisor.stderr.log')} </dev/null &"
            )
            _remote_zsh(host, remote)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-start", action="store_true", help="install files without launchd bootstrap"
    )
    parser.add_argument("--skip-john1", action="store_true")
    args = parser.parse_args()
    _verify_binary(JOHN1_BINARY)
    if not args.skip_john1:
        _install_john1(start=not args.no_start)
    secrets_path = JOHN1_ROOT / "config/secrets.env"
    if not secrets_path.exists():
        raise SystemExit("john1 secrets do not exist; install john1 first")
    secrets_value = secrets_path.read_bytes()
    for host, (ip, root) in NODES.items():
        _install_compute(host, ip, root, secrets_value, start=not args.no_start)
    print("Bacalhau v1.9.0 fabric installed on john1, john2, john3, and john4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
