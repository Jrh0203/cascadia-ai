from __future__ import annotations

import subprocess
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
LAUNCHER = REPOSITORY / "infra/minio/run-storage.zsh"


def test_storage_launcher_is_valid_zsh() -> None:
    subprocess.run(["/bin/zsh", "-n", str(LAUNCHER)], check=True)


def test_storage_launcher_keeps_colima_ports_loopback_and_tailnet_private() -> None:
    source = LAUNCHER.read_text()
    assert '127.0.0.1:$REGISTRY_LOOPBACK_PORT:5000' in source
    assert '127.0.0.1:$MINIO_API_LOOPBACK_PORT:9000' in source
    assert '127.0.0.1:$MINIO_CONSOLE_LOOPBACK_PORT:9001' in source
    assert 'serve --bg --tcp 5000' in source
    assert 'serve --bg --tcp 9000' in source
    assert 'serve --bg --tcp 9001' in source
    assert '-p "$TAILSCALE_IP:' not in source
