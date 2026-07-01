from __future__ import annotations

import shlex

import cluster_fabric_install as subject


def test_remote_zsh_preserves_the_complete_command_as_one_argument(monkeypatch) -> None:
    observed: list[tuple[str, ...]] = []
    monkeypatch.setattr(subject, "_run", lambda *args, **_kwargs: observed.append(args))
    command = "pid=$(<state/pid); kill -TERM \"$pid\"; nohup worker >/tmp/log &"
    subject._remote_zsh("john2", command)
    assert observed == [
        ("ssh", "john2", "/bin/zsh", "-lc", shlex.quote(command))
    ]


def test_compute_installer_includes_john4() -> None:
    assert subject.NODES["john4"] == ("100.118.7.103", "/Users/john4/cascadia-cluster")
