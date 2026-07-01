from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[2] / "tools" / "v1_champion_benchmark.py"
SPEC = importlib.util.spec_from_file_location("v1_champion_benchmark", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_parse_players_reads_all_four_seats() -> None:
    text = """
SYMPLAYER p=2 base=94 bonus=95 hab=28 wl=60 tok=6 bear=11 elk=13 salmon=14 hawk=8 fox=14
SYMPLAYER p=0 base=98 bonus=101 hab=32 wl=62 tok=4 bear=11 elk=11 salmon=5 hawk=11 fox=24
SYMPLAYER p=3 base=94 bonus=100 hab=32 wl=59 tok=3 bear=11 elk=17 salmon=7 hawk=11 fox=13
SYMPLAYER p=1 base=91 bonus=98 hab=28 wl=63 tok=0 bear=19 elk=4 salmon=16 hawk=8 fox=16
"""
    players = MODULE.parse_players(text)
    assert [player["seat"] for player in players] == [0, 1, 2, 3]
    assert [player["base"] for player in players] == [98, 91, 94, 94]


def test_parse_players_rejects_incomplete_game() -> None:
    with pytest.raises(ValueError, match="seats"):
        MODULE.parse_players(
            "SYMPLAYER p=0 base=98 bonus=101 hab=32 wl=62 tok=4 "
            "bear=11 elk=11 salmon=5 hawk=11 fox=24"
        )
