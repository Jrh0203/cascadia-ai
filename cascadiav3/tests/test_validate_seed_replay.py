"""Contract tests for the one-seed raw-game replay validator."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cascadiav3.validate_seed_replay import (
    ReplayValidationError,
    install_replay,
    pinned_totals_from_report,
    validate_replay,
)

SEED = 2027070962
RULES = "cascadia_research_aaaaa_4p_card_a_no_habitat_bonus_rules_2026_07_09"


def make_decision(ply: int, action: int, refresh=None) -> dict:
    # Real ledgers use content-hash action identities, not ints.
    return {
        "type": "gumbel_decision",
        "seed": SEED,
        "ply": ply,
        "chosen_action_id": f"sha256:{action:064x}",
        "free_three_of_a_kind_choice": refresh,
        "ruleset_id": RULES,
    }


def make_score(total: int) -> dict:
    # wildlife 5 + habitat 5 + tokens: components sum exactly to total.
    return {
        "wildlife": [total - 20, 5, 5, 5, 0],
        "habitat": [1, 1, 1, 1, 1],
        "nature_tokens": 0,
        "total": total,
    }


def make_replay(totals=(97, 98, 97, 100), action_offset=0, refresh_flip=False):
    rows = []
    for ply in range(80):
        refresh = "accept" if (ply == 12 and not refresh_flip) else (
            "decline" if ply == 12 else None
        )
        rows.append(make_decision(ply, 1000 + ply + action_offset, refresh))
    rows.append(
        {
            "type": "gumbel_game_done",
            "seed": SEED,
            "ruleset_id": RULES,
            "scores": [make_score(t) for t in totals],
        }
    )
    return rows


def make_ledger():
    return [
        make_decision(ply, 1000 + ply, "accept" if ply == 12 else None)
        for ply in range(80)
    ]


class ValidateReplayTest(unittest.TestCase):
    def test_happy_path_returns_done_row(self) -> None:
        done = validate_replay(
            replayed_rows=make_replay(),
            ledger_rows=make_ledger(),
            pinned_seat_totals=[97, 98, 97, 100],
            seed=SEED,
        )
        self.assertEqual([s["total"] for s in done["scores"]], [97, 98, 97, 100])

    def test_action_mismatch_fails(self) -> None:
        with self.assertRaisesRegex(ReplayValidationError, "chosen action"):
            validate_replay(
                replayed_rows=make_replay(action_offset=1),
                ledger_rows=make_ledger(),
                pinned_seat_totals=[97, 98, 97, 100],
                seed=SEED,
            )

    def test_refresh_decision_mismatch_fails(self) -> None:
        with self.assertRaisesRegex(ReplayValidationError, "refresh decision"):
            validate_replay(
                replayed_rows=make_replay(refresh_flip=True),
                ledger_rows=make_ledger(),
                pinned_seat_totals=[97, 98, 97, 100],
                seed=SEED,
            )

    def test_seat_total_mismatch_fails(self) -> None:
        with self.assertRaisesRegex(ReplayValidationError, "pinned"):
            validate_replay(
                replayed_rows=make_replay(totals=(97, 98, 97, 99)),
                ledger_rows=make_ledger(),
                pinned_seat_totals=[97, 98, 97, 100],
                seed=SEED,
            )

    def test_category_sum_mismatch_fails(self) -> None:
        rows = make_replay()
        rows[-1]["scores"][2]["nature_tokens"] = 3  # break the component sum
        with self.assertRaisesRegex(ReplayValidationError, "components sum"):
            validate_replay(
                replayed_rows=rows,
                ledger_rows=make_ledger(),
                pinned_seat_totals=[97, 98, 97, 100],
                seed=SEED,
            )

    def test_wrong_decision_count_fails(self) -> None:
        with self.assertRaisesRegex(ReplayValidationError, "decision rows"):
            validate_replay(
                replayed_rows=make_replay()[1:],
                ledger_rows=make_ledger(),
                pinned_seat_totals=[97, 98, 97, 100],
                seed=SEED,
            )

    def test_pinned_totals_lookup(self) -> None:
        report = {
            "candidate_per_seed": [
                {"seed": SEED, "seat_scores": [97.0, 98.0, 97.0, 100.0]}
            ]
        }
        self.assertEqual(
            pinned_totals_from_report(report, SEED), [97.0, 98.0, 97.0, 100.0]
        )
        with self.assertRaisesRegex(ReplayValidationError, "no candidate_per_seed"):
            pinned_totals_from_report(report, SEED + 1)


class InstallReplayTest(unittest.TestCase):
    def _dir_with_seeds(self, tmp: str, seeds) -> Path:
        raw = Path(tmp) / "raw_games"
        raw.mkdir()
        for s in seeds:
            (raw / f"gumbel_game_seed_{s}.jsonl").write_text("{}\n", encoding="utf-8")
        return raw

    def test_installs_missing_seed(self) -> None:
        with TemporaryDirectory() as tmp:
            raw = self._dir_with_seeds(tmp, [1, 2, 4])
            replay = Path(tmp) / "replayed.jsonl"
            replay.write_text('{"type":"x"}\n', encoding="utf-8")
            target = install_replay(
                replayed_path=replay, raw_games_dir=raw, seed=3, expected_total_seeds=4
            )
            self.assertTrue(target.exists())

    def test_refuses_existing_seed(self) -> None:
        with TemporaryDirectory() as tmp:
            raw = self._dir_with_seeds(tmp, [1, 2, 3])
            replay = Path(tmp) / "replayed.jsonl"
            replay.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ReplayValidationError, "refusing to overwrite"):
                install_replay(
                    replayed_path=replay,
                    raw_games_dir=raw,
                    seed=3,
                    expected_total_seeds=4,
                )

    def test_refuses_wrong_precount(self) -> None:
        with TemporaryDirectory() as tmp:
            raw = self._dir_with_seeds(tmp, [1])
            replay = Path(tmp) / "replayed.jsonl"
            replay.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ReplayValidationError, "expected exactly"):
                install_replay(
                    replayed_path=replay,
                    raw_games_dir=raw,
                    seed=3,
                    expected_total_seeds=4,
                )


if __name__ == "__main__":
    unittest.main()
