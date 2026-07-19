"""Scoring-card selection contract for the benchmark harnesses.

The exporter binary resolves --scoring-cards {aaaaa,cbddb} to a ruleset id
that is stamped on every emitted record. These tests pin the Python side of
that contract: the harness id mapping matches the exporter's constants, the
resolved id is what report validation expects, and the flag is passed through
to the binary invocation only when non-default so default (aaaaa) invocations
stay byte-identical and replayable against older pinned binaries.
"""

import unittest
from pathlib import Path

from cascadiav3 import torch_cascadiaformer_game_benchmark as game_benchmark
from cascadiav3 import torch_cascadiaformer_gumbel_benchmark as gumbel_benchmark
from cascadiav3 import torch_cascadiaformer_search_benchmark as search_benchmark

AAAAA_ID = "cascadia_research_aaaaa_4p_card_a_no_habitat_bonus_rules_2026_07_16"
CBDDB_ID = "cascadia_research_cbddb_4p_no_habitat_bonus_rules_2026_07_19"


class RulesetIdMappingTest(unittest.TestCase):
    def test_gumbel_and_game_mappings_agree_and_pin_exporter_ids(self) -> None:
        expected = {"aaaaa": AAAAA_ID, "cbddb": CBDDB_ID}
        self.assertEqual(gumbel_benchmark.RULESET_IDS_BY_SCORING_CARDS, expected)
        self.assertEqual(game_benchmark.RULESET_IDS_BY_SCORING_CARDS, expected)

    def test_default_expected_ruleset_id_is_aaaaa(self) -> None:
        self.assertEqual(gumbel_benchmark.EXPECTED_RULESET_ID, AAAAA_ID)
        self.assertEqual(game_benchmark.EXPECTED_RULESET_ID, AAAAA_ID)

    def test_expected_ruleset_id_for_resolves_both_selections(self) -> None:
        for module in (gumbel_benchmark, game_benchmark):
            self.assertEqual(module.expected_ruleset_id_for("aaaaa"), AAAAA_ID)
            self.assertEqual(module.expected_ruleset_id_for("cbddb"), CBDDB_ID)

    def test_expected_ruleset_id_for_rejects_unknown_selection(self) -> None:
        for module in (gumbel_benchmark, game_benchmark):
            with self.assertRaises(ValueError):
                module.expected_ruleset_id_for("abcde")


class BinaryCommandPassThroughTest(unittest.TestCase):
    def test_game_benchmark_default_command_has_no_scoring_cards_flag(self) -> None:
        command = game_benchmark._binary_command(
            Path("exporter"), seed=7, max_actions=8
        )
        self.assertNotIn("--scoring-cards", command)

    def test_game_benchmark_cbddb_command_passes_flag_through(self) -> None:
        command = game_benchmark._binary_command(
            Path("exporter"), seed=7, max_actions=8, scoring_cards="cbddb"
        )
        index = command.index("--scoring-cards")
        self.assertEqual(command[index + 1], "cbddb")

    def test_search_benchmark_default_command_has_no_scoring_cards_flag(self) -> None:
        command = search_benchmark._binary_command(
            Path("exporter"),
            seed=7,
            max_actions=8,
            rollouts_per_action=1,
            rollout_top_k=1,
        )
        self.assertNotIn("--scoring-cards", command)

    def test_search_benchmark_cbddb_command_passes_flag_through(self) -> None:
        command = search_benchmark._binary_command(
            Path("exporter"),
            seed=7,
            max_actions=8,
            rollouts_per_action=1,
            rollout_top_k=1,
            scoring_cards="cbddb",
        )
        index = command.index("--scoring-cards")
        self.assertEqual(command[index + 1], "cbddb")


if __name__ == "__main__":
    unittest.main()
