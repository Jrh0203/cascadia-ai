"""Durable-first raw-games evidence contract for the Gumbel benchmark.

The 2026-07-09 seed-2027070908 category loss happened because per-seed raw
game files streamed to a process-owned temporary directory with a side-car
watcher as the only durability mechanism. These tests pin the replacement
contract: production entry points write raw files to a durable directory
derived from the report path, stale raw files are refused rather than mixed,
and only an explicit ephemeral opt-in may use a temporary directory.
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cascadiav3.torch_cascadiaformer_gumbel_benchmark import (
    default_raw_games_dir,
    prepare_raw_games_dir,
)


class DefaultRawGamesDirTest(unittest.TestCase):
    def test_derives_sibling_directory_from_report_path(self) -> None:
        out = Path("cascadiav3/reports/rules_20260709_distq_k8_n1024_d16.json")
        self.assertEqual(
            default_raw_games_dir(out),
            Path("cascadiav3/reports/rules_20260709_distq_k8_n1024_d16_raw_games"),
        )

    def test_matches_existing_raw_games_naming_convention(self) -> None:
        out = Path("/anywhere/gumbel_cycle4_gate_n256.json")
        self.assertEqual(
            default_raw_games_dir(out).name, "gumbel_cycle4_gate_n256_raw_games"
        )


class PrepareRawGamesDirTest(unittest.TestCase):
    def test_creates_missing_directory(self) -> None:
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "nested" / "arm_raw_games"
            self.assertEqual(prepare_raw_games_dir(target), target)
            self.assertTrue(target.is_dir())

    def test_accepts_existing_empty_directory_and_non_raw_files(self) -> None:
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "arm_raw_games"
            target.mkdir()
            (target / "gumbel_batch_0.stderr.log").write_text("", encoding="utf-8")
            self.assertEqual(prepare_raw_games_dir(target), target)

    def test_refuses_stale_per_seed_raw_files(self) -> None:
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "arm_raw_games"
            target.mkdir()
            (target / "gumbel_game_seed_2027070900.jsonl").write_text(
                "{}\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "refusing to mix runs"):
                prepare_raw_games_dir(target)

    def test_refuses_stale_slice_raw_files(self) -> None:
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "arm_raw_games"
            target.mkdir()
            (target / "gumbel_0.jsonl").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                prepare_raw_games_dir(target)


class CliWiringTest(unittest.TestCase):
    def test_cli_defaults_to_durable_directory_beside_out(self) -> None:
        import cascadiav3.torch_cascadiaformer_gumbel_benchmark as bench

        source = Path(bench.__file__).read_text(encoding="utf-8")
        self.assertIn("--raw-games-dir", source)
        self.assertIn("--ephemeral-raw-games", source)
        self.assertIn("default_raw_games_dir(Path(args.out))", source)

    def test_report_records_raw_games_dir_provenance(self) -> None:
        import cascadiav3.torch_cascadiaformer_gumbel_benchmark as bench

        source = Path(bench.__file__).read_text(encoding="utf-8")
        self.assertIn('"raw_games_dir": str(raw_games_dir)', source)


if __name__ == "__main__":
    unittest.main()
