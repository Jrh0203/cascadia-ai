import json
import tempfile
import unittest
from pathlib import Path

from cascadiav3 import harvest_d1_tranche as harvest


def census_row(seed, ply, hard, gap=0.02, se=0.05, eligible=True):
    return {
        "type": "hard_root",
        "seed": seed,
        "ply": ply,
        "eligible": eligible,
        "hard": hard,
        "top1_top2_gap": gap,
        "pairwise_se": se,
    }


def write_census(path: Path, rows) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def synthetic_census(seed_count=200, plies=80, hard_every=2):
    """Roots across many games: even plies hard, odd plies non-hard."""
    rows = []
    for index in range(seed_count):
        seed = 3_000_000 + index
        for ply in range(plies):
            hard = (ply % hard_every) == 0
            gap = 0.01 + 0.001 * (ply % 17)
            se = 0.05 + 0.001 * (ply % 13)
            rows.append(census_row(seed, ply, hard, gap=gap, se=se))
    return rows


class HarvestD1TrancheTest(unittest.TestCase):
    def run_harvest(self, rows):
        with tempfile.TemporaryDirectory() as tempdir:
            census = Path(tempdir) / "hard_roots.jsonl"
            write_census(census, rows)
            registry = harvest.harvest(census, Path(tempdir) / "out")
            tranche = [
                json.loads(line)
                for line in (Path(tempdir) / "out" / "d1_tranche_probe_roots.jsonl")
                .read_text()
                .splitlines()
            ]
            sentinel = [
                json.loads(line)
                for line in (Path(tempdir) / "out" / "d1_sentinel_probe_roots.jsonl")
                .read_text()
                .splitlines()
            ]
            return registry, tranche, sentinel

    def test_full_pool_meets_targets_and_caps(self):
        registry, tranche, sentinel = self.run_harvest(synthetic_census(seed_count=2000))
        self.assertEqual(registry["tranche"]["selected"], 15000)
        self.assertEqual(registry["tranche"]["shortfall"], {})
        self.assertEqual(len(tranche), 15000)
        self.assertEqual(registry["sentinel"]["selected"], len(sentinel))
        self.assertGreaterEqual(len(sentinel), 1490)
        self.assertLessEqual(len(sentinel), 1510)
        # Per-game caps: nothing over the top-up cap; tranche and sentinel
        # budgets are independent by design.
        per_game = {}
        for row in tranche:
            per_game[row["seed"]] = per_game.get(row["seed"], 0) + 1
        self.assertLessEqual(max(per_game.values()), harvest.TOP_UP_PER_GAME_CAP)
        # Tranche is disjoint from sentinel (hard vs non-hard pools).
        tranche_keys = {(row["seed"], row["ply"]) for row in tranche}
        sentinel_keys = {(row["seed"], row["ply"]) for row in sentinel}
        self.assertFalse(tranche_keys & sentinel_keys)
        self.assertEqual(len(tranche_keys), len(tranche))

    def test_phase_split_matches_frozen_targets(self):
        registry, tranche, _ = self.run_harvest(synthetic_census(seed_count=2000))
        phases = {"opening": 0, "mid": 0, "late": 0}
        for row in tranche:
            tile_count = 3 + row["ply"] // 4
            phases[harvest.phase_for_tile_count(tile_count)] += 1
        self.assertEqual(phases, harvest.TRANCHE_PHASE_TARGETS)

    def test_determinism_byte_identical_masks(self):
        rows = synthetic_census(seed_count=2000)
        first, _, _ = self.run_harvest(rows)
        second, _, _ = self.run_harvest(rows)
        self.assertEqual(first["tranche_mask_sha256"], second["tranche_mask_sha256"])
        self.assertEqual(first["sentinel_mask_sha256"], second["sentinel_mask_sha256"])

    def test_scarce_pool_fills_phases_evenly_not_sequentially(self):
        registry, tranche, _ = self.run_harvest(synthetic_census(seed_count=200))
        phases = {"opening": 0, "mid": 0, "late": 0}
        for row in tranche:
            phases[harvest.phase_for_tile_count(3 + row["ply"] // 4)] += 1
        # 200 games x 16-cap = 3200 max total; every phase must get a share.
        self.assertGreater(phases["late"], 0)
        self.assertGreater(phases["opening"], 0)
        self.assertGreater(phases["mid"], 0)

    def test_shortfall_is_recorded_not_fatal(self):
        # Only 100 games x 8 hard opening roots each = 800 opening hard
        # roots; the opening quota (6000) cannot be met even at cap 16.
        rows = []
        for index in range(100):
            seed = 4_000_000 + index
            for ply in range(8):
                rows.append(census_row(seed, ply, True))
            for ply in range(8, 80):
                rows.append(census_row(seed, ply, ply % 2 == 0))
        registry, tranche, _ = self.run_harvest(rows)
        self.assertIn("opening", registry["tranche"]["shortfall"])
        self.assertLess(registry["tranche"]["selected"], 15000)

    def test_ineligible_rows_are_excluded(self):
        rows = synthetic_census(seed_count=50)
        rows.append(census_row(9_999_999, 0, True, eligible=False))
        registry, tranche, _ = self.run_harvest(rows)
        self.assertNotIn(
            (9_999_999, 0), {(row["seed"], row["ply"]) for row in tranche}
        )

    def test_first_pass_cap_respected_when_pool_is_plentiful(self):
        # With 200 games supplying 40 hard roots each (8000 hard total per
        # phase bucket roughly), the 15k tranche needs the top-up pass; with
        # 2000 games it does not, and the first-pass cap must bind.
        registry, tranche, _ = self.run_harvest(synthetic_census(seed_count=2000))
        per_game = {}
        for row in tranche:
            per_game[row["seed"]] = per_game.get(row["seed"], 0) + 1
        self.assertLessEqual(max(per_game.values()), harvest.FIRST_PASS_PER_GAME_CAP)


if __name__ == "__main__":
    unittest.main()
