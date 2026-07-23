from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.aaaaa_wildlife_catalog import solve_one
from tools.aaaaa_wildlife_exact import KNOWN_INCUMBENT_TOKENS
from tools.wildlife_catalog_sharding import load_taskset, select_shard
from tools.wildlife_catalog_taskset import build_taskset


class WildlifeCatalogShardingTest(unittest.TestCase):
    def test_shards_are_disjoint_and_reconstruct_order(self) -> None:
        tasks = [{"counts": [index, 0, 0, 0, 0]} for index in range(11)]
        shards = [
            select_shard(tasks, shard_index=index, shard_count=3) for index in range(3)
        ]
        observed = [tuple(task["counts"]) for shard in shards for task in shard]
        self.assertEqual(11, len(observed))
        self.assertEqual(11, len(set(observed)))
        for index, task in enumerate(tasks):
            self.assertIn(task, shards[index % 3])

    def test_taskset_is_ruleset_and_count_checked(self) -> None:
        canonical = [(6, 4, 6, 0, 4), (3, 6, 6, 0, 5)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": "wildlife-catalog-taskset-v1",
                        "scoring_cards": "AAAAA",
                        "task_count": 1,
                        "counts": [[3, 6, 6, 0, 5]],
                    }
                ),
                encoding="utf-8",
            )
            selected, record = load_taskset(
                path, scoring_cards="AAAAA", canonical_counts=canonical
            )
        self.assertEqual({(3, 6, 6, 0, 5)}, selected)
        self.assertEqual(1, record["task_count"])
        self.assertEqual(64, len(record["sha256"]))

    def test_taskset_rejects_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": "wildlife-catalog-taskset-v1",
                        "scoring_cards": "CBDDB",
                        "counts": [[3, 6, 6, 0, 5], [3, 6, 6, 0, 5]],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_taskset(
                    path,
                    scoring_cards="CBDDB",
                    canonical_counts=[(3, 6, 6, 0, 5)],
                )

    def test_taskset_generator_excludes_only_valid_complete_rows(self) -> None:
        result = solve_one(
            {
                "counts": [6, 4, 6, 0, 4],
                "tokens": KNOWN_INCUMBENT_TOKENS,
                "solver_workers": 1,
                "relaxation_time_limit": 1.0,
                "connected_time_limit": 1.0,
                "seed": 1,
                "proof_provenance": {"test": True},
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": "aaaaa-wildlife-optimal-catalog-v2",
                        "results": [result],
                    }
                ),
                encoding="utf-8",
            )
            payload = build_taskset("AAAAA", path)
        self.assertEqual(825, payload["task_count"])
        self.assertEqual(1, payload["completed_count"])
        self.assertNotIn([6, 4, 6, 0, 4], payload["counts"])


if __name__ == "__main__":
    unittest.main()
