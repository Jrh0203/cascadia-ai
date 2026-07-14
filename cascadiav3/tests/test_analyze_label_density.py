"""Stage 0 label-noise audit math on synthetic packed v4 shards."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def _v4_metadata() -> dict:
    return {
        "schema_id": "cascadiav3.expert_tensor_shard.v4",
        "ruleset_id": "test-ruleset",
        "source_revision": "test-source-revision",
        "mode": "gumbel_selfplay_tensor_corpus",
        "scientific_eligibility": "gumbel_selfplay_expert_iteration",
        "search": {
            "n_simulations": 4,
            "top_m": 2,
            "depth_rounds": 1,
            "determinization_samples": 2,
            "market_decision_samples": 2,
            "exact_endgame_turns": 1,
            "rollout_blend_weight": 0.5,
            "exploration": True,
            "peek": False,
            "table_total": False,
            "table_native_q": False,
            "leaf_softmix": None,
            "tta": 1,
            "k_interior": 2,
            "max_root_actions": None,
            "root_menu": 16,
        },
        "execution": {
            "rayon_threads_requested": 1,
            "rayon_current_num_threads": 1,
            "model_sessions_requested": 1,
            "shared_model_session": True,
            "seed_scheduler": "dynamic_atomic_queue",
            "model_session_topology": "one_shared_bridge_with_worker_clients",
        },
        "teacher_model": {
            "manifest": {"sha256": "1" * 64, "bytes": 10},
            "weights": {"sha256": "2" * 64, "bytes": 20},
        },
        "generator": {"sha256": "3" * 64, "bytes": 30},
        "created_unix_seconds": 1_700_000_000,
        "canonical_targets": [
            "improved_policy",
            "search_root_value",
            "exact_endgame",
            "active_seat",
            "exact_afterstate_score_decomposition_active",
        ],
    }


def _player_token_row(seat: int, tile_count: int):  # type: ignore[no-untyped-def]
    """Packed player token per feature_tensors.rs::public_token_features."""
    import numpy as np

    row = np.zeros((41,), dtype=np.float16)
    row[0] = 1.0  # token-kind one-hot: player
    row[6] = seat / 3.0  # owner_seat / 3
    row[7] = 0.0  # relative_seat 0 == active player
    row[16] = tile_count / 23.0  # tile_count / 23
    return row


def _other_token_row():  # type: ignore[no-untyped-def]
    import numpy as np

    row = np.zeros((41,), dtype=np.float16)
    row[1] = 1.0  # token-kind one-hot: placed_tile
    return row


def write_v4_fixture(path: Path, *, zero_player_tokens: bool = False) -> None:
    """Four hand-computed records, three actions each, via the real writer.

    Records (active seat, active tile_count -> own turn -> phase):
      r0: seat 0, tiles 3  -> turn 0  -> opening
      r1: seat 1, tiles 4  -> turn 1  -> opening
      r2: seat 2, tiles 22 -> turn 19 -> endgame
      r3: seat 3, tiles 22 -> turn 19 -> endgame

    Outcomes (final_score_vector[active_seat]): 20, 10, 30, 40.
    search_root_value: 22, 9, 30, 41 -> errors (2, -1, 0, 1),
    RMSE sqrt(1.5), bias +0.5. Phase outcome means: opening 15, endgame 35,
    so the within-phase baseline error is +/-5 everywhere -> baseline RMSE 5.
    r0 and r1 share one final_score_vector row, as do r2 and r3 (two
    adjacency runs of length 2 with seat +1 cycling).
    """
    import numpy as np

    from cascadiav3.expert_tensor_shards import _save_expert_tensor_shard

    seats = [0, 1, 2, 3]
    tile_counts = [3, 4, 22, 22]
    tokens = []
    for seat, tile_count in zip(seats, tile_counts, strict=True):
        player = _player_token_row(seat, tile_count)
        if zero_player_tokens:
            player = _other_token_row()
        tokens.extend([player, _other_token_row()])
    tokens = np.stack(tokens, axis=0)

    final_scores = np.asarray(
        [
            [20.0, 10.0, 55.0, 65.0],
            [20.0, 10.0, 55.0, 65.0],
            [12.0, 11.0, 30.0, 40.0],
            [12.0, 11.0, 30.0, 40.0],
        ],
        dtype=np.float32,
    )
    score_decomposition = np.zeros((4, 3, 4), dtype=np.float32)
    score_decomposition[:, 0, :] = final_scores

    target_q = np.asarray(
        [10.0, 9.6, 7.0, 20.0, 10.0, 5.0, 15.0, 14.0, 13.0, 8.0, 8.5, 3.0],
        dtype=np.float32,
    )
    q_valid = np.asarray([1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0], dtype=np.uint8)
    visits = np.asarray(
        [4.0, 2.0, 0.0, 4.0, 4.0, 2.0, 5.0, 0.0, 0.0, 2.0, 2.0, 0.0],
        dtype=np.float32,
    )
    q_variance = np.asarray(
        [1.0, 0.5, 0.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 2.0, 2.0, 0.0],
        dtype=np.float32,
    )
    improved_policy = np.asarray(
        [0.5, 0.3, 0.2, 0.7, 0.2, 0.1, 0.9, 0.06, 0.04, 0.4, 0.5, 0.1],
        dtype=np.float32,
    )
    exact_components = np.zeros((12, 3), dtype=np.float32)
    exact_components[:, 0] = 1.0

    _save_expert_tensor_shard(
        out_path=path,
        metadata=_v4_metadata(),
        tokens=tokens,
        actions=np.zeros((12, 61), dtype=np.float16),
        token_offsets=np.asarray([0, 2, 4, 6, 8], dtype=np.int64),
        action_offsets=np.asarray([0, 3, 6, 9, 12], dtype=np.int64),
        relation_edges=np.zeros((0, 3), dtype=np.int32),
        relation_offsets=np.asarray([0, 0, 0, 0, 0], dtype=np.int64),
        selected_action_index=np.asarray([0, 0, 0, 0], dtype=np.int16),
        target_q=target_q,
        target_score_to_go=target_q - exact_components.sum(axis=1),
        q_valid=q_valid,
        priors=np.full((12,), 1.0 / 3.0, dtype=np.float32),
        visits=visits,
        q_variance=q_variance,
        q_count=visits.copy(),
        truncated_count=np.zeros((12,), dtype=np.float32),
        exact_afterstate_score_active=exact_components.sum(axis=1),
        exact_afterstate_score_decomposition_active=exact_components,
        active_seat=np.asarray(seats, dtype=np.uint8),
        final_score_vector=final_scores,
        rank_vector=np.tile(np.asarray([1, 2, 3, 4], dtype=np.int16), (4, 1)),
        score_decomposition=score_decomposition,
        improved_policy=improved_policy,
        search_root_value=np.asarray([22.0, 9.0, 30.0, 41.0], dtype=np.float32),
        exact_endgame=np.asarray([0, 0, 1, 1], dtype=np.uint8),
    )


def write_v1_fixture(path: Path) -> None:
    """Minimal v1 shard: no improved_policy, no search_root_value."""
    import numpy as np

    from cascadiav3.expert_tensor_shards import _save_expert_tensor_shard

    _save_expert_tensor_shard(
        out_path=path,
        metadata={"schema_id": "cascadiav3.expert_tensor_shard.v1"},
        tokens=np.zeros((2, 41), dtype=np.float16),
        actions=np.zeros((3, 61), dtype=np.float16),
        token_offsets=np.asarray([0, 2], dtype=np.int64),
        action_offsets=np.asarray([0, 3], dtype=np.int64),
        relation_edges=np.zeros((0, 3), dtype=np.int32),
        relation_offsets=np.asarray([0, 0], dtype=np.int64),
        selected_action_index=np.asarray([0], dtype=np.int16),
        target_q=np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        target_score_to_go=np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        q_valid=np.asarray([1, 1, 0], dtype=np.uint8),
        priors=np.asarray([0.5, 0.3, 0.2], dtype=np.float32),
        visits=np.asarray([2.0, 1.0, 0.0], dtype=np.float32),
        q_variance=np.zeros((3,), dtype=np.float32),
        q_count=np.asarray([2.0, 1.0, 0.0], dtype=np.float32),
        truncated_count=np.zeros((3,), dtype=np.float32),
        exact_afterstate_score_active=np.zeros((3,), dtype=np.float32),
        final_score_vector=np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32),
        rank_vector=np.asarray([[1, 2, 3, 4]], dtype=np.int16),
        score_decomposition=np.zeros((1, 3, 4), dtype=np.float32),
    )


class AnalyzeLabelDensityTest(unittest.TestCase):
    def test_density_census_on_hand_computed_fixture(self) -> None:
        from cascadiav3.analyze_label_density import analyze

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shard.npz"
            write_v4_fixture(path)
            report = analyze(str(path))

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["records_total"], 4)
        self.assertEqual(report["records_analyzed"], 4)
        census = report["density_census"]
        # q-valid fractions per root: 2/3, 3/3, 1/3, 2/3.
        self.assertAlmostEqual(census["q_valid_fraction"]["mean"], 2.0 / 3.0, places=5)
        self.assertAlmostEqual(census["q_valid_fraction"]["median"], 2.0 / 3.0, places=5)
        self.assertAlmostEqual(census["q_valid_fraction"]["p10"], 1.0 / 3.0 + 0.1, places=5)
        self.assertAlmostEqual(census["q_valid_fraction"]["p90"], 0.9, places=5)
        # Improved-policy mass on unvisited actions: 0.2, 0.0, 0.1, 0.1.
        self.assertAlmostEqual(census["unvisited_policy_mass"]["mean"], 0.1, places=5)
        self.assertAlmostEqual(census["unvisited_policy_mass"]["median"], 0.1, places=5)
        # Top-1 visit fractions: 4/6, 4/10, 5/5, 2/4.
        self.assertAlmostEqual(
            census["top1_visit_fraction"]["mean"],
            (4.0 / 6.0 + 0.4 + 1.0 + 0.5) / 4.0,
            places=5,
        )
        self.assertEqual(census["records_without_visits"], 0)

        stratification = report["stratification"]
        self.assertEqual(stratification["mode"], "active_player_tile_count")
        self.assertEqual(stratification["unknown_phase_records"], 0)
        self.assertEqual(stratification["token_seat_mismatch_records"], 0)
        self.assertEqual(stratification["tile_count_min"], 3)
        self.assertEqual(stratification["tile_count_max"], 22)

    def test_v1_falsifier_exact_numbers(self) -> None:
        from cascadiav3.analyze_label_density import analyze

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shard.npz"
            write_v4_fixture(path)
            report = analyze(str(path))

        falsifier = report["v1_falsifier"]
        self.assertEqual(falsifier["records"], 4)
        self.assertAlmostEqual(falsifier["rmse_search_root_value"], math.sqrt(1.5), places=5)
        self.assertAlmostEqual(falsifier["bias_search_root_value"], 0.5, places=6)
        self.assertAlmostEqual(falsifier["rmse_baseline_within_phase"], 5.0, places=6)
        self.assertAlmostEqual(
            falsifier["rmse_baseline_global_mean"], math.sqrt(125.0), places=5
        )
        self.assertAlmostEqual(
            falsifier["rmse_reduction_vs_within_phase_baseline_pct"],
            100.0 * (1.0 - math.sqrt(1.5) / 5.0),
            places=3,
        )
        self.assertAlmostEqual(
            falsifier["rmse_reduction_vs_global_baseline_pct"],
            100.0 * (1.0 - math.sqrt(1.5) / math.sqrt(125.0)),
            places=3,
        )
        bar = falsifier["preregistered_bar"]
        self.assertEqual(bar["rmse_reduction_pct_required"], 20.0)
        self.assertEqual(bar["abs_bias_max"], 0.5)
        self.assertTrue(bar["passes"])

        self.assertEqual(sorted(falsifier["by_phase"]), ["endgame", "opening"])
        opening = falsifier["by_phase"]["opening"]
        self.assertEqual(opening["records"], 2)
        self.assertAlmostEqual(opening["rmse_search_root_value"], math.sqrt(2.5), places=5)
        self.assertAlmostEqual(opening["bias_search_root_value"], 0.5, places=6)
        self.assertAlmostEqual(opening["rmse_baseline_within_phase"], 5.0, places=6)
        endgame = falsifier["by_phase"]["endgame"]
        self.assertEqual(endgame["records"], 2)
        self.assertAlmostEqual(endgame["rmse_search_root_value"], math.sqrt(0.5), places=5)
        self.assertAlmostEqual(endgame["bias_search_root_value"], 0.5, places=6)

    def test_hard_root_census_exact_fractions(self) -> None:
        from cascadiav3.analyze_label_density import analyze

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shard.npz"
            write_v4_fixture(path)
            report = analyze(str(path))

        hard = report["hard_root_census"]
        # r2 has one q-valid action -> ineligible; r0/r1/r3 eligible.
        self.assertEqual(hard["records"], 4)
        self.assertEqual(hard["eligible_records"], 3)
        self.assertAlmostEqual(hard["eligible_coverage"], 0.75, places=6)
        # r0: gap 0.4 < sqrt(1/4 + 0.5/2) ~ 0.707 -> hard.
        # r1: gap 10 >= sqrt(1/4 + 1/4)         -> not hard.
        # r3: gap 0.5 < sqrt(2/2 + 2/2) ~ 1.414 -> hard.
        self.assertEqual(hard["hard_records"], 2)
        self.assertAlmostEqual(hard["hard_fraction"], 2.0 / 3.0, places=5)
        self.assertAlmostEqual(
            hard["by_phase"]["opening"]["hard_fraction"], 0.5, places=6
        )
        self.assertEqual(hard["by_phase"]["opening"]["eligible_records"], 2)
        self.assertAlmostEqual(
            hard["by_phase"]["endgame"]["hard_fraction"], 1.0, places=6
        )
        self.assertEqual(hard["by_phase"]["endgame"]["eligible_records"], 1)

        adjacency = report["trajectory_adjacency"]
        self.assertEqual(adjacency["record_count"], 4)
        self.assertEqual(adjacency["run_count"], 2)
        self.assertEqual(adjacency["max_run_length"], 2)
        self.assertAlmostEqual(adjacency["fraction_records_in_runs_ge2"], 1.0, places=6)
        self.assertAlmostEqual(adjacency["seat_cycle_fraction_within_runs"], 1.0, places=6)

    def test_max_records_subsampling_is_even_and_deterministic(self) -> None:
        from cascadiav3.analyze_label_density import analyze

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shard.npz"
            write_v4_fixture(path)
            report = analyze(str(path), max_records=2)
            repeat = analyze(str(path), max_records=2)

        # Even stride over 4 records keeps indices 0 and 2 (r0 and r2).
        self.assertEqual(report["records_analyzed"], 2)
        self.assertEqual(report["max_records"], 2)
        falsifier = report["v1_falsifier"]
        # Errors: r0 +2, r2 0 -> RMSE sqrt(2), bias +1.
        self.assertAlmostEqual(falsifier["rmse_search_root_value"], math.sqrt(2.0), places=5)
        self.assertAlmostEqual(falsifier["bias_search_root_value"], 1.0, places=6)
        # One record per phase -> degenerate zero baseline, no reduction claim.
        self.assertAlmostEqual(falsifier["rmse_baseline_within_phase"], 0.0, places=6)
        self.assertIsNone(falsifier["rmse_reduction_vs_within_phase_baseline_pct"])
        self.assertFalse(falsifier["preregistered_bar"]["passes"])
        self.assertEqual(report["v1_falsifier"], repeat["v1_falsifier"])
        # The adjacency probe still sees the full shard.
        self.assertEqual(report["trajectory_adjacency"]["record_count"], 4)

    def test_v1_shard_without_search_root_value_is_rejected(self) -> None:
        from cascadiav3.analyze_label_density import analyze

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v1.npz"
            write_v1_fixture(path)
            with self.assertRaisesRegex(ValueError, "requires v2\\+ expert tensor shards"):
                analyze(str(path))

    def test_record_index_fallback_when_player_tokens_missing(self) -> None:
        from cascadiav3.analyze_label_density import analyze

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shard.npz"
            write_v4_fixture(path, zero_player_tokens=True)
            report = analyze(str(path))

        self.assertEqual(report["stratification"]["mode"], "record_index_within_shard")
        self.assertEqual(report["stratification"]["unknown_phase_records"], 4)
        # Positions 0, 1/3, 2/3, 1 -> quartiles opening/early_mid/late_mid/endgame.
        self.assertEqual(
            sorted(report["v1_falsifier"]["by_phase"]),
            ["early_mid", "endgame", "late_mid", "opening"],
        )

    def test_cli_directory_input_and_report_files(self) -> None:
        from cascadiav3.analyze_label_density import main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_v4_fixture(root / "shard_a.npz")
            write_v4_fixture(root / "shard_b.npz")
            out = root / "report.json"
            summary = root / "report.md"
            argv = [
                "analyze_label_density",
                "--shards",
                str(root),
                "--out",
                str(out),
                "--summary-out",
                str(summary),
            ]
            with mock.patch("sys.argv", argv):
                self.assertEqual(main(), 0)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "pass")
            self.assertEqual(len(report["shards"]), 2)
            self.assertEqual(report["records_analyzed"], 8)
            self.assertEqual(report["trajectory_adjacency"]["record_count"], 8)
            markdown = summary.read_text(encoding="utf-8")
            self.assertIn("# Stage 0 Label-Noise Audit (R1.4)", markdown)
            self.assertIn("## V1 falsifier", markdown)
            self.assertIn("## Hard-root census (D1)", markdown)
            self.assertIn("## Trajectory adjacency", markdown)


if __name__ == "__main__":
    unittest.main()
