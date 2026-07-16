"""Contract tests for the group-sequential chunk-report merger.

The merger must recompute every aggregate exactly from the underlying rows
(concatenated decision JSONLs + per-seed score breakdowns) via the real
benchmark summarizers, refuse mismatched or overlapping chunks loudly, and
emit a report whose comparator-read fields are indistinguishable from a
single monolithic benchmark run over the union of the seeds.
"""

import json
import unittest
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory

from cascadiav3.merge_benchmark_reports import build_merged_report, main
from cascadiav3.torch_cascadiaformer_gumbel_benchmark import (
    summarize_market_decisions,
    summarize_score_categories,
)
from cascadiav3.torch_cascadiaformer_search_benchmark import (
    _percentile,
    summarize_game_results,
)

RULES = "cascadia_research_aaaaa_4p_card_a_no_habitat_bonus_rules_2026_07_16"
SEARCH = {
    "n_simulations": 256,
    "top_m": 16,
    "depth_rounds": 1,
    "determinizations": 4,
    "market_decision_samples": 8,
    "exact_endgame_turns": 1,
    "blend_weight": 0.5,
    "k_interior": 1,
    "max_root_actions": None,
}
CONTROL = {
    "kind": "none",
    "max_actions": 256,
    "rollouts_per_action": 8,
    "rollout_top_k": 5,
    "rollout_determinize": True,
}

# seed -> (seat totals, per-decision (decision_seconds, model_score_seconds, choice))
CHUNK_A = {
    11: ([96.0, 98.0, 100.0, 102.0], [(0.10, 0.01, "accept"), (0.20, 0.02, "not_available"), (0.30, 0.03, "decline")]),
    12: ([90.0, 95.0, 100.0, 105.0], [(0.40, 0.04, "not_available"), (0.50, 0.05, "accept")]),
}
CHUNK_B = {
    13: ([88.0, 92.0, 96.0, 104.0], [(0.60, 0.06, "decline"), (0.70, 0.07, "not_available")]),
    14: ([100.0, 100.0, 100.0, 100.0], [(0.80, 0.08, "not_available"), (0.90, 0.09, "accept")]),
    15: ([85.0, 95.0, 105.0, 115.0], [(1.00, 0.10, "accept"), (1.10, 0.11, "decline"), (1.20, 0.12, "not_available")]),
}


def seat_breakdown(total: float) -> dict:
    """A breakdown satisfying summarize_score_categories' category-sum check."""
    return {
        "wildlife": [total - 8.0, 3.0],
        "habitat": [2.0, 2.0],
        "nature_tokens": 1.0,
        "total": total,
    }


def decision_rows(games: dict) -> list[dict]:
    rows = []
    for seed, (_totals, decisions) in games.items():
        for ply, (decision_seconds, model_seconds, choice) in enumerate(decisions):
            row = {
                "type": "gumbel_decision",
                "seed": seed,
                "ply": ply,
                "ruleset_id": RULES,
                "decision_seconds": decision_seconds,
                "model_score_seconds": model_seconds,
                "free_three_of_a_kind_choice": choice,
                "simulations_run": 64,
                "total_simulations_run": 96,
            }
            if choice in {"accept", "decline"}:
                row["market_chance_samples"] = 8
            rows.append(row)
    return sorted(rows, key=lambda row: (row["seed"], row["ply"]))


def make_chunk(games: dict, wall_seconds: float, revision: str = "rev-test") -> dict:
    """Builds a chunk report exactly the way build_report shapes one."""
    rows = decision_rows(games)
    by_seed: dict[int, list[dict]] = {}
    for row in rows:
        by_seed.setdefault(row["seed"], []).append(row)
    results = [
        {
            "seed": seed,
            "done": {"scores": [seat_breakdown(total) for total in totals]},
            "decisions": by_seed[seed],
        }
        for seed, (totals, _decisions) in sorted(games.items())
    ]
    seconds = [row["decision_seconds"] for row in rows]
    return {
        "status": "pass",
        "ruleset_id": RULES,
        "source_revision": revision,
        "candidate_per_seed": [
            {
                "seed": result["seed"],
                "mean_score_per_seat": mean(s["total"] for s in result["done"]["scores"]),
                "seat_scores": [s["total"] for s in result["done"]["scores"]],
                "seat_score_breakdowns": result["done"]["scores"],
            }
            for result in results
        ],
        "scientific_eligibility": "candidate_only_search_arm",
        "experiment_id": "chunk_experiment",
        "execution": {"runner": "gumbel-policy-game", "requested_jobs": 4},
        "artifacts": {"weights_sha256": "abc123"},
        "raw_games_dir": "/tmp/chunk_raw_games",
        "binary": "target/release/exporter",
        "manifest": "checkpoints/x/best_locked_val.manifest.json",
        "model_service": "auto",
        "seeds": sorted(games),
        "search": dict(SEARCH),
        "market_decisions": summarize_market_decisions(rows),
        "candidate_score_breakdown": summarize_score_categories(results),
        "control_score_breakdown": None,
        "control": dict(CONTROL),
        "strategies": {"gumbel-search": summarize_game_results(results), "control": None},
        "candidate_decision_seconds_p50": _percentile(seconds, 0.50),
        "candidate_decision_seconds_p95": _percentile(seconds, 0.95),
        "candidate_wall_seconds": wall_seconds,
        "paired_score_deltas": [],
        "paired_delta_stats": {"n": 0},
        "gate": None,
    }


def write_chunk(tmp: str, name: str, games: dict, wall_seconds: float, **kwargs) -> tuple[Path, Path]:
    report = make_chunk(games, wall_seconds, **kwargs)
    report_path = Path(tmp) / f"{name}.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    decisions_path = Path(tmp) / f"{name}.decisions.jsonl"
    decisions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in decision_rows(games)),
        encoding="utf-8",
    )
    return report_path, decisions_path


class MergeHappyPathTest(unittest.TestCase):
    def merge_two_chunks(self, tmp: str) -> dict:
        chunk_a, decisions_a = write_chunk(tmp, "a", CHUNK_A, 100.0)
        chunk_b, decisions_b = write_chunk(tmp, "b", CHUNK_B, 250.0)
        return build_merged_report(
            [chunk_a, chunk_b], [decisions_a, decisions_b], "merged_exp"
        )

    def test_seeds_are_sorted_union(self) -> None:
        with TemporaryDirectory() as tmp:
            merged = self.merge_two_chunks(tmp)
            self.assertEqual(merged["seeds"], [11, 12, 13, 14, 15])
            self.assertEqual(
                [row["seed"] for row in merged["candidate_per_seed"]], [11, 12, 13, 14, 15]
            )

    def test_mean_seat_score_is_exact_mean_over_all_seat_totals(self) -> None:
        all_totals = [
            total
            for games in (CHUNK_A, CHUNK_B)
            for totals, _decisions in games.values()
            for total in totals
        ]
        self.assertEqual(len(all_totals), 20)
        with TemporaryDirectory() as tmp:
            merged = self.merge_two_chunks(tmp)
            summary = merged["strategies"]["gumbel-search"]
            self.assertEqual(summary["mean_seat_score"], mean(all_totals))
            self.assertEqual(summary["games"], 5)

    def test_decision_counts_and_timing_are_recomputed_from_rows(self) -> None:
        rows = decision_rows(CHUNK_A) + decision_rows(CHUNK_B)
        seconds = [row["decision_seconds"] for row in rows]
        totals = [
            row["decision_seconds"] + row["model_score_seconds"] for row in rows
        ]
        with TemporaryDirectory() as tmp:
            merged = self.merge_two_chunks(tmp)
            summary = merged["strategies"]["gumbel-search"]
            self.assertEqual(summary["decisions"], len(rows))
            self.assertEqual(summary["mean_total_decision_seconds"], mean(totals))
            self.assertEqual(
                merged["candidate_decision_seconds_p50"], _percentile(seconds, 0.50)
            )
            self.assertEqual(
                merged["candidate_decision_seconds_p95"], _percentile(seconds, 0.95)
            )
            self.assertEqual(merged["market_decisions"]["total_decisions"], len(rows))
            self.assertEqual(merged["market_decisions"]["accepted"], 4)
            self.assertEqual(merged["market_decisions"]["declined"], 3)

    def test_wall_seconds_sum_and_candidate_only_shape(self) -> None:
        with TemporaryDirectory() as tmp:
            merged = self.merge_two_chunks(tmp)
            self.assertEqual(merged["candidate_wall_seconds"], 350.0)
            self.assertEqual(merged["status"], "pass")
            self.assertEqual(merged["ruleset_id"], RULES)
            self.assertEqual(merged["control"], CONTROL)
            self.assertIsNone(merged["strategies"]["control"])
            self.assertIsNone(merged["gate"])
            self.assertIsNone(merged["raw_games_dir"])
            self.assertEqual(merged["paired_score_deltas"], [])
            self.assertEqual(merged["paired_delta_stats"]["n"], 0)
            self.assertEqual(
                merged["scientific_eligibility"], "candidate_only_search_arm"
            )
            self.assertEqual(merged["experiment_id"], "merged_exp")
            self.assertEqual(merged["execution"]["merged_chunks"], 2)
            self.assertEqual(
                merged["execution"]["merged_runner"], "merge_benchmark_reports"
            )
            self.assertEqual(merged["execution"]["requested_jobs"], 4)
            self.assertEqual(merged["search"], SEARCH)
            self.assertEqual(
                [entry["seeds"] for entry in merged["merged_from"]],
                [[11, 12], [13, 15]],
            )
            self.assertEqual(
                [entry["candidate_wall_seconds"] for entry in merged["merged_from"]],
                [100.0, 250.0],
            )
            breakdown = merged["candidate_score_breakdown"]
            self.assertEqual(breakdown["seat_scores_at_least_100"], 11)

    def test_single_chunk_merge_reproduces_comparator_fields(self) -> None:
        with TemporaryDirectory() as tmp:
            chunk_a, decisions_a = write_chunk(tmp, "a", CHUNK_A, 100.0)
            chunk = json.loads(chunk_a.read_text(encoding="utf-8"))
            merged = build_merged_report([chunk_a], [decisions_a], "solo")
            self.assertEqual(merged["seeds"], chunk["seeds"])
            self.assertEqual(merged["candidate_per_seed"], chunk["candidate_per_seed"])
            self.assertEqual(
                merged["strategies"]["gumbel-search"]["mean_seat_score"],
                chunk["strategies"]["gumbel-search"]["mean_seat_score"],
            )
            self.assertEqual(
                merged["strategies"]["gumbel-search"]["mean_total_decision_seconds"],
                chunk["strategies"]["gumbel-search"]["mean_total_decision_seconds"],
            )
            self.assertEqual(
                merged["strategies"]["gumbel-search"],
                chunk["strategies"]["gumbel-search"],
            )
            self.assertEqual(merged["candidate_wall_seconds"], chunk["candidate_wall_seconds"])
            self.assertEqual(
                merged["candidate_decision_seconds_p50"],
                chunk["candidate_decision_seconds_p50"],
            )
            self.assertEqual(
                merged["candidate_decision_seconds_p95"],
                chunk["candidate_decision_seconds_p95"],
            )
            self.assertEqual(merged["market_decisions"], chunk["market_decisions"])
            self.assertEqual(
                merged["candidate_score_breakdown"], chunk["candidate_score_breakdown"]
            )

    def test_cli_writes_report_and_summary(self) -> None:
        with TemporaryDirectory() as tmp:
            chunk_a, decisions_a = write_chunk(tmp, "a", CHUNK_A, 100.0)
            chunk_b, decisions_b = write_chunk(tmp, "b", CHUNK_B, 250.0)
            out = Path(tmp) / "merged.json"
            summary = Path(tmp) / "merged.md"
            code = main(
                [
                    "--chunk", str(chunk_a),
                    "--chunk", str(chunk_b),
                    "--decisions", str(decisions_a),
                    "--decisions", str(decisions_b),
                    "--experiment-id", "merged_exp",
                    "--out", str(out),
                    "--summary-out", str(summary),
                ]
            )
            self.assertEqual(code, 0)
            merged = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(merged["seeds"], [11, 12, 13, 14, 15])
            text = summary.read_text(encoding="utf-8")
            self.assertIn("merged_exp", text)
            self.assertIn("Total seeds: `5`", text)
            self.assertIn("Chunks merged: `2`", text)


class MergeValidationTest(unittest.TestCase):
    def test_overlapping_seed_sets_are_refused(self) -> None:
        overlapping = {12: CHUNK_B[13], 13: CHUNK_B[14]}
        with TemporaryDirectory() as tmp:
            chunk_a, decisions_a = write_chunk(tmp, "a", CHUNK_A, 100.0)
            chunk_b, decisions_b = write_chunk(tmp, "b", overlapping, 250.0)
            with self.assertRaisesRegex(ValueError, "seed sets overlap"):
                build_merged_report(
                    [chunk_a, chunk_b], [decisions_a, decisions_b], "merged_exp"
                )

    def test_search_dict_mismatch_is_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            chunk_a, decisions_a = write_chunk(tmp, "a", CHUNK_A, 100.0)
            chunk_b, decisions_b = write_chunk(tmp, "b", CHUNK_B, 250.0)
            payload = json.loads(chunk_b.read_text(encoding="utf-8"))
            payload["search"]["determinizations"] = 8
            chunk_b.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "search settings mismatch"):
                build_merged_report(
                    [chunk_a, chunk_b], [decisions_a, decisions_b], "merged_exp"
                )

    def test_source_revision_mismatch_is_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            chunk_a, decisions_a = write_chunk(tmp, "a", CHUNK_A, 100.0)
            chunk_b, decisions_b = write_chunk(
                tmp, "b", CHUNK_B, 250.0, revision="rev-other"
            )
            with self.assertRaisesRegex(ValueError, "source_revision mismatch"):
                build_merged_report(
                    [chunk_a, chunk_b], [decisions_a, decisions_b], "merged_exp"
                )

    def test_decisions_seed_set_mismatch_is_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            chunk_a, decisions_a = write_chunk(tmp, "a", CHUNK_A, 100.0)
            chunk_b, decisions_b = write_chunk(tmp, "b", CHUNK_B, 250.0)
            # Swapped decision files: each covers the other chunk's seeds.
            with self.assertRaisesRegex(ValueError, "covers seeds"):
                build_merged_report(
                    [chunk_a, chunk_b], [decisions_b, decisions_a], "merged_exp"
                )

    def test_chunk_and_decisions_count_mismatch_is_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            chunk_a, decisions_a = write_chunk(tmp, "a", CHUNK_A, 100.0)
            chunk_b, _decisions_b = write_chunk(tmp, "b", CHUNK_B, 250.0)
            with self.assertRaisesRegex(ValueError, "must match --chunk count"):
                build_merged_report([chunk_a, chunk_b], [decisions_a], "merged_exp")

    def test_non_pass_status_is_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            chunk_a, decisions_a = write_chunk(tmp, "a", CHUNK_A, 100.0)
            payload = json.loads(chunk_a.read_text(encoding="utf-8"))
            payload["status"] = "fail"
            chunk_a.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not passing"):
                build_merged_report([chunk_a], [decisions_a], "merged_exp")

    def test_non_decision_row_type_is_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            chunk_a, decisions_a = write_chunk(tmp, "a", CHUNK_A, 100.0)
            with decisions_a.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"type": "gumbel_game_done", "seed": 11}) + "\n")
            with self.assertRaisesRegex(ValueError, "expected 'gumbel_decision'"):
                build_merged_report([chunk_a], [decisions_a], "merged_exp")

    def test_control_arm_chunk_is_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            chunk_a, decisions_a = write_chunk(tmp, "a", CHUNK_A, 100.0)
            payload = json.loads(chunk_a.read_text(encoding="utf-8"))
            payload["control"]["kind"] = "full-search"
            chunk_a.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "candidate-only"):
                build_merged_report([chunk_a], [decisions_a], "merged_exp")

    def test_per_seed_coverage_mismatch_is_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            chunk_a, decisions_a = write_chunk(tmp, "a", CHUNK_A, 100.0)
            payload = json.loads(chunk_a.read_text(encoding="utf-8"))
            payload["candidate_per_seed"] = payload["candidate_per_seed"][:1]
            chunk_a.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "candidate_per_seed seeds"):
                build_merged_report([chunk_a], [decisions_a], "merged_exp")

    def test_mismatched_execution_jobs_are_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            chunk_a, decisions_a = write_chunk(tmp, "a", CHUNK_A, 100.0)
            chunk_b, decisions_b = write_chunk(tmp, "b", CHUNK_B, 250.0)
            payload = json.loads(chunk_b.read_text(encoding="utf-8"))
            payload["execution"]["requested_jobs"] = 8
            chunk_b.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "requested_jobs mismatch"):
                build_merged_report(
                    [chunk_a, chunk_b], [decisions_a, decisions_b], "merged_exp"
                )


if __name__ == "__main__":
    unittest.main()
