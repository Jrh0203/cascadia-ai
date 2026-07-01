from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools/full_legal_regret_audit.py"
SPEC = importlib.util.spec_from_file_location("full_legal_regret_audit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def test_rank_and_resource_parsers(tmp_path: Path) -> None:
    assert audit.rank_bucket(1) == "01-08"
    assert audit.rank_bucket(32) == "09-32"
    assert audit.rank_bucket(64) == "33-64"
    assert audit.rank_bucket(65) == "65-128"
    assert audit.rank_bucket(999) == "129+"

    time_file = tmp_path / "seed.time"
    time_file.write_text(
        "      12.50 real        40.25 user         3.75 sys\n"
        "           123456789  maximum resident set size\n"
        "                   0  swaps\n"
    )
    parsed = audit.parse_time_file(time_file)
    assert parsed is not None
    assert parsed["real_seconds"] == 12.5
    assert parsed["user_seconds"] == 40.25
    assert parsed["system_seconds"] == 3.75
    assert parsed["maximum_resident_bytes"] == 123_456_789
    assert parsed["swaps"] == 0

    corrupt_time_file = tmp_path / "corrupt.time"
    corrupt_time_file.write_text("runner script replaced while active\n")
    corrupt = audit.parse_time_file(corrupt_time_file)
    assert corrupt is not None
    assert corrupt["real_seconds"] is None
    assert corrupt["maximum_resident_bytes"] is None
    assert corrupt["swaps"] is None

    swap_file = tmp_path / "swap.txt"
    swap_file.write_text("total = 4096.00M  used = 1409.69M  free = 2686.31M  (encrypted)\n")
    swap = audit.parse_system_swap_file(swap_file)
    assert swap is not None
    assert swap["used_bytes"] == int(1409.69 * 1024**2)


def test_report_discovery_excludes_seed_sidecars(tmp_path: Path) -> None:
    host = tmp_path / "john2"
    host.mkdir()
    shard = host / "seed-61006.json"
    shard.write_text("{}")
    (host / "seed-61006.summary-float-repair.json").write_text("{}")
    (host / "seed-not-a-number.json").write_text("{}")

    assert audit.discover_reports([host]) == [(shard, "john2")]


def test_block_statistic_resamples_whole_games() -> None:
    decisions = [
        {"seed": 1, "value": 1.0},
        {"seed": 1, "value": 3.0},
        {"seed": 2, "value": 5.0},
        {"seed": 2, "value": 7.0},
    ]
    result = audit.block_statistic(
        decisions,
        lambda decision: decision["value"],
        bootstrap_samples=2_000,
        bootstrap_seed=17,
    )
    assert result["count"] == 4
    assert result["games"] == 2
    assert result["mean"] == 4.0
    assert result["bootstrap_confidence_95"][0] == pytest.approx(2.0)
    assert result["bootstrap_confidence_95"][1] == pytest.approx(6.0)


def test_reference_shard_contract_and_compaction() -> None:
    path = ROOT / "artifacts/performance/full-legal-audit-reference-v1/reference.json"
    report = audit.json.loads(path.read_text())
    seed = audit.validate_report_contract(
        report,
        path=path,
        host="reference",
        expected_decisions_per_game=3,
        require_frozen_config=False,
    )
    assert seed == 60_999
    compact = audit.compact_decision(seed, report["games"][0]["decisions"][1])
    assert compact["phase"] == "middle"
    assert compact["champion_regret"] == pytest.approx(0.5613354037267015)
    assert compact["frontier_regret"] == pytest.approx(0.5613354037267015)
    assert compact["selection_regret"] == 0.0
    assert compact["winner_screen_rank"] == 31
    assert compact["change_kind"] == "tile_placement"
    assert compact["paid"] is not None
    assert compact["hidden"] is not None
    assert compact["frontier_regret"] <= compact["champion_regret"]


def test_screen_contract_fingerprint_excludes_recovery_only_fields() -> None:
    path = ROOT / "artifacts/performance/full-legal-audit-reference-v1/reference.json"
    report = audit.json.loads(path.read_text())
    baseline = audit.screen_contract_sha256(report)

    recovery_only = copy.deepcopy(report)
    decision = recovery_only["games"][0]["decisions"][0]
    decision["timings"]["substantial_seconds"] += 100.0
    decision["actions"][0]["sources"]["top_complete_screen"] = not decision["actions"][0][
        "sources"
    ]["top_complete_screen"]
    decision["actions"][0]["substantial_r1200"] = None
    assert audit.screen_contract_sha256(recovery_only) == baseline

    screen_drift = copy.deepcopy(report)
    screen_drift["games"][0]["decisions"][0]["actions"][0]["screen_value"] += 0.25
    assert audit.screen_contract_sha256(screen_drift) != baseline


def test_compaction_uses_best_high_confidence_frontier_finalist() -> None:
    path = ROOT / "artifacts/performance/full-legal-audit-reference-v1/reference.json"
    report = audit.json.loads(path.read_text())
    decision = copy.deepcopy(report["games"][0]["decisions"][0])
    actions = {
        action["canonical_hash"]: action
        for action in decision["actions"]
        if action["high_confidence_r4800"] is not None
    }
    champion = actions[decision["champion_action_hash"]]
    alternate_frontier = next(
        action
        for action in actions.values()
        if action["sources"]["champion_frontier"]
        and action["canonical_hash"] != champion["canonical_hash"]
    )
    winner = next(
        action
        for action in actions.values()
        if action["canonical_hash"]
        not in {champion["canonical_hash"], alternate_frontier["canonical_hash"]}
    )
    winner["sources"]["champion_frontier"] = False
    winner["high_confidence_r4800"]["mean"] = 100.0
    alternate_frontier["high_confidence_r4800"]["mean"] = 99.0
    champion["high_confidence_r4800"]["mean"] = 98.0
    decision["best_complete_screen_hash"] = winner["canonical_hash"]
    decision["champion_regret"]["points"] = 2.0
    decision["champion_frontier_regret"]["points"] = 2.0

    compact = audit.compact_decision(60_999, decision)

    assert compact["frontier_hash"] == alternate_frontier["canonical_hash"]
    assert compact["frontier_regret"] == 1.0
    assert compact["selection_regret"] == 1.0
    assert compact["frontier_comparator_corrected"]


def test_analysis_rejects_incomplete_seed_coverage() -> None:
    with pytest.raises(ValueError, match="seed coverage mismatch"):
        audit.analyze(
            [],
            expected_first_seed=1,
            expected_games=1,
            expected_decisions=1,
            expected_hidden_per_game=0,
            bootstrap_samples=100,
        )


def test_analysis_rejects_reference_seed_or_fingerprint_drift() -> None:
    path = ROOT / "artifacts/performance/full-legal-audit-reference-v1/reference.json"
    raw = audit.json.loads(path.read_text())
    game = raw["games"][0]
    item = audit.AuditInput(
        path=path,
        host="reference",
        seed=60_999,
        sha256=audit.sha256_file(path),
        screen_contract_sha256=audit.screen_contract_sha256(raw),
        bytes=path.stat().st_size,
        config=raw["config"],
        provenance=raw["provenance"],
        summary=raw["summary"],
        final_scores=game["final_scores"],
        decisions=[
            {
                **audit.compact_decision(60_999, decision),
                "host": "reference",
            }
            for decision in game["decisions"]
        ],
    )
    with pytest.raises(ValueError, match="reference seed coverage mismatch"):
        audit.analyze(
            [item],
            expected_first_seed=60_999,
            expected_games=1,
            expected_decisions=3,
            expected_hidden_per_game=3,
            bootstrap_samples=100,
            reference_screen_contract_by_seed={61_000: item.screen_contract_sha256},
        )

    report = audit.analyze(
        [item],
        expected_first_seed=60_999,
        expected_games=1,
        expected_decisions=3,
        expected_hidden_per_game=3,
        bootstrap_samples=100,
        reference_screen_contract_by_seed={60_999: "not-the-frozen-fingerprint"},
    )
    assert not report["screen_contract"]["pairing_passed"]
    assert report["screen_contract"]["mismatched_seeds"] == [60_999]
    assert not report["substantive_gates"]["exact_screen_contract_pairing"]
    assert not report["substantive_gates_passed"]


def test_optional_dominance_gate_preserves_diagnostic_without_blocking() -> None:
    path = ROOT / "artifacts/performance/full-legal-audit-reference-v1/reference.json"
    raw = audit.json.loads(path.read_text())
    game = raw["games"][0]
    item = audit.AuditInput(
        path=path,
        host="reference",
        seed=60_999,
        sha256=audit.sha256_file(path),
        screen_contract_sha256=audit.screen_contract_sha256(raw),
        bytes=path.stat().st_size,
        config=raw["config"],
        provenance=raw["provenance"],
        summary=raw["summary"],
        final_scores=game["final_scores"],
        decisions=[
            {
                **audit.compact_decision(60_999, decision),
                "host": "reference",
            }
            for decision in game["decisions"]
        ],
    )
    report = audit.analyze(
        [item],
        expected_first_seed=60_999,
        expected_games=1,
        expected_decisions=3,
        expected_hidden_per_game=3,
        bootstrap_samples=100,
        require_dominant_error_gate=False,
    )
    assert not report["dominant_error"]["required_for_substantive_gate"]
    assert report["substantive_gates"]["dominant_error_source_supported_by_game_block_ci"]


def test_reference_shard_streaming_analysis_completes() -> None:
    path = ROOT / "artifacts/performance/full-legal-audit-reference-v1/reference.json"
    raw = audit.json.loads(path.read_text())
    game = raw["games"][0]
    item = audit.AuditInput(
        path=path,
        host="reference",
        seed=60_999,
        sha256=audit.sha256_file(path),
        screen_contract_sha256=audit.screen_contract_sha256(raw),
        bytes=path.stat().st_size,
        config=raw["config"],
        provenance=raw["provenance"],
        summary=raw["summary"],
        final_scores=game["final_scores"],
        decisions=[
            {
                **audit.compact_decision(60_999, decision),
                "host": "reference",
            }
            for decision in game["decisions"]
        ],
    )
    report = audit.analyze(
        [item],
        expected_first_seed=60_999,
        expected_games=1,
        expected_decisions=3,
        expected_hidden_per_game=3,
        bootstrap_samples=500,
    )
    assert report["coverage"]["actions_screened"] == 11_594
    assert report["screen_contract"]["by_seed"]["60999"] == (item.screen_contract_sha256)
    assert not report["screen_contract"]["pairing_required"]
    assert report["screen_contract"]["pairing_passed"]
    assert report["decompositions"]["host"]["reference"]["screen_recall"]["mean"] == 1.0
    assert report["public_decision_regret"]["champion"]["mean"] == pytest.approx(
        0.21346239996125385
    )
    assert report["public_decision_regret"]["top64_recall"]["mean"] == 1.0
    assert (
        report["public_decision_regret"]["top64_or_champion_frontier_union_recall"]["mean"] == 1.0
    )
    assert report["public_decision_regret"]["screen_rank_recall_curve"]["64"]["mean"] == 1.0
    assert report["public_decision_regret"]["smallest_observed_width_at_98_percent"] == 64
    assert report["public_decision_regret"]["maximum_observed_winner_screen_rank"] == 31
    assert report["substantive_gates"]["paid_wipe_diagnostic_complete"]
    assert report["substantive_gates"]["realized_hidden_diagnostic_complete"]
