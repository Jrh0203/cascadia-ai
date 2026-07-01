from __future__ import annotations

import v3_capacity_projection as capacity


def test_projection_accounts_for_all_ten_cycles_and_stays_bounded() -> None:
    value = capacity.project(
        {"decisions_per_second": 30.0},
        {"seconds_per_game": 4.0},
        {"r600_seconds_per_game": 20.0},
        {"projected_part2_seconds": 20_000.0, "examples_per_second": 8_000.0},
        {"bytes": 400_000, "games": 100, "elapsed_seconds": 20.0},
    )
    assert value["assumptions"]["expert_cycles"] == 10
    assert value["assumptions"]["teacher_roots"] == 145_000
    assert value["assumptions"]["expected_v3_seat_equivalents_across_cycles"] == 15.4
    assert value["assumptions"]["qualified_v1_opponent_fraction"] == 0.8
    assert value["active_wall_seconds"] > value["measured_components_seconds"]["teacher_search"]
    assert value["projected_campaign_bytes"] < 40 * 1024**3
    assert value["protected_seed_values_opened"] is False


def test_direct_report_can_supply_compact_replay_measurement() -> None:
    direct = {
        "seconds_per_game": 1.0,
        "elapsed_seconds": 100.0,
        "compact_shard": {"bytes": 1_000, "games": 100},
    }
    value = capacity.project(
        {"decisions_per_second": 10.0},
        direct,
        {"r600_seconds_per_game": 20.0},
        {"projected_part2_seconds": 1_000.0, "examples_per_second": 3_000.0},
        direct,
    )
    assert value["rates"]["compact_bytes_per_game"] == 10
