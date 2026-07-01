from __future__ import annotations

import v3_training_schedule as schedule


def test_schedule_freezes_all_ten_cycles_and_registered_mix() -> None:
    value = schedule.build(8_192)
    assert len(value["cycles"]) == 10
    assert value["cycles"][0]["exploration_epsilon"] == 0.10
    assert value["cycles"][-1]["exploration_epsilon"] == 0.02
    assert sum(value["cycles"][4]["data_mix"].values()) == 1.0
    assert value["bootstrap"]["origins"] == 3
    assert value["bootstrap"]["total_exposures_including_calibration"] == 120_000_000
    assert all(cycle["total_exposures_per_origin"] == 1_200_000 for cycle in value["cycles"])
    assert value["cycles"][0]["source_quotas_per_pass"]["preceding_three_cycles"] == 0
    assert value["cycles"][1]["source_quotas_per_pass"]["preceding_three_cycles"] == 120_000
    assert all(cycle["equalize_score_quantile_within_phase"] for cycle in value["cycles"])
    blocks = value["bootstrap"]["blocks"]
    assert len(blocks) == 12
    assert sum(block["exposures"] for block in blocks) == 36_000_000
    assert [block["kind"] for block in blocks].count("broad") == 4
    assert [block["kind"] for block in blocks].count("broad-teacher-50-50") == 6
    assert [block["kind"] for block in blocks].count("low-rate-consolidation") == 2
    lambdas = [block["teacher_lambda"] for block in blocks if block["teacher_lambda"] is not None]
    assert lambdas[0] == 1.0
    assert lambdas[-1] == 0.75
    assert value["topology"]["next_cycle_generation_during_training"] is False
