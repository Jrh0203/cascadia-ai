import pytest

from tools.all_wildlife_profile_taskset import build_taskset, parse_case

CASES = [
    "AAAAA:6,1,6,2,5:69",
    "AAAAA:4,2,6,2,6:69",
    "CADAC:0,2,6,6,6:67",
]


def test_frozen_calibration_has_expected_profile_counts() -> None:
    taskset = build_taskset(CASES)

    assert taskset["schema"] == "all-wildlife-score-profile-taskset-v1"
    assert [case["profile_count"] for case in taskset["cases"]] == [2, 6, 29]
    assert taskset["task_count"] == 37
    assert [task["task_index"] for task in taskset["tasks"]] == list(range(37))
    assert all(
        sum(task["score_profile"]) >= task["threshold"]
        for task in taskset["tasks"]
    )


def test_case_parser_rejects_invalid_counts_and_threshold() -> None:
    with pytest.raises(ValueError, match="invalid count vector"):
        parse_case("AAAAA:4,4,4,4,3:60")
    with pytest.raises(ValueError, match="threshold exceeds"):
        parse_case("AAAAA:6,1,6,2,5:99")


def test_taskset_rejects_duplicate_cases() -> None:
    with pytest.raises(ValueError, match="duplicate case"):
        build_taskset([CASES[0], CASES[0]])
