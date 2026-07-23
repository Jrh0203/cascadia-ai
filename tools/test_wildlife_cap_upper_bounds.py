import math

from tools import aaaaa_wildlife_exact as aaaaa
from tools import cbddb_wildlife_exact as cbddb
from tools import wildlife_cap_upper_bounds as bounds


def test_cap_six_parity_with_existing_relaxations() -> None:
    for counts, expected in aaaaa.count_vectors():
        assert bounds.aaaaa_count_upper(counts, incidence_aware=False) == expected
        assert bounds.aaaaa_count_upper(counts, incidence_aware=True) == expected
    for counts, expected in cbddb.count_vectors():
        assert bounds.cbddb_count_upper(counts) == expected


def test_seventh_token_standalone_scores_follow_rules() -> None:
    assert bounds.bear_a_standalone(7) == 19
    assert bounds.elk_ab_standalone(7) == 22
    assert bounds.salmon_a_standalone(7) == 25
    assert bounds.hawk_a_standalone(7) == 22
    assert bounds.bear_c_standalone(7) == 20
    assert bounds.salmon_d_standalone(7) == 29


def test_cap_seven_count_space_and_maxima() -> None:
    result = bounds.analyze_cap(7)
    expected_count = (
        math.comb(24, 4) - 5 * math.comb(16, 4) + 10 * math.comb(8, 4)
    )
    assert result["allocation_count"] == expected_count == 2226
    assert result["aaaaa_geometry_free"] == {
        "maximum": 75,
        "maximizing_counts": [[1, 4, 7, 1, 7], [2, 3, 7, 1, 7], [4, 1, 7, 1, 7]],
    }
    assert result["aaaaa_incidence_aware"]["maximum"] == 74
    assert result["aaaaa_incidence_aware"]["maximizing_counts"] == [
        [2, 2, 7, 2, 7],
        [2, 3, 7, 1, 7],
        [2, 4, 7, 1, 6],
    ]
    assert result["cbddb_geometry_free"] == {
        "maximum": 102,
        "maximizing_counts": [[0, 3, 6, 4, 7], [0, 4, 3, 6, 7]],
    }


def test_every_cap_seven_maximizer_is_a_legal_allocation() -> None:
    result = bounds.analyze_cap(7)
    for key in (
        "aaaaa_geometry_free",
        "aaaaa_incidence_aware",
        "cbddb_geometry_free",
    ):
        for counts in result[key]["maximizing_counts"]:
            assert len(counts) == 5
            assert sum(counts) == 20
            assert all(0 <= count <= 7 for count in counts)
