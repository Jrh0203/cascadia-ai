from tools.aaaaa_wildlife_split_salmon_dp_screen import CASES


def test_screen_cases() -> None:
    assert CASES == (
        ((4, 5, 2, 3, 6), 67),
        ((5, 5, 2, 2, 6), 64),
        ((3, 5, 2, 4, 6), 63),
        ((3, 6, 2, 3, 6), 63),
    )
