from cascadia_mlx.r2_map_tensor_contract import (
    BOARD_TOKEN_CAPACITY,
    MAX_BOARD_TILES,
    MAX_LEGAL_FRONTIER_TOKENS,
    MAX_LEGAL_HABITAT_COMPONENT_TOKENS,
    MAX_LEGAL_WILDLIFE_MOTIF_TOKENS,
    TOKEN_CAPACITY,
)
from cascadia_mlx.r2_sparse_mlx_cache import (
    BOARD_TOKEN_CAPACITY as FOUNDATION_BOARD_TOKEN_CAPACITY,
)


def test_live_capacity_is_rules_complete_and_distinct_from_frozen_foundation() -> None:
    assert FOUNDATION_BOARD_TOKEN_CAPACITY == 92
    assert MAX_BOARD_TILES == 23
    assert MAX_LEGAL_FRONTIER_TOKENS == 50
    assert MAX_LEGAL_HABITAT_COMPONENT_TOKENS == 46
    assert MAX_LEGAL_WILDLIFE_MOTIF_TOKENS == 20
    assert BOARD_TOKEN_CAPACITY == 139
    assert TOKEN_CAPACITY == 4 * 139
    for occupied in range(3, MAX_BOARD_TILES + 1):
        bound = occupied + (2 * occupied + 4) + 2 * occupied + (occupied - 3)
        assert bound == 6 * occupied + 1
        assert bound <= BOARD_TOKEN_CAPACITY
