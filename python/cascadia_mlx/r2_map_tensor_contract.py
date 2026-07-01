"""Rules-complete live tensor shape for R2-MAP.

The historical sparse-foundation cache remains frozen at 92 rows per board.
Live R2-MAP must represent every legal 23-tile Cascadia afterstate: for a
connected ``n``-tile board, occupied + frontier + habitat components + wildlife
motifs is bounded by ``n + (2n + 4) + 2n + (n - 3) = 6n + 1 <= 139``.
"""

from cascadia_mlx.r2_sparse_mlx_cache import (
    BOARD_OWNERSHIP_ENCODING as BOARD_OWNERSHIP_ENCODING,
)
from cascadia_mlx.r2_sparse_mlx_cache import BOARD_SLOTS as BOARD_SLOTS
from cascadia_mlx.r2_sparse_mlx_cache import GLOBAL_FEATURES as GLOBAL_FEATURES
from cascadia_mlx.r2_sparse_mlx_cache import MARKET_FEATURES as MARKET_FEATURES
from cascadia_mlx.r2_sparse_mlx_cache import PLAYER_FEATURES as PLAYER_FEATURES
from cascadia_mlx.r2_sparse_mlx_cache import TARGET_DIM as TARGET_DIM
from cascadia_mlx.r2_sparse_mlx_cache import TOKEN_FEATURES as TOKEN_FEATURES
from cascadia_mlx.r2_sparse_mlx_cache import (
    TOKEN_PAYLOAD_WIDTH as TOKEN_PAYLOAD_WIDTH,
)

MAX_BOARD_TILES = 23
MAX_LEGAL_FRONTIER_TOKENS = 2 * MAX_BOARD_TILES + 4
MAX_LEGAL_HABITAT_COMPONENT_TOKENS = 2 * MAX_BOARD_TILES
MAX_LEGAL_WILDLIFE_MOTIF_TOKENS = MAX_BOARD_TILES - 3
BOARD_TOKEN_CAPACITY = (
    MAX_BOARD_TILES
    + MAX_LEGAL_FRONTIER_TOKENS
    + MAX_LEGAL_HABITAT_COMPONENT_TOKENS
    + MAX_LEGAL_WILDLIFE_MOTIF_TOKENS
)
TOKEN_CAPACITY = BOARD_SLOTS * BOARD_TOKEN_CAPACITY

assert BOARD_TOKEN_CAPACITY == 139
