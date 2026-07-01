"""CPU-only pre-GPU scaffold for CascadiaFormer-Zero planning.

This package intentionally depends only on the Python standard library. It is a
contract and validation scaffold, not a trained model or strength test.
"""

from .hex import RADIUS6, RADIUS6_CELL_COUNT, coord_ref, in_radius6
from .schema import SCHEMA_ID, SchemaError, validate_search_root_record

__all__ = [
    "RADIUS6",
    "RADIUS6_CELL_COUNT",
    "SCHEMA_ID",
    "SchemaError",
    "coord_ref",
    "in_radius6",
    "validate_search_root_record",
]
