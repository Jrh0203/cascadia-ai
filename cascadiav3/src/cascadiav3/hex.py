"""Radius-6 hex coordinate contract with exact overflow.

The canonical fast path is a regular axial/cube hex disk of radius 6. Coordinates
outside that disk are represented exactly as overflow coordinates instead of
being clipped, remapped, or dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

RADIUS6 = 6
RADIUS6_CELL_COUNT = 1 + 3 * RADIUS6 * (RADIUS6 + 1)

HEX_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
)


@dataclass(frozen=True, order=True)
class AxialCoord:
    q: int
    r: int

    @property
    def s(self) -> int:
        return -self.q - self.r

    @property
    def radius(self) -> int:
        return max(abs(self.q), abs(self.r), abs(self.s))

    @property
    def radius6_member(self) -> bool:
        return self.radius <= RADIUS6

    def rotate_right(self) -> "AxialCoord":
        """Rotate cube coordinate (q, r, s) 60 degrees clockwise."""

        return AxialCoord(-self.r, -self.s)

    def reflect_q(self) -> "AxialCoord":
        """Reflect across the q axis in cube coordinates."""

        return AxialCoord(self.q, self.s)

    def neighbor(self, direction: int) -> "AxialCoord":
        dq, dr = HEX_DIRECTIONS[direction % len(HEX_DIRECTIONS)]
        return AxialCoord(self.q + dq, self.r + dr)

    def as_dict(self) -> dict[str, int]:
        return {"q": self.q, "r": self.r, "s": self.s}


def _stable_radius6_coords() -> tuple[AxialCoord, ...]:
    coords = [
        AxialCoord(q, r)
        for q in range(-RADIUS6, RADIUS6 + 1)
        for r in range(-RADIUS6, RADIUS6 + 1)
        if AxialCoord(q, r).radius6_member
    ]
    coords.sort(key=lambda coord: (coord.radius, coord.q, coord.r))
    return tuple(coords)


RADIUS6_COORDS: tuple[AxialCoord, ...] = _stable_radius6_coords()
if len(RADIUS6_COORDS) != RADIUS6_CELL_COUNT:
    raise RuntimeError("radius-6 coordinate table has the wrong size")

_COORD_TO_INDEX = {coord: idx for idx, coord in enumerate(RADIUS6_COORDS)}


def in_radius6(q: int, r: int) -> bool:
    return AxialCoord(q, r).radius6_member


def cell_index(q: int, r: int) -> int | None:
    return _COORD_TO_INDEX.get(AxialCoord(q, r))


def coord_for_index(index: int) -> AxialCoord:
    if index < 0 or index >= len(RADIUS6_COORDS):
        raise IndexError(f"radius-6 cell index out of range: {index}")
    return RADIUS6_COORDS[index]


def coord_ref(
    q: int,
    r: int,
    *,
    owner_seat: int | None = None,
    placement_id: int | None = None,
) -> dict[str, object]:
    """Return a schema-ready coordinate reference."""

    coord = AxialCoord(q, r)
    if coord.radius6_member:
        return {
            "kind": "canonical",
            "q": coord.q,
            "r": coord.r,
            "s": coord.s,
            "radius6_member": True,
            "cell_index": cell_index(coord.q, coord.r),
        }

    if owner_seat is None or placement_id is None:
        raise ValueError("overflow coord_ref requires owner_seat and placement_id")

    return {
        "kind": "overflow",
        "q": coord.q,
        "r": coord.r,
        "s": coord.s,
        "radius6_member": False,
        "owner_seat": owner_seat,
        "placement_id": placement_id,
    }


def distance(a: AxialCoord, b: AxialCoord) -> int:
    return max(abs(a.q - b.q), abs(a.r - b.r), abs(a.s - b.s))


def distance_bucket(a: AxialCoord, b: AxialCoord) -> str:
    d = distance(a, b)
    return str(d) if d <= RADIUS6 else "7+"
