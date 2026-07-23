#!/usr/bin/env python3
"""Independent scorer and sound count bounds for all 1,024 wildlife rulesets."""

from __future__ import annotations

import itertools
from functools import cache
from typing import Any

SPECIES = ("bear", "elk", "salmon", "hawk", "fox")
VARIANTS = ("A", "B", "C", "D")
TOKEN_COUNT = 20
COUNT_CAP = 6
DIRECTIONS = ((1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1))


def parse_ruleset(ruleset: str) -> tuple[str, str, str, str, str]:
    normalized = ruleset.upper()
    if len(normalized) != len(SPECIES) or any(card not in VARIANTS for card in normalized):
        raise ValueError(f"invalid wildlife ruleset {ruleset!r}; expected five A/B/C/D cards")
    return tuple(normalized)  # type: ignore[return-value]


def rulesets() -> list[str]:
    return ["".join(cards) for cards in itertools.product(VARIANTS, repeat=len(SPECIES))]


def count_vectors() -> list[tuple[int, int, int, int, int]]:
    return [
        counts
        for counts in itertools.product(range(COUNT_CAP + 1), repeat=len(SPECIES))
        if sum(counts) == TOKEN_COUNT
    ]


def neighbors(coord: tuple[int, int]) -> set[tuple[int, int]]:
    q, r = coord
    return {(q + dq, r + dr) for dq, dr in DIRECTIONS}


def components(coords: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    result = []
    remaining = set(coords)
    while remaining:
        component = {remaining.pop()}
        frontier = list(component)
        while frontier:
            found = neighbors(frontier.pop()) & remaining
            component.update(found)
            frontier.extend(found)
            remaining.difference_update(found)
        result.append(component)
    return result


def _maximum_weight_matching(token_count: int, edges: list[tuple[int, int, int]]) -> int:
    by_left: list[list[tuple[int, int]]] = [[] for _ in range(token_count)]
    for left, right, score in edges:
        by_left[left].append((right, score))
        by_left[right].append((left, score))

    @cache
    def solve(available: int) -> int:
        if not available:
            return 0
        first = (available & -available).bit_length() - 1
        without_first = available & ~(1 << first)
        best = solve(without_first)
        for other, score in by_left[first]:
            if without_first & (1 << other):
                best = max(best, score + solve(without_first & ~(1 << other)))
        return best

    return solve((1 << token_count) - 1)


def _maximum_disjoint_groups(token_count: int, groups: set[tuple[int, int]]) -> int:
    dp = [0] * (1 << token_count)
    for state in range(1, len(dp)):
        first = (state & -state).bit_length() - 1
        for group, score in groups:
            if group & (1 << first) and group & state == group:
                dp[state] = max(dp[state], score + dp[state & ~group])
    return dp[-1]


def _score_bear(coords: set[tuple[int, int]], variant: str) -> int:
    sizes = [len(component) for component in components(coords)]
    if variant == "A":
        return (0, 4, 11, 19, 27)[min(sum(size == 2 for size in sizes), 4)]
    if variant == "B":
        return 10 * sum(size == 3 for size in sizes)
    if variant == "C":
        seen = {size for size in sizes if 1 <= size <= 3}
        return sum({1: 2, 2: 5, 3: 8}.get(size, 0) for size in sizes) + 3 * int(
            seen == {1, 2, 3}
        )
    if variant == "D":
        return sum({2: 5, 3: 8, 4: 13}.get(size, 0) for size in sizes)
    raise AssertionError(variant)


def _local_adjacency(ordered: list[tuple[int, int]]) -> list[int]:
    index = {coord: position for position, coord in enumerate(ordered)}
    adjacency = [0] * len(ordered)
    for left, coord in enumerate(ordered):
        for other in neighbors(coord):
            if other in index:
                adjacency[left] |= 1 << index[other]
    return adjacency


def _score_elk_a(ordered: list[tuple[int, int]]) -> int:
    index = {coord: position for position, coord in enumerate(ordered)}
    groups = {(1 << position, 2) for position in range(len(ordered))}
    for start, coord in enumerate(ordered):
        for dq, dr in DIRECTIONS[:3]:
            mask = 1 << start
            q, r = coord
            for _length, score in ((2, 5), (3, 9), (4, 13)):
                q += dq
                r += dr
                if (q, r) not in index:
                    break
                mask |= 1 << index[(q, r)]
                groups.add((mask, score))
    return _maximum_disjoint_groups(len(ordered), groups)


def _score_elk_b(ordered: list[tuple[int, int]]) -> int:
    adjacency = _local_adjacency(ordered)
    groups = {(1 << position, 2) for position in range(len(ordered))}
    for left, right in itertools.combinations(range(len(ordered)), 2):
        if not adjacency[left] & (1 << right):
            continue
        groups.add(((1 << left) | (1 << right), 5))
        for third in range(right + 1, len(ordered)):
            if not (
                adjacency[left] & (1 << third) and adjacency[right] & (1 << third)
            ):
                continue
            triangle = (1 << left) | (1 << right) | (1 << third)
            groups.add((triangle, 9))
            for fourth in range(len(ordered)):
                if triangle & (1 << fourth):
                    continue
                attached = sum(
                    bool(adjacency[member] & (1 << fourth))
                    for member in (left, right, third)
                )
                if attached >= 2:
                    groups.add((triangle | (1 << fourth), 13))
    return _maximum_disjoint_groups(len(ordered), groups)


def _connected_subset(mask: int, adjacency: list[int]) -> bool:
    if not mask:
        return False
    reached = mask & -mask
    while True:
        expanded = reached
        frontier = reached
        while frontier:
            bit = frontier & -frontier
            frontier ^= bit
            expanded |= adjacency[bit.bit_length() - 1] & mask
        if expanded == reached:
            return reached == mask
        reached = expanded


def _score_connected_elk_c(component: set[tuple[int, int]]) -> int:
    ordered = sorted(component)
    adjacency = _local_adjacency(ordered)
    dp = [0] * (1 << len(ordered))
    table = (0, 2, 4, 7, 10, 14, 18, 23)
    for state in range(1, len(dp)):
        first_bit = state & -state
        others = state ^ first_bit
        subset = others
        while True:
            candidate = subset | first_bit
            if _connected_subset(candidate, adjacency):
                size = candidate.bit_count()
                score = table[size] if size < len(table) else 28
                dp[state] = max(dp[state], score + dp[state & ~candidate])
            if subset == 0:
                break
            subset = (subset - 1) & others
    return dp[-1]


def _score_elk_d(ordered: list[tuple[int, int]]) -> int:
    index = {coord: position for position, coord in enumerate(ordered)}
    rings = set()
    for elk in ordered:
        for center in neighbors(elk):
            mask = 0
            for cell in neighbors(center):
                if cell in index:
                    mask |= 1 << index[cell]
            if mask:
                rings.add(mask)
    dp = [0] * (1 << len(ordered))
    table = (0, 2, 5, 8, 12, 16, 21)
    for state in range(1, len(dp)):
        first = (state & -state).bit_length() - 1
        for ring in rings:
            if not ring & (1 << first):
                continue
            claimed = ring & state
            dp[state] = max(dp[state], table[claimed.bit_count()] + dp[state & ~claimed])
    return dp[-1]


def _score_elk(coords: set[tuple[int, int]], variant: str) -> int:
    ordered = sorted(coords)
    if variant == "A":
        return _score_elk_a(ordered)
    if variant == "B":
        return _score_elk_b(ordered)
    if variant == "C":
        return sum(_score_connected_elk_c(component) for component in components(coords))
    if variant == "D":
        return _score_elk_d(ordered)
    raise AssertionError(variant)


def _score_salmon(
    salmon: set[tuple[int, int]],
    occupants: dict[tuple[int, int], str],
    variant: str,
) -> int:
    total = 0
    for component in components(salmon):
        if any(len(neighbors(coord) & component) > 2 for coord in component):
            continue
        size = len(component)
        if variant == "A":
            total += (0, 2, 5, 8, 12, 16, 20)[size] if size <= 6 else 25
        elif variant == "B":
            total += (0, 2, 4, 9, 11)[size] if size <= 4 else 17
        elif variant == "C":
            total += (0, 0, 0, 10, 12)[size] if size <= 4 else 15
        elif variant == "D":
            if size < 3:
                continue
            adjacent_non_salmon = {
                other
                for coord in component
                for other in neighbors(coord)
                if other in occupants and occupants[other] != "salmon"
            }
            total += size + len(adjacent_non_salmon)
        else:
            raise AssertionError(variant)
    return total


def _ray_between(
    left: tuple[int, int], right: tuple[int, int]
) -> tuple[tuple[int, int], ...] | None:
    dq = right[0] - left[0]
    dr = right[1] - left[1]
    for step_q, step_r in DIRECTIONS:
        distance: int | None = None
        if step_q:
            if dq % step_q:
                continue
            distance = dq // step_q
        elif dq:
            continue
        if step_r:
            if dr % step_r:
                continue
            r_distance = dr // step_r
            if distance is not None and distance != r_distance:
                continue
            distance = r_distance
        elif dr:
            continue
        if distance is not None and distance > 1:
            return tuple(
                (left[0] + step * step_q, left[1] + step * step_r)
                for step in range(1, distance)
            )
    return None


def _hawk_lines(
    hawks: list[tuple[int, int]],
    occupants: dict[tuple[int, int], str],
) -> list[tuple[int, int, tuple[tuple[int, int], ...]]]:
    result = []
    for left, right in itertools.combinations(range(len(hawks)), 2):
        between = _ray_between(hawks[left], hawks[right])
        if between is None or any(occupants.get(coord) == "hawk" for coord in between):
            continue
        result.append((left, right, between))
    return result


def _score_hawk(
    coords: set[tuple[int, int]],
    occupants: dict[tuple[int, int], str],
    variant: str,
) -> int:
    hawks = sorted(coords)
    isolated = [
        not any(occupants.get(other) == "hawk" for other in neighbors(hawk))
        for hawk in hawks
    ]
    lines = _hawk_lines(hawks, occupants)
    if variant == "A":
        count = sum(isolated)
        return (0, 2, 5, 8, 11, 14, 18)[count] if count <= 6 else (
            22 if count == 7 else 26
        )
    if variant == "B":
        qualifying = [False] * len(hawks)
        for left, right, _ in lines:
            qualifying[left] |= isolated[left]
            qualifying[right] |= isolated[right]
        count = sum(qualifying)
        return (0, 0, 5, 9, 12, 16, 20)[count] if count <= 6 else (
            24 if count == 7 else 28
        )
    if variant == "C":
        return 3 * len(lines)
    if variant == "D":
        edges = []
        for left, right, between in lines:
            distinct = {occupants[coord] for coord in between if coord in occupants}
            score = (0, 4, 7, 9)[min(len(distinct), 3)]
            if score:
                edges.append((left, right, score))
        return _maximum_weight_matching(len(hawks), edges)
    raise AssertionError(variant)


def _score_fox(
    coords: set[tuple[int, int]],
    occupants: dict[tuple[int, int], str],
    variant: str,
) -> int:
    foxes = sorted(coords)
    if variant == "A":
        return sum(
            len({occupants[other] for other in neighbors(fox) if other in occupants})
            for fox in foxes
        )
    if variant == "B":
        total = 0
        for fox in foxes:
            counts = [
                sum(occupants.get(other) == species for other in neighbors(fox))
                for species in SPECIES[:-1]
            ]
            total += (0, 3, 5, 7)[min(sum(count >= 2 for count in counts), 3)]
        return total
    if variant == "C":
        return sum(
            max(
                (
                    sum(occupants.get(other) == species for other in neighbors(fox))
                    for species in SPECIES[:-1]
                ),
                default=0,
            )
            for fox in foxes
        )
    if variant == "D":
        edges = []
        for left, right in itertools.combinations(range(len(foxes)), 2):
            if foxes[right] not in neighbors(foxes[left]):
                continue
            surrounding = (neighbors(foxes[left]) | neighbors(foxes[right])) - {
                foxes[left],
                foxes[right],
            }
            counts = [
                sum(occupants.get(coord) == species for coord in surrounding)
                for species in SPECIES[:-1]
            ]
            doubled = sum(count >= 2 for count in counts)
            score = (0, 5, 7, 9, 11)[doubled]
            if score:
                edges.append((left, right, score))
        return _maximum_weight_matching(len(foxes), edges)
    raise AssertionError(variant)


def normalized_tokens(rows: list[dict[str, Any]]) -> list[dict[str, int | str]]:
    tokens = [
        {"q": int(row["q"]), "r": int(row["r"]), "wildlife": str(row["wildlife"])}
        for row in rows
    ]
    if len(tokens) != TOKEN_COUNT:
        raise ValueError(f"expected {TOKEN_COUNT} tokens, got {len(tokens)}")
    if any(str(row["wildlife"]) not in SPECIES for row in tokens):
        raise ValueError("unknown wildlife species")
    occupied = {(int(row["q"]), int(row["r"])) for row in tokens}
    if len(occupied) != TOKEN_COUNT:
        raise ValueError("token coordinates overlap")
    tokens.sort(key=lambda row: (int(row["r"]), int(row["q"]), str(row["wildlife"])))
    return tokens


def score_tokens(
    rows: list[dict[str, Any]],
    ruleset: str,
) -> tuple[int, int, int, int, int]:
    cards = parse_ruleset(ruleset)
    tokens = normalized_tokens(rows)
    occupants = {
        (int(row["q"]), int(row["r"])): str(row["wildlife"]) for row in tokens
    }
    positions = {
        species: {coord for coord, wildlife in occupants.items() if wildlife == species}
        for species in SPECIES
    }
    return (
        _score_bear(positions["bear"], cards[0]),
        _score_elk(positions["elk"], cards[1]),
        _score_salmon(positions["salmon"], occupants, cards[2]),
        _score_hawk(positions["hawk"], occupants, cards[3]),
        _score_fox(positions["fox"], occupants, cards[4]),
    )


def _partition_upper(token_count: int, scores: tuple[int, ...]) -> int:
    best = [0] * (token_count + 1)
    for used in range(1, token_count + 1):
        best[used] = max(
            best[used - 1],
            *(
                scores[size] + best[used - size]
                for size in range(1, min(used, len(scores) - 1) + 1)
            ),
        )
    return best[token_count]


def _standalone_bear(count: int, variant: str) -> int:
    if variant == "A":
        return (0, 0, 4, 4, 11, 11, 19)[count]
    if variant == "B":
        return 10 * (count // 3)
    if variant == "C":
        return (0, 2, 5, 8, 10, 13, 18)[count]
    if variant == "D":
        return _partition_upper(count, (0, 0, 5, 8, 13))
    raise AssertionError(variant)


def _standalone_elk(count: int, variant: str) -> int:
    if variant in ("A", "B"):
        return _partition_upper(count, (0, 2, 5, 9, 13))
    if variant == "C":
        return (0, 2, 4, 7, 10, 14, 18)[count]
    if variant == "D":
        return (0, 2, 5, 8, 12, 16, 21)[count]
    raise AssertionError(variant)


def _standalone_salmon(count: int, variant: str, non_salmon: int) -> int:
    if variant == "A":
        return _partition_upper(count, (0, 2, 5, 8, 12, 16, 20))
    if variant == "B":
        return _partition_upper(count, (0, 2, 4, 9, 11, 17, 17))
    if variant == "C":
        return _partition_upper(count, (0, 0, 0, 10, 12, 15, 15))
    if variant == "D":
        best = [0] * (count + 1)
        for used in range(1, count + 1):
            best[used] = best[used - 1]
            for size in range(3, used + 1):
                score = size + min(non_salmon, 2 * size + 4)
                best[used] = max(best[used], score + best[used - size])
        return best[count]
    raise AssertionError(variant)


_BIPARTITE_HEX_EDGE_MAXIMUM = (
    (0, 0, 0, 0, 0, 0, 0),
    (0, 1, 2, 3, 4, 5, 6),
    (0, 2, 4, 5, 6, 7, 8),
    (0, 3, 5, 7, 9, 10, 11),
    (0, 4, 6, 9, 10, 12, 14),
    (0, 5, 7, 10, 12, 14, 15),
    (0, 6, 8, 11, 14, 15, 17),
)


def _bipartite_hex_edge_upper(left: int, right: int) -> int:
    """Exact cap-six adjacency maximum between two disjoint token classes."""
    if not (0 <= left <= COUNT_CAP and 0 <= right <= COUNT_CAP):
        raise ValueError("bipartite edge table is defined through the count cap")
    return _BIPARTITE_HEX_EDGE_MAXIMUM[left][right]


@cache
def _fox_c_upper(foxes: int, targets: tuple[int, int, int, int]) -> int:
    # Assign each fox to the species whose adjacent count it scores.  Each
    # group induces a simple planar bipartite graph with its target species.
    best = 0
    for assigned in itertools.product(range(foxes + 1), repeat=len(targets)):
        if sum(assigned) <= foxes:
            best = max(
                best,
                sum(
                    _bipartite_hex_edge_upper(group, target)
                    for group, target in zip(assigned, targets, strict=True)
                ),
            )
    return best


@cache
def _fox_b_upper(foxes: int, targets: tuple[int, int, int, int]) -> int:
    # Exact lattice maximum for fox vertices with at least two neighbors in
    # one target class, derived componentwise through the cap and then combined
    # over disconnected support.
    qualification_maximum = (
        (0, 0, 0, 0, 0, 0, 0),
        (0, 0, 1, 1, 1, 1, 1),
        (0, 0, 2, 2, 2, 2, 2),
        (0, 0, 2, 3, 3, 3, 3),
        (0, 0, 2, 4, 4, 4, 4),
        (0, 0, 2, 4, 5, 5, 5),
        (0, 0, 2, 4, 6, 6, 6),
    )
    qualifications = sum(
        qualification_maximum[foxes][target] for target in targets
    )
    maximum_types = min(3, sum(target >= 2 for target in targets))
    dp = [0] + [-1] * qualifications
    table = (0, 3, 5, 7)
    for _ in range(foxes):
        updated = dp[:]
        for used, score in enumerate(dp):
            if score < 0:
                continue
            for count in range(1, maximum_types + 1):
                if used + count <= qualifications:
                    updated[used + count] = max(updated[used + count], score + table[count])
        dp = updated
    return max(dp)


@cache
def _fox_a_upper_cached(foxes: int, targets: tuple[int, int, int, int]) -> int:
    subset_capacities = []
    for size in range(2, len(targets) + 1):
        for species in itertools.combinations(range(len(targets)), size):
            capacity = 2 if size == 2 else 1
            for index in species:
                capacity *= targets[index]
            mask = sum(1 << index for index in species)
            subset_capacities.append((mask, min(foxes, capacity)))

    best = 0
    for observed_masks in itertools.combinations_with_replacement(
        range(1 << len(targets)), foxes
    ):
        score = sum(mask.bit_count() for mask in observed_masks)
        if score <= best:
            continue
        if all(
            sum(mask & subset == subset for mask in observed_masks) <= capacity
            for subset, capacity in subset_capacities
        ):
            best = score
    return best + (foxes if foxes >= 2 else 0)


def _fox_a_upper(foxes: int, targets: tuple[int, int, int, int]) -> int:
    return _fox_a_upper_cached(foxes, tuple(sorted(targets)))


def count_upper(
    counts: tuple[int, int, int, int, int],
    ruleset: str,
) -> int:
    if len(counts) != len(SPECIES) or sum(counts) != TOKEN_COUNT:
        raise ValueError(f"invalid counts: {counts}")
    if any(count < 0 or count > COUNT_CAP for count in counts):
        raise ValueError(f"invalid counts: {counts}")
    cards = parse_ruleset(ruleset)
    bear, elk, salmon, hawk, fox = counts
    total = _standalone_bear(bear, cards[0])
    total += _standalone_elk(elk, cards[1])
    total += _standalone_salmon(salmon, cards[2], TOKEN_COUNT - salmon)
    if cards[3] == "A":
        total += (0, 2, 5, 8, 11, 14, 18)[hawk]
    elif cards[3] == "B":
        total += (0, 0, 5, 9, 12, 16, 20)[hawk]
    elif cards[3] == "C":
        # Along each of the three hex axes, only consecutive hawks on one
        # coordinate line can see one another.  For 2..6 distinct lattice
        # points the three projection sets contain at least n+3 values total,
        # so the visibility graph has at most 3n-(n+3)=2n-3 edges.  Scaling a
        # compact pattern by two shows the bound is tight through the cap.
        total += 3 * (0, 0, 1, 3, 5, 7, 9)[hawk]
    else:
        distinct_between = sum(count > 0 for count in (bear, elk, salmon, fox))
        total += (hawk // 2) * (0, 4, 7, 9)[min(distinct_between, 3)]
    if cards[4] == "A":
        total += _fox_a_upper(fox, counts[:4])
    elif cards[4] == "B":
        total += _fox_b_upper(fox, counts[:4])
    elif cards[4] == "C":
        total += _fox_c_upper(fox, counts[:4])
    else:
        doubled = sum(count >= 2 for count in counts[:4])
        total += (fox // 2) * (0, 5, 7, 9, 11)[doubled]
    return total


def global_count_upper(ruleset: str) -> tuple[int, list[tuple[int, int, int, int, int]]]:
    scored = [(counts, count_upper(counts, ruleset)) for counts in count_vectors()]
    maximum = max(score for _, score in scored)
    return maximum, [counts for counts, score in scored if score == maximum]
