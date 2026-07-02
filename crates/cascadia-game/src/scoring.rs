use arrayvec::ArrayVec;
use serde::{Deserialize, Serialize};

use crate::{
    Board, GameMode, GameState, HabitatAnalysis, HexCoord, MAX_BOARD_TILES, ScoringCards,
    ScoringVariant, Terrain, Tile, TileNeighborContext, TilePlacement, Wildlife,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ScoreBreakdown {
    pub habitat: [u16; 5],
    pub wildlife: [u16; 5],
    pub nature_tokens: u16,
    pub habitat_bonus: [u8; 5],
    pub base_total: u16,
    pub total: u16,
}

pub fn score_game(game: &GameState) -> Vec<ScoreBreakdown> {
    let config = game.config();
    let mut scores: Vec<_> = game
        .boards()
        .iter()
        .map(|board| score_board(board, config.scoring_cards))
        .collect();

    if config.habitat_bonuses {
        match config.mode {
            GameMode::Solo => {
                let habitat = scores[0].habitat;
                for (terrain, bonus) in Terrain::ALL
                    .into_iter()
                    .zip(scores[0].habitat_bonus.iter_mut())
                {
                    if habitat[terrain as usize] >= 7 {
                        *bonus = 2;
                    }
                }
            }
            GameMode::Standard => apply_multiplayer_habitat_bonuses(&mut scores),
        }
    }

    for score in &mut scores {
        score.total = score.base_total
            + score
                .habitat_bonus
                .iter()
                .map(|bonus| u16::from(*bonus))
                .sum::<u16>();
    }
    scores
}

pub fn score_board(board: &Board, cards: ScoringCards) -> ScoreBreakdown {
    let mut score = ScoreBreakdown::default();
    for terrain in Terrain::ALL {
        score.habitat[terrain as usize] = u16::from(board.largest_habitat(terrain));
    }
    score.wildlife = [
        score_bears(board, cards.bear),
        score_elk(board, cards.elk),
        score_salmon(board, cards.salmon),
        score_hawks(board, cards.hawk),
        score_foxes(board, cards.fox),
    ];
    score.nature_tokens = u16::from(board.nature_tokens());
    score.base_total = score.habitat.iter().sum::<u16>()
        + score.wildlife.iter().sum::<u16>()
        + score.nature_tokens;
    score.total = score.base_total;
    score
}

pub fn rescore_after_placement(
    board: &Board,
    cards: ScoringCards,
    baseline: ScoreBreakdown,
    placed_tile: crate::Tile,
    placed_wildlife: Option<Wildlife>,
) -> ScoreBreakdown {
    rescore_after_placement_with(
        board,
        cards,
        baseline,
        placed_tile,
        placed_wildlife,
        |terrain| board.largest_habitat(terrain),
    )
}

pub fn rescore_after_placement_with_habitat_analysis(
    board: &Board,
    cards: ScoringCards,
    baseline: ScoreBreakdown,
    analysis: &HabitatAnalysis,
    placement: TilePlacement,
    placed_tile: Tile,
    placed_wildlife: Option<Wildlife>,
) -> ScoreBreakdown {
    let after_tile = rescore_after_tile_with_habitat_analysis(
        board,
        cards,
        baseline,
        analysis,
        placement,
        placed_tile,
    );
    placed_wildlife.map_or(after_tile, |wildlife| {
        rescore_after_wildlife_placement(board, cards, after_tile, wildlife)
    })
}

pub fn rescore_after_tile_with_habitat_analysis(
    board: &Board,
    cards: ScoringCards,
    baseline: ScoreBreakdown,
    analysis: &HabitatAnalysis,
    placement: TilePlacement,
    placed_tile: Tile,
) -> ScoreBreakdown {
    rescore_after_placement_with(board, cards, baseline, placed_tile, None, |terrain| {
        analysis.largest_after_tile(
            board,
            placement.coord,
            placed_tile,
            placement.rotation,
            terrain,
        )
    })
}

/// Equivalent of [`rescore_after_tile_with_habitat_analysis`] that reuses a
/// prebuilt [`TileNeighborContext`] for the placement coordinate instead of
/// re-reading the neighbor cells on every rotation and terrain probe.
pub fn rescore_after_tile_with_neighbor_context(
    board: &Board,
    cards: ScoringCards,
    baseline: ScoreBreakdown,
    analysis: &HabitatAnalysis,
    context: &TileNeighborContext,
    rotation: crate::Rotation,
    placed_tile: Tile,
) -> ScoreBreakdown {
    rescore_after_placement_with(board, cards, baseline, placed_tile, None, |terrain| {
        analysis.largest_after_tile_with_context(context, placed_tile, rotation, terrain)
    })
}

pub fn rescore_after_wildlife_placement(
    board: &Board,
    cards: ScoringCards,
    baseline: ScoreBreakdown,
    placed_wildlife: Wildlife,
) -> ScoreBreakdown {
    let mut score = baseline;
    score.wildlife[placed_wildlife as usize] = score_wildlife(board, cards, placed_wildlife);
    if placed_wildlife != Wildlife::Fox {
        score.wildlife[Wildlife::Fox as usize] = score_wildlife(board, cards, Wildlife::Fox);
    }
    if placed_wildlife != Wildlife::Salmon && cards.salmon == ScoringVariant::D {
        score.wildlife[Wildlife::Salmon as usize] = score_wildlife(board, cards, Wildlife::Salmon);
    }
    if placed_wildlife != Wildlife::Hawk && cards.hawk == ScoringVariant::D {
        score.wildlife[Wildlife::Hawk as usize] = score_wildlife(board, cards, Wildlife::Hawk);
    }
    finish_score(board, &mut score);
    score
}

pub fn rescore_with_wildlife_scores(
    board: &Board,
    baseline: ScoreBreakdown,
    wildlife_scores: [u16; 5],
) -> ScoreBreakdown {
    let mut score = baseline;
    score.wildlife = wildlife_scores;
    finish_score(board, &mut score);
    score
}

fn rescore_after_placement_with(
    board: &Board,
    cards: ScoringCards,
    baseline: ScoreBreakdown,
    placed_tile: Tile,
    placed_wildlife: Option<Wildlife>,
    mut largest_habitat: impl FnMut(Terrain) -> u8,
) -> ScoreBreakdown {
    let mut score = baseline;
    score.habitat[placed_tile.terrain_a as usize] =
        u16::from(largest_habitat(placed_tile.terrain_a));
    if let Some(terrain) = placed_tile.terrain_b {
        score.habitat[terrain as usize] = u16::from(largest_habitat(terrain));
    }
    if let Some(wildlife) = placed_wildlife {
        score.wildlife[wildlife as usize] = score_wildlife(board, cards, wildlife);
        if wildlife != Wildlife::Fox {
            score.wildlife[Wildlife::Fox as usize] = score_wildlife(board, cards, Wildlife::Fox);
        }
        if wildlife != Wildlife::Salmon && cards.salmon == ScoringVariant::D {
            score.wildlife[Wildlife::Salmon as usize] =
                score_wildlife(board, cards, Wildlife::Salmon);
        }
        if wildlife != Wildlife::Hawk && cards.hawk == ScoringVariant::D {
            score.wildlife[Wildlife::Hawk as usize] = score_wildlife(board, cards, Wildlife::Hawk);
        }
    }
    finish_score(board, &mut score);
    score
}

fn finish_score(board: &Board, score: &mut ScoreBreakdown) {
    score.nature_tokens = u16::from(board.nature_tokens());
    score.base_total = score.habitat.iter().sum::<u16>()
        + score.wildlife.iter().sum::<u16>()
        + score.nature_tokens;
    score.total = score.base_total;
}

fn score_wildlife(board: &Board, cards: ScoringCards, wildlife: Wildlife) -> u16 {
    match wildlife {
        Wildlife::Bear => score_bears(board, cards.bear),
        Wildlife::Elk => score_elk(board, cards.elk),
        Wildlife::Salmon => score_salmon(board, cards.salmon),
        Wildlife::Hawk => score_hawks(board, cards.hawk),
        Wildlife::Fox => score_foxes(board, cards.fox),
    }
}

fn apply_multiplayer_habitat_bonuses(scores: &mut [ScoreBreakdown]) {
    for terrain in Terrain::ALL {
        let terrain_index = terrain as usize;
        let sizes: Vec<_> = scores
            .iter()
            .map(|score| score.habitat[terrain_index])
            .collect();
        if scores.len() == 2 {
            if sizes[0] == sizes[1] {
                scores[0].habitat_bonus[terrain_index] = 1;
                scores[1].habitat_bonus[terrain_index] = 1;
            } else {
                let winner = usize::from(sizes[1] > sizes[0]);
                scores[winner].habitat_bonus[terrain_index] = 2;
            }
            continue;
        }

        let largest = *sizes.iter().max().expect("standard games have players");
        let leaders: Vec<_> = sizes
            .iter()
            .enumerate()
            .filter_map(|(player, size)| (*size == largest).then_some(player))
            .collect();
        if leaders.len() == 1 {
            scores[leaders[0]].habitat_bonus[terrain_index] = 3;
            let second = sizes
                .iter()
                .copied()
                .filter(|size| *size < largest)
                .max()
                .unwrap_or(0);
            let runners_up: Vec<_> = sizes
                .iter()
                .enumerate()
                .filter_map(|(player, size)| (*size == second).then_some(player))
                .collect();
            if runners_up.len() == 1 {
                scores[runners_up[0]].habitat_bonus[terrain_index] = 1;
            }
        } else {
            let tied_bonus = if leaders.len() == 2 { 2 } else { 1 };
            for player in leaders {
                scores[player].habitat_bonus[terrain_index] = tied_bonus;
            }
        }
    }
}

fn wildlife_components(
    board: &Board,
    wildlife: Wildlife,
) -> ArrayVec<ArrayVec<HexCoord, MAX_BOARD_TILES>, MAX_BOARD_TILES> {
    let positions = board.wildlife_positions(wildlife);
    let mut remaining = positions.clone();
    let mut components = ArrayVec::new();
    while let Some(start) = remaining.pop() {
        let mut component = ArrayVec::new();
        component.push(start);
        let mut stack = ArrayVec::<HexCoord, MAX_BOARD_TILES>::new();
        stack.push(start);
        while let Some(coord) = stack.pop() {
            for neighbor in coord.neighbors() {
                if let Some(index) = remaining
                    .iter()
                    .position(|candidate| *candidate == neighbor)
                {
                    let found = remaining.swap_remove(index);
                    component.push(found);
                    stack.push(found);
                }
            }
        }
        components.push(component);
    }
    components
}

fn score_bears(board: &Board, variant: ScoringVariant) -> u16 {
    let sizes: Vec<_> = wildlife_components(board, Wildlife::Bear)
        .into_iter()
        .map(|component| component.len())
        .collect();
    match variant {
        ScoringVariant::A => match sizes.iter().filter(|size| **size == 2).count() {
            0 => 0,
            1 => 4,
            2 => 11,
            3 => 19,
            _ => 27,
        },
        ScoringVariant::B => sizes.iter().filter(|size| **size == 3).count() as u16 * 10,
        ScoringVariant::C => {
            let mut seen = [false; 3];
            let mut total = 0;
            for size in sizes {
                total += match size {
                    1 => {
                        seen[0] = true;
                        2
                    }
                    2 => {
                        seen[1] = true;
                        5
                    }
                    3 => {
                        seen[2] = true;
                        8
                    }
                    _ => 0,
                };
            }
            total + u16::from(seen.into_iter().all(|value| value)) * 3
        }
        ScoringVariant::D => sizes
            .into_iter()
            .map(|size| match size {
                2 => 5,
                3 => 8,
                4 => 13,
                _ => 0,
            })
            .sum(),
    }
}

fn score_elk(board: &Board, variant: ScoringVariant) -> u16 {
    let positions = board.wildlife_positions(Wildlife::Elk);
    match variant {
        ScoringVariant::A => score_elk_lines(&positions),
        ScoringVariant::B => score_elk_shapes(&positions),
        ScoringVariant::C => wildlife_components(board, Wildlife::Elk)
            .into_iter()
            .map(|component| score_connected_elk_component(&component))
            .sum(),
        ScoringVariant::D => score_elk_rings(&positions),
    }
}

fn score_elk_lines(positions: &[HexCoord]) -> u16 {
    if positions.is_empty() {
        return 0;
    }
    let mut groups = Vec::new();
    for index in 0..positions.len() {
        groups.push((1u32 << index, 2));
    }
    for (start_index, start) in positions.iter().enumerate() {
        for &(dq, dr) in &HexCoord::DIRECTIONS[..3] {
            let mut mask = 1u32 << start_index;
            let mut current = *start;
            for length in 2..=4 {
                current = HexCoord::new(current.q + dq, current.r + dr);
                let Some(index) = positions.iter().position(|coord| *coord == current) else {
                    break;
                };
                mask |= 1u32 << index;
                groups.push((
                    mask,
                    match length {
                        2 => 5,
                        3 => 9,
                        _ => 13,
                    },
                ));
            }
        }
    }
    maximize_disjoint_groups(positions.len(), &groups)
}

fn score_elk_shapes(positions: &[HexCoord]) -> u16 {
    let n = positions.len();
    if n == 0 {
        return 0;
    }
    let adjacency = local_adjacency(positions);
    let mut groups = Vec::new();
    for i in 0..n {
        groups.push((1u32 << i, 2));
        for j in (i + 1)..n {
            if adjacency[i] & (1 << j) == 0 {
                continue;
            }
            groups.push(((1 << i) | (1 << j), 5));
            for k in (j + 1)..n {
                let triangle = adjacency[i] & (1 << j) != 0
                    && adjacency[i] & (1 << k) != 0
                    && adjacency[j] & (1 << k) != 0;
                if !triangle {
                    continue;
                }
                let triangle_mask = (1 << i) | (1 << j) | (1 << k);
                groups.push((triangle_mask, 9));
                for l in 0..n {
                    if triangle_mask & (1 << l) == 0
                        && [i, j, k]
                            .into_iter()
                            .filter(|triangle_elk| adjacency[*triangle_elk] & (1 << l) != 0)
                            .count()
                            >= 2
                    {
                        groups.push((triangle_mask | (1 << l), 13));
                    }
                }
            }
        }
    }
    groups.sort_unstable();
    groups.dedup();
    maximize_disjoint_groups(n, &groups)
}

fn score_connected_elk_component(component: &[HexCoord]) -> u16 {
    let n = component.len();
    if n == 0 {
        return 0;
    }
    let adjacency = local_adjacency(component);
    let state_count = 1usize << n;
    let mut dp = vec![0u16; state_count];
    for state in 1..state_count {
        let first = state.trailing_zeros() as usize;
        let first_bit = 1usize << first;
        let others = state ^ first_bit;
        let mut subset = others;
        let mut best = 0;
        loop {
            let candidate = subset | first_bit;
            if connected_subset(candidate as u32, &adjacency) {
                let score = match candidate.count_ones() {
                    1 => 2,
                    2 => 4,
                    3 => 7,
                    4 => 10,
                    5 => 14,
                    6 => 18,
                    7 => 23,
                    _ => 28,
                };
                best = best.max(score + dp[state & !candidate]);
            }
            if subset == 0 {
                break;
            }
            subset = (subset - 1) & others;
        }
        dp[state] = best;
    }
    dp[state_count - 1]
}

fn score_elk_rings(positions: &[HexCoord]) -> u16 {
    let n = positions.len();
    if n == 0 {
        return 0;
    }
    let mut rings = Vec::new();
    for elk in positions {
        for center in elk.neighbors() {
            let mut mask = 0u32;
            for neighbor in center.neighbors() {
                if let Some(index) = positions.iter().position(|coord| *coord == neighbor) {
                    mask |= 1 << index;
                }
            }
            if mask != 0 {
                rings.push(mask);
            }
        }
    }
    rings.sort_unstable();
    rings.dedup();

    let state_count = 1usize << n;
    let mut dp = vec![0u16; state_count];
    for state in 1..state_count {
        let first = state.trailing_zeros();
        for ring in &rings {
            if ring & (1 << first) == 0 {
                continue;
            }
            let claimed = *ring & state as u32;
            let score = match claimed.count_ones() {
                0 => 0,
                1 => 2,
                2 => 5,
                3 => 8,
                4 => 12,
                5 => 16,
                _ => 21,
            };
            dp[state] = dp[state].max(score + dp[state & !(claimed as usize)]);
        }
    }
    dp[state_count - 1]
}

fn score_salmon(board: &Board, variant: ScoringVariant) -> u16 {
    wildlife_components(board, Wildlife::Salmon)
        .into_iter()
        .filter(|component| {
            component.iter().all(|coord| {
                coord
                    .neighbors()
                    .into_iter()
                    .filter(|neighbor| board.wildlife_at(*neighbor) == Some(Wildlife::Salmon))
                    .count()
                    <= 2
            })
        })
        .map(|run| match variant {
            ScoringVariant::A => match run.len() {
                1 => 2,
                2 => 5,
                3 => 8,
                4 => 12,
                5 => 16,
                6 => 20,
                _ => 25,
            },
            ScoringVariant::B => match run.len() {
                1 => 2,
                2 => 4,
                3 => 9,
                4 => 11,
                _ => 17,
            },
            ScoringVariant::C => match run.len() {
                0..=2 => 0,
                3 => 10,
                4 => 12,
                _ => 15,
            },
            ScoringVariant::D => {
                if run.len() < 3 {
                    return 0;
                }
                let mut adjacent = Vec::new();
                for salmon in &run {
                    for neighbor in salmon.neighbors() {
                        if let Some(wildlife) = board.wildlife_at(neighbor)
                            && wildlife != Wildlife::Salmon
                            && !adjacent.contains(&neighbor)
                        {
                            adjacent.push(neighbor);
                        }
                    }
                }
                run.len() as u16 + adjacent.len() as u16
            }
        })
        .sum()
}

fn score_hawks(board: &Board, variant: ScoringVariant) -> u16 {
    let positions = board.wildlife_positions(Wildlife::Hawk);
    let isolated: ArrayVec<_, MAX_BOARD_TILES> = positions
        .iter()
        .map(|hawk| {
            !hawk
                .neighbors()
                .into_iter()
                .any(|neighbor| board.wildlife_at(neighbor) == Some(Wildlife::Hawk))
        })
        .collect();
    match variant {
        ScoringVariant::A => hawk_count_score(isolated.iter().filter(|value| **value).count()),
        ScoringVariant::B => {
            let mut qualifying = [false; MAX_BOARD_TILES];
            for (left, right, _) in hawk_lines_of_sight(board, &positions) {
                if isolated[left] {
                    qualifying[left] = true;
                }
                if isolated[right] {
                    qualifying[right] = true;
                }
            }
            match qualifying[..positions.len()]
                .iter()
                .filter(|value| **value)
                .count()
            {
                0 | 1 => 0,
                2 => 5,
                3 => 9,
                4 => 12,
                5 => 16,
                6 => 20,
                7 => 24,
                _ => 28,
            }
        }
        ScoringVariant::C => hawk_lines_of_sight(board, &positions).len() as u16 * 3,
        ScoringVariant::D => {
            let edges: Vec<_> = hawk_lines_of_sight(board, &positions)
                .into_iter()
                .filter_map(|(left, right, between)| {
                    let unique_types = between.count_ones();
                    let score = match unique_types {
                        0 => 0,
                        1 => 4,
                        2 => 7,
                        _ => 9,
                    };
                    (score > 0).then_some((left, right, score))
                })
                .collect();
            maximum_weight_matching(positions.len(), &edges)
        }
    }
}

fn hawk_count_score(count: usize) -> u16 {
    match count {
        0 => 0,
        1 => 2,
        2 => 5,
        3 => 8,
        4 => 11,
        5 => 14,
        6 => 18,
        7 => 22,
        _ => 26,
    }
}

fn hawk_lines_of_sight(board: &Board, positions: &[HexCoord]) -> Vec<(usize, usize, u8)> {
    let mut pairs = Vec::new();
    for (left_index, left) in positions.iter().enumerate() {
        for &(dq, dr) in &HexCoord::DIRECTIONS {
            let mut current = HexCoord::new(left.q + dq, left.r + dr);
            let mut distance = 1;
            let mut between = 0u8;
            loop {
                if let Some(right_index) = positions.iter().position(|hawk| *hawk == current) {
                    if left_index < right_index && distance > 1 {
                        pairs.push((left_index, right_index, between));
                    }
                    break;
                }
                if let Some(wildlife) = board.wildlife_at(current) {
                    between |= 1 << wildlife as u8;
                }
                let next = HexCoord::new(current.q + dq, current.r + dr);
                if next.to_index().is_none() {
                    break;
                }
                current = next;
                distance += 1;
            }
        }
    }
    pairs
}

fn score_foxes(board: &Board, variant: ScoringVariant) -> u16 {
    let positions = board.wildlife_positions(Wildlife::Fox);
    match variant {
        ScoringVariant::A => positions
            .iter()
            .map(|fox| {
                fox.neighbors()
                    .into_iter()
                    .filter_map(|neighbor| board.wildlife_at(neighbor))
                    .fold(0u8, |mask, wildlife| mask | (1 << wildlife as u8))
                    .count_ones() as u16
            })
            .sum(),
        ScoringVariant::B => positions
            .iter()
            .map(|fox| {
                match adjacent_non_fox_counts(board, *fox)
                    .into_iter()
                    .filter(|count| *count >= 2)
                    .count()
                {
                    0 => 0,
                    1 => 3,
                    2 => 5,
                    _ => 7,
                }
            })
            .sum(),
        ScoringVariant::C => positions
            .iter()
            .map(|fox| {
                u16::from(
                    adjacent_non_fox_counts(board, *fox)
                        .into_iter()
                        .max()
                        .unwrap_or(0),
                )
            })
            .sum(),
        ScoringVariant::D => {
            let mut edges = Vec::new();
            for left in 0..positions.len() {
                for right in (left + 1)..positions.len() {
                    if positions[left].distance(positions[right]) != 1 {
                        continue;
                    }
                    let mut surrounding = ArrayVec::<HexCoord, 12>::new();
                    for center in [positions[left], positions[right]] {
                        for neighbor in center.neighbors() {
                            if neighbor != positions[left]
                                && neighbor != positions[right]
                                && !surrounding.contains(&neighbor)
                            {
                                surrounding.push(neighbor);
                            }
                        }
                    }
                    let mut counts = [0u8; 5];
                    for coord in surrounding {
                        if let Some(wildlife) = board.wildlife_at(coord)
                            && wildlife != Wildlife::Fox
                        {
                            counts[wildlife as usize] += 1;
                        }
                    }
                    let score = match counts.into_iter().filter(|count| *count >= 2).count() {
                        0 => 0,
                        1 => 5,
                        2 => 7,
                        3 => 9,
                        _ => 11,
                    };
                    if score > 0 {
                        edges.push((left, right, score));
                    }
                }
            }
            maximum_weight_matching(positions.len(), &edges)
        }
    }
}

fn adjacent_non_fox_counts(board: &Board, fox: HexCoord) -> [u8; 5] {
    let mut counts = [0u8; 5];
    for neighbor in fox.neighbors() {
        if let Some(wildlife) = board.wildlife_at(neighbor)
            && wildlife != Wildlife::Fox
        {
            counts[wildlife as usize] += 1;
        }
    }
    counts
}

fn local_adjacency(positions: &[HexCoord]) -> ArrayVec<u32, MAX_BOARD_TILES> {
    let mut adjacency: ArrayVec<u32, MAX_BOARD_TILES> =
        std::iter::repeat_n(0, positions.len()).collect();
    for left in 0..positions.len() {
        for right in (left + 1)..positions.len() {
            if positions[left].distance(positions[right]) == 1 {
                adjacency[left] |= 1 << right;
                adjacency[right] |= 1 << left;
            }
        }
    }
    adjacency
}

fn connected_subset(mask: u32, adjacency: &[u32]) -> bool {
    let mut visited = 1u32 << mask.trailing_zeros();
    loop {
        let mut expanded = visited;
        let mut frontier = visited;
        while frontier != 0 {
            let index = frontier.trailing_zeros() as usize;
            frontier &= frontier - 1;
            expanded |= adjacency[index] & mask;
        }
        if expanded == visited {
            return visited == mask;
        }
        visited = expanded;
    }
}

fn maximize_disjoint_groups(n: usize, groups: &[(u32, u16)]) -> u16 {
    let state_count = 1usize << n;
    let mut dp = vec![0u16; state_count];
    for state in 1..state_count {
        let first = state.trailing_zeros();
        for (group, score) in groups {
            if group & (1 << first) != 0 && group & state as u32 == *group {
                dp[state] = dp[state].max(*score + dp[state & !(*group as usize)]);
            }
        }
    }
    dp[state_count - 1]
}

fn maximum_weight_matching(n: usize, edges: &[(usize, usize, u16)]) -> u16 {
    let state_count = 1usize << n;
    let mut dp = vec![0u16; state_count];
    for state in 1..state_count {
        let first = state.trailing_zeros() as usize;
        let without_first = state & !(1 << first);
        let mut best = dp[without_first];
        for &(left, right, score) in edges {
            let partner = if left == first {
                right
            } else if right == first {
                left
            } else {
                continue;
            };
            if without_first & (1 << partner) != 0 {
                best = best.max(score + dp[without_first & !(1 << partner)]);
            }
        }
        dp[state] = best;
    }
    dp[state_count - 1]
}

#[cfg(test)]
mod tests {
    use crate::{Rotation, Tile, WildlifeMask};

    use super::*;

    fn wildlife_tile(id: u8) -> Tile {
        Tile {
            id: crate::TileId(id),
            terrain_a: Terrain::Forest,
            terrain_b: None,
            wildlife: WildlifeMask::from_bits(0b1_1111),
            keystone: false,
        }
    }

    fn board_with_wildlife(entries: &[(HexCoord, Wildlife)]) -> Board {
        let mut board = Board::empty();
        for (index, (coord, wildlife)) in entries.iter().enumerate() {
            board
                .insert_scoring_fixture(
                    *coord,
                    wildlife_tile(200 + index as u8),
                    Rotation::ZERO,
                    Some(*wildlife),
                )
                .unwrap();
        }
        board
    }

    #[test]
    fn aaaaa_reference_patterns_match_card_tables() {
        let bear_board = board_with_wildlife(&[
            (HexCoord::new(0, 0), Wildlife::Bear),
            (HexCoord::new(1, 0), Wildlife::Bear),
            (HexCoord::new(4, 0), Wildlife::Bear),
            (HexCoord::new(5, 0), Wildlife::Bear),
        ]);
        assert_eq!(score_bears(&bear_board, ScoringVariant::A), 11);

        let elk_board = board_with_wildlife(&[
            (HexCoord::new(0, 0), Wildlife::Elk),
            (HexCoord::new(1, 0), Wildlife::Elk),
            (HexCoord::new(2, 0), Wildlife::Elk),
            (HexCoord::new(3, 0), Wildlife::Elk),
        ]);
        assert_eq!(score_elk(&elk_board, ScoringVariant::A), 13);

        let salmon_board = board_with_wildlife(&[
            (HexCoord::new(0, 0), Wildlife::Salmon),
            (HexCoord::new(1, 0), Wildlife::Salmon),
            (HexCoord::new(2, 0), Wildlife::Salmon),
            (HexCoord::new(3, 0), Wildlife::Salmon),
        ]);
        assert_eq!(score_salmon(&salmon_board, ScoringVariant::A), 12);

        let hawk_board = board_with_wildlife(&[
            (HexCoord::new(0, 0), Wildlife::Hawk),
            (HexCoord::new(3, 0), Wildlife::Hawk),
        ]);
        assert_eq!(score_hawks(&hawk_board, ScoringVariant::A), 5);

        let fox_board = board_with_wildlife(&[
            (HexCoord::new(0, 0), Wildlife::Fox),
            (HexCoord::new(1, 0), Wildlife::Bear),
            (HexCoord::new(0, 1), Wildlife::Elk),
            (HexCoord::new(-1, 1), Wildlife::Bear),
        ]);
        assert_eq!(score_foxes(&fox_board, ScoringVariant::A), 2);
    }

    #[test]
    fn elk_a_scores_crossing_lines_with_no_double_counting() {
        let board = board_with_wildlife(&[
            (HexCoord::new(-2, 0), Wildlife::Elk),
            (HexCoord::new(-1, 0), Wildlife::Elk),
            (HexCoord::new(0, 0), Wildlife::Elk),
            (HexCoord::new(1, 0), Wildlife::Elk),
            (HexCoord::new(0, -1), Wildlife::Elk),
            (HexCoord::new(0, 1), Wildlife::Elk),
        ]);
        assert_eq!(score_elk(&board, ScoringVariant::A), 17);
    }

    #[test]
    fn branching_salmon_run_scores_zero() {
        let board = board_with_wildlife(&[
            (HexCoord::new(0, 0), Wildlife::Salmon),
            (HexCoord::new(1, 0), Wildlife::Salmon),
            (HexCoord::new(-1, 0), Wildlife::Salmon),
            (HexCoord::new(0, -1), Wildlife::Salmon),
        ]);
        assert_eq!(score_salmon(&board, ScoringVariant::A), 0);
    }

    #[test]
    fn salmon_d_requires_a_run_of_at_least_three() {
        let short = board_with_wildlife(&[
            (HexCoord::new(0, 0), Wildlife::Salmon),
            (HexCoord::new(1, 0), Wildlife::Salmon),
            (HexCoord::new(0, 1), Wildlife::Bear),
        ]);
        let scoring = board_with_wildlife(&[
            (HexCoord::new(0, 0), Wildlife::Salmon),
            (HexCoord::new(1, 0), Wildlife::Salmon),
            (HexCoord::new(2, 0), Wildlife::Salmon),
            (HexCoord::new(0, 1), Wildlife::Bear),
        ]);

        assert_eq!(score_salmon(&short, ScoringVariant::D), 0);
        assert_eq!(score_salmon(&scoring, ScoringVariant::D), 4);
    }

    #[test]
    fn hawk_b_requires_isolation_as_well_as_line_of_sight() {
        let board = board_with_wildlife(&[
            (HexCoord::new(0, 0), Wildlife::Hawk),
            (HexCoord::new(1, 0), Wildlife::Hawk),
            (HexCoord::new(3, 0), Wildlife::Hawk),
        ]);
        assert_eq!(score_hawks(&board, ScoringVariant::B), 0);
    }

    #[test]
    fn score_breakdown_keeps_base_and_bonus_separate() {
        let game = GameState::new(
            crate::GameConfig::research_aaaaa(2).unwrap(),
            crate::GameSeed::from_u64(1),
        )
        .unwrap();
        let scores = score_game(&game);
        assert!(scores.iter().all(|score| score.total == score.base_total));
    }
}
