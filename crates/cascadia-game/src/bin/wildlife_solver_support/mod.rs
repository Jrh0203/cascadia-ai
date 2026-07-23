use std::collections::VecDeque;

use cascadia_game::{
    Board, HexCoord, Rotation, ScoringCards, Terrain, Tile, TileId, Wildlife, WildlifeMask,
    score_board,
};

pub const TOKEN_COUNT: usize = 20;
pub const SPECIES_COUNT: usize = 5;
pub const COUNT_CAP: u8 = 6;
const SEARCH_RADIUS: i8 = 9;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Token {
    pub coord: HexCoord,
    pub wildlife: Wildlife,
}

#[derive(Clone, Debug)]
pub struct Layout {
    pub tokens: Vec<Token>,
}

impl Layout {
    pub fn wildlife_at(&self, coord: HexCoord) -> Option<Wildlife> {
        self.tokens
            .iter()
            .find_map(|token| (token.coord == coord).then_some(token.wildlife))
    }

    pub fn occupied(&self, coord: HexCoord) -> bool {
        self.tokens.iter().any(|token| token.coord == coord)
    }

    pub fn positions(&self, wildlife: Wildlife) -> Vec<HexCoord> {
        self.tokens
            .iter()
            .filter_map(|token| (token.wildlife == wildlife).then_some(token.coord))
            .collect()
    }

    pub fn counts(&self) -> [u8; SPECIES_COUNT] {
        let mut counts = [0; SPECIES_COUNT];
        for token in &self.tokens {
            counts[token.wildlife as usize] += 1;
        }
        counts
    }

    pub fn is_connected(&self) -> bool {
        if self.tokens.is_empty() {
            return true;
        }
        let mut seen = vec![false; self.tokens.len()];
        let mut queue = VecDeque::from([0]);
        seen[0] = true;
        while let Some(index) = queue.pop_front() {
            for neighbor in self.tokens[index].coord.neighbors() {
                if let Some(other) = self.tokens.iter().position(|token| token.coord == neighbor)
                    && !seen[other]
                {
                    seen[other] = true;
                    queue.push_back(other);
                }
            }
        }
        seen.into_iter().all(|value| value)
    }

    fn frontier(&self) -> Vec<HexCoord> {
        let mut frontier = Vec::with_capacity(self.tokens.len() * 3);
        for token in &self.tokens {
            for neighbor in token.coord.neighbors() {
                if !self.occupied(neighbor)
                    && neighbor.q.abs() <= SEARCH_RADIUS
                    && neighbor.r.abs() <= SEARCH_RADIUS
                    && (neighbor.q + neighbor.r).abs() <= SEARCH_RADIUS
                {
                    frontier.push(neighbor);
                }
            }
        }
        frontier.sort_unstable();
        frontier.dedup();
        frontier
    }

    pub fn normalize(&mut self) {
        let min_q = self
            .tokens
            .iter()
            .map(|token| token.coord.q)
            .min()
            .unwrap_or(0);
        let min_r = self
            .tokens
            .iter()
            .map(|token| token.coord.r)
            .min()
            .unwrap_or(0);
        for token in &mut self.tokens {
            token.coord = HexCoord::new(token.coord.q - min_q, token.coord.r - min_r);
        }
        self.tokens
            .sort_by_key(|token| (token.coord.r, token.coord.q, token.wildlife as u8));
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WildlifeScore {
    pub by_species: [u16; SPECIES_COUNT],
}

impl WildlifeScore {
    pub fn total(self) -> u16 {
        self.by_species.into_iter().sum()
    }
}

pub fn components(layout: &Layout, wildlife: Wildlife) -> Vec<Vec<HexCoord>> {
    let mut remaining = layout.positions(wildlife);
    let mut result = Vec::new();
    while let Some(start) = remaining.pop() {
        let mut component = vec![start];
        let mut stack = vec![start];
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
        result.push(component);
    }
    result
}

pub fn maximize_disjoint_groups(n: usize, groups: &[(u32, u16)]) -> u16 {
    let mut dp = vec![0; 1usize << n];
    for state in 1..(1usize << n) {
        let first = state.trailing_zeros();
        for &(group, score) in groups {
            if group & (1 << first) != 0 && group & state as u32 == group {
                dp[state] = dp[state].max(score + dp[state & !(group as usize)]);
            }
        }
    }
    dp[(1usize << n) - 1]
}

pub fn enumerate_count_vectors(
    count_relaxation: impl Fn([u8; SPECIES_COUNT]) -> u16,
) -> Vec<([u8; SPECIES_COUNT], u16)> {
    let mut vectors = Vec::new();
    for bear in 0..=COUNT_CAP {
        for elk in 0..=COUNT_CAP {
            for salmon in 0..=COUNT_CAP {
                for hawk in 0..=COUNT_CAP {
                    for fox in 0..=COUNT_CAP {
                        let counts = [bear, elk, salmon, hawk, fox];
                        if counts.into_iter().map(usize::from).sum::<usize>() == TOKEN_COUNT {
                            vectors.push((counts, count_relaxation(counts)));
                        }
                    }
                }
            }
        }
    }
    vectors.sort_by_key(|(counts, bound)| (std::cmp::Reverse(*bound), *counts));
    vectors
}

#[derive(Clone, Copy)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn next_u64(&mut self) -> u64 {
        let mut value = self.0;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.0 = value;
        value
    }

    fn usize(&mut self, upper: usize) -> usize {
        (self.next_u64() as usize) % upper
    }

    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    fn shuffle<T>(&mut self, values: &mut [T]) {
        for index in (1..values.len()).rev() {
            values.swap(index, self.usize(index + 1));
        }
    }
}

fn initial_shape() -> Vec<HexCoord> {
    let mut cells = Vec::new();
    for q in -2i8..=2 {
        for r in -2i8..=2 {
            if q.abs().max(r.abs()).max((q + r).abs()) <= 2 {
                cells.push(HexCoord::new(q, r));
            }
        }
    }
    cells.push(HexCoord::new(3, 0));
    cells
}

fn random_layout(counts: [u8; SPECIES_COUNT], rng: &mut Rng) -> Layout {
    let mut wildlife = Vec::with_capacity(TOKEN_COUNT);
    for species in Wildlife::ALL {
        wildlife.extend(std::iter::repeat_n(
            species,
            counts[species as usize] as usize,
        ));
    }
    rng.shuffle(&mut wildlife);
    let mut cells = initial_shape();
    rng.shuffle(&mut cells);
    Layout {
        tokens: cells
            .into_iter()
            .zip(wildlife)
            .map(|(coord, wildlife)| Token { coord, wildlife })
            .collect(),
    }
}

fn mutate_fixed_counts(layout: &mut Layout, rng: &mut Rng) -> bool {
    match rng.usize(10) {
        0..=6 => {
            let left = rng.usize(TOKEN_COUNT);
            let mut right = rng.usize(TOKEN_COUNT - 1);
            if right >= left {
                right += 1;
            }
            layout.tokens.swap(left, right);
            let left_coord = layout.tokens[left].coord;
            let right_coord = layout.tokens[right].coord;
            layout.tokens[left].coord = right_coord;
            layout.tokens[right].coord = left_coord;
            true
        }
        _ => {
            let frontier = layout.frontier();
            if frontier.is_empty() {
                return false;
            }
            let index = rng.usize(TOKEN_COUNT);
            let prior = layout.tokens[index].coord;
            layout.tokens[index].coord = frontier[rng.usize(frontier.len())];
            if layout.is_connected() {
                true
            } else {
                layout.tokens[index].coord = prior;
                false
            }
        }
    }
}

pub fn anneal_fixed_counts(
    counts: [u8; SPECIES_COUNT],
    restarts: usize,
    iterations: usize,
    seed: u64,
    score_layout: impl Fn(&Layout) -> WildlifeScore,
    count_relaxation: impl Fn([u8; SPECIES_COUNT]) -> u16,
) -> (Layout, WildlifeScore, u64) {
    let mut rng = Rng::new(seed);
    let mut global_layout = random_layout(counts, &mut rng);
    let mut global_score = score_layout(&global_layout);
    let mut evaluated = 1u64;

    for _ in 0..restarts {
        let mut current = random_layout(counts, &mut rng);
        let mut current_score = score_layout(&current);
        evaluated += 1;
        for iteration in 0..iterations {
            let prior = current.clone();
            if !mutate_fixed_counts(&mut current, &mut rng) {
                continue;
            }
            let candidate_score = score_layout(&current);
            evaluated += 1;
            let fraction = iteration as f64 / iterations.max(1) as f64;
            let temperature = 4.0 * (0.025f64 / 4.0).powf(fraction);
            let delta = f64::from(candidate_score.total()) - f64::from(current_score.total());
            if delta >= 0.0 || rng.unit() < (delta / temperature).exp() {
                current_score = candidate_score;
                if current_score.total() > global_score.total() {
                    global_layout = current.clone();
                    global_score = current_score;
                    if global_score.total() == count_relaxation(counts) {
                        break;
                    }
                }
            } else {
                current = prior;
            }
        }
        if global_score.total() == count_relaxation(counts) {
            break;
        }
    }

    global_layout.normalize();
    (global_layout, global_score, evaluated)
}

pub fn production_score(layout: &Layout, cards: ScoringCards) -> [u16; SPECIES_COUNT] {
    assert!(layout.is_connected());
    let mut board = Board::empty();
    let mut pending = layout.tokens.clone();
    let first = pending.remove(0);
    let tile = |id| Tile {
        id: TileId(id),
        terrain_a: Terrain::Forest,
        terrain_b: None,
        wildlife: WildlifeMask::from_bits(0b1_1111),
        keystone: false,
    };
    board
        .place_tile(first.coord, tile(200), Rotation::ZERO)
        .expect("first synthetic tile places");
    board
        .place_wildlife(first.coord, first.wildlife)
        .expect("first synthetic wildlife places");
    let mut next_id = 201;
    while !pending.is_empty() {
        let index = pending
            .iter()
            .position(|token| {
                token
                    .coord
                    .neighbors()
                    .into_iter()
                    .any(|neighbor| board.tile_at(neighbor).is_some())
            })
            .expect("connected layout has an attachable pending cell");
        let token = pending.remove(index);
        board
            .place_tile(token.coord, tile(next_id), Rotation::ZERO)
            .expect("synthetic tile places");
        board
            .place_wildlife(token.coord, token.wildlife)
            .expect("synthetic wildlife places");
        next_id += 1;
    }
    score_board(&board, cards).wildlife
}

pub fn wildlife_name(wildlife: Wildlife) -> &'static str {
    match wildlife {
        Wildlife::Bear => "bear",
        Wildlife::Elk => "elk",
        Wildlife::Salmon => "salmon",
        Wildlife::Hawk => "hawk",
        Wildlife::Fox => "fox",
    }
}
