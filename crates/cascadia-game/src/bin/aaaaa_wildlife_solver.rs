//! Optimize a pure-wildlife, all-Card-A Cascadia layout.
//!
//! The search state is exactly twenty occupied cells on one connected hex
//! polyhex. Habitat, tile compatibility, Nature tokens, drafting, and every
//! other game mechanic are deliberately absent. Each wildlife species may
//! appear at most six times.

use std::collections::{BTreeMap, VecDeque};
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use cascadia_game::{
    Board, HexCoord, Rotation, ScoringCards, Terrain, Tile, TileId, Wildlife, WildlifeMask,
    score_board,
};
use serde::{Deserialize, Serialize};

const TOKEN_COUNT: usize = 20;
const SPECIES_COUNT: usize = 5;
const COUNT_CAP: u8 = 6;
const SEARCH_RADIUS: i8 = 7;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Token {
    coord: HexCoord,
    wildlife: Wildlife,
}

#[derive(Clone, Debug)]
struct Layout {
    tokens: Vec<Token>,
}

impl Layout {
    fn wildlife_at(&self, coord: HexCoord) -> Option<Wildlife> {
        self.tokens
            .iter()
            .find_map(|token| (token.coord == coord).then_some(token.wildlife))
    }

    fn occupied(&self, coord: HexCoord) -> bool {
        self.tokens.iter().any(|token| token.coord == coord)
    }

    fn positions(&self, wildlife: Wildlife) -> Vec<HexCoord> {
        self.tokens
            .iter()
            .filter_map(|token| (token.wildlife == wildlife).then_some(token.coord))
            .collect()
    }

    fn counts(&self) -> [u8; SPECIES_COUNT] {
        let mut counts = [0; SPECIES_COUNT];
        for token in &self.tokens {
            counts[token.wildlife as usize] += 1;
        }
        counts
    }

    fn is_connected(&self) -> bool {
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

    fn normalize(&mut self) {
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
struct WildlifeScore {
    by_species: [u16; SPECIES_COUNT],
}

impl WildlifeScore {
    fn total(self) -> u16 {
        self.by_species.into_iter().sum()
    }
}

fn components(layout: &Layout, wildlife: Wildlife) -> Vec<Vec<HexCoord>> {
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

fn maximize_disjoint_groups(n: usize, groups: &[(u32, u16)]) -> u16 {
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

fn score_layout(layout: &Layout) -> WildlifeScore {
    let mut score = WildlifeScore::default();

    let bear_pairs = components(layout, Wildlife::Bear)
        .into_iter()
        .filter(|component| component.len() == 2)
        .count();
    score.by_species[Wildlife::Bear as usize] = match bear_pairs {
        0 => 0,
        1 => 4,
        2 => 11,
        3 => 19,
        _ => 27,
    };

    let elk = layout.positions(Wildlife::Elk);
    let mut elk_groups: Vec<(u32, u16)> = (0..elk.len()).map(|index| (1 << index, 2)).collect();
    for (start_index, start) in elk.iter().enumerate() {
        for &(dq, dr) in &HexCoord::DIRECTIONS[..3] {
            let mut mask = 1 << start_index;
            let mut current = *start;
            for length in 2..=4 {
                current = HexCoord::new(current.q + dq, current.r + dr);
                let Some(index) = elk.iter().position(|coord| *coord == current) else {
                    break;
                };
                mask |= 1 << index;
                elk_groups.push((
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
    score.by_species[Wildlife::Elk as usize] = maximize_disjoint_groups(elk.len(), &elk_groups);

    score.by_species[Wildlife::Salmon as usize] = components(layout, Wildlife::Salmon)
        .into_iter()
        .filter(|component| {
            component.iter().all(|coord| {
                coord
                    .neighbors()
                    .into_iter()
                    .filter(|neighbor| layout.wildlife_at(*neighbor) == Some(Wildlife::Salmon))
                    .count()
                    <= 2
            })
        })
        .map(|component| match component.len() {
            1 => 2,
            2 => 5,
            3 => 8,
            4 => 12,
            5 => 16,
            6 => 20,
            _ => 25,
        })
        .sum();

    let isolated_hawks = layout
        .positions(Wildlife::Hawk)
        .into_iter()
        .filter(|hawk| {
            !hawk
                .neighbors()
                .into_iter()
                .any(|neighbor| layout.wildlife_at(neighbor) == Some(Wildlife::Hawk))
        })
        .count();
    score.by_species[Wildlife::Hawk as usize] = match isolated_hawks {
        0 => 0,
        1 => 2,
        2 => 5,
        3 => 8,
        4 => 11,
        5 => 14,
        6 => 18,
        7 => 22,
        _ => 26,
    };

    score.by_species[Wildlife::Fox as usize] = layout
        .positions(Wildlife::Fox)
        .into_iter()
        .map(|fox| {
            fox.neighbors()
                .into_iter()
                .filter_map(|neighbor| layout.wildlife_at(neighbor))
                .fold(0u8, |mask, wildlife| mask | (1 << wildlife as u8))
                .count_ones() as u16
        })
        .sum();

    score
}

fn standalone_score(wildlife: Wildlife, count: u8) -> u16 {
    match wildlife {
        Wildlife::Bear => [0, 0, 4, 4, 11, 11, 19][count as usize],
        Wildlife::Elk => [0, 2, 5, 9, 13, 15, 18][count as usize],
        Wildlife::Salmon => [0, 2, 5, 8, 12, 16, 20][count as usize],
        Wildlife::Hawk => [0, 2, 5, 8, 11, 14, 18][count as usize],
        Wildlife::Fox => 0,
    }
}

fn count_relaxation(counts: [u8; SPECIES_COUNT]) -> u16 {
    let non_fox_types = counts[..4].iter().filter(|count| **count > 0).count() as u16;
    let fox_types = non_fox_types + u16::from(counts[Wildlife::Fox as usize] >= 2);
    Wildlife::ALL[..4]
        .iter()
        .map(|wildlife| standalone_score(*wildlife, counts[*wildlife as usize]))
        .sum::<u16>()
        + u16::from(counts[Wildlife::Fox as usize]) * fox_types
}

fn enumerate_count_vectors() -> Vec<([u8; SPECIES_COUNT], u16)> {
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
struct Rng(u64);

impl Rng {
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

fn known_optimal_layout() -> Layout {
    use Wildlife::{Bear as B, Elk as E, Fox as F, Salmon as S};
    Layout {
        tokens: [
            (3, 0, E),
            (1, 1, B),
            (2, 1, F),
            (3, 1, E),
            (0, 2, B),
            (1, 2, S),
            (2, 2, F),
            (3, 2, E),
            (1, 3, S),
            (2, 3, B),
            (3, 3, E),
            (0, 4, S),
            (1, 4, B),
            (2, 4, F),
            (3, 4, F),
            (0, 5, S),
            (1, 5, S),
            (2, 5, S),
            (3, 5, B),
            (2, 6, B),
        ]
        .into_iter()
        .map(|(q, r, wildlife)| Token {
            coord: HexCoord::new(q, r),
            wildlife,
        })
        .collect(),
    }
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

fn mutate(layout: &mut Layout, rng: &mut Rng) -> bool {
    match rng.usize(10) {
        0..=4 => {
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
        5..=6 => {
            let index = rng.usize(TOKEN_COUNT);
            let counts = layout.counts();
            let mut replacement = Wildlife::ALL[rng.usize(SPECIES_COUNT)];
            for _ in 0..SPECIES_COUNT {
                if replacement != layout.tokens[index].wildlife
                    && counts[replacement as usize] < COUNT_CAP
                {
                    layout.tokens[index].wildlife = replacement;
                    return true;
                }
                replacement = Wildlife::ALL[(replacement as usize + 1) % SPECIES_COUNT];
            }
            false
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

fn anneal(
    restarts: usize,
    iterations: usize,
    seed: u64,
    progress: bool,
) -> (Layout, WildlifeScore, u64) {
    let count_vectors = enumerate_count_vectors();
    let eligible: Vec<_> = count_vectors
        .iter()
        .filter(|(_, bound)| *bound >= 66)
        .copied()
        .collect();
    let mut rng = Rng(seed.max(1));
    let mut global_layout = random_layout(count_vectors[0].0, &mut rng);
    let mut global_score = score_layout(&global_layout);
    let mut evaluated = 1u64;

    for restart in 0..restarts {
        let counts = eligible[restart % eligible.len()].0;
        let mut current = random_layout(counts, &mut rng);
        let mut current_score = score_layout(&current);
        evaluated += 1;
        let mut local_best = current.clone();
        let mut local_best_score = current_score;

        for iteration in 0..iterations {
            let prior = current.clone();
            if !mutate(&mut current, &mut rng) {
                continue;
            }
            let candidate_score = score_layout(&current);
            evaluated += 1;
            let fraction = iteration as f64 / iterations.max(1) as f64;
            let temperature = 4.0 * (0.04f64 / 4.0).powf(fraction);
            let delta = f64::from(candidate_score.total()) - f64::from(current_score.total());
            if delta >= 0.0 || rng.unit() < (delta / temperature).exp() {
                current_score = candidate_score;
                if current_score.total() > local_best_score.total() {
                    local_best = current.clone();
                    local_best_score = current_score;
                }
                if current_score.total() > global_score.total() {
                    global_layout = current.clone();
                    global_score = current_score;
                    if progress {
                        eprintln!(
                            "best={} parts={:?} counts={:?} restart={} iteration={}",
                            global_score.total(),
                            global_score.by_species,
                            global_layout.counts(),
                            restart,
                            iteration
                        );
                    }
                }
            } else {
                current = prior;
            }
        }

        current = local_best;
        current_score = local_best_score;
        for _ in 0..(iterations / 10) {
            let prior = current.clone();
            if !mutate(&mut current, &mut rng) {
                continue;
            }
            let candidate_score = score_layout(&current);
            evaluated += 1;
            if candidate_score.total() >= current_score.total() {
                current_score = candidate_score;
                if current_score.total() > global_score.total() {
                    global_layout = current.clone();
                    global_score = current_score;
                }
            } else {
                current = prior;
            }
        }
    }

    global_layout.normalize();
    (global_layout, global_score, evaluated)
}

fn anneal_fixed_counts(
    counts: [u8; SPECIES_COUNT],
    restarts: usize,
    iterations: usize,
    seed: u64,
) -> (Layout, WildlifeScore, u64) {
    let mut rng = Rng(seed.max(1));
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
            let temperature = 3.0 * (0.025f64 / 3.0).powf(fraction);
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

#[derive(Debug, Serialize)]
struct CatalogToken {
    q: i8,
    r: i8,
    wildlife: &'static str,
}

#[derive(Debug, Serialize)]
struct CatalogCandidate {
    counts: [u8; SPECIES_COUNT],
    count_relaxation: u16,
    score: u16,
    score_breakdown: [u16; SPECIES_COUNT],
    upper_bound_matched: bool,
    states_evaluated: u64,
    tokens: Vec<CatalogToken>,
}

#[derive(Debug, Serialize)]
struct CatalogCandidateFile {
    schema: &'static str,
    token_count: usize,
    count_cap: u8,
    seed: u64,
    threads: usize,
    restarts_per_count: usize,
    iterations_per_restart: usize,
    elapsed_seconds: f64,
    candidates: Vec<CatalogCandidate>,
}

fn wildlife_name(wildlife: Wildlife) -> &'static str {
    match wildlife {
        Wildlife::Bear => "bear",
        Wildlife::Elk => "elk",
        Wildlife::Salmon => "salmon",
        Wildlife::Hawk => "hawk",
        Wildlife::Fox => "fox",
    }
}

fn parse_wildlife(name: &str) -> Wildlife {
    match name {
        "bear" => Wildlife::Bear,
        "elk" => Wildlife::Elk,
        "salmon" => Wildlife::Salmon,
        "hawk" => Wildlife::Hawk,
        "fox" => Wildlife::Fox,
        _ => panic!("unknown wildlife in catalog: {name}"),
    }
}

#[derive(Debug, Deserialize)]
struct VerificationToken {
    q: i8,
    r: i8,
    wildlife: String,
}

#[derive(Debug, Deserialize)]
struct VerificationResult {
    counts: [u8; SPECIES_COUNT],
    optimum: u16,
    score_breakdown: [u16; SPECIES_COUNT],
    proof_complete: bool,
    tokens: Vec<VerificationToken>,
}

#[derive(Debug, Deserialize)]
struct VerificationCatalog {
    proof_complete: bool,
    allocation_count: usize,
    results: Vec<VerificationResult>,
}

fn verify_catalog(path: &Path) {
    let encoded = fs::read_to_string(path).expect("read exact catalog");
    let catalog: VerificationCatalog =
        serde_json::from_str(&encoded).expect("parse exact catalog JSON");
    assert!(
        catalog.proof_complete,
        "catalog does not claim a complete proof"
    );
    let vectors = enumerate_count_vectors();
    assert_eq!(catalog.allocation_count, vectors.len());
    assert_eq!(catalog.results.len(), vectors.len());

    for (index, (result, (expected_counts, _))) in
        catalog.results.iter().zip(vectors.iter()).enumerate()
    {
        assert!(result.proof_complete, "result {index} proof is incomplete");
        assert_eq!(
            &result.counts, expected_counts,
            "result {index} count order"
        );
        assert_eq!(
            result.tokens.len(),
            TOKEN_COUNT,
            "result {index} token count"
        );
        let layout = Layout {
            tokens: result
                .tokens
                .iter()
                .map(|token| Token {
                    coord: HexCoord::new(token.q, token.r),
                    wildlife: parse_wildlife(&token.wildlife),
                })
                .collect(),
        };
        let unique: BTreeMap<_, _> = layout
            .tokens
            .iter()
            .map(|token| (token.coord, token.wildlife))
            .collect();
        assert_eq!(unique.len(), TOKEN_COUNT, "result {index} overlaps");
        assert!(layout.is_connected(), "result {index} is disconnected");
        assert_eq!(layout.counts(), result.counts, "result {index} counts");
        let custom = score_layout(&layout);
        assert_eq!(
            custom.by_species, result.score_breakdown,
            "result {index} custom score"
        );
        assert_eq!(custom.total(), result.optimum, "result {index} optimum");
        assert_eq!(
            production_score(&layout),
            result.score_breakdown,
            "result {index} production score"
        );
    }
    println!(
        "verified {} exact AAAAA catalog boards with the production scorer",
        catalog.results.len()
    );
}

fn catalog_candidates(
    output: &Path,
    threads: usize,
    restarts: usize,
    iterations: usize,
    seed: u64,
) {
    let started = Instant::now();
    let vectors = Arc::new(enumerate_count_vectors());
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<Option<CatalogCandidate>>> =
        Mutex::new((0..vectors.len()).map(|_| None).collect());
    let thread_count = threads.max(1).min(vectors.len());

    std::thread::scope(|scope| {
        for _ in 0..thread_count {
            let vectors = Arc::clone(&vectors);
            let next = &next;
            let results = &results;
            scope.spawn(move || {
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    let Some(&(counts, bound)) = vectors.get(index) else {
                        break;
                    };
                    let count_seed = seed ^ (index as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15);
                    let (layout, score, states_evaluated) =
                        anneal_fixed_counts(counts, restarts, iterations, count_seed);
                    assert_eq!(layout.counts(), counts);
                    assert!(layout.is_connected());
                    assert_eq!(production_score(&layout), score.by_species);
                    let candidate = CatalogCandidate {
                        counts,
                        count_relaxation: bound,
                        score: score.total(),
                        score_breakdown: score.by_species,
                        upper_bound_matched: score.total() == bound,
                        states_evaluated,
                        tokens: layout
                            .tokens
                            .iter()
                            .map(|token| CatalogToken {
                                q: token.coord.q,
                                r: token.coord.r,
                                wildlife: wildlife_name(token.wildlife),
                            })
                            .collect(),
                    };
                    results.lock().expect("catalog result lock")[index] = Some(candidate);
                    eprintln!(
                        "candidate {}/{} counts={counts:?} score={}/{}",
                        index + 1,
                        vectors.len(),
                        score.total(),
                        bound
                    );
                }
            });
        }
    });

    let candidates = results
        .into_inner()
        .expect("catalog result lock")
        .into_iter()
        .map(|candidate| candidate.expect("every catalog task completed"))
        .collect();
    let payload = CatalogCandidateFile {
        schema: "aaaaa-wildlife-candidates-v1",
        token_count: TOKEN_COUNT,
        count_cap: COUNT_CAP,
        seed,
        threads: thread_count,
        restarts_per_count: restarts,
        iterations_per_restart: iterations,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        candidates,
    };
    let encoded = serde_json::to_string_pretty(&payload).expect("catalog JSON serializes") + "\n";
    let temporary_extension = output.extension().map_or_else(
        || "tmp".to_owned(),
        |extension| format!("{}.tmp", extension.to_string_lossy()),
    );
    let temporary = output.with_extension(temporary_extension);
    fs::write(&temporary, encoded).expect("write candidate catalog");
    fs::rename(&temporary, output).expect("publish candidate catalog atomically");
}

fn production_score(layout: &Layout) -> [u16; SPECIES_COUNT] {
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
    score_board(&board, ScoringCards::AAAAA).wildlife
}

fn species_letter(wildlife: Wildlife) -> char {
    match wildlife {
        Wildlife::Bear => 'B',
        Wildlife::Elk => 'E',
        Wildlife::Salmon => 'S',
        Wildlife::Hawk => 'H',
        Wildlife::Fox => 'F',
    }
}

fn render(
    layout: &Layout,
    score: WildlifeScore,
    evaluated: u64,
    elapsed: f64,
    optimality: &str,
) -> String {
    let production = production_score(layout);
    assert_eq!(score.by_species, production);
    let mut output = String::new();
    let count_vectors = enumerate_count_vectors();
    writeln!(output, "AAAAA pure-wildlife optimization").unwrap();
    writeln!(output, "score: {}", score.total()).unwrap();
    writeln!(output, "optimality: {optimality}").unwrap();
    writeln!(
        output,
        "breakdown [bear, elk, salmon, hawk, fox]: {:?}",
        score.by_species
    )
    .unwrap();
    writeln!(
        output,
        "counts    [bear, elk, salmon, hawk, fox]: {:?}",
        layout.counts()
    )
    .unwrap();
    writeln!(
        output,
        "count-only relaxation upper bound: {}",
        count_vectors[0].1
    )
    .unwrap();
    writeln!(output, "production scorer verified: yes").unwrap();
    writeln!(output, "connected: {}", layout.is_connected()).unwrap();
    writeln!(output, "states evaluated: {evaluated}").unwrap();
    writeln!(output, "elapsed seconds: {elapsed:.3}").unwrap();
    writeln!(output).unwrap();
    writeln!(output, "coordinates (q, r, wildlife):").unwrap();
    for token in &layout.tokens {
        writeln!(
            output,
            "  ({:>2}, {:>2}) {} {:?}",
            token.coord.q,
            token.coord.r,
            species_letter(token.wildlife),
            token.wildlife
        )
        .unwrap();
    }
    writeln!(output).unwrap();
    writeln!(output, "row view (axial r; dots are unoccupied):").unwrap();
    let by_coord: BTreeMap<_, _> = layout
        .tokens
        .iter()
        .map(|token| (token.coord, species_letter(token.wildlife)))
        .collect();
    let min_q = layout
        .tokens
        .iter()
        .map(|token| token.coord.q)
        .min()
        .unwrap();
    let max_q = layout
        .tokens
        .iter()
        .map(|token| token.coord.q)
        .max()
        .unwrap();
    let min_r = layout
        .tokens
        .iter()
        .map(|token| token.coord.r)
        .min()
        .unwrap();
    let max_r = layout
        .tokens
        .iter()
        .map(|token| token.coord.r)
        .max()
        .unwrap();
    for r in min_r..=max_r {
        write!(output, "r={r:>2} {}", " ".repeat((r - min_r) as usize)).unwrap();
        for q in min_q..=max_q {
            let glyph = by_coord.get(&HexCoord::new(q, r)).copied().unwrap_or('.');
            write!(output, " {glyph}").unwrap();
        }
        writeln!(output).unwrap();
    }
    output
}

fn parse_usize(args: &[String], name: &str, default: usize) -> usize {
    args.windows(2)
        .find_map(|pair| (pair[0] == name).then(|| pair[1].parse().expect("numeric CLI value")))
        .unwrap_or(default)
}

fn parse_u64(args: &[String], name: &str, default: u64) -> u64 {
    args.windows(2)
        .find_map(|pair| (pair[0] == name).then(|| pair[1].parse().expect("numeric CLI value")))
        .unwrap_or(default)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if let Some(input) = args
        .windows(2)
        .find_map(|pair| (pair[0] == "--verify-catalog").then(|| Path::new(&pair[1]).to_path_buf()))
    {
        verify_catalog(&input);
        return;
    }
    if let Some(output) = args.windows(2).find_map(|pair| {
        (pair[0] == "--catalog-candidates").then(|| Path::new(&pair[1]).to_path_buf())
    }) {
        let threads = parse_usize(
            &args,
            "--threads",
            std::thread::available_parallelism().map_or(1, usize::from),
        );
        let restarts = parse_usize(&args, "--restarts-per-count", 8);
        let iterations = parse_usize(&args, "--iterations-per-restart", 20_000);
        let seed = parse_u64(&args, "--seed", 0x5eed_a5a5_2026_0722);
        catalog_candidates(&output, threads, restarts, iterations, seed);
        return;
    }
    if args.iter().any(|arg| arg == "--show-optimum") {
        let layout = known_optimal_layout();
        let score = score_layout(&layout);
        print!(
            "{}",
            render(
                &layout,
                score,
                0,
                0.0,
                "certified; exact solver excludes every score >=69"
            )
        );
        return;
    }
    let restarts = parse_usize(&args, "--restarts", 160);
    let iterations = parse_usize(&args, "--iterations", 250_000);
    let seed = parse_u64(&args, "--seed", 0x5eed_a5a5_2026_0722);
    let started = Instant::now();
    let (layout, score, evaluated) = anneal(restarts, iterations, seed, true);
    print!(
        "{}",
        render(
            &layout,
            score,
            evaluated,
            started.elapsed().as_secs_f64(),
            "incumbent only; run the exact solver to certify"
        )
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(entries: &[(i8, i8, Wildlife)]) -> Layout {
        Layout {
            tokens: entries
                .iter()
                .map(|&(q, r, wildlife)| Token {
                    coord: HexCoord::new(q, r),
                    wildlife,
                })
                .collect(),
        }
    }

    #[test]
    fn count_vector_space_and_relaxation_are_pinned() {
        let vectors = enumerate_count_vectors();
        assert_eq!(vectors.len(), 826);
        assert_eq!(vectors[0], ([6, 1, 6, 1, 6], 73));
    }

    #[test]
    fn custom_score_matches_reference_patterns() {
        let sample = layout(&[
            (0, 0, Wildlife::Bear),
            (1, 0, Wildlife::Bear),
            (3, 0, Wildlife::Elk),
            (4, 0, Wildlife::Elk),
            (5, 0, Wildlife::Elk),
            (6, 0, Wildlife::Elk),
            (0, 1, Wildlife::Salmon),
            (1, 1, Wildlife::Salmon),
            (2, 1, Wildlife::Salmon),
            (3, 1, Wildlife::Hawk),
            (5, 1, Wildlife::Hawk),
            (2, 0, Wildlife::Fox),
        ]);
        assert_eq!(score_layout(&sample).by_species, [4, 13, 8, 5, 3]);
    }

    #[test]
    fn short_search_is_legal_and_production_verified() {
        let (candidate, score, _) = anneal(2, 2_000, 7, false);
        assert_eq!(candidate.tokens.len(), TOKEN_COUNT);
        assert!(candidate.is_connected());
        assert!(
            candidate
                .counts()
                .into_iter()
                .all(|count| count <= COUNT_CAP)
        );
        assert_eq!(score.by_species, production_score(&candidate));
    }

    #[test]
    fn bundled_optimum_is_production_verified() {
        let optimum = known_optimal_layout();
        assert!(optimum.is_connected());
        assert_eq!(optimum.counts(), [6, 4, 6, 0, 4]);
        assert_eq!(score_layout(&optimum).by_species, [19, 13, 20, 0, 16]);
        assert_eq!(production_score(&optimum), [19, 13, 20, 0, 16]);
    }
}
