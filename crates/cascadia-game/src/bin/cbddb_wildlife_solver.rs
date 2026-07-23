//! Candidate generator and production verifier for pure-wildlife CBDDB boards.

mod wildlife_solver_support;

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use cascadia_game::{HexCoord, ScoringCards, Wildlife};
use serde::{Deserialize, Serialize};
use wildlife_solver_support::{
    COUNT_CAP, Layout, SPECIES_COUNT, TOKEN_COUNT, Token, WildlifeScore, anneal_fixed_counts,
    components, enumerate_count_vectors, maximize_disjoint_groups, production_score, wildlife_name,
};

const BEAR_STANDALONE: [u16; 7] = [0, 2, 5, 8, 10, 13, 18];
const ELK_STANDALONE: [u16; 7] = [0, 2, 5, 9, 13, 15, 18];
const SALMON_STANDALONE: [u16; 7] = [0, 0, 0, 13, 16, 19, 26];

fn score_layout(layout: &Layout) -> WildlifeScore {
    let mut score = WildlifeScore::default();

    let bear_sizes: Vec<_> = components(layout, Wildlife::Bear)
        .into_iter()
        .map(|component| component.len())
        .collect();
    let mut bear_seen = [false; 3];
    let mut bear_score = 0;
    for size in bear_sizes {
        bear_score += match size {
            1 => {
                bear_seen[0] = true;
                2
            }
            2 => {
                bear_seen[1] = true;
                5
            }
            3 => {
                bear_seen[2] = true;
                8
            }
            _ => 0,
        };
    }
    score.by_species[Wildlife::Bear as usize] =
        bear_score + u16::from(bear_seen.into_iter().all(|value| value)) * 3;

    let elk = layout.positions(Wildlife::Elk);
    let mut elk_adjacency = vec![0u32; elk.len()];
    for left in 0..elk.len() {
        for right in (left + 1)..elk.len() {
            if elk[left].distance(elk[right]) == 1 {
                elk_adjacency[left] |= 1 << right;
                elk_adjacency[right] |= 1 << left;
            }
        }
    }
    let mut elk_groups: Vec<(u32, u16)> = (0..elk.len()).map(|index| (1 << index, 2)).collect();
    for left in 0..elk.len() {
        for right in (left + 1)..elk.len() {
            if elk_adjacency[left] & (1 << right) == 0 {
                continue;
            }
            elk_groups.push(((1 << left) | (1 << right), 5));
            for third in (right + 1)..elk.len() {
                if elk_adjacency[left] & (1 << third) == 0
                    || elk_adjacency[right] & (1 << third) == 0
                {
                    continue;
                }
                let triangle = (1 << left) | (1 << right) | (1 << third);
                elk_groups.push((triangle, 9));
                for fourth in 0..elk.len() {
                    if triangle & (1 << fourth) != 0 {
                        continue;
                    }
                    let attached = [left, right, third]
                        .into_iter()
                        .filter(|triangle_elk| elk_adjacency[*triangle_elk] & (1 << fourth) != 0)
                        .count();
                    if attached >= 2 {
                        elk_groups.push((triangle | (1 << fourth), 13));
                    }
                }
            }
        }
    }
    elk_groups.sort_unstable();
    elk_groups.dedup();
    score.by_species[Wildlife::Elk as usize] = maximize_disjoint_groups(elk.len(), &elk_groups);

    score.by_species[Wildlife::Salmon as usize] = components(layout, Wildlife::Salmon)
        .into_iter()
        .filter(|component| {
            component.len() >= 3
                && component.iter().all(|coord| {
                    coord
                        .neighbors()
                        .into_iter()
                        .filter(|neighbor| layout.wildlife_at(*neighbor) == Some(Wildlife::Salmon))
                        .count()
                        <= 2
                })
        })
        .map(|component| {
            let mut adjacent = Vec::new();
            for salmon in &component {
                for neighbor in salmon.neighbors() {
                    if let Some(wildlife) = layout.wildlife_at(neighbor)
                        && wildlife != Wildlife::Salmon
                        && !adjacent.contains(&neighbor)
                    {
                        adjacent.push(neighbor);
                    }
                }
            }
            (component.len() + adjacent.len()) as u16
        })
        .sum();

    let hawks = layout.positions(Wildlife::Hawk);
    let mut hawk_edges = Vec::new();
    for left in 0..hawks.len() {
        for right in (left + 1)..hawks.len() {
            let Some(between) = ray_between(hawks[left], hawks[right]) else {
                continue;
            };
            if between
                .iter()
                .any(|coord| layout.wildlife_at(*coord) == Some(Wildlife::Hawk))
            {
                continue;
            }
            let mask = between
                .iter()
                .filter_map(|coord| layout.wildlife_at(*coord))
                .fold(0u8, |mask, wildlife| mask | (1 << wildlife as u8));
            let edge_score = match mask.count_ones() {
                0 => 0,
                1 => 4,
                2 => 7,
                _ => 9,
            };
            if edge_score > 0 {
                hawk_edges.push((left, right, edge_score));
            }
        }
    }
    score.by_species[Wildlife::Hawk as usize] = maximum_weight_matching(hawks.len(), &hawk_edges);

    score.by_species[Wildlife::Fox as usize] = layout
        .positions(Wildlife::Fox)
        .into_iter()
        .map(|fox| {
            let mut counts = [0u8; SPECIES_COUNT];
            for neighbor in fox.neighbors() {
                if let Some(wildlife) = layout.wildlife_at(neighbor)
                    && wildlife != Wildlife::Fox
                {
                    counts[wildlife as usize] += 1;
                }
            }
            match counts.into_iter().filter(|count| *count >= 2).count() {
                0 => 0,
                1 => 3,
                2 => 5,
                _ => 7,
            }
        })
        .sum();

    score
}

fn ray_between(left: HexCoord, right: HexCoord) -> Option<Vec<HexCoord>> {
    let delta_q = right.q - left.q;
    let delta_r = right.r - left.r;
    for (step_q, step_r) in HexCoord::DIRECTIONS {
        let mut distance = None;
        if step_q != 0 {
            if delta_q % step_q != 0 {
                continue;
            }
            distance = Some(delta_q / step_q);
        } else if delta_q != 0 {
            continue;
        }
        if step_r != 0 {
            if delta_r % step_r != 0 {
                continue;
            }
            let r_distance = delta_r / step_r;
            if distance.is_some_and(|value| value != r_distance) {
                continue;
            }
            distance = Some(r_distance);
        } else if delta_r != 0 {
            continue;
        }
        if let Some(distance) = distance.filter(|distance| *distance > 1) {
            return Some(
                (1..distance)
                    .map(|step| HexCoord::new(left.q + step * step_q, left.r + step * step_r))
                    .collect(),
            );
        }
    }
    None
}

fn maximum_weight_matching(n: usize, edges: &[(usize, usize, u16)]) -> u16 {
    let mut dp = vec![0u16; 1usize << n];
    for state in 1..dp.len() {
        let first = state.trailing_zeros() as usize;
        let without_first = state & !(1 << first);
        dp[state] = dp[without_first];
        for &(left, right, edge_score) in edges {
            let other = if left == first {
                right
            } else if right == first {
                left
            } else {
                continue;
            };
            if without_first & (1 << other) != 0 {
                dp[state] = dp[state].max(edge_score + dp[without_first & !(1 << other)]);
            }
        }
    }
    dp[(1 << n) - 1]
}

fn count_relaxation(counts: [u8; SPECIES_COUNT]) -> u16 {
    let [bear, elk, salmon, hawk, fox] = counts;
    let between_types = [bear, elk, salmon, fox]
        .into_iter()
        .filter(|count| *count > 0)
        .count();
    let hawk_pair = [0, 4, 7, 9][between_types.min(3)];
    let doubled_types = [bear, elk, salmon, hawk]
        .into_iter()
        .filter(|count| *count >= 2)
        .count();
    let fox_score = [0, 3, 5, 7][doubled_types.min(3)];
    BEAR_STANDALONE[bear as usize]
        + ELK_STANDALONE[elk as usize]
        + SALMON_STANDALONE[salmon as usize]
        + u16::from(hawk / 2) * hawk_pair
        + u16::from(fox) * fox_score
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
    scoring_cards: &'static str,
    token_count: usize,
    count_cap: u8,
    seed: u64,
    threads: usize,
    restarts_per_count: usize,
    iterations_per_restart: usize,
    elapsed_seconds: f64,
    candidates: Vec<CatalogCandidate>,
}

fn catalog_candidates(
    output: &Path,
    threads: usize,
    restarts: usize,
    iterations: usize,
    seed: u64,
) {
    let started = Instant::now();
    let vectors = Arc::new(enumerate_count_vectors(count_relaxation));
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
                    let (layout, score, states_evaluated) = anneal_fixed_counts(
                        counts,
                        restarts,
                        iterations,
                        count_seed,
                        score_layout,
                        count_relaxation,
                    );
                    assert_eq!(layout.counts(), counts);
                    assert!(layout.is_connected());
                    assert_eq!(
                        production_score(&layout, ScoringCards::CBDDB),
                        score.by_species
                    );
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
        schema: "cbddb-wildlife-candidates-v1",
        scoring_cards: "CBDDB",
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
    let temporary = output.with_extension("json.tmp");
    fs::write(&temporary, encoded).expect("write candidate catalog");
    fs::rename(&temporary, output).expect("publish candidate catalog atomically");
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
struct VerificationAssumptions {
    scoring_cards: String,
}

#[derive(Debug, Deserialize)]
struct VerificationCatalog {
    proof_complete: bool,
    allocation_count: usize,
    assumptions: VerificationAssumptions,
    results: Vec<VerificationResult>,
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

fn verify_catalog(path: &Path) {
    let encoded = fs::read_to_string(path).expect("read exact catalog");
    let catalog: VerificationCatalog =
        serde_json::from_str(&encoded).expect("parse exact catalog JSON");
    assert!(catalog.proof_complete, "catalog proof is incomplete");
    assert_eq!(catalog.assumptions.scoring_cards, "CBDDB");
    let vectors = enumerate_count_vectors(count_relaxation);
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
        assert_eq!(custom.by_species, result.score_breakdown);
        assert_eq!(custom.total(), result.optimum);
        assert_eq!(
            production_score(&layout, ScoringCards::CBDDB),
            result.score_breakdown,
            "result {index} production score"
        );
    }
    println!(
        "verified {} exact CBDDB catalog boards with the production scorer",
        catalog.results.len()
    );
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
    let output = args
        .windows(2)
        .find_map(|pair| {
            (pair[0] == "--catalog-candidates").then(|| Path::new(&pair[1]).to_path_buf())
        })
        .expect("use --catalog-candidates OUTPUT or --verify-catalog INPUT");
    let threads = parse_usize(
        &args,
        "--threads",
        std::thread::available_parallelism().map_or(1, usize::from),
    );
    let restarts = parse_usize(&args, "--restarts-per-count", 8);
    let iterations = parse_usize(&args, "--iterations-per-restart", 20_000);
    let seed = parse_u64(&args, "--seed", 0x5eed_cbdd_2026_0723);
    catalog_candidates(&output, threads, restarts, iterations, seed);
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
    fn count_vector_space_and_top_relaxation_are_pinned() {
        let vectors = enumerate_count_vectors(count_relaxation);
        assert_eq!(vectors.len(), 826);
        assert_eq!(vectors[0], ([0, 2, 6, 6, 6], 100));
        assert_eq!(vectors[1], ([2, 0, 6, 6, 6], 100));
    }

    #[test]
    fn custom_scorer_matches_all_card_patterns() {
        use Wildlife::{Bear as B, Elk as E, Fox as F, Hawk as H, Salmon as S};
        let board = layout(&[
            (0, 0, B),
            (3, 0, B),
            (4, 0, B),
            (0, 3, B),
            (1, 3, B),
            (0, 4, B),
            (6, 0, E),
            (7, 0, E),
            (6, 1, E),
            (7, 1, E),
            (0, 7, S),
            (1, 7, S),
            (2, 7, S),
            (0, 8, F),
            (1, 8, F),
            (2, 8, E),
            (0, 10, H),
            (4, 10, H),
            (1, 10, B),
            (3, 10, E),
        ]);
        assert_eq!(score_layout(&board).by_species, [20, 17, 6, 7, 6]);
    }

    #[test]
    fn line_of_sight_and_matching_are_exact() {
        assert_eq!(
            ray_between(HexCoord::new(0, 0), HexCoord::new(4, 0)).unwrap(),
            vec![
                HexCoord::new(1, 0),
                HexCoord::new(2, 0),
                HexCoord::new(3, 0)
            ]
        );
        assert!(ray_between(HexCoord::new(0, 0), HexCoord::new(2, 1)).is_none());
        assert_eq!(
            maximum_weight_matching(4, &[(0, 1, 9), (1, 2, 9), (2, 3, 7)]),
            16
        );
    }

    #[test]
    fn custom_scorer_matches_production_on_varied_connected_boards() {
        let cases = [
            [0, 2, 6, 6, 6],
            [6, 0, 6, 6, 2],
            [6, 6, 2, 0, 6],
            [4, 4, 4, 4, 4],
            [6, 4, 3, 2, 5],
        ];
        for (index, counts) in cases.into_iter().enumerate() {
            let (layout, custom, _) = anneal_fixed_counts(
                counts,
                1,
                500,
                20260723 + index as u64,
                score_layout,
                count_relaxation,
            );
            assert_eq!(layout.counts(), counts);
            assert!(layout.is_connected());
            assert_eq!(
                production_score(&layout, ScoringCards::CBDDB),
                custom.by_species,
                "count case {counts:?}"
            );
        }
    }
}
