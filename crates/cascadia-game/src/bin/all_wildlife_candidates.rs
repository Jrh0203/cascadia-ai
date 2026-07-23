//! Heuristic incumbent generator for all 1,024 pure-wildlife scoring-card sets.

#[allow(dead_code)]
mod wildlife_solver_support;

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use serde::Serialize;
use wildlife_solver_support::{
    COUNT_CAP, Layout, SPECIES_COUNT, TOKEN_COUNT, WildlifeScore, anneal_any_counts,
    production_score_all, wildlife_name,
};

const RULESET_COUNT: usize = 4usize.pow(SPECIES_COUNT as u32);

fn ruleset(index: usize) -> ([usize; SPECIES_COUNT], String) {
    assert!(index < RULESET_COUNT);
    let mut value = index;
    let mut cards = [0; SPECIES_COUNT];
    for species in (0..SPECIES_COUNT).rev() {
        cards[species] = value % 4;
        value /= 4;
    }
    let text = cards
        .into_iter()
        .map(|card| char::from(b'A' + card as u8))
        .collect();
    (cards, text)
}

fn partition_upper(token_count: u8, scores: &[u16]) -> u16 {
    let mut best = vec![0; token_count as usize + 1];
    for used in 1..=token_count as usize {
        best[used] = best[used - 1];
        for size in 1..=used.min(scores.len() - 1) {
            best[used] = best[used].max(scores[size] + best[used - size]);
        }
    }
    best[token_count as usize]
}

fn bear_upper(count: u8, card: usize) -> u16 {
    match card {
        0 => [0, 0, 4, 4, 11, 11, 19][count as usize],
        1 => u16::from(count / 3) * 10,
        2 => [0, 2, 5, 8, 10, 13, 18][count as usize],
        3 => partition_upper(count, &[0, 0, 5, 8, 13]),
        _ => unreachable!(),
    }
}

fn elk_upper(count: u8, card: usize) -> u16 {
    match card {
        0 | 1 => partition_upper(count, &[0, 2, 5, 9, 13]),
        2 => [0, 2, 4, 7, 10, 14, 18][count as usize],
        3 => [0, 2, 5, 8, 12, 16, 21][count as usize],
        _ => unreachable!(),
    }
}

fn salmon_upper(count: u8, card: usize) -> u16 {
    match card {
        0 => partition_upper(count, &[0, 2, 5, 8, 12, 16, 20]),
        1 => partition_upper(count, &[0, 2, 4, 9, 11, 17, 17]),
        2 => partition_upper(count, &[0, 0, 0, 10, 12, 15, 15]),
        3 => {
            let mut best = vec![0; count as usize + 1];
            for used in 1..=count as usize {
                best[used] = best[used - 1];
                for size in 3..=used {
                    let score = size + (TOKEN_COUNT - count as usize).min(2 * size + 4);
                    best[used] = best[used].max(score as u16 + best[used - size]);
                }
            }
            best[count as usize]
        }
        _ => unreachable!(),
    }
}

fn bipartite_hex_edge_upper(left: u8, right: u8) -> u16 {
    if left == 0 || right == 0 {
        return 0;
    }
    let planar = if left + right == 2 {
        1
    } else {
        2 * u16::from(left + right) - 4
    };
    [
        u16::from(left) * u16::from(right),
        6 * u16::from(left),
        6 * u16::from(right),
        planar,
    ]
    .into_iter()
    .min()
    .unwrap()
}

fn fox_c_upper(foxes: u8, targets: [u8; 4]) -> u16 {
    let mut best = 0;
    for first in 0..=foxes {
        for second in 0..=foxes - first {
            for third in 0..=foxes - first - second {
                for fourth in 0..=foxes - first - second - third {
                    let assigned = [first, second, third, fourth];
                    let score = assigned
                        .into_iter()
                        .zip(targets)
                        .map(|(group, target)| bipartite_hex_edge_upper(group, target))
                        .sum();
                    best = best.max(score);
                }
            }
        }
    }
    best
}

fn count_upper(counts: [u8; SPECIES_COUNT], cards: [usize; SPECIES_COUNT]) -> u16 {
    let [bear, elk, salmon, hawk, fox] = counts;
    let mut total =
        bear_upper(bear, cards[0]) + elk_upper(elk, cards[1]) + salmon_upper(salmon, cards[2]);
    total += match cards[3] {
        0 => [0, 2, 5, 8, 11, 14, 18][hawk as usize],
        1 => [0, 0, 5, 9, 12, 16, 20][hawk as usize],
        // Only consecutive hawks on each of the three axial line families
        // see one another.  The tight cap-six edge maxima are 0,0,1,3,5,7,9.
        2 => 3 * [0, 0, 1, 3, 5, 7, 9][hawk as usize],
        3 => {
            let distinct = [bear, elk, salmon, fox]
                .into_iter()
                .filter(|count| *count > 0)
                .count();
            u16::from(hawk / 2) * [0, 4, 7, 9][distinct.min(3)]
        }
        _ => unreachable!(),
    };
    total += match cards[4] {
        0 => {
            let observed = [bear, elk, salmon, hawk]
                .into_iter()
                .filter(|count| *count > 0)
                .count()
                + usize::from(fox >= 2);
            u16::from(fox) * observed as u16
        }
        1 => {
            let doubled = [bear, elk, salmon, hawk]
                .into_iter()
                .filter(|count| *count >= 2)
                .count();
            u16::from(fox) * [0, 3, 5, 7][doubled.min(3)]
        }
        2 => fox_c_upper(fox, [bear, elk, salmon, hawk]),
        3 => {
            let doubled = [bear, elk, salmon, hawk]
                .into_iter()
                .filter(|count| *count >= 2)
                .count();
            u16::from(fox / 2) * [0, 5, 7, 9, 11][doubled]
        }
        _ => unreachable!(),
    };
    total
}

fn global_upper(cards: [usize; SPECIES_COUNT]) -> u16 {
    let mut best = 0;
    for bear in 0..=COUNT_CAP {
        for elk in 0..=COUNT_CAP {
            for salmon in 0..=COUNT_CAP {
                for hawk in 0..=COUNT_CAP {
                    for fox in 0..=COUNT_CAP {
                        let counts = [bear, elk, salmon, hawk, fox];
                        if counts.into_iter().map(usize::from).sum::<usize>() == TOKEN_COUNT {
                            best = best.max(count_upper(counts, cards));
                        }
                    }
                }
            }
        }
    }
    best
}

fn score_layout(layout: &Layout, cards: [usize; SPECIES_COUNT]) -> WildlifeScore {
    let all = production_score_all(layout);
    WildlifeScore {
        by_species: std::array::from_fn(|species| all[cards[species]][species]),
    }
}

#[derive(Serialize)]
struct CatalogToken {
    q: i8,
    r: i8,
    wildlife: &'static str,
}

#[derive(Serialize)]
struct Candidate {
    index: usize,
    ruleset: String,
    count_upper: u16,
    score: u16,
    score_breakdown: [u16; SPECIES_COUNT],
    counts: [u8; SPECIES_COUNT],
    upper_bound_matched: bool,
    states_evaluated: u64,
    tokens: Vec<CatalogToken>,
}

#[derive(Serialize)]
struct CandidateFile {
    schema: &'static str,
    token_count: usize,
    count_cap: u8,
    range_start: usize,
    range_end: usize,
    seed: u64,
    threads: usize,
    restarts_per_ruleset: usize,
    iterations_per_restart: usize,
    elapsed_seconds: f64,
    candidates: Vec<Candidate>,
}

#[derive(Clone, Copy)]
struct Config {
    start: usize,
    end: usize,
    threads: usize,
    restarts: usize,
    iterations: usize,
    seed: u64,
}

fn generate(output: &Path, config: Config) {
    assert!(config.start < config.end && config.end <= RULESET_COUNT);
    let started = Instant::now();
    let task_count = config.end - config.start;
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<Option<Candidate>>> =
        Mutex::new((0..task_count).map(|_| None).collect());
    let thread_count = config.threads.max(1).min(task_count);

    std::thread::scope(|scope| {
        for _ in 0..thread_count {
            let next = &next;
            let results = &results;
            scope.spawn(move || {
                loop {
                    let local_index = next.fetch_add(1, Ordering::Relaxed);
                    if local_index >= task_count {
                        break;
                    }
                    let index = config.start + local_index;
                    let (cards, ruleset) = ruleset(index);
                    let upper = global_upper(cards);
                    let candidate_seed =
                        config.seed ^ (index as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15);
                    let (layout, score, states_evaluated) = anneal_any_counts(
                        config.restarts,
                        config.iterations,
                        candidate_seed,
                        |layout| score_layout(layout, cards),
                        upper,
                    );
                    assert!(layout.is_connected());
                    assert!(layout.counts().into_iter().all(|count| count <= COUNT_CAP));
                    assert_eq!(score_layout(&layout, cards), score);
                    let candidate = Candidate {
                        index,
                        ruleset: ruleset.clone(),
                        count_upper: upper,
                        score: score.total(),
                        score_breakdown: score.by_species,
                        counts: layout.counts(),
                        upper_bound_matched: score.total() == upper,
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
                    results.lock().expect("catalog result lock")[local_index] = Some(candidate);
                    eprintln!(
                        "candidate {}/{} index={} ruleset={} score={}/{}",
                        local_index + 1,
                        task_count,
                        index,
                        ruleset,
                        score.total(),
                        upper
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
    let payload = CandidateFile {
        schema: "all-wildlife-candidates-v1",
        token_count: TOKEN_COUNT,
        count_cap: COUNT_CAP,
        range_start: config.start,
        range_end: config.end,
        seed: config.seed,
        threads: thread_count,
        restarts_per_ruleset: config.restarts,
        iterations_per_restart: config.iterations,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        candidates,
    };
    let encoded = serde_json::to_string_pretty(&payload).expect("candidate JSON serializes") + "\n";
    let temporary = output.with_extension("json.tmp");
    fs::write(&temporary, encoded).expect("write candidate catalog");
    fs::rename(&temporary, output).expect("publish candidate catalog atomically");
}

fn parse<T: std::str::FromStr>(args: &[String], index: usize, default: T, name: &str) -> T {
    args.get(index)
        .map(|value| value.parse().unwrap_or_else(|_| panic!("invalid {name}")))
        .unwrap_or(default)
}

fn main() {
    let args: Vec<_> = env::args().collect();
    let output = PathBuf::from(args.get(1).unwrap_or_else(|| {
        eprintln!(
            "usage: all_wildlife_candidates OUTPUT [START=0] [END=1024] \
                 [THREADS=1] [RESTARTS=12] [ITERATIONS=100000] [SEED=1]"
        );
        std::process::exit(2);
    }));
    let config = Config {
        start: parse(&args, 2, 0, "start"),
        end: parse(&args, 3, RULESET_COUNT, "end"),
        threads: parse(&args, 4, 1, "threads"),
        restarts: parse(&args, 5, 12, "restarts"),
        iterations: parse(&args, 6, 100_000, "iterations"),
        seed: parse(&args, 7, 1, "seed"),
    };
    generate(&output, config);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ruleset_indices_are_lexicographic() {
        assert_eq!(ruleset(0).1, "AAAAA");
        assert_eq!(ruleset(1).1, "AAAAB");
        assert_eq!(ruleset(4).1, "AAABA");
        assert_eq!(ruleset(RULESET_COUNT - 1).1, "DDDDD");
    }

    #[test]
    fn count_upper_matches_known_rulesets() {
        assert_eq!(global_upper(ruleset(0).0), 73);
        assert_eq!(global_upper([2, 1, 3, 3, 1]), 100);
        assert_eq!(global_upper(ruleset(RULESET_COUNT - 1).0), 87);
    }
}
