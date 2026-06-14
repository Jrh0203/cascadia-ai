// Quick standalone measurement using the public APIs.
use cascadia_ai::mce::default_greedy_mce_candidates;
use cascadia_core::{game::GameState, types::ScoringCards};
use rand::{rngs::StdRng, SeedableRng};

fn main() {
    let mut hist = [0usize; 21]; // bucket by 0-4, 5-9, ..., 95-99, 100+
    let mut max = 0;
    let mut total = 0;
    let mut over_24 = 0;
    let mut games = 0;
    for seed in 0..50u64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut g = GameState::new(4, ScoringCards::all_a(), &mut rng);
        games += 1;
        while !g.is_game_over() {
            let cands = default_greedy_mce_candidates(&g);
            let n = cands.len();
            total += 1;
            max = max.max(n);
            if n > 24 {
                over_24 += 1;
            }
            let b = (n / 5).min(20);
            hist[b] += 1;
            // Use the first candidate to advance (just to walk through game).
            if cands.is_empty() {
                break;
            }
            if !cascadia_ai::search::execute_scored_move(&mut g, &cands[0]) {
                break;
            }
        }
    }
    println!(
        "games: {}, decisions: {}, max_candidates: {}",
        games, total, max
    );
    println!(
        "decisions with >24 candidates: {} ({:.1}%)",
        over_24,
        100.0 * over_24 as f64 / total as f64
    );
    println!("histogram (5-wide buckets):");
    for (i, &c) in hist.iter().enumerate() {
        if c > 0 {
            let lo = i * 5;
            let hi = lo + 4;
            let last = i == hist.len() - 1;
            let label = if last {
                format!("{}+", lo)
            } else {
                format!("{:3}-{:3}", lo, hi)
            };
            println!("  {} : {:4}", label, c);
        }
    }
}
