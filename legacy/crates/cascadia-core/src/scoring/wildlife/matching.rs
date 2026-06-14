//! Maximum-weight matching on a small undirected graph.
//!
//! Used by Hawk D and Fox D scoring, where each hawk/fox can participate in at
//! most one scoring pair and we want the pairing that maximises total points.
//! Solved by bitmask DP over node subsets.

/// Compute the maximum total weight of a matching on `n` nodes (`n` ≤ 24)
/// given the list of (node_a, node_b, weight) edges. Each node appears in
/// at most one edge of the chosen matching.
pub fn max_weight_matching(n: usize, edges: &[(u8, u8, u16)]) -> u16 {
    debug_assert!(n <= 24, "matching: n={} exceeds bitmask capacity", n);
    if n < 2 || edges.is_empty() {
        return 0;
    }

    // Per-node adjacency: list of (other_node, weight) pairs.
    let mut nbrs: Vec<arrayvec::ArrayVec<(u8, u16), 24>> = vec![Default::default(); n];
    for &(a, b, w) in edges {
        let _ = nbrs[a as usize].try_push((b, w));
        let _ = nbrs[b as usize].try_push((a, w));
    }

    let size: usize = 1 << n;
    let mut dp: Vec<u16> = vec![0; size];
    // Iterate subsets in increasing-mask order so subproblems are ready.
    for state in 1..size {
        // Pick the lowest-set bit `i` to "decide" first.
        let i = (state as u32).trailing_zeros() as usize;
        let without_i = state & !(1 << i);
        // Option A: leave i unmatched.
        let mut best = dp[without_i];
        // Option B: pair i with each available neighbor j.
        for &(j, w) in &nbrs[i] {
            let jb = 1usize << j as usize;
            if (state & jb) == 0 {
                continue;
            }
            let after = without_i & !jb;
            let val = dp[after] + w;
            if val > best {
                best = val;
            }
        }
        dp[state] = best;
    }
    dp[size - 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty() {
        assert_eq!(max_weight_matching(0, &[]), 0);
        assert_eq!(max_weight_matching(4, &[]), 0);
    }

    #[test]
    fn single_edge() {
        assert_eq!(max_weight_matching(2, &[(0, 1, 7)]), 7);
    }

    #[test]
    fn picks_best_disjoint_pair() {
        // Triangle: edges all share nodes.
        let edges = &[(0, 1, 5), (1, 2, 7), (0, 2, 3)];
        assert_eq!(max_weight_matching(3, edges), 7);
    }

    #[test]
    fn two_disjoint_pairs() {
        let edges = &[(0, 1, 4), (2, 3, 6)];
        assert_eq!(max_weight_matching(4, edges), 10);
    }

    #[test]
    fn chain_of_four_picks_best_partition() {
        // 0-1 (w=5), 1-2 (w=10), 2-3 (w=5)
        // Either (0,1)+(2,3)=10 or (1,2)=10. Same here.
        let edges = &[(0, 1, 5), (1, 2, 10), (2, 3, 5)];
        assert_eq!(max_weight_matching(4, edges), 10);
    }

    #[test]
    fn six_node_clique() {
        // 0-1=10, 2-3=10, 4-5=10, plus crossing edges of 1.
        let edges = &[
            (0, 1, 10),
            (2, 3, 10),
            (4, 5, 10),
            (0, 2, 1),
            (0, 3, 1),
            (1, 4, 1),
        ];
        assert_eq!(max_weight_matching(6, edges), 30);
    }
}
