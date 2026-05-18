# MCTS with UCB + Tree Reuse — Design

## Why

Current MCE does NO state persistence across turns. Every turn rebuilds candidates and rolls out from scratch.  
UCB is now allowed (Apr 16, 2026). Tree reuse can amortize compute across turns. Together they change flat-uniform search into selective deepening.

**Realistic expectation**: +5-15% compute efficiency within a turn (from UCB vs halving), +5-20% cross-turn (from tree reuse). Compounding, maybe net +1-2 game score points. Not the 3-5x I originally suggested; cascadia's per-turn branching (market × candidates × opponent-sequences) works against classical tree-reuse.

## Structure

Single file `crates/cascadia-ai/src/mcts_tree.rs`, ~400 lines.

```rust
/// Persistent search context — one per AI player for the whole game.
pub struct MctsContext {
    /// Current root node. Contains children, visit stats.
    root: Option<Box<TreeNode>>,
    /// Zobrist-style hash of the game state at the root (for match-checking).
    root_hash: u64,
    /// Weights shared across the tree.
    net: Arc<NNUENetwork>,
    /// Total simulations this game (for diagnostics).
    total_sims: usize,
}

/// Tree node = decision point or chance outcome.
enum NodeKind {
    /// AI is about to move. Children = one per explored candidate move.
    /// Uses UCB1 for selection; progressive widening adds candidates as N grows.
    Decision {
        candidates: Vec<ScoredMove>,  // all possible moves in this market
        children: Vec<Box<TreeNode>>,  // one per expanded candidate; None slots lazy
        expansion_frontier: usize,     // next candidate index to expand
    },
    /// After a move, before next player moves. Chance = random market refill.
    /// Children = one per observed outcome (lazily added).
    Chance {
        outcomes: Vec<(MarketRefillKey, Box<TreeNode>)>,
    },
}

struct TreeNode {
    visits: u32,
    sum_rewards: f64,
    sum_sq_rewards: f64,  // for variance-aware UCB
    kind: NodeKind,
    depth: u16,           // plies from root; for depth-limit bailout
}
```

## Core Algorithm

```rust
impl MctsContext {
    /// Called from main game loop each turn. Finds best move for current state.
    pub fn best_move(&mut self, game: &GameState, sim_budget: usize, rng: &mut StdRng) -> ScoredMove {
        self.reuse_or_create_root(game);
        for _ in 0..sim_budget {
            self.simulate_one(game, rng);
        }
        self.select_best_root_child()
    }

    /// Simulate one traversal: selection → expansion → rollout → backprop.
    fn simulate_one(&mut self, game: &GameState, rng: &mut StdRng) {
        let mut path = vec![];
        let mut node = self.root.as_mut().unwrap();
        let mut gs = game.clone();

        // SELECTION + EXPANSION
        loop {
            match &mut node.kind {
                NodeKind::Decision { candidates, children, expansion_frontier } => {
                    // Progressive widening: expand new candidate if sqrt(N) > #children
                    if (node.visits as f32).sqrt() > *expansion_frontier as f32
                        && *expansion_frontier < candidates.len() {
                        let mv = candidates[*expansion_frontier];
                        *expansion_frontier += 1;
                        execute_scored_move(&mut gs, &mv);
                        // New leaf — rollout from here
                        let reward = self.rollout(gs, rng);
                        // ... add new child + backprop
                        return;
                    }
                    // Otherwise UCB1 selection
                    let ci = select_ucb(children, node.visits);
                    path.push(ci);
                    execute_scored_move(&mut gs, &candidates[ci]);
                    node = children[ci].as_mut().unwrap();
                }
                NodeKind::Chance { outcomes } => {
                    // Sample one: market refill (from bag) + opponents' moves
                    let outcome_key = simulate_next_state(&mut gs, rng);
                    // Find matching child or add new
                    match outcomes.iter().position(|(k, _)| *k == outcome_key) {
                        Some(i) => { node = outcomes[i].1.as_mut().unwrap(); }
                        None => {
                            outcomes.push((outcome_key, new_decision_node(&gs)));
                            node = outcomes.last_mut().unwrap().1.as_mut().unwrap();
                        }
                    }
                }
            }
            if node.depth >= MAX_DEPTH || gs.is_game_over() {
                break;
            }
        }
        let reward = self.leaf_eval(&gs);
        self.backprop(&path, reward);
    }

    /// UCB1 with variance-aware exploration (based on Rust's existing Thompson impl).
    fn select_ucb(children: &[Option<Box<TreeNode>>], parent_visits: u32) -> usize {
        let log_n = (parent_visits.max(1) as f64).ln();
        let c = 0.5;  // exploration constant; tune empirically (typical MCTS values 0.5-1.0)
        children.iter().enumerate().filter_map(|(i, c)| {
            c.as_ref().map(|n| {
                let mean = n.sum_rewards / n.visits.max(1) as f64;
                let explore = c * (log_n / n.visits.max(1) as f64).sqrt();
                (i, mean / 100.0 + explore)
            })
        }).max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap().0
    }

    /// Rebase root to the child corresponding to the actual move + actual state.
    /// This is the tree-reuse step: discard the rest of the tree, keep the subtree
    /// relevant to the move we committed + actual chance outcome.
    pub fn commit_move(&mut self, mv: &ScoredMove, resulting_state: &GameState) {
        // Navigate old_root → decision_child[mv] → chance_child with matching refill key
        // If no match (chance outcome never sampled), re-root to empty node.
        let new_root_hash = hash_game_state(resulting_state);
        if let Some(new_root) = self.try_descend(mv, new_root_hash) {
            self.root = Some(new_root);
            self.root_hash = new_root_hash;
        } else {
            self.root = Some(new_decision_node(resulting_state));
            self.root_hash = new_root_hash;
        }
    }
}
```

## Leaf Evaluation

- If `gs.is_game_over()`: use final `ScoreBreakdown::compute` total.
- Else: `actual_score + NNUE_remaining + tier_bonus` — matches existing MCE leaf.
- Optional: a short greedy rollout to terminal, average with NNUE leaf.

## Chance Node Handling (Cascadia-Specific)

After our candidate move, the state transitions through:
1. **Market refill** (random from bag): 1 tile + 1 wildlife drawn per emptied slot.
2. **Opponent turns** (3 players, each picks + market refills).

Two options:
- **Fine-grained chance**: separate chance node per (our-move → market-refill), then decision node for each opponent, chance again, etc. Deep trees.
- **Coarse-grained chance** (recommended for v1): single chance node per "our-move → end of opponents' turns." `MarketRefillKey` is a hash of (state after all opponents played). Simpler, fewer node types.

Opponents play greedy (or NNUE) deterministically given their market — so variability only comes from *market refills*, which are bag draws. Branching factor at each chance node ≈ (# ways the bag could refill) ≈ 5^k for k empty slots. We cap at `MAX_CHANCE_CHILDREN = 64` per node — if we see a new outcome beyond that, reuse the closest-hash existing child.

## Cross-Turn Tree Reuse

When the real game advances:
1. We committed move M.
2. Market actually refilled + opponents actually played → we're at state S_next.
3. Descend: old_root → decision_child[M] → chance_child with `MarketRefillKey(S_next)`.
4. If the chance_child exists and its state hash matches → promote to new root.
5. Otherwise → fresh empty decision node.

In practice, match rate will be low in Cascadia (specific opponent move sequences are rarely in our tree). Expected match rate: 5-15%. When it matches, we inherit the subtree's visits (amortized search across turns). When it doesn't, we lose nothing.

Additional saving regardless of match: the **value estimates of already-evaluated board states** bleed through the network's generalization (NNUE is the same).

## Integration Into CLI

Add tag `"mcts_tree"` in `pick_move_by_tag` (main.rs:381). Requires per-game persistent state, which means `MctsContext` must live in the game loop, not just per-call. Extend `simulate_game_inner` to instantiate a MctsContext once and pass it to `pick_move_by_tag` via a thread-local or arg.

## Validation Plan

1. Unit: simulate one fixed position, run mcts_tree with sim_budget=500, compare chosen move to brute-force MCE(2000) — they should agree ≥ 80% of the time.
2. Local: 20-game bench `mcts_tree` vs champion, seed=42. Wall-clock and score.
3. Modal HH: 100 games, `mcts_tree` vs `mce_wide_v1` vs `mce93` vs champion.

## Risks / Known Pitfalls

- **Memory**: tree can grow unboundedly. Cap chance-node children; prune deep decision nodes after commit.
- **Thread safety**: MctsContext is `&mut` from a single thread. Rollouts can still parallelize internally within `simulate_one`'s leaf evaluation if needed, but tree updates stay single-threaded.
- **NNUE eval cost**: leaf eval is ~30μs; at 2000 sims per turn × 20 turns = 40K evals/game = 1.2s of NNUE time — cheap.
- **Opponent model**: greedy opponents in the tree's chance node predictions may misestimate. Use NNUE for opponents if budget permits.

## Not doing in v1

- UCB-V (Auer et al.): variance-aware UCB. Maybe later.
- Virtual losses for parallel tree search.
- RAVE / AMAF for move-correlation priors.
- Learned policy priors.
