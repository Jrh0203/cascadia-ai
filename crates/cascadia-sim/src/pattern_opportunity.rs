use std::collections::HashMap;

use cascadia_game::{Board, GameState, PublicGameState, ScoringCards, Wildlife, score_board};

use super::SimulationError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct WildlifeMarketState {
    pub(super) market: [u8; 5],
    pub(super) bag: [u8; 5],
}

pub(super) struct OpponentConditionedOpportunity {
    opponent_values: Vec<[f64; 5]>,
    replacement_kernel: ReplacementKernel,
    terminal_cache: HashMap<(WildlifeMarketState, u8), Vec<(WildlifeMarketState, f64)>>,
}

impl OpponentConditionedOpportunity {
    pub(super) fn new(game: &GameState, acting_seat: usize, cards: ScoringCards) -> Self {
        let player_count = game.boards().len();
        let opponent_values = (1..player_count)
            .map(|offset| {
                let opponent = (acting_seat + offset) % player_count;
                wildlife_marginal_gains(&game.boards()[opponent], cards)
            })
            .collect();
        Self {
            opponent_values,
            replacement_kernel: ReplacementKernel::default(),
            terminal_cache: HashMap::new(),
        }
    }

    pub(super) fn evaluate(
        &mut self,
        state: &PublicGameState,
        acting_seat: usize,
        cards: ScoringCards,
        future_market_draws: usize,
        future_turns: usize,
    ) -> Result<f64, SimulationError> {
        if future_turns == 0 || state.is_game_over() {
            return Ok(0.0);
        }

        let (market_state, missing) = public_wildlife_market_state(state)?;
        let terminal = if let Some(cached) = self.terminal_cache.get(&(market_state, missing)) {
            cached
        } else {
            let distribution = terminal_market_distribution(
                market_state,
                missing,
                &self.opponent_values,
                &mut self.replacement_kernel,
            )?;
            self.terminal_cache
                .insert((market_state, missing), distribution);
            self.terminal_cache
                .get(&(market_state, missing))
                .expect("terminal market distribution was just cached")
        };
        let actor_values = wildlife_continuation_values(
            &state.boards()[acting_seat],
            cards,
            state.unplaced_wildlife_counts(),
            future_market_draws,
            future_turns,
        );
        expected_actor_market_value(terminal, actor_values)
    }

    pub(super) fn evaluate_premium(
        &mut self,
        state: &PublicGameState,
        acting_seat: usize,
        cards: ScoringCards,
        future_market_draws: usize,
        future_turns: usize,
    ) -> Result<f64, SimulationError> {
        if future_turns == 0 || state.is_game_over() {
            return Ok(0.0);
        }

        let one_turn_values = wildlife_marginal_gains(&state.boards()[acting_seat], cards);
        let optimistic_one_turn = expected_max_without_replacement(
            state.unplaced_wildlife_counts(),
            one_turn_values,
            future_market_draws,
        );
        if future_turns == 1 {
            return Ok(optimistic_one_turn);
        }

        let two_turn_values = wildlife_continuation_values(
            &state.boards()[acting_seat],
            cards,
            state.unplaced_wildlife_counts(),
            future_market_draws,
            future_turns,
        );
        let (market_state, missing) = public_wildlife_market_state(state)?;
        let terminal = if let Some(cached) = self.terminal_cache.get(&(market_state, missing)) {
            cached
        } else {
            let distribution = terminal_market_distribution(
                market_state,
                missing,
                &self.opponent_values,
                &mut self.replacement_kernel,
            )?;
            self.terminal_cache
                .insert((market_state, missing), distribution);
            self.terminal_cache
                .get(&(market_state, missing))
                .expect("terminal market distribution was just cached")
        };
        let conditioned_one_turn = expected_actor_market_value(terminal, one_turn_values)?;
        let conditioned_two_turn = expected_actor_market_value(terminal, two_turn_values)?;
        let premium = conditioned_two_turn - conditioned_one_turn;
        if premium < -1e-12 {
            return Err(SimulationError::Strategy(format!(
                "conditioned commitment premium became negative: {premium}"
            )));
        }
        Ok(optimistic_one_turn + premium.max(0.0))
    }
}

fn public_wildlife_market_state(
    state: &PublicGameState,
) -> Result<(WildlifeMarketState, u8), SimulationError> {
    let mut market = [0u8; 5];
    for wildlife in state.market().wildlife.iter().flatten() {
        market[*wildlife as usize] += 1;
    }
    let mut bag = state.unplaced_wildlife_counts();
    for index in 0..bag.len() {
        bag[index] = bag[index].checked_sub(market[index]).ok_or_else(|| {
            SimulationError::Strategy(
                "public wildlife market exceeds inferred unplaced supply".to_owned(),
            )
        })?;
    }
    let missing = 4usize.saturating_sub(market.iter().map(|count| usize::from(*count)).sum());
    Ok((
        WildlifeMarketState { market, bag },
        u8::try_from(missing).expect("a wildlife market has four slots"),
    ))
}

fn wildlife_continuation_values(
    board: &Board,
    cards: ScoringCards,
    unplaced_counts: [u8; 5],
    draws: usize,
    future_turns: usize,
) -> [f64; 5] {
    if future_turns <= 1 {
        return wildlife_marginal_gains(board, cards);
    }

    let baseline = score_board(board, cards).base_total;
    std::array::from_fn(|index| {
        if unplaced_counts[index] == 0 {
            return 0.0;
        }
        let wildlife = Wildlife::ALL[index];
        board
            .wildlife_placements(wildlife)
            .into_iter()
            .filter_map(|coord| {
                let mut after = board.clone();
                after.place_wildlife(coord, wildlife).ok()?;
                let mut remaining = unplaced_counts;
                remaining[index] -= 1;
                let immediate = score_board(&after, cards)
                    .base_total
                    .saturating_sub(baseline);
                Some(
                    f64::from(immediate)
                        + future_wildlife_opportunity(
                            &after,
                            cards,
                            remaining,
                            draws,
                            future_turns - 1,
                        ),
                )
            })
            .fold(0.0, f64::max)
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ReplacementRequest {
    state: WildlifeMarketState,
    draws: u8,
    set_aside: [u8; 5],
}

#[derive(Default)]
pub(super) struct ReplacementKernel {
    memo: HashMap<ReplacementRequest, Vec<(WildlifeMarketState, f64)>>,
    stable_four_memo: HashMap<[u8; 5], Vec<([u8; 5], f64)>>,
}

impl ReplacementKernel {
    fn draw_and_stabilize(
        &mut self,
        state: WildlifeMarketState,
        draws: u8,
        set_aside: [u8; 5],
    ) -> Result<Vec<(WildlifeMarketState, f64)>, SimulationError> {
        let request = ReplacementRequest {
            state,
            draws,
            set_aside,
        };
        if let Some(cached) = self.memo.get(&request) {
            return Ok(cached.clone());
        }

        let distribution = if draws > 0 {
            let total = state.bag.iter().map(|count| u16::from(*count)).sum::<u16>();
            if total < u16::from(draws) {
                Vec::new()
            } else {
                let allocations = draw_allocations(state.bag, draws);
                let mut outcomes = HashMap::new();
                for (allocation, probability) in allocations {
                    let mut after = state;
                    for (index, drawn) in allocation.into_iter().enumerate() {
                        after.bag[index] -= drawn;
                        after.market[index] += drawn;
                    }
                    for (stable, branch_probability) in self.stabilize(after, set_aside)? {
                        *outcomes.entry(stable).or_insert(0.0) += probability * branch_probability;
                    }
                }
                sorted_distribution(outcomes)
            }
        } else {
            self.stabilize(state, set_aside)?
        };
        self.memo.insert(request, distribution.clone());
        Ok(distribution)
    }

    fn stabilize(
        &mut self,
        state: WildlifeMarketState,
        mut set_aside: [u8; 5],
    ) -> Result<Vec<(WildlifeMarketState, f64)>, SimulationError> {
        if let Some(index) = state.market.iter().position(|count| *count == 4) {
            set_aside[index] = set_aside[index].checked_add(4).ok_or_else(|| {
                SimulationError::Strategy(
                    "opponent market replacement set-aside overflowed".to_owned(),
                )
            })?;
            let final_markets = self.stable_four_draws(state.bag);
            let mut outcomes = Vec::with_capacity(final_markets.len());
            for (market, probability) in final_markets {
                let mut bag = state.bag;
                for index in 0..bag.len() {
                    bag[index] = bag[index]
                        .checked_sub(market[index])
                        .and_then(|remaining| remaining.checked_add(set_aside[index]))
                        .ok_or_else(|| {
                            SimulationError::Strategy(
                                "opponent market stabilization violated token conservation"
                                    .to_owned(),
                            )
                        })?;
                }
                outcomes.push((WildlifeMarketState { market, bag }, probability));
            }
            return Ok(outcomes);
        }

        let before = market_token_total(state) + set_aside.iter().sum::<u8>();
        let mut stable = state;
        for (bag_count, set_aside_count) in stable.bag.iter_mut().zip(set_aside) {
            *bag_count += set_aside_count;
        }
        debug_assert_eq!(market_token_total(stable), before);
        Ok(vec![(stable, 1.0)])
    }

    pub(super) fn stable_four_draws(&mut self, initial_bag: [u8; 5]) -> Vec<([u8; 5], f64)> {
        if let Some(cached) = self.stable_four_memo.get(&initial_bag) {
            return cached.clone();
        }

        let arithmetic = StableFourArithmetic::new(initial_bag);
        let mut outcomes = draw_allocations(initial_bag, 4)
            .into_iter()
            .filter(|(allocation, _)| !allocation.contains(&4))
            .map(|(allocation, _)| (allocation, arithmetic.probability(allocation)))
            .filter(|(_, probability)| *probability > 0.0)
            .collect::<Vec<_>>();
        outcomes.sort_by_key(|(market, _)| *market);
        self.stable_four_memo.insert(initial_bag, outcomes.clone());
        outcomes
    }
}

const MAX_REJECTION_COUNTS: usize = 26;

struct StableFourArithmetic {
    maximum_rejections: usize,
    rejection_scale: [f64; MAX_REJECTION_COUNTS],
    species_factors: [[Vec<f64>; 4]; 5],
}

impl StableFourArithmetic {
    fn new(initial_bag: [u8; 5]) -> Self {
        let total = initial_bag
            .iter()
            .map(|count| usize::from(*count))
            .sum::<usize>();
        let maximum_rejections = total.saturating_sub(4) / 4;
        let mut factorial = 1.0;
        let mut denominator = 1.0;
        let mut rejection_scale = [0.0; MAX_REJECTION_COUNTS];
        for (rejected, scale) in rejection_scale
            .iter_mut()
            .enumerate()
            .take(maximum_rejections + 1)
        {
            if rejected > 0 {
                factorial *= rejected as f64;
            }
            let remaining = total - 4 * rejected;
            denominator *= choose_small(
                u8::try_from(remaining).expect("wildlife supply fits in u8"),
                4,
            ) as f64;
            *scale = factorial / denominator;
        }

        let species_factors = std::array::from_fn(|species| {
            std::array::from_fn(|final_count| {
                let initial = initial_bag[species];
                let final_count =
                    u8::try_from(final_count).expect("stable market count fits in u8");
                if final_count > initial {
                    return Vec::new();
                }
                let maximum_species_rejections = usize::from((initial - final_count) / 4);
                let mut factor = Vec::with_capacity(maximum_species_rejections + 1);
                let mut mono_product = 1.0;
                let mut species_factorial = 1.0;
                for rejected in 0..=maximum_species_rejections {
                    let remaining =
                        initial - 4 * u8::try_from(rejected).expect("rejections fit in u8");
                    factor.push(
                        mono_product * choose_small(remaining, final_count) as f64
                            / species_factorial,
                    );
                    if rejected < maximum_species_rejections {
                        mono_product *= choose_small(remaining, 4) as f64;
                        species_factorial *= (rejected + 1) as f64;
                    }
                }
                factor
            })
        });
        Self {
            maximum_rejections,
            rejection_scale,
            species_factors,
        }
    }

    fn probability(&self, allocation: [u8; 5]) -> f64 {
        let mut coefficient = [0.0; MAX_REJECTION_COUNTS];
        coefficient[0] = 1.0;
        let mut coefficient_len = 1;
        for (species, final_count) in allocation.into_iter().enumerate() {
            let factor = &self.species_factors[species][usize::from(final_count)];
            let mut next = [0.0; MAX_REJECTION_COUNTS];
            for left_index in 0..coefficient_len {
                for (right_index, right_value) in factor.iter().enumerate() {
                    next[left_index + right_index] += coefficient[left_index] * right_value;
                }
            }
            coefficient_len += factor.len() - 1;
            coefficient = next;
        }
        coefficient
            .into_iter()
            .zip(self.rejection_scale)
            .take(self.maximum_rejections + 1)
            .map(|(value, scale)| value * scale)
            .sum()
    }
}

pub(super) fn draw_allocations(counts: [u8; 5], draws: u8) -> Vec<([u8; 5], f64)> {
    fn enumerate(
        counts: [u8; 5],
        index: usize,
        remaining: u8,
        allocation: &mut [u8; 5],
        numerator: u64,
        denominator: f64,
        outcomes: &mut Vec<([u8; 5], f64)>,
    ) {
        if index == counts.len() {
            if remaining == 0 {
                outcomes.push((*allocation, numerator as f64 / denominator));
            }
            return;
        }
        let maximum = counts[index].min(remaining);
        for taken in 0..=maximum {
            allocation[index] = taken;
            enumerate(
                counts,
                index + 1,
                remaining - taken,
                allocation,
                numerator * choose_small(counts[index], taken),
                denominator,
                outcomes,
            );
        }
        allocation[index] = 0;
    }

    let total = counts.iter().map(|count| u16::from(*count)).sum::<u16>();
    if total < u16::from(draws) {
        return Vec::new();
    }
    let denominator = choose_small(
        u8::try_from(total).expect("wildlife supply fits in u8"),
        draws,
    ) as f64;
    let mut outcomes = Vec::new();
    enumerate(counts, 0, draws, &mut [0; 5], 1, denominator, &mut outcomes);
    outcomes
}

fn choose_small(n: u8, k: u8) -> u64 {
    let k = k.min(n - k);
    (0..k).fold(1u64, |value, index| {
        value * u64::from(n - index) / u64::from(index + 1)
    })
}

pub(super) fn terminal_market_distribution(
    state: WildlifeMarketState,
    missing: u8,
    opponent_values: &[[f64; 5]],
    kernel: &mut ReplacementKernel,
) -> Result<Vec<(WildlifeMarketState, f64)>, SimulationError> {
    let mut distribution = apply_market_draw(vec![(state, 1.0)], missing, [0; 5], kernel)?;
    for turn in 0..=opponent_values.len() {
        distribution = prepare_market_distribution(distribution, kernel)?;
        if turn == opponent_values.len() {
            break;
        }

        let mut drafted = HashMap::new();
        for (market_state, state_probability) in distribution {
            for (index, choice_probability) in
                opponent_draft_choices(market_state, opponent_values[turn])?
            {
                let mut after = market_state;
                after.market[index] -= 1;
                *drafted.entry(after).or_insert(0.0) += state_probability * choice_probability;
            }
        }
        distribution = apply_market_draw(sorted_distribution(drafted), 1, [0; 5], kernel)?;
    }
    Ok(distribution)
}

fn prepare_market_distribution(
    distribution: Vec<(WildlifeMarketState, f64)>,
    kernel: &mut ReplacementKernel,
) -> Result<Vec<(WildlifeMarketState, f64)>, SimulationError> {
    let mut prepared = HashMap::new();
    for (state, state_probability) in distribution {
        if let Some(index) = state.market.iter().position(|count| *count == 3) {
            let mut after = state;
            after.market[index] -= 3;
            let mut set_aside = [0; 5];
            set_aside[index] = 3;
            for (stable, branch_probability) in kernel.draw_and_stabilize(after, 3, set_aside)? {
                *prepared.entry(stable).or_insert(0.0) += state_probability * branch_probability;
            }
        } else {
            *prepared.entry(state).or_insert(0.0) += state_probability;
        }
    }
    Ok(sorted_distribution(prepared))
}

fn apply_market_draw(
    distribution: Vec<(WildlifeMarketState, f64)>,
    draws: u8,
    set_aside: [u8; 5],
    kernel: &mut ReplacementKernel,
) -> Result<Vec<(WildlifeMarketState, f64)>, SimulationError> {
    let mut drawn = HashMap::new();
    for (state, state_probability) in distribution {
        for (stable, branch_probability) in kernel.draw_and_stabilize(state, draws, set_aside)? {
            *drawn.entry(stable).or_insert(0.0) += state_probability * branch_probability;
        }
    }
    Ok(sorted_distribution(drawn))
}

fn sorted_distribution(
    distribution: HashMap<WildlifeMarketState, f64>,
) -> Vec<(WildlifeMarketState, f64)> {
    let mut sorted = distribution.into_iter().collect::<Vec<_>>();
    sorted.sort_by_key(|(state, _)| *state);
    sorted
}

fn expected_actor_market_value(
    terminal: &[(WildlifeMarketState, f64)],
    actor_values: [f64; 5],
) -> Result<f64, SimulationError> {
    let mut weighted_value = 0.0;
    let mut total_probability = 0.0;
    for (state, probability) in terminal {
        let value = state
            .market
            .iter()
            .enumerate()
            .filter(|(_, count)| **count > 0)
            .map(|(index, _)| actor_values[index])
            .fold(0.0, f64::max);
        weighted_value += probability * value;
        total_probability += probability;
    }
    if total_probability == 0.0 {
        return Err(SimulationError::Strategy(
            "opponent market model found no valid replacement path".to_owned(),
        ));
    }
    Ok(weighted_value / total_probability)
}

#[cfg(test)]
pub(super) fn expected_market_value(
    state: WildlifeMarketState,
    missing: u8,
    opponent_values: &[[f64; 5]],
    actor_values: [f64; 5],
) -> Result<f64, SimulationError> {
    let terminal = terminal_market_distribution(
        state,
        missing,
        opponent_values,
        &mut ReplacementKernel::default(),
    )?;
    expected_actor_market_value(&terminal, actor_values)
}

pub(super) fn market_token_total(state: WildlifeMarketState) -> u8 {
    state.market.iter().sum::<u8>() + state.bag.iter().sum::<u8>()
}

pub(super) fn opponent_draft_choices(
    state: WildlifeMarketState,
    values: [f64; 5],
) -> Result<Vec<(usize, f64)>, SimulationError> {
    let best = state
        .market
        .iter()
        .enumerate()
        .filter(|(_, count)| **count > 0)
        .map(|(index, _)| values[index])
        .fold(f64::NEG_INFINITY, f64::max);
    if !best.is_finite() {
        return Err(SimulationError::Strategy(
            "opponent market model found no wildlife to draft".to_owned(),
        ));
    }
    let tied_tokens = state
        .market
        .iter()
        .enumerate()
        .filter(|(index, count)| **count > 0 && values[*index] == best)
        .map(|(_, count)| usize::from(*count))
        .sum::<usize>();
    Ok(values
        .into_iter()
        .enumerate()
        .filter(|(index, value)| state.market[*index] > 0 && *value == best)
        .map(|(index, _)| (index, f64::from(state.market[index]) / tied_tokens as f64))
        .collect())
}

pub fn future_market_opportunity(
    board: &Board,
    cards: ScoringCards,
    unplaced_counts: [u8; 5],
    draws: usize,
) -> f64 {
    future_wildlife_opportunity(board, cards, unplaced_counts, draws, 1)
}

pub fn future_wildlife_opportunity(
    board: &Board,
    cards: ScoringCards,
    unplaced_counts: [u8; 5],
    draws: usize,
    future_turns: usize,
) -> f64 {
    if future_turns == 0 {
        return 0.0;
    }
    if future_turns == 1 {
        let marginal_gains = wildlife_marginal_gains(board, cards);
        return expected_max_without_replacement(unplaced_counts, marginal_gains, draws);
    }

    let continuation_values =
        wildlife_continuation_values(board, cards, unplaced_counts, draws, future_turns);
    expected_max_without_replacement(unplaced_counts, continuation_values, draws)
}

pub fn wildlife_marginal_gains(board: &Board, cards: ScoringCards) -> [f64; 5] {
    let baseline = score_board(board, cards).base_total;
    std::array::from_fn(|index| {
        let wildlife = Wildlife::ALL[index];
        board
            .wildlife_placements(wildlife)
            .into_iter()
            .filter_map(|coord| {
                let mut after = board.clone();
                after.place_wildlife(coord, wildlife).ok()?;
                Some(
                    score_board(&after, cards)
                        .base_total
                        .saturating_sub(baseline),
                )
            })
            .max()
            .map_or(0.0, f64::from)
    })
}

pub(super) fn expected_max_without_replacement(
    counts: [u8; 5],
    values: [f64; 5],
    draws: usize,
) -> f64 {
    fn recurse(counts: &mut [u8; 5], values: [f64; 5], draws: usize, best: f64) -> f64 {
        let total = counts
            .iter()
            .map(|count| usize::from(*count))
            .sum::<usize>();
        if draws == 0 || total == 0 {
            return best;
        }
        let mut expectation = 0.0;
        for index in 0..counts.len() {
            if counts[index] == 0 {
                continue;
            }
            let probability = f64::from(counts[index]) / total as f64;
            counts[index] -= 1;
            expectation +=
                probability * recurse(counts, values, draws - 1, best.max(values[index]));
            counts[index] += 1;
        }
        expectation
    }

    recurse(&mut counts.clone(), values, draws, 0.0)
}
