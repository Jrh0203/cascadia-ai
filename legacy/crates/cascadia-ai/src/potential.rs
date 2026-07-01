use cascadia_core::board::Board;
use cascadia_core::hex::{HexCoord, ADJACENCY};
use cascadia_core::types::{ScoringCardVariant, ScoringCards, Wildlife};

#[derive(Clone)]
pub(crate) struct BoardPotentialContext {
    frontier: [bool; 441],
    habitat_neighbor_counts: [[u8; 5]; 441],
    habitat_contributions: [i16; 441],
    habitat_total: i32,
    empty_acceptance: [u16; 5],
    wildlife_contributions: [i32; 5],
    wildlife_total: i32,
    aaaaa: Option<AaaaaPotentialContext>,
}

#[derive(Clone)]
struct AaaaaPotentialContext {
    bear: BearAPotentialContext,
    salmon: SalmonAPotentialContext,
    hawk: HawkAPotentialContext,
    fox: FoxAPotentialContext,
}

#[derive(Clone)]
struct BearAPotentialContext {
    component_ids: [u8; 441],
    component_sizes: arrayvec::ArrayVec<u8, 24>,
    half_eligible: [bool; 441],
    penalized: [bool; 441],
    pair_count: u8,
    half_pairs: i32,
    penalty: i32,
    total: i32,
}

#[derive(Clone)]
struct SalmonAComponent {
    positions: arrayvec::ArrayVec<u16, 24>,
    contribution: i32,
}

#[derive(Clone)]
struct SalmonAPotentialContext {
    component_ids: [u8; 441],
    components: arrayvec::ArrayVec<SalmonAComponent, 24>,
    total: i32,
}

#[derive(Clone)]
struct HawkAPotentialContext {
    isolated: [bool; 441],
    safety_contributions: [i8; 441],
    safe_slots: [bool; 441],
    isolated_count: i32,
    safety_total: i32,
    safe_slot_count: i32,
    total: i32,
}

#[derive(Clone)]
struct FoxAPotentialContext {
    contributions: [i16; 441],
    total: i32,
}

impl BoardPotentialContext {
    pub(crate) fn new(board: &Board, cards: &ScoringCards, frontier: &[u16]) -> Self {
        let mut stored_frontier = [false; 441];
        let mut habitat_neighbor_counts = [[0u8; 5]; 441];
        let mut habitat_contributions = [0i16; 441];
        let mut habitat_total = 0;
        for &index in frontier {
            let counts = habitat_counts_around_cell(board, index as usize);
            let contribution = habitat_potential_from_counts(counts);
            stored_frontier[index as usize] = true;
            habitat_neighbor_counts[index as usize] = counts;
            habitat_contributions[index as usize] = contribution as i16;
            habitat_total += contribution;
        }
        let aaaaa = cards
            .cards
            .iter()
            .all(|&variant| variant == ScoringCardVariant::A)
            .then(|| AaaaaPotentialContext::new(board));
        let wildlife_contributions = if let Some(context) = &aaaaa {
            [
                context.bear.total,
                elk_potential(board),
                context.salmon.total,
                context.hawk.total,
                context.fox.total,
            ]
        } else {
            std::array::from_fn(|index| wildlife_potential(board, cards, Wildlife::ALL[index]))
        };

        Self {
            frontier: stored_frontier,
            habitat_neighbor_counts,
            habitat_contributions,
            habitat_total,
            empty_acceptance: empty_acceptance_counts(board),
            wildlife_total: wildlife_contributions.iter().sum(),
            wildlife_contributions,
            aaaaa,
        }
    }

    fn frontier_contribution(&self, index: u16) -> Option<i32> {
        self.frontier[index as usize].then_some(self.habitat_contributions[index as usize] as i32)
    }
}

impl AaaaaPotentialContext {
    fn new(board: &Board) -> Self {
        Self {
            bear: BearAPotentialContext::new(board),
            salmon: SalmonAPotentialContext::new(board),
            hawk: HawkAPotentialContext::new(board),
            fox: FoxAPotentialContext::new(board),
        }
    }
}

impl BearAPotentialContext {
    fn new(board: &Board) -> Self {
        let positions = &board.wildlife_positions[Wildlife::Bear as usize];
        let mut component_ids = [u8::MAX; 441];
        let mut component_sizes = arrayvec::ArrayVec::<u8, 24>::new();
        let mut half_eligible = [false; 441];
        let mut penalized = [false; 441];
        let mut pair_count = 0u8;
        let mut half_pairs = 0i32;
        let mut penalty = 0i32;

        for &position in positions {
            let index = position as usize;
            let (half, penalized_position) = bear_a_local_terms(board, index);
            half_eligible[index] = half;
            penalized[index] = penalized_position;
            half_pairs += i32::from(half);
            penalty -= 20 * i32::from(penalized_position);

            if component_ids[index] != u8::MAX {
                continue;
            }
            let component_id = component_sizes.len() as u8;
            let mut size = 0u8;
            let mut queue = arrayvec::ArrayVec::<u16, 24>::new();
            queue.push(position);
            component_ids[index] = component_id;
            while let Some(current) = queue.pop() {
                size += 1;
                for neighbor in ADJACENCY.neighbors_of(current as usize) {
                    if component_ids[neighbor] == u8::MAX
                        && board.grid.get(neighbor).placed_wildlife() == Some(Wildlife::Bear)
                    {
                        component_ids[neighbor] = component_id;
                        queue.push(neighbor as u16);
                    }
                }
            }
            pair_count += u8::from(size == 2);
            component_sizes.push(size);
        }

        let total = bear_a_total(pair_count, half_pairs, penalty);
        Self {
            component_ids,
            component_sizes,
            half_eligible,
            penalized,
            pair_count,
            half_pairs,
            penalty,
            total,
        }
    }

    fn after_single_move(
        &self,
        board: &Board,
        placed_tile_index: usize,
        placed_wildlife: Option<(usize, Wildlife)>,
    ) -> i32 {
        let mut pair_count = self.pair_count;
        let mut half_pairs = self.half_pairs;
        let mut penalty = self.penalty;
        let mut affected = arrayvec::ArrayVec::<u16, 12>::new();
        push_neighboring_wildlife(board, placed_tile_index, Wildlife::Bear, &mut affected);
        if let Some((wildlife_index, _)) = placed_wildlife {
            push_neighboring_wildlife(board, wildlife_index, Wildlife::Bear, &mut affected);
        }

        for &position in &affected {
            let index = position as usize;
            if placed_wildlife == Some((index, Wildlife::Bear)) {
                continue;
            }
            half_pairs -= i32::from(self.half_eligible[index]);
            penalty += 20 * i32::from(self.penalized[index]);
            let (half, penalized_position) = bear_a_local_terms(board, index);
            half_pairs += i32::from(half);
            penalty -= 20 * i32::from(penalized_position);
        }

        if let Some((wildlife_index, Wildlife::Bear)) = placed_wildlife {
            let mut adjacent_components = arrayvec::ArrayVec::<u8, 6>::new();
            let mut merged_size = 1u8;
            for neighbor in ADJACENCY.neighbors_of(wildlife_index) {
                let component_id = self.component_ids[neighbor];
                if component_id != u8::MAX && !adjacent_components.contains(&component_id) {
                    adjacent_components.push(component_id);
                    let size = self.component_sizes[component_id as usize];
                    merged_size += size;
                    pair_count -= u8::from(size == 2);
                }
            }
            pair_count += u8::from(merged_size == 2);

            let (half, penalized_position) = bear_a_local_terms(board, wildlife_index);
            half_pairs += i32::from(half);
            penalty -= 20 * i32::from(penalized_position);
        }

        bear_a_total(pair_count, half_pairs, penalty)
    }
}

impl SalmonAPotentialContext {
    fn new(board: &Board) -> Self {
        let positions = &board.wildlife_positions[Wildlife::Salmon as usize];
        let mut component_ids = [u8::MAX; 441];
        let mut components = arrayvec::ArrayVec::<SalmonAComponent, 24>::new();
        let mut total = 0i32;

        for &position in positions {
            let index = position as usize;
            if component_ids[index] != u8::MAX {
                continue;
            }
            let component_id = components.len() as u8;
            let mut component = arrayvec::ArrayVec::<u16, 24>::new();
            let mut queue = arrayvec::ArrayVec::<u16, 24>::new();
            queue.push(position);
            component_ids[index] = component_id;
            while let Some(current) = queue.pop() {
                component.push(current);
                for neighbor in ADJACENCY.neighbors_of(current as usize) {
                    if component_ids[neighbor] == u8::MAX
                        && board.grid.get(neighbor).placed_wildlife() == Some(Wildlife::Salmon)
                    {
                        component_ids[neighbor] = component_id;
                        queue.push(neighbor as u16);
                    }
                }
            }
            let contribution = salmon_a_component_potential(board, &component);
            total += contribution;
            components.push(SalmonAComponent {
                positions: component,
                contribution,
            });
        }

        Self {
            component_ids,
            components,
            total,
        }
    }

    fn after_single_move(
        &self,
        board: &Board,
        placed_tile_index: usize,
        placed_wildlife: Option<(usize, Wildlife)>,
    ) -> i32 {
        let mut affected_components = arrayvec::ArrayVec::<u8, 12>::new();
        self.push_adjacent_components(placed_tile_index, &mut affected_components);
        if let Some((wildlife_index, _)) = placed_wildlife {
            self.push_adjacent_components(wildlife_index, &mut affected_components);
        }

        let mut total = self.total;
        for &component_id in &affected_components {
            total -= self.components[component_id as usize].contribution;
        }

        let mut merged_components = arrayvec::ArrayVec::<u8, 6>::new();
        if let Some((wildlife_index, Wildlife::Salmon)) = placed_wildlife {
            self.push_adjacent_components(wildlife_index, &mut merged_components);
            let mut merged_positions = arrayvec::ArrayVec::<u16, 24>::new();
            for &component_id in &merged_components {
                for &position in &self.components[component_id as usize].positions {
                    merged_positions.push(position);
                }
            }
            merged_positions.push(wildlife_index as u16);
            total += salmon_a_component_potential(board, &merged_positions);
        }

        for &component_id in &affected_components {
            if !merged_components.contains(&component_id) {
                let component = &self.components[component_id as usize];
                total += salmon_a_component_potential(board, &component.positions);
            }
        }
        total
    }

    fn push_adjacent_components<const N: usize>(
        &self,
        index: usize,
        components: &mut arrayvec::ArrayVec<u8, N>,
    ) {
        for neighbor in ADJACENCY.neighbors_of(index) {
            let component_id = self.component_ids[neighbor];
            if component_id != u8::MAX && !components.contains(&component_id) {
                components.push(component_id);
            }
        }
    }
}

impl HawkAPotentialContext {
    fn new(board: &Board) -> Self {
        let mut isolated = [false; 441];
        let mut safety_contributions = [0i8; 441];
        let mut isolated_count = 0i32;
        let mut safety_total = 0i32;
        for &position in &board.wildlife_positions[Wildlife::Hawk as usize] {
            let index = position as usize;
            let (is_isolated, contribution) = hawk_a_local_terms(board, index);
            isolated[index] = is_isolated;
            safety_contributions[index] = contribution as i8;
            isolated_count += i32::from(is_isolated);
            safety_total += contribution;
        }

        let mut safe_slots = [false; 441];
        let mut safe_slot_count = 0i32;
        for &position in &board.placed_tiles {
            let index = position as usize;
            let safe = hawk_a_safe_slot(board, index);
            safe_slots[index] = safe;
            safe_slot_count += i32::from(safe);
        }

        let total = hawk_a_total(isolated_count, safety_total, safe_slot_count);
        Self {
            isolated,
            safety_contributions,
            safe_slots,
            isolated_count,
            safety_total,
            safe_slot_count,
            total,
        }
    }

    fn after_single_move(
        &self,
        board: &Board,
        placed_tile_index: usize,
        placed_wildlife: Option<(usize, Wildlife)>,
    ) -> i32 {
        let mut isolated_count = self.isolated_count;
        let mut safety_total = self.safety_total;
        let mut affected_hawks = arrayvec::ArrayVec::<u16, 12>::new();
        push_neighboring_wildlife(
            board,
            placed_tile_index,
            Wildlife::Hawk,
            &mut affected_hawks,
        );
        if let Some((wildlife_index, _)) = placed_wildlife {
            push_neighboring_wildlife(board, wildlife_index, Wildlife::Hawk, &mut affected_hawks);
        }

        for &position in &affected_hawks {
            let index = position as usize;
            if placed_wildlife == Some((index, Wildlife::Hawk)) {
                continue;
            }
            isolated_count -= i32::from(self.isolated[index]);
            safety_total -= i32::from(self.safety_contributions[index]);
            let (isolated, contribution) = hawk_a_local_terms(board, index);
            isolated_count += i32::from(isolated);
            safety_total += contribution;
        }
        if let Some((wildlife_index, Wildlife::Hawk)) = placed_wildlife {
            let (isolated, contribution) = hawk_a_local_terms(board, wildlife_index);
            isolated_count += i32::from(isolated);
            safety_total += contribution;
        }

        let mut affected_slots = arrayvec::ArrayVec::<u16, 8>::new();
        push_unique_u16(&mut affected_slots, placed_tile_index as u16);
        if let Some((wildlife_index, wildlife)) = placed_wildlife {
            push_unique_u16(&mut affected_slots, wildlife_index as u16);
            if wildlife == Wildlife::Hawk {
                for neighbor in ADJACENCY.neighbors_of(wildlife_index) {
                    push_unique_u16(&mut affected_slots, neighbor as u16);
                }
            }
        }
        let mut safe_slot_count = self.safe_slot_count;
        for &position in &affected_slots {
            let index = position as usize;
            safe_slot_count -= i32::from(self.safe_slots[index]);
            safe_slot_count += i32::from(hawk_a_safe_slot(board, index));
        }

        hawk_a_total(isolated_count, safety_total, safe_slot_count)
    }
}

impl FoxAPotentialContext {
    fn new(board: &Board) -> Self {
        let mut contributions = [0i16; 441];
        let mut total = 0i32;
        for &position in &board.wildlife_positions[Wildlife::Fox as usize] {
            let index = position as usize;
            let contribution = fox_a_position_potential(board, index);
            contributions[index] = contribution as i16;
            total += contribution;
        }
        Self {
            contributions,
            total,
        }
    }

    fn after_single_move(
        &self,
        board: &Board,
        placed_tile_index: usize,
        placed_wildlife: Option<(usize, Wildlife)>,
    ) -> i32 {
        let mut total = self.total;
        let mut affected_foxes = arrayvec::ArrayVec::<u16, 12>::new();
        push_neighboring_wildlife(board, placed_tile_index, Wildlife::Fox, &mut affected_foxes);
        if let Some((wildlife_index, _)) = placed_wildlife {
            push_neighboring_wildlife(board, wildlife_index, Wildlife::Fox, &mut affected_foxes);
        }
        for &position in &affected_foxes {
            let index = position as usize;
            if placed_wildlife == Some((index, Wildlife::Fox)) {
                continue;
            }
            total -= i32::from(self.contributions[index]);
            total += fox_a_position_potential(board, index);
        }
        if let Some((wildlife_index, Wildlife::Fox)) = placed_wildlife {
            total += fox_a_position_potential(board, wildlife_index);
        }
        total
    }
}

fn push_neighboring_wildlife<const N: usize>(
    board: &Board,
    index: usize,
    wildlife: Wildlife,
    positions: &mut arrayvec::ArrayVec<u16, N>,
) {
    for neighbor in ADJACENCY.neighbors_of(index) {
        if board.grid.get(neighbor).placed_wildlife() == Some(wildlife) {
            push_unique_u16(positions, neighbor as u16);
        }
    }
}

fn push_unique_u16<const N: usize>(values: &mut arrayvec::ArrayVec<u16, N>, value: u16) {
    if !values.contains(&value) {
        values.push(value);
    }
}

fn bear_a_local_terms(board: &Board, index: usize) -> (bool, bool) {
    let bear_neighbors = ADJACENCY
        .neighbors_of(index)
        .filter(|&neighbor| board.grid.get(neighbor).placed_wildlife() == Some(Wildlife::Bear))
        .count();
    let half_eligible = bear_neighbors == 0
        && ADJACENCY.neighbors_of(index).any(|neighbor| {
            let cell = board.grid.get(neighbor);
            cell.is_present() && cell.can_place_wildlife(Wildlife::Bear)
        });
    (half_eligible, bear_neighbors >= 2)
}

fn bear_a_total(pair_count: u8, half_pairs: i32, penalty: i32) -> i32 {
    let next_pair_marginal = match pair_count {
        0 => 40,
        1 => 70,
        _ => 80,
    };
    (half_pairs / 2).min(2) * next_pair_marginal / 2 + penalty
}

fn salmon_a_component_potential(board: &Board, component: &[u16]) -> i32 {
    let valid = component.iter().all(|&position| {
        ADJACENCY
            .neighbors_of(position as usize)
            .filter(|&neighbor| {
                board.grid.get(neighbor).placed_wildlife() == Some(Wildlife::Salmon)
            })
            .count()
            <= 2
    });
    if !valid {
        return -30;
    }

    let run_len = component.len() as u16;
    let extendable = component.iter().any(|&position| {
        let salmon_neighbors = ADJACENCY
            .neighbors_of(position as usize)
            .filter(|&neighbor| {
                board.grid.get(neighbor).placed_wildlife() == Some(Wildlife::Salmon)
            })
            .count();
        salmon_neighbors <= 1
            && ADJACENCY.neighbors_of(position as usize).any(|neighbor| {
                let cell = board.grid.get(neighbor);
                cell.is_present()
                    && cell.can_place_wildlife(Wildlife::Salmon)
                    && cell.placed_wildlife() != Some(Wildlife::Salmon)
            })
    });
    if !extendable {
        return 0;
    }
    match run_len {
        1 => 10,
        2 => 15,
        3 | 4 => 20,
        5 => 25,
        6 => 30,
        _ => 0,
    }
}

fn hawk_a_local_terms(board: &Board, index: usize) -> (bool, i32) {
    let isolated = !ADJACENCY
        .neighbors_of(index)
        .any(|neighbor| board.grid.get(neighbor).placed_wildlife() == Some(Wildlife::Hawk));
    if !isolated {
        return (false, 0);
    }
    let danger_slots = ADJACENCY.neighbors_of(index).any(|neighbor| {
        let cell = board.grid.get(neighbor);
        cell.is_present() && cell.can_place_wildlife(Wildlife::Hawk)
    });
    (true, if danger_slots { 5 } else { 10 })
}

fn hawk_a_safe_slot(board: &Board, index: usize) -> bool {
    board.grid.get(index).can_place_wildlife(Wildlife::Hawk)
        && !ADJACENCY
            .neighbors_of(index)
            .any(|neighbor| board.grid.get(neighbor).placed_wildlife() == Some(Wildlife::Hawk))
}

fn hawk_a_total(isolated_count: i32, safety_total: i32, safe_slot_count: i32) -> i32 {
    let next_marginal = match isolated_count {
        0 => 20,
        1..=4 => 30,
        5..=6 => 40,
        _ => 60,
    };
    safety_total + safe_slot_count.min(2) * next_marginal / 2
}

fn fox_a_position_potential(board: &Board, index: usize) -> i32 {
    let mut seen_mask = 0u8;
    let mut empty_slots = 0u8;
    for neighbor in ADJACENCY.neighbors_of(index) {
        let cell = board.grid.get(neighbor);
        if let Some(wildlife) = cell.placed_wildlife() {
            seen_mask |= 1 << wildlife as u8;
        } else if cell.is_present() && cell.allowed_wildlife().count() > 0 {
            empty_slots += 1;
        }
    }
    let current_types = seen_mask.count_ones() as i32;
    (5 - current_types).min(empty_slots as i32) * 5
}

/// Compute a potential bonus estimating future scoring opportunities.
/// Returns a value in "fractional points" (scaled by 10 to avoid floats).
/// Per-animal heuristics dispatch to the active scoring-card variant.
pub fn board_potential(board: &Board, cards: &ScoringCards) -> i32 {
    let frontier = board.frontier();
    board_potential_with_frontier(board, cards, &frontier)
}

/// Compute board potential when the caller already has the exact frontier.
///
/// Candidate generation places one tile at a time and can update its existing
/// frontier in O(frontier + 6), avoiding a full scan of every placed tile for
/// every candidate while preserving the same frontier order and score.
pub(crate) fn board_potential_with_frontier(
    board: &Board,
    cards: &ScoringCards,
    frontier: &[u16],
) -> i32 {
    let mut potential: i32 = 0;

    potential += habitat_potential(board, frontier);
    potential += bear_potential_dispatch(board, cards.variant_for(Wildlife::Bear));
    potential += elk_potential_dispatch(board, cards.variant_for(Wildlife::Elk));
    potential += salmon_potential_dispatch(board, cards.variant_for(Wildlife::Salmon));
    potential += hawk_potential_dispatch(board, cards.variant_for(Wildlife::Hawk));
    potential += fox_potential_dispatch(board, cards.variant_for(Wildlife::Fox));
    potential += empty_slot_potential(board);

    potential
}

/// Compute the same potential as `board_potential` after one tile and at most
/// one wildlife placement, reusing the unchanged frontier and empty-slot work
/// from the parent board.
pub(crate) fn board_potential_after_single_move(
    board: &Board,
    cards: &ScoringCards,
    context: &BoardPotentialContext,
    placed_tile_index: usize,
    placed_wildlife: Option<(usize, Wildlife)>,
) -> i32 {
    if full_potential_recompute_enabled() {
        return board_potential(board, cards);
    }

    let mut habitat = context.habitat_total;
    if let Some(removed) = context.frontier_contribution(placed_tile_index as u16) {
        habitat -= removed;
    }

    let placed_cell = board.grid.get(placed_tile_index);
    for neighbor in ADJACENCY.neighbors_of(placed_tile_index) {
        let neighbor = neighbor as u16;
        if let Some(before) = context.frontier_contribution(neighbor) {
            let mut counts = context.habitat_neighbor_counts[neighbor as usize];
            add_cell_terrains_to_habitat_counts(&mut counts, placed_cell);
            habitat += habitat_potential_from_counts(counts) - before;
        } else if !board.grid.get(neighbor as usize).is_present() {
            let mut counts = [0u8; 5];
            add_cell_terrains_to_habitat_counts(&mut counts, placed_cell);
            habitat += habitat_potential_from_counts(counts);
        }
    }

    let mut affected_wildlife = placed_cell.allowed_wildlife().0;
    let mut acceptance = context.empty_acceptance;
    for wildlife in Wildlife::ALL {
        if placed_cell.allowed_wildlife().contains(wildlife) {
            acceptance[wildlife as usize] += 1;
        }
    }
    if let Some((wildlife_index, wildlife)) = placed_wildlife {
        let occupied = board.grid.get(wildlife_index);
        affected_wildlife |= occupied.allowed_wildlife().0 | (1 << wildlife as u8);
        for wildlife in Wildlife::ALL {
            if occupied.allowed_wildlife().contains(wildlife) {
                acceptance[wildlife as usize] = acceptance[wildlife as usize].saturating_sub(1);
            }
        }
    }

    // Fox setup values inspect all adjacent wildlife and open neighboring
    // slots. Salmon D and Hawk D also inspect non-matching wildlife. Recompute
    // these cross-species terms whenever the board gains a tile or token.
    affected_wildlife |= 1 << Wildlife::Fox as u8;
    if cards.variant_for(Wildlife::Salmon) == ScoringCardVariant::D {
        affected_wildlife |= 1 << Wildlife::Salmon as u8;
    }
    if cards.variant_for(Wildlife::Hawk) == ScoringCardVariant::D {
        affected_wildlife |= 1 << Wildlife::Hawk as u8;
    }

    let mut wildlife_total = context.wildlife_total;
    if let Some(aaaaa) = &context.aaaaa {
        if affected_wildlife & (1 << Wildlife::Bear as u8) != 0 {
            wildlife_total -= context.wildlife_contributions[Wildlife::Bear as usize];
            wildlife_total +=
                aaaaa
                    .bear
                    .after_single_move(board, placed_tile_index, placed_wildlife);
        }
        if affected_wildlife & (1 << Wildlife::Elk as u8) != 0 {
            wildlife_total -= context.wildlife_contributions[Wildlife::Elk as usize];
            wildlife_total += elk_potential(board);
        }
        if affected_wildlife & (1 << Wildlife::Salmon as u8) != 0 {
            wildlife_total -= context.wildlife_contributions[Wildlife::Salmon as usize];
            wildlife_total +=
                aaaaa
                    .salmon
                    .after_single_move(board, placed_tile_index, placed_wildlife);
        }
        if affected_wildlife & (1 << Wildlife::Hawk as u8) != 0 {
            wildlife_total -= context.wildlife_contributions[Wildlife::Hawk as usize];
            wildlife_total +=
                aaaaa
                    .hawk
                    .after_single_move(board, placed_tile_index, placed_wildlife);
        }
        wildlife_total -= context.wildlife_contributions[Wildlife::Fox as usize];
        wildlife_total += aaaaa
            .fox
            .after_single_move(board, placed_tile_index, placed_wildlife);
    } else {
        for wildlife in Wildlife::ALL {
            if affected_wildlife & (1 << wildlife as u8) != 0 {
                wildlife_total -= context.wildlife_contributions[wildlife as usize];
                wildlife_total += wildlife_potential(board, cards, wildlife);
            }
        }
    }

    let result = habitat + wildlife_total + empty_slot_potential_from_counts(acceptance);
    #[cfg(debug_assertions)]
    if std::env::var_os("CASCADIA_POTENTIAL_DIAGNOSTICS").is_some() {
        let exact = board_potential(board, cards);
        if result != exact {
            let exact_wildlife: [i32; 5] =
                std::array::from_fn(|index| wildlife_potential(board, cards, Wildlife::ALL[index]));
            let fast_wildlife = context.aaaaa.as_ref().map(|aaaaa| {
                [
                    aaaaa
                        .bear
                        .after_single_move(board, placed_tile_index, placed_wildlife),
                    elk_potential(board),
                    aaaaa
                        .salmon
                        .after_single_move(board, placed_tile_index, placed_wildlife),
                    aaaaa
                        .hawk
                        .after_single_move(board, placed_tile_index, placed_wildlife),
                    aaaaa
                        .fox
                        .after_single_move(board, placed_tile_index, placed_wildlife),
                ]
            });
            eprintln!(
                "potential mismatch result={result} exact={exact} habitat={habitat} empty={} base_wildlife={:?} exact_wildlife={exact_wildlife:?} fast_wildlife={fast_wildlife:?} tile={placed_tile_index} wildlife={placed_wildlife:?}",
                empty_slot_potential_from_counts(acceptance),
                context.wildlife_contributions,
            );
        }
    }
    result
}

fn full_potential_recompute_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("LEGACY_TEACHER_POTENTIAL_FULL_RECOMPUTE")
            .ok()
            .is_some_and(|value| !value.is_empty() && value != "0")
    })
}

#[inline]
fn wildlife_potential(board: &Board, cards: &ScoringCards, wildlife: Wildlife) -> i32 {
    match wildlife {
        Wildlife::Bear => bear_potential_dispatch(board, cards.variant_for(wildlife)),
        Wildlife::Elk => elk_potential_dispatch(board, cards.variant_for(wildlife)),
        Wildlife::Salmon => salmon_potential_dispatch(board, cards.variant_for(wildlife)),
        Wildlife::Hawk => hawk_potential_dispatch(board, cards.variant_for(wildlife)),
        Wildlife::Fox => fox_potential_dispatch(board, cards.variant_for(wildlife)),
    }
}

#[inline]
fn bear_potential_dispatch(board: &Board, v: ScoringCardVariant) -> i32 {
    match v {
        ScoringCardVariant::A => bear_potential(board),
        ScoringCardVariant::C => bear_potential_c(board),
        // Bear B (groups of exactly 3) and D (sizes 2/3/4) have different sweet
        // spots. For now use Card A's "extend toward small groups" heuristic as
        // a rough proxy — it's directionally correct (more bears → more points).
        _ => bear_potential(board),
    }
}

#[inline]
fn elk_potential_dispatch(board: &Board, v: ScoringCardVariant) -> i32 {
    match v {
        ScoringCardVariant::A => elk_potential(board),
        ScoringCardVariant::B => elk_potential_b(board),
        // Elk C (any contiguous group, super-linear) and D (rings around any hex
        // center) prefer dense clusters too — Card B is the best single proxy.
        ScoringCardVariant::C | ScoringCardVariant::D => elk_potential_b(board),
    }
}

#[inline]
fn salmon_potential_dispatch(board: &Board, v: ScoringCardVariant) -> i32 {
    match v {
        ScoringCardVariant::A => salmon_potential(board),
        ScoringCardVariant::D => salmon_potential_d(board),
        // Salmon B/C use the same chain rule as A, just with different score
        // tables — A's "extend the run" heuristic is directionally correct.
        _ => salmon_potential(board),
    }
}

#[inline]
fn hawk_potential_dispatch(board: &Board, v: ScoringCardVariant) -> i32 {
    match v {
        ScoringCardVariant::A => hawk_potential(board),
        ScoringCardVariant::D => hawk_potential_d(board),
        // Hawk B/C also reward LOS — Card A's "isolate" heuristic is the
        // OPPOSITE of what they want, but we don't have variant-specific
        // potentials yet. Fall back to A and accept the mismatch.
        _ => hawk_potential(board),
    }
}

#[inline]
fn fox_potential_dispatch(board: &Board, v: ScoringCardVariant) -> i32 {
    match v {
        ScoringCardVariant::A => fox_potential(board),
        ScoringCardVariant::B => fox_potential_b(board),
        // Fox C (single-type max count) and D (pair scoring) need their own
        // heuristics; B's "encourage same-type adjacency" is closer to C/D
        // than A's "encourage diversity".
        ScoringCardVariant::C | ScoringCardVariant::D => fox_potential_b(board),
    }
}

/// Habitat potential: bonus for frontier cells adjacent to same-terrain tiles,
/// especially when they could bridge separate groups.
fn habitat_potential(board: &Board, frontier: &[u16]) -> i32 {
    frontier
        .iter()
        .map(|&index| habitat_cell_potential(board, index as usize))
        .sum()
}

fn habitat_cell_potential(board: &Board, index: usize) -> i32 {
    habitat_potential_from_counts(habitat_counts_around_cell(board, index))
}

fn habitat_counts_around_cell(board: &Board, index: usize) -> [u8; 5] {
    let mut counts = [0u8; 5];
    for neighbor in ADJACENCY.neighbors_of(index) {
        add_cell_terrains_to_habitat_counts(&mut counts, board.grid.get(neighbor));
    }
    counts
}

fn add_cell_terrains_to_habitat_counts(counts: &mut [u8; 5], cell: cascadia_core::types::Cell) {
    if let Some(primary) = cell.primary_terrain() {
        counts[primary as usize] += 1;
        if let Some(secondary) = cell.secondary_terrain() {
            if secondary != primary {
                counts[secondary as usize] += 1;
            }
        }
    }
}

fn habitat_potential_from_counts(counts: [u8; 5]) -> i32 {
    counts
        .into_iter()
        .map(|count| {
            if count >= 2 {
                15
            } else if count == 1 {
                2
            } else {
                0
            }
        })
        .sum()
}

/// Bear potential (Card A): reward setups toward completing bear pairs.
/// Uses actual marginal values: 0→1 pair = +4, 1→2 = +7, 2→3 = +8, 3→4+ = +8.
fn bear_potential(board: &Board) -> i32 {
    let positions = &board.wildlife_positions[Wildlife::Bear as usize];
    let adj = &*ADJACENCY;

    // Count current pairs using the same algorithm as scoring
    let current_pairs = count_bear_pairs(board);

    // Marginal value of next pair (in tenths of a point)
    let next_pair_marginal: i32 = match current_pairs {
        0 => 40, // +4 points
        1 => 70, // +7 points
        2 => 80, // +8 points
        _ => 80, // +8 points
    };

    let mut half_pairs = 0i32;
    let mut penalty = 0i32;

    for &pos in positions.iter() {
        let idx = pos as usize;
        let bear_neighbors: usize = adj
            .neighbors_of(idx)
            .filter(|&nidx| board.grid.get(nidx).placed_wildlife() == Some(Wildlife::Bear))
            .count();

        if bear_neighbors == 0 {
            let bear_slots: usize = adj
                .neighbors_of(idx)
                .filter(|&nidx| {
                    let cell = board.grid.get(nidx);
                    cell.is_present() && cell.can_place_wildlife(Wildlife::Bear)
                })
                .count();
            if bear_slots >= 1 {
                half_pairs += 1;
            }
        }
        // Bears in groups of 3+ broke a potential pair
        if bear_neighbors >= 2 {
            penalty -= 20;
        }
    }

    let completable_pairs = (half_pairs / 2).min(2);
    completable_pairs * next_pair_marginal / 2 + penalty
}

fn count_bear_pairs(board: &Board) -> u16 {
    let positions = &board.wildlife_positions[Wildlife::Bear as usize];
    let adj = &*ADJACENCY;
    let mut visited = [false; 441];
    let mut pairs = 0u16;

    for &pos in positions.iter() {
        let idx = pos as usize;
        if visited[idx] {
            continue;
        }

        let mut size = 0u16;
        let mut queue = arrayvec::ArrayVec::<u16, 24>::new();
        queue.push(pos);
        visited[idx] = true;

        while let Some(current) = queue.pop() {
            size += 1;
            for nidx in adj.neighbors_of(current as usize) {
                if !visited[nidx] && board.grid.get(nidx).placed_wildlife() == Some(Wildlife::Bear)
                {
                    visited[nidx] = true;
                    queue.push(nidx as u16);
                }
            }
        }

        if size == 2 {
            pairs += 1;
        }
    }
    pairs
}

/// Elk potential (Card A): reward extendable lines based on actual marginals.
/// Line of 1→2: +3pts, 2→3: +4pts, 3→4+: +4pts. Lines of 4+ have no extension value.
fn elk_potential(board: &Board) -> i32 {
    let positions = &board.wildlife_positions[Wildlife::Elk as usize];
    if positions.is_empty() {
        return 0;
    }
    let adj = &*ADJACENCY;
    let mut potential: i32 = 0;

    for &position in positions {
        let start = position as usize;
        for direction in 0..3 {
            let backward = adj.neighbors[start][direction + 3];
            if backward == u16::MAX {
                continue;
            }
            let extension_index = backward as usize;
            if !board
                .grid
                .get(extension_index)
                .can_place_wildlife(Wildlife::Elk)
            {
                continue;
            }

            let mut line_len = 0u8;
            let mut current = adj.neighbors[extension_index][direction];
            while current != u16::MAX && line_len < 4 {
                let current_index = current as usize;
                if board.grid.get(current_index).placed_wildlife() != Some(Wildlife::Elk) {
                    break;
                }
                line_len += 1;
                current = adj.neighbors[current_index][direction];
            }
            potential += 20 * i32::from(matches!(line_len, 2 | 3));
        }
    }

    potential
}

#[cfg(test)]
fn elk_potential_reference(board: &Board) -> i32 {
    let positions = &board.wildlife_positions[Wildlife::Elk as usize];
    if positions.is_empty() {
        return 0;
    }
    let mut potential = 0;

    for &position in positions {
        let coord = HexCoord::from_index(position as usize);
        for &(dq, dr) in &HexCoord::LINE_DIRECTIONS {
            let forward = HexCoord::new(coord.q + dq, coord.r + dr);
            if !forward
                .to_index()
                .is_some_and(|index| board.grid.get(index).placed_wildlife() == Some(Wildlife::Elk))
            {
                continue;
            }

            let mut line_len = 1u16;
            let mut current = forward;
            while let Some(index) = current.to_index() {
                if board.grid.get(index).placed_wildlife() != Some(Wildlife::Elk) {
                    break;
                }
                line_len += 1;
                current = HexCoord::new(current.q + dq, current.r + dr);
            }
            current = HexCoord::new(coord.q - dq, coord.r - dr);
            while let Some(index) = current.to_index() {
                if board.grid.get(index).placed_wildlife() != Some(Wildlife::Elk) {
                    break;
                }
                line_len += 1;
                current = HexCoord::new(current.q - dq, current.r - dr);
            }

            if line_len >= 4 {
                continue;
            }
            let backward = HexCoord::new(coord.q - dq, coord.r - dr);
            if backward
                .to_index()
                .is_some_and(|index| board.grid.get(index).can_place_wildlife(Wildlife::Elk))
            {
                potential += match line_len {
                    1 => 15,
                    2 | 3 => 20,
                    _ => 0,
                };
            }
        }
    }

    potential
}

/// Salmon potential (Card A): reward extendable runs based on actual marginals.
/// Run scoring: 1→2: +2, 2→3: +3, 3→4: +4, 4→5: +4, 5→6: +5, 6→7+: +6
fn salmon_potential(board: &Board) -> i32 {
    let positions = &board.wildlife_positions[Wildlife::Salmon as usize];
    if positions.is_empty() {
        return 0;
    }
    let adj = &*ADJACENCY;
    let mut potential: i32 = 0;

    // Find connected components and their sizes (same as scorer)
    let mut visited = [false; 441];

    for &pos in positions.iter() {
        let idx = pos as usize;
        if visited[idx] {
            continue;
        }

        let mut component = arrayvec::ArrayVec::<u16, 24>::new();
        let mut queue = arrayvec::ArrayVec::<u16, 24>::new();
        queue.push(pos);
        visited[idx] = true;

        while let Some(current) = queue.pop() {
            component.push(current);
            for nidx in adj.neighbors_of(current as usize) {
                if !visited[nidx]
                    && board.grid.get(nidx).placed_wildlife() == Some(Wildlife::Salmon)
                {
                    visited[nidx] = true;
                    queue.push(nidx as u16);
                }
            }
        }

        let run_len = component.len() as u16;

        // Check if this is a valid run (no branching)
        let is_valid = component.iter().all(|&p| {
            adj.neighbors_of(p as usize)
                .filter(|&n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Salmon))
                .count()
                <= 2
        });

        if !is_valid {
            // Branching run — penalty
            potential -= 30;
            continue;
        }

        // Find endpoints (salmon with exactly 0 or 1 salmon neighbor)
        for &p in &component {
            let salmon_neighbors: usize = adj
                .neighbors_of(p as usize)
                .filter(|&n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Salmon))
                .count();

            let is_endpoint = salmon_neighbors <= 1;
            if !is_endpoint {
                continue;
            }

            // Check if endpoint has an adjacent empty slot accepting salmon
            let can_extend = adj.neighbors_of(p as usize).any(|nidx| {
                let cell = board.grid.get(nidx);
                cell.is_present()
                    && cell.can_place_wildlife(Wildlife::Salmon)
                    && cell.placed_wildlife() != Some(Wildlife::Salmon)
            });

            if can_extend {
                // Marginal value based on what the run would become
                let marginal = match run_len {
                    1 => 20, // 1→2: +2 points
                    2 => 30, // 2→3: +3
                    3 => 40, // 3→4: +4
                    4 => 40, // 4→5: +4
                    5 => 50, // 5→6: +5
                    6 => 60, // 6→7: +6
                    _ => 0,  // 7+ already maxed
                };
                // Discount for uncertainty, only count once per run
                potential += marginal / 2;
                break; // only count one endpoint per run
            }
        }
    }

    potential
}

/// Hawk potential (Card A): reward isolated hawks and safe placement opportunities.
/// Marginals: 0→1: +2, 1→2: +3, 2→3: +3, 3→4: +3, 4→5: +3, 5→6: +4, 6→7: +4, 7→8+: +6
fn hawk_potential(board: &Board) -> i32 {
    let positions = &board.wildlife_positions[Wildlife::Hawk as usize];
    let adj = &*ADJACENCY;
    let mut potential: i32 = 0;

    // Count current isolated hawks
    let mut isolated_count = 0u16;
    for &pos in positions.iter() {
        let has_adjacent_hawk = adj
            .neighbors_of(pos as usize)
            .any(|nidx| board.grid.get(nidx).placed_wildlife() == Some(Wildlife::Hawk));
        if !has_adjacent_hawk {
            isolated_count += 1;
        }
    }

    // Value of maintaining each isolated hawk's safety
    for &pos in positions.iter() {
        let idx = pos as usize;
        let has_adjacent_hawk = adj
            .neighbors_of(idx)
            .any(|nidx| board.grid.get(nidx).placed_wildlife() == Some(Wildlife::Hawk));

        if has_adjacent_hawk {
            continue;
        } // not isolated

        // Count danger slots (empty slots adjacent to this hawk that accept hawks)
        let danger_slots: usize = adj
            .neighbors_of(idx)
            .filter(|&nidx| {
                let cell = board.grid.get(nidx);
                cell.is_present() && cell.can_place_wildlife(Wildlife::Hawk)
            })
            .count();

        if danger_slots == 0 {
            potential += 10; // perfectly safe isolated hawk
        } else {
            // Some risk of losing isolation
            potential += 5;
        }
    }

    // Bonus for potential to add MORE isolated hawks
    // Count empty tiles that accept hawks and aren't adjacent to any hawk
    let mut safe_hawk_slots = 0i32;
    for &tile_idx in &board.placed_tiles {
        let idx = tile_idx as usize;
        let cell = board.grid.get(idx);
        if cell.has_wildlife() || !cell.can_place_wildlife(Wildlife::Hawk) {
            continue;
        }
        let adjacent_to_hawk = adj
            .neighbors_of(idx)
            .any(|nidx| board.grid.get(nidx).placed_wildlife() == Some(Wildlife::Hawk));
        if !adjacent_to_hawk {
            safe_hawk_slots += 1;
        }
    }

    // Marginal value of adding the next isolated hawk
    let next_marginal = match isolated_count {
        0 => 20,     // +2
        1..=4 => 30, // +3
        5..=6 => 40, // +4
        _ => 60,     // +6
    };

    // Potential for adding isolated hawks (capped at 2 for uncertainty)
    let addable = safe_hawk_slots.min(2);
    potential += addable * next_marginal / 2;

    potential
}

/// Fox potential (Card A): reward diversity setup. Each unique adjacent type = +1.
fn fox_potential(board: &Board) -> i32 {
    let positions = &board.wildlife_positions[Wildlife::Fox as usize];
    let adj = &*ADJACENCY;
    let mut potential: i32 = 0;

    for &pos in positions.iter() {
        let idx = pos as usize;
        let mut seen_mask = 0u8;
        let mut empty_slots = 0u8;

        for nidx in adj.neighbors_of(idx) {
            let cell = board.grid.get(nidx);
            if let Some(w) = cell.placed_wildlife() {
                seen_mask |= 1 << (w as u8);
            } else if cell.is_present() && cell.allowed_wildlife().count() > 0 {
                empty_slots += 1;
            }
        }

        let current_types = seen_mask.count_ones() as i32;
        let missing_types = 5 - current_types;
        // Each empty slot could potentially add a new type
        let addable = missing_types.min(empty_slots as i32);
        // Each new type = +1 point = 10 in our scale
        potential += addable * 10 / 2; // discount for uncertainty
    }

    potential
}

/// Bear potential (Card C): reward growing groups toward sizes 1/2/3.
/// Marginals: an empty pair-completion is +5 (2-pt loss as singleton + 5-pt pair = +3 net),
/// a 2→3 extension is +3 (8-pt triple − 5-pt pair). Sizes >3 score 0 so they're penalised.
/// +3 set-completion bonus rewarded if board has size-1 + size-2 components and a 3 is reachable.
fn bear_potential_c(board: &Board) -> i32 {
    let positions = &board.wildlife_positions[Wildlife::Bear as usize];
    let adj = &*ADJACENCY;
    if positions.is_empty() {
        return 0;
    }

    // Per-component sizes (BFS), and per-component "can extend by 1" flag.
    let mut visited = [false; 441];
    let mut sizes = arrayvec::ArrayVec::<u16, 32>::new();
    let mut extendable = arrayvec::ArrayVec::<bool, 32>::new();

    for &pos in positions.iter() {
        let idx = pos as usize;
        if visited[idx] {
            continue;
        }
        let mut comp = arrayvec::ArrayVec::<u16, 32>::new();
        let mut q = arrayvec::ArrayVec::<u16, 32>::new();
        q.push(pos);
        visited[idx] = true;
        while let Some(c) = q.pop() {
            let _ = comp.try_push(c);
            for n in adj.neighbors_of(c as usize) {
                if !visited[n] && board.grid.get(n).placed_wildlife() == Some(Wildlife::Bear) {
                    visited[n] = true;
                    let _ = q.try_push(n as u16);
                }
            }
        }
        let mut can_ext = false;
        for &p in &comp {
            for n in adj.neighbors_of(p as usize) {
                let cell = board.grid.get(n);
                if cell.is_present()
                    && !cell.has_wildlife()
                    && cell.can_place_wildlife(Wildlife::Bear)
                {
                    can_ext = true;
                    break;
                }
            }
            if can_ext {
                break;
            }
        }
        let _ = sizes.try_push(comp.len() as u16);
        let _ = extendable.try_push(can_ext);
    }

    let mut p: i32 = 0;
    let mut has_1 = false;
    let mut has_2 = false;
    let mut has_3 = false;
    for (&s, &ext) in sizes.iter().zip(extendable.iter()) {
        match s {
            1 => {
                has_1 = true;
                if ext {
                    p += 30;
                } // +3 to grow to pair (5 pts vs 2 pts)
            }
            2 => {
                has_2 = true;
                if ext {
                    p += 30;
                } // +3 to grow to triple (8 pts vs 5 pts)
            }
            3 => {
                has_3 = true;
                if ext {
                    p -= 80;
                } // BIG penalty: 4+ scores 0, lose all 8 pts
            }
            _ => {
                p -= 30;
            } // already in dead zone
        }
    }
    // Bonus completion: if we have 2 of the 3 sizes, partial credit toward +3 bonus.
    let sizes_present = (has_1 as i32) + (has_2 as i32) + (has_3 as i32);
    p += sizes_present * 10; // gradient toward the bonus

    p
}

/// Elk potential (Card B): reward DENSE shapes (triangle, rhombus) rather than lines.
/// Marginals: single = 2, pair = 5, triangle = 9, rhombus = 13.
/// "Triangle" = 3 mutually-adjacent elk. "Rhombus" = triangle + 1 adjacent elk.
fn elk_potential_b(board: &Board) -> i32 {
    let positions = &board.wildlife_positions[Wildlife::Elk as usize];
    if positions.is_empty() {
        return 0;
    }
    let adj = &*ADJACENCY;
    let mut p: i32 = 0;

    for &pos in positions.iter() {
        let idx = pos as usize;
        // Count elk neighbors and "elk-able" empty neighbors.
        let mut elk_n = 0u8;
        let mut grow = 0u8;
        for n in adj.neighbors_of(idx) {
            let cell = board.grid.get(n);
            if cell.placed_wildlife() == Some(Wildlife::Elk) {
                elk_n += 1;
            } else if cell.is_present()
                && !cell.has_wildlife()
                && cell.can_place_wildlife(Wildlife::Elk)
            {
                grow += 1;
            }
        }
        // Singleton with growth potential → reward becoming a pair (5 vs 2 = +3).
        if elk_n == 0 && grow > 0 {
            p += 15;
        }
        // Pair-end (1 elk neighbor) with growth → could become triangle (9 vs 5 = +4).
        // Highly valuable if the growth slot is also adjacent to the OTHER elk (forms triangle).
        if elk_n == 1 && grow > 0 {
            p += 20;
        }
        // 2-neighbor elk (already in a triad-like config) with growth → rhombus (13 vs 9 = +4).
        if elk_n == 2 && grow > 0 {
            p += 20;
        }
        // 3+ neighbor elk (already a tight cluster) — diminishing returns.
        if elk_n >= 3 {
            p -= 5;
        }
    }
    p
}

/// Salmon potential (Card D): reward salmon ADJACENT to non-salmon wildlife,
/// and runs of length ≥ 3. (Each adjacent non-salmon token = +1 pt under Card D.)
fn salmon_potential_d(board: &Board) -> i32 {
    let positions = &board.wildlife_positions[Wildlife::Salmon as usize];
    if positions.is_empty() {
        return 0;
    }
    let adj = &*ADJACENCY;
    let mut p: i32 = 0;

    // Component sizes (same as scorer)
    let mut visited = [false; 441];
    for &pos in positions.iter() {
        let idx = pos as usize;
        if visited[idx] {
            continue;
        }
        let mut comp = arrayvec::ArrayVec::<u16, 32>::new();
        let mut q = arrayvec::ArrayVec::<u16, 32>::new();
        q.push(pos);
        visited[idx] = true;
        while let Some(c) = q.pop() {
            let _ = comp.try_push(c);
            for n in adj.neighbors_of(c as usize) {
                if !visited[n] && board.grid.get(n).placed_wildlife() == Some(Wildlife::Salmon) {
                    visited[n] = true;
                    let _ = q.try_push(n as u16);
                }
            }
        }
        let len = comp.len();
        // Validity check (≤ 2 salmon neighbors per cell)
        let valid = comp.iter().all(|&p| {
            adj.neighbors_of(p as usize)
                .filter(|&n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Salmon))
                .count()
                <= 2
        });
        if !valid {
            p -= 30;
            continue;
        }

        // Reward growth toward length 3 (the qualifying threshold).
        // Runs at length 1 and 2 score 0 outright but are PROGRESS toward
        // the 3+ payoff. Give a graded reward by distance to threshold,
        // multiplied by the already-accumulated adjacent-animal bonus
        // (which will cash in once len hits 3).
        if len < 3 {
            // Find an endpoint with extension room.
            let mut can_extend = false;
            for &c in &comp {
                let salmon_n: usize = adj
                    .neighbors_of(c as usize)
                    .filter(|&n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Salmon))
                    .count();
                if salmon_n > 1 {
                    continue;
                }
                if adj.neighbors_of(c as usize).any(|n| {
                    let cell = board.grid.get(n);
                    cell.is_present()
                        && !cell.has_wildlife()
                        && cell.can_place_wildlife(Wildlife::Salmon)
                }) {
                    can_extend = true;
                    break;
                }
            }
            // Count unique non-salmon adjacent tokens already next to this seed run
            // — they'll add 1 pt each as soon as the run qualifies.
            let mut seen = [false; 441];
            let mut adj_animals = 0i32;
            for &c in &comp {
                for n in adj.neighbors_of(c as usize) {
                    if seen[n] {
                        continue;
                    }
                    seen[n] = true;
                    if let Some(w) = board.grid.get(n).placed_wildlife() {
                        if w != Wildlife::Salmon {
                            adj_animals += 1;
                        }
                    }
                }
            }
            // Base seed reward: how far along the run is. Length 2 is half-credit
            // toward length-3 qualification; length 1 is quarter-credit. Each is
            // also worth 1 pt directly once qualified (per-salmon scoring).
            let seed_value: i32 = match len {
                1 => 10, // singletons: small nudge to plant a seed
                2 => 25, // 1 step away — strong nudge to complete
                _ => 0,
            };
            // If the run can be extended, the seed-value AND the staged adj-animal
            // bonus (worth ~adj_animals pts at qualification) are realisable.
            if can_extend {
                p += seed_value;
                // Discount: bonus only realised after one more salmon placement,
                // which may not happen — half-credit.
                p += adj_animals * 5;
            } else {
                // Trapped seed (no extension slot) — small consolation, since the
                // adjacent animals at least set up future positioning. But mostly
                // a wasted salmon: penalise modestly so greedy doesn't plant
                // dead-end singletons.
                p += seed_value / 4;
                p -= 10;
            }
        } else {
            // Run already qualifies — count unique non-salmon adjacent tokens (the actual D bonus).
            let mut seen = [false; 441];
            let mut bonus = 0i32;
            for &c in &comp {
                for n in adj.neighbors_of(c as usize) {
                    if seen[n] {
                        continue;
                    }
                    seen[n] = true;
                    if let Some(w) = board.grid.get(n).placed_wildlife() {
                        if w != Wildlife::Salmon {
                            bonus += 10;
                        }
                    }
                }
            }
            // Reward the existing bonus + a small add-one-more-salmon nudge.
            p += bonus / 2; // already partially captured by current scoring
                            // Reward extension: each new salmon adds 1 pt + potentially adj-animal bonuses.
            let mut endpoints_extendable = 0;
            for &c in &comp {
                let salmon_n: usize = adj
                    .neighbors_of(c as usize)
                    .filter(|&n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Salmon))
                    .count();
                if salmon_n > 1 {
                    continue;
                }
                if adj.neighbors_of(c as usize).any(|n| {
                    let cell = board.grid.get(n);
                    cell.is_present()
                        && !cell.has_wildlife()
                        && cell.can_place_wildlife(Wildlife::Salmon)
                }) {
                    endpoints_extendable += 1;
                }
            }
            p += (endpoints_extendable.min(2) as i32) * 15;
        }
    }
    p
}

/// Hawk potential (Card D): reward hawks placed where they could form a non-adjacent
/// LOS pair with another hawk WITH non-hawk wildlife in the cells between.
/// (Per pair: 1 type=4, 2=7, 3+=9. Each hawk in ≤ 1 pair.)
fn hawk_potential_d(board: &Board) -> i32 {
    let positions = &board.wildlife_positions[Wildlife::Hawk as usize];
    if positions.len() < 2 {
        return 0;
    }
    let mut hawk_set = [false; 441];
    for &p in positions.iter() {
        hawk_set[p as usize] = true;
    }
    let mut p: i32 = 0;

    // For each hawk, walk each direction. If we find another hawk at distance ≥ 2
    // (non-adjacent) without an intervening hawk, count unique non-hawk wildlife
    // types in between. Bigger crowd = bigger pair-score.
    let mut paired = [false; 441];
    for (i, &pos) in positions.iter().enumerate() {
        if paired[pos as usize] {
            continue;
        }
        let coord = HexCoord::from_index(pos as usize);
        for &(dq, dr) in &cascadia_core::hex::HexCoord::DIRECTIONS {
            let mut cur = HexCoord::new(coord.q + dq, coord.r + dr);
            let mut steps = 1u16;
            let mut types_mask = 0u8;
            loop {
                match cur.to_index() {
                    Some(idx) => {
                        if hawk_set[idx] {
                            if steps >= 2 && !paired[idx] {
                                let unique =
                                    (types_mask & !(1 << Wildlife::Hawk as u8)).count_ones();
                                let pair_pts: i32 = match unique {
                                    0 => 0,
                                    1 => 40,
                                    2 => 70,
                                    _ => 90,
                                };
                                if pair_pts > 0 {
                                    p += pair_pts / 2; // discount for partial credit
                                    paired[pos as usize] = true;
                                    paired[idx] = true;
                                }
                            }
                            break;
                        }
                        if let Some(w) = board.grid.get(idx).placed_wildlife() {
                            types_mask |= 1 << (w as u8);
                        }
                    }
                    None => break,
                }
                cur = HexCoord::new(cur.q + dq, cur.r + dr);
                steps += 1;
            }
        }
        let _ = i;
    }
    p
}

/// Fox potential (Card B): reward foxes with ≥ 2 of the same non-fox type adjacent
/// (a "pair-type"). Per fox: 1 pair-type=3, 2=5, 3=7.
fn fox_potential_b(board: &Board) -> i32 {
    let positions = &board.wildlife_positions[Wildlife::Fox as usize];
    if positions.is_empty() {
        return 0;
    }
    let adj = &*ADJACENCY;
    let mut p: i32 = 0;

    for &pos in positions.iter() {
        let mut counts = [0u8; 5];
        let mut empty_with_type: [u8; 5] = [0; 5]; // empties that could become this type
        for n in adj.neighbors_of(pos as usize) {
            let cell = board.grid.get(n);
            if let Some(w) = cell.placed_wildlife() {
                if w != Wildlife::Fox {
                    counts[w as usize] += 1;
                }
            } else if cell.is_present() && !cell.has_wildlife() {
                for w in 0..5 {
                    let wl = Wildlife::from_u8(w as u8).unwrap();
                    if wl == Wildlife::Fox {
                        continue;
                    }
                    if cell.can_place_wildlife(wl) {
                        empty_with_type[w] += 1;
                    }
                }
            }
        }
        let pair_types_now = counts.iter().filter(|&&c| c >= 2).count() as i32;
        // Score the achieved state at half-credit (already in the live scoring).
        p += match pair_types_now {
            0 => 0,
            1 => 15,
            2 => 25,
            _ => 35,
        };
        // Reward types that are 1-away from being a pair (i.e., have count = 1
        // AND have an empty neighbor that could host that type).
        for w in 0..5 {
            if counts[w] == 1 && empty_with_type[w] >= 1 {
                p += 10; // could grow to a pair-type
            }
        }
    }
    p
}

/// Bonus for wildlife types that have very few remaining placement slots.
fn empty_slot_potential(board: &Board) -> i32 {
    empty_slot_potential_from_counts(empty_acceptance_counts(board))
}

fn empty_acceptance_counts(board: &Board) -> [u16; 5] {
    let mut acceptance_count = [0u16; 5];
    for &tile_idx in &board.placed_tiles {
        let cell = board.grid.get(tile_idx as usize);
        if cell.has_wildlife() {
            continue;
        }
        for w in Wildlife::ALL {
            if cell.allowed_wildlife().contains(w) {
                acceptance_count[w as usize] += 1;
            }
        }
    }
    acceptance_count
}

fn empty_slot_potential_from_counts(acceptance_count: [u16; 5]) -> i32 {
    let mut potential = 0;
    for wildlife in Wildlife::ALL {
        let count = acceptance_count[wildlife as usize];
        if count == 1 {
            potential += 15;
        } else if count == 2 {
            potential += 5;
        }
    }

    potential
}

#[cfg(test)]
mod tests {
    use super::*;
    use cascadia_core::{game::GameState, types::ScoringCards};
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn adjacency_elk_potential_matches_coordinate_reference() {
        for game_index in 0..24 {
            let mut rng = StdRng::seed_from_u64(0xe1ca_0000 + game_index);
            let mut game = GameState::new(4, ScoringCards::all_a(), &mut rng);
            while !game.is_game_over() {
                for board in &game.boards {
                    assert_eq!(elk_potential(board), elk_potential_reference(board));
                }
                let Some(movement) = crate::search::greedy_move(&game) else {
                    break;
                };
                assert!(crate::search::execute_scored_move(&mut game, &movement));
            }
        }
    }
}
