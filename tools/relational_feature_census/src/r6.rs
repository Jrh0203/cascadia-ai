use std::{collections::BTreeMap, hint::black_box, time::Instant};

use cascadia_game::Terrain;
use r2_sparse_entity_census::{
    FrontierToken, HabitatComponentToken, OccupiedTileToken, SparsePublicState,
};
use r3_action_edit_census::{
    ActionEdit, AxialCoord, BoardObjectChanges, BoardTileToken, ComponentChanges, ComponentObject,
    FrontierChanges, MarketSlotEdit, MarketSlotToken, MarketSnapshot, MotifChanges, ObjectUpdate,
    PlayerPublicSummary, PreparedPublicStateTrunk, PublicStateTrunk, SupplySnapshot, TileSemantic,
    WildlifeMotifObject,
};
use serde::{Deserialize, Serialize};

use crate::{
    CommonConfig, DistributionSummary, ExperimentLane, ReportEnvelope, Result,
    common::{deterministic_index, envelope, run_games, unix_ms},
    invalid,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct R6Metrics {
    pub positions: u64,
    pub complete_actions: u64,
    pub exact_apply_checks: u64,
    pub exact_apply_failures: u64,
    pub exact_undo_checks: u64,
    pub exact_undo_failures: u64,
    pub authoritative_apply_ns: u64,
    pub incremental_apply_undo_ns: u64,
    pub incremental_speedup_ppm: u64,
    pub actions_per_position: DistributionSummary,
    pub accumulator_bytes: DistributionSummary,
    pub exact_parity_pass: bool,
    pub throughput_gate_pass: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct AccumulatorSnapshot {
    board: Vec<BoardTileToken>,
    market: MarketSnapshot,
    supply: SupplySnapshot,
    active_player: PlayerPublicSummary,
    frontier: Vec<StableFrontierToken>,
    components: Vec<ComponentObject>,
    motifs: Vec<WildlifeMotifObject>,
    completed_turns: u8,
    current_relative_seat: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct StableFrontierTouch {
    terrain: Terrain,
    component_key: [u8; 32],
    component_size: u16,
    contact_edge_bits: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct StableFrontierToken {
    relative_seat: u8,
    coord: AxialCoord,
    neighbor_presence_bits: u8,
    neighbor_facing_terrains: [Option<Terrain>; 6],
    adjacent_wildlife_counts: [u8; 5],
    occupied_neighbor_runs: u8,
    opposite_neighbor_pair_bits: u8,
    touched_habitat_components: Vec<StableFrontierTouch>,
    resulting_size_by_terrain: [u16; 5],
    habitat_bridge_terrain_bits: u8,
    repeated_component_contact_terrain_bits: u8,
}

#[derive(Debug, Clone)]
pub struct UndoJournal {
    market: MarketSnapshot,
    supply: SupplySnapshot,
    active_player: PlayerPublicSummary,
    completed_turns: u8,
    current_relative_seat: u8,
}

#[derive(Debug, Clone)]
pub struct IncrementalSparseAccumulator {
    state_trunk_blake3: [u8; 32],
    board: BTreeMap<AxialCoord, BoardTileToken>,
    market: MarketSnapshot,
    supply: SupplySnapshot,
    active_player: PlayerPublicSummary,
    frontier: BTreeMap<AxialCoord, StableFrontierToken>,
    components: BTreeMap<[u8; 32], ComponentObject>,
    motifs: BTreeMap<AxialCoord, WildlifeMotifObject>,
    completed_turns: u8,
    current_relative_seat: u8,
}

impl IncrementalSparseAccumulator {
    pub fn from_prepared(prepared: &PreparedPublicStateTrunk<'_>) -> Result<Self> {
        let trunk = prepared.trunk();
        let sparse = &trunk.sparse;
        let board = sparse
            .occupied_tiles
            .iter()
            .filter(|tile| tile.relative_seat == 0)
            .map(|tile| {
                let token = board_tile_from_r2(tile);
                (token.coord, token)
            })
            .collect();
        let components = sparse
            .habitat_components
            .iter()
            .filter(|token| token.relative_seat == 0)
            .map(component_from_r2)
            .map(|component| (component.object_key, component))
            .collect::<BTreeMap<_, _>>();
        let frontier = sparse
            .legal_frontier
            .iter()
            .filter(|token| token.relative_seat == 0)
            .map(|token| stable_frontier_from_r2(token, &components))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .map(|token| (token.coord, token))
            .collect();
        let motifs = sparse
            .wildlife_motifs
            .iter()
            .filter(|token| token.relative_seat == 0)
            .map(motif_from_r2)
            .map(|motif| (motif.coord, motif))
            .collect();
        Ok(Self {
            state_trunk_blake3: prepared.canonical_hash(),
            board,
            market: market_from_sparse(sparse)?,
            supply: trunk.supply.clone(),
            active_player: player_summary_from_sparse(sparse, 0)?,
            frontier,
            components,
            motifs,
            completed_turns: sparse.global.turn,
            current_relative_seat: sparse.global.current_relative_seat,
        })
    }

    fn snapshot(&self) -> AccumulatorSnapshot {
        AccumulatorSnapshot {
            board: self.board.values().cloned().collect(),
            market: self.market.clone(),
            supply: self.supply.clone(),
            active_player: self.active_player.clone(),
            frontier: self.frontier.values().cloned().collect(),
            components: self.components.values().cloned().collect(),
            motifs: self.motifs.values().cloned().collect(),
            completed_turns: self.completed_turns,
            current_relative_seat: self.current_relative_seat,
        }
    }

    pub fn canonical_blake3(&self) -> Result<String> {
        Ok(blake3::hash(&postcard::to_allocvec(&self.snapshot())?)
            .to_hex()
            .to_string())
    }

    pub fn apply(&mut self, edit: &ActionEdit) -> Result<UndoJournal> {
        if edit.state_trunk_blake3 != self.state_trunk_blake3 {
            return Err(invalid(
                "R6 action edit belongs to a different parent trunk",
            ));
        }
        if self.completed_turns != edit.turn.completed_turns_before
            || self.current_relative_seat != edit.turn.current_relative_seat_before
        {
            return Err(invalid("R6 turn precondition mismatch"));
        }
        let journal = UndoJournal {
            market: self.market.clone(),
            supply: self.supply.clone(),
            active_player: self.active_player.clone(),
            completed_turns: self.completed_turns,
            current_relative_seat: self.current_relative_seat,
        };

        if self.market != edit.prelude.market_before
            || self.active_player != edit.prelude.active_player_before
        {
            return Err(invalid("R6 prelude precondition mismatch"));
        }
        apply_market_edits(&mut self.market, &edit.prelude.market_edits)?;
        if self.market != edit.prelude.market_after {
            return Err(invalid("R6 prelude market result mismatch"));
        }
        self.active_player = edit.prelude.active_player_after.clone();
        self.supply = edit.prelude.supply.apply(&self.supply)?;

        if self.market != edit.placement.market_before
            || self.active_player != edit.placement.active_player_before
        {
            return Err(invalid("R6 placement precondition mismatch"));
        }
        apply_board_changes(&mut self.board, &edit.placement.board)?;
        apply_market_edits(&mut self.market, &edit.placement.market_edits)?;
        if self.market != edit.placement.market_after {
            return Err(invalid("R6 placement market result mismatch"));
        }
        self.active_player = edit.placement.active_player_after.clone();
        self.supply = edit.placement.supply.apply(&self.supply)?;
        remove_frontier_before(&mut self.frontier, &edit.frontier, &self.components)?;
        apply_component_changes(&mut self.components, &edit.components)?;
        insert_frontier_after(&mut self.frontier, &edit.frontier, &self.components)?;
        apply_motif_changes(&mut self.motifs, &edit.motifs)?;
        self.completed_turns = edit.turn.completed_turns_after;
        self.current_relative_seat = edit.turn.current_relative_seat_after;
        Ok(journal)
    }

    pub fn undo(&mut self, edit: &ActionEdit, journal: UndoJournal) -> Result<()> {
        undo_motif_changes(&mut self.motifs, &edit.motifs)?;
        remove_frontier_after(&mut self.frontier, &edit.frontier, &self.components)?;
        undo_component_changes(&mut self.components, &edit.components)?;
        insert_frontier_before(&mut self.frontier, &edit.frontier, &self.components)?;
        undo_board_changes(&mut self.board, &edit.placement.board)?;
        self.market = journal.market;
        self.supply = journal.supply;
        self.active_player = journal.active_player;
        self.completed_turns = journal.completed_turns;
        self.current_relative_seat = journal.current_relative_seat;
        Ok(())
    }

    pub fn matches_authoritative(
        &self,
        applied: &r3_action_edit_census::AppliedPublicState,
        edit: &ActionEdit,
        parent: &PublicStateTrunk,
    ) -> Result<bool> {
        Ok(self.snapshot() == authoritative_snapshot(applied, edit, parent)?)
    }
}

fn authoritative_snapshot(
    applied: &r3_action_edit_census::AppliedPublicState,
    edit: &ActionEdit,
    parent: &PublicStateTrunk,
) -> Result<AccumulatorSnapshot> {
    let mut geometry_record = applied.record.clone();
    geometry_record.market_entities = parent.public_record()?.market_entities;
    let sparse = SparsePublicState::from_position_record(&geometry_record, None)?;
    let mut authoritative_components = sparse
        .habitat_components
        .iter()
        .filter(|token| token.relative_seat == 0)
        .map(component_from_r2)
        .map(|component| (component.object_key, component))
        .collect::<BTreeMap<_, _>>()
        .into_values()
        .collect::<Vec<_>>();
    authoritative_components.sort_by_key(|component| component.object_key);
    let authoritative_component_map = authoritative_components
        .iter()
        .cloned()
        .map(|component| (component.object_key, component))
        .collect::<BTreeMap<_, _>>();
    Ok(AccumulatorSnapshot {
        board: sparse
            .occupied_tiles
            .iter()
            .filter(|tile| tile.relative_seat == 0)
            .map(board_tile_from_r2)
            .collect(),
        market: edit.placement.market_after.clone(),
        supply: applied.supply.clone(),
        active_player: player_summary_from_sparse(&sparse, 0)?,
        frontier: sparse
            .legal_frontier
            .iter()
            .filter(|token| token.relative_seat == 0)
            .map(|token| stable_frontier_from_r2(token, &authoritative_component_map))
            .collect::<Result<Vec<_>>>()?,
        components: authoritative_components,
        motifs: sparse
            .wildlife_motifs
            .iter()
            .filter(|token| token.relative_seat == 0)
            .map(motif_from_r2)
            .collect(),
        completed_turns: sparse.global.turn,
        current_relative_seat: sparse.global.current_relative_seat,
    })
}

#[derive(Default)]
struct SeedMetrics {
    positions: u64,
    complete_actions: u64,
    exact_apply_checks: u64,
    exact_apply_failures: u64,
    exact_undo_checks: u64,
    exact_undo_failures: u64,
    authoritative_apply_ns: u128,
    incremental_apply_undo_ns: u128,
    actions_per_position: Vec<u64>,
    accumulator_bytes: Vec<u64>,
}

impl SeedMetrics {
    fn merge(&mut self, other: Self) {
        self.positions += other.positions;
        self.complete_actions += other.complete_actions;
        self.exact_apply_checks += other.exact_apply_checks;
        self.exact_apply_failures += other.exact_apply_failures;
        self.exact_undo_checks += other.exact_undo_checks;
        self.exact_undo_failures += other.exact_undo_failures;
        self.authoritative_apply_ns += other.authoritative_apply_ns;
        self.incremental_apply_undo_ns += other.incremental_apply_undo_ns;
        self.actions_per_position.extend(other.actions_per_position);
        self.accumulator_bytes.extend(other.accumulator_bytes);
    }
}

pub fn run_r6(config: CommonConfig) -> Result<ReportEnvelope<R6Metrics>> {
    if config.lane != ExperimentLane::R6Incremental {
        return Err(invalid("R6 runner received a non-R6 lane"));
    }
    let started = unix_ms()?;
    let per_seed = run_games(&config, run_seed)?;
    let mut combined = SeedMetrics::default();
    for (_, metrics) in per_seed {
        combined.merge(metrics);
    }
    let authoritative_apply_ns = u64::try_from(combined.authoritative_apply_ns)?;
    let incremental_apply_undo_ns = u64::try_from(combined.incremental_apply_undo_ns)?;
    let incremental_speedup_ppm = authoritative_apply_ns
        .checked_mul(1_000_000)
        .ok_or_else(|| invalid("R6 speedup ratio overflowed"))?
        / incremental_apply_undo_ns.max(1);
    let exact_parity_pass = combined.exact_apply_failures == 0
        && combined.exact_undo_failures == 0
        && combined.exact_apply_checks == combined.complete_actions
        && combined.exact_undo_checks == combined.complete_actions;
    let throughput_gate_pass = incremental_speedup_ppm >= 2_000_000;
    let passed = exact_parity_pass && throughput_gate_pass;
    let classification = if passed {
        "r6_incremental_apply_undo_promoted"
    } else if !exact_parity_pass {
        "r6_incremental_exactness_failed"
    } else {
        "r6_incremental_throughput_failed"
    };
    envelope(
        config,
        R6Metrics {
            positions: combined.positions,
            complete_actions: combined.complete_actions,
            exact_apply_checks: combined.exact_apply_checks,
            exact_apply_failures: combined.exact_apply_failures,
            exact_undo_checks: combined.exact_undo_checks,
            exact_undo_failures: combined.exact_undo_failures,
            authoritative_apply_ns,
            incremental_apply_undo_ns,
            incremental_speedup_ppm,
            actions_per_position: DistributionSummary::from_values(combined.actions_per_position)?,
            accumulator_bytes: DistributionSummary::from_values(combined.accumulator_bytes)?,
            exact_parity_pass,
            throughput_gate_pass,
        },
        passed,
        classification,
        started,
    )
}

fn run_seed(seed: u64, mut game: cascadia_game::GameState) -> Result<SeedMetrics> {
    let mut metrics = SeedMetrics::default();
    while !game.is_game_over() {
        metrics.positions += 1;
        let game_index = seed
            .checked_mul(100)
            .and_then(|value| value.checked_add(u64::from(game.completed_turns())))
            .ok_or_else(|| invalid("R6 game index overflowed"))?;
        let trunk = PublicStateTrunk::observe(&game, game_index)?;
        let prepared = trunk.prepare_action_edits()?;
        let (prelude, _) = game.preview_free_three_of_a_kind_if_feasible()?;
        let observed = prepared.observe_legal_actions(&game, &prelude)?;
        if observed.is_empty() {
            return Err(invalid("R6 nonterminal position has no legal actions"));
        }
        metrics.complete_actions += observed.len() as u64;
        metrics.actions_per_position.push(observed.len() as u64);

        let authority_started = Instant::now();
        for (_, edit) in &observed {
            let applied = prepared.apply(edit)?;
            black_box(applied.canonical_record_hash());
        }
        metrics.authoritative_apply_ns += authority_started.elapsed().as_nanos();

        let mut accumulator = IncrementalSparseAccumulator::from_prepared(&prepared)?;
        metrics
            .accumulator_bytes
            .push(postcard::to_allocvec(&accumulator.snapshot())?.len() as u64);
        let incremental_started = Instant::now();
        for (_, edit) in &observed {
            let journal = accumulator.apply(edit)?;
            black_box(accumulator.completed_turns);
            accumulator.undo(edit, journal)?;
        }
        metrics.incremental_apply_undo_ns += incremental_started.elapsed().as_nanos();

        let parent_digest = accumulator.canonical_blake3()?;
        for (_, edit) in &observed {
            let authoritative = prepared.apply(edit)?;
            let journal = accumulator.apply(edit)?;
            metrics.exact_apply_checks += 1;
            if !accumulator.matches_authoritative(&authoritative, edit, &trunk)? {
                metrics.exact_apply_failures += 1;
            }
            accumulator.undo(edit, journal)?;
            metrics.exact_undo_checks += 1;
            if accumulator.canonical_blake3()? != parent_digest {
                metrics.exact_undo_failures += 1;
            }
        }

        let selected = observed
            [deterministic_index(seed, game.completed_turns(), observed.len(), b"r6-advance")]
        .0
        .clone();
        game.apply(&selected)?;
    }
    if metrics.positions != 80 {
        return Err(invalid(format!(
            "R6 seed {seed} produced {} positions instead of 80",
            metrics.positions
        )));
    }
    Ok(metrics)
}

fn board_tile_from_r2(tile: &OccupiedTileToken) -> BoardTileToken {
    BoardTileToken {
        coord: AxialCoord::new(tile.coord.q, tile.coord.r),
        tile: TileSemantic::new(
            tile.terrain_a,
            tile.terrain_b,
            tile.wildlife_eligibility,
            tile.keystone,
        ),
        rotation: tile.rotation.get(),
        directed_edge_terrains: tile.directed_edge_terrains,
        placed_wildlife: tile.placed_wildlife,
    }
}

fn component_from_r2(token: &HabitatComponentToken) -> ComponentObject {
    let members = token
        .members
        .iter()
        .map(|coord| AxialCoord::new(coord.q, coord.r))
        .collect::<Vec<_>>();
    ComponentObject {
        object_key: component_key(token.relative_seat, token.terrain as u8, &members),
        relative_seat: token.relative_seat,
        terrain: token.terrain,
        members,
        member_count: token.member_count,
        matching_internal_edge_count: token.matching_internal_edge_count,
        open_boundary_edge_count: token.open_boundary_edge_count,
        frontier_contact_count: token.frontier_contact_count,
    }
}

fn component_key(relative_seat: u8, terrain: u8, members: &[AxialCoord]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"cascadia-r3-component-object-v1");
    hasher.update(&[relative_seat, terrain]);
    hasher.update(&(members.len() as u16).to_le_bytes());
    for member in members {
        hasher.update(&member.q.to_le_bytes());
        hasher.update(&member.r.to_le_bytes());
    }
    *hasher.finalize().as_bytes()
}

fn motif_from_r2(token: &r2_sparse_entity_census::WildlifeMotifToken) -> WildlifeMotifObject {
    WildlifeMotifObject {
        relative_seat: token.relative_seat,
        coord: AxialCoord::new(token.coord.q, token.coord.r),
        wildlife: token.wildlife,
        neighbor_wildlife: token.neighbor_wildlife,
        adjacent_wildlife_counts: token.adjacent_wildlife_counts,
        same_species_neighbor_bits: token.same_species_neighbor_bits,
    }
}

fn stable_frontier_from_r2(
    token: &FrontierToken,
    components: &BTreeMap<[u8; 32], ComponentObject>,
) -> Result<StableFrontierToken> {
    let coord = AxialCoord::new(token.coord.q, token.coord.r);
    let mut touches = Vec::with_capacity(token.touched_habitat_components.len());
    for touch in &token.touched_habitat_components {
        let member = (0..6)
            .find(|edge| touch.contact_edge_bits & (1 << edge) != 0)
            .map(|edge| coord_neighbor(coord, edge))
            .ok_or_else(|| invalid("R6 frontier touch has no contact edge"))?;
        let component = components
            .values()
            .find(|component| {
                component.terrain == touch.terrain && component.members.contains(&member)
            })
            .ok_or_else(|| invalid("R6 frontier touch cannot resolve a stable component"))?;
        if component.member_count != touch.component_size {
            return Err(invalid("R6 frontier touch component size mismatch"));
        }
        touches.push(StableFrontierTouch {
            terrain: touch.terrain,
            component_key: component.object_key,
            component_size: touch.component_size,
            contact_edge_bits: touch.contact_edge_bits,
        });
    }
    touches.sort_by_key(|touch| (touch.terrain as u8, touch.component_key));
    Ok(StableFrontierToken {
        relative_seat: token.relative_seat,
        coord,
        neighbor_presence_bits: token.neighbor_presence_bits,
        neighbor_facing_terrains: token.neighbor_facing_terrains,
        adjacent_wildlife_counts: token.adjacent_wildlife_counts,
        occupied_neighbor_runs: token.occupied_neighbor_runs,
        opposite_neighbor_pair_bits: token.opposite_neighbor_pair_bits,
        touched_habitat_components: touches,
        resulting_size_by_terrain: token.resulting_size_by_terrain,
        habitat_bridge_terrain_bits: token.habitat_bridge_terrain_bits,
        repeated_component_contact_terrain_bits: token.repeated_component_contact_terrain_bits,
    })
}

fn coord_neighbor(coord: AxialCoord, edge: usize) -> AxialCoord {
    const DIRECTIONS: [(i16, i16); 6] = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)];
    let (dq, dr) = DIRECTIONS[edge % 6];
    AxialCoord::new(coord.q + dq, coord.r + dr)
}

fn market_from_sparse(sparse: &SparsePublicState) -> Result<MarketSnapshot> {
    let slots = sparse
        .market
        .iter()
        .map(|token| MarketSlotToken {
            slot: token.slot,
            tile: token.tile.map(|tile| {
                TileSemantic::new(
                    tile.terrain_a,
                    tile.terrain_b,
                    tile.wildlife_eligibility,
                    tile.keystone,
                )
            }),
            wildlife: token.wildlife,
        })
        .collect::<Vec<_>>()
        .try_into()
        .map_err(|_| invalid("R6 sparse market does not contain four slots"))?;
    Ok(MarketSnapshot { slots })
}

fn player_summary_from_sparse(
    sparse: &SparsePublicState,
    relative_seat: usize,
) -> Result<PlayerPublicSummary> {
    let player = sparse
        .players
        .get(relative_seat)
        .ok_or_else(|| invalid("R6 missing player summary"))?;
    Ok(PlayerPublicSummary {
        occupied_count: player.occupied_count,
        nature_tokens: player.nature_tokens,
        wildlife_counts: player.wildlife_counts,
        largest_habitats: player.largest_habitats,
    })
}

fn apply_market_edits(market: &mut MarketSnapshot, edits: &[MarketSlotEdit]) -> Result<()> {
    for edit in edits {
        let slot = usize::from(edit.slot);
        let current = market
            .slots
            .get_mut(slot)
            .ok_or_else(|| invalid("R6 market edit references an invalid slot"))?;
        if current != &edit.before {
            return Err(invalid("R6 market edit precondition mismatch"));
        }
        *current = edit.after.clone();
    }
    Ok(())
}

fn apply_board_changes(
    board: &mut BTreeMap<AxialCoord, BoardTileToken>,
    changes: &BoardObjectChanges,
) -> Result<()> {
    apply_object_changes(
        board,
        &changes.removed,
        &changes.updated,
        &changes.added,
        |token| token.coord,
        "board",
    )
}

fn undo_board_changes(
    board: &mut BTreeMap<AxialCoord, BoardTileToken>,
    changes: &BoardObjectChanges,
) -> Result<()> {
    apply_object_changes(
        board,
        &changes.added,
        &changes
            .updated
            .iter()
            .map(|update| ObjectUpdate {
                before: update.after.clone(),
                after: update.before.clone(),
            })
            .collect::<Vec<_>>(),
        &changes.removed,
        |token| token.coord,
        "board undo",
    )
}

fn remove_frontier_before(
    frontier: &mut BTreeMap<AxialCoord, StableFrontierToken>,
    changes: &FrontierChanges,
    components: &BTreeMap<[u8; 32], ComponentObject>,
) -> Result<()> {
    for token in changes
        .removed
        .iter()
        .chain(changes.updated.iter().map(|update| &update.before))
    {
        let stable = stable_frontier_from_r2(token, components)?;
        match frontier.remove(&stable.coord) {
            Some(actual) if actual == stable => {}
            _ => return Err(invalid("R6 frontier before-state removal mismatch")),
        }
    }
    Ok(())
}

fn insert_frontier_after(
    frontier: &mut BTreeMap<AxialCoord, StableFrontierToken>,
    changes: &FrontierChanges,
    components: &BTreeMap<[u8; 32], ComponentObject>,
) -> Result<()> {
    for token in changes
        .updated
        .iter()
        .map(|update| &update.after)
        .chain(changes.added.iter())
    {
        let stable = stable_frontier_from_r2(token, components)?;
        if frontier.insert(stable.coord, stable).is_some() {
            return Err(invalid("R6 frontier after-state insertion collided"));
        }
    }
    Ok(())
}

fn remove_frontier_after(
    frontier: &mut BTreeMap<AxialCoord, StableFrontierToken>,
    changes: &FrontierChanges,
    components: &BTreeMap<[u8; 32], ComponentObject>,
) -> Result<()> {
    for token in changes
        .added
        .iter()
        .chain(changes.updated.iter().map(|update| &update.after))
    {
        let stable = stable_frontier_from_r2(token, components)?;
        match frontier.remove(&stable.coord) {
            Some(actual) if actual == stable => {}
            _ => return Err(invalid("R6 frontier after-state removal mismatch")),
        }
    }
    Ok(())
}

fn insert_frontier_before(
    frontier: &mut BTreeMap<AxialCoord, StableFrontierToken>,
    changes: &FrontierChanges,
    components: &BTreeMap<[u8; 32], ComponentObject>,
) -> Result<()> {
    for token in changes
        .updated
        .iter()
        .map(|update| &update.before)
        .chain(changes.removed.iter())
    {
        let stable = stable_frontier_from_r2(token, components)?;
        if frontier.insert(stable.coord, stable).is_some() {
            return Err(invalid("R6 frontier before-state insertion collided"));
        }
    }
    Ok(())
}

fn apply_component_changes(
    components: &mut BTreeMap<[u8; 32], ComponentObject>,
    changes: &ComponentChanges,
) -> Result<()> {
    apply_object_changes(
        components,
        &changes.removed,
        &changes.updated,
        &changes.added,
        |token| token.object_key,
        "component",
    )
}

fn undo_component_changes(
    components: &mut BTreeMap<[u8; 32], ComponentObject>,
    changes: &ComponentChanges,
) -> Result<()> {
    apply_object_changes(
        components,
        &changes.added,
        &changes
            .updated
            .iter()
            .map(|update| ObjectUpdate {
                before: update.after.clone(),
                after: update.before.clone(),
            })
            .collect::<Vec<_>>(),
        &changes.removed,
        |token| token.object_key,
        "component undo",
    )
}

fn apply_motif_changes(
    motifs: &mut BTreeMap<AxialCoord, WildlifeMotifObject>,
    changes: &MotifChanges,
) -> Result<()> {
    apply_object_changes(
        motifs,
        &changes.removed,
        &changes.updated,
        &changes.added,
        |token| token.coord,
        "motif",
    )
}

fn undo_motif_changes(
    motifs: &mut BTreeMap<AxialCoord, WildlifeMotifObject>,
    changes: &MotifChanges,
) -> Result<()> {
    apply_object_changes(
        motifs,
        &changes.added,
        &changes
            .updated
            .iter()
            .map(|update| ObjectUpdate {
                before: update.after.clone(),
                after: update.before.clone(),
            })
            .collect::<Vec<_>>(),
        &changes.removed,
        |token| token.coord,
        "motif undo",
    )
}

fn apply_object_changes<K, T>(
    objects: &mut BTreeMap<K, T>,
    removed: &[T],
    updated: &[ObjectUpdate<T>],
    added: &[T],
    key: impl Fn(&T) -> K,
    label: &str,
) -> Result<()>
where
    K: Ord + Copy,
    T: Clone + PartialEq,
{
    for token in removed {
        let object_key = key(token);
        match objects.remove(&object_key) {
            Some(actual) if actual == *token => {}
            _ => return Err(invalid(format!("R6 {label} removal mismatch"))),
        }
    }
    for update in updated {
        let before_key = key(&update.before);
        let after_key = key(&update.after);
        match objects.remove(&before_key) {
            Some(actual) if actual == update.before => {}
            _ => return Err(invalid(format!("R6 {label} update mismatch"))),
        }
        if objects.insert(after_key, update.after.clone()).is_some() {
            return Err(invalid(format!("R6 {label} update collided")));
        }
    }
    for token in added {
        if objects.insert(key(token), token.clone()).is_some() {
            return Err(invalid(format!("R6 {label} addition collided")));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use cascadia_game::{GameConfig, GameSeed, GameState};

    use super::*;

    #[test]
    fn accumulator_matches_and_undoes_every_opening_action() {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(22),
        )
        .unwrap();
        let trunk = PublicStateTrunk::observe(&game, 22).unwrap();
        let prepared = trunk.prepare_action_edits().unwrap();
        let (prelude, _) = game.preview_free_three_of_a_kind_if_feasible().unwrap();
        let observed = prepared.observe_legal_actions(&game, &prelude).unwrap();
        let mut accumulator = IncrementalSparseAccumulator::from_prepared(&prepared).unwrap();
        let parent = accumulator.canonical_blake3().unwrap();
        for (_, edit) in observed {
            let expected = prepared.apply(&edit).unwrap();
            let journal = accumulator.apply(&edit).unwrap();
            let actual_snapshot = accumulator.snapshot();
            let expected_snapshot = authoritative_snapshot(&expected, &edit, &trunk).unwrap();
            if actual_snapshot != expected_snapshot {
                eprintln!(
                    "board={} market={} supply={} player={} frontier={} components={} motifs={} turn={} seat={}",
                    actual_snapshot.board == expected_snapshot.board,
                    actual_snapshot.market == expected_snapshot.market,
                    actual_snapshot.supply == expected_snapshot.supply,
                    actual_snapshot.active_player == expected_snapshot.active_player,
                    actual_snapshot.frontier == expected_snapshot.frontier,
                    actual_snapshot.components == expected_snapshot.components,
                    actual_snapshot.motifs == expected_snapshot.motifs,
                    actual_snapshot.completed_turns == expected_snapshot.completed_turns,
                    actual_snapshot.current_relative_seat
                        == expected_snapshot.current_relative_seat,
                );
            }
            assert_eq!(actual_snapshot, expected_snapshot);
            accumulator.undo(&edit, journal).unwrap();
            assert_eq!(accumulator.canonical_blake3().unwrap(), parent);
        }
    }
}
