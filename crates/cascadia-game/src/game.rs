use blake3::Hasher;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    Board, BoardError, HexCoord, Market, MarketSlot, Rotation, STANDARD_TILES, STARTER_CLUSTERS,
    ScoringCards, Terrain, Tile, Wildlife,
};

const STATE_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GameMode {
    Standard,
    Solo,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GameConfig {
    pub player_count: u8,
    pub mode: GameMode,
    pub scoring_cards: ScoringCards,
    pub habitat_bonuses: bool,
}

impl GameConfig {
    pub fn standard(player_count: u8, scoring_cards: ScoringCards) -> Result<Self, RuleError> {
        if !(2..=4).contains(&player_count) {
            return Err(RuleError::InvalidPlayerCount(player_count));
        }
        Ok(Self {
            player_count,
            mode: GameMode::Standard,
            scoring_cards,
            habitat_bonuses: true,
        })
    }

    pub const fn solo(scoring_cards: ScoringCards) -> Self {
        Self {
            player_count: 1,
            mode: GameMode::Solo,
            scoring_cards,
            habitat_bonuses: true,
        }
    }

    pub fn research_aaaaa(player_count: u8) -> Result<Self, RuleError> {
        let mut config = Self::standard(player_count, ScoringCards::AAAAA)?;
        config.habitat_bonuses = false;
        Ok(config)
    }

    fn validate(self) -> Result<(), RuleError> {
        match (self.mode, self.player_count) {
            (GameMode::Solo, 1) | (GameMode::Standard, 2..=4) => Ok(()),
            _ => Err(RuleError::InvalidPlayerCount(self.player_count)),
        }
    }

    fn tile_stack_size(self) -> usize {
        match self.player_count {
            1 | 2 => 43,
            3 => 63,
            4 => 83,
            _ => unreachable!("validated player count"),
        }
    }

    fn total_turns(self) -> u16 {
        if self.mode == GameMode::Solo {
            20
        } else {
            20 * u16::from(self.player_count)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct GameSeed(pub [u8; 32]);

impl GameSeed {
    pub fn from_u64(seed: u64) -> Self {
        let mut hasher = Hasher::new();
        hasher.update(b"cascadia-v2-game-seed");
        hasher.update(&seed.to_le_bytes());
        Self(*hasher.finalize().as_bytes())
    }

    fn rng(self, domain: &[u8]) -> ChaCha8Rng {
        let mut hasher = Hasher::new();
        hasher.update(b"cascadia-v2-rng-domain");
        hasher.update(&self.0);
        hasher.update(domain);
        ChaCha8Rng::from_seed(*hasher.finalize().as_bytes())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DraftChoice {
    Paired {
        slot: MarketSlot,
    },
    Independent {
        tile_slot: MarketSlot,
        wildlife_slot: MarketSlot,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TilePlacement {
    pub coord: HexCoord,
    pub rotation: Rotation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WildlifeWipe {
    pub slots: Vec<MarketSlot>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct MarketPrelude {
    pub replace_three_of_a_kind: bool,
    pub wildlife_wipes: Vec<WildlifeWipe>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnAction {
    pub replace_three_of_a_kind: bool,
    pub wildlife_wipes: Vec<WildlifeWipe>,
    pub draft: DraftChoice,
    pub tile: TilePlacement,
    pub wildlife: Option<HexCoord>,
}

impl TurnAction {
    pub fn paired(slot: MarketSlot, coord: HexCoord, rotation: Rotation) -> Self {
        Self {
            replace_three_of_a_kind: false,
            wildlife_wipes: Vec::new(),
            draft: DraftChoice::Paired { slot },
            tile: TilePlacement { coord, rotation },
            wildlife: None,
        }
    }

    pub fn prelude(&self) -> MarketPrelude {
        MarketPrelude {
            replace_three_of_a_kind: self.replace_three_of_a_kind,
            wildlife_wipes: self.wildlife_wipes.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GameState {
    schema_version: u16,
    config: GameConfig,
    seed: GameSeed,
    boards: Vec<Board>,
    market: Market,
    tile_stack: Vec<Tile>,
    wildlife_bag: Vec<Wildlife>,
    excluded_tiles: Vec<Tile>,
    discarded_tiles: Vec<Tile>,
    discarded_wildlife: Vec<Wildlife>,
    current_player: u8,
    completed_turns: u16,
    wildlife_return_counter: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicGameState {
    config: GameConfig,
    boards: Vec<Board>,
    market: Market,
    current_player: u8,
    completed_turns: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicSupply {
    pub wildlife_bag: [u8; 5],
    pub unseen_tile_terrain_capacity: [u8; 5],
    pub unseen_tile_wildlife_capacity: [u8; 5],
    pub unseen_keystones_by_terrain: [u8; 5],
    pub unseen_dual_terrain_pairs: [u8; 10],
}

impl PublicGameState {
    pub fn config(&self) -> GameConfig {
        self.config
    }

    pub fn boards(&self) -> &[Board] {
        &self.boards
    }

    pub fn market(&self) -> &Market {
        &self.market
    }

    pub fn current_player(&self) -> usize {
        usize::from(self.current_player)
    }

    pub fn completed_turns(&self) -> u16 {
        self.completed_turns
    }

    pub fn total_turns(&self) -> u16 {
        self.config.total_turns()
    }

    pub fn turns_remaining_for_player(&self, player: usize) -> u16 {
        turns_remaining_for_player(self.completed_turns, self.boards.len(), player)
    }

    pub fn unplaced_wildlife_counts(&self) -> [u8; 5] {
        unplaced_wildlife_counts(&self.boards)
    }

    pub fn is_game_over(&self) -> bool {
        self.completed_turns == self.total_turns()
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        postcard::to_allocvec(self).expect("serializing an in-memory public game state cannot fail")
    }

    pub fn canonical_hash(&self) -> blake3::Hash {
        blake3::hash(&self.canonical_bytes())
    }
}

impl GameState {
    pub fn new(config: GameConfig, seed: GameSeed) -> Result<Self, RuleError> {
        config.validate()?;

        let mut tiles = STANDARD_TILES.to_vec();
        tiles.shuffle(&mut seed.rng(b"tile-stack"));
        let excluded_tiles = tiles.split_off(config.tile_stack_size());
        let mut tile_stack = tiles;

        let mut wildlife_bag = Vec::with_capacity(100);
        for wildlife in Wildlife::ALL {
            wildlife_bag.extend(std::iter::repeat_n(wildlife, 20));
        }
        wildlife_bag.shuffle(&mut seed.rng(b"wildlife-bag"));

        let mut starter_indices = [0usize, 1, 2, 3, 4];
        starter_indices.shuffle(&mut seed.rng(b"starter-clusters"));
        let boards = starter_indices
            .into_iter()
            .take(usize::from(config.player_count))
            .map(|index| Board::from_starter(&STARTER_CLUSTERS[index]))
            .collect();

        let mut market = Market::empty();
        for slot in MarketSlot::ALL {
            market.tiles[slot.index()] = tile_stack.pop();
            market.wildlife[slot.index()] = wildlife_bag.pop();
        }

        let mut state = Self {
            schema_version: STATE_SCHEMA_VERSION,
            config,
            seed,
            boards,
            market,
            tile_stack,
            wildlife_bag,
            excluded_tiles,
            discarded_tiles: Vec::new(),
            discarded_wildlife: Vec::new(),
            current_player: 0,
            completed_turns: 0,
            wildlife_return_counter: 0,
        };
        state.resolve_automatic_overpopulation()?;
        state.validate().map_err(RuleError::Invariant)?;
        Ok(state)
    }

    pub fn config(&self) -> GameConfig {
        self.config
    }

    pub fn seed(&self) -> GameSeed {
        self.seed
    }

    pub fn boards(&self) -> &[Board] {
        &self.boards
    }

    pub fn market(&self) -> &Market {
        &self.market
    }

    pub fn current_player(&self) -> usize {
        usize::from(self.current_player)
    }

    pub fn completed_turns(&self) -> u16 {
        self.completed_turns
    }

    pub fn total_turns(&self) -> u16 {
        self.config.total_turns()
    }

    pub fn turns_remaining(&self) -> u16 {
        self.total_turns() - self.completed_turns
    }

    pub fn turns_remaining_for_player(&self, player: usize) -> u16 {
        turns_remaining_for_player(self.completed_turns, self.boards.len(), player)
    }

    pub fn unplaced_wildlife_counts(&self) -> [u8; 5] {
        unplaced_wildlife_counts(&self.boards)
    }

    pub fn public_state(&self) -> PublicGameState {
        PublicGameState {
            config: self.config,
            boards: self.boards.clone(),
            market: self.market.clone(),
            current_player: self.current_player,
            completed_turns: self.completed_turns,
        }
    }

    pub fn public_supply(&self) -> PublicSupply {
        let mut supply = PublicSupply {
            wildlife_bag: [0; 5],
            unseen_tile_terrain_capacity: [0; 5],
            unseen_tile_wildlife_capacity: [0; 5],
            unseen_keystones_by_terrain: [0; 5],
            unseen_dual_terrain_pairs: [0; 10],
        };
        for wildlife in &self.wildlife_bag {
            supply.wildlife_bag[*wildlife as usize] += 1;
        }
        for tile in self.tile_stack.iter().chain(&self.excluded_tiles) {
            for terrain in Terrain::ALL {
                if tile.contains_terrain(terrain) {
                    supply.unseen_tile_terrain_capacity[terrain as usize] += 1;
                }
            }
            for wildlife in Wildlife::ALL {
                if tile.wildlife.contains(wildlife) {
                    supply.unseen_tile_wildlife_capacity[wildlife as usize] += 1;
                }
            }
            if tile.keystone {
                supply.unseen_keystones_by_terrain[tile.terrain_a as usize] += 1;
            } else if let Some(terrain_b) = tile.terrain_b {
                let index = terrain_pair_index(tile.terrain_a, terrain_b);
                supply.unseen_dual_terrain_pairs[index] += 1;
            }
        }
        supply
    }

    pub fn is_game_over(&self) -> bool {
        self.completed_turns == self.total_turns()
    }

    pub fn transition(&self, action: &TurnAction) -> Result<Self, RuleError> {
        let mut next = self.clone();
        next.apply_in_place(action)?;
        next.validate().map_err(RuleError::Invariant)?;
        Ok(next)
    }

    pub fn apply(&mut self, action: &TurnAction) -> Result<(), RuleError> {
        *self = self.transition(action)?;
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        postcard::to_allocvec(self).expect("serializing an in-memory game state cannot fail")
    }

    pub fn canonical_hash(&self) -> blake3::Hash {
        blake3::hash(&self.canonical_bytes())
    }

    pub fn redeterminize_hidden(&mut self, determinization_seed: GameSeed) {
        let tile_stack_len = self.tile_stack.len();
        let mut unseen_tiles = std::mem::take(&mut self.tile_stack);
        unseen_tiles.append(&mut self.excluded_tiles);
        unseen_tiles.sort_by_key(|tile| tile.id.0);
        unseen_tiles.shuffle(&mut determinization_seed.rng(b"hidden-tiles"));
        self.excluded_tiles = unseen_tiles.split_off(tile_stack_len);
        self.tile_stack = unseen_tiles;
        self.wildlife_bag.sort_by_key(|wildlife| *wildlife as u8);
        self.wildlife_bag
            .shuffle(&mut determinization_seed.rng(b"hidden-wildlife"));
    }

    pub fn legal_wildlife_wipes(&self) -> Vec<WildlifeWipe> {
        if self.is_game_over() || self.boards[self.current_player()].nature_tokens() == 0 {
            return Vec::new();
        }
        (1u8..16)
            .map(|mask| WildlifeWipe {
                slots: MarketSlot::ALL
                    .into_iter()
                    .filter(|slot| mask & (1 << slot.index()) != 0)
                    .collect(),
            })
            .collect()
    }

    pub fn legal_turn_actions(
        &self,
        prelude: &MarketPrelude,
    ) -> Result<Vec<TurnAction>, RuleError> {
        Ok(self
            .evaluate_legal_turn_actions(prelude, |_| ())?
            .into_iter()
            .map(|(action, ())| action)
            .collect())
    }

    pub fn legal_turn_actions_for_draft(
        &self,
        prelude: &MarketPrelude,
        draft: DraftChoice,
    ) -> Result<Vec<TurnAction>, RuleError> {
        Ok(self
            .evaluate_legal_draft_actions(prelude, draft, |_| ())?
            .into_iter()
            .map(|(action, ())| action)
            .collect())
    }

    pub fn evaluate_legal_turn_actions<T>(
        &self,
        prelude: &MarketPrelude,
        mut evaluate: impl FnMut(&Board) -> T,
    ) -> Result<Vec<(TurnAction, T)>, RuleError> {
        self.evaluate_legal_turn_actions_with_context(prelude, |board, _, _, _| evaluate(board))
    }

    pub fn evaluate_legal_turn_actions_with_context<T>(
        &self,
        prelude: &MarketPrelude,
        mut evaluate: impl FnMut(&Board, TilePlacement, Tile, Option<Wildlife>) -> T,
    ) -> Result<Vec<(TurnAction, T)>, RuleError> {
        self.evaluate_legal_turn_actions_with_tile_context(
            prelude,
            |_, placement, tile| (placement, tile),
            |board, &(placement, tile), placed_wildlife| {
                evaluate(
                    board,
                    placement,
                    tile,
                    placed_wildlife.map(|(wildlife, _)| wildlife),
                )
            },
        )
    }

    pub fn evaluate_legal_turn_actions_with_tile_context<C, T>(
        &self,
        prelude: &MarketPrelude,
        mut prepare_tile: impl FnMut(&Board, TilePlacement, Tile) -> C,
        mut evaluate: impl FnMut(&Board, &C, Option<(Wildlife, HexCoord)>) -> T,
    ) -> Result<Vec<(TurnAction, T)>, RuleError> {
        if self.is_game_over() {
            return Ok(Vec::new());
        }

        let mut staged = self.clone();
        staged.apply_market_prelude(prelude)?;
        let mut drafts: Vec<_> = MarketSlot::ALL
            .into_iter()
            .filter(|slot| staged.market.paired(*slot).is_some())
            .map(|slot| DraftChoice::Paired { slot })
            .collect();
        if staged.boards[staged.current_player()].nature_tokens() > 0 {
            for tile_slot in MarketSlot::ALL {
                for wildlife_slot in MarketSlot::ALL {
                    if staged.market.tiles[tile_slot.index()].is_some()
                        && staged.market.wildlife[wildlife_slot.index()].is_some()
                    {
                        drafts.push(DraftChoice::Independent {
                            tile_slot,
                            wildlife_slot,
                        });
                    }
                }
            }
        }

        staged.evaluate_staged_drafts_with_tile_context(
            prelude,
            &drafts,
            &mut prepare_tile,
            &mut evaluate,
        )
    }

    pub fn evaluate_legal_draft_actions<T>(
        &self,
        prelude: &MarketPrelude,
        draft: DraftChoice,
        mut evaluate: impl FnMut(&Board) -> T,
    ) -> Result<Vec<(TurnAction, T)>, RuleError> {
        if self.is_game_over() {
            return Ok(Vec::new());
        }

        let mut staged = self.clone();
        staged.apply_market_prelude(prelude)?;
        staged.preview_draft(draft)?;
        staged.evaluate_staged_drafts_with_tile_context(
            prelude,
            &[draft],
            &mut |_, _, _| (),
            &mut |board, &(), _| evaluate(board),
        )
    }

    pub fn preview_market_prelude(&self, prelude: &MarketPrelude) -> Result<Self, RuleError> {
        if self.is_game_over() {
            return Err(RuleError::GameOver);
        }
        let mut staged = self.clone();
        staged.apply_market_prelude(prelude)?;
        staged.validate().map_err(RuleError::Invariant)?;
        Ok(staged)
    }

    pub fn preview_free_three_of_a_kind_if_feasible(
        &self,
    ) -> Result<(MarketPrelude, Self), RuleError> {
        let prelude = MarketPrelude {
            replace_three_of_a_kind: self.market.three_of_a_kind().is_some(),
            wildlife_wipes: Vec::new(),
        };
        if !prelude.replace_three_of_a_kind {
            return Ok((prelude, self.clone()));
        }
        match self.preview_market_prelude(&prelude) {
            Ok(staged) => Ok((prelude, staged)),
            Err(RuleError::WildlifeBagEmpty) => Ok((MarketPrelude::default(), self.clone())),
            Err(error) => Err(error),
        }
    }

    fn evaluate_staged_drafts_with_tile_context<C, T>(
        &self,
        prelude: &MarketPrelude,
        drafts: &[DraftChoice],
        prepare_tile: &mut impl FnMut(&Board, TilePlacement, Tile) -> C,
        evaluate: &mut impl FnMut(&Board, &C, Option<(Wildlife, HexCoord)>) -> T,
    ) -> Result<Vec<(TurnAction, T)>, RuleError> {
        let player = self.current_player();
        let mut board = self.boards[player].clone();
        let frontier = board.frontier();
        let wildlife_placements: [_; 5] =
            std::array::from_fn(|index| board.wildlife_placements(Wildlife::ALL[index]));
        let mut evaluated = Vec::new();
        for &draft in drafts {
            let (tile, wildlife) = self.preview_draft(draft)?;
            let independent = matches!(draft, DraftChoice::Independent { .. });
            if independent && !board.spend_nature_token() {
                return Err(RuleError::NoNatureTokens);
            }
            let rotations = if tile.terrain_b.is_some() {
                &Rotation::ALL[..]
            } else {
                &Rotation::ALL[..1]
            };
            for coord in &frontier {
                for rotation in rotations {
                    let tile_delta = board.place_tile(*coord, tile, *rotation)?;
                    let mut valid_wildlife_placements =
                        wildlife_placements[wildlife as usize].clone();
                    if tile.wildlife.contains(wildlife) {
                        valid_wildlife_placements.push(*coord);
                    }
                    debug_assert_eq!(
                        valid_wildlife_placements,
                        board.wildlife_placements(wildlife)
                    );
                    let tile_placement = TilePlacement {
                        coord: *coord,
                        rotation: *rotation,
                    };
                    let tile_context = prepare_tile(&board, tile_placement, tile);
                    evaluated.push((
                        TurnAction {
                            replace_three_of_a_kind: prelude.replace_three_of_a_kind,
                            wildlife_wipes: prelude.wildlife_wipes.clone(),
                            draft,
                            tile: tile_placement,
                            wildlife: None,
                        },
                        evaluate(&board, &tile_context, None),
                    ));
                    for wildlife_coord in valid_wildlife_placements {
                        let wildlife_delta = board.place_wildlife(wildlife_coord, wildlife)?;
                        evaluated.push((
                            TurnAction {
                                replace_three_of_a_kind: prelude.replace_three_of_a_kind,
                                wildlife_wipes: prelude.wildlife_wipes.clone(),
                                draft,
                                tile: tile_placement,
                                wildlife: Some(wildlife_coord),
                            },
                            evaluate(&board, &tile_context, Some((wildlife, wildlife_coord))),
                        ));
                        board.undo(wildlife_delta)?;
                    }
                    board.undo(tile_delta)?;
                }
            }
            if independent {
                board.refund_nature_token();
            }
        }
        Ok(evaluated)
    }

    pub fn preview_active_board(&self, action: &TurnAction) -> Result<Board, RuleError> {
        if self.is_game_over() {
            return Err(RuleError::GameOver);
        }
        if action.replace_three_of_a_kind || !action.wildlife_wipes.is_empty() {
            let mut staged = self.clone();
            staged.apply_market_prelude(&action.prelude())?;
            staged.preview_board_after_prelude(action)
        } else {
            self.preview_board_after_prelude(action)
        }
    }

    pub fn preview_public_afterstate(
        &self,
        action: &TurnAction,
    ) -> Result<PublicGameState, RuleError> {
        if self.is_game_over() {
            return Err(RuleError::GameOver);
        }

        let mut staged = self.clone();
        staged.apply_market_prelude(&action.prelude())?;

        let mut placement = action.clone();
        placement.replace_three_of_a_kind = false;
        placement.wildlife_wipes.clear();
        let board = staged.preview_board_after_prelude(&placement)?;

        let mut market = staged.market.clone();
        match placement.draft {
            DraftChoice::Paired { slot } => {
                market
                    .take_paired(slot)
                    .ok_or(RuleError::UnavailableMarketSlot(slot))?;
            }
            DraftChoice::Independent {
                tile_slot,
                wildlife_slot,
            } => {
                market.take_independent(tile_slot, wildlife_slot).ok_or(
                    RuleError::UnavailableIndependentDraft {
                        tile_slot,
                        wildlife_slot,
                    },
                )?;
            }
        }

        let mut boards = staged.boards.clone();
        boards[staged.current_player()] = board;
        let completed_turns = staged.completed_turns + 1;
        let current_player =
            ((staged.current_player() + 1) % usize::from(staged.config.player_count)) as u8;
        Ok(PublicGameState {
            config: staged.config,
            boards,
            market,
            current_player,
            completed_turns,
        })
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.schema_version != STATE_SCHEMA_VERSION {
            return Err("unsupported game state schema");
        }
        if self.boards.len() != usize::from(self.config.player_count) {
            return Err("board count does not match player count");
        }
        if self.completed_turns > self.total_turns() {
            return Err("completed turns exceed game length");
        }
        if usize::from(self.current_player) >= self.boards.len() {
            return Err("current player is out of range");
        }
        for board in &self.boards {
            board.validate()?;
        }
        self.market.validate(self.is_game_over())?;

        let standard_tiles_on_boards = self
            .boards
            .iter()
            .flat_map(Board::placed_tiles)
            .filter(|(_, placed)| placed.tile.id.0 < 85)
            .count();
        let tile_total = self.tile_stack.len()
            + self.excluded_tiles.len()
            + self.discarded_tiles.len()
            + self.market.tiles.iter().flatten().count()
            + standard_tiles_on_boards;
        if tile_total != 85 {
            return Err("habitat tile conservation failed");
        }

        let placed_wildlife = self
            .boards
            .iter()
            .flat_map(Board::placed_tiles)
            .filter(|(_, placed)| placed.wildlife.is_some())
            .count();
        let wildlife_total = self.wildlife_bag.len()
            + self.discarded_wildlife.len()
            + self.market.wildlife.iter().flatten().count()
            + placed_wildlife;
        if wildlife_total != 100 {
            return Err("wildlife token conservation failed");
        }
        Ok(())
    }

    fn apply_in_place(&mut self, action: &TurnAction) -> Result<(), RuleError> {
        if self.is_game_over() {
            return Err(RuleError::GameOver);
        }
        let player = self.current_player();

        self.apply_market_prelude(&action.prelude())?;

        let (tile, wildlife) = match action.draft {
            DraftChoice::Paired { slot } => self
                .market
                .take_paired(slot)
                .ok_or(RuleError::UnavailableMarketSlot(slot))?,
            DraftChoice::Independent {
                tile_slot,
                wildlife_slot,
            } => {
                if !self.boards[player].spend_nature_token() {
                    return Err(RuleError::NoNatureTokens);
                }
                self.market
                    .take_independent(tile_slot, wildlife_slot)
                    .ok_or(RuleError::UnavailableIndependentDraft {
                        tile_slot,
                        wildlife_slot,
                    })?
            }
        };

        self.boards[player].place_tile(action.tile.coord, tile, action.tile.rotation)?;
        if let Some(coord) = action.wildlife {
            self.boards[player].place_wildlife(coord, wildlife)?;
        } else {
            self.return_wildlife(wildlife);
        }

        self.finish_turn()?;
        Ok(())
    }

    fn apply_market_prelude(&mut self, prelude: &MarketPrelude) -> Result<(), RuleError> {
        let player = self.current_player();
        if prelude.replace_three_of_a_kind {
            let wildlife = self
                .market
                .three_of_a_kind()
                .ok_or(RuleError::NoThreeOfAKind)?;
            let slots = self.market.wildlife_slots(wildlife);
            self.replace_wildlife(&slots)?;
        }

        for wipe in &prelude.wildlife_wipes {
            validate_wipe_slots(&wipe.slots)?;
            if !self.boards[player].spend_nature_token() {
                return Err(RuleError::NoNatureTokens);
            }
            self.replace_wildlife(&wipe.slots)?;
        }
        Ok(())
    }

    fn preview_draft(&self, draft: DraftChoice) -> Result<(Tile, Wildlife), RuleError> {
        match draft {
            DraftChoice::Paired { slot } => self
                .market
                .paired(slot)
                .ok_or(RuleError::UnavailableMarketSlot(slot)),
            DraftChoice::Independent {
                tile_slot,
                wildlife_slot,
            } => {
                if self.boards[self.current_player()].nature_tokens() == 0 {
                    return Err(RuleError::NoNatureTokens);
                }
                match (
                    self.market.tiles[tile_slot.index()],
                    self.market.wildlife[wildlife_slot.index()],
                ) {
                    (Some(tile), Some(wildlife)) => Ok((tile, wildlife)),
                    _ => Err(RuleError::UnavailableIndependentDraft {
                        tile_slot,
                        wildlife_slot,
                    }),
                }
            }
        }
    }

    fn preview_board_after_prelude(&self, action: &TurnAction) -> Result<Board, RuleError> {
        let player = self.current_player();
        let (tile, wildlife) = self.preview_draft(action.draft)?;
        let mut board = self.boards[player].clone();
        if matches!(action.draft, DraftChoice::Independent { .. }) && !board.spend_nature_token() {
            return Err(RuleError::NoNatureTokens);
        }
        board.place_tile(action.tile.coord, tile, action.tile.rotation)?;
        if let Some(coord) = action.wildlife {
            board.place_wildlife(coord, wildlife)?;
        }
        Ok(board)
    }

    fn replace_wildlife(&mut self, slots: &[MarketSlot]) -> Result<(), RuleError> {
        let mut set_aside = Vec::with_capacity(slots.len() + 4);
        for slot in slots {
            let wildlife = self.market.wildlife[slot.index()]
                .take()
                .ok_or(RuleError::UnavailableMarketSlot(*slot))?;
            set_aside.push(wildlife);
        }
        self.fill_empty_wildlife()?;

        let mut automatic_wipes = 0;
        while self.market.four_of_a_kind().is_some() {
            automatic_wipes += 1;
            if automatic_wipes > 100 {
                return Err(RuleError::AutomaticOverpopulationDidNotResolve);
            }
            for slot in MarketSlot::ALL {
                set_aside.push(
                    self.market.wildlife[slot.index()]
                        .take()
                        .expect("four-of-a-kind market is full"),
                );
            }
            self.fill_empty_wildlife()?;
        }

        for wildlife in set_aside {
            self.return_wildlife(wildlife);
        }
        Ok(())
    }

    fn resolve_automatic_overpopulation(&mut self) -> Result<(), RuleError> {
        if self.market.four_of_a_kind().is_some() {
            self.replace_wildlife(&MarketSlot::ALL)?;
        }
        Ok(())
    }

    fn fill_empty_wildlife(&mut self) -> Result<(), RuleError> {
        for slot in MarketSlot::ALL {
            if self.market.wildlife[slot.index()].is_none() {
                self.market.wildlife[slot.index()] =
                    Some(self.wildlife_bag.pop().ok_or(RuleError::WildlifeBagEmpty)?);
            }
        }
        Ok(())
    }

    fn return_wildlife(&mut self, wildlife: Wildlife) {
        let mut hasher = Hasher::new();
        hasher.update(b"cascadia-v2-wildlife-return");
        hasher.update(&self.seed.0);
        hasher.update(&self.wildlife_return_counter.to_le_bytes());
        let bytes = hasher.finalize();
        let mut position_bytes = [0u8; 8];
        position_bytes.copy_from_slice(&bytes.as_bytes()[..8]);
        let position =
            (u64::from_le_bytes(position_bytes) % (self.wildlife_bag.len() as u64 + 1)) as usize;
        self.wildlife_bag.insert(position, wildlife);
        self.wildlife_return_counter += 1;
    }

    fn finish_turn(&mut self) -> Result<(), RuleError> {
        self.completed_turns += 1;
        self.current_player = ((usize::from(self.current_player) + 1) % self.boards.len()) as u8;

        if self.is_game_over() {
            return Ok(());
        }

        match self.config.mode {
            GameMode::Standard => self.refill_standard_market()?,
            GameMode::Solo => self.refill_solo_market()?,
        }
        self.resolve_automatic_overpopulation()
    }

    fn refill_standard_market(&mut self) -> Result<(), RuleError> {
        for slot in MarketSlot::ALL {
            if self.market.tiles[slot.index()].is_none() {
                self.market.tiles[slot.index()] =
                    Some(self.tile_stack.pop().ok_or(RuleError::TileStackEmpty)?);
            }
            if self.market.wildlife[slot.index()].is_none() {
                self.market.wildlife[slot.index()] =
                    Some(self.wildlife_bag.pop().ok_or(RuleError::WildlifeBagEmpty)?);
            }
        }
        Ok(())
    }

    fn refill_solo_market(&mut self) -> Result<(), RuleError> {
        let tile_to_discard = self
            .market
            .tiles
            .iter()
            .rposition(Option::is_some)
            .ok_or(RuleError::TileStackEmpty)?;
        self.discarded_tiles
            .push(self.market.tiles[tile_to_discard].take().unwrap());

        let wildlife_to_discard = self
            .market
            .wildlife
            .iter()
            .rposition(Option::is_some)
            .ok_or(RuleError::WildlifeBagEmpty)?;
        self.discarded_wildlife
            .push(self.market.wildlife[wildlife_to_discard].take().unwrap());

        self.market.compact_away_from_draw_stack();
        for slot in MarketSlot::ALL {
            if self.market.tiles[slot.index()].is_none() {
                self.market.tiles[slot.index()] =
                    Some(self.tile_stack.pop().ok_or(RuleError::TileStackEmpty)?);
            }
            if self.market.wildlife[slot.index()].is_none() {
                self.market.wildlife[slot.index()] =
                    Some(self.wildlife_bag.pop().ok_or(RuleError::WildlifeBagEmpty)?);
            }
        }
        Ok(())
    }
}

fn turns_remaining_for_player(completed_turns: u16, player_count: usize, player: usize) -> u16 {
    if player >= player_count {
        return 0;
    }
    let player_count = player_count as u16;
    let completed_rounds = completed_turns / player_count;
    let completed_in_round = usize::from(completed_turns % player_count);
    let completed_by_player = completed_rounds + u16::from(player < completed_in_round);
    20u16.saturating_sub(completed_by_player)
}

fn unplaced_wildlife_counts(boards: &[Board]) -> [u8; 5] {
    let mut counts = [20u8; 5];
    for board in boards {
        for wildlife in Wildlife::ALL {
            counts[wildlife as usize] = counts[wildlife as usize]
                .saturating_sub(board.wildlife_positions(wildlife).len() as u8);
        }
    }
    counts
}

fn terrain_pair_index(left: Terrain, right: Terrain) -> usize {
    let (low, high) = if (left as u8) < (right as u8) {
        (left as usize, right as usize)
    } else {
        (right as usize, left as usize)
    };
    let mut index = 0;
    for first in 0..5 {
        for second in first + 1..5 {
            if first == low && second == high {
                return index;
            }
            index += 1;
        }
    }
    unreachable!("dual terrain tile must contain two distinct terrains")
}

fn validate_wipe_slots(slots: &[MarketSlot]) -> Result<(), RuleError> {
    if slots.is_empty() {
        return Err(RuleError::EmptyWildlifeWipe);
    }
    let mut seen = [false; 4];
    for slot in slots {
        if std::mem::replace(&mut seen[slot.index()], true) {
            return Err(RuleError::DuplicateWildlifeWipeSlot(*slot));
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RuleError {
    #[error("player count {0} is invalid for the selected game mode")]
    InvalidPlayerCount(u8),
    #[error("the game is already over")]
    GameOver,
    #[error("the market does not contain exactly three matching wildlife tokens")]
    NoThreeOfAKind,
    #[error("a paid wildlife wipe must select at least one slot")]
    EmptyWildlifeWipe,
    #[error("wildlife wipe repeats market slot {0:?}")]
    DuplicateWildlifeWipeSlot(MarketSlot),
    #[error("market slot {0:?} is unavailable")]
    UnavailableMarketSlot(MarketSlot),
    #[error("independent draft cannot take tile {tile_slot:?} and wildlife {wildlife_slot:?}")]
    UnavailableIndependentDraft {
        tile_slot: MarketSlot,
        wildlife_slot: MarketSlot,
    },
    #[error("the active player does not have enough nature tokens")]
    NoNatureTokens,
    #[error("the habitat tile stack is unexpectedly empty")]
    TileStackEmpty,
    #[error("the wildlife bag is unexpectedly empty")]
    WildlifeBagEmpty,
    #[error("automatic four-token overpopulation did not resolve")]
    AutomaticOverpopulationDidNotResolve,
    #[error(transparent)]
    Board(#[from] BoardError),
    #[error("game invariant failed: {0}")]
    Invariant(&'static str),
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    fn game(players: u8, seed: u64) -> GameState {
        GameState::new(
            GameConfig::research_aaaaa(players).unwrap(),
            GameSeed::from_u64(seed),
        )
        .unwrap()
    }

    fn first_legal_skip_action(game: &GameState) -> TurnAction {
        TurnAction::paired(
            MarketSlot::ZERO,
            game.boards[game.current_player()].frontier()[0],
            Rotation::ZERO,
        )
    }

    fn force_market_wildlife(game: &mut GameState, wildlife: [Wildlife; 4]) {
        for slot in MarketSlot::ALL {
            game.wildlife_bag
                .push(game.market.wildlife[slot.index()].take().unwrap());
        }
        for (slot, desired) in MarketSlot::ALL.into_iter().zip(wildlife) {
            let index = game
                .wildlife_bag
                .iter()
                .position(|candidate| *candidate == desired)
                .unwrap();
            game.market.wildlife[slot.index()] = Some(game.wildlife_bag.swap_remove(index));
        }
    }

    #[test]
    fn public_afterstate_never_exposes_the_hidden_refill() {
        let game = game(4, 812);
        let action = game
            .legal_turn_actions(&MarketPrelude::default())
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        let DraftChoice::Paired { slot } = action.draft else {
            panic!("the first default action uses a paired draft");
        };

        let mut redetermined = game.clone();
        redetermined.redeterminize_hidden(GameSeed::from_u64(813));
        let left = game.preview_public_afterstate(&action).unwrap();
        let right = redetermined.preview_public_afterstate(&action).unwrap();

        assert_eq!(left, right);
        assert_eq!(left.completed_turns(), game.completed_turns() + 1);
        assert_eq!(left.current_player(), 1);
        assert_eq!(
            left.boards()[0].tile_count(),
            game.boards()[0].tile_count() + 1
        );
        assert_eq!(left.market().tiles[slot.index()], None);
        assert_eq!(left.market().wildlife[slot.index()], None);
    }

    #[test]
    fn independent_public_afterstate_preserves_unchosen_market_components() {
        let mut game = game(4, 814);
        game.boards[0].grant_nature_tokens(1);
        let tile_slot = MarketSlot::ZERO;
        let wildlife_slot = MarketSlot::ONE;
        let action = game
            .legal_turn_actions_for_draft(
                &MarketPrelude::default(),
                DraftChoice::Independent {
                    tile_slot,
                    wildlife_slot,
                },
            )
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        let original_tile = game.market.tiles[wildlife_slot.index()];
        let original_wildlife = game.market.wildlife[tile_slot.index()];

        let afterstate = game.preview_public_afterstate(&action).unwrap();

        assert_eq!(afterstate.market().tiles[tile_slot.index()], None);
        assert_eq!(
            afterstate.market().wildlife[tile_slot.index()],
            original_wildlife
        );
        assert_eq!(
            afterstate.market().tiles[wildlife_slot.index()],
            original_tile
        );
        assert_eq!(afterstate.market().wildlife[wildlife_slot.index()], None);
        assert_eq!(afterstate.boards()[0].nature_tokens(), 0);
    }

    #[test]
    fn setup_uses_official_stack_sizes_and_starter_counts() {
        for (players, expected_stack_after_market) in [(2, 39), (3, 59), (4, 79)] {
            let game = game(players, 7);
            assert_eq!(game.tile_stack.len(), expected_stack_after_market);
            assert_eq!(game.boards.len(), usize::from(players));
            assert!(game.boards.iter().all(|board| board.tile_count() == 3));
            game.validate().unwrap();
        }
    }

    #[test]
    fn seeded_setup_is_reproducible_and_domain_separated() {
        let left = game(4, 123);
        let right = game(4, 123);
        let different = game(4, 124);

        assert_eq!(left.canonical_hash(), right.canonical_hash());
        assert_ne!(left.canonical_hash(), different.canonical_hash());
    }

    #[test]
    fn hidden_redeterminization_preserves_every_public_game_fact() {
        let mut game = game(4, 124);
        let original = game.clone();
        let original_public = game.public_state();
        let original_public_hash = original_public.canonical_hash();
        let original_supply = game.public_supply();
        let original_legal = game.legal_turn_actions(&MarketPrelude::default()).unwrap();

        game.redeterminize_hidden(GameSeed::from_u64(9001));

        assert_eq!(game.public_state(), original_public);
        assert_eq!(game.public_state().canonical_hash(), original_public_hash);
        assert_eq!(game.config, original.config);
        assert_eq!(game.boards, original.boards);
        assert_eq!(game.market, original.market);
        assert_eq!(game.current_player, original.current_player);
        assert_eq!(game.completed_turns, original.completed_turns);
        assert_eq!(game.discarded_tiles, original.discarded_tiles);
        assert_eq!(game.discarded_wildlife, original.discarded_wildlife);
        assert_eq!(game.public_supply(), original_supply);
        assert_eq!(
            game.legal_turn_actions(&MarketPrelude::default()).unwrap(),
            original_legal
        );
        game.validate().unwrap();
        assert_ne!(game.canonical_hash(), original.canonical_hash());
    }

    #[test]
    fn public_supply_counts_only_publicly_unseen_resources() {
        let game = game(4, 901);
        let supply = game.public_supply();

        assert_eq!(supply.wildlife_bag.iter().sum::<u8>(), 96);
        assert_eq!(
            supply.unseen_keystones_by_terrain.iter().sum::<u8>()
                + supply.unseen_dual_terrain_pairs.iter().sum::<u8>(),
            81
        );
        assert!(
            supply
                .unseen_tile_terrain_capacity
                .iter()
                .all(|count| *count > 0)
        );
        assert!(
            supply
                .unseen_tile_wildlife_capacity
                .iter()
                .all(|count| *count > 0)
        );
    }

    #[test]
    fn hidden_redeterminization_is_seeded_and_reproducible() {
        let original = game(4, 125);
        let mut left = original.clone();
        let mut right = original.clone();
        let mut different = original.clone();
        let mut previously_redetermined = original;

        left.redeterminize_hidden(GameSeed::from_u64(77));
        right.redeterminize_hidden(GameSeed::from_u64(77));
        different.redeterminize_hidden(GameSeed::from_u64(78));
        previously_redetermined.redeterminize_hidden(GameSeed::from_u64(79));
        previously_redetermined.redeterminize_hidden(GameSeed::from_u64(77));

        assert_eq!(left.canonical_hash(), right.canonical_hash());
        assert_eq!(
            left.canonical_hash(),
            previously_redetermined.canonical_hash()
        );
        assert_ne!(left.canonical_hash(), different.canonical_hash());
    }

    #[test]
    fn public_supply_and_per_player_turn_counts_track_completed_turns() {
        let mut game = game(4, 126);
        assert_eq!(game.unplaced_wildlife_counts(), [20; 5]);
        assert_eq!(
            (0..4)
                .map(|player| game.turns_remaining_for_player(player))
                .collect::<Vec<_>>(),
            vec![20, 20, 20, 20]
        );

        let action = game
            .legal_turn_actions(&MarketPrelude::default())
            .unwrap()
            .into_iter()
            .find(|action| action.wildlife.is_some())
            .expect("initial market has a legal wildlife placement");
        game.apply(&action).unwrap();

        assert_eq!(game.unplaced_wildlife_counts().iter().sum::<u8>(), 99);
        assert_eq!(
            (0..4)
                .map(|player| game.turns_remaining_for_player(player))
                .collect::<Vec<_>>(),
            vec![19, 20, 20, 20]
        );
        assert_eq!(game.turns_remaining_for_player(4), 0);
    }

    #[test]
    fn invalid_turn_is_transactional() {
        let mut game = game(4, 1);
        let before = game.canonical_hash();
        let action = TurnAction::paired(MarketSlot::ZERO, HexCoord::new(20, 20), Rotation::ZERO);

        assert!(matches!(
            game.apply(&action),
            Err(RuleError::Board(BoardError::Detached(_)))
        ));
        assert_eq!(game.canonical_hash(), before);
    }

    #[test]
    fn valid_turn_advances_without_losing_components() {
        let mut game = game(4, 2);
        let action = first_legal_skip_action(&game);
        game.apply(&action).unwrap();

        assert_eq!(game.completed_turns(), 1);
        assert_eq!(game.current_player(), 1);
        game.validate().unwrap();
    }

    #[test]
    fn paid_wipe_accepts_any_nonempty_subset_and_charges_one_token() {
        let mut game = game(2, 3);
        game.boards[0].grant_nature_tokens(2);
        let before_tokens = game.boards[0].nature_tokens();
        let mut action = first_legal_skip_action(&game);
        action.wildlife_wipes.push(WildlifeWipe {
            slots: vec![MarketSlot::ONE, MarketSlot::THREE],
        });

        game.apply(&action).unwrap();
        assert_eq!(game.boards[0].nature_tokens(), before_tokens - 1);
    }

    #[test]
    fn duplicate_paid_wipe_slots_are_rejected_transactionally() {
        let mut game = game(2, 4);
        game.boards[0].grant_nature_tokens(1);
        let before = game.canonical_hash();
        let mut action = first_legal_skip_action(&game);
        action.wildlife_wipes.push(WildlifeWipe {
            slots: vec![MarketSlot::ONE, MarketSlot::ONE],
        });

        assert_eq!(
            game.apply(&action),
            Err(RuleError::DuplicateWildlifeWipeSlot(MarketSlot::ONE))
        );
        assert_eq!(game.canonical_hash(), before);
    }

    #[test]
    fn independent_draft_costs_one_token() {
        let mut game = game(2, 5);
        game.boards[0].grant_nature_tokens(1);
        let preserved_tile = game.market.tiles[1];
        let preserved_wildlife = game.market.wildlife[0];
        let mut action = first_legal_skip_action(&game);
        action.draft = DraftChoice::Independent {
            tile_slot: MarketSlot::ZERO,
            wildlife_slot: MarketSlot::ONE,
        };

        game.apply(&action).unwrap();
        assert_eq!(game.boards[0].nature_tokens(), 0);
        assert_eq!(game.market.tiles[1], preserved_tile);
        assert_eq!(game.market.wildlife[0], preserved_wildlife);
    }

    #[test]
    fn same_slot_independent_draft_is_generated_and_costs_one_token() {
        let mut game = game(2, 5_001);
        game.boards[0].grant_nature_tokens(1);
        let prelude = MarketPrelude::default();
        let actions = game.legal_turn_actions(&prelude).unwrap();
        let action = actions
            .into_iter()
            .find(|action| {
                matches!(
                    action.draft,
                    DraftChoice::Independent {
                        tile_slot,
                        wildlife_slot,
                    } if tile_slot == wildlife_slot
                )
            })
            .expect("the official any-tile, any-token ability includes the same slot");
        assert!(matches!(
            action.draft,
            DraftChoice::Independent {
                tile_slot,
                wildlife_slot,
            } if tile_slot == wildlife_slot
        ));
        let before_tokens = game.boards[0].nature_tokens();
        game.apply(&action).unwrap();

        assert_eq!(game.boards[0].nature_tokens(), before_tokens - 1);
    }

    #[test]
    fn solo_turn_discards_one_tile_and_wildlife_and_draws_two() {
        let config = GameConfig::solo(ScoringCards::AAAAA);
        let mut game = GameState::new(config, GameSeed::from_u64(6)).unwrap();
        let before_tiles = game.tile_stack.len();
        let action = first_legal_skip_action(&game);
        game.apply(&action).unwrap();

        assert_eq!(game.tile_stack.len(), before_tiles - 2);
        assert_eq!(game.discarded_tiles.len(), 1);
        assert_eq!(game.discarded_wildlife.len(), 1);
        game.validate().unwrap();
    }

    #[test]
    fn state_round_trips_through_json_and_postcard() {
        let game = game(4, 99);
        let json = serde_json::to_string(&game).unwrap();
        let from_json: GameState = serde_json::from_str(&json).unwrap();
        let bytes = game.canonical_bytes();
        let from_postcard: GameState = postcard::from_bytes(&bytes).unwrap();

        assert_eq!(from_json, game);
        assert_eq!(from_postcard, game);
        assert_eq!(from_postcard.canonical_hash(), game.canonical_hash());
    }

    #[test]
    fn automatic_four_token_overpopulation_is_never_a_stable_state() {
        let mut game = game(2, 101);
        force_market_wildlife(&mut game, [Wildlife::Bear; 4]);
        assert_eq!(game.market.four_of_a_kind(), Some(Wildlife::Bear));

        game.resolve_automatic_overpopulation().unwrap();

        assert!(game.market.four_of_a_kind().is_none());
        game.validate().unwrap();
    }

    #[test]
    fn optional_three_token_replacement_is_part_of_the_transaction() {
        let mut game = game(2, 102);
        force_market_wildlife(
            &mut game,
            [
                Wildlife::Bear,
                Wildlife::Bear,
                Wildlife::Bear,
                Wildlife::Elk,
            ],
        );
        let mut action = first_legal_skip_action(&game);
        action.replace_three_of_a_kind = true;

        game.apply(&action).unwrap();

        assert_eq!(game.completed_turns(), 1);
        game.validate().unwrap();
    }

    #[test]
    fn legal_generator_only_emits_accepted_complete_turns() {
        let game = game(2, 103);
        let actions = game.legal_turn_actions(&MarketPrelude::default()).unwrap();

        assert!(!actions.is_empty());
        for action in actions {
            game.transition(&action).unwrap();
        }
    }

    #[test]
    fn tile_context_is_prepared_once_per_draft_and_tile_placement() {
        let game = game(2, 106);
        let prepared = std::cell::Cell::new(0usize);
        let evaluations = game
            .evaluate_legal_turn_actions_with_tile_context(
                &MarketPrelude::default(),
                |_, _, _| {
                    let context = prepared.get();
                    prepared.set(context + 1);
                    context
                },
                |_, context, _| *context,
            )
            .unwrap();

        let mut groups = Vec::<(DraftChoice, TilePlacement, usize)>::new();
        for (action, context) in evaluations {
            if let Some((_, _, existing)) = groups
                .iter()
                .find(|(draft, tile, _)| *draft == action.draft && *tile == action.tile)
            {
                assert_eq!(*existing, context);
            } else {
                groups.push((action.draft, action.tile, context));
            }
        }
        assert_eq!(groups.len(), prepared.get());
    }

    #[test]
    fn fast_board_preview_matches_the_transactional_transition() {
        let game = game(2, 105);
        let actions = game.legal_turn_actions(&MarketPrelude::default()).unwrap();
        for action in actions.into_iter().take(100) {
            let preview = game.preview_active_board(&action).unwrap();
            let transitioned = game.transition(&action).unwrap();
            assert_eq!(&preview, &transitioned.boards[0]);
        }
    }

    #[test]
    fn legal_wildlife_wipes_cover_every_nonempty_slot_subset() {
        let mut game = game(2, 104);
        assert!(game.legal_wildlife_wipes().is_empty());
        game.boards[0].grant_nature_tokens(1);

        let wipes = game.legal_wildlife_wipes();
        assert_eq!(wipes.len(), 15);
        assert!(wipes.iter().all(|wipe| !wipe.slots.is_empty()));
    }

    #[test]
    fn complete_skip_games_preserve_invariants_for_every_player_count() {
        for players in 1..=4 {
            let config = if players == 1 {
                GameConfig::solo(ScoringCards::AAAAA)
            } else {
                GameConfig::research_aaaaa(players).unwrap()
            };
            let mut game =
                GameState::new(config, GameSeed::from_u64(200 + u64::from(players))).unwrap();
            while !game.is_game_over() {
                let action = first_legal_skip_action(&game);
                game.apply(&action).unwrap();
                game.validate().unwrap();
            }

            assert_eq!(game.completed_turns(), game.total_turns());
            assert_eq!(
                game.boards
                    .iter()
                    .map(Board::tile_count)
                    .collect::<Vec<_>>(),
                vec![23; usize::from(players)]
            );
            crate::score_game(&game);
        }
    }

    #[test]
    fn market_prelude_preview_does_not_advance_or_mutate_original_state() {
        let mut game = game(2, 106);
        game.boards[0].grant_nature_tokens(1);
        let before = game.canonical_hash();
        let prelude = MarketPrelude {
            replace_three_of_a_kind: false,
            wildlife_wipes: vec![WildlifeWipe {
                slots: vec![MarketSlot::ZERO],
            }],
        };

        let staged = game.preview_market_prelude(&prelude).unwrap();

        assert_eq!(game.canonical_hash(), before);
        assert_eq!(staged.completed_turns(), game.completed_turns());
        assert_eq!(
            staged.boards()[0].nature_tokens(),
            game.boards()[0].nature_tokens() - 1
        );
        assert_ne!(staged.canonical_hash(), before);
    }

    #[test]
    fn infeasible_optional_three_of_a_kind_replacement_is_declined() {
        let mut game = game(4, 108);
        force_market_wildlife(
            &mut game,
            [
                Wildlife::Bear,
                Wildlife::Bear,
                Wildlife::Bear,
                Wildlife::Elk,
            ],
        );
        let mut retained = Vec::new();
        for _ in 0..3 {
            let index = game
                .wildlife_bag
                .iter()
                .position(|wildlife| *wildlife == Wildlife::Elk)
                .unwrap();
            retained.push(game.wildlife_bag.swap_remove(index));
        }
        game.discarded_wildlife.append(&mut game.wildlife_bag);
        game.wildlife_bag = retained;
        game.validate().unwrap();

        let requested = MarketPrelude {
            replace_three_of_a_kind: true,
            wildlife_wipes: Vec::new(),
        };
        assert!(matches!(
            game.preview_market_prelude(&requested),
            Err(RuleError::WildlifeBagEmpty)
        ));

        let before = game.canonical_hash();
        let (prelude, staged) = game.preview_free_three_of_a_kind_if_feasible().unwrap();
        assert_eq!(prelude, MarketPrelude::default());
        assert_eq!(staged.canonical_hash(), before);
        staged.validate().unwrap();
    }

    #[test]
    fn draft_specific_actions_match_the_full_legal_action_subset() {
        let game = game(2, 107);
        let prelude = MarketPrelude::default();
        let draft = DraftChoice::Paired {
            slot: MarketSlot::TWO,
        };
        let expected: Vec<_> = game
            .legal_turn_actions(&prelude)
            .unwrap()
            .into_iter()
            .filter(|action| action.draft == draft)
            .collect();

        assert_eq!(
            game.legal_turn_actions_for_draft(&prelude, draft).unwrap(),
            expected
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(12))]

        #[test]
        fn seeded_skip_games_replay_to_the_identical_hash(
            seed in any::<u64>(),
            players in 1u8..=4,
        ) {
            let config = if players == 1 {
                GameConfig::solo(ScoringCards::AAAAA)
            } else {
                GameConfig::research_aaaaa(players).unwrap()
            };
            let game_seed = GameSeed::from_u64(seed);
            let mut game = GameState::new(config, game_seed).unwrap();
            let mut replay = crate::Replay::new(config, game_seed);
            while !game.is_game_over() {
                let action = first_legal_skip_action(&game);
                game.apply(&action).unwrap();
                replay.turns.push(action);
            }
            replay.final_state_hash = Some(*game.canonical_hash().as_bytes());

            let replayed = replay.play().unwrap();
            prop_assert_eq!(replayed.canonical_hash(), game.canonical_hash());
            prop_assert_eq!(replayed, game);
        }
    }
}
