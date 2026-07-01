mod common;

use cascadia_data::{ExactSemanticSupply, PositionRecord, TARGET_DIM};
use cascadia_game::{D6Transform, DraftChoice, HexCoord, MarketPrelude, MarketSlot, Rotation};
use common::{
    canonical_actions, game, representative_drafts, representative_dual_action,
    state_with_nature_token, state_with_nature_tokens,
};
use r2_sparse_entity_census::SparsePublicState;
use r3_action_edit_census::{
    ACTION_EDIT_MAGIC, ActionEdit, CensusConfig, PublicStateTrunk, R3_CENSUS_PROTOCOL_ID,
    R3_EXPERIMENT_ID, STATE_TRUNK_MAGIC, SupplySnapshot,
};

#[test]
fn experiment_identity_and_scientific_protocol_are_frozen() {
    let config = CensusConfig::default();
    assert_eq!(config.experiment_id, R3_EXPERIMENT_ID);
    assert_eq!(config.protocol_id, R3_CENSUS_PROTOCOL_ID);
    assert_eq!(
        serde_json::to_vec(&config).unwrap(),
        serde_json::to_vec(&CensusConfig::default()).unwrap()
    );
    config.validate().unwrap();

    let mut wrong_id = config.clone();
    wrong_id.experiment_id = "renamed-r3".to_owned();
    assert!(wrong_id.validate().is_err());

    let mut skipped_gate = config;
    skipped_gate.corpus.d6_sentinel_per_position = false;
    assert!(skipped_gate.validate().is_err());
}

#[test]
fn state_trunk_is_canonical_public_only_and_hidden_order_invariant() {
    let state = game(41);
    let trunk = PublicStateTrunk::observe(&state, 7).unwrap();
    let packed = trunk.to_packed_bytes().unwrap();
    assert_eq!(PublicStateTrunk::from_packed_bytes(&packed).unwrap(), trunk);
    assert_eq!(&packed[..8], STATE_TRUNK_MAGIC);
    assert!(trunk.sparse.global.targets_omitted);

    for seed in 0..16 {
        let mut redetermined = state.clone();
        redetermined.redeterminize_hidden(cascadia_game::GameSeed::from_u64(90_000 + seed));
        assert_eq!(PublicStateTrunk::observe(&redetermined, 7).unwrap(), trunk);
    }

    let record = PositionRecord::observe(&state, 7);
    let mut target_mutation = record.clone();
    target_mutation.targets = [u16::MAX; TARGET_DIM];
    assert_eq!(
        SparsePublicState::from_position_record(&record, None).unwrap(),
        SparsePublicState::from_position_record(&target_mutation, None).unwrap()
    );
}

#[test]
fn every_action_in_generated_paired_and_independent_screens_reproduces_authority() {
    let state = state_with_nature_token();
    let game_index = 81_000 + u64::from(state.completed_turns());
    let trunk = PublicStateTrunk::observe(&state, game_index).unwrap();
    let prepared = trunk.prepare_action_edits().unwrap();
    let trunk_bytes = prepared.packed_bytes().to_vec();
    let trunk_hash = prepared.canonical_hash();
    let (prelude, legal_actions) = canonical_actions(&state);
    let (paired, independent) = representative_drafts(&state);
    let observed = prepared.observe_legal_actions(&state, &prelude).unwrap();
    assert_eq!(
        observed
            .iter()
            .map(|(action, _)| action)
            .collect::<Vec<_>>(),
        legal_actions.iter().collect::<Vec<_>>()
    );
    assert!(
        observed
            .iter()
            .all(|(_, edit)| edit.state_trunk_blake3 == trunk_hash)
    );

    let mut tested = 0usize;
    for draft in [paired, independent] {
        let actions = observed
            .iter()
            .filter(|(action, _)| action.draft == draft)
            .collect::<Vec<_>>();
        assert!(!actions.is_empty());
        for (action, edit) in actions {
            assert_eq!(edit.state_trunk_blake3, trunk_hash);
            let applied = prepared.apply(edit).unwrap();
            let public_afterstate = state.preview_public_afterstate(action).unwrap();
            let authoritative = PositionRecord::observe_public_for_seat(
                &public_afterstate,
                game_index,
                state.current_player(),
            );
            assert_eq!(applied.record.to_bytes(), authoritative.to_bytes());
            assert_eq!(
                applied.supply,
                SupplySnapshot::from_exact(
                    &ExactSemanticSupply::from_public_state(&public_afterstate).unwrap()
                )
            );
            let packed = edit.to_packed_bytes().unwrap();
            let decoded = ActionEdit::from_packed_bytes(&packed).unwrap();
            assert_eq!(&decoded, edit);
            assert_eq!(prepared.apply(&decoded).unwrap(), applied);
            assert_eq!(&packed[..8], ACTION_EDIT_MAGIC);
            tested += 1;
        }
    }
    assert_eq!(prepared.packed_bytes(), trunk_bytes);
    assert!(tested > 20);
}

#[test]
fn draft_batched_observation_matches_the_complete_screen() {
    let state = state_with_nature_token();
    let game_index = 81_500 + u64::from(state.completed_turns());
    let trunk = PublicStateTrunk::observe(&state, game_index).unwrap();
    let prepared = trunk.prepare_action_edits().unwrap();
    let (prelude, _) = canonical_actions(&state);
    let complete = prepared.observe_legal_actions(&state, &prelude).unwrap();
    let (paired, independent) = representative_drafts(&state);

    for draft in [paired, independent] {
        let expected = complete
            .iter()
            .filter(|(action, _)| action.draft == draft)
            .cloned()
            .collect::<Vec<_>>();
        let observed = prepared
            .observe_draft_actions(&state, &prelude, draft)
            .unwrap();
        assert_eq!(observed, expected);
    }
}

#[test]
fn mlx_action_tokens_round_trip_real_actions_and_reject_corruption() {
    let state = state_with_nature_token();
    let game_index = 81_600 + u64::from(state.completed_turns());
    let trunk = PublicStateTrunk::observe(&state, game_index).unwrap();
    let prepared = trunk.prepare_action_edits().unwrap();
    let (prelude, _) = canonical_actions(&state);
    let observed = prepared.observe_legal_actions(&state, &prelude).unwrap();

    for (action, edit) in observed
        .iter()
        .step_by((observed.len() / 32).max(1))
        .take(32)
    {
        let encoding = edit.mlx_action_encoding().unwrap();
        assert_eq!(encoding.decode_canonical_view().unwrap(), edit.canonical);
        assert!(encoding.tokens.len() >= 38);
        assert!(
            encoding
                .tokens
                .iter()
                .all(|token| token.token_type > 0 && token.token_type <= 8)
        );
        let transform =
            D6Transform::from_id(prepared.canonical_transform_id(edit).unwrap()).unwrap();
        assert_eq!(
            transform.transform_tile_rotation(
                edit.selected.tile.as_tile(),
                Rotation::new(edit.factors.tile_rotation).unwrap(),
            ),
            Rotation::ZERO
        );
        prepared.apply(edit).unwrap();
        state.preview_public_afterstate(action).unwrap();
    }

    let mut corrupted = observed[0].1.mlx_action_encoding().unwrap();
    corrupted.tokens[0].payload[63] = 1;
    assert!(corrupted.decode_canonical_view().is_err());
}

#[test]
fn paid_wipe_prelude_is_exact_and_variable_length() {
    let state = state_with_nature_tokens(2);
    let game_index = 82_000 + u64::from(state.completed_turns());
    let trunk = PublicStateTrunk::observe(&state, game_index).unwrap();
    let prepared = trunk.prepare_action_edits().unwrap();
    let (free, staged) = state.preview_free_three_of_a_kind_if_feasible().unwrap();
    let wipe = staged
        .legal_wildlife_wipes()
        .into_iter()
        .find(|wipe| wipe.slots.len() == 4)
        .unwrap();
    let prelude = MarketPrelude {
        replace_three_of_a_kind: free.replace_three_of_a_kind,
        wildlife_wipes: vec![wipe.clone(), wipe],
    };
    let observed = prepared.observe_legal_actions(&state, &prelude).unwrap();
    let (action, edit) = observed
        .into_iter()
        .find(|(action, _)| action.wildlife.is_some())
        .unwrap();
    assert_eq!(edit.state_trunk_blake3, prepared.canonical_hash());
    assert_eq!(edit.factors.wildlife_wipe_masks, vec![0x0f, 0x0f]);
    assert_eq!(
        i16::from(edit.prelude.active_player_after.nature_tokens)
            - i16::from(edit.prelude.active_player_before.nature_tokens),
        -2
    );
    assert!(!edit.prelude.market_edits.is_empty());
    let applied = prepared.apply(&edit).unwrap();
    let public_afterstate = state.preview_public_afterstate(&action).unwrap();
    assert_eq!(
        applied.record.to_bytes(),
        PositionRecord::observe_public_for_seat(
            &public_afterstate,
            game_index,
            state.current_player()
        )
        .to_bytes()
    );
    assert_eq!(
        applied.supply,
        SupplySnapshot::from_exact(
            &ExactSemanticSupply::from_public_state(&public_afterstate).unwrap()
        )
    );
}

#[test]
fn all_twelve_d6_transforms_preserve_the_canonical_action_edit() {
    let state = game(137);
    let action = representative_dual_action(&state);
    let game_index = 83_000;
    let trunk = PublicStateTrunk::observe(&state, game_index).unwrap();
    let source = ActionEdit::observe(&state, &trunk, &action).unwrap();

    for transform in D6Transform::ALL {
        let transformed_state = state.transformed(transform).unwrap();
        let transformed_action = state.transform_turn_action(&action, transform).unwrap();
        let transformed_trunk = PublicStateTrunk::observe(&transformed_state, game_index).unwrap();
        let transformed =
            ActionEdit::observe(&transformed_state, &transformed_trunk, &transformed_action)
                .unwrap();
        assert_eq!(transformed.canonical, source.canonical);
        assert_eq!(transformed.selected, source.selected);
        assert_eq!(transformed.score_delta, source.score_delta);
        assert_eq!(transformed.radius_coverage, source.radius_coverage);
        transformed.apply(&transformed_trunk).unwrap();
    }
}

#[test]
fn d6_regression_seed_4100003_turn_zero_is_exact() {
    let state = game(4_100_003);
    let (_, actions) = canonical_actions(&state);
    let action = actions
        .into_iter()
        .find(|action| {
            !action.replace_three_of_a_kind
                && action.wildlife_wipes.is_empty()
                && action.draft
                    == DraftChoice::Paired {
                        slot: MarketSlot::ONE,
                    }
                && action.tile.coord == HexCoord::new(-1, 1)
                && action.tile.rotation == Rotation::new(2).unwrap()
                && action.wildlife.is_none()
        })
        .unwrap();
    let game_index = 410_000_300;
    let trunk = PublicStateTrunk::observe(&state, game_index).unwrap();
    let source = ActionEdit::observe(&state, &trunk, &action).unwrap();
    let transform = D6Transform::from_id(2).unwrap();
    let transformed_state = state.transformed(transform).unwrap();
    let transformed_action = state.transform_turn_action(&action, transform).unwrap();
    let transformed_trunk = PublicStateTrunk::observe(&transformed_state, game_index).unwrap();
    let transformed =
        ActionEdit::observe(&transformed_state, &transformed_trunk, &transformed_action).unwrap();

    assert_eq!(transformed.canonical, source.canonical);
    assert_eq!(transformed.radius_coverage, source.radius_coverage);
}

#[test]
fn paired_and_independent_draft_factors_remain_distinct() {
    let state = state_with_nature_token();
    let trunk = PublicStateTrunk::observe(&state, 84_000).unwrap();
    let (prelude, _) = canonical_actions(&state);
    let observed = ActionEdit::observe_legal_actions(&state, &trunk, &prelude).unwrap();
    let paired = observed
        .iter()
        .find(|(action, _)| matches!(action.draft, DraftChoice::Paired { .. }))
        .unwrap();
    let independent = observed
        .iter()
        .find(|(action, _)| matches!(action.draft, DraftChoice::Independent { .. }))
        .unwrap();
    assert_ne!(paired.1.factors.draft, independent.1.factors.draft);
    assert_ne!(
        paired.1.placement.active_player_after.nature_tokens,
        independent.1.placement.active_player_after.nature_tokens
    );
}

#[test]
fn codecs_reject_corruption_truncation_and_trailing_bytes() {
    let state = game(211);
    let trunk = PublicStateTrunk::observe(&state, 85_000).unwrap();
    let action = canonical_actions(&state).1.remove(0);
    let edit = ActionEdit::observe(&state, &trunk, &action).unwrap();

    let trunk_bytes = trunk.to_packed_bytes().unwrap();
    let edit_bytes = edit.to_packed_bytes().unwrap();
    for bytes in [&trunk_bytes, &edit_bytes] {
        assert!(bytes.len() > 16);
    }

    let mut bad_trunk_magic = trunk_bytes.clone();
    bad_trunk_magic[0] ^= 0xff;
    assert!(PublicStateTrunk::from_packed_bytes(&bad_trunk_magic).is_err());
    assert!(PublicStateTrunk::from_packed_bytes(&trunk_bytes[..trunk_bytes.len() - 1]).is_err());
    let mut trailing_trunk = trunk_bytes;
    trailing_trunk.push(0);
    assert!(PublicStateTrunk::from_packed_bytes(&trailing_trunk).is_err());

    let mut bad_edit_magic = edit_bytes.clone();
    bad_edit_magic[0] ^= 0xff;
    assert!(ActionEdit::from_packed_bytes(&bad_edit_magic).is_err());
    assert!(ActionEdit::from_packed_bytes(&edit_bytes[..edit_bytes.len() - 1]).is_err());
    let mut trailing_edit = edit_bytes;
    trailing_edit.push(0);
    assert!(ActionEdit::from_packed_bytes(&trailing_edit).is_err());
}
