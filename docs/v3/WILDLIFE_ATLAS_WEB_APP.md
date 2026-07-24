# Wildlife Atlas Web App

The Wildlife Atlas is the interactive companion to the complete cap-six
pure-wildlife catalog. It shows the best validated board currently available
for every ordered combination of five wildlife scoring cards.

Open the existing Vite app at:

```text
http://127.0.0.1:5187/wildlife-catalog
```

A specific card set can be linked directly:

```text
http://127.0.0.1:5187/wildlife-catalog?rules=CBDDB
```

## What the atlas shows

- All 1,024 ordered Bear/Elk/Salmon/Hawk/Fox A–D card combinations.
- One connected 20-token board for every card set.
- At most six tokens of any wildlife species.
- The independently validated animal score and its five-species breakdown.
- The best-known animal counts and axial coordinates.
- A clear distinction between a certified optimum and an unproven incumbent.
- The sound per-ruleset upper bound and remaining proof gap.
- Ranked views of the full catalog, the 80 proven rows, and the eight
  85-point leaders.

The atlas deliberately does **not** call all 1,024 boards optimal. The source
catalog contains 80 certified optima and 944 validated best-known incumbents.
For an unresolved row, the displayed interval `[score, sound upper]` is the
honest current claim.

## Controls

- Enter a five-letter ruleset ID or select A–D independently for each animal.
- Use Previous/Next, the left/right arrow keys, or Random to traverse the
  catalog.
- Click a ranked-ledger row to load it.
- Select a species in the board legend or score anatomy to isolate its
  placement pattern.
- Zoom the board or restore its fitted view.
- Copy a durable URL for the selected card set.

The page is responsive. On a narrow viewport it reads top-to-bottom as a
field guide: cards, board, score dossier, then a bounded ranked ledger.

## Data pipeline and integrity

The browser does not load the 13 MB research artifact directly. A deterministic
export produces a 267 KB purpose-built asset:

```bash
python3 tools/export_wildlife_atlas.py \
  docs/v3/evidence/all_wildlife_catalog_bound_probe_complete659_2026-07-24.json \
  apps/web/public/wildlife-atlas.json
```

The exporter fails closed unless:

- all 1,024 rulesets are unique and in catalog-index order;
- every board contains exactly 20 non-overlapping tokens;
- token species agree with the stored animal counts;
- all five score parts sum to the stored total;
- the count cap is respected; and
- every exact row has `score == sound upper`, while every other upper is at
  least its incumbent.

The compact asset preserves the source artifact SHA-256. The client repeats
the core schema, row-count, score-sum, token-count, and score-interval checks
before rendering.

Current provenance:

- Source catalog SHA-256:
  `6a4ba86d67b1bf4b44b5ef9a84791e078698dea2fc4d2b760324ad010a279b43`
- Compact web asset SHA-256:
  `d1061538c135bb614ce5fd17fcbf1d27808067423f86b336b5db05d1b0bfb74b`

## Implementation map

- `apps/web/src/WildlifeAtlas.tsx`: route shell, card selector, ranking,
  score/proof dossier, URL state.
- `apps/web/src/components/WildlifeAtlasBoard.tsx`: responsive pointy-top
  axial hex renderer and species focus.
- `apps/web/src/wildlifeCatalogData.ts`: types, rule references, catalog
  indexing, ranking, and runtime validation.
- `apps/web/public/wildlife-atlas.json`: compact curated catalog.
- `tools/export_wildlife_atlas.py`: deterministic validated exporter.

The renderer follows the established Cascadia UI geometry and animal
iconography rather than introducing a second board convention.

## Validation

From `apps/web`:

```bash
npm test
npm run lint
npm run build
npm run test:e2e
```

The end-to-end suite covers the atlas at 1440×960 and an iPhone 13 viewport,
including direct-link loading, a scoring-card change, score/proof updates, and
species isolation. The full web suite also covers the playable board and
cluster dashboard.

## Best next steps

The atlas makes the catalog useful for inspection; the next highest-value work
is to make the score structure legible and the remaining proof effort
selective.

1. Add rules-aware overlays: Hawk sight lines, Salmon runs, Bear components,
   Elk formations, and Fox neighborhoods. The current species isolation is a
   good general-purpose first layer, but these overlays would turn the atlas
   into a real teaching tool.
2. Canonicalize and cluster boards into recurring geometric archetypes. Many
   rulesets reuse the same or equivalent layouts; naming those patterns will
   yield more practical strategy than treating 1,024 rows as unrelated.
3. Prioritize exact proofs by decision value: the eight score-85 leaders,
   high-score low-gap rows, and rulesets whose proof could change a rule-card
   ranking. Proving all 944 unresolved rows uniformly is much less useful.
4. Regenerate the compact asset automatically whenever a new catalog becomes
   the durable source. The exporter makes this a deterministic one-command
   refresh.
5. Only after the pure-wildlife patterns are understood, add habitat
   feasibility as a separate layer. Mixing it in now would erase the clean
   causal signal this catalog was built to expose.
