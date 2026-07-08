import { titleCase } from "../game";
import type {
  DraftChoice,
  MarketPairView,
  TurnOptions,
} from "../types";
import { TileToken } from "./TileToken";
import { WildlifeMark } from "./WildlifeMark";

/** Legacy-v1 style market rail: a vertical list of tile/wildlife pairs.
 * Click a pair to draft it (phase select_market); non-selected pairs dim
 * once drafting starts. Independent (nature token) drafting highlights the
 * two halves separately, mirroring the v1 "pinecone" flow. Prelude moves
 * (wipe / free replacement) surface as stacked buttons underneath, like
 * v1's overflow/mulligan buttons. */
interface MarketRailProps {
  market: MarketPairView[];
  turnOptions: TurnOptions | null;
  draft: DraftChoice | null;
  independent: boolean;
  selectedTileSlot: number | null;
  selectedWildlifeSlot: number | null;
  phase: "select_market" | "place_tile" | "place_wildlife";
  glowTileSlot: number | null;
  glowWildlifeSlot: number | null;
  glowSplit: boolean;
  busy: boolean;
  freeReplacement: boolean;
  freeReplacementSelected: boolean;
  preludeActive: boolean;
  natureTokens: number;
  onSelectDraftComponent: (slot: number, component: "tile" | "wildlife") => void;
  onSetIndependent: (independent: boolean) => void;
  onWipeAll: () => void;
  onToggleFreeReplacement: () => void;
  onResetPrelude: () => void;
}

export function MarketRail({
  market,
  turnOptions,
  draft,
  independent,
  selectedTileSlot,
  selectedWildlifeSlot,
  phase,
  glowTileSlot,
  glowWildlifeSlot,
  glowSplit,
  busy,
  freeReplacement,
  freeReplacementSelected,
  preludeActive,
  natureTokens,
  onSelectDraftComponent,
  onSetIndependent,
  onWipeAll,
  onToggleFreeReplacement,
  onResetPrelude,
}: MarketRailProps) {
  const selectable = phase === "select_market" && !busy;
  const canWipe = turnOptions?.can_add_paid_wipe ?? false;
  return (
    <div className="market-rail">
      {market.map((pair) => {
        const slot = pair.slot;
        const tileSelected = selectedTileSlot === slot;
        const wildlifeSelected = selectedWildlifeSlot === slot;
        const dimmed = draft !== null && !tileSelected && !wildlifeSelected;
        const pairGlow = !glowSplit && glowTileSlot === slot;
        const tileGlow = glowSplit && glowTileSlot === slot;
        const wildlifeGlow = glowSplit && glowWildlifeSlot === slot;
        if (!pair.tile || !pair.wildlife) {
          return (
            <div key={slot} className="market-pair empty">
              <span className="market-empty-note">empty slot</span>
            </div>
          );
        }
        const terrains = [pair.tile.terrain_a, pair.tile.terrain_b]
          .filter((terrain): terrain is NonNullable<typeof terrain> => terrain !== null)
          .map(titleCase)
          .join("/");
        return (
          <div
            key={slot}
            className={`market-pair${tileSelected || wildlifeSelected ? " selected" : ""}${dimmed ? " dimmed" : ""}${pairGlow ? " suggest-glow" : ""}`}
          >
            <button
              type="button"
              className={`market-half${tileSelected ? " picked" : ""}${tileGlow ? " suggest-glow" : ""}`}
              disabled={!selectable}
              onClick={() => onSelectDraftComponent(slot, "tile")}
              title={independent ? "Draft this tile" : "Draft this pair"}
            >
              <TileToken tile={pair.tile} />
            </button>
            <button
              type="button"
              className={`market-half${wildlifeSelected ? " picked" : ""}${wildlifeGlow ? " suggest-glow" : ""}`}
              disabled={!selectable}
              onClick={() => onSelectDraftComponent(slot, "wildlife")}
              title={independent ? "Draft this wildlife" : "Draft this pair"}
            >
              <WildlifeMark wildlife={pair.wildlife} />
            </button>
            <div className="market-info">
              {terrains}
              {pair.tile.keystone ? " ★" : ""}
              <div className="market-allowed">
                {pair.tile.wildlife.map((slotWildlife) => (
                  <WildlifeMark key={slotWildlife} wildlife={slotWildlife} size="small" />
                ))}
              </div>
            </div>
          </div>
        );
      })}

      <div className="market-actions">
        {canWipe && (
          <button
            type="button"
            className="rail-button danger"
            onClick={onWipeAll}
            disabled={busy || draft !== null}
            title="Spend a nature token to wipe the most repeated wildlife from the market"
          >
            Overflow: wipe repeated wildlife
          </button>
        )}
        {freeReplacement && (
          <button
            type="button"
            className={`rail-button${freeReplacementSelected ? " active" : ""}`}
            onClick={onToggleFreeReplacement}
            disabled={busy || draft !== null}
          >
            {freeReplacementSelected ? "✓ " : ""}Free three-of-a-kind refresh
          </button>
        )}
        <button
          type="button"
          className={`rail-button${independent ? " active" : ""}`}
          onClick={() => onSetIndependent(!independent)}
          disabled={busy || natureTokens === 0 || draft !== null}
          title="Spend a nature token to pick tile and wildlife from different slots"
        >
          {independent ? "✓ " : ""}Nature token: split draft ({natureTokens})
        </button>
        {preludeActive && (
          <button
            type="button"
            className="rail-button"
            onClick={onResetPrelude}
            disabled={busy}
          >
            Reset turn setup
          </button>
        )}
      </div>
    </div>
  );
}
