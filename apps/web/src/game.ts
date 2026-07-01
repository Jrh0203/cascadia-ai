import type {
  DraftChoice,
  HexCoord,
  MarketPrelude,
  ScoringCards,
  SeatConfig,
  Terrain,
  Wildlife,
} from "./types";

export const TERRAINS: Terrain[] = [
  "mountain",
  "forest",
  "prairie",
  "wetland",
  "river",
];
export const WILDLIFE: Wildlife[] = ["bear", "elk", "salmon", "hawk", "fox"];
export const DEFAULT_CARDS: ScoringCards = {
  bear: "A",
  elk: "A",
  salmon: "A",
  hawk: "A",
  fox: "A",
};
export const EMPTY_PRELUDE: MarketPrelude = {
  replace_three_of_a_kind: false,
  wildlife_wipes: [],
};

export function defaultSeats(playerCount: number): SeatConfig[] {
  return Array.from({ length: playerCount }, (_, player) => ({
    kind: player === 0 ? "human" : "ai",
    strength: player === 0 ? "instant" : "interactive",
  }));
}

export function coordKey(coord: HexCoord): string {
  return `${coord.q},${coord.r}`;
}

export function sameCoord(left: HexCoord | null, right: HexCoord | null): boolean {
  return left?.q === right?.q && left?.r === right?.r;
}

export function pairedDraft(slot: number): DraftChoice {
  return { Paired: { slot } };
}

export function independentDraft(tileSlot: number, wildlifeSlot: number): DraftChoice {
  return { Independent: { tile_slot: tileSlot, wildlife_slot: wildlifeSlot } };
}

export function draftSlots(draft: DraftChoice | null): {
  tile: number | null;
  wildlife: number | null;
  independent: boolean;
} {
  if (!draft) return { tile: null, wildlife: null, independent: false };
  if ("Paired" in draft) {
    return {
      tile: draft.Paired.slot,
      wildlife: draft.Paired.slot,
      independent: false,
    };
  }
  return {
    tile: draft.Independent.tile_slot,
    wildlife: draft.Independent.wildlife_slot,
    independent: true,
  };
}

export function actionSummary(draft: DraftChoice): string {
  if ("Paired" in draft) return `Pair ${draft.Paired.slot + 1}`;
  return `Tile ${draft.Independent.tile_slot + 1} + wildlife ${
    draft.Independent.wildlife_slot + 1
  }`;
}

export function titleCase(value: string): string {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

export function randomSeed(): number {
  return crypto.getRandomValues(new Uint32Array(1))[0];
}
