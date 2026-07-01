import { describe, expect, it } from "vitest";

import {
  coordKey,
  defaultSeats,
  draftSlots,
  independentDraft,
  pairedDraft,
  sameCoord,
} from "./game";

describe("game UI helpers", () => {
  it("creates one human seat followed by local AI seats", () => {
    expect(defaultSeats(4)).toEqual([
      { kind: "human", strength: "instant" },
      { kind: "ai", strength: "interactive" },
      { kind: "ai", strength: "interactive" },
      { kind: "ai", strength: "interactive" },
    ]);
  });

  it("normalizes draft selections", () => {
    expect(draftSlots(pairedDraft(2))).toEqual({
      tile: 2,
      wildlife: 2,
      independent: false,
    });
    expect(draftSlots(independentDraft(1, 3))).toEqual({
      tile: 1,
      wildlife: 3,
      independent: true,
    });
  });

  it("compares and keys axial coordinates", () => {
    expect(coordKey({ q: -2, r: 4 })).toBe("-2,4");
    expect(sameCoord({ q: 1, r: 2 }, { q: 1, r: 2 })).toBe(true);
    expect(sameCoord({ q: 1, r: 2 }, null)).toBe(false);
  });
});
