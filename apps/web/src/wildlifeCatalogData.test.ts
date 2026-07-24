import { describe, expect, it } from "vitest";

import {
  assertAtlasDocument,
  indexToRuleset,
  rankedRows,
  replaceRulesetCard,
  requestedRuleset,
  rulesetToIndex,
  type WildlifeAtlasDocument,
  type WildlifeAtlasRow,
} from "./wildlifeCatalogData";

function row(
  id: string,
  score: number,
  upper: number,
  exact = false,
): WildlifeAtlasRow {
  return {
    id,
    score,
    upper,
    exact,
    counts: [4, 4, 4, 4, 4],
    parts: [score, 0, 0, 0, 0],
    tokens: Array.from({ length: 20 }, (_, index) => [index, 0, index % 5]),
  };
}

describe("wildlife atlas helpers", () => {
  it("maps all five scoring cards to the catalog's base-four index", () => {
    expect(rulesetToIndex("AAAAA")).toBe(0);
    expect(rulesetToIndex("AAACA")).toBe(8);
    expect(rulesetToIndex("ADCCB")).toBe(233);
    expect(rulesetToIndex("DDDDD")).toBe(1023);
    expect(indexToRuleset(233)).toBe("ADCCB");
  });

  it("changes one card without disturbing the other species", () => {
    expect(replaceRulesetCard("AAAAA", 3, "C")).toBe("AAACA");
    expect(replaceRulesetCard("CBDDB", 4, "D")).toBe("CBDDD");
  });

  it("accepts only valid share-link rulesets", () => {
    expect(requestedRuleset("?rules=cbddb")).toBe("CBDDB");
    expect(requestedRuleset("?rules=ZAAAA", "AAAAA")).toBe("AAAAA");
  });

  it("ranks score first, then proof quality and proof gap", () => {
    const rows = [
      row("AAAAA", 70, 76),
      row("AAAAB", 72, 80),
      row("AAAAC", 72, 72, true),
      row("AAAAD", 72, 75),
    ];
    expect(rankedRows(rows, "all").map((candidate) => candidate.id)).toEqual([
      "AAAAC",
      "AAAAD",
      "AAAAB",
      "AAAAA",
    ]);
    expect(rankedRows(rows, "exact").map((candidate) => candidate.id)).toEqual([
      "AAAAC",
    ]);
    expect(rankedRows(rows, "leaders").map((candidate) => candidate.id)).toEqual([
      "AAAAC",
      "AAAAD",
      "AAAAB",
    ]);
  });

  it("fails closed on incomplete catalog payloads", () => {
    const document = {
      schema: "cascadia-wildlife-atlas-v1",
      rulesetCount: 1024,
      rows: [],
    } as unknown as WildlifeAtlasDocument;
    expect(() => assertAtlasDocument(document)).toThrow(/unsupported or incomplete/);
  });
});
