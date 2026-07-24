import type { ScoringVariant, Wildlife } from "./types";

export const WILDLIFE_ORDER = [
  "bear",
  "elk",
  "salmon",
  "hawk",
  "fox",
] as const satisfies readonly Wildlife[];

export const SCORING_VARIANTS = [
  "A",
  "B",
  "C",
  "D",
] as const satisfies readonly ScoringVariant[];

export interface WildlifeProfile {
  id: Wildlife;
  label: string;
  emoji: string;
  short: string;
}

export const WILDLIFE_PROFILES: readonly WildlifeProfile[] = [
  { id: "bear", label: "Bear", emoji: "🐻", short: "BR" },
  { id: "elk", label: "Elk", emoji: "🫎", short: "EL" },
  { id: "salmon", label: "Salmon", emoji: "🐟", short: "SA" },
  { id: "hawk", label: "Hawk", emoji: "🦅", short: "HK" },
  { id: "fox", label: "Fox", emoji: "🦊", short: "FX" },
];

export const SCORING_CARD_RULES: Record<
  Wildlife,
  Record<ScoringVariant, string>
> = {
  bear: {
    A: "Score separate pairs: 4 / 11 / 19 / 27 for one through four pairs.",
    B: "Each isolated group of exactly three bears scores 10.",
    C: "Groups of one, two, or three score 2 / 5 / 8; collect all sizes for +3.",
    D: "Groups of exactly two, three, or four score 5 / 8 / 13.",
  },
  elk: {
    A: "Partition elk into straight formations of up to four: 2 / 5 / 9 / 13.",
    B: "Partition elk into compact line, triangle, or diamond formations: 2 / 5 / 9 / 13.",
    C: "Partition connected herds; larger connected groups climb to 28 at eight or more.",
    D: "Arrange elk around shared centers; ring groups of one through six score 2 / 5 / 8 / 12 / 16 / 21.",
  },
  salmon: {
    A: "Non-branching runs score 2 / 5 / 8 / 12 / 16 / 20 / 25 at seven or more.",
    B: "Non-branching runs score 2 / 4 / 9 / 11, then 17 at five or more.",
    C: "Only runs of at least three score: 10 / 12 / 15 at five or more.",
    D: "Runs of at least three score their length plus adjacent non-salmon wildlife.",
  },
  hawk: {
    A: "Count non-adjacent hawks: 2 / 5 / 8 / 11 / 14 / 18, up to 26 at eight.",
    B: "Count isolated hawks with unobstructed line of sight to another hawk.",
    C: "Every unobstructed hawk-to-hawk sight line scores 3.",
    D: "Match disjoint sight-line pairs for 4 / 7 / 9 based on wildlife variety between them.",
  },
  fox: {
    A: "Each fox scores the number of distinct adjacent wildlife species.",
    B: "Each fox scores 3 / 5 / 7 for one, two, or three adjacent non-fox species appearing in pairs.",
    C: "Each fox scores its largest adjacent count of any one non-fox species.",
    D: "Match adjacent fox pairs; surrounding doubled species score 5 / 7 / 9 / 11.",
  },
};

export type CompactToken = [q: number, r: number, wildlifeIndex: number];

export interface WildlifeAtlasRow {
  id: string;
  score: number;
  upper: number;
  exact: boolean;
  counts: number[];
  parts: number[];
  tokens: CompactToken[];
}

export interface WildlifeAtlasDocument {
  schema: "cascadia-wildlife-atlas-v1";
  sourceSchema: string;
  sourceSha256: string;
  rulesetCount: number;
  tokenCount: number;
  countCap: number;
  completedRulesets: number;
  incumbentHolisticMaximum: number;
  holisticSoundUpper: number;
  leaders: string[];
  rows: WildlifeAtlasRow[];
}

export type AtlasListFilter = "all" | "exact" | "leaders";

export function isRuleset(value: string): boolean {
  return /^[A-D]{5}$/.test(value);
}

export function rulesetToIndex(ruleset: string): number {
  if (!isRuleset(ruleset)) {
    throw new Error(`Invalid wildlife ruleset: ${ruleset}`);
  }
  return [...ruleset].reduce(
    (index, variant) => index * SCORING_VARIANTS.length + variant.charCodeAt(0) - 65,
    0,
  );
}

export function indexToRuleset(index: number): string {
  if (!Number.isInteger(index) || index < 0 || index >= 4 ** WILDLIFE_ORDER.length) {
    throw new Error(`Invalid wildlife catalog index: ${index}`);
  }
  let remainder = index;
  const cards = Array.from({ length: WILDLIFE_ORDER.length }, () => "A");
  for (let position = cards.length - 1; position >= 0; position -= 1) {
    cards[position] = SCORING_VARIANTS[remainder % SCORING_VARIANTS.length];
    remainder = Math.floor(remainder / SCORING_VARIANTS.length);
  }
  return cards.join("");
}

export function replaceRulesetCard(
  ruleset: string,
  position: number,
  variant: ScoringVariant,
): string {
  if (!isRuleset(ruleset) || position < 0 || position >= WILDLIFE_ORDER.length) {
    throw new Error("Cannot replace card in invalid wildlife ruleset");
  }
  return `${ruleset.slice(0, position)}${variant}${ruleset.slice(position + 1)}`;
}

export function requestedRuleset(search: string, fallback = "ADCCB"): string {
  const candidate = new URLSearchParams(search).get("rules")?.toUpperCase() ?? "";
  return isRuleset(candidate) ? candidate : fallback;
}

export function rankedRows(
  rows: readonly WildlifeAtlasRow[],
  filter: AtlasListFilter,
): WildlifeAtlasRow[] {
  const maximum =
    filter === "leaders" ? Math.max(...rows.map((candidate) => candidate.score)) : 0;
  return [...rows]
    .filter((row) => {
      if (filter === "exact") return row.exact;
      if (filter === "leaders") return row.score === maximum;
      return true;
    })
    .sort(
      (left, right) =>
        right.score - left.score ||
        Number(right.exact) - Number(left.exact) ||
        left.upper - right.upper ||
        left.id.localeCompare(right.id),
    );
}

export function assertAtlasDocument(value: unknown): WildlifeAtlasDocument {
  if (!value || typeof value !== "object") {
    throw new Error("Wildlife atlas data is not an object");
  }
  const document = value as Partial<WildlifeAtlasDocument>;
  if (
    document.schema !== "cascadia-wildlife-atlas-v1" ||
    document.rulesetCount !== 1024 ||
    document.rows?.length !== document.rulesetCount
  ) {
    throw new Error("Wildlife atlas data has an unsupported or incomplete schema");
  }
  const ids = new Set(document.rows.map((row) => row.id));
  if (
    ids.size !== document.rulesetCount ||
    document.rows.some(
      (row, index) =>
        !isRuleset(row.id) ||
        row.id !== indexToRuleset(index) ||
        row.tokens.length !== document.tokenCount ||
        row.counts.length !== WILDLIFE_ORDER.length ||
        row.parts.length !== WILDLIFE_ORDER.length ||
        row.counts.reduce((total, count) => total + count, 0) !==
          document.tokenCount ||
        Math.max(...row.counts) > (document.countCap ?? 0) ||
        new Set(row.tokens.map(([q, r]) => `${q},${r}`)).size !==
          document.tokenCount ||
        row.tokens.some(
          ([, , wildlifeIndex]) =>
            wildlifeIndex < 0 || wildlifeIndex >= WILDLIFE_ORDER.length,
        ) ||
        WILDLIFE_ORDER.some(
          (_, wildlifeIndex) =>
            row.tokens.filter(([, , tokenWildlife]) => tokenWildlife === wildlifeIndex)
              .length !== row.counts[wildlifeIndex],
        ) ||
        row.parts.reduce((total, score) => total + score, 0) !== row.score ||
        row.upper < row.score ||
        (row.exact && row.upper !== row.score),
    )
  ) {
    throw new Error("Wildlife atlas data failed integrity checks");
  }
  return document as WildlifeAtlasDocument;
}
