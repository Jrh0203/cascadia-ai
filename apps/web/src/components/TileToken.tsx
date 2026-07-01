import type { TileView } from "../types";
import { WildlifeMark } from "./WildlifeMark";

export function TileToken({
  tile,
  compact = false,
}: {
  tile: TileView;
  compact?: boolean;
}) {
  return (
    <div
      className={`tile-token ${compact ? "tile-token-compact" : ""}`}
      aria-label={`${tile.terrain_a}${tile.terrain_b ? ` and ${tile.terrain_b}` : ""} habitat tile`}
    >
      <svg viewBox="0 0 88 76" aria-hidden="true">
        <defs>
          <clipPath id={`tile-${tile.id}`}>
            <polygon points="22,2 66,2 86,38 66,74 22,74 2,38" />
          </clipPath>
        </defs>
        <polygon
          className={`terrain-fill terrain-${tile.terrain_a}`}
          points="22,2 66,2 86,38 66,74 22,74 2,38"
        />
        {tile.terrain_b && (
          <polygon
            className={`terrain-fill terrain-${tile.terrain_b}`}
            points="44,38 66,2 86,38 66,74"
            clipPath={`url(#tile-${tile.id})`}
          />
        )}
        <polygon
          className="tile-token-outline"
          points="22,2 66,2 86,38 66,74 22,74 2,38"
        />
      </svg>
      <div className="tile-token-wildlife">
        {tile.wildlife.map((wildlife) => (
          <WildlifeMark key={wildlife} wildlife={wildlife} size="small" />
        ))}
      </div>
      {tile.keystone && <span className="keystone-mark" aria-label="keystone tile" />}
    </div>
  );
}
