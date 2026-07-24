import { LocateFixed, Minus, Plus } from "lucide-react";
import { useId, useMemo, useState } from "react";

import {
  WILDLIFE_ORDER,
  WILDLIFE_PROFILES,
  type CompactToken,
} from "../wildlifeCatalogData";
import type { Wildlife } from "../types";

const HEX_SIZE = 47;
const SQRT_3 = Math.sqrt(3);
const FORWARD_NEIGHBORS = new Set(["1,0", "1,-1", "0,-1"]);

interface WildlifeAtlasBoardProps {
  ruleset: string;
  tokens: CompactToken[];
  focusedWildlife: Wildlife | null;
  onFocusWildlife: (wildlife: Wildlife | null) => void;
}

interface Point {
  x: number;
  y: number;
}

function center(q: number, r: number): Point {
  return {
    x: HEX_SIZE * SQRT_3 * (q + r / 2),
    y: HEX_SIZE * 1.5 * r,
  };
}

function cornerPoints(q: number, r: number): string {
  const { x, y } = center(q, r);
  return Array.from({ length: 6 }, (_, index) => {
    const angle = ((60 * index - 30) * Math.PI) / 180;
    return `${x + HEX_SIZE * Math.cos(angle)},${y + HEX_SIZE * Math.sin(angle)}`;
  }).join(" ");
}

function coordinateKey(q: number, r: number): string {
  return `${q},${r}`;
}

export function WildlifeAtlasBoard({
  ruleset,
  tokens,
  focusedWildlife,
  onFocusWildlife,
}: WildlifeAtlasBoardProps) {
  const [zoom, setZoom] = useState(1);
  const rawPatternId = useId();
  const patternId = rawPatternId.replaceAll(":", "");
  const geometry = useMemo(() => {
    const points = tokens.map(([q, r]) => center(q, r));
    const xs = points.map((point) => point.x);
    const ys = points.map((point) => point.y);
    const padding = HEX_SIZE * 1.7;
    const minX = Math.min(...xs) - padding;
    const maxX = Math.max(...xs) + padding;
    const minY = Math.min(...ys) - padding;
    const maxY = Math.max(...ys) + padding;
    const occupied = new Set(tokens.map(([q, r]) => coordinateKey(q, r)));
    const links: [Point, Point][] = [];
    for (const [q, r] of tokens) {
      for (const delta of FORWARD_NEIGHBORS) {
        const [dq, dr] = delta.split(",").map(Number);
        if (occupied.has(coordinateKey(q + dq, r + dr))) {
          links.push([center(q, r), center(q + dq, r + dr)]);
        }
      }
    }
    return {
      bounds: {
        x: minX,
        y: minY,
        width: maxX - minX,
        height: maxY - minY,
      },
      links,
    };
  }, [tokens]);

  const centerX = geometry.bounds.x + geometry.bounds.width / 2;
  const centerY = geometry.bounds.y + geometry.bounds.height / 2;
  const transform = `translate(${centerX} ${centerY}) scale(${zoom}) translate(${-centerX} ${-centerY})`;

  return (
    <div className="atlas-board-stage">
      <svg
        className="atlas-hex-board"
        viewBox={`${geometry.bounds.x} ${geometry.bounds.y} ${geometry.bounds.width} ${geometry.bounds.height}`}
        role="img"
        aria-label={`${ruleset} best-known wildlife board with ${tokens.length} animals`}
      >
        <defs>
          <pattern
            id={patternId}
            width="34"
            height="34"
            patternUnits="userSpaceOnUse"
            patternTransform="rotate(18)"
          >
            <path d="M 0 17 H 34" className="atlas-contour-line" />
            <circle cx="4" cy="5" r="1.1" className="atlas-contour-dot" />
          </pattern>
          <filter id={`${patternId}-shadow`} x="-40%" y="-40%" width="180%" height="180%">
            <feDropShadow dx="0" dy="5" stdDeviation="5" floodOpacity="0.38" />
          </filter>
        </defs>
        <rect
          x={geometry.bounds.x}
          y={geometry.bounds.y}
          width={geometry.bounds.width}
          height={geometry.bounds.height}
          fill={`url(#${patternId})`}
        />
        <g transform={transform} className="atlas-board-geometry">
          <g className="atlas-board-links" aria-hidden="true">
            {geometry.links.map(([from, to], index) => (
              <line
                key={`${from.x}-${from.y}-${to.x}-${to.y}-${index}`}
                x1={from.x}
                y1={from.y}
                x2={to.x}
                y2={to.y}
              />
            ))}
          </g>
          {tokens.map(([q, r, wildlifeIndex], tokenIndex) => {
            const wildlife = WILDLIFE_ORDER[wildlifeIndex];
            const profile = WILDLIFE_PROFILES[wildlifeIndex];
            const { x, y } = center(q, r);
            const isMuted = focusedWildlife !== null && focusedWildlife !== wildlife;
            const isFocused = focusedWildlife === wildlife;
            return (
              <g
                key={`${q}-${r}-${wildlife}`}
                className={`atlas-token wildlife-${wildlife}-fill ${
                  isMuted ? "is-muted" : ""
                } ${isFocused ? "is-focused" : ""}`}
                onClick={() =>
                  onFocusWildlife(focusedWildlife === wildlife ? null : wildlife)
                }
              >
                <title>
                  {profile.label} {tokenIndex + 1} at ({q}, {r})
                </title>
                <polygon
                  className="atlas-token-hex"
                  points={cornerPoints(q, r)}
                />
                <circle
                  className="atlas-token-disc"
                  cx={x}
                  cy={y}
                  r={HEX_SIZE * 0.34}
                  filter={`url(#${patternId}-shadow)`}
                />
                <text className="atlas-token-emoji" x={x} y={y + 1}>
                  {profile.emoji}
                </text>
                <text className="atlas-token-coordinate" x={x} y={y + 34}>
                  {q},{r}
                </text>
              </g>
            );
          })}
        </g>
      </svg>
      <div className="atlas-board-tools" aria-label="Board view controls">
        <button
          type="button"
          title="Zoom in"
          aria-label="Zoom in"
          onClick={() => setZoom((value) => Math.min(1.7, value * 1.12))}
        >
          <Plus aria-hidden="true" />
        </button>
        <button
          type="button"
          title="Zoom out"
          aria-label="Zoom out"
          onClick={() => setZoom((value) => Math.max(0.72, value / 1.12))}
        >
          <Minus aria-hidden="true" />
        </button>
        <button
          type="button"
          title="Fit board"
          aria-label="Fit board"
          onClick={() => setZoom(1)}
        >
          <LocateFixed aria-hidden="true" />
        </button>
      </div>
    </div>
  );
}
