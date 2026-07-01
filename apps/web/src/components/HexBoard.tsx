import { LocateFixed, Minus, Plus } from "lucide-react";
import {
  type PointerEvent as ReactPointerEvent,
  type WheelEvent,
  useMemo,
  useRef,
  useState,
} from "react";

import { coordKey, sameCoord } from "../game";
import type {
  BoardView,
  HexCoord,
  PlacementOption,
  TileView,
  Wildlife,
} from "../types";

const HEX_SIZE = 43;
const SQRT_3 = Math.sqrt(3);

interface PreviewTile {
  coord: HexCoord;
  tile: TileView;
  rotation: number;
  wildlife: Wildlife | null;
}

interface HexBoardProps {
  board: BoardView;
  active: boolean;
  placements: PlacementOption[];
  selectedCoord: HexCoord | null;
  wildlifeTargets: HexCoord[];
  selectedWildlife: HexCoord | null;
  draftedWildlife: Wildlife | null;
  preview: PreviewTile | null;
  onSelectCoord: (coord: HexCoord) => void;
  onSelectWildlife: (coord: HexCoord) => void;
}

function center(coord: HexCoord): [number, number] {
  return [
    HEX_SIZE * SQRT_3 * (coord.q + coord.r / 2),
    HEX_SIZE * 1.5 * coord.r,
  ];
}

function cornerPoints(coord: HexCoord): [number, number][] {
  const [cx, cy] = center(coord);
  return Array.from({ length: 6 }, (_, index) => {
    const angle = ((60 * index - 30) * Math.PI) / 180;
    return [
      cx + HEX_SIZE * Math.cos(angle),
      cy + HEX_SIZE * Math.sin(angle),
    ];
  });
}

function pointsAttribute(points: [number, number][]): string {
  return points.map(([x, y]) => `${x},${y}`).join(" ");
}

function splitPoints(coord: HexCoord, rotation: number): string {
  const corners = cornerPoints(coord);
  return pointsAttribute(
    Array.from({ length: 4 }, (_, index) => corners[(rotation + index) % 6]),
  );
}

function wildlifeLabel(wildlife: Wildlife): string {
  return {
    bear: "BR",
    elk: "EL",
    salmon: "SA",
    hawk: "HK",
    fox: "FX",
  }[wildlife];
}

export function HexBoard({
  board,
  active,
  placements,
  selectedCoord,
  wildlifeTargets,
  selectedWildlife,
  draftedWildlife,
  preview,
  onSelectCoord,
  onSelectWildlife,
}: HexBoardProps) {
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const drag = useRef<{ x: number; y: number; panX: number; panY: number } | null>(
    null,
  );
  const legalCoords = useMemo(
    () => new Set(placements.map((placement) => coordKey(placement.coord))),
    [placements],
  );
  const targetCoords = useMemo(
    () => new Set(wildlifeTargets.map(coordKey)),
    [wildlifeTargets],
  );
  const allCoords = useMemo(
    () => [...board.tiles.map((tile) => tile.coord), ...board.frontier],
    [board],
  );
  const bounds = useMemo(() => {
    if (allCoords.length === 0) return { x: -400, y: -300, width: 800, height: 600 };
    const centers = allCoords.map(center);
    const xs = centers.map(([x]) => x);
    const ys = centers.map(([, y]) => y);
    const minX = Math.min(...xs) - HEX_SIZE * 2;
    const maxX = Math.max(...xs) + HEX_SIZE * 2;
    const minY = Math.min(...ys) - HEX_SIZE * 2;
    const maxY = Math.max(...ys) + HEX_SIZE * 2;
    return { x: minX, y: minY, width: maxX - minX, height: maxY - minY };
  }, [allCoords]);

  function pointerDown(event: ReactPointerEvent<SVGSVGElement>) {
    if ((event.target as Element).closest("[data-board-action]")) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    drag.current = {
      x: event.clientX,
      y: event.clientY,
      panX: pan.x,
      panY: pan.y,
    };
  }

  function pointerMove(event: ReactPointerEvent<SVGSVGElement>) {
    if (!drag.current) return;
    setPan({
      x: drag.current.panX + event.clientX - drag.current.x,
      y: drag.current.panY + event.clientY - drag.current.y,
    });
  }

  function pointerUp(event: ReactPointerEvent<SVGSVGElement>) {
    if (!drag.current) return;
    drag.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }

  function wheel(event: WheelEvent<SVGSVGElement>) {
    event.preventDefault();
    setZoom((value) => Math.min(2.2, Math.max(0.55, value * (event.deltaY > 0 ? 0.9 : 1.1))));
  }

  const transform = `translate(${pan.x} ${pan.y}) scale(${zoom})`;

  return (
    <div className="board-stage">
      <svg
        className="hex-board"
        viewBox={`${bounds.x} ${bounds.y} ${bounds.width} ${bounds.height}`}
        role="application"
        aria-label={`Player ${board.player + 1} Cascadia board`}
        onPointerDown={pointerDown}
        onPointerMove={pointerMove}
        onPointerUp={pointerUp}
        onPointerCancel={() => {
          drag.current = null;
        }}
        onWheel={wheel}
      >
        <g transform={transform}>
          {board.frontier.map((coord) => {
            const legal = active && legalCoords.has(coordKey(coord));
            const selected = sameCoord(coord, selectedCoord);
            return (
              <polygon
                key={`frontier-${coordKey(coord)}`}
                data-board-action={legal ? "placement" : undefined}
                className={`hex-frontier ${legal ? "is-legal" : ""} ${
                  selected ? "is-selected" : ""
                }`}
                points={pointsAttribute(cornerPoints(coord))}
                onClick={() => legal && onSelectCoord(coord)}
                role={legal ? "button" : undefined}
                aria-label={legal ? `Place tile at ${coord.q}, ${coord.r}` : undefined}
              />
            );
          })}
          {board.tiles.map((placed) => (
            <TileHex
              key={`tile-${coordKey(placed.coord)}`}
              coord={placed.coord}
              tile={placed.tile}
              rotation={placed.rotation}
              wildlife={placed.wildlife}
              pendingWildlife={
                sameCoord(placed.coord, selectedWildlife) ? draftedWildlife : null
              }
              target={active && targetCoords.has(coordKey(placed.coord))}
              selectedWildlife={sameCoord(placed.coord, selectedWildlife)}
              onSelectWildlife={onSelectWildlife}
            />
          ))}
          {preview && (
            <TileHex
              coord={preview.coord}
              tile={preview.tile}
              rotation={preview.rotation}
              wildlife={preview.wildlife}
              pendingWildlife={
                sameCoord(preview.coord, selectedWildlife) ? draftedWildlife : null
              }
              preview
              target={targetCoords.has(coordKey(preview.coord))}
              selectedWildlife={sameCoord(preview.coord, selectedWildlife)}
              onSelectWildlife={onSelectWildlife}
            />
          )}
        </g>
      </svg>
      <div className="board-tools" aria-label="Board view controls">
        <button
          className="icon-button"
          type="button"
          title="Zoom in"
          onClick={() => setZoom((value) => Math.min(2.2, value * 1.15))}
        >
          <Plus aria-hidden="true" />
        </button>
        <button
          className="icon-button"
          type="button"
          title="Zoom out"
          onClick={() => setZoom((value) => Math.max(0.55, value / 1.15))}
        >
          <Minus aria-hidden="true" />
        </button>
        <button
          className="icon-button"
          type="button"
          title="Center board"
          onClick={() => {
            setZoom(1);
            setPan({ x: 0, y: 0 });
          }}
        >
          <LocateFixed aria-hidden="true" />
        </button>
      </div>
    </div>
  );
}

function TileHex({
  coord,
  tile,
  rotation,
  wildlife,
  pendingWildlife,
  preview = false,
  target,
  selectedWildlife,
  onSelectWildlife,
}: {
  coord: HexCoord;
  tile: TileView;
  rotation: number;
  wildlife: Wildlife | null;
  pendingWildlife: Wildlife | null;
  preview?: boolean;
  target: boolean;
  selectedWildlife: boolean;
  onSelectWildlife: (coord: HexCoord) => void;
}) {
  const [cx, cy] = center(coord);
  const canSelect = target && wildlife === null;
  const visibleWildlife = wildlife ?? pendingWildlife;
  return (
    <g
      className={`board-tile ${preview ? "is-preview" : ""} ${
        target ? "is-wildlife-target" : ""
      } ${selectedWildlife ? "is-wildlife-selected" : ""}`}
      data-board-action={canSelect ? "wildlife" : undefined}
      onClick={() => canSelect && onSelectWildlife(coord)}
      role={canSelect ? "button" : undefined}
      aria-label={canSelect ? `Place wildlife at ${coord.q}, ${coord.r}` : undefined}
    >
      <polygon
        className={`terrain-fill terrain-${tile.terrain_a}`}
        points={pointsAttribute(cornerPoints(coord))}
      />
      {tile.terrain_b && (
        <polygon
          className={`terrain-fill terrain-${tile.terrain_b}`}
          points={splitPoints(coord, rotation)}
        />
      )}
      <polygon className="board-tile-outline" points={pointsAttribute(cornerPoints(coord))} />
      {!visibleWildlife &&
        tile.wildlife.map((allowed, index) => {
          const angle = (index / tile.wildlife.length) * Math.PI * 2;
          return (
            <circle
              key={allowed}
              className={`allowed-dot wildlife-${allowed}-fill`}
              cx={cx + Math.cos(angle) * 14}
              cy={cy + Math.sin(angle) * 14}
              r="4"
            />
          );
        })}
      {visibleWildlife && (
        <g
          className={`board-wildlife wildlife-${visibleWildlife}-fill ${
            pendingWildlife ? "is-pending" : ""
          }`}
        >
          <circle cx={cx} cy={cy} r="19" />
          <text x={cx} y={cy + 1}>
            {wildlifeLabel(visibleWildlife)}
          </text>
        </g>
      )}
      {tile.keystone && !visibleWildlife && (
        <path
          className="board-keystone"
          d={`M ${cx} ${cy - 8} l 7 8 -7 8 -7 -8 z`}
        />
      )}
    </g>
  );
}
