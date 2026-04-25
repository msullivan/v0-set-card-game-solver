"use client"

import { type SetCard } from "@/lib/set-game"
import { cn } from "@/lib/utils"
import { Pencil } from "lucide-react"

interface SetCardDisplayProps {
  card: SetCard
  highlighted?: boolean
  size?: "sm" | "md" | "lg"
  onClick?: () => void
}

const bgColorMap = {
  red: "bg-red-50 border-red-200",
  green: "bg-emerald-50 border-emerald-200",
  purple: "bg-purple-50 border-purple-200",
}

// SVG shapes for Set cards. All paths use viewBox 0 0 100 50.
const SHAPE_PATHS = {
  diamond: "M 3,25 L 50,3 L 97,25 L 50,47 Z",
  oval: "M 25,3 L 75,3 A 22,22 0 0 1 75,47 L 25,47 A 22,22 0 0 1 25,3 Z",
  // Traced from test-images/crops/103.jpg via scripts/trace-shape.py
  // (RDP eps=4 → Chaikin x2 corner-cutting → polyline). Chaikin stays
  // inside the convex hull, so no overshoot/pinching at sparse joins.
  squiggle:
    "M 78.21,4.45 L 72.82,6.85 L 67.06,8.26 L 60.94,8.69 L 54.45,8.13 L 47.60,6.59 L 41.78,5.46 L 36.99,4.73 L 33.22,4.41 L 30.48,4.49 L 27.68,4.79 L 24.81,5.31 L 21.88,6.04 L 18.88,6.98 L 16.01,8.30 L 13.27,10.02 L 10.66,12.11 L 8.18,14.60 L 6.04,17.19 L 4.24,19.88 L 2.78,22.69 L 1.67,25.60 L 0.92,28.64 L 0.54,31.81 L 0.51,35.10 L 0.86,38.53 L 1.48,41.44 L 2.38,43.84 L 3.55,45.72 L 5.01,47.09 L 6.49,48.03 L 7.98,48.54 L 9.50,48.63 L 11.04,48.29 L 13.06,47.47 L 15.54,46.19 L 18.49,44.43 L 21.92,42.21 L 24.81,40.50 L 27.16,39.30 L 28.98,38.61 L 30.27,38.44 L 31.91,38.31 L 33.93,38.23 L 36.30,38.18 L 39.04,38.18 L 42.72,38.70 L 47.35,39.73 L 52.91,41.27 L 59.42,43.32 L 65.39,44.71 L 70.83,45.44 L 75.73,45.51 L 80.09,44.91 L 83.97,43.88 L 87.35,42.42 L 90.24,40.54 L 92.64,38.23 L 94.67,35.66 L 96.34,32.83 L 97.65,29.75 L 98.59,26.41 L 99.25,23.31 L 99.64,20.44 L 99.74,17.81 L 99.57,15.41 L 99.25,13.18 L 98.78,11.13 L 98.16,9.25 L 97.39,7.53 L 96.58,6.04 L 95.72,4.75 L 94.82,3.68 L 93.88,2.83 L 92.79,2.14 L 91.55,1.63 L 90.15,1.28 L 88.61,1.11 L 86.11,1.58 L 82.64,2.70 Z",
} as const

const FILL_COLOR = {
  red: "#ef4444",
  green: "#10b981",
  purple: "#a855f7",
} as const

function ShapeRenderer({
  shape,
  shading,
  color,
}: {
  shape: keyof typeof SHAPE_PATHS
  shading: string
  color: keyof typeof FILL_COLOR
}) {
  const fillColor = FILL_COLOR[color]
  const patternId = `stripe-${color}`
  const fill =
    shading === "solid" ? fillColor : shading === "striped" ? `url(#${patternId})` : "none"
  return (
    <svg viewBox="0 0 100 50" className="w-[70%] h-auto">
      <defs>
        <pattern id={patternId} patternUnits="userSpaceOnUse" width="4" height="4">
          <line x1="0" y1="0" x2="0" y2="4" stroke={fillColor} strokeWidth="1.5" />
        </pattern>
      </defs>
      <path
        d={SHAPE_PATHS[shape]}
        fill={fill}
        stroke={fillColor}
        strokeWidth="2"
        strokeLinejoin="round"
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  )
}

export function SetCardDisplay({ card, highlighted, size = "md", onClick }: SetCardDisplayProps) {
  const sizeClasses = {
    sm: "w-[60px]",
    md: "w-[80px]",
    lg: "w-[100px]",
  }

  return (
    <div
      onClick={onClick}
      role={onClick ? "button" : undefined}
      tabIndex={onClick ? 0 : undefined}
      aria-label={onClick ? `Edit card: ${card.number} ${card.color} ${card.shading} ${card.shape}` : undefined}
      onKeyDown={onClick ? (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onClick() } } : undefined}
      className={cn(
        "rounded-lg border-2 aspect-[5/7] flex flex-col items-center justify-center transition-all relative",
        bgColorMap[card.color],
        sizeClasses[size],
        highlighted && "ring-2 ring-primary ring-offset-2 scale-105",
        onClick && "cursor-pointer group hover:brightness-95"
      )}
    >
      <div className="flex flex-col items-center gap-0.5">
        {Array.from({ length: card.number }).map((_, i) => (
          <ShapeRenderer
            key={i}
            shape={card.shape}
            shading={card.shading}
            color={card.color}
          />
        ))}
      </div>
      {onClick && (
        <div className="absolute bottom-1 right-1 opacity-0 group-hover:opacity-100 transition-opacity">
          <div className="bg-black/40 rounded p-0.5">
            <Pencil className="w-2.5 h-2.5 text-white" />
          </div>
        </div>
      )}
    </div>
  )
}

interface CardGridProps {
  cards: SetCard[]
  highlightedIds?: string[]
  onCardClick?: (card: SetCard) => void
}

export function CardGrid({ cards, highlightedIds = [], onCardClick }: CardGridProps) {
  const cols = Math.ceil(cards.length / 3)
  return (
    <div className="grid gap-3" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}>
      {cards.map((card) => (
        <SetCardDisplay
          key={card.id}
          card={card}
          highlighted={highlightedIds.includes(card.id)}
          onClick={onCardClick ? () => onCardClick(card) : undefined}
        />
      ))}
    </div>
  )
}
