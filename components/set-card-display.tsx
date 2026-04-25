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
  // Traced from test-images/crops/103.jpg (a real solid-red squiggle card),
  // contour simplified with RDP eps=3, smoothed via midpoint quadratic Béziers.
  squiggle:
    "M 80.99,3.94 Q 75.68,7.02 70.21,8.73 Q 64.73,10.45 61.13,10.27 Q 57.53,10.10 47.43,7.19 Q 37.33,4.28 31.85,4.45 Q 26.37,4.62 20.38,6.51 Q 14.38,8.39 9.42,13.36 Q 4.45,18.32 2.23,24.14 Q 0,29.97 0.68,36.82 Q 1.37,43.66 4.28,46.40 Q 7.19,49.14 10.27,48.46 Q 13.36,47.77 20.21,43.32 Q 27.05,38.87 29.62,38.53 Q 32.19,38.18 37.67,38.18 Q 43.15,38.18 56.16,42.29 Q 69.18,46.40 73.97,46.40 Q 78.77,46.40 82.71,45.21 Q 86.64,44.01 91.44,39.38 Q 96.23,34.76 98.12,28.08 Q 100,21.40 99.66,16.61 Q 99.32,11.82 97.77,8.39 Q 96.23,4.97 94.35,3.25 Q 92.47,1.54 89.55,1.20 Q 86.64,0.86 86.47,0.86 Q 86.30,0.86 80.99,3.94 Z",
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
