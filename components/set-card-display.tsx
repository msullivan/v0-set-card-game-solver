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
  // Hand-tuned cubic outline based on the trace from test-images/crops/103.jpg.
  // The left lobe is deliberately rounded out to avoid the traced pinch.
  squiggle:
    "M 75.5,5.6 C 66.5,9.7 58.0,8.7 48.0,6.7 C 37.0,4.5 26.8,3.1 17.5,7.3 C 8.8,11.2 2.7,20.2 1.1,31.0 C -0.4,41.4 3.3,47.5 9.8,48.5 C 15.8,49.4 21.6,45.5 28.2,40.7 C 37.2,34.1 47.8,37.8 59.4,41.6 C 70.9,45.4 82.5,47.5 91.1,39.4 C 98.0,32.9 100.6,19.2 97.5,10.0 C 95.1,2.7 90.5,-0.4 85.2,1.5 C 82.1,2.6 79.0,4.0 75.5,5.6 Z",
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
