"use client"

import { type SetCard } from "@/lib/set-game"
import { cn } from "@/lib/utils"

interface SetCardDisplayProps {
  card: SetCard
  highlighted?: boolean
  size?: "sm" | "md" | "lg"
}

const colorMap = {
  red: "text-red-500",
  green: "text-emerald-500",
  purple: "text-purple-500",
}

const bgColorMap = {
  red: "bg-red-50 border-red-200",
  green: "bg-emerald-50 border-emerald-200",
  purple: "bg-purple-50 border-purple-200",
}

// SVG shapes for Set cards
function DiamondShape({ shading, color }: { shading: string; color: string }) {
  const fillColor = color === "red" ? "#ef4444" : color === "green" ? "#10b981" : "#a855f7"

  return (
    <svg viewBox="0 0 60 40" className="w-[60%] h-auto">
      <polygon
        points="5,20 30,5 55,20 30,35"
        fill={shading === "solid" ? fillColor : shading === "striped" ? `url(#stripe-${color})` : "none"}
        stroke={fillColor}
        strokeWidth="2"
      />
      <defs>
        <pattern id={`stripe-${color}`} patternUnits="userSpaceOnUse" width="4" height="4">
          <line x1="0" y1="0" x2="0" y2="4" stroke={fillColor} strokeWidth="1.5" />
        </pattern>
      </defs>
    </svg>
  )
}

function OvalShape({ shading, color }: { shading: string; color: string }) {
  const fillColor = color === "red" ? "#ef4444" : color === "green" ? "#10b981" : "#a855f7"

  return (
    <svg viewBox="0 0 60 40" className="w-[60%] h-auto">
      <ellipse
        cx="30"
        cy="20"
        rx="25"
        ry="15"
        fill={shading === "solid" ? fillColor : shading === "striped" ? `url(#stripe-oval-${color})` : "none"}
        stroke={fillColor}
        strokeWidth="2"
      />
      <defs>
        <pattern id={`stripe-oval-${color}`} patternUnits="userSpaceOnUse" width="4" height="4">
          <line x1="0" y1="0" x2="0" y2="4" stroke={fillColor} strokeWidth="1.5" />
        </pattern>
      </defs>
    </svg>
  )
}

function SquiggleShape({ shading, color }: { shading: string; color: string }) {
  const fillColor = color === "red" ? "#ef4444" : color === "green" ? "#10b981" : "#a855f7"

  return (
    <svg viewBox="0 0 60 40" className="w-[60%] h-auto">
      <path
        d="M5,28 C5,15 16,2 30,15 C44,22 55,2 55,12 C55,22 44,38 30,25 C16,18 5,38 5,28Z"
        fill={shading === "solid" ? fillColor : shading === "striped" ? `url(#stripe-squiggle-${color})` : "none"}
        stroke={fillColor}
        strokeWidth="2"
        strokeLinejoin="round"
      />
      <defs>
        <pattern id={`stripe-squiggle-${color}`} patternUnits="userSpaceOnUse" width="4" height="4">
          <line x1="0" y1="0" x2="0" y2="4" stroke={fillColor} strokeWidth="1.5" />
        </pattern>
      </defs>
    </svg>
  )
}

function ShapeRenderer({ shape, shading, color }: { shape: string; shading: string; color: string }) {
  switch (shape) {
    case "diamond":
      return <DiamondShape shading={shading} color={color} />
    case "oval":
      return <OvalShape shading={shading} color={color} />
    case "squiggle":
      return <SquiggleShape shading={shading} color={color} />
    default:
      return null
  }
}

export function SetCardDisplay({ card, highlighted, size = "md" }: SetCardDisplayProps) {
  const sizeClasses = {
    sm: "w-[60px]",
    md: "w-[80px]",
    lg: "w-[100px]",
  }

  return (
    <div
      className={cn(
        "rounded-lg border-2 aspect-[5/7] flex flex-col items-center justify-center transition-all",
        bgColorMap[card.color],
        sizeClasses[size],
        highlighted && "ring-2 ring-primary ring-offset-2 scale-105"
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
    </div>
  )
}

interface CardGridProps {
  cards: SetCard[]
  highlightedIds?: string[]
}

export function CardGrid({ cards, highlightedIds = [] }: CardGridProps) {
  const cols = Math.ceil(cards.length / 3)
  return (
    <div className="grid gap-3" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}>
      {cards.map((card) => (
        <SetCardDisplay
          key={card.id}
          card={card}
          highlighted={highlightedIds.includes(card.id)}
        />
      ))}
    </div>
  )
}
