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
    <svg viewBox="0 0 40 60" className="w-6 h-9">
      <polygon
        points="20,5 35,30 20,55 5,30"
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
    <svg viewBox="0 0 40 60" className="w-6 h-9">
      <ellipse
        cx="20"
        cy="30"
        rx="15"
        ry="25"
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
    <svg viewBox="0 0 40 60" className="w-6 h-9">
      <path
        d="M10,10 Q5,20 15,30 Q25,40 15,50 Q10,55 20,55 Q35,55 30,45 Q25,35 30,25 Q35,15 25,10 Q15,5 10,10"
        fill={shading === "solid" ? fillColor : shading === "striped" ? `url(#stripe-squiggle-${color})` : "none"}
        stroke={fillColor}
        strokeWidth="2"
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
    sm: "p-2 min-w-[60px]",
    md: "p-3 min-w-[80px]",
    lg: "p-4 min-w-[100px]",
  }

  return (
    <div
      className={cn(
        "rounded-lg border-2 flex flex-col items-center justify-center gap-1 transition-all",
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
  return (
    <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 gap-3">
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
