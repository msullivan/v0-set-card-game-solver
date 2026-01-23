"use client"

import { useState } from "react"
import { type SetCard, type ValidSet, formatCard } from "@/lib/set-game"
import { SetCardDisplay, CardGrid } from "@/components/set-card-display"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { CheckCircle2, AlertCircle, Sparkles } from "lucide-react"
import { cn } from "@/lib/utils"

interface SetResultsProps {
  cards: SetCard[]
  validSets: ValidSet[]
  confidence: "high" | "medium" | "low"
  notes?: string
}

export function SetResults({ cards, validSets, confidence, notes }: SetResultsProps) {
  const [hoveredSetIndex, setHoveredSetIndex] = useState<number | null>(null)

  const highlightedIds =
    hoveredSetIndex !== null
      ? validSets[hoveredSetIndex].cards.map((c) => c.id)
      : []

  const confidenceConfig = {
    high: { label: "High confidence", color: "bg-emerald-100 text-emerald-700" },
    medium: { label: "Medium confidence", color: "bg-yellow-100 text-yellow-700" },
    low: { label: "Low confidence", color: "bg-red-100 text-red-700" },
  }

  return (
    <div className="space-y-6">
      {/* Summary */}
      <div className="flex flex-wrap items-center gap-3">
        <Badge variant="secondary" className="text-sm py-1 px-3">
          {cards.length} cards detected
        </Badge>
        <Badge
          variant="secondary"
          className={cn("text-sm py-1 px-3", confidenceConfig[confidence].color)}
        >
          {confidenceConfig[confidence].label}
        </Badge>
        {validSets.length > 0 ? (
          <Badge className="text-sm py-1 px-3 bg-primary text-primary-foreground">
            <Sparkles className="w-3 h-3 mr-1" />
            {validSets.length} valid {validSets.length === 1 ? "set" : "sets"} found
          </Badge>
        ) : (
          <Badge variant="destructive" className="text-sm py-1 px-3">
            No valid sets found
          </Badge>
        )}
      </div>

      {notes && (
        <div className="flex items-start gap-2 p-3 rounded-lg bg-muted">
          <AlertCircle className="w-4 h-4 text-muted-foreground mt-0.5 flex-shrink-0" />
          <p className="text-sm text-muted-foreground">{notes}</p>
        </div>
      )}

      {/* Detected Cards */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-lg">Detected Cards</CardTitle>
        </CardHeader>
        <CardContent>
          <CardGrid cards={cards} highlightedIds={highlightedIds} />
        </CardContent>
      </Card>

      {/* Valid Sets */}
      {validSets.length > 0 && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center gap-2">
              <CheckCircle2 className="w-5 h-5 text-primary" />
              Valid Sets
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {validSets.map((set, index) => (
                <div
                  key={index}
                  className={cn(
                    "p-4 rounded-lg border-2 transition-all cursor-pointer",
                    hoveredSetIndex === index
                      ? "border-primary bg-accent"
                      : "border-border hover:border-primary/50"
                  )}
                  onMouseEnter={() => setHoveredSetIndex(index)}
                  onMouseLeave={() => setHoveredSetIndex(null)}
                >
                  <div className="flex items-center gap-2 mb-3">
                    <Badge variant="outline" className="font-mono">
                      Set {index + 1}
                    </Badge>
                  </div>
                  <div className="flex items-center gap-3 mb-3">
                    {set.cards.map((card, cardIndex) => (
                      <SetCardDisplay key={card.id} card={card} size="sm" />
                    ))}
                  </div>
                  <p className="text-sm text-muted-foreground">
                    {set.reason}
                  </p>
                  <div className="mt-2 flex flex-wrap gap-1">
                    {set.cards.map((card) => (
                      <Badge key={card.id} variant="secondary" className="text-xs">
                        {formatCard(card)}
                      </Badge>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* No Sets Found */}
      {validSets.length === 0 && cards.length >= 3 && (
        <Card className="border-dashed">
          <CardContent className="py-8 text-center">
            <AlertCircle className="w-12 h-12 mx-auto text-muted-foreground mb-3" />
            <p className="text-lg font-medium text-foreground">No valid sets found</p>
            <p className="text-sm text-muted-foreground mt-1">
              The {cards.length} detected cards don't contain any valid Set combinations.
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
