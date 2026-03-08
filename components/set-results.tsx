"use client"

import { useState } from "react"
import { type SetCard, type ValidSet, type CardColor, type CardShape, type CardShading, type CardNumber, formatCard, findAllSets } from "@/lib/set-game"
import { SetCardDisplay, CardGrid } from "@/components/set-card-display"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { CheckCircle2, AlertCircle, Sparkles } from "lucide-react"
import { cn } from "@/lib/utils"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"

interface SetResultsProps {
  cards: SetCard[]
  validSets: ValidSet[]
  confidence: "high" | "medium" | "low"
  notes?: string
}

const COLORS: CardColor[] = ["red", "green", "purple"]
const SHAPES: CardShape[] = ["diamond", "oval", "squiggle"]
const SHADINGS: CardShading[] = ["solid", "striped", "empty"]
const NUMBERS: CardNumber[] = [1, 2, 3]

const colorDot: Record<CardColor, string> = {
  red: "bg-red-500",
  green: "bg-emerald-500",
  purple: "bg-purple-500",
}

export function SetResults({ cards: initialCards, validSets: initialValidSets, confidence, notes }: SetResultsProps) {
  const [cards, setCards] = useState<SetCard[]>(initialCards)
  const [validSets, setValidSets] = useState<ValidSet[]>(initialValidSets)
  const [hoveredSetIndex, setHoveredSetIndex] = useState<number | null>(null)
  const [editingCard, setEditingCard] = useState<SetCard | null>(null)

  const highlightedIds =
    hoveredSetIndex !== null
      ? validSets[hoveredSetIndex].cards.map((c) => c.id)
      : []

  const confidenceConfig = {
    high: { label: "High confidence", color: "bg-emerald-100 text-emerald-700" },
    medium: { label: "Medium confidence", color: "bg-yellow-100 text-yellow-700" },
    low: { label: "Low confidence", color: "bg-red-100 text-red-700" },
  }

  const handleCardClick = (card: SetCard) => {
    setEditingCard({ ...card })
  }

  const handleSaveCard = () => {
    if (!editingCard) return
    const newCards = cards.map((c) => (c.id === editingCard.id ? editingCard : c))
    setCards(newCards)
    setValidSets(findAllSets(newCards))
    setEditingCard(null)
  }

  const updateDraft = (patch: Partial<SetCard>) => {
    setEditingCard((prev) => (prev ? { ...prev, ...patch } : null))
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
          <p className="text-sm text-muted-foreground">Click a card to correct a misidentification</p>
        </CardHeader>
        <CardContent>
          <CardGrid cards={cards} highlightedIds={highlightedIds} onCardClick={handleCardClick} />
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
                    {set.cards.map((card) => (
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
              The {cards.length} detected cards don&apos;t contain any valid Set combinations.
            </p>
          </CardContent>
        </Card>
      )}

      {/* Card Edit Dialog */}
      <Dialog open={editingCard !== null} onOpenChange={(open) => !open && setEditingCard(null)}>
        <DialogContent className="sm:max-w-sm">
          <DialogHeader>
            <DialogTitle>Edit Card</DialogTitle>
            <DialogDescription>
              Correct any misidentified attributes for this card.
            </DialogDescription>
          </DialogHeader>

          {editingCard && (
            <div className="space-y-4">
              {/* Preview */}
              <div className="flex justify-center py-2">
                <SetCardDisplay card={editingCard} size="lg" />
              </div>

              {/* Color */}
              <div className="space-y-1.5">
                <p className="text-sm font-medium">Color</p>
                <div className="flex gap-2">
                  {COLORS.map((color) => (
                    <Button
                      key={color}
                      variant={editingCard.color === color ? "default" : "outline"}
                      size="sm"
                      className="flex-1"
                      onClick={() => updateDraft({ color })}
                    >
                      <span className={cn("w-3 h-3 rounded-full mr-1.5 border border-white/30", colorDot[color])} />
                      {color.charAt(0).toUpperCase() + color.slice(1)}
                    </Button>
                  ))}
                </div>
              </div>

              {/* Shape */}
              <div className="space-y-1.5">
                <p className="text-sm font-medium">Shape</p>
                <div className="flex gap-2">
                  {SHAPES.map((shape) => (
                    <Button
                      key={shape}
                      variant={editingCard.shape === shape ? "default" : "outline"}
                      size="sm"
                      className="flex-1 capitalize"
                      onClick={() => updateDraft({ shape })}
                    >
                      {shape}
                    </Button>
                  ))}
                </div>
              </div>

              {/* Shading */}
              <div className="space-y-1.5">
                <p className="text-sm font-medium">Shading</p>
                <div className="flex gap-2">
                  {SHADINGS.map((shading) => (
                    <Button
                      key={shading}
                      variant={editingCard.shading === shading ? "default" : "outline"}
                      size="sm"
                      className="flex-1 capitalize"
                      onClick={() => updateDraft({ shading })}
                    >
                      {shading}
                    </Button>
                  ))}
                </div>
              </div>

              {/* Number */}
              <div className="space-y-1.5">
                <p className="text-sm font-medium">Number</p>
                <div className="flex gap-2">
                  {NUMBERS.map((number) => (
                    <Button
                      key={number}
                      variant={editingCard.number === number ? "default" : "outline"}
                      size="sm"
                      className="flex-1"
                      onClick={() => updateDraft({ number })}
                    >
                      {number}
                    </Button>
                  ))}
                </div>
              </div>
            </div>
          )}

          <DialogFooter>
            <Button variant="outline" onClick={() => setEditingCard(null)}>
              Cancel
            </Button>
            <Button onClick={handleSaveCard}>Save</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
