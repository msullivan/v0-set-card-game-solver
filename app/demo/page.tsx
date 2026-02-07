"use client"

import { SetCardDisplay } from "@/components/set-card-display"
import { type CardColor, type CardShape, type CardShading, type CardNumber, type SetCard } from "@/lib/set-game"

const colors: CardColor[] = ["red", "green", "purple"]
const shapes: CardShape[] = ["diamond", "oval", "squiggle"]
const shadings: CardShading[] = ["solid", "striped", "empty"]
const numbers: CardNumber[] = [1, 2, 3]

function allCards(): SetCard[] {
  const cards: SetCard[] = []
  let id = 0
  for (const color of colors) {
    for (const shape of shapes) {
      for (const shading of shadings) {
        for (const number of numbers) {
          cards.push({ id: `card-${++id}`, color, shape, shading, number })
        }
      }
    }
  }
  return cards
}

export default function DemoPage() {
  const cards = allCards()

  return (
    <main className="min-h-screen bg-background p-8">
      <h1 className="text-2xl font-bold mb-8">All 81 Set Cards</h1>
      {colors.map((color) => (
        <div key={color} className="mb-10">
          <h2 className="text-xl font-semibold capitalize mb-4">{color}</h2>
          {shapes.map((shape) => (
            <div key={shape} className="mb-6">
              <h3 className="text-sm font-medium text-muted-foreground capitalize mb-2">{shape}</h3>
              <div className="grid grid-cols-9 gap-3 max-w-3xl">
                {shadings.map((shading) =>
                  numbers.map((number) => {
                    const card = cards.find(
                      (c) => c.color === color && c.shape === shape && c.shading === shading && c.number === number
                    )!
                    return <SetCardDisplay key={card.id} card={card} size="lg" />
                  })
                )}
              </div>
              <div className="grid grid-cols-9 gap-3 max-w-3xl mt-1">
                {shadings.flatMap((shading) =>
                  numbers.map((number) => (
                    <span key={`${shading}-${number}`} className="text-[10px] text-muted-foreground text-center">
                      {number} {shading}
                    </span>
                  ))
                )}
              </div>
            </div>
          ))}
        </div>
      ))}
    </main>
  )
}
