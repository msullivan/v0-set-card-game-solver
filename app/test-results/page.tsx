"use client"

// Test fixture page for Playwright tests — not intended for production use.
// Renders SetResults with hardcoded mock cards so the edit-card feature
// can be tested without going through the file upload + AI analysis flow.

import { SetResults } from "@/components/set-results"
import { type SetCard } from "@/lib/set-game"

const MOCK_CARDS: SetCard[] = [
  { id: "c1", color: "red",    shape: "diamond", shading: "solid", number: 1 },
  { id: "c2", color: "green",  shape: "diamond", shading: "solid", number: 1 },
  { id: "c3", color: "purple", shape: "diamond", shading: "solid", number: 1 },
]

const MOCK_VALID_SETS = [
  {
    cards: MOCK_CARDS as [SetCard, SetCard, SetCard],
    reason: "different color (red, green, purple), all diamond shape, all solid shading, all 1 number",
  },
]

export default function TestResultsPage() {
  return (
    <main className="max-w-4xl mx-auto px-4 py-8">
      <SetResults
        cards={MOCK_CARDS}
        validSets={MOCK_VALID_SETS}
        confidence="high"
      />
    </main>
  )
}
