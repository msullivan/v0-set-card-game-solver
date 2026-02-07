import { readFileSync, writeFileSync, readdirSync, mkdirSync } from "fs"
import { join, dirname } from "path"
import { fileURLToPath } from "url"
import { findAllSets, type SetCard } from "../lib/set-game"

const __dirname = dirname(fileURLToPath(import.meta.url))

const sonnetDir = join(__dirname, "..", "test-images", "results", "sonnet-4")
const haikuDir = join(__dirname, "..", "test-images", "results", "haiku-4")
const outputDir = join(__dirname, "..", "test-images", "results", "merged")

// Corrections: use haiku's value for these specific cards
const corrections: Record<string, Record<string, Partial<SetCard>>> = {
  "003": { "card-7": { number: 2 } },  // haiku had 2, sonnet had 3
  "011": { "card-7": { number: 2 } },  // haiku had 2, sonnet had 3
}

mkdirSync(outputDir, { recursive: true })

const files = readdirSync(sonnetDir).filter((f) => f.endsWith(".json"))

for (const file of files) {
  const name = file.replace(".json", "")
  const data = JSON.parse(readFileSync(join(sonnetDir, file), "utf-8"))

  if (corrections[name]) {
    for (const [cardId, fix] of Object.entries(corrections[name])) {
      const card = data.cards.find((c: SetCard) => c.id === cardId)
      if (card) Object.assign(card, fix)
    }
    // Recalculate sets with corrected cards
    data.validSets = findAllSets(data.cards)
    data.totalSets = data.validSets.length
  }

  writeFileSync(join(outputDir, file), JSON.stringify(data, null, 2))
  console.log(`${file}: ${data.totalCards} cards, ${data.totalSets} sets`)
}

console.log(`\nMerged results written to ${outputDir}`)
