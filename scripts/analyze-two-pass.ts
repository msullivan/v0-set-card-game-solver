import { readFileSync, writeFileSync, mkdirSync, readdirSync } from "fs"
import { join, dirname, basename } from "path"
import { fileURLToPath } from "url"

const __dirname = dirname(fileURLToPath(import.meta.url))

// Load .env.local
const envPath = join(__dirname, "..", ".env.local")
for (const line of readFileSync(envPath, "utf-8").split("\n")) {
  const match = line.match(/^\s*([^#=]+?)\s*=\s*(.*)\s*$/)
  if (match) process.env[match[1]] = match[2]
}

import { detectCards } from "../lib/detect-cards"
import { analyzeCard } from "../lib/analyze-card"
import { findAllSets, type SetCard } from "../lib/set-game"

async function analyzeImage(imagePath: string, model: string) {
  const imageBuffer = readFileSync(imagePath)

  // Pass 1: CV card detection
  const { crops } = await detectCards(imageBuffer)
  console.log(`  CV detected ${crops.length} cards`)

  // Pass 2: AI analysis of each card (parallel)
  const cards = await Promise.all(
    crops.map(async (crop, i) => {
      const attrs = await analyzeCard(crop, model)
      console.log(`    card-${i + 1}: ${attrs.number} ${attrs.color} ${attrs.shape} ${attrs.shading}`)
      return {
        id: `card-${i + 1}`,
        color: attrs.color,
        shape: attrs.shape,
        shading: attrs.shading,
        number: parseInt(attrs.number, 10) as 1 | 2 | 3,
        position: { x: 0, y: 0 },
      } as SetCard
    })
  )

  const validSets = findAllSets(cards)

  return {
    cards,
    validSets,
    confidence: "high" as const,
    notes: "",
    totalCards: cards.length,
    totalSets: validSets.length,
  }
}

async function main() {
  const model = process.argv[2]
  const outName = process.argv[3]

  if (!model || !outName) {
    console.error("Usage: npx tsx scripts/analyze-two-pass.ts <model> <output-dir>")
    console.error("Example: npx tsx scripts/analyze-two-pass.ts anthropic/claude-sonnet-4-20250514 sonnet-4-two-pass")
    process.exit(1)
  }

  const inputDir = join(__dirname, "..", "test-images", "fixed")
  const outputDir = join(__dirname, "..", "test-images", "results", outName)

  const files = readdirSync(inputDir)
    .filter(f => /\.(jpg|jpeg|png)$/i.test(f))
    .sort()

  if (files.length === 0) {
    console.log("No images found in", inputDir)
    process.exit(1)
  }

  mkdirSync(outputDir, { recursive: true })

  for (const file of files) {
    const imagePath = join(inputDir, file)
    const outputName = basename(file, ".jpg") + ".json"
    const outputPath = join(outputDir, outputName)

    console.log(`Analyzing ${file}...`)
    try {
      const result = await analyzeImage(imagePath, model)
      writeFileSync(outputPath, JSON.stringify(result, null, 2))
      console.log(`  → ${result.totalCards} cards, ${result.totalSets} sets`)
    } catch (error) {
      console.error(`  → Error analyzing ${file}:`, error)
    }
  }

  console.log(`\nResults written to ${outputDir}`)
}

main()
