import { readFileSync, writeFileSync, mkdirSync, readdirSync } from "fs"
import { join, dirname, basename } from "path"
import { fileURLToPath } from "url"

import { detectCards } from "../lib/detect-cards"
import { findAllSets, type SetCard } from "../lib/set-game"

const __dirname = dirname(fileURLToPath(import.meta.url))

const SERVER_URL = process.env.SET_ID_URL || "http://127.0.0.1:8000"

type CardLogits = {
  number: Record<string, number>
  color: Record<string, number>
  shape: Record<string, number>
  shading: Record<string, number>
}

type PredResp = {
  predictions: {
    color: SetCard["color"]
    shape: SetCard["shape"]
    shading: SetCard["shading"]
    number: "1" | "2" | "3"
    logits: CardLogits
  }[]
}

async function predictBatch(crops: Buffer[]): Promise<PredResp> {
  const images = crops.map((b) => b.toString("base64"))
  const resp = await fetch(`${SERVER_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ images }),
  })
  if (!resp.ok) {
    throw new Error(`predict ${resp.status}: ${await resp.text()}`)
  }
  return (await resp.json()) as PredResp
}

async function analyzeImage(imagePath: string) {
  const imageBuffer = readFileSync(imagePath)

  const { crops } = await detectCards(imageBuffer)
  console.log(`  CV detected ${crops.length} cards`)

  if (crops.length === 0) {
    return { cards: [], validSets: [], confidence: "high" as const, notes: "no cards detected", totalCards: 0, totalSets: 0 }
  }

  const { predictions } = await predictBatch(crops)
  const cards = predictions.map((p, i) => {
    console.log(`    card-${i + 1}: ${p.number} ${p.color} ${p.shape} ${p.shading}`)
    return {
      id: `card-${i + 1}`,
      color: p.color,
      shape: p.shape,
      shading: p.shading,
      number: parseInt(p.number, 10) as 1 | 2 | 3,
      position: { x: 0, y: 0 },
      logits: p.logits,
    }
  })

  const validSets = findAllSets(cards as unknown as SetCard[])
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
  const outName = process.argv[2]
  if (!outName) {
    console.error("Usage: npx tsx scripts/analyze-two-pass-local.ts <output-dir>")
    console.error("       SET_ID_URL=http://host:port npx tsx ... to override server URL")
    process.exit(1)
  }

  // Fail fast if the server isn't reachable.
  try {
    const health = await fetch(`${SERVER_URL}/health`)
    if (!health.ok) throw new Error(`health ${health.status}`)
    console.log(`server: ${SERVER_URL} — ${await health.text()}`)
  } catch (e) {
    console.error(`server at ${SERVER_URL} not reachable: ${e}`)
    console.error(`start it with: (cd ml && uv run python -m set_id.serve)`)
    process.exit(1)
  }

  const inputDir = join(__dirname, "..", "test-images", "fixed")
  const outputDir = join(__dirname, "..", "test-images", "results", outName)

  const files = readdirSync(inputDir)
    .filter((f) => /\.(jpg|jpeg|png)$/i.test(f))
    .sort()
  if (files.length === 0) {
    console.log("No images found in", inputDir)
    process.exit(1)
  }

  mkdirSync(outputDir, { recursive: true })
  for (const file of files) {
    const imagePath = join(inputDir, file)
    const outputPath = join(outputDir, basename(file, ".jpg") + ".json")
    console.log(`Analyzing ${file}...`)
    try {
      const result = await analyzeImage(imagePath)
      writeFileSync(outputPath, JSON.stringify(result, null, 2))
      console.log(`  → ${result.totalCards} cards, ${result.totalSets} sets`)
    } catch (error) {
      console.error(`  → Error analyzing ${file}:`, error)
    }
  }
  console.log(`\nResults written to ${outputDir}`)
}

main()
