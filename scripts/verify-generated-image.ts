import { readFileSync } from "fs"
import { join, dirname } from "path"
import { fileURLToPath } from "url"

const __dirname = dirname(fileURLToPath(import.meta.url))

// Load .env.local
const envPath = join(__dirname, "..", ".env.local")
for (const line of readFileSync(envPath, "utf-8").split("\n")) {
  const match = line.match(/^\s*([^#=]+?)\s*=\s*(.*)\s*$/)
  if (match) process.env[match[1]] = match[2]
}

import { detectCards } from "../lib/detect-cards-node"
import { analyzeCard } from "../lib/analyze-card"

interface ExpectedCard {
  color: string
  shape: string
  shading: string
  number: number
}

async function main() {
  const imagePath = process.argv[2]
  if (!imagePath) {
    console.error("Usage: npx tsx scripts/verify-generated-image.ts <image-path> [expected.json] [model]")
    console.error("\nIf expected.json is not provided, looks for a .json file next to the image.")
    process.exit(1)
  }

  // Find expected cards JSON
  const jsonPath = process.argv[3] || imagePath.replace(/\.[^.]+$/, ".json")
  let expected: ExpectedCard[] | null = null
  try {
    expected = JSON.parse(readFileSync(jsonPath, "utf-8"))
  } catch {
    console.log(`No expected cards file found at ${jsonPath}, will just print detected cards.\n`)
  }

  const model = process.argv[4] || "anthropic/claude-haiku-4-5-20251001"

  // Run two-pass analysis
  const imageBuffer = readFileSync(imagePath)
  const { crops } = await detectCards(imageBuffer)
  console.log(`CV detected ${crops.length} cards\n`)

  const detected = await Promise.all(
    crops.map(async (crop, i) => {
      const attrs = await analyzeCard(crop, model)
      return {
        index: i,
        color: attrs.color,
        shape: attrs.shape,
        shading: attrs.shading,
        number: parseInt(attrs.number, 10),
      }
    })
  )

  if (!expected) {
    for (const card of detected) {
      console.log(`  card-${card.index + 1}: ${card.number} ${card.color} ${card.shading} ${card.shape}`)
    }
    return
  }

  // Compare
  let correct = 0
  const total = expected.length

  console.log("Expected".padEnd(35) + "Detected".padEnd(35) + "Match")
  console.log("-".repeat(75))

  for (let i = 0; i < total; i++) {
    const exp = expected[i]
    const det = detected[i]

    const expStr = exp ? `${exp.number} ${exp.color} ${exp.shading} ${exp.shape}` : "???"
    const detStr = det ? `${det.number} ${det.color} ${det.shading} ${det.shape}` : "not detected"

    const match = det && exp
      && det.color === exp.color
      && det.shape === exp.shape
      && det.shading === exp.shading
      && det.number === exp.number

    if (match) correct++
    console.log(`  ${expStr.padEnd(33)}${detStr.padEnd(33)}${match ? "✓" : "✗"}`)
  }

  // Check for extra detected cards
  if (detected.length > total) {
    for (let i = total; i < detected.length; i++) {
      const det = detected[i]
      console.log(`  ${"".padEnd(33)}${`${det.number} ${det.color} ${det.shading} ${det.shape}`.padEnd(33)}extra`)
    }
  }

  console.log("-".repeat(75))
  console.log(`\n${correct}/${total} cards match (${detected.length} detected)`)
}

main()
