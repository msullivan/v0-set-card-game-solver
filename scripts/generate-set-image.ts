import { readFileSync, writeFileSync } from "fs"
import { join, dirname } from "path"
import { fileURLToPath } from "url"

const __dirname = dirname(fileURLToPath(import.meta.url))

// Load .env.local
const envPath = join(__dirname, "..", ".env.local")
for (const line of readFileSync(envPath, "utf-8").split("\n")) {
  const match = line.match(/^\s*([^#=]+?)\s*=\s*(.*)\s*$/)
  if (match) process.env[match[1]] = match[2]
}

import { generateText } from "ai"

const COLORS = ["red", "green", "purple"] as const
const SHAPES = ["diamond", "oval", "squiggle"] as const
const SHADINGS = ["solid", "striped", "empty"] as const
const NUMBERS = [1, 2, 3] as const

interface Card {
  color: (typeof COLORS)[number]
  shape: (typeof SHAPES)[number]
  shading: (typeof SHADINGS)[number]
  number: (typeof NUMBERS)[number]
}

function generateCards(n: number = 12): Card[] {
  const all: Card[] = []
  for (const color of COLORS)
    for (const shape of SHAPES)
      for (const shading of SHADINGS)
        for (const number of NUMBERS)
          all.push({ color, shape, shading, number })

  // Fisher-Yates shuffle and take n
  for (let i = all.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[all[i], all[j]] = [all[j], all[i]]
  }
  return all.slice(0, n)
}

function formatCardList(cards: Card[]): string {
  return cards
    .map((card, i) => {
      const row = Math.floor(i / 4) + 1
      const col = (i % 4) + 1
      return `  Row ${row}, Col ${col}: ${card.number} ${card.color} ${card.shading} ${card.shape}(s)`
    })
    .join("\n")
}

function buildPrompt(cards: Card[]): string {
  const cardList = formatCardList(cards).replaceAll("empty", "outline")
  return `This is a reference photo of real Set game cards. The three shapes are:

1. DIAMOND: a four-sided rhombus shape, oriented horizontally (wider than tall)
2. OVAL: a rounded rectangle / stadium shape, oriented horizontally
3. SQUIGGLE: a fat blobby bean/slug shape with smooth organic curves — NOT an S or a 2.

All three shadings exist: solid (100% filled with color), striped (horizontal lines inside the shape), and outline (just the colored border, white/blank inside — 0% fill).

Generate a new photo that looks similar to the reference — same style, same type of cards, same wooden table, same overhead perspective. ALL cards must be in portrait orientation (taller than wide), with shapes stacked vertically. All shapes should be wider than they are tall.

The cards must be exactly these 12, arranged in a 4 columns x 3 rows grid (left to right, top to bottom):
${cardList}`
}

async function main() {
  const args = process.argv.slice(2)
  const referencePath = args[0]
  const outputIdx = args.indexOf("-o")
  const outputPath = outputIdx !== -1 ? args[outputIdx + 1] : null
  const numIdx = args.indexOf("-n")
  const numCards = numIdx !== -1 ? parseInt(args[numIdx + 1], 10) : 12

  if (!referencePath || !outputPath) {
    console.error("Usage: npx tsx scripts/generate-set-image.ts <reference-image> -o <output-path> [-n num-cards]")
    process.exit(1)
  }

  const refImage = readFileSync(referencePath)
  const cards = generateCards(numCards)
  const prompt = buildPrompt(cards)

  console.log("Generated cards:")
  console.log(formatCardList(cards))
  console.log()

  console.log("Generating Set card image with Gemini Pro...")
  const result = await generateText({
    model: "google/gemini-3-pro-image",
    messages: [
      {
        role: "user",
        content: [
          { type: "image", image: refImage },
          { type: "text", text: prompt },
        ],
      },
    ],
  })

  if (result.files.length === 0) {
    console.error("No image returned.")
    if (result.text) console.error("Text response:", result.text.slice(0, 500))
    process.exit(1)
  }

  const imageFile = result.files[0]
  const imgBuffer = Buffer.from(imageFile.uint8Array)
  writeFileSync(outputPath, imgBuffer)
  console.log(`Image saved to ${outputPath}`)

  // Save expected cards as JSON sidecar
  const jsonPath = outputPath.replace(/\.[^.]+$/, ".json")
  writeFileSync(jsonPath, JSON.stringify(cards, null, 2))
  console.log(`Expected cards saved to ${jsonPath}`)
}

main()
