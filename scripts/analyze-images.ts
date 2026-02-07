import { readFileSync, writeFileSync, readdirSync, mkdirSync } from "fs"
import { join, basename, dirname } from "path"
import { fileURLToPath } from "url"

const __dirname = dirname(fileURLToPath(import.meta.url))

// Load .env.local
const envPath = join(__dirname, "..", ".env.local")
for (const line of readFileSync(envPath, "utf-8").split("\n")) {
  const match = line.match(/^\s*([^#=]+?)\s*=\s*(.*)\s*$/)
  if (match) process.env[match[1]] = match[2]
}

import { generateObject } from "ai"
import { z } from "zod"
import { findAllSets, type SetCard } from "../lib/set-game"

const CardSchema = z.object({
  id: z.string().describe("Unique identifier like card-1, card-2, etc."),
  color: z.enum(["red", "green", "purple"]).describe("The color of the shapes on the card"),
  shape: z.enum(["diamond", "oval", "squiggle"]).describe("The shape type on the card"),
  shading: z.enum(["solid", "striped", "empty"]).describe("solid=filled, striped=has lines, empty=outline only"),
  number: z.enum(["1", "2", "3"]).describe("Count of shapes on the card as a string"),
  positionX: z.number().describe("Approximate x position (0-100) of the card in the image"),
  positionY: z.number().describe("Approximate y position (0-100) of the card in the image"),
})

const ResponseSchema = z.object({
  cards: z.array(CardSchema).describe("All Set game cards detected in the image"),
  confidence: z
    .enum(["high", "medium", "low"])
    .describe("How confident you are in the card detection"),
  notes: z
    .string()
    .describe("Any notes about card detection issues or unclear cards, or empty string if none"),
})

const PROMPT = `You are analyzing a photo of Set game cards. Carefully examine each card and identify its 4 attributes:

**COLOR** (look at the actual color of the shapes):
- RED: shapes are red/pink colored
- GREEN: shapes are green colored
- PURPLE: shapes are purple/violet colored

**SHAPE** (the geometric form):
- DIAMOND: pointed at top and bottom, like a rhombus/playing card diamond
- OVAL: rounded elongated shape, like a pill or stadium
- SQUIGGLE: wavy/irregular blob shape with curves

**SHADING** (how the shape is filled - look very carefully):
- SOLID: completely filled in with solid color, no white showing inside
- STRIPED: has horizontal lines running through it, you can see lines/stripes inside the shape
- EMPTY: just an outline with white/blank inside, only the border is colored

**NUMBER**: Count the shapes on the card - exactly 1, 2, or 3

For position, estimate where each card is located in the image as x,y percentages (0-100).

Important: The most common mistake is confusing shading. Look closely:
- If the inside is completely one solid color = SOLID
- If you see lines/stripes through it = STRIPED
- If the inside is white/empty = EMPTY

Examine each card systematically, one by one. Assign IDs as "card-1", "card-2", etc.`

async function analyzeImage(imagePath: string) {
  const imageBuffer = readFileSync(imagePath)
  const base64 = imageBuffer.toString("base64")
  const mimeType = "image/jpeg"

  const result = await generateObject({
    model: "anthropic/claude-sonnet-4-20250514",
    schema: ResponseSchema,
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: PROMPT },
          { type: "image", image: `data:${mimeType};base64,${base64}` },
        ],
      },
    ],
  })

  const { cards, confidence, notes } = result.object

  const parsedCards: SetCard[] = cards.map((card) => ({
    id: card.id,
    color: card.color,
    shape: card.shape,
    shading: card.shading,
    number: parseInt(card.number, 10) as 1 | 2 | 3,
    position: { x: card.positionX, y: card.positionY },
  }))

  const validSets = findAllSets(parsedCards)

  return {
    cards: parsedCards,
    validSets,
    confidence,
    notes,
    totalCards: parsedCards.length,
    totalSets: validSets.length,
  }
}

async function main() {
  const inputDir = join(__dirname, "..", "test-images", "fixed")
  const outputDir = join(__dirname, "..", "test-images", "results")

  const files = readdirSync(inputDir).filter((f) =>
    /\.(jpg|jpeg|png)$/i.test(f)
  )

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
      const result = await analyzeImage(imagePath)
      writeFileSync(outputPath, JSON.stringify(result, null, 2))
      console.log(
        `  → ${result.totalCards} cards, ${result.totalSets} sets (${result.confidence} confidence)`
      )
    } catch (error) {
      console.error(`  → Error analyzing ${file}:`, error)
    }
  }

  console.log(`\nResults written to ${outputDir}`)
}

main()
