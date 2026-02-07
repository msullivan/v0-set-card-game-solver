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
import sharp from "sharp"
import { findAllSets, type SetCard } from "../lib/set-game"

// --- Schemas ---

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

const BBoxSchema = z.object({
  cards: z.array(z.object({
    id: z.string().describe("Unique identifier like card-1, card-2, etc."),
    centerX: z.number().describe("Center x position as percentage (0-100)"),
    centerY: z.number().describe("Center y position as percentage (0-100)"),
  })).describe("Center points of all Set game cards in the image"),
})

const SingleCardSchema = z.object({
  color: z.enum(["red", "green", "purple"]).describe("The color of the shapes on the card"),
  shape: z.enum(["diamond", "oval", "squiggle"]).describe("The shape type on the card"),
  shading: z.enum(["solid", "striped", "empty"]).describe("solid=filled, striped=has lines, empty=outline only"),
  number: z.enum(["1", "2", "3"]).describe("Count of shapes on the card"),
})

// --- Prompts ---

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

const BBOX_PROMPT = `You are analyzing a photo of Set game cards laid out on a table. Identify the center point of each card in the image.

For each card, provide:
- A unique ID (card-1, card-2, etc.)
- The center point as percentages of the image dimensions (0-100):
  - centerX: horizontal center of the card
  - centerY: vertical center of the card

Assign IDs left-to-right, top-to-bottom.`

const SINGLE_CARD_PROMPT = `You are looking at a cropped photo centered on a single Set game card. Focus on the card closest to the center of the image. Identify its 4 attributes:

**COLOR**: red, green, or purple
**SHAPE**: diamond, oval, or squiggle
**SHADING**: solid (completely filled), striped (has lines through it), or empty (just an outline)
**NUMBER**: Count the shapes on that one card - exactly 1, 2, or 3

Look very carefully at the shading and count. Ignore any other cards partially visible at the edges.`

// --- Single-pass analysis ---

async function analyzeImage(imagePath: string, model: string) {
  const imageBuffer = readFileSync(imagePath)
  const base64 = imageBuffer.toString("base64")
  const mimeType = "image/jpeg"

  const result = await generateObject({
    model,
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

// --- Two-pass analysis ---

async function analyzeImageTwoPass(imagePath: string, model: string, outputDir: string) {
  const imageBuffer = readFileSync(imagePath)
  const base64 = imageBuffer.toString("base64")
  const mimeType = "image/jpeg"
  const metadata = await sharp(imageBuffer).metadata()
  const imgWidth = metadata.width!
  const imgHeight = metadata.height!

  // Pass 1: full-image analysis to get card positions and initial attributes
  console.log("    Pass 1: full-image analysis...")
  const pass1Result = await generateObject({
    model,
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

  const pass1Cards = pass1Result.object.cards
  console.log(`    Found ${pass1Cards.length} cards, cropping for verification...`)

  // Crop each card using pass 1 positions with generous fixed size
  const cropW = Math.round(imgWidth * 0.25)
  const cropH = Math.round(imgHeight * 0.25)

  const crops = await Promise.all(
    pass1Cards.map(async (card) => {
      const cx = Math.round((card.positionX / 100) * imgWidth)
      const cy = Math.round((card.positionY / 100) * imgHeight)
      const left = Math.max(0, Math.round(cx - cropW / 2))
      const top = Math.max(0, Math.round(cy - cropH / 2))
      const width = Math.min(cropW, imgWidth - left)
      const height = Math.min(cropH, imgHeight - top)
      const cropped = await sharp(imageBuffer)
        .extract({ left, top, width, height })
        .jpeg()
        .toBuffer()
      // Save crops for debugging
      const debugDir = join(outputDir, "crops")
      mkdirSync(debugDir, { recursive: true })
      writeFileSync(join(debugDir, `${card.id}.jpg`), cropped)
      return { id: card.id, crop: cropped, positionX: card.positionX, positionY: card.positionY }
    })
  )

  // Pass 2: re-identify each card from its crop in parallel
  console.log("    Pass 2: verifying each card individually...")
  const cardResults = await Promise.all(
    crops.map(async (crop) => {
      const cropBase64 = crop.crop.toString("base64")
      try {
        const result = await generateObject({
          model,
          schema: SingleCardSchema,
          messages: [
            {
              role: "user",
              content: [
                { type: "text", text: SINGLE_CARD_PROMPT },
                { type: "image", image: `data:image/jpeg;base64,${cropBase64}` },
              ],
            },
          ],
        })
        console.log(`      ${crop.id}: ${result.object.number} ${result.object.color} ${result.object.shading} ${result.object.shape}`)
        return {
          id: crop.id,
          ...result.object,
          positionX: crop.positionX,
          positionY: crop.positionY,
        }
      } catch (err) {
        console.error(`      ${crop.id} FAILED:`, err instanceof Error ? err.message : String(err))
        throw err
      }
    })
  )

  // Combine results
  const parsedCards: SetCard[] = cardResults.map((card) => ({
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
    confidence: "high" as const,
    notes: "Two-pass analysis: full image analyzed first, then each card verified individually from crop.",
    totalCards: parsedCards.length,
    totalSets: validSets.length,
  }
}

// --- Main ---

async function main() {
  const args = process.argv.slice(2)
  const twoPass = args.includes("--two-pass")
  const positional = args.filter((a) => !a.startsWith("--"))

  const model = positional[0]
  const outName = positional[1]

  if (!model || !outName) {
    console.error("Usage: npx tsx scripts/analyze-images.ts [--two-pass] <model> <output-dir>")
    console.error("Example: npx tsx scripts/analyze-images.ts anthropic/claude-sonnet-4-20250514 sonnet-4")
    console.error("Example: npx tsx scripts/analyze-images.ts --two-pass anthropic/claude-sonnet-4-20250514 sonnet-4-two-pass")
    process.exit(1)
  }

  const inputDir = join(__dirname, "..", "test-images", "fixed")
  const outputDir = join(__dirname, "..", "test-images", "results", outName)

  const files = readdirSync(inputDir).filter((f) =>
    /\.(jpg|jpeg|png)$/i.test(f)
  )

  if (files.length === 0) {
    console.log("No images found in", inputDir)
    process.exit(1)
  }

  mkdirSync(outputDir, { recursive: true })

  console.log(`Mode: ${twoPass ? "two-pass" : "single-pass"}`)
  console.log(`Model: ${model}\n`)

  for (const file of files.slice(1, 2)) {
    const imagePath = join(inputDir, file)
    const outputName = basename(file, ".jpg") + ".json"
    const outputPath = join(outputDir, outputName)

    console.log(`Analyzing ${file}...`)
    try {
      const result = twoPass
        ? await analyzeImageTwoPass(imagePath, model, outputDir)
        : await analyzeImage(imagePath, model)
      writeFileSync(outputPath, JSON.stringify(result, null, 2))
      console.log(
        `  → ${result.totalCards} cards, ${result.totalSets} sets (${result.confidence} confidence)`
      )
    } catch (error) {
      console.error(`  → Error analyzing ${file}:`, error instanceof Error ? error.message : String(error))
    }
  }

  console.log(`\nResults written to ${outputDir}`)
}

main()
