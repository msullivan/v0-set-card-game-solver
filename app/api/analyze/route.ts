import { generateObject } from "ai"
import { z } from "zod"
import { findAllSets, type SetCard } from "@/lib/set-game"
import { detectCards } from "@/lib/detect-cards"

const SingleCardSchema = z.object({
  color: z.enum(["red", "green", "purple"]).describe("The color of the shapes"),
  shape: z.enum(["diamond", "oval", "squiggle"]).describe("The shape type"),
  shading: z.enum(["solid", "striped", "empty"]).describe("solid=filled, striped=has lines, empty=outline only"),
  number: z.enum(["1", "2", "3"]).describe("Count of shapes on the card"),
})

const SINGLE_CARD_PROMPT = `This is a photo of a single Set game card. Identify its 4 attributes:

**COLOR**: red, green, or purple
**SHAPE**: diamond (rhombus), oval (pill/stadium), or squiggle (wavy blob)
**SHADING**: solid (completely filled), striped (lines through it), or empty (outline only)
**NUMBER**: count the shapes - exactly 1, 2, or 3

Look carefully at the shading:
- SOLID = completely filled with color, no white inside
- STRIPED = has lines/stripes visible inside the shape
- EMPTY = just an outline, white/blank inside`

async function analyzeCard(cropBuffer: Buffer, model: string) {
  const base64 = cropBuffer.toString("base64")

  const result = await generateObject({
    model,
    schema: SingleCardSchema,
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: SINGLE_CARD_PROMPT },
          { type: "image", image: `data:image/jpeg;base64,${base64}` },
        ],
      },
    ],
  })

  return result.object
}

export async function POST(req: Request) {
  try {
    const { image } = await req.json()

    if (!image) {
      return Response.json({ error: "No image provided" }, { status: 400 })
    }

    // Decode base64 image
    const base64Data = image.replace(/^data:image\/\w+;base64,/, "")
    const imageBuffer = Buffer.from(base64Data, "base64")

    // Pass 1: CV card detection
    const crops = await detectCards(imageBuffer)

    if (crops.length === 0) {
      return Response.json(
        { error: "No cards detected in the image. Please try a clearer photo." },
        { status: 400 }
      )
    }

    // Pass 2: AI analysis of each card (parallel)
    const model = "anthropic/claude-sonnet-4-20250514"
    const cardResults = await Promise.all(
      crops.map(async (crop, i) => {
        const attrs = await analyzeCard(crop, model)
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

    const validSets = findAllSets(cardResults)

    return Response.json({
      cards: cardResults,
      validSets,
      confidence: "high",
      notes: "",
      totalCards: cardResults.length,
      totalSets: validSets.length,
    })
  } catch (error) {
    console.error("Error analyzing image:", error)
    return Response.json(
      { error: "Failed to analyze image. Please try again." },
      { status: 500 }
    )
  }
}
