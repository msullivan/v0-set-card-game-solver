import { generateObject } from "ai"
import { z } from "zod"
import { findAllSets, type SetCard } from "@/lib/set-game"

const CardSchema = z.object({
  id: z.string(),
  color: z.enum(["red", "green", "purple"]),
  shape: z.enum(["diamond", "oval", "squiggle"]),
  shading: z.enum(["solid", "striped", "empty"]),
  number: z.union([z.literal(1), z.literal(2), z.literal(3)]),
  position: z
    .object({
      x: z.number().describe("Approximate x position (0-100) of the card in the image"),
      y: z.number().describe("Approximate y position (0-100) of the card in the image"),
    })
    .optional(),
})

const ResponseSchema = z.object({
  cards: z.array(CardSchema).describe("All Set game cards detected in the image"),
  confidence: z
    .enum(["high", "medium", "low"])
    .describe("How confident you are in the card detection"),
  notes: z
    .string()
    .optional()
    .describe("Any notes about card detection issues or unclear cards"),
})

export async function POST(req: Request) {
  try {
    const { image } = await req.json()

    if (!image) {
      return Response.json({ error: "No image provided" }, { status: 400 })
    }

    // Use GPT-4o to analyze the image
    const result = await generateObject({
      model: "openai/gpt-4o",
      schema: ResponseSchema,
      messages: [
        {
          role: "user",
          content: [
            {
              type: "text",
              text: `Analyze this image of Set game cards. For each card visible, identify:
1. Color: red, green, or purple
2. Shape: diamond, oval, or squiggle  
3. Shading: solid (completely filled), striped (has lines/pattern), or empty (just outline)
4. Number: 1, 2, or 3 (count of shapes on the card)

Also estimate the approximate position of each card in the image (x, y as percentages 0-100).

Be precise about the shading - solid means completely filled with color, striped has lines through it, empty is just an outline.

Return all cards you can clearly identify. Assign each card a unique id like "card-1", "card-2", etc.`,
            },
            {
              type: "image",
              image: image,
            },
          ],
        },
      ],
    })

    const { cards, confidence, notes } = result.object

    // Find all valid sets
    const validSets = findAllSets(cards as SetCard[])

    return Response.json({
      cards,
      validSets,
      confidence,
      notes,
      totalCards: cards.length,
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
