import { generateObject } from "ai"
import { z } from "zod"
import { findAllSets, type SetCard } from "@/lib/set-game"

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

    // Convert string numbers to actual numbers for the SetCard type
    const parsedCards: SetCard[] = cards.map((card) => ({
      id: card.id,
      color: card.color,
      shape: card.shape,
      shading: card.shading,
      number: parseInt(card.number, 10) as 1 | 2 | 3,
      position: { x: card.positionX, y: card.positionY },
    }))

    // Find all valid sets
    const validSets = findAllSets(parsedCards)

    return Response.json({
      cards: parsedCards,
      validSets,
      confidence,
      notes,
      totalCards: parsedCards.length,
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
