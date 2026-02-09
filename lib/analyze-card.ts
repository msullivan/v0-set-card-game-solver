import { generateObject } from "ai"
import { z } from "zod"

export const SingleCardSchema = z.object({
  color: z.enum(["red", "green", "purple"]).describe("The color of the shapes"),
  shape: z.enum(["diamond", "oval", "squiggle"]).describe("The shape type"),
  shading: z.enum(["solid", "striped", "empty"]).describe("solid=filled, striped=has lines, empty=outline only"),
  number: z.enum(["1", "2", "3"]).describe("Count of shapes on the card"),
})

export const SINGLE_CARD_PROMPT = `This is a photo of a single Set game card. Identify its 4 attributes:

**COLOR**: red, green, or purple
**SHAPE**: diamond (rhombus), oval (pill/stadium), or squiggle (wavy blob)
**SHADING**: solid (completely filled), striped (lines through it), or empty (outline only)
**NUMBER**: count the shapes - exactly 1, 2, or 3

Look carefully at the shading:
- SOLID = completely filled with color, no white inside
- STRIPED = has lines/stripes visible inside the shape
- EMPTY = just an outline, white/blank inside`

export async function analyzeCard(cropBuffer: Buffer, model: string) {
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
