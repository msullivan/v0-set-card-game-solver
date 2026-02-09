import { findAllSets, type SetCard } from "@/lib/set-game"
import { detectCards } from "@/lib/detect-cards"
import { analyzeCard } from "@/lib/analyze-card"

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
    const { crops, timing: cvTiming } = await detectCards(imageBuffer)

    if (crops.length === 0) {
      return Response.json(
        { error: "No cards detected in the image. Please try a clearer photo." },
        { status: 400 }
      )
    }

    // Pass 2: AI analysis of each card (parallel)
    const aiStart = performance.now()
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
    const aiTime = performance.now() - aiStart

    // Find valid sets
    const setsStart = performance.now()
    const validSets = findAllSets(cardResults)
    const setsTime = performance.now() - setsStart

    return Response.json({
      cards: cardResults,
      validSets,
      confidence: "high",
      notes: "",
      totalCards: cardResults.length,
      totalSets: validSets.length,
      timing: {
        cvInit: Math.round(cvTiming.cvInit),
        cvProcess: Math.round(cvTiming.cvProcess),
        ai: Math.round(aiTime),
        sets: Math.round(setsTime),
      },
    })
  } catch (error) {
    console.error("Error analyzing image:", error)
    return Response.json(
      { error: "Failed to analyze image. Please try again." },
      { status: 500 }
    )
  }
}
