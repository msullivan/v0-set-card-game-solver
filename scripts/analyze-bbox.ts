import { readFileSync, writeFileSync, readdirSync, mkdirSync } from "fs"
import { join, dirname, basename } from "path"
import { fileURLToPath } from "url"

const __dirname = dirname(fileURLToPath(import.meta.url))

// Load .env.local
const envPath = join(__dirname, "..", ".env.local")
for (const line of readFileSync(envPath, "utf-8").split("\n")) {
  const match = line.match(/^\s*([^#=]+?)\s*=\s*(.*)\s*$/)
  if (match) process.env[match[1]] = match[2]
}

import { generateText } from "ai"
import sharp from "sharp"
import { findAllSets, type SetCard } from "../lib/set-game"
import { analyzeCard } from "../lib/analyze-card"

const BBOX_MODEL = "google/gemini-2.5-flash-lite"
const CARD_MODEL = "anthropic/claude-haiku-4-5-20251001"

// Prompt following Gemini's documented bounding box format
const BBOX_PROMPT = `Detect all of the cards in the image. The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-1000.`

async function analyzeImage(imagePath: string, cropsDir?: string) {
  const imageBuffer = readFileSync(imagePath)
  const { width: imgWidth, height: imgHeight } = await sharp(imageBuffer).metadata()

  console.log(`  image: ${imgWidth}x${imgHeight}`)

  const base64 = imageBuffer.toString("base64")

  // Pass 1: Gemini 2.5 Flash finds bounding boxes
  console.log(`  pass 1 (${BBOX_MODEL}): locating cards...`)
  // We use generateText instead of generateObject here because generateObject sends a JSON
  // schema to Gemini which interferes with its native object detection pipeline. With a schema,
  // Gemini tries to reason about coordinates from scratch rather than using its trained spatial
  // detection model, producing much worse results. The free-form response is clean enough to
  // parse manually.

  type BBox = { label: string; box_2d: [number, number, number, number] }
  let boxes: BBox[] | null = null
  const MAX_RETRIES = 3

  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    const bboxResult = await generateText({
      model: BBOX_MODEL,
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: BBOX_PROMPT },
            { type: "image", image: `data:image/jpeg;base64,${base64}` },
          ],
        },
      ],
    })

    console.log(`  raw response (attempt ${attempt}): ${bboxResult.text}`)

    try {
      // Gemini wraps JSON in markdown code fences even though we didn't ask it to. Annoying.
      const rawText = bboxResult.text.replace(/^```(?:json)?\n?/, "").replace(/\n?```$/, "").trim()
      const parsed = JSON.parse(rawText)

      // Validate: must be an array of objects with box_2d arrays
      if (
        Array.isArray(parsed) &&
        parsed.length > 0 &&
        parsed.every((b: unknown) =>
          typeof b === "object" && b !== null &&
          "box_2d" in b && Array.isArray((b as BBox).box_2d) && (b as BBox).box_2d.length === 4
        )
      ) {
        boxes = parsed as BBox[]
        break
      } else {
        console.warn(`  attempt ${attempt}: unexpected format, retrying...`)
      }
    } catch (e) {
      console.warn(`  attempt ${attempt}: failed to parse JSON (${e instanceof Error ? e.message : e}), retrying...`)
    }
  }

  if (!boxes) throw new Error(`Failed to get valid bounding boxes after ${MAX_RETRIES} attempts`)

  console.log(`  found ${boxes.length} cards`)
  for (const box of boxes) console.log(`    ${box.label ?? "(no label)"}: ${JSON.stringify(box.box_2d)}`)

  // Pass 2: Haiku 4.5 identifies each card from its crop (at original resolution)
  console.log(`  pass 2 (${CARD_MODEL}): identifying cards...`)
  const cardResults = await Promise.all(
    boxes.map(async (box, i) => {
      // box_2d is [ymin, xmin, ymax, xmax], values 0-1000
      const [yn1, xn1, yn2, xn2] = box.box_2d
      const left  = Math.max(0,          Math.round((xn1 / 1000) * imgWidth!))
      const top   = Math.max(0,          Math.round((yn1 / 1000) * imgHeight!))
      const right  = Math.min(imgWidth!,  Math.round((xn2 / 1000) * imgWidth!))
      const bottom = Math.min(imgHeight!, Math.round((yn2 / 1000) * imgHeight!))
      const width  = Math.max(1, right - left)
      const height = Math.max(1, bottom - top)

      const id = `card-${i + 1}`

      const crop = await sharp(imageBuffer)
        .extract({ left, top, width, height })
        .jpeg()
        .toBuffer()

      if (cropsDir) {
        mkdirSync(cropsDir, { recursive: true })
        writeFileSync(join(cropsDir, `${id}.jpg`), crop)
      }

      const attrs = await analyzeCard(crop, CARD_MODEL)
      console.log(`    ${id}: ${attrs.number} ${attrs.color} ${attrs.shape} ${attrs.shading}`)

      return {
        id,
        color: attrs.color,
        shape: attrs.shape,
        shading: attrs.shading,
        number: parseInt(attrs.number, 10) as 1 | 2 | 3,
        position: { x: left, y: top },
      } as SetCard
    })
  )

  const validSets = findAllSets(cardResults)

  return {
    cards: cardResults,
    validSets,
    confidence: "high" as const,
    notes: "",
    totalCards: cardResults.length,
    totalSets: validSets.length,
  }
}

async function main() {
  const outName = process.argv[2]
  const fileFilter = process.argv[3] // optional: e.g. "001" to only process 001.jpg

  if (!outName) {
    console.error("Usage: npx tsx scripts/analyze-bbox.ts <output-dir> [file-prefix]")
    console.error("Example: npx tsx scripts/analyze-bbox.ts gemini-bbox-haiku-id")
    console.error("Example: npx tsx scripts/analyze-bbox.ts gemini-bbox-haiku-id 001")
    process.exit(1)
  }

  const inputDir = join(__dirname, "..", "test-images", "fixed")
  const outputDir = join(__dirname, "..", "test-images", "results", outName)
  mkdirSync(outputDir, { recursive: true })

  const files = readdirSync(inputDir)
    .filter((f) => /\.(jpg|jpeg|png)$/i.test(f) && (!fileFilter || f.startsWith(fileFilter)))
    .sort()

  for (const file of files) {
    const imagePath = join(inputDir, file)
    const outputPath = join(outputDir, basename(file, ".jpg") + ".json")
    const cropsDir = join(outputDir, "crops", basename(file, ".jpg"))

    console.log(`\nAnalyzing ${file}...`)
    try {
      const result = await analyzeImage(imagePath, cropsDir)
      writeFileSync(outputPath, JSON.stringify(result, null, 2))
      console.log(`  → ${result.totalCards} cards, ${result.totalSets} sets`)
    } catch (error) {
      console.error(`  → error:`, error instanceof Error ? error.message : String(error))
    }
  }

  console.log(`\nResults written to ${outputDir}`)
}

main()
