import { readFileSync, writeFileSync, mkdirSync, readdirSync } from "fs"
import { join, dirname, basename } from "path"
import { fileURLToPath } from "url"
import sharp from "sharp"

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

const cv = require("@techstark/opencv-js")

// --- CV card detection ---

const CV_MAX_DIM = 1000

async function detectCards(imageBuffer: Buffer): Promise<Buffer[]> {
  const metadata = await sharp(imageBuffer).metadata()
  const origWidth = metadata.width!
  const origHeight = metadata.height!

  const scale = Math.min(1, CV_MAX_DIM / Math.max(origWidth, origHeight))
  const width = Math.round(origWidth * scale)
  const height = Math.round(origHeight * scale)
  const rawData = await sharp(imageBuffer)
    .resize(width, height)
    .removeAlpha()
    .raw()
    .toBuffer()

  const mat = new cv.Mat(height, width, cv.CV_8UC3)
  mat.data.set(rawData)

  const gray = new cv.Mat()
  cv.cvtColor(mat, gray, cv.COLOR_RGB2GRAY)

  const bgBlurSize = Math.round(151 * scale) | 1
  const background = new cv.Mat()
  cv.GaussianBlur(gray, background, new cv.Size(bgBlurSize, bgBlurSize), 0)

  const normalized = new cv.Mat()
  cv.subtract(gray, background, normalized)
  const scaled = new cv.Mat()
  normalized.convertTo(scaled, cv.CV_8U, 3.0, 0)

  const blurSize = Math.round(31 * scale) | 1
  const blurred = new cv.Mat()
  cv.GaussianBlur(scaled, blurred, new cv.Size(blurSize, blurSize), 0)

  const thresh = new cv.Mat()
  cv.threshold(blurred, thresh, 40, 255, cv.THRESH_BINARY)

  const border = Math.max(5, Math.round(50 * scale))
  cv.rectangle(thresh, new cv.Point(0, 0), new cv.Point(width, border), new cv.Scalar(0), cv.FILLED)
  cv.rectangle(thresh, new cv.Point(0, height - border), new cv.Point(width, height), new cv.Scalar(0), cv.FILLED)
  cv.rectangle(thresh, new cv.Point(0, 0), new cv.Point(border, height), new cv.Scalar(0), cv.FILLED)
  cv.rectangle(thresh, new cv.Point(width - border, 0), new cv.Point(width, height), new cv.Scalar(0), cv.FILLED)

  const erodeSize = Math.max(3, Math.round(15 * scale))
  const erodeKernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(erodeSize, erodeSize))
  const eroded = new cv.Mat()
  cv.erode(thresh, eroded, erodeKernel)

  const contours = new cv.MatVector()
  const hierarchy = new cv.Mat()
  cv.findContours(eroded, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

  const imgArea = width * height
  const rects: { x: number; y: number; w: number; h: number }[] = []

  for (let i = 0; i < contours.size(); i++) {
    const contour = contours.get(i)
    const area = cv.contourArea(contour)
    const rect = cv.boundingRect(contour)

    const areaRatio = area / imgArea
    if (areaRatio < 0.005 || areaRatio > 0.08) continue

    const aspect = rect.height / rect.width
    if (aspect < 0.9 || aspect > 2.2) continue

    const rectArea = rect.width * rect.height
    const rectangularity = area / rectArea
    if (rectangularity < 0.65) continue

    rects.push({ x: rect.x, y: rect.y, w: rect.width, h: rect.height })
  }

  rects.sort((a, b) => {
    const rowA = Math.round(a.y / (height * 0.1))
    const rowB = Math.round(b.y / (height * 0.1))
    if (rowA !== rowB) return rowA - rowB
    return a.x - b.x
  })

  // Crop at full resolution
  const crops: Buffer[] = []
  for (const r of rects) {
    const pad = Math.round(10 / scale)
    const left = Math.max(0, Math.round(r.x / scale) - pad)
    const top = Math.max(0, Math.round(r.y / scale) - pad)
    const cropWidth = Math.min(Math.round(r.w / scale) + pad * 2, origWidth - left)
    const cropHeight = Math.min(Math.round(r.h / scale) + pad * 2, origHeight - top)

    const cropped = await sharp(imageBuffer)
      .extract({ left, top, width: cropWidth, height: cropHeight })
      .jpeg()
      .toBuffer()
    crops.push(cropped)
  }

  // Cleanup
  mat.delete(); gray.delete(); background.delete(); normalized.delete()
  scaled.delete(); blurred.delete(); thresh.delete()
  erodeKernel.delete(); eroded.delete()
  contours.delete(); hierarchy.delete()

  return crops
}

// --- AI single-card analysis ---

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

// --- Main pipeline ---

async function analyzeImage(imagePath: string, model: string) {
  const imageBuffer = readFileSync(imagePath)

  // Pass 1: CV card detection
  const crops = await detectCards(imageBuffer)
  console.log(`  CV detected ${crops.length} cards`)

  // Pass 2: AI analysis of each card (parallel)
  const cardResults = await Promise.all(
    crops.map(async (crop, i) => {
      const attrs = await analyzeCard(crop, model)
      console.log(`    card-${i + 1}: ${attrs.number} ${attrs.color} ${attrs.shape} ${attrs.shading}`)
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
  const cards = cardResults

  const validSets = findAllSets(cards)

  return {
    cards,
    validSets,
    confidence: "high" as const,
    notes: "",
    totalCards: cards.length,
    totalSets: validSets.length,
  }
}

async function main() {
  // Wait for OpenCV WASM init
  await new Promise<void>((resolve) => {
    if (typeof cv.Mat === "function") { resolve(); return }
    cv.onRuntimeInitialized = () => resolve()
  })

  const model = process.argv[2]
  const outName = process.argv[3]

  if (!model || !outName) {
    console.error("Usage: npx tsx scripts/analyze-two-pass.ts <model> <output-dir>")
    console.error("Example: npx tsx scripts/analyze-two-pass.ts anthropic/claude-sonnet-4-20250514 sonnet-4-two-pass")
    process.exit(1)
  }

  const inputDir = join(__dirname, "..", "test-images", "fixed")
  const outputDir = join(__dirname, "..", "test-images", "results", outName)

  const files = readdirSync(inputDir)
    .filter(f => /\.(jpg|jpeg|png)$/i.test(f))
    .sort()

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
      const result = await analyzeImage(imagePath, model)
      writeFileSync(outputPath, JSON.stringify(result, null, 2))
      console.log(`  → ${result.totalCards} cards, ${result.totalSets} sets`)
    } catch (error) {
      console.error(`  → Error analyzing ${file}:`, error)
    }
  }

  console.log(`\nResults written to ${outputDir}`)
}

main()
