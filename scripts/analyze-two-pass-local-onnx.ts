// Two-pass analysis using local ONNX inference instead of the Python server.
//
// Mirrors scripts/analyze-two-pass-local.ts so `compare-results.ts` can diff
// ONNX output against server output directly.

import { readFileSync, writeFileSync, mkdirSync, readdirSync } from "fs"
import { join, dirname, basename } from "path"
import { fileURLToPath } from "url"
import sharp from "sharp"

import { detectCards } from "../lib/detect-cards-node"
import {
  classifyCards,
  loadSetIdModel,
  type ModelMeta,
  type RGBImage,
  type SetIdSession,
} from "../lib/analyze-card-local"
import { findAllSets, type SetCard } from "../lib/set-game"

const __dirname = dirname(fileURLToPath(import.meta.url))

const MODEL_PATH =
  process.env.SET_ID_MODEL ||
  join(__dirname, "..", "public", "models", "set_id_smaller.onnx")

async function decodeRGB(buf: Buffer): Promise<RGBImage> {
  const { data, info } = await sharp(buf)
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true })
  return { data: new Uint8Array(data), width: info.width, height: info.height }
}

async function analyzeImage(imagePath: string, model: SetIdSession) {
  const imageBuffer = readFileSync(imagePath)

  const { crops } = await detectCards(imageBuffer)
  console.log(`  CV detected ${crops.length} cards`)

  if (crops.length === 0) {
    return {
      cards: [],
      validSets: [],
      confidence: "high" as const,
      notes: "no cards detected",
      totalCards: 0,
      totalSets: 0,
    }
  }

  const rgbs = await Promise.all(crops.map(decodeRGB))
  const preds = await classifyCards(model, rgbs)
  const cards = preds.map((p, i) => {
    console.log(
      `    card-${i + 1}: ${p.number} ${p.color} ${p.shape} ${p.shading}`,
    )
    return {
      id: `card-${i + 1}`,
      color: p.color,
      shape: p.shape,
      shading: p.shading,
      number: parseInt(p.number, 10) as 1 | 2 | 3,
      position: { x: 0, y: 0 },
      logits: p.logits,
    }
  })

  const validSets = findAllSets(cards as unknown as SetCard[])
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
  const outName = process.argv[2]
  if (!outName) {
    console.error(
      "Usage: npx tsx scripts/analyze-two-pass-local-onnx.ts <output-dir>",
    )
    console.error(
      "       SET_ID_MODEL=/path/to/model.onnx npx tsx ... to override model path",
    )
    process.exit(1)
  }

  const modelBytes = readFileSync(MODEL_PATH)
  const metaPath = MODEL_PATH.replace(/\.onnx$/, ".json")
  let meta: ModelMeta | undefined
  try {
    meta = JSON.parse(readFileSync(metaPath, "utf8")) as ModelMeta
  } catch {
    // No sidecar — will fall back to DEFAULT_IMG_SIZE.
  }
  const model = await loadSetIdModel(modelBytes, { meta })
  console.log(
    `loaded model: ${MODEL_PATH} (${(modelBytes.byteLength / 1024).toFixed(1)} KB, img_size=${model.imgSize})`,
  )

  const inputDir = join(__dirname, "..", "test-images", "fixed")
  const outputDir = join(__dirname, "..", "test-images", "results", outName)

  const files = readdirSync(inputDir)
    .filter((f) => /\.(jpg|jpeg|png)$/i.test(f))
    .sort()
  if (files.length === 0) {
    console.log("No images found in", inputDir)
    process.exit(1)
  }

  mkdirSync(outputDir, { recursive: true })
  for (const file of files) {
    const imagePath = join(inputDir, file)
    const outputPath = join(outputDir, basename(file, ".jpg") + ".json")
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
