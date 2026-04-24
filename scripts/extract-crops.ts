import { readFileSync, writeFileSync, readdirSync, mkdirSync } from "fs"
import { join, basename, dirname } from "path"
import { fileURLToPath } from "url"

const __dirname = dirname(fileURLToPath(import.meta.url))

import { detectCards } from "../lib/detect-cards-node"
import type { SetCard } from "../lib/set-game"

type Label = {
  color: SetCard["color"]
  shape: SetCard["shape"]
  shading: SetCard["shading"]
  number: 1 | 2 | 3
}

async function main() {
  const inputDir = join(__dirname, "..", "test-images", "fixed")
  const outputDir = join(__dirname, "..", "test-images", "crops")
  const canonicalDir = join(__dirname, "..", "test-images", "results", "canonical")

  mkdirSync(outputDir, { recursive: true })

  const files = readdirSync(inputDir)
    .filter((f) => /\.(jpg|jpeg|png)$/i.test(f))
    .sort()

  const allLabels: Record<string, Label> = {}
  const summary: Record<string, { cards: number; canonical_cards: number; matched: boolean }> = {}

  for (const file of files) {
    const imageId = basename(file, ".jpg").replace(/\.jpeg$/i, "").replace(/\.png$/i, "")
    console.log(`${imageId}...`)

    const imageBuffer = readFileSync(join(inputDir, file))
    const { crops } = await detectCards(imageBuffer)

    const imageOutDir = join(outputDir, imageId)
    mkdirSync(imageOutDir, { recursive: true })

    let canonicalCards: Label[] = []
    try {
      const canon = JSON.parse(readFileSync(join(canonicalDir, imageId + ".json"), "utf-8"))
      canonicalCards = canon.cards.map((c: any) => ({
        color: c.color,
        shape: c.shape,
        shading: c.shading,
        number: typeof c.number === "number" ? c.number : parseInt(c.number, 10),
      }))
    } catch {
      console.log(`  WARN: no canonical for ${imageId}`)
    }

    const matched = crops.length === canonicalCards.length
    summary[imageId] = { cards: crops.length, canonical_cards: canonicalCards.length, matched }

    for (let i = 0; i < crops.length; i++) {
      const cropName = `card-${String(i + 1).padStart(2, "0")}.jpg`
      writeFileSync(join(imageOutDir, cropName), crops[i])
      if (matched) {
        allLabels[`${imageId}/${cropName}`] = canonicalCards[i]
      }
    }

    console.log(`  ${crops.length} crops, canonical: ${canonicalCards.length}${matched ? " ✓" : " (count mismatch — no labels assigned)"}`)
  }

  writeFileSync(
    join(outputDir, "labels.json"),
    JSON.stringify(
      {
        source: "canonical (position-matched to CV crop order)",
        note: "021 and 024 canonicals were built from Opus full-image analysis, not CV crops — card-to-position mapping may differ from other images. Eyeball those before trusting.",
        summary,
        labels: allLabels,
      },
      null,
      2
    )
  )

  const matching = Object.values(summary).filter((s) => s.matched).length
  console.log(`\n${matching}/${files.length} images labeled. Labels → ${join(outputDir, "labels.json")}`)
}

main()
