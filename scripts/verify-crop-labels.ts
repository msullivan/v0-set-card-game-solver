import { readFileSync, writeFileSync } from "fs"
import { join, dirname } from "path"
import { fileURLToPath } from "url"

const __dirname = dirname(fileURLToPath(import.meta.url))

const envPath = join(__dirname, "..", ".env.local")
for (const line of readFileSync(envPath, "utf-8").split("\n")) {
  const match = line.match(/^\s*([^#=]+?)\s*=\s*(.*)\s*$/)
  if (match) process.env[match[1]] = match[2]
}

import { analyzeCard } from "../lib/analyze-card"
import type { SetCard } from "../lib/set-game"

const MODEL = "anthropic/claude-haiku-4-5-20251001"

type Label = {
  color: SetCard["color"]
  shape: SetCard["shape"]
  shading: SetCard["shading"]
  number: 1 | 2 | 3
}

function key(l: Label) {
  return `${l.number} ${l.color} ${l.shape} ${l.shading}`
}

async function main() {
  const cropsDir = join(__dirname, "..", "test-images", "crops")
  const labels = JSON.parse(readFileSync(join(cropsDir, "labels.json"), "utf-8")).labels as Record<string, Label>

  const disagreements: { path: string; canonical: string; predicted: string }[] = []
  const imageIds = [...new Set(Object.keys(labels).map((k) => k.split("/")[0]))].sort()

  for (const imageId of imageIds) {
    const imageCrops = Object.keys(labels).filter((k) => k.startsWith(`${imageId}/`)).sort()
    process.stdout.write(`${imageId}: `)

    const results = await Promise.all(
      imageCrops.map(async (path) => {
        const buf = readFileSync(join(cropsDir, path))
        const attrs = await analyzeCard(buf, MODEL)
        const pred: Label = {
          color: attrs.color,
          shape: attrs.shape,
          shading: attrs.shading,
          number: parseInt(attrs.number, 10) as 1 | 2 | 3,
        }
        return { path, pred }
      })
    )

    let agree = 0
    for (const { path, pred } of results) {
      const canon = labels[path]
      if (key(pred) === key(canon)) {
        agree++
      } else {
        disagreements.push({ path, canonical: key(canon), predicted: key(pred) })
      }
    }

    console.log(`${agree}/${results.length}`)
  }

  console.log(`\n${disagreements.length} disagreements out of ${Object.keys(labels).length} crops:`)
  for (const d of disagreements) {
    console.log(`  ${d.path}  canonical: ${d.canonical}  haiku: ${d.predicted}`)
  }

  writeFileSync(
    join(cropsDir, "haiku-verify.json"),
    JSON.stringify({ model: MODEL, disagreements }, null, 2)
  )
}

main()
