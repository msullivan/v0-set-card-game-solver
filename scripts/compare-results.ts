import { readFileSync } from "fs"
import { join, dirname } from "path"
import { fileURLToPath } from "url"

const __dirname = dirname(fileURLToPath(import.meta.url))

const baseDir = process.argv[2] || "merged"
const compareDir = process.argv[3] || "gpt-5.1"

const basePath = join(__dirname, "..", "test-images", "results", baseDir)
const comparePath = join(__dirname, "..", "test-images", "results", compareDir)

function cardKey(c: any): string {
  return `${c.number} ${c.color} ${c.shape} ${c.shading}`
}

for (const f of ["001","002","003","005","006","007","008","009","010","011"]) {
  const base = JSON.parse(readFileSync(join(basePath, f + ".json"), "utf-8"))
  const comp = JSON.parse(readFileSync(join(comparePath, f + ".json"), "utf-8"))

  const baseKeys = base.cards.map(cardKey).sort()
  const compKeys = comp.cards.map(cardKey).sort()

  const diffs: string[] = []
  const baseCounts = new Map<string, number>()
  const compCounts = new Map<string, number>()
  for (const k of baseKeys) baseCounts.set(k, (baseCounts.get(k) || 0) + 1)
  for (const k of compKeys) compCounts.set(k, (compCounts.get(k) || 0) + 1)

  const allKeys = new Set([...baseCounts.keys(), ...compCounts.keys()])
  for (const k of [...allKeys].sort()) {
    const b = baseCounts.get(k) || 0
    const c = compCounts.get(k) || 0
    if (b > c) diffs.push(`- ${k}` + (b - c > 1 ? ` (×${b - c})` : ""))
    if (c > b) diffs.push(`+ ${k}` + (c - b > 1 ? ` (×${c - b})` : ""))
  }

  if (diffs.length || base.totalSets !== comp.totalSets) {
    console.log("=== " + f + " ===")
    for (const d of diffs) console.log("  " + d)
    if (base.totalSets !== comp.totalSets)
      console.log("  sets: " + base.totalSets + " → " + comp.totalSets)
    console.log()
  }
}
