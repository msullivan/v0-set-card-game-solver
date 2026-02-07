import { readFileSync } from "fs"
import { join, dirname } from "path"
import { fileURLToPath } from "url"

const __dirname = dirname(fileURLToPath(import.meta.url))

const baseDir = process.argv[2] || "merged"
const compareDir = process.argv[3] || "gpt-5.1"

const basePath = join(__dirname, "..", "test-images", "results", baseDir)
const comparePath = join(__dirname, "..", "test-images", "results", compareDir)

for (const f of ["001","002","003","004","005","006","007","008","009","010","011"]) {
  const base = JSON.parse(readFileSync(join(basePath, f + ".json"), "utf-8"))
  const comp = JSON.parse(readFileSync(join(comparePath, f + ".json"), "utf-8"))
  const diffs: string[] = []

  for (const bc of base.cards) {
    const cc = comp.cards.find((c: any) => c.id === bc.id)
    if (!cc) {
      diffs.push(bc.id + ": missing in " + compareDir)
      continue
    }
    const attrs: string[] = []
    for (const attr of ["color", "shape", "shading", "number"]) {
      if (bc[attr] !== cc[attr]) attrs.push(attr + ": " + bc[attr] + " → " + cc[attr])
    }
    if (attrs.length) diffs.push(bc.id + ": " + attrs.join(", "))
  }

  if (diffs.length || base.totalSets !== comp.totalSets) {
    console.log("=== " + f + " ===")
    for (const d of diffs) console.log("  " + d)
    console.log("  sets: " + base.totalSets + " → " + comp.totalSets)
    console.log()
  }
}
