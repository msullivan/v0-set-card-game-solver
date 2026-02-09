import { readFileSync, writeFileSync, readdirSync, statSync } from "fs"
import { join, dirname, basename } from "path"
import { fileURLToPath } from "url"
import { detectCards } from "../lib/detect-cards"
import { mkdirSync } from "fs"

const __dirname = dirname(fileURLToPath(import.meta.url))

async function processImage(imagePath: string, outputDir: string) {
  console.log(`Processing ${imagePath}...`)
  const imageBuffer = readFileSync(imagePath)

  const { crops, timing } = await detectCards(imageBuffer, outputDir)
  console.log(`  CV: ${(timing.cvProcess / 1000).toFixed(1)}s`)
  console.log(`  Found ${crops.length} cards`)

  // Save crops
  mkdirSync(outputDir, { recursive: true })
  for (let i = 0; i < crops.length; i++) {
    const name = `card-${String(i + 1).padStart(2, "0")}.jpg`
    writeFileSync(join(outputDir, name), crops[i])
    console.log(`    ${name}`)
  }
}

async function main() {
  const inputPath = process.argv[2] || join(__dirname, "..", "test-images", "fixed")
  const outputBase = process.argv[3] || join(__dirname, "..", "test-images", "cv-crops")

  const stat = statSync(inputPath)
  if (stat.isDirectory()) {
    const files = readdirSync(inputPath)
      .filter(f => /\.(jpg|jpeg|png)$/i.test(f))
      .sort()
    console.log(`Found ${files.length} images in ${inputPath}\n`)
    for (const file of files) {
      const name = basename(file, ".jpg").replace(/\.jpeg$/i, "").replace(/\.png$/i, "")
      await processImage(join(inputPath, file), join(outputBase, name))
      console.log()
    }
  } else {
    await processImage(inputPath, outputBase)
  }
}

main()
