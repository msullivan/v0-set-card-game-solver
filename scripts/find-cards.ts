import { readFileSync, writeFileSync, mkdirSync } from "fs"
import { join, dirname } from "path"
import { fileURLToPath } from "url"
import sharp from "sharp"

const __dirname = dirname(fileURLToPath(import.meta.url))
const cv = require("@techstark/opencv-js")

async function main() {
  // Wait for OpenCV WASM init
  await new Promise<void>((resolve) => {
    if (typeof cv.Mat === "function") { resolve(); return }
    cv.onRuntimeInitialized = () => resolve()
  })

  const imagePath = process.argv[2] || join(__dirname, "..", "test-images", "fixed", "001.jpg")
  const outputDir = process.argv[3] || join(__dirname, "..", "test-images", "cv-crops")

  console.log(`Processing ${imagePath}...`)
  const imageBuffer = readFileSync(imagePath)
  const metadata = await sharp(imageBuffer).metadata()
  const width = metadata.width!
  const height = metadata.height!
  const rawData = await sharp(imageBuffer).removeAlpha().raw().toBuffer()
  console.log(`  Image: ${width}x${height}`)

  // Create OpenCV mat from raw RGB data
  const mat = new cv.Mat(height, width, cv.CV_8UC3)
  mat.data.set(rawData)

  // Convert to grayscale
  const gray = new cv.Mat()
  cv.cvtColor(mat, gray, cv.COLOR_RGB2GRAY)

  // Estimate background illumination with a very large blur
  const background = new cv.Mat()
  cv.GaussianBlur(gray, background, new cv.Size(151, 151), 0)

  // Subtract background to normalize lighting (removes shadow gradient)
  // Result: cards become bright, table becomes ~uniform gray
  const normalized = new cv.Mat()
  cv.subtract(gray, background, normalized)
  // The subtraction clips at 0; cards (brighter than bg) stay positive,
  // table (close to bg) goes near 0. Add offset and scale for visibility.
  const scaled = new cv.Mat()
  normalized.convertTo(scaled, cv.CV_8U, 3.0, 0)

  // Blur to smooth out shapes inside cards
  const blurred = new cv.Mat()
  cv.GaussianBlur(scaled, blurred, new cv.Size(31, 31), 0)

  // Global threshold on the normalized image (shadow-free)
  const thresh = new cv.Mat()
  cv.threshold(blurred, thresh, 40, 255, cv.THRESH_BINARY)

  // Morphological close to fill remaining gaps from shapes inside cards
  // Keep kernel small to avoid bridging between closely-placed cards
  const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(20, 20))
  const closed = new cv.Mat()
  cv.morphologyEx(thresh, closed, cv.MORPH_CLOSE, kernel)

  // Clear border to prevent edge noise from merging with cards
  const border = 50
  cv.rectangle(closed, new cv.Point(0, 0), new cv.Point(width, border), new cv.Scalar(0), cv.FILLED)
  cv.rectangle(closed, new cv.Point(0, height - border), new cv.Point(width, height), new cv.Scalar(0), cv.FILLED)
  cv.rectangle(closed, new cv.Point(0, 0), new cv.Point(border, height), new cv.Scalar(0), cv.FILLED)
  cv.rectangle(closed, new cv.Point(width - border, 0), new cv.Point(width, height), new cv.Scalar(0), cv.FILLED)

  // Erode slightly to break thin connections between cards and edge noise
  const erodeKernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(15, 15))
  const eroded = new cv.Mat()
  cv.erode(closed, eroded, erodeKernel)

  // Debug: save intermediate images
  mkdirSync(outputDir, { recursive: true })
  const scaledBuf = Buffer.from(scaled.data)
  await sharp(scaledBuf, { raw: { width, height, channels: 1 } })
    .jpeg()
    .toFile(join(outputDir, "debug-normalized.jpg"))
  console.log("  Saved debug-normalized.jpg")

  const threshBuf = Buffer.from(eroded.data)
  await sharp(threshBuf, { raw: { width, height, channels: 1 } })
    .jpeg()
    .toFile(join(outputDir, "debug-threshold.jpg"))
  console.log("  Saved debug-threshold.jpg")

  // Find contours on eroded image
  const contours = new cv.MatVector()
  const hierarchy = new cv.Mat()
  cv.findContours(eroded, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

  // Filter by area and aspect ratio
  const imgArea = width * height
  const cards: { x: number; y: number; w: number; h: number }[] = []

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

    cards.push({ x: rect.x, y: rect.y, w: rect.width, h: rect.height })
  }

  // Sort top-to-bottom, left-to-right
  cards.sort((a, b) => {
    const rowA = Math.round(a.y / (height * 0.1))
    const rowB = Math.round(b.y / (height * 0.1))
    if (rowA !== rowB) return rowA - rowB
    return a.x - b.x
  })

  console.log(`  Found ${cards.length} cards`)

  // Save crops
  mkdirSync(outputDir, { recursive: true })

  for (let i = 0; i < cards.length; i++) {
    const card = cards[i]
    const pad = 10
    const left = Math.max(0, card.x - pad)
    const top = Math.max(0, card.y - pad)
    const cropWidth = Math.min(card.w + pad * 2, width - left)
    const cropHeight = Math.min(card.h + pad * 2, height - top)

    const cropped = await sharp(imageBuffer)
      .extract({ left, top, width: cropWidth, height: cropHeight })
      .jpeg()
      .toBuffer()

    const name = `card-${String(i + 1).padStart(2, "0")}.jpg`
    writeFileSync(join(outputDir, name), cropped)
    console.log(`    ${name}: ${card.w}x${card.h} at (${card.x}, ${card.y})`)
  }

  // Cleanup
  mat.delete(); gray.delete(); background.delete(); normalized.delete()
  scaled.delete(); blurred.delete(); thresh.delete()
  kernel.delete(); closed.delete(); erodeKernel.delete(); eroded.delete()
  contours.delete(); hierarchy.delete()
}

main()
