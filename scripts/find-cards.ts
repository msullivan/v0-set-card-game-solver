import { readFileSync, writeFileSync, mkdirSync, readdirSync } from "fs"
import { join, dirname, basename } from "path"
import { fileURLToPath } from "url"
import sharp from "sharp"

const __dirname = dirname(fileURLToPath(import.meta.url))
const cv = require("@techstark/opencv-js")

const CV_MAX_DIM = 1000 // max dimension for CV processing

async function processImage(imagePath: string, outputDir: string) {
  console.log(`Processing ${imagePath}...`)
  const imageBuffer = readFileSync(imagePath)
  const metadata = await sharp(imageBuffer).metadata()
  const origWidth = metadata.width!
  const origHeight = metadata.height!

  // Downscale for CV processing
  const scale = Math.min(1, CV_MAX_DIM / Math.max(origWidth, origHeight))
  const width = Math.round(origWidth * scale)
  const height = Math.round(origHeight * scale)
  const rawData = await sharp(imageBuffer)
    .resize(width, height)
    .removeAlpha()
    .raw()
    .toBuffer()
  console.log(`  Image: ${origWidth}x${origHeight} → ${width}x${height} for CV (${scale.toFixed(2)}x)`)

  // Create OpenCV mat from raw RGB data
  const mat = new cv.Mat(height, width, cv.CV_8UC3)
  mat.data.set(rawData)

  // Convert to grayscale
  const gray = new cv.Mat()
  cv.cvtColor(mat, gray, cv.COLOR_RGB2GRAY)

  // Estimate background illumination with a very large blur
  // Scale kernel sizes proportionally (must be odd)
  const bgBlurSize = Math.round(151 * scale) | 1
  const background = new cv.Mat()
  cv.GaussianBlur(gray, background, new cv.Size(bgBlurSize, bgBlurSize), 0)

  // Subtract background to normalize lighting (removes shadow gradient)
  // Result: cards become bright, table becomes ~uniform gray
  const normalized = new cv.Mat()
  cv.subtract(gray, background, normalized)
  // The subtraction clips at 0; cards (brighter than bg) stay positive,
  // table (close to bg) goes near 0. Add offset and scale for visibility.
  const scaled = new cv.Mat()
  normalized.convertTo(scaled, cv.CV_8U, 3.0, 0)

  // Blur to smooth out shapes inside cards
  const blurSize = Math.round(31 * scale) | 1
  const blurred = new cv.Mat()
  cv.GaussianBlur(scaled, blurred, new cv.Size(blurSize, blurSize), 0)

  // Global threshold on the normalized image (shadow-free)
  const thresh = new cv.Mat()
  cv.threshold(blurred, thresh, 40, 255, cv.THRESH_BINARY)

  // Clear border to prevent edge noise from merging with cards
  const border = Math.max(5, Math.round(50 * scale))
  cv.rectangle(thresh, new cv.Point(0, 0), new cv.Point(width, border), new cv.Scalar(0), cv.FILLED)
  cv.rectangle(thresh, new cv.Point(0, height - border), new cv.Point(width, height), new cv.Scalar(0), cv.FILLED)
  cv.rectangle(thresh, new cv.Point(0, 0), new cv.Point(border, height), new cv.Scalar(0), cv.FILLED)
  cv.rectangle(thresh, new cv.Point(width - border, 0), new cv.Point(width, height), new cv.Scalar(0), cv.FILLED)

  // Erode slightly to break thin connections between cards and edge noise
  const erodeSize = Math.max(3, Math.round(15 * scale))
  const erodeKernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(erodeSize, erodeSize))
  const eroded = new cv.Mat()
  cv.erode(thresh, eroded, erodeKernel)

  // Debug: save intermediate images
  mkdirSync(outputDir, { recursive: true })
  const scaledBuf = Buffer.from(scaled.data)
  await sharp(scaledBuf, { raw: { width, height, channels: 1 } })
    .jpeg()
    .toFile(join(outputDir, "debug-normalized.jpg"))

  const threshBuf = Buffer.from(eroded.data)
  await sharp(threshBuf, { raw: { width, height, channels: 1 } })
    .jpeg()
    .toFile(join(outputDir, "debug-threshold.jpg"))

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

  // Save crops (map coordinates back to original resolution)
  for (let i = 0; i < cards.length; i++) {
    const card = cards[i]
    const pad = Math.round(10 / scale)
    const origX = Math.round(card.x / scale)
    const origY = Math.round(card.y / scale)
    const origW = Math.round(card.w / scale)
    const origH = Math.round(card.h / scale)
    const left = Math.max(0, origX - pad)
    const top = Math.max(0, origY - pad)
    const cropWidth = Math.min(origW + pad * 2, origWidth - left)
    const cropHeight = Math.min(origH + pad * 2, origHeight - top)

    const cropped = await sharp(imageBuffer)
      .extract({ left, top, width: cropWidth, height: cropHeight })
      .jpeg()
      .toBuffer()

    const name = `card-${String(i + 1).padStart(2, "0")}.jpg`
    writeFileSync(join(outputDir, name), cropped)
    console.log(`    ${name}: ${origW}x${origH} at (${origX}, ${origY})`)
  }

  // Cleanup
  mat.delete(); gray.delete(); background.delete(); normalized.delete()
  scaled.delete(); blurred.delete(); thresh.delete()
  erodeKernel.delete(); eroded.delete()
  contours.delete(); hierarchy.delete()
}

async function main() {
  // Wait for OpenCV WASM init
  await new Promise<void>((resolve) => {
    if (typeof cv.Mat === "function") { resolve(); return }
    cv.onRuntimeInitialized = () => resolve()
  })

  const inputPath = process.argv[2] || join(__dirname, "..", "test-images", "fixed")
  const outputBase = process.argv[3] || join(__dirname, "..", "test-images", "cv-crops")

  // Check if input is a directory or a single file
  const stat = require("fs").statSync(inputPath)
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
