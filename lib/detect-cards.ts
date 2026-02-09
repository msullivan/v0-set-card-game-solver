import sharp from "sharp"
import path from "path"
import { mkdir } from "fs/promises"

// eslint-disable-next-line @typescript-eslint/no-require-imports
const cv = require("@techstark/opencv-js")

const CV_MAX_DIM = 1000

let cvReady: Promise<void> | null = null

function ensureCVReady(): Promise<void> {
  if (!cvReady) {
    cvReady = new Promise<void>((resolve) => {
      if (typeof cv.Mat === "function") { resolve(); return }
      cv.onRuntimeInitialized = () => resolve()
    })
  }
  return cvReady
}

async function detectCardsFromBuffer(imageBuffer: Buffer, debugDir?: string): Promise<Buffer[]> {
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

  // Background subtraction to normalize lighting
  const bgBlurSize = Math.round(151 * scale) | 1
  const background = new cv.Mat()
  cv.GaussianBlur(gray, background, new cv.Size(bgBlurSize, bgBlurSize), 0)

  const normalized = new cv.Mat()
  cv.subtract(gray, background, normalized)
  const scaled = new cv.Mat()
  normalized.convertTo(scaled, cv.CV_8U, 3.0, 0)

  // Blur to smooth shapes inside cards
  const blurSize = Math.round(31 * scale) | 1
  const blurred = new cv.Mat()
  cv.GaussianBlur(scaled, blurred, new cv.Size(blurSize, blurSize), 0)

  // Threshold
  const thresh = new cv.Mat()
  cv.threshold(blurred, thresh, 40, 255, cv.THRESH_BINARY)

  // Clear border to prevent edge noise
  const border = Math.max(5, Math.round(50 * scale))
  cv.rectangle(thresh, new cv.Point(0, 0), new cv.Point(width, border), new cv.Scalar(0), cv.FILLED)
  cv.rectangle(thresh, new cv.Point(0, height - border), new cv.Point(width, height), new cv.Scalar(0), cv.FILLED)
  cv.rectangle(thresh, new cv.Point(0, 0), new cv.Point(border, height), new cv.Scalar(0), cv.FILLED)
  cv.rectangle(thresh, new cv.Point(width - border, 0), new cv.Point(width, height), new cv.Scalar(0), cv.FILLED)

  // Erode to break thin connections
  const erodeSize = Math.max(3, Math.round(15 * scale))
  const erodeKernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(erodeSize, erodeSize))
  const eroded = new cv.Mat()
  cv.erode(thresh, eroded, erodeKernel)

  // Write debug images if requested
  if (debugDir) {
    await mkdir(debugDir, { recursive: true })
    await sharp(Buffer.from(scaled.data), { raw: { width, height, channels: 1 } })
      .jpeg()
      .toFile(path.join(debugDir, "debug-normalized.jpg"))
    await sharp(Buffer.from(eroded.data), { raw: { width, height, channels: 1 } })
      .jpeg()
      .toFile(path.join(debugDir, "debug-threshold.jpg"))
  }

  // Find contours
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

  // Sort top-to-bottom, left-to-right
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

export interface DetectCardsResult {
  crops: Buffer[]
  timing: { cvInit: number; cvProcess: number }
}

export async function detectCards(imageBuffer: Buffer, debugDir?: string): Promise<DetectCardsResult> {
  const initStart = performance.now()
  await ensureCVReady()
  const cvInit = performance.now() - initStart

  const processStart = performance.now()
  let crops = await detectCardsFromBuffer(imageBuffer, debugDir)
  if (crops.length === 0) {
    // No cards found — try rotating 90 degrees in case image is landscape
    const rotated = await sharp(imageBuffer).rotate(90).jpeg().toBuffer()
    crops = await detectCardsFromBuffer(rotated, debugDir)
  }
  const cvProcess = performance.now() - processStart

  return { crops, timing: { cvInit, cvProcess } }
}
