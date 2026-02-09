import sharp from "sharp"
import path from "path"
import { mkdir } from "fs/promises"

// eslint-disable-next-line @typescript-eslint/no-require-imports
const cv = require("@techstark/opencv-js")

const CV_MAX_DIM = 1000
const GRID_MIN_CARDS = 6

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

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
}

function clusterValues(values: number[], threshold: number): number[][] {
  const sorted = [...values].sort((a, b) => a - b)
  const clusters: number[][] = [[sorted[0]]]
  for (let i = 1; i < sorted.length; i++) {
    if (sorted[i] - sorted[i - 1] <= threshold) {
      clusters[clusters.length - 1].push(sorted[i])
    } else {
      clusters.push([sorted[i]])
    }
  }
  return clusters
}

/**
 * Infer missing cards from grid geometry.
 * When most cards in a regular grid are detected, compute the expected grid
 * positions and add synthetic rects for any empty cells.
 */
function inferMissingGridCards(
  rects: { x: number; y: number; w: number; h: number }[],
  debugDir?: string,
): { x: number; y: number; w: number; h: number }[] {
  if (rects.length < GRID_MIN_CARDS) return rects

  const medW = median(rects.map(r => r.w))
  const medH = median(rects.map(r => r.h))

  // Cluster centers into columns and rows
  const xCenters = rects.map(r => r.x + r.w / 2)
  const yCenters = rects.map(r => r.y + r.h / 2)

  const colClusters = clusterValues(xCenters, medW * 0.5)
  const rowClusters = clusterValues(yCenters, medH * 0.5)

  // Need at least 2 rows and 2 columns to form a grid
  if (colClusters.length < 2 || rowClusters.length < 2) return rects

  const colCenters = colClusters.map(c => median(c)).sort((a, b) => a - b)
  const rowCenters = rowClusters.map(c => median(c)).sort((a, b) => a - b)

  const totalCells = colCenters.length * rowCenters.length
  // Only infer if we have most of the grid already (at least 75%)
  if (rects.length < totalCells * 0.75) return rects

  const result = [...rects]
  let added = 0

  for (const rowY of rowCenters) {
    for (const colX of colCenters) {
      // Check if a detected card exists near this grid intersection
      const nearby = rects.some(r => {
        const cx = r.x + r.w / 2
        const cy = r.y + r.h / 2
        return Math.abs(cx - colX) < medW * 0.5 && Math.abs(cy - rowY) < medH * 0.5
      })
      if (!nearby) {
        const synth = {
          x: Math.max(0, Math.round(colX - medW / 2)),
          y: Math.max(0, Math.round(rowY - medH / 2)),
          w: Math.round(medW),
          h: Math.round(medH),
        }
        result.push(synth)
        added++
        if (debugDir) {
          console.log(`  grid inferred card at (${synth.x},${synth.y}) ${synth.w}x${synth.h}`)
        }
      }
    }
  }

  if (debugDir && added > 0) {
    console.log(`  grid inference: ${colCenters.length} cols × ${rowCenters.length} rows, added ${added} card(s)`)
  }

  return result
}

async function detectCardsFromBuffer(imageBuffer: Buffer, debugDir?: string): Promise<Buffer[]> {
  // Apply EXIF rotation so pixel dimensions match visual orientation
  imageBuffer = await sharp(imageBuffer).rotate().jpeg().toBuffer()
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

  // Local contrast normalization: divide by local max to equalize dim/bright areas
  const normKernelSize = Math.round(101 * scale) | 1
  const normKernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(normKernelSize, normKernelSize))
  const localMax = new cv.Mat()
  cv.dilate(normalized, localMax, normKernel)
  // Clamp localMax to min 1 to avoid division by zero
  const ones = new cv.Mat(height, width, cv.CV_8U, new cv.Scalar(1))
  cv.max(localMax, ones, localMax)
  ones.delete()
  const scaled = new cv.Mat()
  cv.divide(normalized, localMax, scaled, 255.0)

  // Blur to smooth shapes inside cards
  const blurSize = Math.round(31 * scale) | 1
  const blurred = new cv.Mat()
  cv.GaussianBlur(scaled, blurred, new cv.Size(blurSize, blurSize), 0)

  // Threshold
  const thresh = new cv.Mat()
  cv.threshold(blurred, thresh, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

  // Clear border to prevent edge noise
  const border = Math.max(5, Math.round(50 * scale))
  cv.rectangle(thresh, new cv.Point(0, 0), new cv.Point(width, border), new cv.Scalar(0), cv.FILLED)
  cv.rectangle(thresh, new cv.Point(0, height - border), new cv.Point(width, height), new cv.Scalar(0), cv.FILLED)
  cv.rectangle(thresh, new cv.Point(0, 0), new cv.Point(border, height), new cv.Scalar(0), cv.FILLED)
  cv.rectangle(thresh, new cv.Point(width - border, 0), new cv.Point(width, height), new cv.Scalar(0), cv.FILLED)

  // Erode to break thin connections between adjacent cards
  const erodeSize = Math.max(3, Math.round(10 * scale))
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
    const aspect = rect.height / rect.width
    const rectArea = rect.width * rect.height
    const rectangularity = area / rectArea

    if (debugDir) {
      const dominated = areaRatio >= 0.005 && areaRatio <= 0.08
      if (dominated || areaRatio >= 0.002) {
        console.log(`  contour ${i}: area=${areaRatio.toFixed(4)} aspect=${aspect.toFixed(2)} rect=${rectangularity.toFixed(2)} ${rect.width}x${rect.height} at (${rect.x},${rect.y})${areaRatio < 0.005 || areaRatio > 0.08 ? ' REJECT:area' : aspect < 0.9 || aspect > 2.2 ? ' REJECT:aspect' : rectangularity < 0.25 ? ' REJECT:rect' : ' OK'}`)
      }
    }

    if (areaRatio < 0.005 || areaRatio > 0.08) continue
    if (aspect < 0.9 || aspect > 2.2) continue
    if (rectangularity < 0.25) continue

    rects.push({ x: rect.x, y: rect.y, w: rect.width, h: rect.height })
  }

  // Infer missing cards from grid geometry
  const allRects = inferMissingGridCards(rects, debugDir)

  // Sort top-to-bottom, left-to-right
  allRects.sort((a, b) => {
    const rowA = Math.round(a.y / (height * 0.1))
    const rowB = Math.round(b.y / (height * 0.1))
    if (rowA !== rowB) return rowA - rowB
    return a.x - b.x
  })

  // Crop at full resolution
  const crops: Buffer[] = []
  for (const r of allRects) {
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
  normKernel.delete(); localMax.delete()
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
