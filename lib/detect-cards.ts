// Browser-only card detection. Mirrors detect-cards-node.ts but uses
// createImageBitmap + OffscreenCanvas instead of sharp. OpenCV.js WASM is
// lazy-loaded on first call so the ~10MB blob doesn't bloat initial page load.

const CV_MAX_DIM = 1000
const GRID_MIN_CARDS = 6

/* eslint-disable @typescript-eslint/no-explicit-any */
declare global { interface Window { cv: any } }

// opencv.js is loaded via <Script> in app/layout.tsx so its 10 MB WASM blob
// stays out of Turbopack's module graph. Matches the three-state init pattern
// from @techstark/opencv-js's README, adapted for script-tag loading: wait
// for window.cv, then for its runtime. We return `{ cv }` (not `cv` directly)
// because Emscripten modules have a `.then` method and passing one to
// `resolve()` makes the Promise adopt a thenable chain that never settles.
async function getOpenCv(): Promise<{ cv: any }> {
  while (!window.cv) await new Promise((r) => setTimeout(r, 20))
  const cvModule = window.cv
  if (cvModule.Mat) return { cv: cvModule }
  await new Promise<void>((resolve) => { cvModule.onRuntimeInitialized = () => resolve() })
  return { cv: cvModule }
}

let cvPromise: Promise<{ cv: any }> | null = null
function ensureCV(): Promise<{ cv: any }> {
  return cvPromise ??= getOpenCv()
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

type Rect = { x: number; y: number; w: number; h: number }

// When most cards in a regular grid were detected, fill in the empty cells
// with synthetic rects at the expected grid positions. Lets a single missed
// contour (e.g. a card that got merged with a neighbor) still produce a crop.
function inferMissingGridCards(rects: Rect[]): Rect[] {
  if (rects.length < GRID_MIN_CARDS) return rects
  const medW = median(rects.map((r) => r.w))
  const medH = median(rects.map((r) => r.h))
  const xCenters = rects.map((r) => r.x + r.w / 2)
  const yCenters = rects.map((r) => r.y + r.h / 2)
  const colClusters = clusterValues(xCenters, medW * 0.5)
  const rowClusters = clusterValues(yCenters, medH * 0.5)
  // Need at least a 2x2 grid to infer geometry.
  if (colClusters.length < 2 || rowClusters.length < 2) return rects
  const colCenters = colClusters.map(median).sort((a, b) => a - b)
  const rowCenters = rowClusters.map(median).sort((a, b) => a - b)
  const totalCells = colCenters.length * rowCenters.length
  // Only infer when most of the grid is already detected, otherwise a sparse
  // layout would spawn a flood of false-positive cells.
  if (rects.length < totalCells * 0.75) return rects
  const result = [...rects]
  for (const rowY of rowCenters) {
    for (const colX of colCenters) {
      const nearby = rects.some((r) => {
        const cx = r.x + r.w / 2
        const cy = r.y + r.h / 2
        return Math.abs(cx - colX) < medW * 0.5 && Math.abs(cy - rowY) < medH * 0.5
      })
      if (!nearby) {
        result.push({
          x: Math.max(0, Math.round(colX - medW / 2)),
          y: Math.max(0, Math.round(rowY - medH / 2)),
          w: Math.round(medW),
          h: Math.round(medH),
        })
      }
    }
  }
  return result
}

async function toBitmap(input: Blob | string): Promise<ImageBitmap> {
  if (typeof input === "string") {
    const resp = await fetch(input)
    return createImageBitmap(await resp.blob(), { imageOrientation: "from-image" })
  }
  return createImageBitmap(input, { imageOrientation: "from-image" })
}

async function rotateBitmap90(bitmap: ImageBitmap): Promise<ImageBitmap> {
  const w = bitmap.height
  const h = bitmap.width
  const canvas = new OffscreenCanvas(w, h)
  const ctx = canvas.getContext("2d")!
  ctx.translate(w / 2, h / 2)
  ctx.rotate(Math.PI / 2)
  ctx.drawImage(bitmap, -bitmap.width / 2, -bitmap.height / 2)
  return createImageBitmap(canvas)
}

async function detectCardsFromBitmap(bitmap: ImageBitmap): Promise<Blob[]> {
  const { cv } = await ensureCV()
  const origWidth = bitmap.width
  const origHeight = bitmap.height

  const scale = Math.min(1, CV_MAX_DIM / Math.max(origWidth, origHeight))
  const width = Math.round(origWidth * scale)
  const height = Math.round(origHeight * scale)

  const scaledCanvas = new OffscreenCanvas(width, height)
  const sctx = scaledCanvas.getContext("2d", { willReadFrequently: true })!
  sctx.drawImage(bitmap, 0, 0, width, height)
  const imgData = sctx.getImageData(0, 0, width, height)

  const mat = cv.matFromImageData(imgData) // 4-channel RGBA from canvas
  const gray = new cv.Mat()
  cv.cvtColor(mat, gray, cv.COLOR_RGBA2GRAY)

  // Subtract a heavily-blurred copy to flatten uneven lighting and shadows
  // before thresholding — keeps cards on dark table surfaces recoverable.
  const bgBlurSize = Math.round(151 * scale) | 1
  const background = new cv.Mat()
  cv.GaussianBlur(gray, background, new cv.Size(bgBlurSize, bgBlurSize), 0)

  const normalized = new cv.Mat()
  cv.subtract(gray, background, normalized)

  // Local contrast normalization: divide by local max so dim and bright
  // regions end up on the same footing. Clamp localMax to >=1 first so the
  // divide can't hit zero.
  const normKernelSize = Math.round(101 * scale) | 1
  const normKernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(normKernelSize, normKernelSize))
  const localMax = new cv.Mat()
  cv.dilate(normalized, localMax, normKernel)
  const ones = new cv.Mat(height, width, cv.CV_8U, new cv.Scalar(1))
  cv.max(localMax, ones, localMax)
  ones.delete()
  const scaledMat = new cv.Mat()
  cv.divide(normalized, localMax, scaledMat, 255.0)

  // Smooth out the symbols printed inside cards so Otsu doesn't latch onto
  // them and carve up each card's interior into multiple contours.
  const blurSize = Math.round(31 * scale) | 1
  const blurred = new cv.Mat()
  cv.GaussianBlur(scaledMat, blurred, new cv.Size(blurSize, blurSize), 0)

  // Threshold
  const thresh = new cv.Mat()
  cv.threshold(blurred, thresh, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

  // Zero out a border so image-edge artifacts can't become contours.
  const border = Math.max(5, Math.round(50 * scale))
  cv.rectangle(thresh, new cv.Point(0, 0), new cv.Point(width, border), new cv.Scalar(0), cv.FILLED)
  cv.rectangle(thresh, new cv.Point(0, height - border), new cv.Point(width, height), new cv.Scalar(0), cv.FILLED)
  cv.rectangle(thresh, new cv.Point(0, 0), new cv.Point(border, height), new cv.Scalar(0), cv.FILLED)
  cv.rectangle(thresh, new cv.Point(width - border, 0), new cv.Point(width, height), new cv.Scalar(0), cv.FILLED)

  // Erode to break thin bridges where adjacent cards touch, so each card
  // ends up as its own connected component.
  const erodeSize = Math.max(3, Math.round(10 * scale))
  const erodeKernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(erodeSize, erodeSize))
  const eroded = new cv.Mat()
  cv.erode(thresh, eroded, erodeKernel)

  // Find contours
  const contours = new cv.MatVector()
  const hierarchy = new cv.Mat()
  cv.findContours(eroded, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

  const imgArea = width * height
  const rects: Rect[] = []

  for (let i = 0; i < contours.size(); i++) {
    const contour = contours.get(i)
    const area = cv.contourArea(contour)
    const rect = cv.boundingRect(contour)
    const aspect = rect.height / rect.width
    const rectArea = rect.width * rect.height
    const bboxAreaRatio = rectArea / imgArea
    const rectangularity = area / rectArea
    // Accept either a clean rectangular contour or a ragged-but-card-shaped
    // one — striped shading can fragment a card's interior into a ring-ish
    // contour with low rectangularity but the correct bounding box.
    const cleanCard = rectangularity >= 0.25 && aspect >= 0.9 && aspect <= 2.2
    const raggedCard = rectangularity >= 0.08 && aspect >= 1.2 && aspect <= 1.8
    const accepted = (cleanCard || raggedCard) && bboxAreaRatio >= 0.015 && bboxAreaRatio <= 0.08
    if (!accepted) continue
    rects.push({ x: rect.x, y: rect.y, w: rect.width, h: rect.height })
  }

  // Infer missing cards from grid geometry
  const allRects = inferMissingGridCards(rects)

  // Sort top-to-bottom, left-to-right using clustering for row assignment
  if (allRects.length >= 2) {
    const medH = median(allRects.map((r) => r.h))
    const rowClusters = clusterValues(allRects.map((r) => r.y + r.h / 2), medH * 0.5)
    const rowCenters = rowClusters.map(median).sort((a, b) => a - b)
    const rowOf = (r: { y: number; h: number }) => {
      let best = 0
      let bestDist = Infinity
      for (let i = 0; i < rowCenters.length; i++) {
        const d = Math.abs(r.y + r.h / 2 - rowCenters[i])
        if (d < bestDist) { best = i; bestDist = d }
      }
      return best
    }
    allRects.sort((a, b) => {
      const rowDiff = rowOf(a) - rowOf(b)
      if (rowDiff !== 0) return rowDiff
      return a.x - b.x
    })
  }

  // Crop at full resolution. The scaled pipeline only chose bounding boxes;
  // we draw the original bitmap to a full-res canvas and slice from there so
  // the cropped JPEGs carry the model's real input pixels, not upscaled ones.
  const fullCanvas = new OffscreenCanvas(origWidth, origHeight)
  const fctx = fullCanvas.getContext("2d")!
  fctx.drawImage(bitmap, 0, 0)

  const crops: Blob[] = []
  for (const r of allRects) {
    const pad = Math.round(10 / scale)
    const left = Math.max(0, Math.round(r.x / scale) - pad)
    const top = Math.max(0, Math.round(r.y / scale) - pad)
    const cropWidth = Math.min(Math.round(r.w / scale) + pad * 2, origWidth - left)
    const cropHeight = Math.min(Math.round(r.h / scale) + pad * 2, origHeight - top)

    const cc = new OffscreenCanvas(cropWidth, cropHeight)
    const cctx = cc.getContext("2d")!
    cctx.drawImage(fullCanvas, left, top, cropWidth, cropHeight, 0, 0, cropWidth, cropHeight)
    crops.push(await cc.convertToBlob({ type: "image/jpeg", quality: 0.9 }))
  }

  // Cleanup
  mat.delete(); gray.delete(); background.delete(); normalized.delete()
  normKernel.delete(); localMax.delete()
  scaledMat.delete(); blurred.delete(); thresh.delete()
  erodeKernel.delete(); eroded.delete()
  contours.delete(); hierarchy.delete()

  return crops
}

export interface DetectCardsResult {
  crops: Blob[]
  timing: { cvInit: number; cvProcess: number }
}

export async function detectCards(input: Blob | string): Promise<DetectCardsResult> {
  const initStart = performance.now()
  await ensureCV()
  const cvInit = performance.now() - initStart

  const processStart = performance.now()
  const bitmap = await toBitmap(input)
  let crops = await detectCardsFromBitmap(bitmap)
  if (crops.length === 0) {
    const rotated = await rotateBitmap90(bitmap)
    crops = await detectCardsFromBitmap(rotated)
    rotated.close()
  }
  bitmap.close()
  const cvProcess = performance.now() - processStart

  return { crops, timing: { cvInit, cvProcess } }
}
