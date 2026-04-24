// Local (in-process) card classifier using an ONNX-exported checkpoint.
//
// Mirrors the response shape of the Python /predict server (see
// `lib/classify-cards.ts`), so callers can swap implementations freely.
// Runs in both Node (for test scripts) and the browser via onnxruntime-web.
//
// Preprocessing must stay numerically close to `ml/set_id/preprocess.py`:
// longest side → img_size (bilinear), center-padded to a square with zeros
// in original pixel space, then ImageNet mean/std normalized.

import * as ort from "onnxruntime-web"

import type { SetCard } from "./set-game"

// Matches ml/set_id/labels_schema.py. Head order in the exported ONNX graph
// is irrelevant because we use named outputs, but class index order must
// match the training-time ordering exactly.
const NUMBERS = ["1", "2", "3"] as const
const COLORS = ["red", "green", "purple"] as const
const SHAPES = ["diamond", "oval", "squiggle"] as const
const SHADINGS = ["empty", "striped", "solid"] as const

const MEAN = [0.485, 0.456, 0.406]
const STD = [0.229, 0.224, 0.225]

// Fallback if neither a sidecar meta nor an explicit imgSize is provided.
// Real img_size comes from the exported checkpoint (see export_onnx.py).
const DEFAULT_IMG_SIZE = 128

export type ModelMeta = {
  arch?: string
  img_size?: number
  opset?: number
}

export type RGBImage = {
  // Interleaved RGB, no alpha. Length must be width * height * 3.
  data: Uint8Array | Uint8ClampedArray
  width: number
  height: number
}

export type CardLogits = {
  number: Record<string, number>
  color: Record<string, number>
  shape: Record<string, number>
  shading: Record<string, number>
}

export type LocalCardPrediction = {
  color: SetCard["color"]
  shape: SetCard["shape"]
  shading: SetCard["shading"]
  number: "1" | "2" | "3"
  logits: CardLogits
}

export type SetIdSession = {
  session: ort.InferenceSession
  imgSize: number
}

export type LoadOptions = {
  // Explicit override. Otherwise taken from `meta.img_size`, then DEFAULT_IMG_SIZE.
  imgSize?: number
  meta?: ModelMeta
  executionProviders?: ort.InferenceSession.SessionOptions["executionProviders"]
}

export async function loadSetIdModel(
  modelBytes: ArrayBuffer | Uint8Array,
  opts: LoadOptions = {},
): Promise<SetIdSession> {
  // Normalize to Uint8Array — ort's overload resolution doesn't accept the union.
  const bytes =
    modelBytes instanceof Uint8Array ? modelBytes : new Uint8Array(modelBytes)
  const session = await ort.InferenceSession.create(bytes, {
    executionProviders: opts.executionProviders ?? ["wasm"],
  })
  const imgSize = opts.imgSize ?? opts.meta?.img_size ?? DEFAULT_IMG_SIZE
  return { session, imgSize }
}

// Bilinear-sample + center-pad + normalize → CHW float32 tensor of
// shape (3, size, size). Writes into `dst` at the given batch offset
// (offset = i * 3 * size * size) to avoid per-card allocations.
function preprocessInto(dst: Float32Array, offset: number, img: RGBImage, size: number): void {
  const scale = size / Math.max(img.width, img.height)
  const newW = Math.max(1, Math.round(img.width * scale))
  const newH = Math.max(1, Math.round(img.height * scale))
  const offX = Math.floor((size - newW) / 2)
  const offY = Math.floor((size - newH) / 2)

  const plane = size * size

  // Fill the whole batch slice with the normalized value of a zero pixel.
  // Anything the resized image doesn't cover stays as "padding" in the
  // same sense as preprocess.py's black canvas.
  for (let c = 0; c < 3; c++) {
    const padVal = (0 - MEAN[c]) / STD[c]
    dst.fill(padVal, offset + c * plane, offset + (c + 1) * plane)
  }

  // Pixel-centered bilinear, matching the common resampling convention.
  // PIL's bilinear differs slightly but the CNN is robust at 128×128.
  const invScaleX = img.width / newW
  const invScaleY = img.height / newH
  const srcW = img.width
  const srcH = img.height
  const src = img.data

  for (let y = 0; y < newH; y++) {
    let srcY = (y + 0.5) * invScaleY - 0.5
    if (srcY < 0) srcY = 0
    else if (srcY > srcH - 1) srcY = srcH - 1
    const y0 = Math.floor(srcY)
    const y1 = Math.min(srcH - 1, y0 + 1)
    const fy = srcY - y0
    const dstRow = (offY + y) * size

    for (let x = 0; x < newW; x++) {
      let srcX = (x + 0.5) * invScaleX - 0.5
      if (srcX < 0) srcX = 0
      else if (srcX > srcW - 1) srcX = srcW - 1
      const x0 = Math.floor(srcX)
      const x1 = Math.min(srcW - 1, x0 + 1)
      const fx = srcX - x0

      const w00 = (1 - fx) * (1 - fy)
      const w01 = fx * (1 - fy)
      const w10 = (1 - fx) * fy
      const w11 = fx * fy

      const i00 = (y0 * srcW + x0) * 3
      const i01 = (y0 * srcW + x1) * 3
      const i10 = (y1 * srcW + x0) * 3
      const i11 = (y1 * srcW + x1) * 3

      const dstCol = offX + x
      const dstBase = dstRow + dstCol

      for (let c = 0; c < 3; c++) {
        const v =
          (src[i00 + c] * w00 +
            src[i01 + c] * w01 +
            src[i10 + c] * w10 +
            src[i11 + c] * w11) /
          255
        dst[offset + c * plane + dstBase] = (v - MEAN[c]) / STD[c]
      }
    }
  }
}

function argmax3(a: Float32Array, base: number): number {
  const v0 = a[base]
  const v1 = a[base + 1]
  const v2 = a[base + 2]
  if (v0 >= v1 && v0 >= v2) return 0
  return v1 >= v2 ? 1 : 2
}

function toLogitMap<T extends readonly string[]>(
  labels: T,
  a: Float32Array,
  base: number,
): Record<T[number], number> {
  const out = {} as Record<T[number], number>
  for (let j = 0; j < labels.length; j++) {
    out[labels[j] as T[number]] = a[base + j]
  }
  return out
}

export async function classifyCards(
  model: SetIdSession,
  images: RGBImage[],
): Promise<LocalCardPrediction[]> {
  if (images.length === 0) return []

  const { session, imgSize } = model
  const N = images.length
  const perSample = 3 * imgSize * imgSize
  const batch = new Float32Array(N * perSample)
  for (let i = 0; i < N; i++) {
    preprocessInto(batch, i * perSample, images[i], imgSize)
  }

  const input = new ort.Tensor("float32", batch, [N, 3, imgSize, imgSize])
  const results = await session.run({ pixels: input })

  const num = results.number_logits.data as Float32Array
  const col = results.color_logits.data as Float32Array
  const shp = results.shape_logits.data as Float32Array
  const shd = results.shading_logits.data as Float32Array

  const out: LocalCardPrediction[] = []
  for (let i = 0; i < N; i++) {
    const b = i * 3
    out.push({
      number: NUMBERS[argmax3(num, b)],
      color: COLORS[argmax3(col, b)],
      shape: SHAPES[argmax3(shp, b)],
      shading: SHADINGS[argmax3(shd, b)],
      logits: {
        number: toLogitMap(NUMBERS, num, b),
        color: toLogitMap(COLORS, col, b),
        shape: toLogitMap(SHAPES, shp, b),
        shading: toLogitMap(SHADINGS, shd, b),
      },
    })
  }
  return out
}
