// Local card classifier using ONNX inference in the browser.
// Replaces the previous HTTP-based path that talked to a Python server.
//
// Uses onnxruntime-web/all for WebGPU support (falls back to WASM).
// The ort import is dynamic to avoid SSR evaluation in Next.js.

import {
  classifyCards,
  type SetIdSession,
  type ModelMeta,
  type RGBImage,
} from "./analyze-card-local"

export type { CardLogits, LocalCardPrediction as CardPrediction } from "./analyze-card-local"

const MODEL_URL = "/models/set_id_smaller.onnx"
const META_URL = "/models/set_id_smaller.json"

let modelPromise: Promise<SetIdSession> | null = null

function getModel(): Promise<SetIdSession> {
  if (!modelPromise) {
    modelPromise = (async () => {
      const ort = await import("onnxruntime-web/all")
      ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/"

      const [modelBuf, metaResp] = await Promise.all([
        fetch(MODEL_URL).then((r) => {
          if (!r.ok) throw new Error(`Failed to load model: ${r.status}`)
          return r.arrayBuffer()
        }),
        fetch(META_URL).then((r) => (r.ok ? r.json() : undefined)),
      ])
      const meta = metaResp as ModelMeta | undefined
      const bytes = new Uint8Array(modelBuf)
      const session = await ort.InferenceSession.create(bytes, {
        executionProviders: ["webgpu", "webgl", "wasm"],
      })
      const imgSize = meta?.img_size ?? 128
      return { session, imgSize }
    })()
  }
  return modelPromise
}

// Kick off model loading immediately when this module is first imported
// (which happens at page load since page.tsx imports us).
if (typeof window !== "undefined") {
  getModel()
}

async function blobToRGB(blob: Blob): Promise<RGBImage> {
  const bmp = await createImageBitmap(blob)
  const canvas = new OffscreenCanvas(bmp.width, bmp.height)
  const ctx = canvas.getContext("2d")!
  ctx.drawImage(bmp, 0, 0)
  bmp.close()
  const { data, width, height } = ctx.getImageData(0, 0, canvas.width, canvas.height)

  // Strip alpha: RGBA → RGB
  const rgb = new Uint8Array(width * height * 3)
  for (let i = 0, j = 0; i < data.length; i += 4, j += 3) {
    rgb[j] = data[i]
    rgb[j + 1] = data[i + 1]
    rgb[j + 2] = data[i + 2]
  }
  return { data: rgb, width, height }
}

export async function classifyCrops(crops: Blob[]) {
  if (crops.length === 0) return { predictions: [] }
  const [model, images] = await Promise.all([
    getModel(),
    Promise.all(crops.map(blobToRGB)),
  ])
  const predictions = await classifyCards(model, images)
  return { predictions }
}
