// Talks to the Python inference server's /predict endpoint.

import type { SetCard } from "./set-game"

const DEFAULT_URL = process.env.NEXT_PUBLIC_SET_ID_URL || "http://127.0.0.1:8001"

export type CardLogits = {
  number: Record<string, number>
  color: Record<string, number>
  shape: Record<string, number>
  shading: Record<string, number>
}

type PredictResponse = {
  predictions: {
    color: SetCard["color"]
    shape: SetCard["shape"]
    shading: SetCard["shading"]
    number: "1" | "2" | "3"
    logits: CardLogits
  }[]
}

async function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => {
      const s = reader.result as string
      const comma = s.indexOf(",")
      resolve(comma >= 0 ? s.slice(comma + 1) : s)
    }
    reader.onerror = () => reject(reader.error)
    reader.readAsDataURL(blob)
  })
}

export async function classifyCrops(crops: Blob[], serverUrl: string = DEFAULT_URL) {
  if (crops.length === 0) return { predictions: [] as PredictResponse["predictions"] }
  const images = await Promise.all(crops.map(blobToBase64))
  const resp = await fetch(`${serverUrl}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ images }),
  })
  if (!resp.ok) {
    throw new Error(`classifier ${resp.status}: ${await resp.text()}`)
  }
  return (await resp.json()) as PredictResponse
}
