// Dispatcher: picks between local ONNX inference (default) and the Python
// inference server (`?server=1` query param). The server path stays around so
// we can compare the two implementations side by side.

type LocalModule = typeof import("./classify-cards-local")
type Impl = { classifyCrops: LocalModule["classifyCrops"] }

function pickImpl(): Promise<Impl> {
  const useServer =
    typeof window !== "undefined" &&
    new URLSearchParams(window.location.search).has("server")
  return useServer
    ? import("./classify-cards-server")
    : import("./classify-cards-local").then((m) => {
        // Kick off model load eagerly so it overlaps with CV detection.
        m.getModel()
        return m
      })
}

const implPromise: Promise<Impl> | null =
  typeof window !== "undefined" ? pickImpl() : null

export async function classifyCrops(crops: Blob[]) {
  const mod = await (implPromise ?? pickImpl())
  return mod.classifyCrops(crops)
}
