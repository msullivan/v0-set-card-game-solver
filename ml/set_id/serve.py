"""Batched inference server.

POST /predict with {"images": ["<base64 jpeg/png>", ...]} returns
{"predictions": [{"color","shape","shading","number"}, ...]} in the same
shape as `lib/analyze-card.ts` so callers can swap the LLM path for this one.

Run with:
    uv run python -m set_id.serve                # defaults to checkpoints/small_best.pt
    SET_ID_CKPT=checkpoints/resnet18_best.pt uv run python -m set_id.serve
"""

from __future__ import annotations

import base64
import binascii
import io
import os
import pathlib
import sys
from pathlib import Path

# Pre-3.14 pickles reference `pathlib._local.PosixPath`; on 3.14 that module
# no longer exists. Alias it so checkpoints saved on older Python still load.
sys.modules.setdefault("pathlib._local", pathlib)

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from set_id.labels_schema import NAME
from set_id.model import build_model
from set_id.preprocess import preprocess

DEFAULT_CKPT = Path(__file__).resolve().parents[1] / "checkpoints" / "smaller_best.pt"


class PredictRequest(BaseModel):
    images: list[str] = Field(
        ...,
        description="Base64-encoded image bytes (JPEG or PNG). A leading `data:image/...;base64,` prefix is tolerated.",
    )


class CardLogits(BaseModel):
    number: dict[str, float]
    color: dict[str, float]
    shape: dict[str, float]
    shading: dict[str, float]


class CardAttrs(BaseModel):
    color: str
    shape: str
    shading: str
    number: str
    logits: CardLogits


class PredictResponse(BaseModel):
    predictions: list[CardAttrs]


def _decode(b64: str) -> Image.Image:
    s = b64.strip()
    if s.startswith("data:") and "," in s:
        s = s.split(",", 1)[1]
    try:
        raw = base64.b64decode(s, validate=False)
    except (binascii.Error, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"base64 decode failed: {e}")
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise HTTPException(status_code=400, detail=f"image decode failed: {e}")


def create_app(
    ckpt_path: Path | None = None,
    arch: str | None = None,
    img_size: int | None = None,
) -> FastAPI:
    ckpt_path = Path(os.environ.get("SET_ID_CKPT") or ckpt_path or DEFAULT_CKPT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state.get("cfg") or {}
    arch = os.environ.get("SET_ID_ARCH") or arch or cfg.get("arch") or "small"
    img_size = int(os.environ.get("SET_ID_IMG_SIZE") or img_size or cfg.get("img_size") or 128)

    model = build_model(arch, pretrained=False)
    model.load_state_dict(state["state_dict"])
    model.to(device).eval()

    app = FastAPI(title="set-id")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "arch": arch,
            "ckpt": str(ckpt_path),
            "img_size": img_size,
            "device": str(device),
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        if not req.images:
            return PredictResponse(predictions=[])
        tensors = [preprocess(_decode(b), img_size) for b in req.images]
        batch = torch.stack(tensors).to(device)
        with torch.inference_mode():
            logits = model(batch)
        # Head order is (number, color, shape, shading) — see model._Heads.
        logit_rows = [l.tolist() for l in logits]
        argmax = [l.argmax(1).tolist() for l in logits]
        out = [
            CardAttrs(
                number=str(NAME["number"][argmax[0][i]]),
                color=NAME["color"][argmax[1][i]],
                shape=NAME["shape"][argmax[2][i]],
                shading=NAME["shading"][argmax[3][i]],
                logits=CardLogits(
                    number={str(NAME["number"][j]): logit_rows[0][i][j] for j in range(3)},
                    color={NAME["color"][j]: logit_rows[1][i][j] for j in range(3)},
                    shape={NAME["shape"][j]: logit_rows[2][i][j] for j in range(3)},
                    shading={NAME["shading"][j]: logit_rows[3][i][j] for j in range(3)},
                ),
            )
            for i in range(batch.size(0))
        ]
        return PredictResponse(predictions=out)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("SET_ID_HOST", "127.0.0.1"),
        port=int(os.environ.get("SET_ID_PORT", "8000")),
    )
