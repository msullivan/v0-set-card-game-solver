# set-classifier

Local CNN classifier for Set card attributes (number, color, shape, shading).
Trained on the 336-crop dataset at `../test-images/crops/`. Replaces the
per-crop LLM classification step in the main app.

## Setup

Pick the extra that matches the machine (they're mutually exclusive — the
`conflicts` block in `pyproject.toml` enforces it):

```bash
uv sync --extra cpu     # local dev / smoke tests
uv sync --extra gpu     # CUDA 12.8
uv sync --extra rocm    # ROCm 7.2 (Linux only)
```

Each extra routes `torch` to the matching PyTorch wheel index. Default groups
are empty, so nothing installs implicitly — add `--group dev` to pull in ruff.

## Smoke test

```bash
uv run python -m set_id.smoke
```

Runs both architectures on a tiny CPU subset, verifies the split has disjoint
card identities, and that forward/backward/metrics all work. Takes a few
seconds.

## Training

```bash
uv run python -m set_id.train --arch resnet18
uv run python -m set_id.train --arch small
```

Useful flags:
- `--arch {resnet18,small}` — backbone (ResNet-18 pretrained on ImageNet, or
  a ~1.2M-param custom CNN intended for browser deployment)
- `--phase1-epochs N` / `--phase2-epochs N` — defaults 10 / 40
- `--batch-size N` — default 32
- `--holdout-identities N` — number of unique card identities held out entirely
  for val (default 15, out of 81)
- `--wandb-mode {online,offline,disabled}` — default `online`
- `--run-name STR` — wandb run name

Best checkpoint (by mean per-attribute val accuracy) saves to
`checkpoints/{arch}_best.pt`.

## Serving

```bash
uv run python -m set_id.serve
```

FastAPI app on `127.0.0.1:8000`. `POST /predict` takes
`{"images": ["<base64>", ...]}` (data-URL prefix tolerated) and returns
`{"predictions": [{"color","shape","shading","number"}, ...]}` — the same
shape `lib/analyze-card.ts` returns. Env overrides: `SET_ID_CKPT`,
`SET_ID_ARCH`, `SET_ID_IMG_SIZE`, `SET_ID_HOST`, `SET_ID_PORT`. `arch` and
`img_size` default to whatever the checkpoint was trained with.

## Layout

```
set_id/
  __init__.py
  labels_schema.py    canonical attr -> class id
  dataset.py          labels.json loader + stratified-by-identity split
  augment.py          Albumentations train/val pipelines
  model.py            ResNet18Net + SmallNet, both with 4 heads
  metrics.py          per-attr + full-card accuracy meter
  train.py            phased training loop + wandb
  serve.py            FastAPI batched inference endpoint
  smoke.py            CPU end-to-end sanity run
```

Phase 1: backbone frozen, heads trained with Adam @ lr=1e-3 (~10 epochs).
Phase 2: full fine-tune, cosine schedule, backbone lr=1e-5 + heads lr=1e-4.
Loss is the unweighted sum of four cross-entropies.
