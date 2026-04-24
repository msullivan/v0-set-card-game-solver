# set-classifier

Local CNN classifier for Set card attributes (number, color, shape, shading).
Trained on the 336-crop dataset at `../test-images/crops/`. Replaces the
per-crop LLM classification step in the main app.

## Setup

```bash
uv sync
```

On a ROCm box, swap torch after the initial sync:

```bash
uv pip install --reinstall torch torchvision \
  --index-url https://download.pytorch.org/whl/rocm6.2
```

(Adjust `rocm6.2` to match the box's ROCm version.) The CPU wheels from PyPI
are fine for smoke-testing and code edits locally.

## Smoke test

```bash
uv run python src/smoke.py
```

Runs both architectures on a tiny CPU subset, verifies the split has disjoint
card identities, and that forward/backward/metrics all work. Takes a few
seconds.

## Training

```bash
uv run python src/train.py --arch resnet18
uv run python src/train.py --arch small
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

## Layout

```
src/
  labels_schema.py    canonical attr -> class id
  dataset.py          labels.json loader + stratified-by-identity split
  augment.py          Albumentations train/val pipelines
  model.py            ResNet18Net + SmallNet, both with 4 heads
  metrics.py          per-attr + full-card accuracy meter
  train.py            phased training loop + wandb
  smoke.py            CPU end-to-end sanity run
```

Phase 1: backbone frozen, heads trained with Adam @ lr=1e-3 (~10 epochs).
Phase 2: full fine-tune, cosine schedule, backbone lr=1e-5 + heads lr=1e-4.
Loss is the unweighted sum of four cross-entropies.
