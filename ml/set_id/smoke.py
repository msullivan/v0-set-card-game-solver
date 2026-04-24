"""CPU smoke test. Verifies the full pipeline runs end-to-end on a tiny subset.

Not intended for real training — just confirms:
  - labels.json loads
  - stratified split produces disjoint identities
  - both architectures forward + backward cleanly
  - metrics.py computes reasonable numbers
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from set_id.augment import train_transform, val_transform
from set_id.dataset import CardDataset, load_samples, stratified_split
from set_id.metrics import AccMeter
from set_id.model import ARCHES, build_model, param_count

ATTRS = ("number", "color", "shape", "shading")


def unpack(batch):
    img, number, color, shape, shading = batch
    return img, {"number": number, "color": color, "shape": shape, "shading": shading}


def main() -> int:
    crops_dir = Path(__file__).resolve().parents[2] / "test-images" / "crops"
    print(f"crops dir: {crops_dir}")

    samples = load_samples(crops_dir)
    print(f"samples: {len(samples)}")

    train, val = stratified_split(samples, holdout_identities=15, seed=0)
    print(f"train: {len(train)}  val: {len(val)}")
    train_ids = {s.identity for s in train}
    val_ids = {s.identity for s in val}
    assert train_ids.isdisjoint(val_ids), "train/val identity overlap"
    print(f"unique train identities: {len(train_ids)}  val: {len(val_ids)}")

    # Tiny subset, tiny image, tiny batch
    train_ds = CardDataset(train[:24], train_transform(img_size=64))
    val_ds = CardDataset(val[:12], val_transform(img_size=64))
    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

    device = torch.device("cpu")

    for arch in ARCHES:
        print(f"\n--- arch: {arch} ---")
        # For smoke, skip ImageNet download on resnet18 by using pretrained=False
        model = build_model(arch, pretrained=False).to(device)
        print(f"params: {param_count(model):,}")

        # one train step
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        batch = next(iter(train_dl))
        img, labels = unpack(batch)
        logits = model(img)
        loss = sum(F.cross_entropy(logits[i], labels[a]) for i, a in enumerate(ATTRS))
        loss.backward()
        opt.step()
        print(f"train loss (1 step): {loss.item():.4f}")

        # one val epoch
        model.eval()
        meter = AccMeter()
        with torch.no_grad():
            for batch in val_dl:
                img, labels = unpack(batch)
                logits = model(img)
                preds = {a: logits[i].argmax(1).tolist() for i, a in enumerate(ATTRS)}
                gts = {a: labels[a].tolist() for a in ATTRS}
                meter.update(preds, gts)
        m = meter.summary()
        print(f"val summary: {m}")
        # Fresh untrained model should be near random (1/3 per attr) — just
        # sanity check the numbers are in [0, 1].
        for k, v in m.items():
            assert 0.0 <= v <= 1.0, f"{k}={v} out of range"

    print("\nsmoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
