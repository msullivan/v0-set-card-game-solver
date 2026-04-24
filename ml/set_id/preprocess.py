"""Inference-time image preprocessing.

Pure PIL + torch — no albumentations/opencv — so inference deployments can skip
those deps. Must stay numerically close to `set_id.augment.val_transform` so
checkpoints trained with that pipeline see the same distribution at serve time:
longest side resized to `img_size`, center-padded to `img_size` square with
zeros, normalized with ImageNet mean/std.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def preprocess(img: Image.Image, img_size: int = 128) -> torch.Tensor:
    """PIL RGB image -> normalized CHW float32 tensor of shape (3, img_size, img_size)."""
    img = img.convert("RGB")
    w, h = img.size
    scale = img_size / max(w, h)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    img = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (img_size, img_size), (0, 0, 0))
    canvas.paste(img, ((img_size - new_w) // 2, (img_size - new_h) // 2))
    arr = torch.from_numpy(np.array(canvas)).float().div_(255.0).permute(2, 0, 1)
    arr.sub_(IMAGENET_MEAN).div_(IMAGENET_STD)
    return arr
