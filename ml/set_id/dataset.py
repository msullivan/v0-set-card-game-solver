"""Dataset + stratified-by-card-identity split.

Holds out whole card identities from train so val measures generalization,
not memorization of specific photos.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from set_id.labels_schema import IDX


@dataclass(frozen=True)
class Sample:
    path: Path
    source_image: str
    number: int
    color: int
    shape: int
    shading: int

    @property
    def identity(self) -> tuple[int, int, int, int]:
        return (self.number, self.color, self.shape, self.shading)


def load_samples(
    crops_dir: Path,
    exclude_sources: Sequence[str] = (),
) -> list[Sample]:
    labels_path = crops_dir / "labels.json"
    data = json.loads(labels_path.read_text())
    excluded = set(exclude_sources)
    out: list[Sample] = []
    for fname, lab in data["labels"].items():
        if lab["source_image"] in excluded:
            continue
        out.append(
            Sample(
                path=crops_dir / fname,
                source_image=lab["source_image"],
                number=IDX["number"][lab["number"]],
                color=IDX["color"][lab["color"]],
                shape=IDX["shape"][lab["shape"]],
                shading=IDX["shading"][lab["shading"]],
            )
        )
    return out


def stratified_split(
    samples: Sequence[Sample],
    holdout_identities: int = 15,
    seed: int = 0,
) -> tuple[list[Sample], list[Sample]]:
    """Hold out `holdout_identities` whole identities for val.

    Every crop of a held-out identity goes to val; none appear in train.
    """
    rng = random.Random(seed)
    identities = sorted({s.identity for s in samples})
    rng.shuffle(identities)
    val_ids = set(identities[:holdout_identities])
    train = [s for s in samples if s.identity not in val_ids]
    val = [s for s in samples if s.identity in val_ids]
    return train, val


class CardDataset(Dataset):
    def __init__(self, samples: Sequence[Sample], transform):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = np.array(Image.open(s.path).convert("RGB"))
        out = self.transform(image=img)["image"]
        return out, s.number, s.color, s.shape, s.shading
