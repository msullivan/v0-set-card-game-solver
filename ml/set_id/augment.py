"""Train/val Albumentations pipelines.

Heavy photometric + light geometric — cards have canonical orientation after
CV crop, and hue shifts must stay small because color is a label.
"""

from __future__ import annotations

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _resize(img_size: int) -> A.BasicTransform:
    # Aspect-ratio-preserving: longest side -> img_size, then pad to square.
    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
            ),
        ]
    )


def train_transform(img_size: int = 128) -> A.Compose:
    return A.Compose(
        [
            _resize(img_size),
            # Geometric (mild)
            A.Rotate(limit=15, p=0.7, border_mode=cv2.BORDER_CONSTANT),
            A.Perspective(scale=(0.02, 0.08), p=0.5),
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), p=0.5),
            # Photometric — where most of the value lives
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.5),
            A.CLAHE(p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            # Sensor / capture noise
            A.ISONoise(p=0.3),
            A.GaussNoise(p=0.3),
            A.MotionBlur(blur_limit=5, p=0.2),
            A.Defocus(radius=(1, 3), p=0.1),
            # Occlusion (glare / fingers)
            A.CoarseDropout(p=0.3),
            # Compression
            A.ImageCompression(quality_range=(60, 95), p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def val_transform(img_size: int = 128) -> A.Compose:
    return A.Compose(
        [
            _resize(img_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
