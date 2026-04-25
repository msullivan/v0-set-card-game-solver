"""Generate augmented versions of card crops using the train Albumentations pipeline."""

import argparse
import random
import sys
from pathlib import Path

import albumentations as A
import cv2
import numpy as np

CROPS_DIR = Path(__file__).parent.parent / "test-images" / "crops"
OUTPUT_DIR = Path(__file__).parent.parent / "test-images" / "augmented-crops"


def augment_pipeline() -> A.Compose:
    return A.Compose(
        [
            A.Rotate(limit=15, p=0.7, border_mode=cv2.BORDER_CONSTANT),
            A.Perspective(scale=(0.02, 0.08), p=0.5),
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.5),
            A.CLAHE(p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.ISONoise(p=0.3),
            A.GaussNoise(p=0.3),
            A.MotionBlur(blur_limit=5, p=0.2),
            A.Defocus(radius=(1, 3), p=0.1),
            A.CoarseDropout(p=0.3),
            A.ImageCompression(quality_range=(60, 95), p=0.5),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate augmented card crop images")
    parser.add_argument("--count", type=int, default=300, help="Number of augmented images to generate")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    crops = sorted(CROPS_DIR.glob("*.jpg"))
    if not crops:
        print(f"No crops found in {CROPS_DIR}", file=sys.stderr)
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)
    pipeline = augment_pipeline()
    rng = random.Random(args.seed)

    print(f"Generating {args.count} augmented images from {len(crops)} source crops → {args.output}")

    for i in range(args.count):
        src = rng.choice(crops)
        img = cv2.imread(str(src))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        augmented = pipeline(image=img)["image"]

        out_path = args.output / f"{i + 1:04d}_from_{src.name}"
        cv2.imwrite(str(out_path), cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{args.count}")

    print(f"Done. {args.count} images saved to {args.output}")


if __name__ == "__main__":
    main()
