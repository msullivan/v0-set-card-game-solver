#!/usr/bin/env python3
"""Trace a Set card shape (squiggle / oval / diamond) from a real photo crop.

Pipeline:
  1. Load the crop, threshold for the red shape (high R, low G/B — saturation
     filter excludes brown wood backgrounds).
  2. cv2.findContours on the largest connected component.
  3. Ramer-Douglas-Peucker simplification at user-supplied epsilon.
  4. Smooth into closed quadratic Bezier curves via the midpoint trick (each
     polygon vertex becomes a Q control point; the curve passes through the
     midpoints of consecutive vertices).
  5. Normalize the bounding box to width=100, centered in viewBox 0 0 100 50.

Run via uv so deps are isolated:

  uv run --with numpy --with pillow --with scipy --with opencv-python-headless \\
    scripts/trace-shape.py test-images/crops/103.jpg 3

Args:
  IMAGE   path to a crop image
  EPSILON optional RDP simplification epsilon in pixels (default 4.0).
          Larger = fewer points / smoother. eps=3 keeps wave detail; eps=5-8
          gives a cleaner, simpler trace.

Saves the binary mask to /tmp/squiggle-mask.png and the path to
/tmp/squiggle-traced.txt.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image


def red_mask(img: np.ndarray) -> np.ndarray:
    r, g, b = img[..., 0].astype(int), img[..., 1].astype(int), img[..., 2].astype(int)
    # Red shape: high R, low G, low B, and high saturation (exclude brown wood).
    return ((r > 140) & (r - g > 80) & (r - b > 80)).astype(np.uint8)


def trace_contour(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return the largest-component outer contour as a clockwise list of (x,y)."""
    import cv2
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    pts = largest.squeeze(1)  # shape (N, 2) as (x, y)
    return [(int(x), int(y)) for x, y in pts]


def rdp(points: list[tuple[float, float]], eps: float) -> list[tuple[float, float]]:
    """Ramer-Douglas-Peucker simplification on a closed polyline."""
    if len(points) < 3:
        return points

    def perp_dist(p, a, b):
        if a == b:
            return ((p[0] - a[0]) ** 2 + (p[1] - a[1]) ** 2) ** 0.5
        x0, y0 = p
        x1, y1 = a
        x2, y2 = b
        num = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        den = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5
        return num / den

    def _rdp(pts):
        if len(pts) < 3:
            return pts
        a, b = pts[0], pts[-1]
        idx, dmax = 0, -1
        for i in range(1, len(pts) - 1):
            d = perp_dist(pts[i], a, b)
            if d > dmax:
                dmax = d
                idx = i
        if dmax > eps:
            left = _rdp(pts[: idx + 1])
            right = _rdp(pts[idx:])
            return left[:-1] + right
        return [a, b]

    return _rdp(points)


def midpoint_quadratic(pts: list[tuple[float, float]]) -> str:
    """Closed quadratic-Bezier path: each polygon vertex is a control,
    midpoints between consecutive vertices are on-curve points.
    Produces a smooth closed shape with no spike artifacts at the seam.
    """
    n = len(pts)
    def mid(a, b):
        return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    m0 = mid(pts[0], pts[1])
    out = [f"M {m0[0]:.2f},{m0[1]:.2f}"]
    for i in range(n):
        ctrl = pts[(i + 1) % n]
        end = mid(pts[(i + 1) % n], pts[(i + 2) % n])
        out.append(f"Q {ctrl[0]:.2f},{ctrl[1]:.2f} {end[0]:.2f},{end[1]:.2f}")
    out.append("Z")
    return " ".join(out)

# Backward-compat name used by main()
catmull_to_bezier = midpoint_quadratic


def main():
    src = Path(sys.argv[1])
    img = np.array(Image.open(src).convert("RGB"))
    mask = red_mask(img)

    # Save mask for inspection
    Image.fromarray(mask * 255).save("/tmp/squiggle-mask.png")

    contour = trace_contour(mask)
    print(f"Contour: {len(contour)} points")
    if not contour:
        sys.exit("no contour found")

    # Convert to floats
    pts = [(float(x), float(y)) for x, y in contour]

    # Simplify
    target_eps = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
    simp = rdp(pts, target_eps)
    if simp and simp[-1] == simp[0]:
        simp = simp[:-1]
    eps = target_eps
    print(f"Chosen eps={eps}, {len(simp)} points")

    # Normalize to viewBox 0 0 100 50.
    xs = [p[0] for p in simp]
    ys = [p[1] for p in simp]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    bw = maxx - minx
    bh = maxy - miny
    # Preserve aspect ratio: scale so width=100. Center vertically.
    sx = 100.0 / bw
    sy = sx  # same scale to preserve aspect
    new_h = bh * sy
    norm = []
    for x, y in simp:
        nx = (x - minx) * sx
        ny = (y - miny) * sy + (50 - new_h) / 2
        norm.append((nx, ny))

    print(f"normalized aspect: width=100, height={new_h:.1f}")

    path = catmull_to_bezier(norm)
    print()
    print(path)
    print()
    Path("/tmp/squiggle-traced.txt").write_text(path)


if __name__ == "__main__":
    main()
