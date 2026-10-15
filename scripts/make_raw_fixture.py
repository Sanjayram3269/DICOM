from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def make_fixture(width: int, height: int, output: Path) -> None:
    y, x = np.mgrid[0:height, 0:width]
    cx, cy = width / 2.0, height / 2.0
    nx = (x - cx) / max(width / 2.0, 1.0)
    ny = (y - cy) / max(height / 2.0, 1.0)
    background = 0.45 + 0.18 * np.exp(-((nx / 0.75) ** 2 + (ny / 0.85) ** 2))
    left_lung = np.exp(-(((nx + 0.28) / 0.22) ** 2 + (ny / 0.55) ** 2))
    right_lung = np.exp(-(((nx - 0.28) / 0.22) ** 2 + (ny / 0.55) ** 2))
    mediastinum = np.exp(-((nx / 0.15) ** 2 + ((ny + 0.02) / 0.43) ** 2))
    image = background - 0.24 * (left_lung + right_lung) + 0.12 * mediastinum
    rng = np.random.default_rng(20261015)
    image += rng.normal(0.0, 0.015, size=image.shape)
    pixels = np.asarray(np.clip(image, 0.0, 1.0) * 65535.0, dtype="<u2")
    output.parent.mkdir(parents=True, exist_ok=True)
    pixels.tofile(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic 16-bit RAW chest-like fixture.")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--output", type=Path, default=Path("data/demo/synthetic_chest_1024x1024.raw"))
    args = parser.parse_args()
    make_fixture(args.width, args.height, args.output)
    print(f"Wrote {args.width}x{args.height} 16-bit fixture to {args.output}")


if __name__ == "__main__":
    main()
