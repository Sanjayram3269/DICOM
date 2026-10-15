from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RawProfile:
    width: int
    height: int
    bit_depth: int = 16
    signed: bool = False
    endian: str = "little"

    @property
    def dtype(self) -> np.dtype:
        if not 1 <= self.bit_depth <= 16:
            raise ValueError(f"Unsupported bit depth: {self.bit_depth}")
        if self.bit_depth <= 8:
            return np.dtype("i1" if self.signed else "u1")
        prefix = "<" if self.endian == "little" else ">"
        return np.dtype(prefix + ("i2" if self.signed else "u2"))

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Width and height must be positive")
        if self.endian not in {"little", "big"}:
            raise ValueError("endian must be 'little' or 'big'")
        if self.bit_depth not in {8, 10, 12, 14, 16}:
            raise ValueError("Supported unpacked bit depths: 8, 10, 12, 14, 16")

    @property
    def expected_file_size(self) -> int:
        self.validate()
        if self.bit_depth <= 8:
            bytes_per_pixel = 1
        else:
            bytes_per_pixel = 2
        return self.width * self.height * bytes_per_pixel


def read_raw(path: str | Path, profile: RawProfile) -> np.ndarray:
    profile.validate()
    path = Path(path)
    expected = profile.expected_file_size
    actual = path.stat().st_size
    if actual != expected:
        raise ValueError(f"RAW size mismatch: expected {expected} bytes, got {actual}")
    pixels = np.fromfile(path, dtype=profile.dtype)
    return pixels.reshape((profile.height, profile.width))


def normalize_float(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    low = float(np.min(image))
    high = float(np.max(image))
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - low) / (high - low), 0.0, 1.0).astype(np.float32)
