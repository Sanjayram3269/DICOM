from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RawProfile:
    """
    Describes the binary layout of a detector RAW image.
    """

    width: int
    height: int
    bit_depth: int = 16
    signed: bool = False
    endian: str = "little"

    @property
    def dtype(self) -> np.dtype:
        """
        Return the NumPy dtype corresponding to the RAW format.
        """

        if self.bit_depth <= 8:
            base = "i1" if self.signed else "u1"

        elif self.bit_depth <= 16:
            base = "i2" if self.signed else "u2"

        elif self.bit_depth <= 32:
            base = "i4" if self.signed else "u4"

        else:
            raise ValueError(
                f"Unsupported bit depth: {self.bit_depth}"
            )

        if self.endian not in ("little", "big"):
            raise ValueError(
                "Endianness must be 'little' or 'big'."
            )

        # 8-bit data has no meaningful byte order.
        if self.bit_depth <= 8:
            return np.dtype(base)

        byte_order = "<" if self.endian == "little" else ">"

        return np.dtype(byte_order + base)


def validate_profile(profile: RawProfile) -> None:
    """
    Validate RAW image configuration before reading.
    """

    if profile.width <= 0:
        raise ValueError("RAW width must be greater than zero.")

    if profile.height <= 0:
        raise ValueError("RAW height must be greater than zero.")

    if profile.bit_depth not in (8, 12, 14, 16, 32):
        raise ValueError(
            "Supported bit depths are 8, 12, 14, 16 and 32."
        )

    if profile.endian not in ("little", "big"):
        raise ValueError(
            "Endianness must be 'little' or 'big'."
        )


def expected_file_size(profile: RawProfile) -> int:
    """
    Calculate expected RAW file size in bytes.

    Note:
        12-bit and 14-bit packed detector formats can use
        different storage layouts. This initial implementation
        assumes one 16-bit container per pixel for <=16-bit data.
    """

    validate_profile(profile)

    bytes_per_pixel = 1 if profile.bit_depth <= 8 else 2

    return (
        profile.width
        * profile.height
        * bytes_per_pixel
    )


def read_raw(
    path: str | Path,
    profile: RawProfile,
) -> np.ndarray:
    """
    Read a detector RAW file and return a 2-D NumPy array.
    """

    validate_profile(profile)

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"RAW file does not exist: {path}"
        )

    expected_pixels = (
        profile.width * profile.height
    )

    data = np.fromfile(
        path,
        dtype=profile.dtype,
    )

    if data.size != expected_pixels:
        expected_bytes = expected_file_size(profile)

        raise ValueError(
            "RAW size mismatch.\n"
            f"Expected pixels: {expected_pixels}\n"
            f"Expected bytes: {expected_bytes}\n"
            f"Actual bytes: {path.stat().st_size}\n"
            "Check width, height, bit depth, packing and endianness."
        )

    image = data.reshape(
        profile.height,
        profile.width,
    )

    return image


def normalize_float(
    image: np.ndarray,
) -> np.ndarray:
    """
    Convert an image to float32 in the [0, 1] range.

    Percentile normalization reduces the influence of extreme
    detector values during visualization and research processing.
    """

    if image.size == 0:
        raise ValueError(
            "Cannot normalize an empty image."
        )

    x = image.astype(
        np.float32,
        copy=False,
    )

    low = float(
        np.percentile(x, 0.5)
    )

    high = float(
        np.percentile(x, 99.5)
    )

    # Fallback for nearly constant images.
    if high <= low:
        low = float(np.min(x))
        high = float(np.max(x))

    if high <= low:
        return np.zeros_like(
            x,
            dtype=np.float32,
        )

    normalized = (
        (x - low)
        / (high - low)
    )

    return np.clip(
        normalized,
        0.0,
        1.0,
    ).astype(
        np.float32,
        copy=False,
    )