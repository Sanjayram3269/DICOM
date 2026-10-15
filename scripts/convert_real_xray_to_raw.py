from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    BASE_DIR
    / "data"
    / "demo"
    / "real"
    / "MCUCXR_0001_0.png"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "demo"
    / "real"
    / "MCUCXR_0001_0_16bit.raw"
)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input X-ray not found:\n{INPUT_PATH}"
        )

    print("=" * 60)
    print("REAL CHEST X-RAY → CONTROLLED 16-BIT RAW")
    print("=" * 60)

    # Read without altering the original bit depth.
    image = cv2.imread(
        str(INPUT_PATH),
        cv2.IMREAD_UNCHANGED,
    )

    if image is None:
        raise RuntimeError(
            f"Could not read image:\n{INPUT_PATH}"
        )

    # We require a single-channel grayscale radiograph.
    if image.ndim != 2:
        raise ValueError(
            f"Expected grayscale X-ray, got shape {image.shape}"
        )

    print(f"Input file      : {INPUT_PATH.name}")
    print(f"Input shape     : {image.shape}")
    print(f"Input dtype     : {image.dtype}")
    print(f"Input min       : {image.min()}")
    print(f"Input max       : {image.max()}")

    # Convert to unsigned 16-bit while preserving the original
    # grayscale values.
    if image.dtype == np.uint16:
        image_16 = image

        # If this is a genuine 12-bit image stored inside uint16,
        # expand 0..4095 -> 0..65520 using an exact bit shift.
        if int(image.max()) <= 4095:
            image_16 = image << 4
            conversion = "12-bit values preserved, shifted left 4 bits"

        else:
            conversion = "already 16-bit, values preserved"

    elif image.dtype == np.uint8:
        # Fallback for an 8-bit source.
        # Expand 0..255 -> 0..65535 exactly.
        image_16 = image.astype(np.uint16) * 257
        conversion = "8-bit values expanded to 16-bit"

    else:
        raise ValueError(
            f"Unsupported input dtype: {image.dtype}"
        )

    image_16 = np.ascontiguousarray(image_16)

    # Write raw little-endian uint16 values.
    raw_bytes = image_16.astype("<u2").tobytes()
    OUTPUT_PATH.write_bytes(raw_bytes)

    expected_size = (
        image_16.shape[0]
        * image_16.shape[1]
        * np.dtype("<u2").itemsize
    )

    actual_size = OUTPUT_PATH.stat().st_size

    print()
    print("CONVERSION")
    print(f"Method          : {conversion}")
    print(f"Output dtype    : {image_16.dtype}")
    print(f"Output min      : {image_16.min()}")
    print(f"Output max      : {image_16.max()}")
    print(f"Width           : {image_16.shape[1]}")
    print(f"Height          : {image_16.shape[0]}")
    print(f"Endian          : Little-endian")
    print(f"Signed          : No")
    print(f"Bits            : 16")

    print()
    print("RAW FILE")
    print(f"Output          : {OUTPUT_PATH}")
    print(f"Expected bytes  : {expected_size:,}")
    print(f"Actual bytes    : {actual_size:,}")

    if actual_size != expected_size:
        raise RuntimeError(
            "RAW size validation failed!"
        )

    print()
    print("PASS: RAW size validation")
    print("PASS: 16-bit unsigned little-endian RAW created")
    print("=" * 60)


if __name__ == "__main__":
    main()