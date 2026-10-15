from pathlib import Path

import cv2
import numpy as np


WIDTH = 1024
HEIGHT = 1024


def create_radiograph_fixture() -> np.ndarray:
    """
    Create a controlled synthetic radiograph-like image.

    This is NOT a medical image and must never be interpreted
    as a real diagnostic radiograph.

    It exists solely to test the RAW ingestion and processing
    pipeline before real public radiograph data is introduced.
    """

    y, x = np.mgrid[0:HEIGHT, 0:WIDTH]

    # ---------------------------------------------------------
    # Background
    # ---------------------------------------------------------

    image = np.full(
        (HEIGHT, WIDTH),
        0.035,
        dtype=np.float32,
    )

    # ---------------------------------------------------------
    # Chest field
    # ---------------------------------------------------------

    cx = WIDTH / 2
    cy = HEIGHT * 0.53

    chest_width = WIDTH * 0.70
    chest_height = HEIGHT * 0.72

    chest = (
        ((x - cx) / (chest_width / 2)) ** 2
        + ((y - cy) / (chest_height / 2)) ** 2
        <= 1
    )

    image[chest] += 0.18

    # ---------------------------------------------------------
    # Left / right lung fields
    # ---------------------------------------------------------

    left_lung = (
        ((x - WIDTH * 0.36) / (WIDTH * 0.19)) ** 2
        + ((y - HEIGHT * 0.52) / (HEIGHT * 0.31)) ** 2
        <= 1
    )

    right_lung = (
        ((x - WIDTH * 0.64) / (WIDTH * 0.19)) ** 2
        + ((y - HEIGHT * 0.52) / (HEIGHT * 0.31)) ** 2
        <= 1
    )

    image[left_lung] += 0.20
    image[right_lung] += 0.20

    # ---------------------------------------------------------
    # Spine
    # ---------------------------------------------------------

    spine = (
        np.abs(x - cx) < WIDTH * 0.025
    ) & (
        y > HEIGHT * 0.25
    ) & (
        y < HEIGHT * 0.86
    )

    image[spine] += 0.18

    # ---------------------------------------------------------
    # Heart region
    # ---------------------------------------------------------

    heart = (
        ((x - WIDTH * 0.56) / (WIDTH * 0.14)) ** 2
        + ((y - HEIGHT * 0.60) / (HEIGHT * 0.20)) ** 2
        <= 1
    )

    image[heart] += 0.12

    # ---------------------------------------------------------
    # Diaphragm
    # ---------------------------------------------------------

    diaphragm = (
        np.exp(
            -(
                (
                    y
                    - (
                        HEIGHT * 0.76
                        + 35
                        * np.sin(
                            (x / WIDTH)
                            * np.pi
                        )
                    )
                )
                ** 2
            )
            / (2 * 25**2)
        )
    )

    image += diaphragm * 0.10

    # ---------------------------------------------------------
    # Rib-like structures
    # ---------------------------------------------------------

    for i in range(7):

        yy = (
            HEIGHT * 0.30
            + i * HEIGHT * 0.065
        )

        curve = (
            yy
            + 45
            * np.sin(
                ((x - cx) / WIDTH)
                * np.pi
            )
        )

        rib = np.exp(
            -(
                (y - curve) ** 2
            )
            / (2 * 3.0**2)
        )

        rib *= (
            ((x - cx) / (WIDTH * 0.40)) ** 2
            < 1
        )

        image += rib * 0.055

    # ---------------------------------------------------------
    # Mild detector noise
    # ---------------------------------------------------------

    rng = np.random.default_rng(20261006)

    gaussian_noise = rng.normal(
        0,
        0.025,
        image.shape,
    ).astype(np.float32)

    image += gaussian_noise

    # ---------------------------------------------------------
    # Mild low-frequency variation
    # ---------------------------------------------------------

    low_frequency = cv2.GaussianBlur(
        rng.normal(
            0,
            1,
            image.shape,
        ).astype(np.float32),
        (0, 0),
        30,
    )

    low_frequency /= (
        np.std(low_frequency) + 1e-8
    )

    image += (
        low_frequency
        * 0.008
    )

    # ---------------------------------------------------------
    # Clamp
    # ---------------------------------------------------------

    image = np.clip(
        image,
        0,
        1,
    )

    return image


def main():

    image = create_radiograph_fixture()

    # Convert to a 16-bit detector-like signal.
    raw16 = np.round(
        image * 65535
    ).astype(
        "<u2"
    )

    output_dir = Path(
        "data/demo"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_path = (
        output_dir
        / "synthetic_chest_1024x1024.raw"
    )

    preview_path = (
        output_dir
        / "synthetic_chest_preview.png"
    )

    raw16.tofile(
        raw_path
    )

    preview = np.round(
        image * 255
    ).astype(
        np.uint8
    )

    cv2.imwrite(
        str(preview_path),
        preview,
    )

    print()
    print("======================================")
    print("Radiograph RAW fixture created")
    print("======================================")
    print(f"Width:       {WIDTH}")
    print(f"Height:      {HEIGHT}")
    print("Bit depth:   16")
    print("Signed:      False")
    print("Endianness:  little")
    print()
    print(f"RAW:     {raw_path}")
    print(f"Preview: {preview_path}")
    print()
    print(
        f"RAW size: {raw_path.stat().st_size:,} bytes"
    )
    print("======================================")


if __name__ == "__main__":
    main()