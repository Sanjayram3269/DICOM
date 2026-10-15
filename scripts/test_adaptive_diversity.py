from pathlib import Path
import cv2
import numpy as np

import sys
from pathlib import Path

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[1]
    )
)

from backend.analysis.quality import analyze_quality
from backend.enhancement.adaptive_engine import select_parameters


DATASET_DIR = Path(
    "data/demo/real/NLM-MontgomeryCXRSet/MontgomerySet/CXR_png"
)


def main():
    images = sorted(
        DATASET_DIR.glob("*.png")
    )

    print(f"Images found: {len(images)}")
    print()

    rows = []

    for path in images:
        image = cv2.imread(
            str(path),
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None:
            print(f"SKIP: {path.name}")
            continue

        image = (
            image.astype(np.float32)
            / 255.0
        )

        quality = analyze_quality(
            image
        )

        params = select_parameters(
            anatomy="CHEST",
            quality=quality,
        )

        rows.append(
            (
                path.name,
                quality["snr_db"],
                quality["local_contrast"],
                quality["laplacian_variance"],
                quality["low_clip_fraction"],
                params.bilateral_sigma_color,
                params.clahe_clip_limit,
                params.gamma,
                params.sharpen_amount,
            )
        )

    print(
        "Image | SNR | LocalContrast | Sharpness | "
        "LowClip | Bilateral | CLAHE | Gamma | Sharpen"
    )

    print("-" * 130)

    for row in rows[:20]:
        print(
            f"{row[0]:<18} "
            f"{row[1]:>6.2f} "
            f"{row[2]:>14.5f} "
            f"{row[3]:>10.6f} "
            f"{row[4]:>8.3f} "
            f"{row[5]:>9.3f} "
            f"{row[6]:>7.3f} "
            f"{row[7]:>6.3f} "
            f"{row[8]:>7.3f}"
        )

    print()
    print("PARAMETER DIVERSITY")
    print("-" * 50)

    parameter_names = [
        "bilateral_sigma_color",
        "clahe_clip_limit",
        "gamma",
        "sharpen_amount",
    ]

    parameter_values = [
        [row[5] for row in rows],
        [row[6] for row in rows],
        [row[7] for row in rows],
        [row[8] for row in rows],
    ]

    for name, values in zip(
        parameter_names,
        parameter_values,
    ):
        print(
            f"{name}: "
            f"min={min(values):.3f}, "
            f"max={max(values):.3f}, "
            f"unique={len(set(values))}"
        )

    print()
    print(
        f"Successfully analyzed: {len(rows)}"
    )


if __name__ == "__main__":
    main()