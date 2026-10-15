from pathlib import Path
import json
import sys

import cv2
import numpy as np

# Allow imports from the project root.
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from backend.ingestion.raw_reader import RawProfile
from backend.pipeline.processor import process_raw


DATASET_DIR = (
    BASE_DIR
    / "data"
    / "demo"
    / "real"
    / "NLM-MontgomeryCXRSet"
    / "MontgomerySet"
    / "CXR_png"
)

VALIDATION_DIR = (
    BASE_DIR
    / "data"
    / "demo"
    / "real"
    / "validation"
)

CASES = [
    "MCUCXR_0001_0.png",
    "MCUCXR_0002_0.png",
    "MCUCXR_0104_1.png",
]


def convert_png_to_raw(
    image: np.ndarray,
    output_path: Path,
) -> tuple[int, int]:
    """
    Convert the source grayscale radiograph to controlled
    unsigned 16-bit little-endian RAW.
    """

    if image.ndim != 2:
        raise ValueError(
            f"Expected grayscale image, got shape {image.shape}"
        )

    if image.dtype == np.uint8:
        image_16 = image.astype(np.uint16) * 257

    elif image.dtype == np.uint16:
        image_16 = image

    else:
        raise ValueError(
            f"Unsupported dtype: {image.dtype}"
        )

    image_16 = np.ascontiguousarray(image_16)

    output_path.write_bytes(
        image_16.astype("<u2").tobytes()
    )

    return image_16.shape[1], image_16.shape[0]


def save_preview(
    image: np.ndarray,
    output_path: Path,
) -> None:
    """
    Save normalized float image as an 8-bit PNG preview.
    """

    image = np.asarray(image)

    if image.dtype != np.float32 and image.dtype != np.float64:
        image = image.astype(np.float32)

    image = np.clip(image, 0.0, 1.0)

    preview = np.round(
        image * 255.0
    ).astype(np.uint8)

    if not cv2.imwrite(
        str(output_path),
        preview,
    ):
        raise RuntimeError(
            f"Could not write preview: {output_path}"
        )


def save_comparison(
    original: np.ndarray,
    enhanced: np.ndarray,
    output_path: Path,
) -> None:
    """
    Save a side-by-side original/enhanced comparison.
    """

    def to_uint8(image: np.ndarray) -> np.ndarray:
        image = np.asarray(image).astype(np.float32)
        image = np.clip(image, 0.0, 1.0)
        return np.round(image * 255.0).astype(np.uint8)

    original_8 = to_uint8(original)
    enhanced_8 = to_uint8(enhanced)

    comparison = np.hstack(
        [original_8, enhanced_8]
    )

    if not cv2.imwrite(
        str(output_path),
        comparison,
    ):
        raise RuntimeError(
            f"Could not write comparison: {output_path}"
        )


def main() -> None:
    print("=" * 72)
    print("REAL RADIOGRAPH — 3 CASE VALIDATION")
    print("=" * 72)

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = []

    for index, filename in enumerate(CASES, start=1):
        print()
        print("-" * 72)
        print(f"CASE {index}/{len(CASES)}: {filename}")
        print("-" * 72)

        source_path = DATASET_DIR / filename

        if not source_path.exists():
            raise FileNotFoundError(
                f"Source X-ray not found:\n{source_path}"
            )

        case_name = Path(filename).stem

        case_dir = (
            VALIDATION_DIR
            / case_name
        )

        case_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ------------------------------------------------------------
        # Read source X-ray
        # ------------------------------------------------------------

        source_image = cv2.imread(
            str(source_path),
            cv2.IMREAD_UNCHANGED,
        )

        if source_image is None:
            raise RuntimeError(
                f"Could not read:\n{source_path}"
            )

        print(f"Source shape    : {source_image.shape}")
        print(f"Source dtype    : {source_image.dtype}")

        # ------------------------------------------------------------
        # Convert to controlled RAW
        # ------------------------------------------------------------

        raw_path = (
            case_dir
            / f"{case_name}_16bit.raw"
        )

        width, height = convert_png_to_raw(
            source_image,
            raw_path,
        )

        print(f"RAW dimensions  : {width} × {height}")
        print(f"RAW file        : {raw_path.name}")

        # ------------------------------------------------------------
        # Existing RAW profile
        # ------------------------------------------------------------

        profile = RawProfile(
            width=width,
            height=height,
            bit_depth=16,
            signed=False,
            endian="little",
        )

        # ------------------------------------------------------------
        # Existing complete pipeline
        # ------------------------------------------------------------

        result = process_raw(
            path=str(raw_path),
            profile=profile,
            output_dir=str(case_dir),
        )

        # ------------------------------------------------------------
        # Save previews
        # ------------------------------------------------------------

        original_preview = (
            case_dir
            / "original.png"
        )

        enhanced_preview = (
            case_dir
            / "enhanced.png"
        )

        comparison_path = (
            case_dir
            / "comparison.png"
        )

        save_preview(
            result["original"],
            original_preview,
        )

        save_preview(
            result["enhanced"],
            enhanced_preview,
        )

        save_comparison(
            result["original"],
            result["enhanced"],
            comparison_path,
        )

        # ------------------------------------------------------------
        # Metrics
        # ------------------------------------------------------------

        before = result["before"]
        after = result["after"]

        metrics = {
            "case": case_name,
            "source_file": str(source_path),
            "width": width,
            "height": height,
            "anatomy": result["anatomy"],
            "confidence": result["confidence"],
            "view": result["view"],
            "model": result["model"],
            "before": before,
            "after": after,
            "improvement": {
                "snr_change_db": round(
                    after["snr_db"]
                    - before["snr_db"],
                    4,
                ),
                "noise_change": round(
                    after["noise"]
                    - before["noise"],
                    6,
                ),
            },
            "parameters": result["parameters"],
            "dicom_path": result["dicom_path"],
        }

        metrics_path = (
            case_dir
            / "metrics.json"
        )

        metrics_path.write_text(
            json.dumps(
                metrics,
                indent=2,
            ),
            encoding="utf-8",
        )

        summary.append(metrics)

        print()
        print("ANATOMY")
        print(
            f"Detected        : {result['anatomy']}"
        )
        print(
            f"Confidence      : {result['confidence']}"
        )
        print(
            f"View            : {result['view']}"
        )

        print()
        print("QUALITY")
        print(
            f"SNR             : "
            f"{before['snr_db']:.2f} → "
            f"{after['snr_db']:.2f} dB"
        )
        print(
            f"Noise           : "
            f"{before['noise']:.6f} → "
            f"{after['noise']:.6f}"
        )
        print(
            f"Contrast        : "
            f"{before['contrast']:.6f} → "
            f"{after['contrast']:.6f}"
        )
        print(
            f"Entropy         : "
            f"{before['entropy']:.4f} → "
            f"{after['entropy']:.4f}"
        )

        print()
        print(
            f"SNR change      : "
            f"{after['snr_db'] - before['snr_db']:+.2f} dB"
        )

        print()
        print(
            f"Comparison      : {comparison_path.name}"
        )
        print(
            f"DICOM           : "
            f"{Path(result['dicom_path']).name}"
        )

    # ------------------------------------------------------------
    # Save overall summary
    # ------------------------------------------------------------

    summary_path = (
        VALIDATION_DIR
        / "validation_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("VALIDATION COMPLETE")
    print("=" * 72)
    print(
        f"Cases processed : {len(summary)}"
    )
    print(
        f"Results folder  : {VALIDATION_DIR}"
    )
    print(
        f"Summary         : {summary_path}"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()