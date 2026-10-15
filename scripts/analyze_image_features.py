from pathlib import Path
import csv
import json
import sys

import cv2
import numpy as np


# ---------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    BASE_DIR
    / "data"
    / "demo"
    / "real"
    / "NLM-MontgomeryCXRSet"
    / "MontgomerySet"
    / "CXR_png"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "demo"
    / "real"
    / "batch_validation"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "image_quality_features.csv"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "image_quality_features.json"
)


# ---------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------

def normalize_image(image):
    image = image.astype(np.float32)

    minimum = float(np.min(image))
    maximum = float(np.max(image))

    if maximum <= minimum:
        return np.zeros_like(image)

    return (
        (image - minimum)
        / (maximum - minimum)
    )


def calculate_features(image):
    normalized = normalize_image(image)

    # -------------------------------------------------------------
    # Histogram statistics
    # -------------------------------------------------------------

    mean_intensity = float(
        np.mean(normalized)
    )

    median_intensity = float(
        np.median(normalized)
    )

    std_intensity = float(
        np.std(normalized)
    )

    p01 = float(
        np.percentile(normalized, 1)
    )

    p05 = float(
        np.percentile(normalized, 5)
    )

    p25 = float(
        np.percentile(normalized, 25)
    )

    p75 = float(
        np.percentile(normalized, 75)
    )

    p95 = float(
        np.percentile(normalized, 95)
    )

    p99 = float(
        np.percentile(normalized, 99)
    )

    percentile_dynamic_range = (
        p99 - p01
    )

    interquartile_range = (
        p75 - p25
    )

    # -------------------------------------------------------------
    # Clipping / saturation
    # -------------------------------------------------------------

    low_clip_fraction = float(
        np.mean(normalized <= 0.01)
    )

    high_clip_fraction = float(
        np.mean(normalized >= 0.99)
    )

    # -------------------------------------------------------------
    # Gradient / edge information
    # -------------------------------------------------------------

    sobel_x = cv2.Sobel(
        normalized,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    sobel_y = cv2.Sobel(
        normalized,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    gradient_magnitude = np.sqrt(
        sobel_x ** 2
        + sobel_y ** 2
    )

    mean_gradient = float(
        np.mean(gradient_magnitude)
    )

    median_gradient = float(
        np.median(gradient_magnitude)
    )

    p90_gradient = float(
        np.percentile(
            gradient_magnitude,
            90,
        )
    )

    strong_edge_fraction = float(
        np.mean(
            gradient_magnitude
            > np.percentile(
                gradient_magnitude,
                90,
            )
        )
    )

    # -------------------------------------------------------------
    # Laplacian sharpness proxy
    # -------------------------------------------------------------

    laplacian = cv2.Laplacian(
        normalized,
        cv2.CV_32F,
    )

    laplacian_variance = float(
        np.var(laplacian)
    )

    # -------------------------------------------------------------
    # Local contrast
    # -------------------------------------------------------------

    local_mean = cv2.GaussianBlur(
        normalized,
        (0, 0),
        sigmaX=7,
    )

    local_variance = cv2.GaussianBlur(
        (
            normalized
            - local_mean
        ) ** 2,
        (0, 0),
        sigmaX=7,
    )

    local_contrast = float(
        np.mean(
            np.sqrt(
                np.maximum(
                    local_variance,
                    0,
                )
            )
        )
    )

    # -------------------------------------------------------------
    # Histogram entropy
    # -------------------------------------------------------------

    histogram, _ = np.histogram(
        normalized,
        bins=256,
        range=(0.0, 1.0),
        density=False,
    )

    probabilities = (
        histogram.astype(np.float64)
        / histogram.sum()
    )

    probabilities = probabilities[
        probabilities > 0
    ]

    entropy = float(
        -np.sum(
            probabilities
            * np.log2(probabilities)
        )
    )

    # -------------------------------------------------------------
    # Return feature set
    # -------------------------------------------------------------

    return {
        "mean_intensity": mean_intensity,
        "median_intensity": median_intensity,
        "std_intensity": std_intensity,

        "p01": p01,
        "p05": p05,
        "p25": p25,
        "p75": p75,
        "p95": p95,
        "p99": p99,

        "percentile_dynamic_range":
            percentile_dynamic_range,

        "interquartile_range":
            interquartile_range,

        "low_clip_fraction":
            low_clip_fraction,

        "high_clip_fraction":
            high_clip_fraction,

        "mean_gradient":
            mean_gradient,

        "median_gradient":
            median_gradient,

        "p90_gradient":
            p90_gradient,

        "strong_edge_fraction":
            strong_edge_fraction,

        "laplacian_variance":
            laplacian_variance,

        "local_contrast":
            local_contrast,

        "entropy":
            entropy,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print("=" * 80)
    print("MONTGOMERY DATASET — IMAGE QUALITY FEATURE ANALYSIS")
    print("=" * 80)

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found:\n{DATASET_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_files = sorted(
        DATASET_DIR.glob("*.png")
    )

    print()
    print(
        f"Radiographs found : {len(image_files)}"
    )

    if not image_files:
        raise RuntimeError(
            "No PNG radiographs found."
        )

    results = []

    for index, image_path in enumerate(
        image_files,
        start=1,
    ):

        print(
            f"[{index:03d}/{len(image_files):03d}] "
            f"{image_path.stem}",
            flush=True,
        )

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_UNCHANGED,
        )

        if image is None:
            print("  FAILED: could not read")
            continue

        if image.ndim != 2:
            print(
                "  FAILED: not grayscale"
            )
            continue

        features = calculate_features(
            image
        )

        record = {
            "case": image_path.stem,
            "filename": image_path.name,
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "dtype": str(image.dtype),
            **features,
        }

        results.append(record)

    # -----------------------------------------------------------------
    # Save CSV
    # -----------------------------------------------------------------

    if not results:
        raise RuntimeError(
            "No images were successfully analyzed."
        )

    fieldnames = list(
        results[0].keys()
    )

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(results)

    # -----------------------------------------------------------------
    # Calculate distributions
    # -----------------------------------------------------------------

    feature_names = [
        key
        for key in results[0].keys()
        if key not in {
            "case",
            "filename",
            "width",
            "height",
            "dtype",
        }
    ]

    distributions = {}

    for feature in feature_names:

        values = np.array(
            [
                float(row[feature])
                for row in results
            ],
            dtype=np.float64,
        )

        distributions[feature] = {
            "mean": float(
                np.mean(values)
            ),
            "median": float(
                np.median(values)
            ),
            "std": float(
                np.std(values)
            ),
            "min": float(
                np.min(values)
            ),
            "p10": float(
                np.percentile(
                    values,
                    10,
                )
            ),
            "p90": float(
                np.percentile(
                    values,
                    90,
                )
            ),
            "max": float(
                np.max(values)
            ),
        }

    # -----------------------------------------------------------------
    # Correlation matrix
    # -----------------------------------------------------------------

    matrix = np.array(
        [
            [
                float(row[feature])
                for feature in feature_names
            ]
            for row in results
        ],
        dtype=np.float64,
    )

    correlation = np.corrcoef(
        matrix,
        rowvar=False,
    )

    correlation_dict = {}

    for i, feature_a in enumerate(
        feature_names
    ):

        correlation_dict[feature_a] = {}

        for j, feature_b in enumerate(
            feature_names
        ):

            correlation_dict[
                feature_a
            ][feature_b] = float(
                correlation[i, j]
            )

    # -----------------------------------------------------------------
    # Feature ranges
    # -----------------------------------------------------------------

    feature_ranges = {}

    for feature in feature_names:

        values = [
            float(row[feature])
            for row in results
        ]

        feature_ranges[feature] = (
            max(values) - min(values)
        )

    # -----------------------------------------------------------------
    # Structured JSON
    # -----------------------------------------------------------------

    output = {
        "dataset": "NLM-MontgomeryCXRSet",
        "total_cases": len(results),
        "features": feature_names,
        "distributions": distributions,
        "feature_ranges": feature_ranges,
        "correlation": correlation_dict,
        "results": results,
    }

    OUTPUT_JSON.write_text(
        json.dumps(
            output,
            indent=2,
        ),
        encoding="utf-8",
    )

    # -----------------------------------------------------------------
    # Print useful summary
    # -----------------------------------------------------------------

    print()
    print("=" * 80)
    print("FEATURE ANALYSIS COMPLETE")
    print("=" * 80)

    print(
        f"Successful cases : {len(results)}"
    )

    print()
    print(
        "FEATURE DISTRIBUTIONS"
    )
    print("-" * 80)

    for feature in feature_names:

        stats = distributions[
            feature
        ]

        print(
            f"{feature:30s} "
            f"mean={stats['mean']:.6f} "
            f"std={stats['std']:.6f} "
            f"range="
            f"{stats['min']:.6f}"
            f"→"
            f"{stats['max']:.6f}"
        )

    print()
    print(
        "OUTPUT"
    )
    print("-" * 80)

    print(
        f"CSV  : {OUTPUT_CSV}"
    )

    print(
        f"JSON : {OUTPUT_JSON}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()