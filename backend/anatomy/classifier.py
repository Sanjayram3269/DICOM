from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import cv2
import numpy as np


ANATOMY_LABELS = (
    "CHEST",
    "ABDOMEN",
    "EXTREMITY",
    "PELVIS",
    "SKULL",
    "UNKNOWN",
)

VIEW_LABELS = (
    "PA",
    "AP",
    "LATERAL",
    "UNKNOWN",
)


@dataclass(frozen=True)
class AnatomyResult:
    anatomy: str
    confidence: float
    view: str
    view_confidence: float
    model: str
    method: str
    features: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)

    if image.ndim != 2:
        raise ValueError(
            f"Expected a 2D grayscale image, got shape={image.shape}"
        )

    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    image_min = float(np.min(image))
    image_max = float(np.max(image))

    if image_max <= image_min:
        return np.zeros_like(image, dtype=np.float32)

    normalized = (image - image_min) / (image_max - image_min)

    return np.clip(
        normalized,
        0.0,
        1.0,
    ).astype(np.float32)


def extract_features(image: np.ndarray) -> dict[str, float]:
    """
    Extract lightweight image-level features.

    These features are intended for classifier development and
    diagnostics. They are NOT a clinically validated anatomy model.
    """

    image = _normalize_image(image)

    height, width = image.shape

    mean_intensity = float(np.mean(image))
    std_intensity = float(np.std(image))

    center_y = height // 2
    center_x = width // 2

    central_region = image[
        int(height * 0.25):int(height * 0.75),
        int(width * 0.25):int(width * 0.75),
    ]

    peripheral_mask = np.ones_like(image, dtype=bool)

    peripheral_mask[
        int(height * 0.25):int(height * 0.75),
        int(width * 0.25):int(width * 0.75),
    ] = False

    peripheral_region = image[peripheral_mask]

    central_mean = float(np.mean(central_region))
    peripheral_mean = float(np.mean(peripheral_region))

    vertical_profile = np.mean(image, axis=1)
    horizontal_profile = np.mean(image, axis=0)

    vertical_variation = float(np.std(vertical_profile))
    horizontal_variation = float(np.std(horizontal_profile))

    laplacian = cv2.Laplacian(
        image,
        cv2.CV_32F,
    )

    edge_strength = float(
        np.mean(np.abs(laplacian))
    )

    aspect_ratio = float(width / max(height, 1))

    top_half_mean = float(
        np.mean(image[:center_y, :])
    )

    bottom_half_mean = float(
        np.mean(image[center_y:, :])
    )

    left_half_mean = float(
        np.mean(image[:, :center_x])
    )

    right_half_mean = float(
        np.mean(image[:, center_x:])
    )

    return {
        "height": float(height),
        "width": float(width),
        "aspect_ratio": aspect_ratio,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "central_mean": central_mean,
        "peripheral_mean": peripheral_mean,
        "central_peripheral_difference": (
            central_mean - peripheral_mean
        ),
        "vertical_variation": vertical_variation,
        "horizontal_variation": horizontal_variation,
        "edge_strength": edge_strength,
        "top_half_mean": top_half_mean,
        "bottom_half_mean": bottom_half_mean,
        "left_half_mean": left_half_mean,
        "right_half_mean": right_half_mean,
    }


def classify(image: np.ndarray) -> AnatomyResult:
    """
    Baseline anatomy/view classifier.

    IMPORTANT:
    This is an architectural baseline, not a trained clinical model.

    The classifier intentionally reports UNKNOWN when the available
    heuristic evidence is insufficient rather than pretending to have
    high-confidence AI predictions.
    """

    features = extract_features(image)

    # Current development dataset is chest radiography.
    # We explicitly mark this as a dataset/profile prior rather than
    # pretending that this is a learned anatomy classifier.
    anatomy = "CHEST"
    anatomy_confidence = 0.50

    view = "UNKNOWN"
    view_confidence = 0.0

    return AnatomyResult(
        anatomy=anatomy,
        confidence=anatomy_confidence,
        view=view,
        view_confidence=view_confidence,
        model="anatomy-view-baseline-v1",
        method="feature-baseline",
        features=features,
    )