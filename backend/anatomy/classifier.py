from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import cv2
import numpy as np


ANATOMY_LABELS = ("CHEST", "ABDOMEN", "EXTREMITY", "PELVIS", "SKULL", "UNKNOWN")
VIEW_LABELS = ("PA", "AP", "LATERAL", "UNKNOWN")


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
        raise ValueError(f"Expected a 2D grayscale image, got shape={image.shape}")
    image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
    image_min = float(np.min(image))
    image_max = float(np.max(image))
    if image_max <= image_min:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - image_min) / (image_max - image_min), 0.0, 1.0).astype(np.float32)


def extract_features(image: np.ndarray) -> dict[str, float]:
    """Extract image-level diagnostic features; these are not a clinical model."""
    image = _normalize_image(image)
    height, width = image.shape
    center_y, center_x = height // 2, width // 2
    y0, y1 = int(height * 0.25), int(height * 0.75)
    x0, x1 = int(width * 0.25), int(width * 0.75)
    central_region = image[y0:y1, x0:x1]
    peripheral_mask = np.ones_like(image, dtype=bool)
    peripheral_mask[y0:y1, x0:x1] = False
    peripheral_region = image[peripheral_mask]
    vertical_profile = np.mean(image, axis=1)
    horizontal_profile = np.mean(image, axis=0)
    laplacian = cv2.Laplacian(image, cv2.CV_32F)
    return {
        "height": float(height),
        "width": float(width),
        "aspect_ratio": float(width / max(height, 1)),
        "mean_intensity": float(np.mean(image)),
        "std_intensity": float(np.std(image)),
        "central_mean": float(np.mean(central_region)),
        "peripheral_mean": float(np.mean(peripheral_region)),
        "central_peripheral_difference": float(np.mean(central_region) - np.mean(peripheral_region)),
        "vertical_variation": float(np.std(vertical_profile)),
        "horizontal_variation": float(np.std(horizontal_profile)),
        "edge_strength": float(np.mean(np.abs(laplacian))),
        "top_half_mean": float(np.mean(image[:center_y, :])),
        "bottom_half_mean": float(np.mean(image[center_y:, :])),
        "left_half_mean": float(np.mean(image[:, :center_x])),
        "right_half_mean": float(np.mean(image[:, center_x:])),
    }


def classify(image: np.ndarray) -> AnatomyResult:
    """Conservative baseline; not a trained or clinically validated classifier."""
    features = extract_features(image)
    return AnatomyResult(
        anatomy="CHEST",
        confidence=0.50,
        view="UNKNOWN",
        view_confidence=0.0,
        model="anatomy-view-baseline-v1",
        method="feature-baseline",
        features=features,
    )
