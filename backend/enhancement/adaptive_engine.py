from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np


@dataclass(frozen=True)
class EnhancementParameters:
    bilateral_diameter: int = 15
    clahe_clip_limit: float = 1.8
    gamma: float = 0.94
    sharpen_strength: float = 0.18

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def select_parameters(quality: dict[str, float]) -> EnhancementParameters:
    noise = float(quality.get("noise", 0.0))
    contrast = float(quality.get("contrast", 0.0))
    diameter = 17 if noise > 0.02 else 13
    clip = 2.0 if contrast < 0.20 else 1.6
    return EnhancementParameters(
        bilateral_diameter=diameter,
        clahe_clip_limit=clip,
        gamma=0.94,
        sharpen_strength=0.18,
    )


def enhance(image: np.ndarray, parameters: EnhancementParameters) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    image = np.clip(image, 0.0, 1.0)
    denoised = cv2.bilateralFilter(
        image,
        d=int(parameters.bilateral_diameter),
        sigmaColor=0.08,
        sigmaSpace=7.0,
    )
    uint8 = np.uint8(np.clip(denoised * 255.0, 0, 255))
    clahe = cv2.createCLAHE(clipLimit=float(parameters.clahe_clip_limit), tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(uint8).astype(np.float32) / 255.0
    gamma = max(float(parameters.gamma), 1e-4)
    gamma_corrected = np.power(np.clip(contrast_enhanced, 0.0, 1.0), gamma)
    blur = cv2.GaussianBlur(gamma_corrected, (0, 0), sigmaX=1.0)
    sharpened = gamma_corrected + float(parameters.sharpen_strength) * (gamma_corrected - blur)
    return np.clip(sharpened, 0.0, 1.0).astype(np.float32)


def generate_candidates(baseline: EnhancementParameters) -> list[EnhancementParameters]:
    candidates = [baseline]
    specs = [
        ("bilateral_diameter", -3),
        ("bilateral_diameter", 3),
        ("clahe_clip_limit", -0.25),
        ("clahe_clip_limit", 0.25),
        ("gamma", -0.025),
        ("gamma", 0.025),
        ("sharpen_strength", -0.05),
        ("sharpen_strength", 0.05),
    ]
    for field, offset in specs:
        values = baseline.to_dict()
        values[field] += offset
        values["bilateral_diameter"] = int(np.clip(values["bilateral_diameter"], 8, 26))
        values["clahe_clip_limit"] = float(np.clip(values["clahe_clip_limit"], 1.2, 2.6))
        values["gamma"] = float(np.clip(values["gamma"], 0.88, 1.0))
        values["sharpen_strength"] = float(np.clip(values["sharpen_strength"], 0.06, 0.30))
        candidate = EnhancementParameters(**values)
        if candidate not in candidates:
            candidates.append(candidate)
    for changes in (
        {"bilateral_diameter": 2, "clahe_clip_limit": 0.15},
        {"clahe_clip_limit": 0.15, "sharpen_strength": 0.03},
    ):
        values = baseline.to_dict()
        for field, offset in changes.items():
            values[field] += offset
        values["bilateral_diameter"] = int(np.clip(values["bilateral_diameter"], 8, 26))
        values["clahe_clip_limit"] = float(np.clip(values["clahe_clip_limit"], 1.2, 2.6))
        values["gamma"] = float(np.clip(values["gamma"], 0.88, 1.0))
        values["sharpen_strength"] = float(np.clip(values["sharpen_strength"], 0.06, 0.30))
        candidate = EnhancementParameters(**values)
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates
