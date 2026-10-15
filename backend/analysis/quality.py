from __future__ import annotations

import cv2
import numpy as np


def _safe_float(value: float) -> float:
    return float(value)


def estimate_noise(image: np.ndarray) -> float:
    image = image.astype(np.float32, copy=False)
    smoothed = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0, sigmaY=1.0)
    residual = image - smoothed
    median_residual = np.median(residual)
    mad = np.median(np.abs(residual - median_residual))
    return _safe_float(1.4826 * mad)


def estimate_signal(image: np.ndarray) -> float:
    return float(np.mean(np.abs(image)))


def estimate_snr_db(signal: float, noise: float) -> float:
    if noise <= 1e-12:
        return 100.0
    return float(20 * np.log10(max(signal, 1e-12) / noise))


def estimate_contrast(image: np.ndarray) -> float:
    return float(np.std(image))


def estimate_dynamic_range(image: np.ndarray) -> float:
    return float(np.max(image) - np.min(image))


def estimate_entropy(image: np.ndarray, bins: int = 256) -> float:
    values = np.clip(image.astype(np.float32, copy=False), 0.0, 1.0)
    histogram, _ = np.histogram(values, bins=bins, range=(0.0, 1.0))
    probabilities = histogram.astype(np.float64)
    probabilities /= max(float(probabilities.sum()), 1.0)
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log2(probabilities)))


def estimate_local_contrast(image: np.ndarray) -> float:
    image = image.astype(np.float32, copy=False)
    local_mean = cv2.GaussianBlur(image, (0, 0), sigmaX=5.0, sigmaY=5.0)
    local_second_moment = cv2.GaussianBlur(image * image, (0, 0), sigmaX=5.0, sigmaY=5.0)
    local_variance = np.maximum(local_second_moment - local_mean * local_mean, 0.0)
    return float(np.mean(np.sqrt(local_variance)))


def estimate_laplacian_variance(image: np.ndarray) -> float:
    image = image.astype(np.float32, copy=False)
    return float(np.var(cv2.Laplacian(image, cv2.CV_32F)))


def estimate_low_clip_fraction(image: np.ndarray, threshold: float = 0.01) -> float:
    return float(np.mean(image <= threshold))


def estimate_high_clip_fraction(image: np.ndarray, threshold: float = 0.99) -> float:
    return float(np.mean(image >= threshold))


def _gradient_magnitude(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(grad_x, grad_y)


def estimate_mean_gradient(image: np.ndarray) -> float:
    return float(np.mean(_gradient_magnitude(image)))


def estimate_median_gradient(image: np.ndarray) -> float:
    return float(np.median(_gradient_magnitude(image)))


def estimate_p90_gradient(image: np.ndarray) -> float:
    return float(np.percentile(_gradient_magnitude(image), 90))


def analyze_quality(image: np.ndarray) -> dict[str, float]:
    image = np.asarray(image, dtype=np.float32)
    noise = estimate_noise(image)
    signal = estimate_signal(image)
    return {
        "snr_db": estimate_snr_db(signal, noise),
        "noise": noise,
        "signal": signal,
        "contrast": estimate_contrast(image),
        "dynamic_range": estimate_dynamic_range(image),
        "entropy": estimate_entropy(image),
        "local_contrast": estimate_local_contrast(image),
        "laplacian_variance": estimate_laplacian_variance(image),
        "low_clip_fraction": estimate_low_clip_fraction(image),
        "high_clip_fraction": estimate_high_clip_fraction(image),
        "mean_gradient": estimate_mean_gradient(image),
        "median_gradient": estimate_median_gradient(image),
        "p90_gradient": estimate_p90_gradient(image),
    }
