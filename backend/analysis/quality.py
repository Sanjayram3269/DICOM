from __future__ import annotations

import cv2
import numpy as np

def _safe_float(value: float) -> float:
    return float(value)

def estimate_noise(image: np.ndarray) -> float:
    """
    Estimate high-frequency noise using a robust residual method.

    The image is smoothed with a small Gaussian kernel and the
    high-frequency residual is measured using MAD (median absolute
    deviation).

    This is an engineering noise proxy, not a clinical noise metric.
    """
    image = image.astype(np.float32, copy=False)

    smoothed = cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=1.0,
        sigmaY=1.0,
    )

    residual = image - smoothed

    median_residual = np.median(residual)

    mad = np.median(
        np.abs(residual - median_residual)
    )

    # Convert MAD to a Gaussian-equivalent standard deviation.
    noise = 1.4826 * mad

    return _safe_float(noise)


def estimate_signal(image: np.ndarray) -> float:
    return _safe_float(
        np.mean(np.abs(image))
    )

def estimate_snr_db(
    signal: float,
    noise: float,
) -> float:
    if noise <= 1e-12:
        return 100.0

    return _safe_float(
        20.0 * np.log10(
            max(signal, 1e-12)
            / noise
        )
    )


def estimate_contrast(image: np.ndarray) -> float:
    return _safe_float(
        np.std(image)
    )


def estimate_dynamic_range(image: np.ndarray) -> float:
    return _safe_float(
        np.max(image) - np.min(image)
    )


def estimate_entropy(
    image: np.ndarray,
    bins: int = 256,
) -> float:
    histogram, _ = np.histogram(
        image,
        bins=bins,
        range=(0.0, 1.0),
    )

    histogram = histogram.astype(
        np.float64
    )

    total = histogram.sum()

    if total <= 0:
        return 0.0

    probabilities = histogram / total

    probabilities = probabilities[
        probabilities > 0
    ]

    return _safe_float(
        -np.sum(
            probabilities
            * np.log2(probabilities)
        )
    )


def estimate_local_contrast(
    image: np.ndarray,
) -> float:
    """
    Estimate local contrast using local standard deviation.
    """
    mean = cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=5.0,
        sigmaY=5.0,
    )

    mean_squared = cv2.GaussianBlur(
        image * image,
        (0, 0),
        sigmaX=5.0,
        sigmaY=5.0,
    )

    variance = np.maximum(
        mean_squared - mean * mean,
        0.0,
    )

    local_std = np.sqrt(variance)

    return _safe_float(
        np.mean(local_std)
    )


def estimate_laplacian_variance(
    image: np.ndarray,
) -> float:
    laplacian = cv2.Laplacian(
        image,
        cv2.CV_32F,
    )

    return _safe_float(
        np.var(laplacian)
    )


def estimate_low_clip_fraction(
    image: np.ndarray,
    threshold: float = 0.01,
) -> float:
    return _safe_float(
        np.mean(image <= threshold)
    )


def estimate_high_clip_fraction(
    image: np.ndarray,
    threshold: float = 0.99,
) -> float:
    return _safe_float(
        np.mean(image >= threshold)
    )


def estimate_mean_gradient(
    image: np.ndarray,
) -> float:
    gx = cv2.Sobel(
        image,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        image,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    magnitude = cv2.magnitude(
        gx,
        gy,
    )

    return _safe_float(
        np.mean(magnitude)
    )


def estimate_median_gradient(
    image: np.ndarray,
) -> float:
    gx = cv2.Sobel(
        image,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        image,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    magnitude = cv2.magnitude(
        gx,
        gy,
    )

    return _safe_float(
        np.median(magnitude)
    )


def estimate_p90_gradient(
    image: np.ndarray,
) -> float:
    gx = cv2.Sobel(
        image,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        image,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    magnitude = cv2.magnitude(
        gx,
        gy,
    )

    return _safe_float(
        np.percentile(
            magnitude,
            90,
        )
    )


def analyze_quality(
    image: np.ndarray,
) -> dict:
    """
    Compute the full engineering quality profile.

    Returns:
        - signal/noise/SNR
        - intensity distribution
        - clipping
        - local contrast
        - gradient/detail
        - Laplacian sharpness
    """

    image = np.asarray(
        image,
        dtype=np.float32,
    )

    signal = estimate_signal(image)
    noise = estimate_noise(image)

    return {
        # Core quality metrics
        "snr_db": estimate_snr_db(
            signal,
            noise,
        ),
        "noise": noise,
        "signal": signal,
        "contrast": estimate_contrast(image),
        "dynamic_range": estimate_dynamic_range(image),
        "entropy": estimate_entropy(image),

        # Intensity distribution
        "mean_intensity": _safe_float(
            np.mean(image)
        ),
        "median_intensity": _safe_float(
            np.median(image)
        ),
        "std_intensity": _safe_float(
            np.std(image)
        ),
        "p01": _safe_float(
            np.percentile(image, 1)
        ),
        "p05": _safe_float(
            np.percentile(image, 5)
        ),
        "p25": _safe_float(
            np.percentile(image, 25)
        ),
        "p75": _safe_float(
            np.percentile(image, 75)
        ),
        "p95": _safe_float(
            np.percentile(image, 95)
        ),
        "p99": _safe_float(
            np.percentile(image, 99)
        ),
        "percentile_dynamic_range": _safe_float(
            np.percentile(image, 99)
            - np.percentile(image, 1)
        ),
        "interquartile_range": _safe_float(
            np.percentile(image, 75)
            - np.percentile(image, 25)
        ),

        # Clipping
        "low_clip_fraction": estimate_low_clip_fraction(
            image
        ),
        "high_clip_fraction": estimate_high_clip_fraction(
            image
        ),

        # Spatial/detail
        "mean_gradient": estimate_mean_gradient(
            image
        ),
        "median_gradient": estimate_median_gradient(
            image
        ),
        "p90_gradient": estimate_p90_gradient(
            image
        ),
        "laplacian_variance": estimate_laplacian_variance(
            image
        ),
        "local_contrast": estimate_local_contrast(
            image
        ),
    }