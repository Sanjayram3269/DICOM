from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class EnhancementParameters:
    bilateral_sigma_color: float
    clahe_clip_limit: float
    gamma: float
    sharpen_amount: float


def _clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    return float(
        np.clip(
            value,
            minimum,
            maximum,
        )
    )


def _normalize(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    """
    Normalize a value into [0, 1] using a bounded engineering range.
    """

    if maximum <= minimum:
        return 0.0

    return float(
        np.clip(
            (value - minimum)
            / (maximum - minimum),
            0.0,
            1.0,
        )
    )


def select_parameters(
    anatomy: str,
    quality: dict,
) -> EnhancementParameters:
    """
    Select the baseline enhancement parameters from image-quality signals.

    This remains the primary adaptive parameter policy.

    9J builds on top of this baseline by generating nearby candidate
    parameter sets and evaluating them through the quality gate.
    """

    # ---------------------------------------------------------
    # Extract quality metrics
    # ---------------------------------------------------------

    snr_db = float(
        quality.get(
            "snr_db",
            20.0,
        )
    )

    local_contrast = float(
        quality.get(
            "local_contrast",
            0.01,
        )
    )

    laplacian_variance = float(
        quality.get(
            "laplacian_variance",
            0.001,
        )
    )

    low_clip_fraction = float(
        quality.get(
            "low_clip_fraction",
            0.0,
        )
    )

    entropy = float(
        quality.get(
            "entropy",
            7.0,
        )
    )

    mean_intensity = float(
        quality.get(
            "mean_intensity",
            0.45,
        )
    )

    # ---------------------------------------------------------
    # 1. Noise score
    #
    # Higher score = noisier image
    # ---------------------------------------------------------

    noise_score = 1.0 - _normalize(
        snr_db,
        15.0,
        40.0,
    )

    # ---------------------------------------------------------
    # 2. Contrast deficit
    #
    # Higher score = more contrast enhancement needed
    # ---------------------------------------------------------

    contrast_score = 1.0 - _normalize(
        local_contrast,
        0.006,
        0.015,
    )

    # ---------------------------------------------------------
    # 3. Sharpness deficit
    #
    # Higher score = softer image
    # ---------------------------------------------------------

    sharpness_score = 1.0 - _normalize(
        laplacian_variance,
        0.0006,
        0.0025,
    )

    # ---------------------------------------------------------
    # 4. Low-end clipping
    #
    # Higher score = more dark clipping
    # ---------------------------------------------------------

    clipping_score = _normalize(
        low_clip_fraction,
        0.02,
        0.40,
    )

    # ---------------------------------------------------------
    # 5. Entropy deficit
    #
    # Lower entropy can indicate reduced image information.
    # ---------------------------------------------------------

    entropy_score = 1.0 - _normalize(
        entropy,
        5.5,
        7.9,
    )

    # ---------------------------------------------------------
    # 6. Combined image-condition scores
    # ---------------------------------------------------------

    denoise_score = (
        0.70 * noise_score
        + 0.30 * (1.0 - sharpness_score)
    )

    contrast_need = (
        0.55 * contrast_score
        + 0.25 * entropy_score
        + 0.20 * clipping_score
    )

    sharpening_need = (
        0.65 * sharpness_score
        + 0.20 * contrast_score
        + 0.15 * entropy_score
    )

    # ---------------------------------------------------------
    # 7. Bilateral filtering strength
    # ---------------------------------------------------------

    bilateral_sigma_color = _clamp(
        8.0
        + 18.0 * denoise_score,
        8.0,
        26.0,
    )

    # ---------------------------------------------------------
    # 8. CLAHE strength
    # ---------------------------------------------------------

    clahe_clip_limit = _clamp(
        1.2
        + 1.4 * contrast_need,
        1.2,
        2.6,
    )

    # ---------------------------------------------------------
    # 9. Gamma
    # ---------------------------------------------------------

    brightness_offset = _normalize(
        mean_intensity,
        0.25,
        0.60,
    )

    gamma = _clamp(
        0.96
        - 0.10 * (1.0 - brightness_offset)
        + 0.04 * brightness_offset,
        0.88,
        1.00,
    )

    # ---------------------------------------------------------
    # 10. Sharpening
    # ---------------------------------------------------------

    sharpen_amount = _clamp(
        0.06
        + 0.24 * sharpening_need,
        0.06,
        0.30,
    )

    # ---------------------------------------------------------
    # 11. Anatomy hook
    # ---------------------------------------------------------

    _ = anatomy

    return EnhancementParameters(
        bilateral_sigma_color=round(
            bilateral_sigma_color,
            3,
        ),
        clahe_clip_limit=round(
            clahe_clip_limit,
            3,
        ),
        gamma=round(
            gamma,
            3,
        ),
        sharpen_amount=round(
            sharpen_amount,
            3,
        ),
    )


def generate_candidates(
    baseline: EnhancementParameters,
) -> list[EnhancementParameters]:
    """
    Generate a small, bounded candidate neighborhood around
    the adaptive baseline.

    The baseline itself is always included.

    Candidate search is deliberately conservative. We are not
    exploring arbitrary enhancement configurations; we are
    testing controlled variations around the image-specific
    recommendation.

    Candidate dimensions:

        bilateral_sigma_color
        clahe_clip_limit
        gamma
        sharpen_amount

    The resulting candidates are deterministic and deduplicated.
    """

    # ---------------------------------------------------------
    # Candidate offsets
    #
    # These represent controlled local exploration around
    # the baseline parameter set.
    # ---------------------------------------------------------

    bilateral_offsets = (
        -3.0,
        0.0,
        3.0,
    )

    clahe_offsets = (
        -0.25,
        0.0,
        0.25,
    )

    gamma_offsets = (
        -0.025,
        0.0,
        0.025,
    )

    sharpen_offsets = (
        -0.05,
        0.0,
        0.05,
    )

    candidates: list[EnhancementParameters] = []

    # ---------------------------------------------------------
    # Primary candidate neighborhood
    #
    # We vary one parameter at a time around the baseline.
    # This keeps the search interpretable and prevents an
    # uncontrolled combinatorial explosion.
    # ---------------------------------------------------------

    candidates.append(
        baseline
    )

    # Bilateral alternatives
    for offset in bilateral_offsets:
        if offset == 0.0:
            continue

        candidates.append(
            EnhancementParameters(
                bilateral_sigma_color=round(
                    _clamp(
                        baseline.bilateral_sigma_color
                        + offset,
                        8.0,
                        26.0,
                    ),
                    3,
                ),
                clahe_clip_limit=baseline.clahe_clip_limit,
                gamma=baseline.gamma,
                sharpen_amount=baseline.sharpen_amount,
            )
        )

    # CLAHE alternatives
    for offset in clahe_offsets:
        if offset == 0.0:
            continue

        candidates.append(
            EnhancementParameters(
                bilateral_sigma_color=baseline.bilateral_sigma_color,
                clahe_clip_limit=round(
                    _clamp(
                        baseline.clahe_clip_limit
                        + offset,
                        1.2,
                        2.6,
                    ),
                    3,
                ),
                gamma=baseline.gamma,
                sharpen_amount=baseline.sharpen_amount,
            )
        )

    # Gamma alternatives
    for offset in gamma_offsets:
        if offset == 0.0:
            continue

        candidates.append(
            EnhancementParameters(
                bilateral_sigma_color=baseline.bilateral_sigma_color,
                clahe_clip_limit=baseline.clahe_clip_limit,
                gamma=round(
                    _clamp(
                        baseline.gamma
                        + offset,
                        0.88,
                        1.00,
                    ),
                    3,
                ),
                sharpen_amount=baseline.sharpen_amount,
            )
        )

    # Sharpening alternatives
    for offset in sharpen_offsets:
        if offset == 0.0:
            continue

        candidates.append(
            EnhancementParameters(
                bilateral_sigma_color=baseline.bilateral_sigma_color,
                clahe_clip_limit=baseline.clahe_clip_limit,
                gamma=baseline.gamma,
                sharpen_amount=round(
                    _clamp(
                        baseline.sharpen_amount
                        + offset,
                        0.06,
                        0.30,
                    ),
                    3,
                ),
            )
        )

    # ---------------------------------------------------------
    # Add two conservative combined candidates.
    #
    # These explore the two most common interactions:
    #
    #   denoise + contrast
    #   contrast + sharpening
    #
    # while keeping the baseline as the reference point.
    # ---------------------------------------------------------

    candidates.append(
        EnhancementParameters(
            bilateral_sigma_color=round(
                _clamp(
                    baseline.bilateral_sigma_color
                    + 2.0,
                    8.0,
                    26.0,
                ),
                3,
            ),
            clahe_clip_limit=round(
                _clamp(
                    baseline.clahe_clip_limit
                    + 0.15,
                    1.2,
                    2.6,
                ),
                3,
            ),
            gamma=baseline.gamma,
            sharpen_amount=baseline.sharpen_amount,
        )
    )

    candidates.append(
        EnhancementParameters(
            bilateral_sigma_color=baseline.bilateral_sigma_color,
            clahe_clip_limit=round(
                _clamp(
                    baseline.clahe_clip_limit
                    + 0.15,
                    1.2,
                    2.6,
                ),
                3,
            ),
            gamma=baseline.gamma,
            sharpen_amount=round(
                _clamp(
                    baseline.sharpen_amount
                    + 0.03,
                    0.06,
                    0.30,
                ),
                3,
            ),
        )
    )

    # ---------------------------------------------------------
    # Deduplicate candidates.
    # ---------------------------------------------------------

    unique_candidates: list[
        EnhancementParameters
    ] = []

    seen: set[
        tuple[float, float, float, float]
    ] = set()

    for candidate in candidates:

        key = (
            candidate.bilateral_sigma_color,
            candidate.clahe_clip_limit,
            candidate.gamma,
            candidate.sharpen_amount,
        )

        if key in seen:
            continue

        seen.add(key)
        unique_candidates.append(
            candidate
        )

    return unique_candidates


def enhance(
    image: np.ndarray,
    params: EnhancementParameters,
) -> np.ndarray:
    """
    Apply deterministic image enhancement using selected parameters.

    Pixel transformation remains deterministic. The intelligence
    layer only chooses the parameters.
    """

    x = np.clip(
        image.astype(np.float32),
        0,
        1,
    )

    # ---------------------------------------------------------
    # Bilateral denoising
    # ---------------------------------------------------------

    u8 = (
        x * 255
    ).astype(
        np.uint8
    )

    u8 = cv2.bilateralFilter(
        u8,
        7,
        params.bilateral_sigma_color,
        7,
    )

    # ---------------------------------------------------------
    # CLAHE
    # ---------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=params.clahe_clip_limit,
        tileGridSize=(8, 8),
    )

    y = (
        clahe.apply(u8)
        .astype(np.float32)
        / 255.0
    )

    # ---------------------------------------------------------
    # Gamma correction
    # ---------------------------------------------------------

    y = np.power(
        np.clip(y, 0, 1),
        params.gamma,
    )

    # ---------------------------------------------------------
    # Conservative unsharp masking
    # ---------------------------------------------------------

    blur = cv2.GaussianBlur(
        y,
        (0, 0),
        1.0,
    )

    y = np.clip(
        y
        + params.sharpen_amount
        * (y - blur),
        0,
        1,
    )

    return y.astype(
        np.float32
    )