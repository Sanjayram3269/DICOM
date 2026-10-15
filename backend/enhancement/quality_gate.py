from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QualityGateResult:
    accepted: bool
    blend_ratio: float
    score_before: float
    score_enhanced: float
    final_score: float
    reason: str

    # Diagnostic fields
    snr_change_db: float
    noise_change: float
    contrast_change: float
    entropy_change: float
    safety_passed: bool


def _clip(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))


def quality_score(quality: dict) -> float:
    """
    Composite engineering quality score.

    This is used only after the hard safety checks pass.
    It is not intended to represent clinical image quality.
    """
    snr_score = _clip(
        quality["snr_db"] / 40.0,
        0.0,
        1.0,
    )

    contrast_score = _clip(
        quality["local_contrast"] / 0.015,
        0.0,
        1.0,
    )

    entropy_score = _clip(
        quality["entropy"] / 8.0,
        0.0,
        1.0,
    )

    sharpness_score = _clip(
        quality["laplacian_variance"] / 0.0025,
        0.0,
        1.0,
    )

    noise_penalty = _clip(
        quality["noise"] / 0.05,
        0.0,
        1.0,
    )

    clipping_penalty = _clip(
        (
            quality["low_clip_fraction"]
            + quality["high_clip_fraction"]
        )
        / 0.50,
        0.0,
        1.0,
    )

    return float(
        0.40 * snr_score
        + 0.20 * contrast_score
        + 0.15 * entropy_score
        + 0.15 * sharpness_score
        - 0.07 * noise_penalty
        - 0.03 * clipping_penalty
    )


def evaluate_enhancement(
    before_quality: dict,
    enhanced_quality: dict,
) -> QualityGateResult:
    """
    Evaluate an enhanced image using two stages:

    Stage 1:
        Hard safety constraints.

    Stage 2:
        Composite quality score.

    The safety stage prevents an enhancement from being accepted
    merely because improvements in one metric compensate for
    degradation in another.
    """

    snr_change = (
        enhanced_quality["snr_db"]
        - before_quality["snr_db"]
    )

    noise_change = (
        enhanced_quality["noise"]
        - before_quality["noise"]
    )

    contrast_change = (
        enhanced_quality["contrast"]
        - before_quality["contrast"]
    )

    entropy_change = (
        enhanced_quality["entropy"]
        - before_quality["entropy"]
    )

    score_before = quality_score(before_quality)
    score_enhanced = quality_score(enhanced_quality)

    # ---------------------------------------------------------
    # Hard safety constraints
    # ---------------------------------------------------------

    # A small amount of numerical variation is tolerated.
    SNR_TOLERANCE_DB = 0.05

    # Allow a small relative increase in the noise proxy.
    NOISE_INCREASE_RATIO = 0.10
    NOISE_INCREASE_FLOOR = 0.00010

    # Do not allow severe loss of global contrast.
    CONTRAST_LOSS_RATIO = 0.08

    # SNR must not materially degrade.
    snr_safe = snr_change >= -SNR_TOLERANCE_DB

    # Noise must not increase excessively.
    baseline_noise = max(
    float(before_quality["noise"]),
    1e-8,
    )

    allowed_noise_increase = max(
        baseline_noise * NOISE_INCREASE_RATIO,
        NOISE_INCREASE_FLOOR,   
    )

    noise_safe = (
        noise_change <= allowed_noise_increase
    )

    # Contrast should not collapse.
    baseline_contrast = max(
        float(before_quality["contrast"]),
        1e-8,
    )

    contrast_safe = (
        contrast_change
        >= -baseline_contrast * CONTRAST_LOSS_RATIO
    )

    safety_passed = (
        snr_safe
        and noise_safe
        and contrast_safe
    )

    # ---------------------------------------------------------
    # Reject immediately if safety constraints fail.
    # ---------------------------------------------------------

    if not safety_passed:
        failed_constraints = []

        if not snr_safe:
            failed_constraints.append("snr_degradation")

        if not noise_safe:
            failed_constraints.append("noise_increase")

        if not contrast_safe:
            failed_constraints.append("contrast_loss")

        return QualityGateResult(
            accepted=False,
            blend_ratio=0.0,
            score_before=score_before,
            score_enhanced=score_enhanced,
            final_score=score_before,
            reason="safety_rejected:" + ",".join(failed_constraints),
            snr_change_db=float(snr_change),
            noise_change=float(noise_change),
            contrast_change=float(contrast_change),
            entropy_change=float(entropy_change),
            safety_passed=False,
        )

    # ---------------------------------------------------------
    # Composite quality decision
    # ---------------------------------------------------------

    improvement = score_enhanced - score_before

    if improvement >= 0.005:
        return QualityGateResult(
            accepted=True,
            blend_ratio=1.0,
            score_before=score_before,
            score_enhanced=score_enhanced,
            final_score=score_enhanced,
            reason="enhancement_improved_quality",
            snr_change_db=float(snr_change),
            noise_change=float(noise_change),
            contrast_change=float(contrast_change),
            entropy_change=float(entropy_change),
            safety_passed=True,
        )

    if improvement >= -0.002:
        return QualityGateResult(
            accepted=True,
            blend_ratio=0.75,
            score_before=score_before,
            score_enhanced=score_enhanced,
            final_score=(
                0.75 * score_enhanced
                + 0.25 * score_before
            ),
            reason="enhancement_within_tolerance",
            snr_change_db=float(snr_change),
            noise_change=float(noise_change),
            contrast_change=float(contrast_change),
            entropy_change=float(entropy_change),
            safety_passed=True,
        )

    return QualityGateResult(
        accepted=False,
        blend_ratio=0.0,
        score_before=score_before,
        score_enhanced=score_enhanced,
        final_score=score_before,
        reason="quality_score_rejected",
        snr_change_db=float(snr_change),
        noise_change=float(noise_change),
        contrast_change=float(contrast_change),
        entropy_change=float(entropy_change),
        safety_passed=True,
    )


def blend_with_original(
    original: np.ndarray,
    enhanced: np.ndarray,
    ratio: float,
) -> np.ndarray:
    """
    Blend the enhanced image with the original image.
    """
    ratio = _clip(ratio, 0.0, 1.0)

    blended = (
        (1.0 - ratio) * original
        + ratio * enhanced
    )

    return np.clip(
        blended,
        0.0,
        1.0,
    ).astype(np.float32)