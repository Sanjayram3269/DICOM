from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SNR_TOLERANCE_DB = 0.05
NOISE_INCREASE_RATIO = 0.10
NOISE_INCREASE_FLOOR = 0.00010
CONTRAST_LOSS_RATIO = 0.08


@dataclass(frozen=True)
class QualityGateResult:
    accepted: bool
    blend_ratio: float
    score_before: float
    score_enhanced: float
    final_score: float
    reason: str
    snr_change_db: float
    noise_change: float
    contrast_change: float
    entropy_change: float
    safety_passed: bool


def quality_score(metrics: dict[str, float]) -> float:
    snr_score = float(np.clip(metrics.get("snr_db", 0.0) / 40.0, 0.0, 1.0))
    contrast_score = float(np.clip(metrics.get("local_contrast", 0.0) / 0.015, 0.0, 1.0))
    entropy_score = float(np.clip(metrics.get("entropy", 0.0) / 8.0, 0.0, 1.0))
    sharpness_score = float(np.clip(metrics.get("laplacian_variance", 0.0) / 0.0025, 0.0, 1.0))
    noise_penalty = float(np.clip(metrics.get("noise", 0.0) / 0.05, 0.0, 1.0))
    clipping = metrics.get("low_clip_fraction", 0.0) + metrics.get("high_clip_fraction", 0.0)
    clipping_penalty = float(np.clip(clipping / 0.50, 0.0, 1.0))
    return 0.40 * snr_score + 0.20 * contrast_score + 0.15 * entropy_score + 0.15 * sharpness_score - 0.07 * noise_penalty - 0.03 * clipping_penalty


def evaluate_quality_gate(
    before: dict[str, float],
    enhanced: dict[str, float],
) -> QualityGateResult:
    snr_change = float(enhanced.get("snr_db", 0.0) - before.get("snr_db", 0.0))
    noise_before = float(before.get("noise", 0.0))
    noise_change = float(enhanced.get("noise", 0.0) - noise_before)
    contrast_before = float(before.get("contrast", 0.0))
    contrast_change = float(enhanced.get("contrast", 0.0) - contrast_before)
    entropy_change = float(enhanced.get("entropy", 0.0) - before.get("entropy", 0.0))
    allowed_noise = max(noise_before * NOISE_INCREASE_RATIO, NOISE_INCREASE_FLOOR)
    constraints = []
    if snr_change < -SNR_TOLERANCE_DB:
        constraints.append("snr_degradation")
    if noise_change > allowed_noise:
        constraints.append("noise_increase")
    if contrast_before > 1e-12 and contrast_change < -CONTRAST_LOSS_RATIO * contrast_before:
        constraints.append("contrast_loss")
    before_score = quality_score(before)
    enhanced_score = quality_score(enhanced)
    improvement = enhanced_score - before_score
    if constraints:
        return QualityGateResult(False, 0.0, before_score, enhanced_score, before_score, "safety_rejected:" + ",".join(constraints), snr_change, noise_change, contrast_change, entropy_change, False)
    if improvement >= 0.005:
        return QualityGateResult(True, 1.0, before_score, enhanced_score, enhanced_score, "accepted_full", snr_change, noise_change, contrast_change, entropy_change, True)
    if improvement >= -0.002:
        return QualityGateResult(True, 0.75, before_score, enhanced_score, before_score + 0.75 * improvement, "accepted_blend", snr_change, noise_change, contrast_change, entropy_change, True)
    return QualityGateResult(False, 0.0, before_score, enhanced_score, before_score, "quality_score_rejected", snr_change, noise_change, contrast_change, entropy_change, True)
