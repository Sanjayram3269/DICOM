from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.ingestion.raw_reader import (
    RawProfile,
    read_raw,
    normalize_float,
)

from backend.analysis.quality import (
    analyze_quality,
)

from backend.anatomy.classifier import (
    classify,
)

from backend.enhancement.adaptive_engine import (
    select_parameters,
    generate_candidates,
    enhance,
)

from backend.enhancement.quality_gate import (
    evaluate_enhancement,
    blend_with_original,
)

from backend.dicom.writer import (
    write_dicom,
)


# ---------------------------------------------------------------------
# 9J COARSE-TO-FINE SEARCH CONFIGURATION
# ---------------------------------------------------------------------

# Candidate search is performed on a smaller representative image.
#
# The final winning candidate is ALWAYS re-run at full resolution and
# evaluated again using the full-resolution quality gate.
#
# This preserves the safety boundary while avoiding 11 expensive
# full-resolution enhancement passes.

SEARCH_MAX_DIMENSION = 1024


def _gate_to_dict(gate) -> dict:
    """
    Convert QualityGateResult into a JSON-safe dictionary.

    Keeps the processor public result independent from the internal
    dataclass implementation.
    """

    return {
        "accepted": bool(gate.accepted),
        "blend_ratio": float(gate.blend_ratio),
        "score_before": float(gate.score_before),
        "score_enhanced": float(gate.score_enhanced),
        "final_score": float(gate.final_score),
        "reason": str(gate.reason),
        "snr_change_db": float(gate.snr_change_db),
        "noise_change": float(gate.noise_change),
        "contrast_change": float(gate.contrast_change),
        "entropy_change": float(gate.entropy_change),
        "safety_passed": bool(gate.safety_passed),
    }


def _downsample_for_search(
    image: np.ndarray,
    max_dimension: int = SEARCH_MAX_DIMENSION,
) -> np.ndarray:
    """
    Create a representative image for adaptive candidate search.

    The search image preserves the original aspect ratio.

    Only the candidate-selection stage uses this image.

    Full-resolution processing and the final quality gate remain
    completely independent from this downsampled representation.
    """

    if image.ndim != 2:
        raise ValueError(
            f"Expected a 2D grayscale image, got shape {image.shape}"
        )

    height, width = image.shape

    largest_dimension = max(height, width)

    if largest_dimension <= max_dimension:
        return image.astype(np.float32, copy=False)

    scale = max_dimension / float(largest_dimension)

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    search_image = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )

    return search_image.astype(np.float32, copy=False)


def _candidate_to_diagnostic(
    candidate: dict[str, Any],
) -> dict:
    """
    Convert an internal candidate result into lightweight diagnostics.

    Never expose image arrays because full-resolution candidate images
    are extremely large.
    """

    gate = candidate["gate"]
    parameters = candidate["parameters"]

    return {
        "index": int(candidate["index"]),
        "parameters": {
            "bilateral_sigma_color": float(
                parameters.bilateral_sigma_color
            ),
            "clahe_clip_limit": float(
                parameters.clahe_clip_limit
            ),
            "gamma": float(
                parameters.gamma
            ),
            "sharpen_amount": float(
                parameters.sharpen_amount
            ),
        },
        "accepted": bool(
            gate.accepted
        ),
        "blend_ratio": float(
            gate.blend_ratio
        ),
        "score_before": float(
            gate.score_before
        ),
        "score_enhanced": float(
            gate.score_enhanced
        ),
        "final_score": float(
            gate.final_score
        ),
        "reason": str(
            gate.reason
        ),
        "snr_change_db": float(
            gate.snr_change_db
        ),
        "noise_change": float(
            gate.noise_change
        ),
        "contrast_change": float(
            gate.contrast_change
        ),
        "entropy_change": float(
            gate.entropy_change
        ),
        "safety_passed": bool(
            gate.safety_passed
        ),
    }


def process_raw(
    path: str,
    profile: RawProfile,
    output_dir: str,
) -> dict:
    """
    Complete RAW radiograph processing pipeline.

    Pipeline:

        RAW
          ↓
        Read
          ↓
        Normalize
          ↓
        Quality analysis
          ↓
        Anatomy classification
          ↓
        Adaptive baseline parameter selection
          ↓
        Candidate generation
          ↓
        Downsampled candidate search
          ↓
        Candidate ranking
          ↓
        Best SAFE candidate
          ↓
        Full-resolution enhancement
          ↓
        Full-resolution quality gate
          ↓
        DICOM export

    Important design principle:

    The adaptive search is performed on a representative
    downsampled image for efficiency.

    The selected parameters are then applied to the ORIGINAL
    full-resolution image.

    The final full-resolution image is independently evaluated
    through the quality/safety gate before DICOM export.

    The enhancement itself remains deterministic.

    The adaptive intelligence is responsible for selecting the
    safest and highest-quality candidate.
    """

    # ---------------------------------------------------------
    # 1. Read RAW
    # ---------------------------------------------------------

    raw_image = read_raw(
        path=path,
        profile=profile,
    )

    # ---------------------------------------------------------
    # 2. Normalize to float32 [0, 1]
    # ---------------------------------------------------------

    image = normalize_float(
        raw_image
    ).astype(np.float32, copy=False)

    # ---------------------------------------------------------
    # 3. Analyze original full-resolution image
    # ---------------------------------------------------------

    before_quality = analyze_quality(
        image
    )

    # ---------------------------------------------------------
    # 4. Detect anatomy
    # ---------------------------------------------------------

    anatomy_result = classify(
        image
    )

    # ---------------------------------------------------------
    # 5. Select adaptive baseline parameters
    # ---------------------------------------------------------

    baseline_parameters = select_parameters(
        anatomy=anatomy_result.anatomy,
        quality=before_quality,
    )

    # ---------------------------------------------------------
    # 6. Generate controlled candidate parameter sets
    # ---------------------------------------------------------

    candidate_parameters = generate_candidates(
        baseline=baseline_parameters,
    )

    # ---------------------------------------------------------
    # 7. Create downsampled search image
    # ---------------------------------------------------------

    search_image = _downsample_for_search(
        image,
        max_dimension=SEARCH_MAX_DIMENSION,
    )

    search_quality = analyze_quality(
        search_image
    )

    # ---------------------------------------------------------
    # 8. Evaluate all candidates on SEARCH IMAGE
    # ---------------------------------------------------------

    candidate_results = []

    for candidate_index, parameters in enumerate(
        candidate_parameters,
        start=1,
    ):

        # -----------------------------------------------------
        # Generate candidate on downsampled image
        # -----------------------------------------------------

        candidate_image = enhance(
            search_image,
            parameters,
        )

        # -----------------------------------------------------
        # Analyze candidate
        # -----------------------------------------------------

        candidate_quality = analyze_quality(
            candidate_image
        )

        # -----------------------------------------------------
        # Apply safety + quality gate
        #
        # IMPORTANT:
        # The search-stage gate compares against the search-image
        # baseline, not the full-resolution baseline.
        # -----------------------------------------------------

        candidate_gate = evaluate_enhancement(
            before_quality=search_quality,
            enhanced_quality=candidate_quality,
        )

        candidate_results.append(
            {
                "index": candidate_index,
                "parameters": parameters,
                "image": candidate_image,
                "quality": candidate_quality,
                "gate": candidate_gate,
            }
        )

    # ---------------------------------------------------------
    # 9. Select best SAFE candidate from search
    # ---------------------------------------------------------

    safe_candidates = [
        candidate
        for candidate in candidate_results
        if candidate["gate"].accepted
    ]

    if safe_candidates:

        # Composite quality score is the primary ranking criterion.
        #
        # A tiny preference for baseline candidate #1 is retained
        # when scores are effectively identical.

        selected_search_candidate = max(
            safe_candidates,
            key=lambda candidate: (
                round(
                    candidate["gate"].final_score,
                    6,
                ),
                candidate["index"] == 1,
            ),
        )

    else:

        # No search candidate passed the safety gate.
        #
        # Never force enhancement when all candidates fail.

        selected_search_candidate = None

    # ---------------------------------------------------------
    # 10. Build candidate diagnostics
    # ---------------------------------------------------------

    candidate_diagnostics = [
        _candidate_to_diagnostic(candidate)
        for candidate in candidate_results
    ]

    # ---------------------------------------------------------
    # 11. Determine selected parameters
    # ---------------------------------------------------------

    if selected_search_candidate is not None:

        selected_parameters = (
            selected_search_candidate["parameters"]
        )

        selected_candidate_index = int(
            selected_search_candidate["index"]
        )

        baseline_selected = (
            selected_candidate_index == 1
        )

    else:

        selected_parameters = baseline_parameters

        selected_candidate_index = None

        baseline_selected = False

    # ---------------------------------------------------------
    # 12. FULL-RESOLUTION WINNER ENHANCEMENT
    # ---------------------------------------------------------

    #
    # This is the key optimization.
    #
    # Previously:
    #
    #   11 candidates × FULL resolution
    #
    # Now:
    #
    #   11 candidates × SMALL search image
    #   +
    #   1 winner × FULL resolution
    #
    # ---------------------------------------------------------

    if selected_search_candidate is not None:

        full_resolution_candidate = enhance(
            image,
            selected_parameters,
        )

        # -----------------------------------------------------
        # Full-resolution final quality gate
        #
        # This is the authoritative safety decision.
        # -----------------------------------------------------

        full_resolution_gate = evaluate_enhancement(
            before_quality=before_quality,
            enhanced_quality=analyze_quality(
                full_resolution_candidate
            ),
        )

        if full_resolution_gate.accepted:

            enhanced_image = blend_with_original(
                original=image,
                enhanced=full_resolution_candidate,
                ratio=full_resolution_gate.blend_ratio,
            )

            selected_gate = full_resolution_gate

        else:

            # -------------------------------------------------
            # Critical safety fallback.
            #
            # Even if a candidate looked good at search
            # resolution, the full-resolution image gets the
            # final say.
            #
            # If it fails, keep the original image.
            # -------------------------------------------------

            enhanced_image = image.copy()

            selected_gate = full_resolution_gate

            selected_parameters = baseline_parameters

            selected_candidate_index = None

            baseline_selected = False

    else:

        # -----------------------------------------------------
        # All search candidates rejected.
        # -----------------------------------------------------

        enhanced_image = image.copy()

        selected_gate = None

        selected_parameters = baseline_parameters

        selected_candidate_index = None

        baseline_selected = False

    # ---------------------------------------------------------
    # 13. Analyze final output
    # ---------------------------------------------------------

    after_quality = analyze_quality(
        enhanced_image
    )

    # ---------------------------------------------------------
    # 14. Final quality-gate response
    # ---------------------------------------------------------

    if selected_gate is not None:

        quality_gate_result = _gate_to_dict(
            selected_gate
        )

    else:

        quality_gate_result = {
            "accepted": False,
            "blend_ratio": 0.0,
            "score_before": 0.0,
            "score_enhanced": 0.0,
            "final_score": 0.0,
            "reason": "all_candidates_rejected",
            "snr_change_db": 0.0,
            "noise_change": 0.0,
            "contrast_change": 0.0,
            "entropy_change": 0.0,
            "safety_passed": False,
        }

    # ---------------------------------------------------------
    # 15. Search summary
    # ---------------------------------------------------------

    search_summary = {
        "candidate_count": len(
            candidate_results
        ),
        "safe_candidate_count": len(
            safe_candidates
        ),
        "selected_candidate_index": (
            selected_candidate_index
        ),
        "baseline_selected": (
            baseline_selected
        ),
        "search_max_dimension": SEARCH_MAX_DIMENSION,
        "search_width": int(
            search_image.shape[1]
        ),
        "search_height": int(
            search_image.shape[0]
        ),
        "full_resolution_width": int(
            image.shape[1]
        ),
        "full_resolution_height": int(
            image.shape[0]
        ),
        "search_resolution_scale": float(
            search_image.shape[0] / image.shape[0]
        ),
    }

    # ---------------------------------------------------------
    # 16. Create output directory
    # ---------------------------------------------------------

    output_path = Path(
        output_dir
    )

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # 17. Generate DICOM
    # ---------------------------------------------------------

    dicom_path = (
        output_path
        / f"{Path(path).stem}_enhanced.dcm"
    )

    write_dicom(
        image=enhanced_image,
        output_path=dicom_path,
        anatomy=anatomy_result.anatomy,
    )

    # ---------------------------------------------------------
    # 18. Return complete processing result
    # ---------------------------------------------------------

    return {
        "anatomy": anatomy_result.to_dict(),

        "before": before_quality,

        "after": after_quality,

        # Preserve existing frontend/backend contract.
        "parameters": selected_parameters.__dict__,

        "quality_gate": quality_gate_result,

        # 9J search summary.
        "candidate_search": search_summary,

        "candidate_count": len(
            candidate_results
        ),

        "safe_candidate_count": len(
            safe_candidates
        ),

        "selected_candidate_index": (
            selected_candidate_index
        ),

        # Lightweight candidate diagnostics.
        "candidates": candidate_diagnostics,

        "dicom_path": str(
            dicom_path
        ),

        # Keep these internally available for
        # the frontend image viewer.
        "original": image,

        "enhanced": enhanced_image,
    }