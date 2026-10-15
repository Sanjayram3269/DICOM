from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.analysis.quality import analyze_quality
from backend.anatomy.classifier import classify
from backend.dicom.writer import write_dicom
from backend.enhancement.adaptive_engine import (
    EnhancementParameters,
    enhance,
    generate_candidates,
    select_parameters,
)
from backend.enhancement.quality_gate import evaluate_quality_gate
from backend.ingestion.raw_reader import RawProfile, normalize_float, read_raw


SEARCH_MAX_DIMENSION = 1024


def _downsample_for_search(image: np.ndarray, max_dimension: int = SEARCH_MAX_DIMENSION) -> np.ndarray:
    height, width = image.shape
    scale = min(1.0, max_dimension / max(height, width))
    if scale >= 1.0:
        return image.astype(np.float32, copy=False)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA).astype(np.float32)


def _gate_to_dict(result: Any) -> dict[str, Any]:
    if hasattr(result, "__dataclass_fields__"):
        return {
            key: getattr(result, key)
            for key in result.__dataclass_fields__
        }
    return dict(result)


def process_raw(
    raw_path: str | Path,
    profile: RawProfile,
    output_dir: str | Path,
) -> dict[str, Any]:
    raw_path = Path(raw_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_image = read_raw(raw_path, profile)
    image = normalize_float(raw_image)
    before_quality = analyze_quality(image)
    anatomy_result = classify(image)
    baseline = select_parameters(before_quality)
    candidates = generate_candidates(baseline)

    search_image = _downsample_for_search(image)
    search_before = analyze_quality(search_image)
    safe_candidates: list[tuple[int, EnhancementParameters, np.ndarray, Any]] = []
    candidate_results: list[dict[str, Any]] = []

    for index, parameters in enumerate(candidates, start=1):
        candidate_image = enhance(search_image, parameters)
        metrics = analyze_quality(candidate_image)
        gate = evaluate_quality_gate(search_before, metrics)
        gate_dict = _gate_to_dict(gate)
        candidate_results.append({
            "index": index,
            "parameters": parameters.to_dict(),
            "quality_gate": gate_dict,
        })
        if gate.accepted:
            safe_candidates.append((index, parameters, candidate_image, gate))

    if safe_candidates:
        # Prefer the highest final score; baseline wins score ties.
        selected = max(
            safe_candidates,
            key=lambda item: (
                round(float(item[3].final_score), 8),
                item[1] == baseline,
            ),
        )
        selected_candidate_index, selected_parameters, _, _ = selected
        candidate_image = enhance(image, selected_parameters)
        after_candidate_quality = analyze_quality(candidate_image)
        full_gate = evaluate_quality_gate(before_quality, after_candidate_quality)
        gate_dict = _gate_to_dict(full_gate)
        if full_gate.accepted:
            ratio = float(full_gate.blend_ratio)
            enhanced_image = np.clip((1.0 - ratio) * image + ratio * candidate_image, 0.0, 1.0)
        else:
            enhanced_image = image.copy()
            selected_candidate_index = None
        quality_gate_result = gate_dict
    else:
        selected_candidate_index = None
        selected_parameters = baseline
        enhanced_image = image.copy()
        quality_gate_result = _gate_to_dict(
            evaluate_quality_gate(before_quality, before_quality)
        )
        quality_gate_result.update({
            "accepted": False,
            "blend_ratio": 0.0,
            "reason": "all_candidates_rejected",
            "safety_passed": False,
        })

    after_quality = analyze_quality(enhanced_image)
    dicom_path = output_dir / f"{raw_path.stem}_enhanced.dcm"
    write_dicom(enhanced_image, dicom_path)

    candidate_counts = {}
    for item in candidate_results:
        key = str(item["index"])
        candidate_counts[key] = candidate_counts.get(key, 0) + 0
    search_summary = {
        "candidate_count": len(candidates),
        "safe_candidate_count": len(safe_candidates),
        "selected_candidate_index": selected_candidate_index,
        "baseline_selected": selected_candidate_index == 1,
        "search_max_dimension": SEARCH_MAX_DIMENSION,
        "search_width": int(search_image.shape[1]),
        "search_height": int(search_image.shape[0]),
        "full_resolution_width": int(image.shape[1]),
        "full_resolution_height": int(image.shape[0]),
        "search_resolution_scale": float(search_image.shape[1] / max(image.shape[1], 1)),
    }

    return {
        "anatomy": anatomy_result.to_dict(),
        "before": before_quality,
        "after": after_quality,
        "parameters": selected_parameters.to_dict(),
        "quality_gate": quality_gate_result,
        "candidate_search": search_summary,
        "candidate_count": len(candidates),
        "safe_candidate_count": len(safe_candidates),
        "selected_candidate_index": selected_candidate_index,
        "candidates": candidate_results,
        "dicom_path": str(dicom_path),
        "original": image,
        "enhanced": enhanced_image,
    }
