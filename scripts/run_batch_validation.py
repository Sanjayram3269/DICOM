from pathlib import Path
import csv
import json
import shutil
import sys
import time

import cv2
import numpy as np
import pydicom

MAX_IMAGES = 5


# ---------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from backend.ingestion.raw_reader import RawProfile
from backend.pipeline.processor import process_raw


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

DATASET_DIR = (
    BASE_DIR
    / "data"
    / "demo"
    / "real"
    / "NLM-MontgomeryCXRSet"
    / "MontgomerySet"
    / "CXR_png"
)

VALIDATION_DIR = (
    BASE_DIR
    / "data"
    / "demo"
    / "real"
    / "batch_validation"
)

WORK_DIR = VALIDATION_DIR / "_work"

SUMMARY_CSV = VALIDATION_DIR / "batch_summary.csv"
SUMMARY_JSON = VALIDATION_DIR / "batch_summary.json"
FAILURES_JSON = VALIDATION_DIR / "failures.json"


# ---------------------------------------------------------------------
# RAW conversion
# ---------------------------------------------------------------------

def convert_to_16bit_raw(
    image: np.ndarray,
    output_path: Path,
) -> tuple[int, int]:

    if image.ndim != 2:
        raise ValueError(
            f"Expected grayscale image, got shape {image.shape}"
        )

    if image.dtype == np.uint8:
        image_16 = image.astype(np.uint16) * 257

    elif image.dtype == np.uint16:
        image_16 = image

    else:
        raise ValueError(
            f"Unsupported source dtype: {image.dtype}"
        )

    image_16 = np.ascontiguousarray(image_16)

    output_path.write_bytes(
        image_16.astype("<u2").tobytes()
    )

    height, width = image_16.shape

    return width, height


# ---------------------------------------------------------------------
# DICOM validation
# ---------------------------------------------------------------------

def validate_dicom(
    dicom_path: Path,
    expected_width: int,
    expected_height: int,
) -> dict:

    if not dicom_path.exists():
        raise FileNotFoundError(
            f"DICOM was not generated: {dicom_path}"
        )

    ds = pydicom.dcmread(
        str(dicom_path),
        stop_before_pixels=False,
    )

    checks = {
        "modality": getattr(ds, "Modality", None) == "DX",

        "rows": int(
            getattr(ds, "Rows", -1)
        ) == expected_height,

        "columns": int(
            getattr(ds, "Columns", -1)
        ) == expected_width,

        "samples_per_pixel": int(
            getattr(ds, "SamplesPerPixel", -1)
        ) == 1,

        "bits_allocated": int(
            getattr(ds, "BitsAllocated", -1)
        ) == 16,

        "bits_stored": int(
            getattr(ds, "BitsStored", -1)
        ) == 16,

        "high_bit": int(
            getattr(ds, "HighBit", -1)
        ) == 15,

        "pixel_representation": int(
            getattr(ds, "PixelRepresentation", -1)
        ) == 0,

        "photometric": getattr(
            ds,
            "PhotometricInterpretation",
            None,
        ) == "MONOCHROME2",
    }

    passed = all(checks.values())

    if not passed:
        failed = [
            name
            for name, value in checks.items()
            if not value
        ]

        raise ValueError(
            "DICOM validation failed: "
            + ", ".join(failed)
        )

    return {
        "sop_class_uid": str(
            getattr(ds, "SOPClassUID", "")
        ),
        "modality": str(
            getattr(ds, "Modality", "")
        ),
        "rows": int(ds.Rows),
        "columns": int(ds.Columns),
        "bits_allocated": int(ds.BitsAllocated),
        "bits_stored": int(ds.BitsStored),
        "photometric": str(
            ds.PhotometricInterpretation
        ),
    }


# ---------------------------------------------------------------------
# Quality-gate helpers
# ---------------------------------------------------------------------

def _get_quality_gate(
    result: dict,
    before: dict,
    after: dict,
) -> dict:

    """
    Normalize quality-gate information.

    The production pipeline owns the actual gate decision.

    This function makes the batch validator tolerant of small
    differences in the returned result structure while keeping
    diagnostics consistent for JSON/CSV reporting.
    """

    gate = result.get("quality_gate")

    if gate is None:
        raise KeyError(
            "process_raw() did not return 'quality_gate'"
        )

    # Always derive these metric deltas directly from the
    # before/after quality dictionaries. This avoids relying
    # on duplicated diagnostic fields inside the gate object.
    snr_change_db = (
        float(after["snr_db"])
        - float(before["snr_db"])
    )

    noise_change = (
        float(after["noise"])
        - float(before["noise"])
    )

    contrast_change = (
        float(after["contrast"])
        - float(before["contrast"])
    )

    entropy_change = (
        float(after["entropy"])
        - float(before["entropy"])
    )

    return {
        "accepted": bool(
            gate.get("accepted", False)
        ),

        "blend_ratio": float(
            gate.get("blend_ratio", 0.0)
        ),

        "score_before": float(
            gate.get("score_before", 0.0)
        ),

        "score_enhanced": float(
            gate.get("score_enhanced", 0.0)
        ),

        "final_score": float(
            gate.get("final_score", 0.0)
        ),

        "reason": str(
            gate.get("reason", "")
        ),

        "snr_change_db": snr_change_db,

        "noise_change": noise_change,

        "contrast_change": contrast_change,

        "entropy_change": entropy_change,

        "safety_passed": bool(
            gate.get("safety_passed", False)
        ),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    print("=" * 80)
    print("MONTGOMERY DATASET — FULL BATCH VALIDATION")
    print("=" * 80)

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found:\n{DATASET_DIR}"
        )

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Clean previous temporary work directory only.
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)

    WORK_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------
    # Find actual PNG radiographs
    # -----------------------------------------------------------------

    image_files = sorted(
        [
            path
            for path in DATASET_DIR.glob("*.png")
            if path.is_file()
        ]
    )

    total_available = len(image_files)

    if MAX_IMAGES is not None:
        image_files = image_files[:MAX_IMAGES]

    print(f"Images selected  : {len(image_files)}")
    print(f"Validation limit : {MAX_IMAGES}")

    print()
    print(f"Dataset directory : {DATASET_DIR}")
    print(f"Radiographs found : {len(image_files)}")

    if not image_files:
        raise RuntimeError(
            "No PNG radiographs found."
        )

    print()

    results = []
    failures = []

    total_start = time.perf_counter()

    # -----------------------------------------------------------------
    # Process every image
    # -----------------------------------------------------------------

    for index, source_path in enumerate(
        image_files,
        start=1,
    ):

        case_name = source_path.stem

        print(
            f"[{index:03d}/{len(image_files):03d}] "
            f"{case_name}",
            end=" ",
            flush=True,
        )

        case_start = time.perf_counter()

        case_work_dir = (
            WORK_DIR / case_name
        )

        case_work_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        raw_path = (
            case_work_dir
            / f"{case_name}_16bit.raw"
        )

        try:

            # ---------------------------------------------------------
            # Read source image
            # ---------------------------------------------------------

            image = cv2.imread(
                str(source_path),
                cv2.IMREAD_UNCHANGED,
            )

            if image is None:
                raise RuntimeError(
                    "Could not read source image"
                )

            source_dtype = str(
                image.dtype
            )

            if image.ndim != 2:
                raise ValueError(
                    f"Expected grayscale image, got shape {image.shape}"
                )

            source_height, source_width = (
                image.shape
            )

            # ---------------------------------------------------------
            # Convert to controlled RAW
            # ---------------------------------------------------------

            width, height = (
                convert_to_16bit_raw(
                    image,
                    raw_path,
                )
            )

            # ---------------------------------------------------------
            # RAW profile
            # ---------------------------------------------------------

            profile = RawProfile(
                width=width,
                height=height,
                bit_depth=16,
                signed=False,
                endian="little",
            )

            # ---------------------------------------------------------
            # Run existing production pipeline
            # ---------------------------------------------------------

            result = process_raw(
                path=str(raw_path),
                profile=profile,
                output_dir=str(case_work_dir),
            )

            # ---------------------------------------------------------
            # Validate generated DICOM
            # ---------------------------------------------------------

            dicom_path = Path(
                result["dicom_path"]
            )

            dicom_info = validate_dicom(
                dicom_path=dicom_path,
                expected_width=width,
                expected_height=height,
            )

            # ---------------------------------------------------------
            # Metrics
            # ---------------------------------------------------------

            before = result["before"]
            after = result["after"]

            snr_change = (
                float(after["snr_db"])
                - float(before["snr_db"])
            )

            noise_change = (
                float(after["noise"])
                - float(before["noise"])
            )

            contrast_change = (
                float(after["contrast"])
                - float(before["contrast"])
            )

            entropy_change = (
                float(after["entropy"])
                - float(before["entropy"])
            )

            # ---------------------------------------------------------
            # Quality gate
            # ---------------------------------------------------------

            quality_gate = _get_quality_gate(
                result=result,
                before=before,
                after=after,
            )

            candidate_search = result.get("candidate_search", {})

            selected_candidate_index = candidate_search.get(
            "selected_candidate_index"
            )

            candidate_count = int(
            candidate_search.get("candidate_count", 0)
            )

            safe_candidate_count = int(
            candidate_search.get("safe_candidate_count", 0)
            )

            elapsed = (
                time.perf_counter()
                - case_start
            )

            # ---------------------------------------------------------
            # Record
            # ---------------------------------------------------------

            record = {
                "case": case_name,

                "source_file": source_path.name,

                "source_dtype": source_dtype,

                "width": width,

                "height": height,

                "source_width": source_width,

                "source_height": source_height,

                "anatomy": result["anatomy"],

                "confidence": float(
                    result["confidence"]
                ),

                "view": result["view"],

                "model": result["model"],

                "snr_before_db": float(
                    before["snr_db"]
                ),

                "snr_after_db": float(
                    after["snr_db"]
                ),

                "snr_change_db": float(
                    snr_change
                ),

                "noise_before": float(
                    before["noise"]
                ),

                "noise_after": float(
                    after["noise"]
                ),

                "noise_change": float(
                    noise_change
                ),

                "contrast_before": float(
                    before["contrast"]
                ),

                "contrast_after": float(
                    after["contrast"]
                ),

                "contrast_change": float(
                    contrast_change
                ),

                "dynamic_range_before": float(
                    before["dynamic_range"]
                ),

                "dynamic_range_after": float(
                    after["dynamic_range"]
                ),

                "entropy_before": float(
                    before["entropy"]
                ),

                "entropy_after": float(
                    after["entropy"]
                ),

                "entropy_change": float(
                    entropy_change
                ),

                "parameters": result[
                    "parameters"
                ],

                "candidate_search": candidate_search,
                "selected_candidate_index": selected_candidate_index,
                "candidate_count": candidate_count,
                "safe_candidate_count": safe_candidate_count,

                "quality_gate": quality_gate,

                "dicom_valid": True,

                "dicom_modality": dicom_info[
                    "modality"
                ],

                "dicom_rows": dicom_info[
                    "rows"
                ],

                "dicom_columns": dicom_info[
                    "columns"
                ],

                "processing_time_seconds": float(
                    elapsed
                ),
            }

            results.append(record)

            print(
                f"OK | "
                f"SNR {before['snr_db']:.2f}"
                f"→{after['snr_db']:.2f} dB | "
                f"Δ {snr_change:+.2f} dB | "
                f"GATE "
                f"{'ACCEPT' if quality_gate['accepted'] else 'REJECT'} | "
                f"CANDIDATE "
                f"{selected_candidate_index if selected_candidate_index is not None else 'FALLBACK'} | "
                f"{elapsed:.2f}s"
            )

        except Exception as exc:

            elapsed = (
                time.perf_counter()
                - case_start
            )

            failure = {
                "case": case_name,

                "source_file": source_path.name,

                "error": str(exc),

                "error_type": type(exc).__name__,

                "processing_time_seconds": float(
                    elapsed
                ),
            }

            failures.append(failure)

            print(
                f"FAILED | "
                f"{type(exc).__name__}: {exc}"
            )

        finally:

            # ---------------------------------------------------------
            # Remove temporary case files.
            #
            # This prevents 138 large DICOM files from accumulating.
            # ---------------------------------------------------------

            if case_work_dir.exists():
                shutil.rmtree(
                    case_work_dir,
                    ignore_errors=True,
                )

    total_elapsed = (
        time.perf_counter()
        - total_start
    )

    # -----------------------------------------------------------------
    # Aggregate statistics
    # -----------------------------------------------------------------

    successful = len(results)
    failed = len(failures)

    def values(key):
        return np.array(
            [
                float(record[key])
                for record in results
            ],
            dtype=np.float64,
        )

        aggregate = {}

    if results:
        # Existing aggregate calculations
        snr_changes = values("snr_change_db")
        noise_changes = values("noise_change")
        contrast_changes = values("contrast_change")
        entropy_changes = values("entropy_change")
        snr_before = values("snr_before_db")
        snr_after = values("snr_after_db")

        gate_accepted = sum(
            bool(record["quality_gate"]["accepted"])
            for record in results
        )

        gate_rejected = successful - gate_accepted

        safety_passed = sum(
            bool(record["quality_gate"]["safety_passed"])
            for record in results
        )

        safety_failed = successful - safety_passed

        aggregate = {
            "snr_before": {
                "mean_db": float(np.mean(snr_before)),
                "median_db": float(np.median(snr_before)),
                "min_db": float(np.min(snr_before)),
                "max_db": float(np.max(snr_before)),
            },
            "snr_after": {
                "mean_db": float(np.mean(snr_after)),
                "median_db": float(np.median(snr_after)),
                "min_db": float(np.min(snr_after)),
                "max_db": float(np.max(snr_after)),
            },
            "snr_change": {
                "mean_db": float(np.mean(snr_changes)),
                "median_db": float(np.median(snr_changes)),
                "min_db": float(np.min(snr_changes)),
                "max_db": float(np.max(snr_changes)),
                "positive_cases": int(np.sum(snr_changes > 0)),
                "negative_cases": int(np.sum(snr_changes < 0)),
                "unchanged_cases": int(np.sum(snr_changes == 0)),
            },
            "noise_change": {
                "mean": float(np.mean(noise_changes)),
                "median": float(np.median(noise_changes)),
                "min": float(np.min(noise_changes)),
                "max": float(np.max(noise_changes)),
                "reduced_noise_cases": int(np.sum(noise_changes < 0)),
                "increased_noise_cases": int(np.sum(noise_changes > 0)),
            },
            "contrast_change": {
                "mean": float(np.mean(contrast_changes)),
                "median": float(np.median(contrast_changes)),
            },
            "entropy_change": {
                "mean": float(np.mean(entropy_changes)),
                "median": float(np.median(entropy_changes)),
            },
            "quality_gate": {
                "accepted_cases": int(gate_accepted),
                "rejected_cases": int(gate_rejected),
                "safety_passed_cases": int(safety_passed),
                "safety_failed_cases": int(safety_failed),
                "acceptance_rate": (
                    float(gate_accepted / successful)
                    if successful
                    else 0.0
                ),
            },
        }

        # -------------------------------------------------------------
        # Candidate-search distribution
        # -------------------------------------------------------------

        candidate_counts = {}

        for record in results:
            candidate_index = record.get("selected_candidate_index")

            if candidate_index is None:
                key = "fallback"
            else:
                key = str(candidate_index)

            candidate_counts[key] = (
                candidate_counts.get(key, 0) + 1
            )

        aggregate["candidate_search"] = {
            "candidate_selection_counts": candidate_counts,
            "fallback_cases": int(
                candidate_counts.get("fallback", 0)
            ),
        }

    if results:

        snr_changes = values(
            "snr_change_db"
        )

        noise_changes = values(
            "noise_change"
        )

        contrast_changes = values(
            "contrast_change"
        )

        entropy_changes = values(
            "entropy_change"
        )

        snr_before = values(
            "snr_before_db"
        )

        snr_after = values(
            "snr_after_db"
        )

        gate_accepted = sum(
            bool(
                record["quality_gate"]["accepted"]
            )
            for record in results
        )

        gate_rejected = (
            successful - gate_accepted
        )

        safety_passed = sum(
            bool(
                record["quality_gate"]["safety_passed"]
            )
            for record in results
        )

        safety_failed = (
            successful - safety_passed
        )

        aggregate = {

            "snr_before": {
                "mean_db": float(
                    np.mean(snr_before)
                ),
                "median_db": float(
                    np.median(snr_before)
                ),
                "min_db": float(
                    np.min(snr_before)
                ),
                "max_db": float(
                    np.max(snr_before)
                ),
            },

            "snr_after": {
                "mean_db": float(
                    np.mean(snr_after)
                ),
                "median_db": float(
                    np.median(snr_after)
                ),
                "min_db": float(
                    np.min(snr_after)
                ),
                "max_db": float(
                    np.max(snr_after)
                ),
            },

            "snr_change": {
                "mean_db": float(
                    np.mean(snr_changes)
                ),
                "median_db": float(
                    np.median(snr_changes)
                ),
                "min_db": float(
                    np.min(snr_changes)
                ),
                "max_db": float(
                    np.max(snr_changes)
                ),
                "positive_cases": int(
                    np.sum(snr_changes > 0)
                ),
                "negative_cases": int(
                    np.sum(snr_changes < 0)
                ),
                "unchanged_cases": int(
                    np.sum(snr_changes == 0)
                ),
            },

            "noise_change": {
                "mean": float(
                    np.mean(noise_changes)
                ),
                "median": float(
                    np.median(noise_changes)
                ),
                "min": float(
                    np.min(noise_changes)
                ),
                "max": float(
                    np.max(noise_changes)
                ),
                "reduced_noise_cases": int(
                    np.sum(noise_changes < 0)
                ),
                "increased_noise_cases": int(
                    np.sum(noise_changes > 0)
                ),
            },

            "contrast_change": {
                "mean": float(
                    np.mean(contrast_changes)
                ),
                "median": float(
                    np.median(contrast_changes)
                ),
            },

            "entropy_change": {
                "mean": float(
                    np.mean(entropy_changes)
                ),
                "median": float(
                    np.median(entropy_changes)
                ),
            },

            "quality_gate": {
                "accepted_cases": int(
                    gate_accepted
                ),
                "rejected_cases": int(
                    gate_rejected
                ),
                "safety_passed_cases": int(
                    safety_passed
                ),
                "safety_failed_cases": int(
                    safety_failed
                ),
                "acceptance_rate": float(
                    gate_accepted / successful
                ) if successful else 0.0,
            },
        }

    # -----------------------------------------------------------------
    # Save CSV
    # -----------------------------------------------------------------

    csv_fields = [
        "case",
        "source_file",
        "source_dtype",
        "width",
        "height",
        "source_width",
        "source_height",
        "anatomy",
        "confidence",
        "view",
        "model",
        "selected_candidate_index",
        "candidate_count",
        "safe_candidate_count",

        "snr_before_db",
        "snr_after_db",
        "snr_change_db",

        "noise_before",
        "noise_after",
        "noise_change",

        "contrast_before",
        "contrast_after",
        "contrast_change",

        "dynamic_range_before",
        "dynamic_range_after",

        "entropy_before",
        "entropy_after",
        "entropy_change",

        "quality_gate_accepted",
        "quality_gate_blend_ratio",
        "quality_gate_score_before",
        "quality_gate_score_enhanced",
        "quality_gate_final_score",
        "quality_gate_reason",
        "quality_gate_snr_change_db",
        "quality_gate_noise_change",
        "quality_gate_contrast_change",
        "quality_gate_entropy_change",
        "quality_gate_safety_passed",

        "dicom_valid",
        "dicom_modality",
        "dicom_rows",
        "dicom_columns",

        "processing_time_seconds",
    ]

    with SUMMARY_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=csv_fields,
        )

        writer.writeheader()

        for record in results:

            gate = record["quality_gate"]

            csv_record = {
                field: record.get(
                    field,
                    "",
                )
                for field in csv_fields
            }

            csv_record.update(
                {
                    "quality_gate_accepted":
                        gate["accepted"],

                    "quality_gate_blend_ratio":
                        gate["blend_ratio"],

                    "quality_gate_score_before":
                        gate["score_before"],

                    "quality_gate_score_enhanced":
                        gate["score_enhanced"],

                    "quality_gate_final_score":
                        gate["final_score"],

                    "quality_gate_reason":
                        gate["reason"],

                    "quality_gate_snr_change_db":
                        gate["snr_change_db"],

                    "quality_gate_noise_change":
                        gate["noise_change"],

                    "quality_gate_contrast_change":
                        gate["contrast_change"],

                    "quality_gate_entropy_change":
                        gate["entropy_change"],

                    "quality_gate_safety_passed":
                        gate["safety_passed"],
                }
            )

            writer.writerow(csv_record)

    # -----------------------------------------------------------------
    # Save JSON summary
    # -----------------------------------------------------------------

    summary = {
        "dataset": "NLM-MontgomeryCXRSet",

        "dataset_directory": str(
            DATASET_DIR
        ),

        "total_images": len(image_files),

        "successful": successful,

        "failed": failed,

        "total_processing_time_seconds": float(
            total_elapsed
        ),

        "average_processing_time_seconds": (
            float(
                total_elapsed / successful
            )
            if successful
            else 0.0
        ),

        "aggregate_metrics": aggregate,

        "results": results,
    }

    SUMMARY_JSON.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    # -----------------------------------------------------------------
    # Save failures
    # -----------------------------------------------------------------

    FAILURES_JSON.write_text(
        json.dumps(
            failures,
            indent=2,
        ),
        encoding="utf-8",
    )

    # -----------------------------------------------------------------
    # Final report
    # -----------------------------------------------------------------

    print()
    print("=" * 80)
    print("BATCH VALIDATION COMPLETE")
    print("=" * 80)

    print(
        f"Total images       : {len(image_files)}"
    )

    print(
        f"Successful         : {successful}"
    )

    print(
        f"Failed             : {failed}"
    )

    print(
        f"Total time         : "
        f"{total_elapsed:.2f} seconds"
    )

    if successful:

        print()
        print(
            f"Average/image     : "
            f"{total_elapsed / successful:.2f} seconds"
        )

        print()
        print("SNR")

        print(
            f"Mean before        : "
            f"{aggregate['snr_before']['mean_db']:.2f} dB"
        )

        print(
            f"Mean after         : "
            f"{aggregate['snr_after']['mean_db']:.2f} dB"
        )

        print(
            f"Mean improvement   : "
            f"{aggregate['snr_change']['mean_db']:+.2f} dB"
        )

        print(
            f"Median improvement : "
            f"{aggregate['snr_change']['median_db']:+.2f} dB"
        )

        print(
            f"Positive cases     : "
            f"{aggregate['snr_change']['positive_cases']}"
        )

        print(
            f"Negative cases     : "
            f"{aggregate['snr_change']['negative_cases']}"
        )

        print()
        print("NOISE")

        print()
        print("CANDIDATE SEARCH")

        candidate_counts = aggregate["candidate_search"][
            "candidate_selection_counts"
        ]

        for candidate_index in sorted(
            (
                key
                for key in candidate_counts
                if key != "fallback"
            ),
            key=int,
        ):
            print(
                f"Candidate {candidate_index:<10}: "
                f"{candidate_counts[candidate_index]}"
            )

        print(
            f"Fallback{'':<10}: "
            f"{candidate_counts.get('fallback', 0)}"
        )

        print(
            f"Reduced noise      : "
            f"{aggregate['noise_change']['reduced_noise_cases']}"
        )

        print(
            f"Increased noise    : "
            f"{aggregate['noise_change']['increased_noise_cases']}"
        )

        print()
        print("QUALITY GATE")

        print(
            f"Accepted cases     : "
            f"{aggregate['quality_gate']['accepted_cases']}"
        )

        print(
            f"Rejected cases     : "
            f"{aggregate['quality_gate']['rejected_cases']}"
        )

        print(
            f"Safety passed      : "
            f"{aggregate['quality_gate']['safety_passed_cases']}"
        )

        print(
            f"Safety failed      : "
            f"{aggregate['quality_gate']['safety_failed_cases']}"
        )

        print(
            f"Acceptance rate    : "
            f"{aggregate['quality_gate']['acceptance_rate'] * 100:.2f}%"
        )

    print()

    print(
        f"CSV                : {SUMMARY_CSV}"
    )

    print(
        f"JSON               : {SUMMARY_JSON}"
    )

    print(
        f"Failures           : {FAILURES_JSON}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()