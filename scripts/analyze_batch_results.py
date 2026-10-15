from pathlib import Path
import csv
import json
import statistics
import sys

import numpy as np


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

BATCH_DIR = (
    BASE_DIR
    / "data"
    / "demo"
    / "real"
    / "batch_validation"
)

CSV_PATH = BATCH_DIR / "batch_summary.csv"
JSON_PATH = BATCH_DIR / "batch_summary.json"

REPORT_PATH = BATCH_DIR / "batch_analysis_report.txt"
ANALYSIS_JSON_PATH = BATCH_DIR / "batch_analysis.json"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def mean(values):
    return float(np.mean(values)) if values else 0.0


def median(values):
    return float(np.median(values)) if values else 0.0


def percentile(values, p):
    return float(np.percentile(values, p)) if values else 0.0


def fmt(value, digits=2):
    return f"{value:.{digits}f}"


def load_csv():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV not found:\n{CSV_PATH}"
        )

    with CSV_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def load_json():
    if not JSON_PATH.exists():
        raise FileNotFoundError(
            f"JSON not found:\n{JSON_PATH}"
        )

    return json.loads(
        JSON_PATH.read_text(
            encoding="utf-8"
        )
    )


def float_column(rows, name):
    return [
        float(row[name])
        for row in rows
    ]


# ---------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------

def main():

    print("=" * 80)
    print("MONTGOMERY DATASET — BATCH RESULT ANALYSIS")
    print("=" * 80)

    rows = load_csv()
    json_data = load_json()

    print()
    print(f"CSV records       : {len(rows)}")
    print(
        f"JSON results      : "
        f"{len(json_data.get('results', []))}"
    )

    if not rows:
        raise RuntimeError(
            "No batch results found."
        )

    # -----------------------------------------------------------------
    # Numeric columns
    # -----------------------------------------------------------------

    snr_before = float_column(
        rows,
        "snr_before_db",
    )

    snr_after = float_column(
        rows,
        "snr_after_db",
    )

    snr_change = float_column(
        rows,
        "snr_change_db",
    )

    noise_before = float_column(
        rows,
        "noise_before",
    )

    noise_after = float_column(
        rows,
        "noise_after",
    )

    noise_change = float_column(
        rows,
        "noise_change",
    )

    contrast_before = float_column(
        rows,
        "contrast_before",
    )

    contrast_after = float_column(
        rows,
        "contrast_after",
    )

    contrast_change = float_column(
        rows,
        "contrast_change",
    )

    entropy_before = float_column(
        rows,
        "entropy_before",
    )

    entropy_after = float_column(
        rows,
        "entropy_after",
    )

    entropy_change = float_column(
        rows,
        "entropy_change",
    )

    processing_time = float_column(
        rows,
        "processing_time_seconds",
    )

    # -----------------------------------------------------------------
    # Core statistics
    # -----------------------------------------------------------------

    positive_snr = [
        value
        for value in snr_change
        if value > 0
    ]

    negative_snr = [
        value
        for value in snr_change
        if value < 0
    ]

    zero_snr = [
        value
        for value in snr_change
        if value == 0
    ]

    reduced_noise = [
        value
        for value in noise_change
        if value < 0
    ]

    increased_noise = [
        value
        for value in noise_change
        if value > 0
    ]

    # -----------------------------------------------------------------
    # Best / worst cases
    # -----------------------------------------------------------------

    sorted_by_snr = sorted(
        rows,
        key=lambda row: float(
            row["snr_change_db"]
        ),
        reverse=True,
    )

    best_cases = sorted_by_snr[:10]
    worst_cases = sorted_by_snr[-10:][::-1]

    # -----------------------------------------------------------------
    # Group analysis based on filename suffix.
    #
    # We intentionally call these "_0" and "_1" groups rather than
    # assigning a clinical meaning to the suffix.
    # -----------------------------------------------------------------

    group_0 = [
        row
        for row in rows
        if row["case"].endswith("_0")
    ]

    group_1 = [
        row
        for row in rows
        if row["case"].endswith("_1")
    ]

    def group_stats(group):

        if not group:
            return {
                "count": 0,
                "mean_snr_before": 0.0,
                "mean_snr_after": 0.0,
                "mean_snr_change": 0.0,
                "median_snr_change": 0.0,
                "positive_snr_cases": 0,
                "negative_snr_cases": 0,
                "mean_noise_change": 0.0,
                "noise_reduced_cases": 0,
                "noise_increased_cases": 0,
            }

        group_snr_before = [
            float(r["snr_before_db"])
            for r in group
        ]

        group_snr_after = [
            float(r["snr_after_db"])
            for r in group
        ]

        group_snr_change = [
            float(r["snr_change_db"])
            for r in group
        ]

        group_noise_change = [
            float(r["noise_change"])
            for r in group
        ]

        return {
            "count": len(group),
            "mean_snr_before": mean(
                group_snr_before
            ),
            "mean_snr_after": mean(
                group_snr_after
            ),
            "mean_snr_change": mean(
                group_snr_change
            ),
            "median_snr_change": median(
                group_snr_change
            ),
            "positive_snr_cases": sum(
                x > 0
                for x in group_snr_change
            ),
            "negative_snr_cases": sum(
                x < 0
                for x in group_snr_change
            ),
            "mean_noise_change": mean(
                group_noise_change
            ),
            "noise_reduced_cases": sum(
                x < 0
                for x in group_noise_change
            ),
            "noise_increased_cases": sum(
                x > 0
                for x in group_noise_change
            ),
        }

    group_0_stats = group_stats(
        group_0
    )

    group_1_stats = group_stats(
        group_1
    )

    # -----------------------------------------------------------------
    # Correlations
    # -----------------------------------------------------------------

    snr_noise_corr = float(
        np.corrcoef(
            snr_change,
            noise_change,
        )[0, 1]
    )

    snr_before_change_corr = float(
        np.corrcoef(
            snr_before,
            snr_change,
        )[0, 1]
    )

    # -----------------------------------------------------------------
    # Parameter usage from JSON
    # -----------------------------------------------------------------

    json_results = json_data.get(
        "results",
        [],
    )

    parameter_profiles = {}

    for result in json_results:

        parameters = result.get(
            "parameters",
            {},
        )

        key = (
            str(
                parameters.get(
                    "denoise_strength",
                    parameters.get(
                        "bilateral_sigma_color",
                        "",
                    ),
                )
            ),
            str(
                parameters.get(
                    "clahe_clip_limit",
                    "",
                )
            ),
            str(
                parameters.get(
                    "gamma",
                    "",
                )
            ),
            str(
                parameters.get(
                    "sharpen_amount",
                    "",
                )
            ),
        )

        parameter_profiles[key] = (
            parameter_profiles.get(
                key,
                0,
            )
            + 1
        )

    # -----------------------------------------------------------------
    # Aggregate analysis object
    # -----------------------------------------------------------------

    analysis = {
        "dataset": json_data.get(
            "dataset",
            "NLM-MontgomeryCXRSet",
        ),

        "total_cases": len(rows),

        "snr": {
            "before_mean_db": mean(
                snr_before
            ),
            "before_median_db": median(
                snr_before
            ),
            "after_mean_db": mean(
                snr_after
            ),
            "after_median_db": median(
                snr_after
            ),
            "change_mean_db": mean(
                snr_change
            ),
            "change_median_db": median(
                snr_change
            ),
            "change_min_db": min(
                snr_change
            ),
            "change_max_db": max(
                snr_change
            ),
            "p10_change_db": percentile(
                snr_change,
                10,
            ),
            "p90_change_db": percentile(
                snr_change,
                90,
            ),
            "positive_cases": len(
                positive_snr
            ),
            "negative_cases": len(
                negative_snr
            ),
            "unchanged_cases": len(
                zero_snr
            ),
        },

        "noise": {
            "before_mean": mean(
                noise_before
            ),
            "after_mean": mean(
                noise_after
            ),
            "change_mean": mean(
                noise_change
            ),
            "change_median": median(
                noise_change
            ),
            "reduced_cases": len(
                reduced_noise
            ),
            "increased_cases": len(
                increased_noise
            ),
        },

        "contrast": {
            "before_mean": mean(
                contrast_before
            ),
            "after_mean": mean(
                contrast_after
            ),
            "change_mean": mean(
                contrast_change
            ),
            "change_median": median(
                contrast_change
            ),
        },

        "entropy": {
            "before_mean": mean(
                entropy_before
            ),
            "after_mean": mean(
                entropy_after
            ),
            "change_mean": mean(
                entropy_change
            ),
            "change_median": median(
                entropy_change
            ),
        },

        "processing": {
            "mean_seconds": mean(
                processing_time
            ),
            "median_seconds": median(
                processing_time
            ),
            "min_seconds": min(
                processing_time
            ),
            "max_seconds": max(
                processing_time
            ),
            "total_seconds": sum(
                processing_time
            ),
        },

        "groups": {
            "_0": group_0_stats,
            "_1": group_1_stats,
        },

        "correlations": {
            "snr_change_vs_noise_change":
                snr_noise_corr,
            "initial_snr_vs_snr_change":
                snr_before_change_corr,
        },

        "parameter_profiles": {
            str(key): count
            for key, count
            in parameter_profiles.items()
        },

        "best_cases": [
            {
                "case": row["case"],
                "snr_change_db": float(
                    row["snr_change_db"]
                ),
                "noise_change": float(
                    row["noise_change"]
                ),
            }
            for row in best_cases
        ],

        "worst_cases": [
            {
                "case": row["case"],
                "snr_change_db": float(
                    row["snr_change_db"]
                ),
                "noise_change": float(
                    row["noise_change"]
                ),
            }
            for row in worst_cases
        ],
    }

    # -----------------------------------------------------------------
    # Save JSON
    # -----------------------------------------------------------------

    ANALYSIS_JSON_PATH.write_text(
        json.dumps(
            analysis,
            indent=2,
        ),
        encoding="utf-8",
    )

    # -----------------------------------------------------------------
    # Build human-readable report
    # -----------------------------------------------------------------

    report = []

    report.append(
        "=" * 80
    )
    report.append(
        "MONTGOMERY DATASET — ENGINEERING ANALYSIS"
    )
    report.append(
        "=" * 80
    )

    report.append("")
    report.append(
        f"Total cases: {len(rows)}"
    )

    # SNR
    report.append("")
    report.append(
        "-" * 80
    )
    report.append(
        "1. SNR ANALYSIS"
    )
    report.append(
        "-" * 80
    )

    report.append(
        f"Mean before       : "
        f"{mean(snr_before):.3f} dB"
    )

    report.append(
        f"Mean after        : "
        f"{mean(snr_after):.3f} dB"
    )

    report.append(
        f"Mean improvement  : "
        f"{mean(snr_change):+.3f} dB"
    )

    report.append(
        f"Median improvement: "
        f"{median(snr_change):+.3f} dB"
    )

    report.append(
        f"10th percentile   : "
        f"{percentile(snr_change, 10):+.3f} dB"
    )

    report.append(
        f"90th percentile   : "
        f"{percentile(snr_change, 90):+.3f} dB"
    )

    report.append(
        f"Minimum change    : "
        f"{min(snr_change):+.3f} dB"
    )

    report.append(
        f"Maximum change    : "
        f"{max(snr_change):+.3f} dB"
    )

    report.append(
        f"Positive cases    : "
        f"{len(positive_snr)}"
    )

    report.append(
        f"Negative cases    : "
        f"{len(negative_snr)}"
    )

    # Noise
    report.append("")
    report.append(
        "-" * 80
    )
    report.append(
        "2. NOISE ANALYSIS"
    )
    report.append(
        "-" * 80
    )

    report.append(
        f"Mean noise before : "
        f"{mean(noise_before):.6f}"
    )

    report.append(
        f"Mean noise after  : "
        f"{mean(noise_after):.6f}"
    )

    report.append(
        f"Mean noise change : "
        f"{mean(noise_change):+.6f}"
    )

    report.append(
        f"Noise reduced     : "
        f"{len(reduced_noise)}"
    )

    report.append(
        f"Noise increased   : "
        f"{len(increased_noise)}"
    )

    # Contrast
    report.append("")
    report.append(
        "-" * 80
    )
    report.append(
        "3. CONTRAST ANALYSIS"
    )
    report.append(
        "-" * 80
    )

    report.append(
        f"Mean before       : "
        f"{mean(contrast_before):.6f}"
    )

    report.append(
        f"Mean after        : "
        f"{mean(contrast_after):.6f}"
    )

    report.append(
        f"Mean change       : "
        f"{mean(contrast_change):+.6f}"
    )

    # Entropy
    report.append("")
    report.append(
        "-" * 80
    )
    report.append(
        "4. ENTROPY ANALYSIS"
    )
    report.append(
        "-" * 80
    )

    report.append(
        f"Mean before       : "
        f"{mean(entropy_before):.4f}"
    )

    report.append(
        f"Mean after        : "
        f"{mean(entropy_after):.4f}"
    )

    report.append(
        f"Mean change       : "
        f"{mean(entropy_change):+.4f}"
    )

    # Processing
    report.append("")
    report.append(
        "-" * 80
    )
    report.append(
        "5. PROCESSING PERFORMANCE"
    )
    report.append(
        "-" * 80
    )

    report.append(
        f"Mean time         : "
        f"{mean(processing_time):.3f} sec"
    )

    report.append(
        f"Median time       : "
        f"{median(processing_time):.3f} sec"
    )

    report.append(
        f"Fastest case      : "
        f"{min(processing_time):.3f} sec"
    )

    report.append(
        f"Slowest case      : "
        f"{max(processing_time):.3f} sec"
    )

    # Groups
    report.append("")
    report.append(
        "-" * 80
    )
    report.append(
        "6. DATASET GROUP ANALYSIS"
    )
    report.append(
        "-" * 80
    )

    for group_name, stats in [
        ("_0", group_0_stats),
        ("_1", group_1_stats),
    ]:

        report.append("")
        report.append(
            f"GROUP {group_name}"
        )

        report.append(
            f"Cases             : "
            f"{stats['count']}"
        )

        report.append(
            f"Mean SNR before   : "
            f"{stats['mean_snr_before']:.3f} dB"
        )

        report.append(
            f"Mean SNR after    : "
            f"{stats['mean_snr_after']:.3f} dB"
        )

        report.append(
            f"Mean SNR change   : "
            f"{stats['mean_snr_change']:+.3f} dB"
        )

        report.append(
            f"Median SNR change : "
            f"{stats['median_snr_change']:+.3f} dB"
        )

        report.append(
            f"Positive SNR      : "
            f"{stats['positive_snr_cases']}"
        )

        report.append(
            f"Negative SNR      : "
            f"{stats['negative_snr_cases']}"
        )

        report.append(
            f"Noise reduced     : "
            f"{stats['noise_reduced_cases']}"
        )

        report.append(
            f"Noise increased   : "
            f"{stats['noise_increased_cases']}"
        )

    # Correlation
    report.append("")
    report.append(
        "-" * 80
    )
    report.append(
        "7. RELATIONSHIPS"
    )
    report.append(
        "-" * 80
    )

    report.append(
        "SNR change vs noise change correlation: "
        f"{snr_noise_corr:+.4f}"
    )

    report.append(
        "Initial SNR vs SNR improvement correlation: "
        f"{snr_before_change_corr:+.4f}"
    )

    # Best
    report.append("")
    report.append(
        "-" * 80
    )
    report.append(
        "8. TOP 10 SNR IMPROVEMENTS"
    )
    report.append(
        "-" * 80
    )

    for index, row in enumerate(
        best_cases,
        start=1,
    ):

        report.append(
            f"{index:02d}. "
            f"{row['case']:18s} "
            f"SNR Δ "
            f"{float(row['snr_change_db']):+.3f} dB | "
            f"Noise Δ "
            f"{float(row['noise_change']):+.6f}"
        )

    # Worst
    report.append("")
    report.append(
        "-" * 80
    )
    report.append(
        "9. LOWEST 10 SNR IMPROVEMENTS"
    )
    report.append(
        "-" * 80
    )

    for index, row in enumerate(
        worst_cases,
        start=1,
    ):

        report.append(
            f"{index:02d}. "
            f"{row['case']:18s} "
            f"SNR Δ "
            f"{float(row['snr_change_db']):+.3f} dB | "
            f"Noise Δ "
            f"{float(row['noise_change']):+.6f}"
        )

    # Parameter profiles
    report.append("")
    report.append(
        "-" * 80
    )
    report.append(
        "10. PARAMETER PROFILES"
    )
    report.append(
        "-" * 80
    )

    sorted_profiles = sorted(
        parameter_profiles.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    for profile, count in sorted_profiles:

        report.append(
            f"{count:03d} cases | "
            f"{profile}"
        )

    report.append("")
    report.append(
        "=" * 80
    )
    report.append(
        "ANALYSIS COMPLETE"
    )
    report.append(
        "=" * 80
    )

    report_text = "\n".join(report)

    REPORT_PATH.write_text(
        report_text,
        encoding="utf-8",
    )

    # -----------------------------------------------------------------
    # Print report to terminal
    # -----------------------------------------------------------------

    print()
    print(report_text)

    print()
    print(
        f"Report JSON : {ANALYSIS_JSON_PATH}"
    )
    print(
        f"Report TXT  : {REPORT_PATH}"
    )


if __name__ == "__main__":
    main()