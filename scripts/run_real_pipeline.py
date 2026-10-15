from pathlib import Path
import sys

# Allow imports from the project root when running this script directly.
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from backend.ingestion.raw_reader import RawProfile
from backend.pipeline.processor import process_raw


RAW_PATH = (
    BASE_DIR
    / "data"
    / "demo"
    / "real"
    / "MCUCXR_0001_0_16bit.raw"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "demo"
    / "real"
    / "output"
)


def main() -> None:
    print("=" * 70)
    print("REAL RADIOGRAPH - END-TO-END PIPELINE TEST")
    print("=" * 70)

    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"RAW file not found:\n{RAW_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    profile = RawProfile(
        width=4892,
        height=4020,
        bit_depth=16,
        signed=False,
        endian="little",
    )

    print()
    print("INPUT")
    print(f"RAW file        : {RAW_PATH.name}")
    print(f"Width           : {profile.width}")
    print(f"Height          : {profile.height}")
    print(f"Bit depth       : {profile.bit_depth}")
    print(f"Signed          : {profile.signed}")
    print(f"Endian          : {profile.endian}")

    print()
    print("PROCESSING...")

    result = process_raw(
        path=str(RAW_PATH),
        profile=profile,
        output_dir=str(OUTPUT_DIR),
    )

    # ------------------------------------------------------------
    # ANATOMY
    # ------------------------------------------------------------

    print()
    print("ANATOMY")
    print(f"Detected anatomy: {result['anatomy']}")
    print(f"Confidence      : {result['confidence']}")
    print(f"View            : {result['view']}")
    print(f"Model           : {result['model']}")

    # ------------------------------------------------------------
    # QUALITY BEFORE
    # ------------------------------------------------------------

    print()
    print("QUALITY - BEFORE")

    for key, value in result["before"].items():
        print(f"{key:28}: {value}")

    # ------------------------------------------------------------
    # 9J ADAPTIVE CANDIDATE SEARCH
    # ------------------------------------------------------------

    candidate_search = result.get("candidate_search", {})

    print()
    print("=" * 70)
    print("9J ADAPTIVE CANDIDATE SEARCH")
    print("=" * 70)

    print(
        f"Candidates evaluated : "
        f"{candidate_search.get('candidate_count', result.get('candidate_count', 'N/A'))}"
    )

    print(
        f"Safe candidates      : "
        f"{candidate_search.get('safe_candidate_count', result.get('safe_candidate_count', 'N/A'))}"
    )

    selected_index = candidate_search.get(
        "selected_candidate_index",
        result.get("selected_candidate_index", "N/A"),
    )

    print(f"Selected candidate   : #{selected_index}")

    baseline_selected = candidate_search.get(
        "baseline_selected",
        False,
    )

    print(f"Baseline selected     : {baseline_selected}")

    # ------------------------------------------------------------
    # DETAILED CANDIDATE DIAGNOSTICS
    # ------------------------------------------------------------

    candidates = result.get("candidates", [])

    print()
    print("CANDIDATE RESULTS")
    print("-" * 70)

    if candidates:
        for candidate in candidates:
            index = candidate.get("index", "?")

            selected = index == selected_index
            marker = "SELECTED" if selected else ""

            parameters = candidate.get("parameters", {})

            accepted = candidate.get("accepted", False)
            final_score = candidate.get("final_score")
            blend_ratio = candidate.get("blend_ratio")
            reason = candidate.get("reason", "N/A")

            snr_change = candidate.get("snr_change_db")
            noise_change = candidate.get("noise_change")
            contrast_change = candidate.get("contrast_change")
            entropy_change = candidate.get("entropy_change")
            safety_passed = candidate.get("safety_passed")

            score_text = (
                f"{final_score:.6f}"
                if isinstance(final_score, (int, float))
                else "N/A"
            )

            blend_text = (
                f"{blend_ratio:.3f}"
                if isinstance(blend_ratio, (int, float))
                else "N/A"
            )

            print(
                f"#{index:<3} "
                f"{marker:<10} "
                f"accepted={str(accepted):<5} "
                f"score={score_text:<10} "
                f"blend={blend_text}"
            )

            print(
                f"     bilateral={parameters.get('bilateral_sigma_color', 'N/A')}, "
                f"clahe={parameters.get('clahe_clip_limit', 'N/A')}, "
                f"gamma={parameters.get('gamma', 'N/A')}, "
                f"sharpen={parameters.get('sharpen_amount', 'N/A')}"
            )

            print(f"     reason={reason}")

            snr_text = (
                f"{snr_change:+.6f}"
                if isinstance(snr_change, (int, float))
                else "N/A"
            )

            noise_text = (
                f"{noise_change:+.6f}"
                if isinstance(noise_change, (int, float))
                else "N/A"
            )

            contrast_text = (
                f"{contrast_change:+.6f}"
                if isinstance(contrast_change, (int, float))
                else "N/A"
            )

            entropy_text = (
                f"{entropy_change:+.6f}"
                if isinstance(entropy_change, (int, float))
                else "N/A"
            )

            print(
                f"     SNR Δ={snr_text}, "
                f"noise Δ={noise_text}, "
                f"contrast Δ={contrast_text}, "
                f"entropy Δ={entropy_text}"
            )

            print(
                f"     safety_passed={safety_passed}"
            )

    else:
        print(
            "No candidate diagnostics returned by process_raw."
        )

    # ------------------------------------------------------------
    # SELECTED PARAMETERS
    # ------------------------------------------------------------

    print()
    print("SELECTED ADAPTIVE PARAMETERS")

    for key, value in result["parameters"].items():
        print(f"{key:28}: {value}")

    # ------------------------------------------------------------
    # QUALITY AFTER
    # ------------------------------------------------------------

    print()
    print("QUALITY - AFTER")

    for key, value in result["after"].items():
        print(f"{key:28}: {value}")

    # ------------------------------------------------------------
    # IMPROVEMENT
    # ------------------------------------------------------------

    before_snr = result["before"]["snr_db"]
    after_snr = result["after"]["snr_db"]

    before_noise = result["before"]["noise"]
    after_noise = result["after"]["noise"]

    print()
    print("IMPROVEMENT")
    print(f"SNR change      : {after_snr - before_snr:+.2f} dB")
    print(f"Noise change    : {after_noise - before_noise:+.6f}")

    # ------------------------------------------------------------
    # QUALITY GATE
    # ------------------------------------------------------------

    print()
    print("QUALITY GATE")

    gate = result["quality_gate"]

    print(f"Accepted        : {gate['accepted']}")
    print(f"Blend ratio     : {gate['blend_ratio']}")
    print(f"Score before    : {gate['score_before']:.6f}")
    print(f"Score enhanced  : {gate['score_enhanced']:.6f}")
    print(f"Final score     : {gate['final_score']:.6f}")
    print(f"Reason          : {gate['reason']}")

    if "safety_passed" in gate:
        print(f"Safety passed   : {gate['safety_passed']}")

    if "snr_change_db" in gate:
        print(f"SNR delta       : {gate['snr_change_db']:+.6f} dB")

    if "noise_change" in gate:
        print(f"Noise delta     : {gate['noise_change']:+.6f}")

    if "contrast_change" in gate:
        print(f"Contrast delta  : {gate['contrast_change']:+.6f}")

    if "entropy_change" in gate:
        print(f"Entropy delta   : {gate['entropy_change']:+.6f}")

    # ------------------------------------------------------------
    # DICOM
    # ------------------------------------------------------------

    print()
    print("DICOM")
    print(f"Output          : {result['dicom_path']}")

    print()
    print("PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()