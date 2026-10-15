import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from backend.ingestion.raw_reader import RawProfile
from backend.pipeline.processor import process_raw


RAW_PATH = (
    Path("data/demo")
    / "synthetic_chest_1024x1024.raw"
)

OUTPUT_DIR = Path(
    "data/demo/output"
)


def main():

    print()
    print("=" * 60)
    print("RADIOGRAPH PROCESSING PIPELINE")
    print("=" * 60)

    # ---------------------------------------------------------
    # RAW configuration
    # ---------------------------------------------------------

    profile = RawProfile(
        width=1024,
        height=1024,
        bit_depth=16,
        signed=False,
        endian="little",
    )

    print()
    print("INPUT")
    print("-" * 60)
    print(f"RAW file:   {RAW_PATH}")
    print(f"Width:      {profile.width}")
    print(f"Height:     {profile.height}")
    print(f"Bit depth:  {profile.bit_depth}")
    print(f"Endian:     {profile.endian}")

    # ---------------------------------------------------------
    # Process
    # ---------------------------------------------------------

    print()
    print("PROCESSING...")
    print("-" * 60)

    result = process_raw(
        path=str(RAW_PATH),
        profile=profile,
        output_dir=str(OUTPUT_DIR),
    )

    # ---------------------------------------------------------
    # Anatomy
    # ---------------------------------------------------------

    print()
    print("ANATOMY")
    print("-" * 60)

    print(
        f"Detected anatomy: "
        f"{result['anatomy']}"
    )

    print(
        f"Confidence: "
        f"{result['confidence']:.3f}"
    )

    print(
        f"View: "
        f"{result['view']}"
    )

    print(
        f"Model: "
        f"{result['model']}"
    )

    # ---------------------------------------------------------
    # Quality before
    # ---------------------------------------------------------

    before = result["before"]

    print()
    print("QUALITY — BEFORE")
    print("-" * 60)

    print(
        f"SNR:            "
        f"{before['snr_db']} dB"
    )

    print(
        f"Noise:          "
        f"{before['noise']}"
    )

    print(
        f"Signal:         "
        f"{before['signal']}"
    )

    print(
        f"Contrast:       "
        f"{before['contrast']}"
    )

    print(
        f"Dynamic range:  "
        f"{before['dynamic_range']}"
    )

    print(
        f"Entropy:        "
        f"{before['entropy']}"
    )

    # ---------------------------------------------------------
    # Parameters
    # ---------------------------------------------------------

    print()
    print("ADAPTIVE PARAMETERS")
    print("-" * 60)

    for name, value in result[
        "parameters"
    ].items():

        print(
            f"{name:20s}: {value}"
        )

    # ---------------------------------------------------------
    # Quality after
    # ---------------------------------------------------------

    after = result["after"]

    print()
    print("QUALITY — AFTER")
    print("-" * 60)

    print(
        f"SNR:            "
        f"{after['snr_db']} dB"
    )

    print(
        f"Noise:          "
        f"{after['noise']}"
    )

    print(
        f"Signal:         "
        f"{after['signal']}"
    )

    print(
        f"Contrast:       "
        f"{after['contrast']}"
    )

    print(
        f"Dynamic range:  "
        f"{after['dynamic_range']}"
    )

    print(
        f"Entropy:        "
        f"{after['entropy']}"
    )

    # ---------------------------------------------------------
    # Improvement
    # ---------------------------------------------------------

    snr_change = (
        after["snr_db"]
        - before["snr_db"]
    )

    noise_change = (
        after["noise"]
        - before["noise"]
    )

    print()
    print("IMPROVEMENT")
    print("-" * 60)

    print(
        f"SNR change:     "
        f"{snr_change:+.2f} dB"
    )

    print(
        f"Noise change:   "
        f"{noise_change:+.6f}"
    )

    # ---------------------------------------------------------
    # DICOM
    # ---------------------------------------------------------

    print()
    print("DICOM")
    print("-" * 60)

    print(
        f"Generated: "
        f"{result['dicom_path']}"
    )

    print()
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()