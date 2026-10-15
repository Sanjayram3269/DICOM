from pathlib import Path

import pydicom


DICOM_PATH = (
    Path("data/demo/output")
    / "synthetic_chest_1024x1024_enhanced.dcm"
)


def main():
    print("=" * 60)
    print("DICOM VALIDATION")
    print("=" * 60)

    print()
    print("FILE")
    print("-" * 60)
    print(f"Path: {DICOM_PATH}")

    if not DICOM_PATH.exists():
        raise FileNotFoundError(
            f"DICOM file not found: {DICOM_PATH}"
        )

    ds = pydicom.dcmread(DICOM_PATH)

    print()
    print("IDENTIFICATION")
    print("-" * 60)
    print(f"SOP Class UID:          {ds.SOPClassUID}")
    print(f"Modality:               {ds.Modality}")
    print(f"Patient Name:           {ds.PatientName}")
    print(f"Patient ID:             {ds.PatientID}")
    print(f"Body Part Examined:     {ds.BodyPartExamined}")
    print(f"View Position:          {getattr(ds, 'ViewPosition', 'N/A')}")

    print()
    print("IMAGE")
    print("-" * 60)
    print(f"Rows:                   {ds.Rows}")
    print(f"Columns:                {ds.Columns}")
    print(f"Samples Per Pixel:      {ds.SamplesPerPixel}")
    print(f"Photometric:            {ds.PhotometricInterpretation}")
    print(f"Bits Allocated:         {ds.BitsAllocated}")
    print(f"Bits Stored:            {ds.BitsStored}")
    print(f"High Bit:               {ds.HighBit}")
    print(f"Pixel Representation:   {ds.PixelRepresentation}")

    print()
    print("PIXEL DATA")
    print("-" * 60)

    image = ds.pixel_array

    print(f"Shape:                  {image.shape}")
    print(f"Dtype:                  {image.dtype}")
    print(f"Minimum:                {image.min()}")
    print(f"Maximum:                {image.max()}")
    print(f"Mean:                   {image.mean():.2f}")
    print(f"Std Dev:                {image.std():.2f}")

    expected_shape = (ds.Rows, ds.Columns)

    print()
    print("VALIDATION")
    print("-" * 60)

    checks = {
        "File exists": DICOM_PATH.exists(),
        "2D image": image.ndim == 2,
        "Shape matches metadata": image.shape == expected_shape,
        "1024 x 1024": image.shape == (1024, 1024),
        "16-bit": image.dtype == "uint16",
        "MONOCHROME2": ds.PhotometricInterpretation == "MONOCHROME2",
        "Single channel": ds.SamplesPerPixel == 1,
    }

    all_passed = True

    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"

        print(f"{status:<8} {name}")

        if not passed:
            all_passed = False

    print()
    print("=" * 60)

    if all_passed:
        print("DICOM VALIDATION PASSED")
    else:
        print("DICOM VALIDATION FAILED")

    print("=" * 60)


if __name__ == "__main__":
    main()