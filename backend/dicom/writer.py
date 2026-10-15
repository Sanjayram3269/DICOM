from pathlib import Path

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import (
    ExplicitVRLittleEndian,
    DigitalXRayImageStorageForPresentation,
    generate_uid,
)


def write_dicom(
    image: np.ndarray,
    output_path: str | Path,
    anatomy: str = "CHEST",
    view: str = "PA",
    patient_name: str = "DEMO^PATIENT",
    patient_id: str = "DEMO001",
) -> Path:
    """
    Write a grayscale image as a DICOM Digital X-Ray object.

    Prototype/research implementation.
    Not intended for clinical diagnostic use without
    appropriate validation and regulatory compliance.
    """

    image = np.asarray(image)

    if image.ndim != 2:
        raise ValueError(
            f"DICOM writer expects a 2D grayscale image, got {image.shape}"
        )

    # ---------------------------------------------------------
    # Convert image to unsigned 16-bit pixel data
    # ---------------------------------------------------------

    if np.issubdtype(image.dtype, np.floating):
        image = np.nan_to_num(
            image,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )

        image = np.clip(image, 0.0, 1.0)

        pixel_array = np.round(
            image * 65535.0
        ).astype(np.uint16)

    else:
        pixel_array = np.clip(
            image,
            0,
            65535,
        ).astype(np.uint16)

    # ---------------------------------------------------------
    # Output path
    # ---------------------------------------------------------

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # DICOM identifiers
    # ---------------------------------------------------------

    sop_instance_uid = generate_uid()

    # ---------------------------------------------------------
    # File Meta Information
    # ---------------------------------------------------------

    file_meta = FileMetaDataset()

    file_meta.MediaStorageSOPClassUID = (
        DigitalXRayImageStorageForPresentation
    )

    file_meta.MediaStorageSOPInstanceUID = (
        sop_instance_uid
    )

    file_meta.TransferSyntaxUID = (
        ExplicitVRLittleEndian
    )

    file_meta.ImplementationClassUID = generate_uid()

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    ds = FileDataset(
        str(output_path),
        {},
        file_meta=file_meta,
        preamble=b"\0" * 128,
    )

    ds.is_little_endian = True
    ds.is_implicit_VR = False

    # ---------------------------------------------------------
    # Patient / Study
    # ---------------------------------------------------------

    ds.PatientName = patient_name
    ds.PatientID = patient_id

    ds.Modality = "DX"

    ds.BodyPartExamined = anatomy
    ds.ViewPosition = view

    # ---------------------------------------------------------
    # UIDs
    # ---------------------------------------------------------

    ds.SOPClassUID = (
        DigitalXRayImageStorageForPresentation
    )

    ds.SOPInstanceUID = sop_instance_uid

    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()

    # ---------------------------------------------------------
    # Image attributes
    # ---------------------------------------------------------

    ds.Rows = pixel_array.shape[0]
    ds.Columns = pixel_array.shape[1]

    ds.SamplesPerPixel = 1

    ds.PhotometricInterpretation = "MONOCHROME2"

    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15

    ds.PixelRepresentation = 0

    # Prototype pixel spacing.
    # Real detector metadata must replace this.
    ds.PixelSpacing = [1.0, 1.0]

    ds.RescaleIntercept = 0
    ds.RescaleSlope = 1

    # ---------------------------------------------------------
    # Pixel data
    # ---------------------------------------------------------

    ds.PixelData = pixel_array.tobytes()

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    ds.save_as(
        str(output_path),
        write_like_original=False,
    )

    return output_path