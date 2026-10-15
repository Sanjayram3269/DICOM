from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import (
    ExplicitVRLittleEndian,
    DigitalXRayImageStorageForPresentation,
    generate_uid,
)


def write_dicom(image: np.ndarray, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.asarray(np.clip(image, 0.0, 1.0) * 65535.0, dtype=np.uint16)
    height, width = pixels.shape
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = DigitalXRayImageStorageForPresentation
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    dataset = FileDataset(str(output_path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.StudyInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = generate_uid()
    dataset.Modality = "DX"
    dataset.PatientName = "DEMO^PATIENT"
    dataset.PatientID = "DEMO001"
    dataset.PatientBirthDate = ""
    dataset.PatientSex = "O"
    dataset.StudyDate = datetime.now().strftime("%Y%m%d")
    dataset.StudyTime = datetime.now().strftime("%H%M%S")
    dataset.BodyPartExamined = "CHEST"
    dataset.ViewPosition = "PA"
    dataset.Rows = int(height)
    dataset.Columns = int(width)
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.PixelData = pixels.tobytes()
    dataset.save_as(output_path, enforce_file_format=True)
    return output_path
