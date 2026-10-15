# Data Plan

## Dataset 1 — NIH ChestX-ray14
Use for the first local chest-radiograph benchmark. It provides 112,120 frontal chest radiographs at 1024×1024 with 14 pathology labels and patient/view metadata.

## Dataset 2 — MIMIC-CXR
Use later for higher-fidelity DICOM experimentation. The source dataset contains DICOM images and reports, but access requires PhysioNet credentialing and a data-use agreement.

## RAW strategy
Neither public dataset is a detector-native universal `.raw` benchmark. Therefore, create controlled RAW fixtures from public images. This validates our parser and processing pipeline, but must never be presented as real detector RAW validation.

## Required future data
A real detector sample plus its acquisition specification is eventually required to validate width/height/bit depth/endianness, offset/gain correction, orientation, bad-pixel correction, and detector-specific calibration.
