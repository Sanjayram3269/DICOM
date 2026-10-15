# Data plan

## Development approach

1. Begin with public chest-radiograph datasets whose usage terms and provenance have been reviewed.
2. Keep source image identifiers and dataset-specific metadata with each converted fixture.
3. Convert images into controlled 16-bit RAW fixtures only when the conversion process is documented and repeatable.
4. Validate parser, enhancement and DICOM output separately so errors can be localized.
5. Do not treat converted public images as equivalent to real detector-native RAW captures.
6. Before supporting any real detector, obtain its actual file specification and calibration information.

## Evaluation safeguards

- Keep training, tuning and evaluation data separated.
- Preserve the original image for comparisons and fallback.
- Report dataset size, preprocessing, metric definitions and failed cases.
- Treat image-quality metrics as engineering proxies, not clinical endpoints.
- Do not publish patient-identifiable data or derived images without appropriate permissions.
