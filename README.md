# Adaptive Radiography Intelligence Platform

> **An end-to-end RAW radiography processing platform — currently in active development.**

A software engineering project exploring how raw detector image data can be ingested, analyzed, adaptively enhanced, quality-checked, and packaged as DICOM for downstream radiology workflows. The goal is to build a reliable, transparent processing platform with conservative failure handling and a professional review interface.

**Development status:** research / engineering prototype. Not clinically validated and not intended for diagnostic use.

---

## Why this project

Radiography systems may produce detector-specific pixel data that needs careful interpretation before it can be handled consistently by downstream software. A robust pipeline must do more than apply a visual filter: it needs explicit input assumptions, measurable image-quality checks, controlled enhancement, valid DICOM output, and traceable fallback behavior.

This project is being developed around those principles.

## Current capabilities

- **RAW ingestion:** explicit input profiles for image dimensions, bit depth, signedness, and endianness, with input validation.
- **Image-quality analysis:** engineering metrics for signal-to-noise ratio (SNR), residual-noise estimation, contrast, entropy, clipping, gradients, and sharpness proxies.
- **Adaptive enhancement search:** generates and evaluates multiple parameter candidates on a reduced-resolution image, then applies the selected candidate at full resolution.
- **Safety-oriented quality gate:** checks changes in SNR, noise, and contrast and can reject an enhancement in favor of the original image.
- **DICOM generation:** creates and validates DICOM output with image pixel metadata.
- **Processing API and viewer:** FastAPI backend and a React/TypeScript frontend with image viewing controls and processing results.
- **Validation tooling:** single-image and batch scripts for repeatable engineering checks and diagnostic reporting.

## Processing workflow

```text
Detector RAW + explicit input profile
                |
                v
        RAW parsing / validation
                |
                v
       Normalize + analyze quality
                |
                v
  Anatomy / view classification interface
                |
                v
   Generate candidate enhancement settings
                |
                v
    Coarse-resolution candidate evaluation
                |
                v
       Safety and quality gate
          /             \
     accepted           rejected
        |                  |
        v                  v
 Full-resolution       Preserve original
 enhancement           image as fallback
        \                  /
         v                v
         Final quality analysis
                  |
                  v
            DICOM generation
                  |
                  v
         API / viewer / export
```

The pipeline separates **analysis and parameter selection** from deterministic pixel transformations. A candidate is not automatically accepted simply because it looks sharper or has a higher composite score.

## Engineering validation snapshot

The project has been exercised against a public chest-radiograph image set converted to a defined 16-bit RAW representation, in addition to a synthetic test fixture.

| Validation item | Observed result |
|---|---|
| Batch execution | 138/138 images processed successfully in a prior validation run |
| Mean SNR change in that batch | +2.17 dB |
| Median SNR change in that batch | +1.80 dB |
| Quality-gate outcomes | 107 accepted; 31 rejected/fallback |
| Candidate-search smoke test | 5-image run selected multiple distinct candidate settings and exercised fallback |
| DICOM checks | Generated outputs passed the project's current structural checks |

**Interpretation matters:** these are engineering results for the current dataset, RAW conversion procedure, and proxy metrics—not clinical performance claims. Successful file processing does not establish diagnostic quality, and SNR proxies are not a substitute for radiologist assessment.

A separate current test-fixture run has shown SNR increasing from **19.4 dB to 25.56 dB**. Results depend on the image and measurement method; do not compare metrics across datasets or scripts without checking their definitions.

## Technology stack

- **Backend:** Python, FastAPI
- **Image processing:** NumPy, OpenCV, SciPy, scikit-image
- **Medical imaging:** pydicom
- **Frontend:** React, TypeScript, Vite, Tailwind CSS
- **Validation:** Python scripts for image-level checks, batch summaries, and DICOM inspection

## Repository layout

```text
backend/
  analysis/       # Image-quality metrics
  anatomy/        # Anatomy/view classifier interface and features
  api/            # API routes
  dicom/          # DICOM writer
  enhancement/    # Adaptive candidate engine and quality gate
  ingestion/      # RAW profiles and parser
  pipeline/       # End-to-end processing orchestration
frontend/         # React + TypeScript application
frontend_streamlit/ # Exploratory Streamlit interface
scripts/          # Fixture generation, validation, and analysis tools
data/demo/        # Local demo fixtures and generated outputs
```

Exact available scripts and datasets may change as development progresses.

## Getting started

The project is under active development, so setup commands may evolve. Use Python 3.12 where possible and create an isolated virtual environment.

### Backend environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install dependencies from the project's dependency file if one is present in the checked-out version. If dependency metadata is not yet committed, the environment is not fully reproducible from a clean clone; dependency pinning is a planned repository-hardening task.

### Run the API

From the repository root, after the backend dependencies are installed:

```powershell
python -m uvicorn backend.api.main:app --reload
```

If the module path or startup command changes, follow the entry point defined in `backend/api/main.py`.

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

The UI's ability to process images depends on the API being available and correctly configured.

## Input-data assumptions and limitations

RAW is a byte stream, not a universal medical-image format. Correct interpretation requires detector-specific information such as dimensions, bit depth, signedness, byte order, pixel packing, orientation, and calibration. Incorrect assumptions can produce a plausible-looking but invalid image.

The current workflow uses explicit profiles and development/test images. Production use would require supported detector specifications, calibration handling, robust metadata management, and validation against representative devices.

## Anatomy and view intelligence: current status

The classifier interface now exposes structured anatomy/view outputs, confidence fields, model/method metadata, and extracted diagnostic features. The current implementation is a **baseline development component**, not a trained or clinically validated anatomy/view model. It can report an unknown view and must not be represented as reliable automated clinical classification.

Planned work includes a properly trained and independently evaluated classifier, calibrated uncertainty, broader anatomy/view coverage, and safe handling when model confidence is insufficient.

## Roadmap

- [x] Explicit RAW profiles and parsing
- [x] Image-quality metrics for engineering diagnostics
- [x] Adaptive candidate generation and coarse-to-fine search
- [x] Conservative quality gate and original-image fallback
- [x] DICOM output and structural validation
- [x] Batch validation/reporting tooling
- [x] Structured anatomy/view result schema
- [ ] Train and evaluate genuine anatomy/view models on suitable labeled datasets
- [ ] Add automated unit, integration, and regression tests to CI
- [ ] Pin dependencies and document fully reproducible clean setup
- [ ] Expand API contracts, error handling, and observability
- [ ] Add richer side-by-side viewer and processing provenance
- [ ] Validate performance across detector profiles and representative devices
- [ ] Obtain expert review and conduct appropriate clinical/quality validation before any clinical claims or deployment

## Responsible-use statement

This repository is an **engineering and research prototype**, not a medical device and not a clinically validated diagnostic system. Do not use its output to make patient-care decisions. Image enhancement can alter appearance and may obscure or exaggerate features; the original image should remain available for review. Any future clinical deployment would require appropriate domain-expert oversight, device-specific verification, data governance, regulatory assessment, and clinical validation.

## Contributing

Issues and pull requests that improve reproducibility, test coverage, input validation, DICOM correctness, performance measurement, accessibility, and documentation are welcome.

## License

No license is declared yet. Until a license is added, assume standard copyright applies and do not assume permission to reuse, modify, or redistribute this repository.

---

**Built as an ongoing engineering project focused on adaptive radiography processing, measurable quality checks, and safe failure behavior.**
