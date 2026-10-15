from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.ingestion.raw_reader import RawProfile
from backend.pipeline.processor import process_raw


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "demo"
UPLOAD_DIR = DATA_DIR / "uploads"
PREVIEW_DIR = DATA_DIR / "previews"
OUTPUT_DIR = DATA_DIR / "output"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


app = FastAPI(
    title="Radiograph Processor",
    version="0.1.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    status: str
    service: str


def save_preview(
    image: np.ndarray,
    output_path: Path,
) -> None:
    """
    Save a normalized grayscale image as an 8-bit PNG.

    This PNG is only a browser preview.
    The DICOM remains the actual processed output.
    """

    image = np.asarray(image)

    if image.ndim != 2:
        raise ValueError(
            f"Preview image must be 2D, got {image.shape}"
        )

    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    image = np.clip(
        image,
        0.0,
        1.0,
    )

    preview = np.round(
        image * 255.0
    ).astype(np.uint8)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success = cv2.imwrite(
        str(output_path),
        preview,
    )

    if not success:
        raise RuntimeError(
            f"Failed to write preview: {output_path}"
        )


@app.get("/")
def root():
    return {
        "service": "Radiograph Processor",
        "version": "0.1.0",
        "status": "running",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health():
    return {
        "status": "ok",
        "service": "radiograph-processor",
    }


@app.post("/process")
async def process_endpoint(
    file: UploadFile = File(...),
    width: int = Form(...),
    height: int = Form(...),
    bit_depth: int = Form(...),
    signed: bool = Form(False),
    endian: str = Form("little"),
):
    # ---------------------------------------------------------
    # 1. Read uploaded RAW
    # ---------------------------------------------------------

    raw_bytes = await file.read()

    if not raw_bytes:
        return {
            "status": "error",
            "message": "Uploaded RAW file is empty.",
        }

    safe_filename = Path(
        file.filename or "radiograph.raw"
    ).name

    upload_path = (
        UPLOAD_DIR / safe_filename
    )

    upload_path.write_bytes(
        raw_bytes
    )

    # ---------------------------------------------------------
    # 2. RAW profile
    # ---------------------------------------------------------

    profile = RawProfile(
        width=width,
        height=height,
        bit_depth=bit_depth,
        signed=signed,
        endian=endian,
    )

    # ---------------------------------------------------------
    # 3. Run processing pipeline
    # ---------------------------------------------------------

    result = process_raw(
        path=str(upload_path),
        profile=profile,
        output_dir=str(OUTPUT_DIR),
    )

    # ---------------------------------------------------------
    # 4. Generate browser previews
    # ---------------------------------------------------------

    stem = Path(
        safe_filename
    ).stem

    original_preview_name = (
        f"{stem}_original.png"
    )

    enhanced_preview_name = (
        f"{stem}_enhanced.png"
    )

    original_preview_path = (
        PREVIEW_DIR
        / original_preview_name
    )

    enhanced_preview_path = (
        PREVIEW_DIR
        / enhanced_preview_name
    )

    save_preview(
        result["original"],
        original_preview_path,
    )

    save_preview(
        result["enhanced"],
        enhanced_preview_path,
    )

    # ---------------------------------------------------------
    # 5. DICOM filename
    # ---------------------------------------------------------

    dicom_path = Path(
        result["dicom_path"]
    )

    # ---------------------------------------------------------
    # 6. Return frontend-safe JSON
    # ---------------------------------------------------------

    return {
        "status": "success",

        "anatomy": result["anatomy"],
        "confidence": result["confidence"],
        "view": result["view"],
        "model": result["model"],

        "before": result["before"],
        "after": result["after"],

        "parameters": result["parameters"],

        "dicom_url": (
            f"/dicom/{dicom_path.name}"
        ),

        "original_preview_url": (
            f"/preview/{original_preview_name}"
        ),

        "enhanced_preview_url": (
            f"/preview/{enhanced_preview_name}"
        ),
    }


@app.get("/preview/{filename}")
def get_preview(
    filename: str,
):
    safe_filename = Path(
        filename
    ).name

    preview_path = (
        PREVIEW_DIR
        / safe_filename
    )

    if not preview_path.exists():
        return {
            "status": "error",
            "message": "Preview not found.",
        }

    return FileResponse(
        preview_path,
        media_type="image/png",
    )


@app.get("/dicom/{filename}")
def get_dicom(
    filename: str,
):
    safe_filename = Path(
        filename
    ).name

    dicom_path = (
        OUTPUT_DIR
        / safe_filename
    )

    if not dicom_path.exists():
        return {
            "status": "error",
            "message": "DICOM file not found.",
        }

    return FileResponse(
        dicom_path,
        media_type="application/dicom",
        filename=safe_filename,
    )