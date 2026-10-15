from pathlib import Path

import numpy as np

from backend.analysis.quality import analyze_quality
from backend.ingestion.raw_reader import RawProfile, normalize_float, read_raw


def test_raw_round_trip(tmp_path: Path) -> None:
    source = np.arange(64, dtype="<u2").reshape(8, 8)
    path = tmp_path / "fixture.raw"
    source.tofile(path)
    loaded = read_raw(path, RawProfile(width=8, height=8, bit_depth=16))
    assert np.array_equal(source, loaded)


def test_normalize_float_constant_image() -> None:
    normalized = normalize_float(np.ones((8, 8), dtype=np.float32))
    assert np.all(normalized == 0.0)


def test_quality_metrics_are_finite() -> None:
    image = np.linspace(0.0, 1.0, 128 * 128, dtype=np.float32).reshape(128, 128)
    metrics = analyze_quality(image)
    assert all(np.isfinite(value) for value in metrics.values())
