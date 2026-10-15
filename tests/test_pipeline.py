import numpy as np
from backend.analysis.quality import analyze
from backend.enhancement.adaptive_engine import select_parameters, enhance

def test_enhancement_pipeline():
    rng = np.random.default_rng(7)
    image = np.clip(rng.normal(0.5, 0.15, (128, 128)), 0, 1).astype(np.float32)
    q = analyze(image)
    out = enhance(image, select_parameters("CHEST", q["snr_db"]))
    assert out.shape == image.shape
    assert np.isfinite(out).all()
    assert 0 <= out.min() <= out.max() <= 1
