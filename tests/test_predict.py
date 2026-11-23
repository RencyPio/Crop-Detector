from pathlib import Path

import numpy as np
import pytest

from src.dl.config import MODEL_PATH, PROJECT_ROOT

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="trained model not in models/")
SANITY_DIR = PROJECT_ROOT / "data" / "sanity_check"


@pytest.fixture(scope="module")
def predictor():
    from src.dl.predict import Predictor
    return Predictor()


def test_probabilities_are_valid(predictor):
    img = next((SANITY_DIR / "Tomato___Late_blight").iterdir(), None) if SANITY_DIR.exists() else None
    if img is None:
        pytest.skip("no sanity-check images")
    probs = predictor.predict(img)
    assert probs.shape == (38,)
    assert np.isclose(probs.sum(), 1.0, atol=1e-3)


def test_path_and_bytes_give_same_result(predictor):
    folder = SANITY_DIR / "Apple___Apple_scab"
    if not folder.exists():
        pytest.skip("no sanity-check images")
    img = sorted(folder.iterdir())[0]
    assert np.allclose(predictor.predict(img), predictor.predict(img.read_bytes()), atol=1e-5)


def test_rebuilt_architecture_matches_saved_model(predictor):
    from src.dl.model import build_model
    rebuilt, _ = build_model(num_classes=38, weights=None)
    rebuilt.set_weights(predictor.model.get_weights())
    x = np.random.default_rng(0).uniform(-1, 1, (1, 224, 224, 3)).astype("float32")
    assert np.allclose(rebuilt.predict(x, verbose=0), predictor.model.predict(x, verbose=0), atol=1e-5)
