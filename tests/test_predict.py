import importlib.util

import numpy as np
import pytest

from src.dl.config import MODEL_PATH, PROJECT_ROOT, TFLITE_PATH

pytestmark = pytest.mark.skipif(not (TFLITE_PATH.exists() or MODEL_PATH.exists()), reason="no model in models/")
SANITY_DIR = PROJECT_ROOT / "data" / "sanity_check"
HAS_TF = importlib.util.find_spec("tensorflow") is not None


@pytest.fixture(scope="module")
def predictor():
    from src.dl.predict import Predictor
    return Predictor()


def sample(folder):
    d = SANITY_DIR / folder
    if not d.exists():
        pytest.skip("no sanity-check images")
    return sorted(d.iterdir())[0]


def test_probabilities_are_valid(predictor):
    probs = predictor.predict(sample("Tomato___Late_blight"))
    assert probs.shape == (38,)
    assert np.isclose(probs.sum(), 1.0, atol=1e-3)


def test_path_and_bytes_give_same_result(predictor):
    img = sample("Apple___Apple_scab")
    assert np.allclose(predictor.predict(img), predictor.predict(img.read_bytes()), atol=1e-5)


def test_model_download_uses_repo_and_token(monkeypatch, tmp_path):
    import src.dl.predict as P
    seen = {}

    class Resp:
        content = b"model-bytes"

        def raise_for_status(self):
            pass

    def fake_get(url, headers=None, timeout=None):
        seen.update(url=url, headers=headers)
        return Resp()

    monkeypatch.setenv("MODEL_REPO", "someone/crop-doctor-model")
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setattr(P.requests, "get", fake_get)
    target = tmp_path / "crop_disease_cnn.tflite"
    assert P.download_model(target) is True
    assert target.read_bytes() == b"model-bytes"
    assert seen["url"] == "https://huggingface.co/someone/crop-doctor-model/resolve/main/crop_disease_cnn.tflite"
    assert seen["headers"] == {"Authorization": "Bearer hf_test"}
    monkeypatch.delenv("MODEL_REPO")
    assert P.download_model(target) is False


def test_numpy_preprocess_matches_mobilenet_preprocess_input():
    if not HAS_TF:
        pytest.skip("tensorflow not installed")
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from src.dl.predict import preprocess
    x = np.random.default_rng(1).uniform(0, 255, (4, 224, 224, 3)).astype("float32")
    assert np.allclose(preprocess(x.copy()), preprocess_input(x.copy()), atol=1e-6)


@pytest.mark.skipif(not (HAS_TF and MODEL_PATH.exists() and TFLITE_PATH.exists()), reason="needs TF + both models")
def test_tflite_matches_keras_model():
    from src.dl.predict import Predictor
    lite, full = Predictor(TFLITE_PATH), Predictor(MODEL_PATH)
    img = sample("Corn_(maize)___Common_rust_")
    assert np.allclose(lite.predict(img), full.predict(img), atol=1e-4)


@pytest.mark.skipif(not (HAS_TF and MODEL_PATH.exists()), reason="needs TF + the Keras model")
def test_rebuilt_architecture_matches_saved_model():
    from src.dl.model import build_model
    from src.dl.predict import Predictor
    full = Predictor(MODEL_PATH)
    rebuilt, _ = build_model(num_classes=38, weights=None)
    rebuilt.set_weights(full.model.get_weights())
    x = np.random.default_rng(0).uniform(-1, 1, (1, 224, 224, 3)).astype("float32")
    assert np.allclose(rebuilt.predict(x, verbose=0), full.model.predict(x, verbose=0), atol=1e-5)
