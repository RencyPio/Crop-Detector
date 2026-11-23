import io
import os
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv
from PIL import Image, ImageOps

from src.dl.config import IMG_SIZE, MODEL_PATH, TFLITE_PATH
from src.dl.labels import load_class_names

load_dotenv()


def load_image(source) -> np.ndarray:
    if isinstance(source, Image.Image):
        img = source
    elif isinstance(source, (bytes, bytearray)):
        img = Image.open(io.BytesIO(source))
    else:
        img = Image.open(Path(source))
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB").resize(IMG_SIZE, Image.NEAREST)
    return np.asarray(img, dtype=np.float32)


def preprocess(x: np.ndarray) -> np.ndarray:
    return x / 127.5 - 1.0


def download_model(path: Path = TFLITE_PATH) -> bool:
    repo = os.getenv("MODEL_REPO")
    if not repo:
        return False
    token = os.getenv("HF_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    r = requests.get(f"https://huggingface.co/{repo}/resolve/main/{path.name}", headers=headers, timeout=180)
    r.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(r.content)
    return True


def _interpreter(path: Path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    return Interpreter(model_path=str(path))


class Predictor:
    def __init__(self, model_path=None):
        if model_path is None:
            if not TFLITE_PATH.exists() and not MODEL_PATH.exists():
                download_model()
            model_path = TFLITE_PATH if TFLITE_PATH.exists() else MODEL_PATH
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Put crop_disease_cnn.tflite in models/ or set MODEL_REPO."
            )
        self.class_names = load_class_names()
        self.backend = "tflite" if model_path.suffix == ".tflite" else "keras"
        if self.backend == "tflite":
            self._interp = _interpreter(model_path)
            self._interp.allocate_tensors()
            self._in = self._interp.get_input_details()[0]["index"]
            self._out = self._interp.get_output_details()[0]["index"]
        else:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(model_path, compile=False)

    def predict(self, source) -> np.ndarray:
        x = preprocess(load_image(source))[np.newaxis, ...].astype(np.float32)
        if self.backend == "tflite":
            self._interp.set_tensor(self._in, x)
            self._interp.invoke()
            return self._interp.get_tensor(self._out)[0]
        return self.model.predict(x, verbose=0)[0]

    def top_k(self, probs: np.ndarray, k: int = 3) -> list[dict]:
        idx = np.argsort(probs)[::-1][:k]
        return [{"label": self.class_names[i], "prob": round(float(probs[i]), 4)} for i in idx]
