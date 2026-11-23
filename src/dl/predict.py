import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from src.dl.config import IMG_SIZE, MODEL_PATH
from src.dl.labels import load_class_names


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


class Predictor:
    def __init__(self, model_path=MODEL_PATH):
        import tensorflow as tf
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Copy crop_disease_cnn_ft_final.keras into models/."
            )
        self._preprocess = preprocess_input
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.class_names = load_class_names()

    def predict(self, source) -> np.ndarray:
        x = self._preprocess(load_image(source))
        return self.model.predict(x[np.newaxis, ...], verbose=0)[0]

    def top_k(self, probs: np.ndarray, k: int = 3) -> list[dict]:
        idx = np.argsort(probs)[::-1][:k]
        return [{"label": self.class_names[i], "prob": round(float(probs[i]), 4)} for i in idx]
