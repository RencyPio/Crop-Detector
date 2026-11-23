import numpy as np

from src.dl.labels import crop_of, load_class_names


def crop_mask(probs: np.ndarray, crop: str | None, class_names: list[str] | None = None):
    class_names = class_names or load_class_names()
    if not crop:
        return probs, 1.0
    keep = np.array([crop_of(n) == crop for n in class_names])
    if not keep.any():
        raise ValueError(f"Unknown crop '{crop}'")
    crop_mass = float(probs[keep].sum())
    masked = np.where(keep, probs, 0.0)
    if crop_mass > 0:
        masked = masked / crop_mass
    return masked, crop_mass
