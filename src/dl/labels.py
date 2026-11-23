import json
from functools import lru_cache

from src.dl.config import CLASS_INDICES_PATH

SEPARATOR = "___"


@lru_cache(maxsize=1)
def load_class_names(path=CLASS_INDICES_PATH) -> list[str]:
    with open(path) as f:
        class_indices = json.load(f)
    return [name for name, _ in sorted(class_indices.items(), key=lambda kv: kv[1])]


def split_label(label: str) -> tuple[str, str]:
    crop, disease = label.split(SEPARATOR, 1)
    return crop, disease


def crop_of(label: str) -> str:
    return split_label(label)[0]


def is_healthy(label: str) -> bool:
    return split_label(label)[1].lower() == "healthy"


def pretty(name: str) -> str:
    return name.replace("_", " ").strip()


def pretty_label(label: str) -> str:
    crop, disease = split_label(label)
    return f"{pretty(crop)} - {pretty(disease)}"


def list_crops(class_names: list[str] | None = None) -> list[str]:
    class_names = class_names or load_class_names()
    seen = []
    for name in class_names:
        crop = crop_of(name)
        if crop not in seen:
            seen.append(crop)
    return seen
