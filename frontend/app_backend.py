import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

API_URL = os.getenv("API_URL")
MODE = "api" if API_URL else "direct"


class Unreachable(Exception):
    pass


def warm_up():
    if MODE == "direct":
        from src.pipeline import _components
        _components()


def health() -> bool:
    if MODE == "direct":
        return True
    try:
        return requests.get(f"{API_URL}/health", timeout=2).ok
    except requests.RequestException:
        return False


def crops() -> list[dict]:
    if MODE == "direct":
        from src.dl.labels import list_crops, pretty
        return [{"value": c, "label": pretty(c)} for c in list_crops()]
    try:
        return requests.get(f"{API_URL}/crops", timeout=10).json()
    except requests.RequestException as e:
        raise Unreachable(str(e))


def detections(limit: int = 200) -> list[dict]:
    if MODE == "direct":
        from src.automation.logger import recent
        rows = recent(limit)
        for r in rows:
            r.pop("trail_json", None)
        return rows
    try:
        return requests.get(f"{API_URL}/detections", params={"limit": limit}, timeout=10).json()
    except requests.RequestException:
        return []


def predict(form: dict, filename: str, data: bytes, mimetype: str) -> tuple[int, dict]:
    if MODE == "api":
        try:
            r = requests.post(f"{API_URL}/predict", timeout=120, data=form,
                              files={"image": (filename, data, mimetype)})
        except requests.RequestException as e:
            raise Unreachable(str(e))
        try:
            body = r.json()
        except ValueError:
            body = {"detail": r.text[:200]}
        return r.status_code, body

    from src.pipeline import run
    from src.validation import ImageError, validate_image
    try:
        img = validate_image(data)
    except ImageError as e:
        return e.status, {"detail": e.message}
    lat = float(form["lat"]) if form.get("lat") else None
    lon = float(form["lon"]) if form.get("lon") else None
    channels = [c for c in form.get("channels", "").split(",") if c]
    result = run(img, crop=form.get("crop") or None, lat=lat, lon=lon, channels=channels, image_name=filename)
    return 200, result
