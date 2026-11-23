import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from src.automation.logger import recent
from src.automation.notify import SENDERS
from src.dl.labels import list_crops, pretty
from src.validation import ImageError
from src.validation import validate_image as _validate

log = logging.getLogger("cropdetector.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    from src.pipeline import _components
    _components()
    log.info("model loaded")
    yield


app = FastAPI(title="Crop Disease Advisory API", version="1.0", lifespan=lifespan)


def validate_image(data: bytes):
    try:
        return _validate(data)
    except ImageError as e:
        raise HTTPException(e.status, e.message)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/crops")
def crops():
    return [{"value": c, "label": pretty(c)} for c in list_crops()]


@app.post("/predict")
async def predict(
    image: UploadFile = File(..., description="Leaf photo (JPG/PNG/WEBP)"),
    crop: str | None = Form(None, description="Crop the farmer grows, e.g. Tomato"),
    lat: float | None = Form(None, ge=-90, le=90),
    lon: float | None = Form(None, ge=-180, le=180),
    channels: str = Form("", description="Comma-separated alert channels: telegram,sms"),
):
    from src.pipeline import run

    if crop == "":
        crop = None
    if crop and crop not in list_crops():
        raise HTTPException(422, f"Unknown crop '{crop}'. See GET /crops.")
    chans = [c.strip() for c in channels.split(",") if c.strip()]
    bad = [c for c in chans if c not in SENDERS]
    if bad:
        raise HTTPException(422, f"Unknown channel(s) {bad}. Allowed: {sorted(SENDERS)}.")

    img = validate_image(await image.read())
    result = run(img, crop=crop, lat=lat, lon=lon, channels=chans, image_name=image.filename)
    log.info("predict image=%s crop=%s decision=%s top1=%s alerts=%s", image.filename, crop,
             result["decision"], result["predictions"][0]["label"], result["alert"]["status"])
    return result


@app.get("/detections")
def detections(limit: int = 50):
    rows = recent(max(1, min(limit, 500)))
    for r in rows:
        r.pop("trail_json", None)
    return rows
