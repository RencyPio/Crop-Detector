import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import src.pipeline
from src.api import main

FAKE_RESULT = {"decision": "healthy", "predictions": [{"label": "Tomato___healthy", "prob": 0.99}],
               "alert": {"status": {}}, "trail": []}


@pytest.fixture
def client(monkeypatch):
    calls = []
    monkeypatch.setattr(src.pipeline, "_components", lambda: None)
    monkeypatch.setattr(src.pipeline, "run", lambda img, **kw: calls.append(kw) or FAKE_RESULT)
    with TestClient(main.app) as c:
        c.calls = calls
        yield c


def jpeg(size=(224, 224)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (40, 120, 40)).save(buf, "JPEG")
    return buf.getvalue()


def test_health_and_crops(client):
    assert client.get("/health").json() == {"status": "ok"}
    crops = client.get("/crops").json()
    assert {"value": "Corn_(maize)", "label": "Corn (maize)"} in crops and len(crops) == 14


def test_predict_passes_form_fields_to_pipeline(client):
    r = client.post("/predict", files={"image": ("leaf.jpg", jpeg(), "image/jpeg")},
                    data={"crop": "Tomato", "lat": "13.08", "lon": "80.27", "channels": "telegram"})
    assert r.status_code == 200 and r.json()["decision"] == "healthy"
    assert client.calls[0] == {"crop": "Tomato", "lat": 13.08, "lon": 80.27, "channels": ["telegram"],
                               "image_name": "leaf.jpg"}


@pytest.mark.parametrize("payload,code", [
    (b"", 400),
    (b"this is not an image", 415),
    (jpeg()[:200], 415),
    (jpeg((20, 20)), 400),
])
def test_bad_uploads_are_rejected(client, payload, code):
    r = client.post("/predict", files={"image": ("x.jpg", payload, "image/jpeg")})
    assert r.status_code == code, r.text
    assert client.calls == []


def test_unknown_crop_and_channel_are_rejected(client):
    f = {"image": ("leaf.jpg", jpeg(), "image/jpeg")}
    assert client.post("/predict", files=f, data={"crop": "Banana"}).status_code == 422
    assert client.post("/predict", files=f, data={"channels": "whatsapp"}).status_code == 422


def test_out_of_range_location_is_rejected(client):
    r = client.post("/predict", files={"image": ("leaf.jpg", jpeg(), "image/jpeg")}, data={"lat": "200"})
    assert r.status_code == 422
