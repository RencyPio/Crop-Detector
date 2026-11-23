import requests

from src.automation import senders
from src.automation.formatter import SMS_LIMIT, format_full, format_short
from src.automation.logger import log_detection, recent
from src.automation.notify import notify

ADVISE = {
    "crop_selected": "Tomato", "decision": "advise", "reason": None, "crop_mass": 0.99,
    "message": "Diagnosis: Tomato - Late blight (95%).",
    "predictions": [{"label": "Tomato___Late_blight", "prob": 0.95},
                    {"label": "Tomato___Early_blight", "prob": 0.04},
                    {"label": "Tomato___healthy", "prob": 0.01}],
    "advice": "**Chemical options:**\n- Spray mancozeb 2 g/l. [1]" + " More detail." * 100,
    "sources": [{"n": 1, "title": "T", "url": "https://agritech.tnau.ac.in/x", "publisher": "TNAU", "region": "India"}],
    "weather": {"rain_prob_next_6h": 0.8, "modified_advice": True, "available": True},
    "trail": [{"step": "rag_advice", "mode": "llm"}, {"step": "decision", "decision": "advise"}],
}


class FakeResp:
    def __init__(self, code, text=""):
        self.status_code, self.text = code, text


def test_short_message_fits_sms_and_warns_about_rain():
    text = format_short(ADVISE)
    assert len(text) <= SMS_LIMIT
    assert "Late blight" in text and "DELAY spraying" in text and "**" not in text


def test_full_message_has_advice_and_sources_without_markdown():
    text = format_full(ADVISE)
    assert "mancozeb" in text and "https://agritech.tnau.ac.in/x" in text
    assert "**" not in text and "[1]" not in text


def test_retry_then_success(monkeypatch):
    calls = iter([requests.ConnectionError("down"), FakeResp(503), FakeResp(200)])
    def fake():
        c = next(calls)
        if isinstance(c, Exception):
            raise c
        return c
    monkeypatch.setattr(senders.time, "sleep", lambda s: None)
    assert senders._with_retry(fake, "telegram") == {"channel": "telegram", "status": "sent", "attempts": 3}


def test_auth_error_is_not_retried(monkeypatch):
    n = []
    monkeypatch.setattr(senders.time, "sleep", lambda s: None)
    out = senders._with_retry(lambda: n.append(1) or FakeResp(401, "bad token"), "sms")
    assert out["status"] == "failed" and len(n) == 1 and "401" in out["error"]


def test_missing_keys_report_not_configured(monkeypatch):
    for k in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    assert senders.send_telegram("hi")["status"] == "not_configured"
    assert senders.send_sms("hi", to="+911234567890")["status"] == "not_configured"


def test_one_broken_channel_does_not_stop_the_others():
    def boom(r):
        raise RuntimeError("bug")
    out = notify(ADVISE, ["sms", "telegram", "pigeon"],
                 senders={"sms": boom, "telegram": lambda r: {"channel": "telegram", "status": "sent"}})
    assert [o["status"] for o in out] == ["failed", "sent", "failed"]


def test_log_round_trip(tmp_path):
    db = tmp_path / "t.db"
    alerts = [{"channel": "telegram", "status": "sent"}, {"channel": "sms", "status": "failed"}]
    row_id = log_detection(ADVISE, alerts, image_name="leaf.jpg", lat=13.08, lon=80.27, db_path=db)
    row = recent(1, db_path=db)[0]
    assert row["id"] == row_id and row["top1"] == "Tomato___Late_blight" and row["decision"] == "advise"
    assert row["alert_status"] == "telegram:sent,sms:failed" and row["delay_spraying"] == 1
    assert row["rag_mode"] == "llm" and row["n_sources"] == 1
