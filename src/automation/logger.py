import json
import sqlite3
from datetime import datetime, timezone

from src.dl.config import PROJECT_ROOT

DB_PATH = PROJECT_ROOT / "data" / "detections.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT NOT NULL,
    image_name    TEXT,
    crop_selected TEXT,
    top1 TEXT, p1 REAL, top2 TEXT, p2 REAL, top3 TEXT, p3 REAL,
    crop_mass     REAL,
    decision      TEXT NOT NULL,
    reason        TEXT,
    advice_text   TEXT,
    rag_mode      TEXT,
    n_sources     INTEGER,
    rain_prob     REAL,
    delay_spraying INTEGER,
    lat REAL, lon REAL,
    alert_channel TEXT,
    alert_status  TEXT,
    trail_json    TEXT
)"""


def connect(db_path=DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(SCHEMA)
    return conn


def log_detection(result: dict, alerts: list[dict], image_name: str | None = None,
                  lat: float | None = None, lon: float | None = None, db_path=DB_PATH) -> int:
    preds = (result.get("predictions") or []) + [{"label": None, "prob": None}] * 3
    weather = result.get("weather") or {}
    rag_step = next((s for s in result.get("trail", []) if s["step"] == "rag_advice"), {})
    row = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "image_name": image_name, "crop_selected": result.get("crop_selected"),
        "top1": preds[0]["label"], "p1": preds[0]["prob"], "top2": preds[1]["label"], "p2": preds[1]["prob"],
        "top3": preds[2]["label"], "p3": preds[2]["prob"], "crop_mass": result.get("crop_mass"),
        "decision": result["decision"], "reason": result.get("reason"), "advice_text": result.get("advice"),
        "rag_mode": rag_step.get("mode"), "n_sources": len(result.get("sources") or []),
        "rain_prob": weather.get("rain_prob_next_6h"), "delay_spraying": int(bool(weather.get("modified_advice"))),
        "lat": lat, "lon": lon,
        "alert_channel": ",".join(a["channel"] for a in alerts),
        "alert_status": ",".join(f"{a['channel']}:{a['status']}" for a in alerts),
        "trail_json": json.dumps(result.get("trail", [])),
    }
    with connect(db_path) as conn:
        cur = conn.execute(f"INSERT INTO detections ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})",
                           list(row.values()))
        return cur.lastrowid


def recent(n: int = 20, db_path=DB_PATH) -> list[dict]:
    with connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute("SELECT * FROM detections ORDER BY id DESC LIMIT ?", (n,))]


if __name__ == "__main__":
    for r in recent():
        print(r["id"], r["ts"], r["crop_selected"], r["top1"], r["p1"], r["decision"], r["alert_status"])
