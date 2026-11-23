import os

import requests
from dotenv import load_dotenv

from src.agent.config import RAIN_WINDOW_HOURS

load_dotenv()
TIMEOUT = 15


def _open_meteo(lat: float, lon: float, hours: int) -> dict:
    r = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={"latitude": lat, "longitude": lon, "hourly": "precipitation_probability,precipitation",
                "forecast_hours": hours, "timezone": "auto"},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    h = r.json()["hourly"]
    probs = [p for p in h["precipitation_probability"] if p is not None]
    return {"rain_prob_next_h": max(probs) / 100 if probs else None,
            "rain_mm_next_h": round(sum(x or 0 for x in h.get("precipitation", [])), 2),
            "provider": "open-meteo", "times": h["time"]}


def _openweathermap(lat: float, lon: float, hours: int, key: str) -> dict:
    r = requests.get(
        "https://api.openweathermap.org/data/2.5/forecast",
        params={"lat": lat, "lon": lon, "appid": key, "units": "metric"},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    steps = r.json()["list"][: max(1, -(-hours // 3))]
    return {"rain_prob_next_h": max(s.get("pop", 0) for s in steps),
            "rain_mm_next_h": round(sum(s.get("rain", {}).get("3h", 0) for s in steps), 2),
            "provider": "openweathermap", "times": [s["dt_txt"] for s in steps]}


def weather_check(lat: float | None, lon: float | None, hours: int = RAIN_WINDOW_HOURS) -> dict:
    if lat is None or lon is None:
        return {"available": False, "reason": "no location given"}
    errors = []
    key = os.getenv("OPENWEATHER_API_KEY")
    providers = ([lambda: _openweathermap(lat, lon, hours, key)] if key else []) + [lambda: _open_meteo(lat, lon, hours)]
    for call in providers:
        try:
            return {"available": True, "window_hours": hours, **call()}
        except Exception as e:
            errors.append(str(e)[:120])
    return {"available": False, "reason": "; ".join(errors)}


if __name__ == "__main__":
    import sys
    lat, lon = (float(sys.argv[1]), float(sys.argv[2])) if len(sys.argv) > 2 else (13.08, 80.27)
    print(weather_check(lat, lon))
