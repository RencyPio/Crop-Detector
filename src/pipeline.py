from functools import lru_cache

from src.agent.decision_agent import SprayDecisionAgent
from src.automation.logger import log_detection
from src.automation.notify import notify


@lru_cache(maxsize=1)
def _components():
    from src.dl.predict import Predictor
    predictor = Predictor()
    return predictor, SprayDecisionAgent(class_names=predictor.class_names)


def run(image, crop: str | None = None, lat: float | None = None, lon: float | None = None,
        channels: list[str] | None = None, image_name: str | None = None, log: bool = True) -> dict:
    predictor, agent = _components()
    result = agent.decide(predictor.predict(image), crop=crop, lat=lat, lon=lon)

    alerts = notify(result, channels)
    result["alert"] = {"channels": [a["channel"] for a in alerts],
                       "status": {a["channel"]: a["status"] for a in alerts}, "details": alerts}
    result["trail"].append({"step": "notify", "results": [{k: a.get(k) for k in ("channel", "status", "error")}
                                                          for a in alerts]})
    if log:
        result["log_id"] = log_detection(result, alerts, image_name=image_name, lat=lat, lon=lon)
    return result


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--crop")
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--channels", default="", help="comma-separated: telegram,sms")
    a = ap.parse_args()

    out = run(a.image, a.crop, a.lat, a.lon, channels=[c for c in a.channels.split(",") if c],
              image_name=Path(a.image).name)
    print(f"Decision: {out['decision']} ({out['reason']})\n{out['message']}\n")
    if out["advice"]:
        print(out["advice"], "\n")
    print("Trail:")
    for step in out["trail"]:
        print("  ", json.dumps(step))
    print(f"\nLogged as detection #{out['log_id']}")
