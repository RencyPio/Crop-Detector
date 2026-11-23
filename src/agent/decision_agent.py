from typing import Callable

import numpy as np

from src.agent import config as C
from src.agent.confusable_groups import are_confusable
from src.agent.crop_mask import crop_mask
from src.dl.labels import is_healthy, load_class_names, pretty, pretty_label

RETAKE_TIPS = "Take one leaf in daylight, fill the frame, keep it in focus, and avoid shadows."


def _top(probs: np.ndarray, names: list[str], k: int = 3) -> list[dict]:
    idx = np.argsort(probs)[::-1][:k]
    return [{"label": names[i], "prob": round(float(probs[i]), 4)} for i in idx]


def involves_spraying(advice: str | None) -> bool:
    return bool(advice) and any(w in advice.lower() for w in C.SPRAY_WORDS)


class SprayDecisionAgent:
    def __init__(self, rag_advice: Callable | None = None, weather_check: Callable | None = None,
                 class_names: list[str] | None = None):
        if rag_advice is None:
            from src.rag.advisor import rag_advice
        if weather_check is None:
            from src.agent.weather_check import weather_check
        self.rag_advice = rag_advice
        self.weather_check = weather_check
        self.class_names = class_names or load_class_names()

    def decide(self, probs: np.ndarray, crop: str | None = None,
               lat: float | None = None, lon: float | None = None) -> dict:
        names, trail = self.class_names, []
        out = {"crop_selected": crop, "predictions_raw": _top(probs, names), "decision": None,
               "reason": None, "message": None, "advice": None, "sources": [],
               "weather": None, "trail": trail}

        probs, crop_mass = crop_mask(probs, crop, names)
        out["crop_mass"] = round(crop_mass, 4)
        top = out["predictions"] = _top(probs, names)
        trail.append({"step": "crop_mask", "crop": crop, "crop_mass": out["crop_mass"],
                      "top1": top[0]["label"], "p1": top[0]["prob"]})
        if crop and crop_mass < C.MIN_CROP_MASS:
            return self._finish(out, "uncertain_retake_photo", "crop_mismatch",
                                f"This photo does not look like a {pretty(crop)} leaf to the model "
                                f"(only {crop_mass:.0%} of its belief was on {pretty(crop)} diseases). "
                                f"Check the crop you selected, or retake the photo. {RETAKE_TIPS}")

        l1, l2 = top[0]["label"], top[1]["label"]
        p1, p2 = top[0]["prob"], top[1]["prob"]
        ambiguous = are_confusable(l1, l2) and (p1 - p2) < C.AMBIGUITY_MARGIN and (p1 + p2) >= C.T_LOW
        trail.append({"step": "confusable_check", "pair": [l1, l2], "gap": round(p1 - p2, 4),
                      "pair_mass": round(p1 + p2, 4), "margin": C.AMBIGUITY_MARGIN, "ambiguous": ambiguous})
        if ambiguous:
            return self._finish(out, "ambiguous_pair", "confusable_pair",
                                f"This could be {pretty_label(l1)} ({p1:.0%}) or {pretty_label(l2)} ({p2:.0%}). "
                                "These look alike but need different treatment. Compare the symptoms "
                                "with both, or show the plant to your local agriculture officer before spraying.")

        trail.append({"step": "confidence_gate", "p1": p1, "threshold": C.T_LOW, "passed": p1 >= C.T_LOW})
        if p1 < C.T_LOW:
            options = ", ".join(f"{pretty_label(t['label'])} ({t['prob']:.0%})" for t in top)
            return self._finish(out, "uncertain_retake_photo", "low_confidence",
                                f"I'm not confident enough to recommend a treatment ({p1:.0%}). "
                                f"Possible: {options}. {RETAKE_TIPS}")

        if is_healthy(l1):
            trail.append({"step": "healthy_shortcut", "label": l1})
            return self._finish(out, "healthy", None,
                                f"The leaf looks healthy ({pretty_label(l1)}, {p1:.0%}). No treatment needed.")

        rag = self.rag_advice(l1)
        out["advice"], out["sources"] = rag["advice"], rag.get("sources", [])
        trail.append({"step": "rag_advice", "label": l1, "mode": rag.get("mode"),
                      "n_sources": len(out["sources"])})

        if rag.get("mode") in ("llm", "extractive", "extractive_fallback") and involves_spraying(out["advice"]):
            out["weather"] = self._weather_step(out, lat, lon, trail)
        else:
            trail.append({"step": "weather_check", "skipped": "advice has no spraying"})

        return self._finish(out, "advise", None, f"Diagnosis: {pretty_label(l1)} ({p1:.0%}).")

    def _weather_step(self, out: dict, lat, lon, trail: list) -> dict:
        w = self.weather_check(lat, lon)
        rain = w.get("rain_prob_next_h") if w.get("available") else None
        delay = rain is not None and rain >= C.RAIN_PROB_THRESHOLD
        if delay:
            out["advice"] += (f"\n\n**Weather:** rain is likely in the next {C.RAIN_WINDOW_HOURS} hours "
                              f"({rain:.0%} chance). Delay spraying until after the rain, because rain "
                              "washes the product off the leaves.")
        elif rain is None:
            out["advice"] += ("\n\n**Weather:** I could not check the forecast. Do not spray if rain is "
                              f"expected in the next {C.RAIN_WINDOW_HOURS} hours.")
        else:
            out["advice"] += (f"\n\n**Weather:** low rain chance in the next {C.RAIN_WINDOW_HOURS} hours "
                              f"({rain:.0%}). Conditions are suitable for spraying.")
        trail.append({"step": "weather_check", "provider": w.get("provider"), "rain_prob": rain,
                      "threshold": C.RAIN_PROB_THRESHOLD, "delay_spraying": delay,
                      "error": None if w.get("available") else w.get("reason")})
        return {"rain_prob_next_6h": rain, "modified_advice": delay, "provider": w.get("provider"),
                "available": bool(w.get("available"))}

    @staticmethod
    def _finish(out: dict, decision: str, reason: str | None, message: str) -> dict:
        out["decision"], out["reason"], out["message"] = decision, reason, message
        out["trail"].append({"step": "decision", "decision": decision, "reason": reason})
        return out
