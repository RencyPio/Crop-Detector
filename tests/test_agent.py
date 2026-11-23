import numpy as np
import pytest

from src.agent.confusable_groups import are_confusable
from src.agent.crop_mask import crop_mask
from src.agent.decision_agent import SprayDecisionAgent, involves_spraying
from src.dl.labels import load_class_names

NAMES = load_class_names()


def probs_for(**label_probs) -> np.ndarray:
    p = np.zeros(len(NAMES))
    for label, v in label_probs.items():
        p[NAMES.index(label)] = v
    rest = 1 - p.sum()
    p[p == 0] = rest / (p == 0).sum()
    return p


def L(name):
    return {"TLB": "Tomato___Late_blight", "TEB": "Tomato___Early_blight", "PEB": "Potato___Early_blight",
            "TH": "Tomato___healthy", "GBR": "Grape___Black_rot", "AS": "Apple___Apple_scab",
            "MITE": "Tomato___Spider_mites Two-spotted_spider_mite", "TS": "Tomato___Target_Spot"}[name]


def P(**kw):
    return probs_for(**{L(k): v for k, v in kw.items()})


def fake_rag(label):
    return {"advice": f"Spray mancozeb 2 g/l for {label}. [1]", "sources": [{"n": 1, "url": "u"}], "mode": "llm"}


def fake_weather(rain):
    return lambda lat, lon: {"available": True, "rain_prob_next_h": rain, "provider": "fake"}


def agent(rain=0.1, rag=fake_rag):
    return SprayDecisionAgent(rag_advice=rag, weather_check=fake_weather(rain), class_names=NAMES)


def test_confident_disease_no_rain_advises_spraying():
    out = agent(rain=0.1).decide(P(TLB=0.95), crop="Tomato", lat=13, lon=80)
    assert out["decision"] == "advise"
    assert out["weather"]["modified_advice"] is False
    assert "suitable for spraying" in out["advice"]


def test_rain_expected_delays_spraying():
    out = agent(rain=0.8).decide(P(TLB=0.95), crop="Tomato", lat=13, lon=80)
    assert out["decision"] == "advise" and out["weather"]["modified_advice"] is True
    assert "Delay spraying" in out["advice"]


def test_rain_exactly_at_threshold_delays():
    out = agent(rain=0.60).decide(P(TLB=0.95), crop="Tomato", lat=13, lon=80)
    assert out["weather"]["modified_advice"] is True


def test_weather_unavailable_warns_instead_of_assuming_dry():
    a = SprayDecisionAgent(rag_advice=fake_rag, class_names=NAMES,
                           weather_check=lambda lat, lon: {"available": False, "reason": "timeout"})
    out = a.decide(P(TLB=0.95), crop="Tomato", lat=13, lon=80)
    assert "could not check the forecast" in out["advice"]
    assert out["weather"]["available"] is False


def test_low_confidence_asks_for_new_photo_and_gives_no_advice():
    calls = []
    out = agent(rag=lambda l: calls.append(l) or fake_rag(l)).decide(P(TLB=0.40, GBR=0.30))
    assert out["decision"] == "uncertain_retake_photo" and out["reason"] == "low_confidence"
    assert out["advice"] is None and calls == []


def test_confusable_close_call_is_ambiguous_not_just_retake():
    out = agent().decide(P(TLB=0.52, TEB=0.44), crop="Tomato")
    assert out["decision"] == "ambiguous_pair"
    assert "Late blight" in out["message"] and "Early blight" in out["message"]


def test_spider_mites_vs_target_spot_is_ambiguous():
    out = agent().decide(P(MITE=0.50, TS=0.40))
    assert out["decision"] == "ambiguous_pair"


def test_close_call_between_unrelated_diseases_is_just_low_confidence():
    out = agent().decide(P(TLB=0.50, GBR=0.45))
    assert out["decision"] == "uncertain_retake_photo" and out["reason"] == "low_confidence"


def test_confusable_pair_with_clear_gap_is_advised():
    out = agent().decide(P(MITE=0.90, TS=0.08), lat=13, lon=80)
    assert out["decision"] == "advise"


def test_healthy_skips_rag_and_weather():
    calls = []
    out = agent(rag=lambda l: calls.append(l) or fake_rag(l)).decide(P(TH=0.97), crop="Tomato")
    assert out["decision"] == "healthy" and calls == [] and out["weather"] is None


def test_crop_selector_fixes_wrong_crop_prediction():
    raw = P(PEB=0.60, TEB=0.36)
    out = agent().decide(raw, crop="Tomato", lat=13, lon=80)
    assert out["predictions_raw"][0]["label"] == L("PEB")
    assert out["predictions"][0]["label"] == L("TEB")
    assert out["decision"] == "advise"


def test_very_low_crop_mass_is_refused_even_if_mask_might_be_right():
    out = agent().decide(P(PEB=0.79, TEB=0.15), crop="Tomato")
    assert out["decision"] == "uncertain_retake_photo" and out["reason"] == "crop_mismatch"


def test_photo_of_another_crop_is_rejected_not_renormalised():
    out = agent().decide(P(GBR=0.97), crop="Tomato")
    assert out["decision"] == "uncertain_retake_photo" and out["reason"] == "crop_mismatch"


def test_disease_outside_knowledge_base_gets_no_weather_step():
    no_kb = lambda l: {"advice": "No documents yet.", "sources": [], "mode": "no_knowledge"}
    out = agent(rag=no_kb).decide(P(AS=0.95), crop="Apple", lat=13, lon=80)
    assert out["decision"] == "advise" and out["weather"] is None


def test_trail_explains_every_step():
    out = agent(rain=0.8).decide(P(TLB=0.95), crop="Tomato", lat=13, lon=80)
    steps = [s["step"] for s in out["trail"]]
    assert steps == ["crop_mask", "confusable_check", "confidence_gate", "rag_advice", "weather_check", "decision"]


def test_helpers():
    assert are_confusable(L("TLB"), L("TEB")) and not are_confusable(L("TLB"), L("PEB"))
    assert involves_spraying("Spray Zineb 0.2%") and not involves_spraying("Remove infected leaves.")
    masked, mass = crop_mask(P(TLB=0.5, PEB=0.3), "Tomato", NAMES)
    assert np.isclose(masked.sum(), 1) and 0.5 < mass < 0.8
    with pytest.raises(ValueError):
        crop_mask(P(TLB=0.5), "Banana", NAMES)
