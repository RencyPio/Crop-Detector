import csv
from collections import Counter
from pathlib import Path

from src.agent.decision_agent import SprayDecisionAgent
from src.dl.labels import crop_of, is_healthy
from src.dl.predict import Predictor

OUT_CSV = Path("reports/agent_scenarios.csv")
EXTS = {".jpg", ".jpeg", ".png"}


def stub_rag(label):
    return {"advice": "(stub) spray advice", "sources": [], "mode": "llm"}


def main():
    pred = Predictor()
    agent = SprayDecisionAgent(rag_advice=stub_rag,
                               weather_check=lambda lat, lon: {"available": False, "reason": "stub"},
                               class_names=pred.class_names)
    rows = []

    for d in sorted(p for p in Path("data/sanity_check").iterdir() if p.is_dir()):
        for f in sorted(p for p in d.iterdir() if p.suffix.lower() in EXTS):
            probs = pred.predict(f)
            out = agent.decide(probs, crop=crop_of(d.name))
            rows.append({"set": "sanity", "image": f.name, "true": d.name, "crop": crop_of(d.name),
                         "cnn_top1": out["predictions_raw"][0]["label"], "agent_top1": out["predictions"][0]["label"],
                         "p1": out["predictions"][0]["prob"], "crop_mass": out["crop_mass"],
                         "decision": out["decision"], "reason": out["reason"]})

    for f in sorted(p for p in Path("data/ood_test_images").iterdir() if p.suffix.lower() in EXTS):
        probs = pred.predict(f)
        for crop in (None, "Tomato"):
            out = agent.decide(probs, crop=crop)
            rows.append({"set": "ood", "image": f.name, "true": "not_a_supported_plant", "crop": crop or "",
                         "cnn_top1": out["predictions_raw"][0]["label"], "agent_top1": out["predictions"][0]["label"],
                         "p1": out["predictions"][0]["prob"], "crop_mass": out["crop_mass"],
                         "decision": out["decision"], "reason": out["reason"]})

    OUT_CSV.parent.mkdir(exist_ok=True)
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    san = [r for r in rows if r["set"] == "sanity"]
    print(f"SANITY SET ({len(san)} images, farmer selects the true crop)")
    print("  decisions:", dict(Counter(r["decision"] for r in san)))
    acted = [r for r in san if r["decision"] in ("advise", "healthy")]
    correct = sum(r["agent_top1"] == r["true"] for r in acted)
    base_wrong = sum(r["cnn_top1"] != r["true"] for r in san)
    agent_wrong = sum(r["agent_top1"] != r["true"] for r in acted)
    wrong_spray_base = sum(r["cnn_top1"] != r["true"] and not is_healthy(r["cnn_top1"]) for r in san)
    wrong_spray_agent = sum(r["agent_top1"] != r["true"] and r["decision"] == "advise" for r in acted)
    print(f"  bare CNN : wrong diagnosis {base_wrong}/{len(san)}, wrong treatment advice {wrong_spray_base}")
    print(f"  agent    : acted on {len(acted)}/{len(san)} ({correct} correct), "
          f"wrong diagnosis acted on {agent_wrong}, wrong treatment advice {wrong_spray_agent}")
    print(f"  coverage cost: {len(san) - len(acted)} images sent back for a retake / expert check")
    for r in san:
        if r["cnn_top1"] != r["true"] or r["decision"] not in ("advise", "healthy"):
            print(f"    {r['true'][:38]:38s} cnn={r['cnn_top1'][:30]:30s} agent={r['agent_top1'][:30]:30s} "
                  f"p1={r['p1']:.2f} mass={r['crop_mass']:.2f} -> {r['decision']} ({r['reason']})")

    print("\nOOD PHOTOS (mint / hibiscus / tulsi: every 'advise' is wrong advice)")
    for crop in ("", "Tomato"):
        sub = [r for r in rows if r["set"] == "ood" and r["crop"] == crop]
        print(f"  crop={crop or 'none':6s}: {dict(Counter(r['decision'] for r in sub))}")
    print(f"\nSaved {OUT_CSV}")


if __name__ == "__main__":
    main()
