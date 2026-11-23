import csv
import json
import re
import statistics
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.agent import config as C
from src.agent.decision_agent import SprayDecisionAgent
from src.dl.config import MODELS_DIR
from src.dl.labels import crop_of, is_healthy, pretty_label

OUT = Path("reports/eval")
EXTS = {".jpg", ".jpeg", ".png"}
BLUE, ORANGE, INK, MUTED, GRID, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#6b6b68", "#e4e3df", "#fcfcfb"


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def per_class_metrics():
    rows = []
    for line in (MODELS_DIR / "classification_report.txt").read_text().splitlines():
        m = re.match(r"\s*(\S.*?___\S.*?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$", line)
        if m:
            rows.append({"class": m.group(1), "precision": float(m.group(2)), "recall": float(m.group(3)),
                         "f1": float(m.group(4)), "support": int(m.group(5))})
    rows.sort(key=lambda r: r["f1"])
    with open(OUT / "per_class_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    fig, ax = plt.subplots(figsize=(8, 10), facecolor=SURFACE)
    names = [pretty_label(r["class"]) for r in rows]
    f1 = [r["f1"] for r in rows]
    colors = [ORANGE if v < 0.90 else BLUE for v in f1]
    ax.barh(names, f1, color=colors, height=0.6)
    ax.axvline(0.90, color=MUTED, linewidth=1, linestyle="--")
    ax.set_xlim(0.6, 1.0)
    ax.set_xlabel("F1 score (validation, 8,129 images)", color=MUTED, fontsize=9)
    for y, (v, r) in enumerate(zip(f1, rows)):
        if v < 0.90:
            ax.text(0.905, y, f"{v:.3f}  (n={r['support']})", va="center", fontsize=8, color=INK)
    style(ax)
    ax.tick_params(axis="y", labelsize=8, colors=INK)
    weak = sum(v < 0.90 for v in f1)
    fig.suptitle(f"{weak} of 38 classes fall below 0.90 F1", x=0.02, ha="left", fontsize=11, color=INK)
    ax.set_title("Orange: look-alike leaf spots, and Potato healthy (n=22)", loc="left", fontsize=9, color=MUTED,
                 pad=8)
    fig.tight_layout()
    fig.savefig(OUT / "per_class_f1.png", dpi=200)
    plt.close(fig)
    return rows


def load_images(predictor):
    items = []
    for d in sorted(p for p in Path("data/sanity_check").iterdir() if p.is_dir()):
        for f in sorted(p for p in d.iterdir() if p.suffix.lower() in EXTS):
            items.append(("sanity", f, d.name))
    for f in sorted(p for p in Path("data/ood_test_images").iterdir() if p.suffix.lower() in EXTS):
        items.append(("ood", f, None))
    times = []
    probs = []
    predictor.predict(items[0][1])
    for _, f, _ in items:
        t0 = time.perf_counter()
        probs.append(predictor.predict(f))
        times.append((time.perf_counter() - t0) * 1000)
    return items, probs, times


def run_agent(items, probs, names, t_low):
    old = C.T_LOW
    C.T_LOW = t_low
    agent = SprayDecisionAgent(rag_advice=lambda l: {"advice": "spray", "sources": [], "mode": "llm"},
                               weather_check=lambda lat, lon: {"available": False}, class_names=names)
    s = {"t_low": t_low, "acted": 0, "correct": 0, "wrong_advice": 0, "deferred": 0, "ood_advised_no_crop": 0,
         "ood_advised_tomato": 0}
    for (kind, _, true), p in zip(items, probs):
        if kind == "sanity":
            out = agent.decide(p, crop=crop_of(true))
            top = out["predictions"][0]["label"]
            if out["decision"] in ("advise", "healthy"):
                s["acted"] += 1
                s["correct"] += top == true
                s["wrong_advice"] += top != true and out["decision"] == "advise"
            else:
                s["deferred"] += 1
        else:
            s["ood_advised_no_crop"] += agent.decide(p)["decision"] == "advise"
            s["ood_advised_tomato"] += agent.decide(p, crop="Tomato")["decision"] == "advise"
    C.T_LOW = old
    return s


def threshold_sweep(items, probs, names):
    n_sanity = sum(k == "sanity" for k, _, _ in items)
    bare_wrong = sum(names[int(np.argmax(p))] != t and not is_healthy(names[int(np.argmax(p))])
                     for (k, _, t), p in zip(items, probs) if k == "sanity")
    rows = [run_agent(items, probs, names, round(t, 2)) for t in np.arange(0.50, 0.96, 0.05)]
    for r in rows:
        r["coverage_pct"] = round(100 * r["acted"] / n_sanity, 1)
    with open(OUT / "threshold_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    t = [r["t_low"] for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.4), facecolor=SURFACE)
    a1.plot(t, [r["coverage_pct"] for r in rows], color=BLUE, linewidth=2, marker="o", markersize=5)
    a1.set_ylim(80, 100)
    a1.set_ylabel("Known images acted on (%)", color=MUTED, fontsize=9)
    a1.set_title("At 0.70 the agent still acts on 92% of known images", loc="left", fontsize=10, color=INK)
    a2.plot(t, [r["ood_advised_tomato"] for r in rows], color=ORANGE, linewidth=2, marker="o", markersize=5,
            label="Non-crop photos advised (crop = Tomato, of 7)")
    a2.plot(t, [r["wrong_advice"] for r in rows], color=BLUE, linewidth=2, marker="o", markersize=5,
            label=f"Wrong advice on known images (bare model: {bare_wrong})")
    a2.set_ylim(-0.3, 7.3)
    a2.set_ylabel("Count", color=MUTED, fontsize=9)
    a2.set_title("From 0.60 up, no wrong advice; non-crop photos still pass", loc="left", fontsize=10, color=INK)
    a2.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1)
    for a in (a1, a2):
        style(a)
        a.grid(axis="y", color=GRID, linewidth=0.8)
        a.grid(axis="x", visible=False)
        a.axvline(C.T_LOW, color=MUTED, linewidth=1, linestyle="--")
        a.set_xlabel("Confidence gate T_LOW (dashed = chosen 0.70)", color=MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "threshold_sweep.png", dpi=200)
    plt.close(fig)
    return rows, bare_wrong


def review_sheet():
    src = Path("reports/rag_all_diseases.csv")
    if not src.exists():
        return 0
    rows = [r for r in csv.DictReader(open(src, encoding="utf-8")) if r["mode"] != "no_knowledge"]
    with open(OUT / "advice_review_sheet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["disease_class", "answer_file", "mode", "sources", "regions", "grounded (Y/N)",
                    "correct (Y/N)", "cited (Y/N)", "safe (Y/N)", "invented or reworded claims", "notes"])
        for r in rows:
            fname = re.sub(r"[^\w.-]", "_", r["disease_class"]) + ".txt"
            w.writerow([r["disease_class"], f"reports/rag_answers/{fname}", r["mode"], r["n_sources"], r["regions"],
                        "", "", "", "", "", ""])
    return len(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    per_class = per_class_metrics()

    from src.dl.predict import Predictor
    t0 = time.perf_counter()
    predictor = Predictor()
    load_s = time.perf_counter() - t0
    items, probs, times = load_images(predictor)
    latency = {"model_load_s": round(load_s, 2), "images": len(times),
               "mean_ms": round(statistics.mean(times), 1), "median_ms": round(statistics.median(times), 1),
               "p95_ms": round(sorted(times)[int(0.95 * len(times)) - 1], 1)}
    (OUT / "latency.json").write_text(json.dumps(latency, indent=2))

    sweep, bare_wrong = threshold_sweep(items, probs, predictor.class_names)
    n_review = review_sheet()

    chosen = next(r for r in sweep if abs(r["t_low"] - C.T_LOW) < 1e-9)
    weak = [r for r in per_class if r["f1"] < 0.90]
    print(f"Weak classes (F1 < 0.90): {len(weak)}")
    for r in weak:
        print(f"  {r['f1']:.3f}  {r['class']}  (n={r['support']})")
    print(f"Latency: {latency}")
    print(f"At T_LOW={C.T_LOW}: coverage {chosen['coverage_pct']}%, wrong advice {chosen['wrong_advice']} "
          f"(bare {bare_wrong}), non-crop advised {chosen['ood_advised_no_crop']}/7 no crop, "
          f"{chosen['ood_advised_tomato']}/7 crop=Tomato")
    print(f"Review sheet rows: {n_review}")
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
