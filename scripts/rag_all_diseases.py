import csv
import re
import sys
import time
from pathlib import Path

from src.dl.labels import is_healthy, load_class_names
from src.rag.advisor import rag_advice

OUT_CSV = Path("reports/rag_all_diseases.csv")
OUT_DIR = Path("reports/rag_answers")


def answer_path(label):
    return OUT_DIR / (re.sub(r"[^\w.-]", "_", label) + ".txt")


def has_llm_answer(label):
    p = answer_path(label)
    return p.exists() and "mode: llm" in p.read_text(encoding="utf-8").splitlines()[:3]


def load_rows():
    if not OUT_CSV.exists():
        return {}
    return {r["disease_class"]: r for r in csv.DictReader(open(OUT_CSV, encoding="utf-8"))}


def save_rows(rows):
    order = [l for l in load_class_names() if l in rows]
    fields = ["disease_class", "mode", "attempts", "n_sources", "regions", "uncited_bullets", "rejected_numbers",
              "rejected_lines"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows[l] for l in order)


def main(redo_all=False):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    for label in load_class_names():
        if is_healthy(label):
            continue
        if not redo_all and has_llm_answer(label):
            print(f"{'kept':20s} {label}")
            continue
        out = rag_advice(label)
        if "tokens per day" in out.get("llm_error", "").lower():
            print("\nGroq daily token limit reached. Stopping here; existing answers were kept.")
            print("Run this script again later and it will continue with the missing diseases.")
            break
        g = out.get("grounding") or {}
        regions = sorted({s["region"] for s in out.get("sources", [])})
        rejected = out.get("rejected", [])
        rows[label] = {"disease_class": label, "mode": out["mode"], "attempts": out.get("attempts", ""),
                       "n_sources": len(out.get("sources", [])), "regions": "+".join(regions),
                       "uncited_bullets": len(g.get("uncited_bullets", [])) if out["mode"] == "llm" else "",
                       "rejected_numbers": " | ".join(" ".join(r["unsupported_numbers"]) for r in rejected),
                       "rejected_lines": " || ".join(ln for r in rejected for ln in r.get("flagged_lines", []))}
        srcs = "\n".join(f"[{s['n']}] {s['title']} - {s['publisher']} ({s['region']}) {s['url']}"
                         for s in out.get("sources", []))
        answer_path(label).write_text(f"{label}\nmode: {out['mode']}\n\n{out['advice']}\n\nSources:\n{srcs}\n",
                                      encoding="utf-8")
        save_rows(rows)
        print(f"{out['mode']:20s} sources={rows[label]['n_sources']} {rows[label]['regions']:20s} {label}")
        time.sleep(8)

    done = [l for l in load_class_names() if not is_healthy(l)]
    llm = sum(has_llm_answer(l) for l in done)
    print(f"\n{llm}/{len(done)} diseases now have a grounded written answer in {OUT_DIR}")


if __name__ == "__main__":
    main(redo_all="--all" in sys.argv)
