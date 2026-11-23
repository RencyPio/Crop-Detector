import csv
import re
import time
from pathlib import Path

from src.dl.labels import is_healthy, load_class_names
from src.rag.advisor import rag_advice

OUT_CSV = Path("reports/rag_all_diseases.csv")
OUT_DIR = Path("reports/rag_answers")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for label in load_class_names():
        if is_healthy(label):
            continue
        out = rag_advice(label)
        g = out.get("grounding") or {}
        regions = sorted({s["region"] for s in out.get("sources", [])})
        rejected = out.get("rejected", [])
        rows.append({"disease_class": label, "mode": out["mode"], "attempts": out.get("attempts", ""),
                     "n_sources": len(out.get("sources", [])), "regions": "+".join(regions),
                     "uncited_bullets": len(g.get("uncited_bullets", [])) if out["mode"] == "llm" else "",
                     "rejected_numbers": " | ".join(" ".join(r["unsupported_numbers"]) for r in rejected),
                     "rejected_lines": " || ".join(ln for r in rejected for ln in r.get("flagged_lines", []))})
        safe = re.sub(r"[^\w.-]", "_", label)
        srcs = "\n".join(f"[{s['n']}] {s['title']} - {s['publisher']} ({s['region']}) {s['url']}" for s in out.get("sources", []))
        (OUT_DIR / f"{safe}.txt").write_text(f"{label}\nmode: {out['mode']}\n\n{out['advice']}\n\nSources:\n{srcs}\n",
                                             encoding="utf-8")
        print(f"{rows[-1]['mode']:20s} sources={rows[-1]['n_sources']} {rows[-1]['regions']:20s} {label}")
        time.sleep(8)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    llm = sum(r["mode"] == "llm" for r in rows)
    print(f"\n{llm}/{len(rows)} diseases got grounded LLM advice; saved {OUT_CSV}")


if __name__ == "__main__":
    main()
