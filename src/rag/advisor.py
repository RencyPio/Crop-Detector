from src.dl.labels import is_healthy, pretty_label
from src.rag.generate import generate_advice
from src.rag.retrieve import retrieve


def rag_advice(disease_class: str) -> dict:
    if is_healthy(disease_class):
        return {"advice": f"{pretty_label(disease_class)}: the leaf looks healthy. No treatment needed.",
                "sources": [], "mode": "healthy"}
    hits = retrieve(disease_class)
    result = generate_advice(disease_class, hits)
    if result["mode"] == "no_knowledge":
        result["advice"] = (f"No documents for {pretty_label(disease_class)} in the knowledge base yet. "
                            "Please consult your local agriculture officer.")
    result.setdefault("rejected", [])
    result["retrieved"] =[{"score": h["score"], "doc_id": h["metadata"]["doc_id"],
                            "section": h["metadata"]["section"]} for h in hits]
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("label", nargs="?", default="Tomato___Late_blight")
    ap.add_argument("--image", help="classify this image first, then advise on the top-1 label")
    args = ap.parse_args()

    label = args.label
    if args.image:
        from src.dl.predict import Predictor
        p = Predictor()
        top = p.top_k(p.predict(args.image), 3)
        print("Prediction:", ", ".join(f"{t['label']} {t['prob']:.1%}" for t in top))
        label = top[0]["label"]

    out = rag_advice(label)
    print(f"\nMode: {out['mode']}  Model: {out.get('model', '-')}  Grounding: {out.get('grounding', '-')}")
    print(f"\n{out['advice']}\n\nSources:")
    for s in out["sources"]:
        print(f"  [{s['n']}] {s['title']} - {s['publisher']} ({s['region']}) {s['url']}")
