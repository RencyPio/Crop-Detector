from src.dl.labels import pretty_label
from src.rag.config import TOP_K
from src.rag.vector_store import embed, get_collection


def build_query(disease_class: str) -> str:
    return f"{pretty_label(disease_class)} symptoms management treatment control fungicide prevention"


def _query(q_emb, where: dict, n: int) -> list[dict]:
    res = get_collection().query(query_embeddings=q_emb, n_results=n, where=where)
    return [{"id": i, "text": doc, "metadata": meta, "score": round(1 - dist, 4)}
            for i, doc, meta, dist in zip(res["ids"][0], res["documents"][0],
                                          res["metadatas"][0], res["distances"][0])]


def retrieve(disease_class: str, query: str | None = None, k: int = TOP_K, india_k: int = 2) -> list[dict]:
    q_emb = embed([query or build_query(disease_class)])
    india = _query(q_emb, {"$and": [{"disease_class": disease_class}, {"region": "India"}]}, india_k)
    rest = _query(q_emb, {"disease_class": disease_class}, k)
    seen = {h["id"] for h in india}
    return (india + [h for h in rest if h["id"] not in seen])[:k]


def has_knowledge(disease_class: str) -> bool:
    return len(get_collection().get(where={"disease_class": disease_class}, limit=1)["ids"]) > 0


if __name__ == "__main__":
    import sys
    label = sys.argv[1] if len(sys.argv) > 1 else "Tomato___Late_blight"
    for h in retrieve(label):
        m = h["metadata"]
        print(f"[{h['score']:.3f}] {m['publisher']} ({m['region']}) / {m['section']}\n  {h['text'][:300]}...\n")
