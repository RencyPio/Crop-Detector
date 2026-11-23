from functools import lru_cache

from src.rag.config import CHROMA_DIR, COLLECTION, EMBED_MODEL


@lru_cache(maxsize=1)
def get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    return get_embedder().encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()


@lru_cache(maxsize=1)
def get_client():
    import chromadb
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection():
    return get_client().get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def build_index():
    from src.rag.chunker import build_chunks

    chunks = build_chunks()
    if not chunks:
        raise SystemExit("No chunks. Run `python -m src.rag.ingest` first.")

    client = get_client()
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = get_collection()
    col.add(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[c.metadata for c in chunks],
        embeddings=embed([c.text for c in chunks]),
    )

    per_disease = {}
    for c in chunks:
        per_disease[c.metadata["disease_class"]] = per_disease.get(c.metadata["disease_class"], 0) + 1
    print(f"Indexed {len(chunks)} chunks into '{COLLECTION}' at {CHROMA_DIR}")
    for d, n in sorted(per_disease.items()):
        print(f"  {n:3d}  {d}")


if __name__ == "__main__":
    build_index()
