import sys
from functools import lru_cache

try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

from src.rag.config import CHROMA_DIR, COLLECTION


@lru_cache(maxsize=1)
def get_embedder():
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    return DefaultEmbeddingFunction()


def embed(texts: list[str]) -> list[list[float]]:
    return [list(map(float, v)) for v in get_embedder()(texts)]


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


def ensure_knowledge_base():
    if get_collection().count() > 0:
        return False
    from src.rag.ingest import main as ingest
    ingest()
    build_index()
    return True


if __name__ == "__main__":
    build_index()
