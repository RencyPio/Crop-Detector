from src.dl.config import PROJECT_ROOT

KB_DIR = PROJECT_ROOT / "data" / "knowledge_base"
SOURCES_CSV = KB_DIR / "sources.csv"
RAW_DIR = KB_DIR / "raw"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"
COLLECTION = "crop_treatment_kb"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_WORDS = 250
OVERLAP_WORDS = 50

TOP_K = 5
