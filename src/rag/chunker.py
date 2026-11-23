import re
from dataclasses import dataclass, field

from src.dl.labels import crop_of
from src.rag.config import CHUNK_WORDS, OVERLAP_WORDS, RAW_DIR
from src.rag.ingest import load_sources

HEADING = re.compile(
    r"^(symptoms?|management|control|cultural|chemical|biological|organic|prevention|"
    r"favou?rable conditions?|epidemiology|disease cycle|mode of spread|spread|pathogen|"
    r"causal organism|identification|resistant varieties?|fungicides?|ipm|idm)\b[^.]{0,40}:?$",
    re.IGNORECASE,
)
CAPTION = re.compile(r"^(figure|fig\.)\s*\d+", re.IGNORECASE)
MENU_WORDS = re.compile(r"\b(home|about us|contact|publications|success stories)\b", re.IGNORECASE)


def is_menu(line: str) -> bool:
    cells = [c.strip() for c in line.split("|")]
    return (len(cells) >= 4 and all(len(c.split()) <= 3 for c in cells)
            and not re.search(r"\d", line) and bool(MENU_WORDS.search(line)))


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


def _paragraphs(text: str):
    section = "general"
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("# "):
            continue
        if CAPTION.match(line):
            continue
        if is_menu(line):
            continue
        if HEADING.match(line) and len(line.split()) <= 6:
            section = line.rstrip(":").strip().lower()
            continue
        yield section, line


def chunk_text(text: str, size: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS):
    chunks, words, sections = [], [], []
    for section, para in _paragraphs(text):
        para_words = para.split()
        if words and len(words) + len(para_words) > size:
            chunks.append((sections[0], " ".join(words)))
            words = words[-overlap:]
            sections = [sections[-1]]
        words.extend(para_words)
        sections.append(section)
        while len(words) > size * 1.5:
            chunks.append((sections[-1], " ".join(words[:size])))
            words = words[size - overlap:]
    if len(words) >= 20:
        chunks.append((sections[0] if sections else "general", " ".join(words)))
    return chunks


def build_chunks() -> list[Chunk]:
    all_chunks = []
    for src in load_sources():
        path = RAW_DIR / f"{src['doc_id']}.txt"
        if not path.exists():
            print(f"missing {path.name} (run python -m src.rag.ingest)")
            continue
        for i, (section, text) in enumerate(chunk_text(path.read_text(encoding="utf-8"))):
            all_chunks.append(Chunk(
                id=f"{src['doc_id']}__{i:03d}",
                text=text,
                metadata={
                    "disease_class": src["disease_class"],
                    "crop": crop_of(src["disease_class"]),
                    "doc_id": src["doc_id"],
                    "source_title": src["title"],
                    "source_url": src["url"],
                    "publisher": src["publisher"],
                    "region": src["region"],
                    "doc_type": src["doc_type"],
                    "section": section,
                },
            ))
    return all_chunks
