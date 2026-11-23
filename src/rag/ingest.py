import csv
import re
import time
from datetime import date

import requests
from bs4 import BeautifulSoup

from src.rag.config import RAW_DIR, SOURCES_CSV

HEADERS = {"User-Agent": "Mozilla/5.0 (student research project; crop disease advisory)"}

DROP_TAGS = ["script", "style", "noscript", "nav", "header", "footer", "form", "iframe", "svg", "button"]
BOILERPLATE = re.compile(
    r"(copyright|all rights reserved|disclaimer|contact us|about us|home\s*\||site map|"
    r"skip to|subscribe|share this|follow us|cookie|print this page|back to top)",
    re.IGNORECASE,
)


def load_sources():
    with open(SOURCES_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


BLOCK_TAGS = ["p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "td", "th", "div", "br", "dt", "dd",
              "blockquote", "section", "table", "ul", "ol"]


def html_to_text(html: bytes | str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(DROP_TAGS):
        tag.decompose()
    for tr in soup.find_all("tr"):
        if tr.find("table") is None:
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            cells = [c for c in cells if c]
            tr.replace_with("\n" + " | ".join(cells) + "\n")
    for tag in soup.find_all(BLOCK_TAGS):
        tag.insert_before("\n")
        tag.insert_after("\n")
    root = soup.find("main") or soup.find("article") or soup.body or soup

    lines = []
    for raw in root.get_text("").splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if len(line.split()) < 3 and not line.endswith(":"):
            continue
        if BOILERPLATE.search(line) and len(line.split()) < 15:
            continue
        if not lines or lines[-1] != line:
            lines.append(line)
    return "\n".join(lines)


def _get(url: str) -> bytes:
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    if resp.status_code != 200 or not resp.content:
        raise RuntimeError(f"HTTP {resp.status_code} with {len(resp.content)} bytes (blocked or rate-limited)")
    return resp.content


def fetch(url: str, retries: int = 3) -> bytes:
    for attempt in range(1, retries + 1):
        try:
            try:
                return _get(url)
            except requests.exceptions.SSLError:
                print(f"  note: HTTPS certificate problem, trying http: {url}")
                return _get(url.replace("https://", "http://", 1))
        except Exception:
            if attempt == retries:
                raise
            time.sleep(5 * attempt)


MIN_WORDS = 80


def pdf_to_text(data: bytes) -> str:
    import io

    import pdfplumber
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        pages = [p.extract_text() or "" for p in pdf.pages]
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in "\n".join(pages).splitlines()]
    return "\n".join(ln for ln in lines if len(ln.split()) >= 3 or ln.endswith(":"))


def to_text(data: bytes) -> str:
    return pdf_to_text(data) if data[:5] == b"%PDF-" else html_to_text(data)


def mentions_disease(text: str, keywords: str) -> bool:
    if not keywords:
        return True
    low = text.lower()
    return any(k.strip().lower() in low for k in keywords.split("|") if k.strip())


def main(refresh: bool = False):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache = {}
    for src in load_sources():
        out = RAW_DIR / f"{src['doc_id']}.txt"
        if out.exists() and not refresh:
            print(f"cached  {src['doc_id']}")
            continue
        try:
            if src["url"] not in cache:
                cache[src["url"]] = to_text(fetch(src["url"]))
                time.sleep(2)
            text = cache[src["url"]]
            if len(text.split()) < MIN_WORDS:
                raise RuntimeError(f"only {len(text.split())} words extracted")
            if not mentions_disease(text, src.get("keywords", "")):
                raise RuntimeError(f"page never mentions '{src['keywords']}' (wrong or missing page?)")
        except Exception as e:
            print(f"FAILED  {src['doc_id']}: {e}")
            continue
        header = f"# {src['title']}\n# {src['url']}\n# retrieved {date.today().isoformat()}\n\n"
        out.write_text(header + text, encoding="utf-8")
        print(f"ok      {src['doc_id']}: {len(text.split())} words")


if __name__ == "__main__":
    import sys
    main(refresh="--refresh" in sys.argv)
