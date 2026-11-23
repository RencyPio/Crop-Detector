import os
import re
import time

from dotenv import load_dotenv

from src.dl.labels import pretty_label

load_dotenv()

DEFAULT_MODEL = "openai/gpt-oss-120b"

DISCLAIMER = ("Always follow the pesticide label and consult your local agriculture officer "
              "before applying any chemical.")

SYSTEM_PROMPT = """You are an agricultural advisory assistant for small farmers.
Rules you must follow:
1. Use ONLY the numbered source passages provided. Do not use outside knowledge.
2. If the passages do not answer something, write "Not found in the knowledge base." for that part.
3. Never state a chemical name, dose or interval that is not written in a passage.
4. End EVERY bullet and sentence in Summary, What to do now and Chemical options with its passage
   number in square brackets, e.g. [2]. Do not add advice that no passage gives.
5. Prefer passages marked region=India for product and dose advice. If you use a dose from a
   non-Indian source, say which country it is from.
6. Write in short, simple sentences a farmer can follow.

Answer in exactly this format:
**Summary:** 1-2 sentences on what the disease is and how serious it is.
**What to do now (cultural / non-chemical):** bullet points.
**Chemical options:** bullet points with product, dose and timing exactly as in the passages.
**When to get expert help:** bullet points."""


def format_context(hits: list[dict]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        m = h["metadata"]
        blocks.append(f"[{i}] (publisher={m['publisher']}; region={m['region']}; section={m['section']})\n{h['text']}")
    return "\n\n".join(blocks)


def sources_from_hits(hits: list[dict]) -> list[dict]:
    seen, out = set(), []
    for i, h in enumerate(hits, 1):
        m = h["metadata"]
        if m["source_url"] not in seen:
            seen.add(m["source_url"])
            out.append({"n": i, "title": m["source_title"], "url": m["source_url"],
                        "publisher": m["publisher"], "region": m["region"]})
    return out


MAX_ATTEMPTS = 2

NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
CITATION = re.compile(r"\[\d+\]")
ODD_CITATION = re.compile(r"【(\d+)[^】]*】")
NUMBER_WORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
                "eight": "8", "nine": "9", "ten": "10", "twice": "2", "thrice": "3", "fortnight": "14"}
NUMBER_WORD_RE = re.compile(r"\b(" + "|".join(NUMBER_WORDS) + r")\b", re.IGNORECASE)


def normalize_citations(text: str) -> str:
    return ODD_CITATION.sub(r"[\1]", text)


def _numbers(text: str) -> list[str]:
    text = CITATION.sub("", text).replace(",", "")
    text = NUMBER_WORD_RE.sub(lambda m: NUMBER_WORDS[m.group(1).lower()], text)
    return NUMBER.findall(text)


def check_grounding(answer: str, hits: list[dict]) -> dict:
    answer = normalize_citations(answer)
    context_numbers = set(_numbers(" ".join(h["text"] for h in hits)))
    unsupported = sorted({n for n in _numbers(answer) if n not in context_numbers})
    flagged = [ln.strip() for ln in answer.splitlines() if set(_numbers(ln)) & set(unsupported)]

    uncited, in_expert = [], False
    for line in answer.splitlines():
        if "expert help" in line.lower():
            in_expert = True
        elif line.lstrip().startswith(("- ", "* ", "• ")) and not in_expert and not CITATION.search(line):
            uncited.append(line.strip())
    return {"unsupported_numbers": unsupported, "flagged_lines": flagged, "uncited_bullets": uncited}


def _extractive(disease_class: str, hits: list[dict]) -> str:
    parts = [f"**What the agricultural guides say about {pretty_label(disease_class)}:**"]
    for i, h in enumerate(hits[:3], 1):
        parts.append(f"[{i}] {h['text'][:600]}...")
    return "\n\n".join(parts)


def generate_advice(disease_class: str, hits: list[dict], model: str | None = None) -> dict:
    if not hits:
        return {"advice": None, "sources": [], "mode": "no_knowledge"}

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return {"advice": _extractive(disease_class, hits) + f"\n\n{DISCLAIMER}",
                "sources": sources_from_hits(hits), "mode": "extractive"}

    from groq import Groq
    client = Groq(api_key=api_key)
    user_msg = (f"Disease diagnosed by the image model: {pretty_label(disease_class)}\n\n"
                f"Source passages:\n{format_context(hits)}\n\n"
                f"Give treatment advice for this disease following the rules and format.")
    model = model or os.getenv("GROQ_MODEL", DEFAULT_MODEL)
    kwargs = dict(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_msg}],
        temperature=0.1,
        max_completion_tokens=2000,
    )
    if "gpt-oss" in model:
        kwargs["reasoning_effort"] = "low"
    def call_llm():
        for wait in (3, 10, None):
            try:
                return client.chat.completions.create(**kwargs)
            except Exception as e:
                daily = "tokens per day" in str(e).lower()
                if wait is None or daily or "429" not in str(e) and "rate limit" not in str(e).lower():
                    raise
                print(f"  Groq rate limit, retrying in {wait}s")
                time.sleep(wait)

    rejected = []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = call_llm()
            text = normalize_citations((resp.choices[0].message.content or "").strip())
            if not text:
                raise RuntimeError("empty answer from the LLM")
        except Exception as e:
            print(f"  LLM call failed ({str(e)[:160]}); using extractive fallback")
            return {"advice": _extractive(disease_class, hits) + f"\n\n{DISCLAIMER}",
                    "sources": sources_from_hits(hits), "mode": "extractive_fallback", "llm_error": str(e)[:300]}
        grounding = check_grounding(text, hits)
        if not grounding["unsupported_numbers"]:
            return {"advice": text + f"\n\n{DISCLAIMER}", "sources": sources_from_hits(hits), "mode": "llm",
                    "model": model, "attempts": attempt, "grounding": grounding, "rejected": rejected}
        print(f"  attempt {attempt}: numbers not in any passage {grounding['unsupported_numbers']}")
        rejected.append(grounding)

    return {"advice": _extractive(disease_class, hits) + f"\n\n{DISCLAIMER}", "sources": sources_from_hits(hits),
            "mode": "extractive_fallback", "attempts": MAX_ATTEMPTS, "grounding": rejected[-1], "rejected": rejected}
