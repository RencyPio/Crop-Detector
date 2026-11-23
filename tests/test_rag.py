import csv

import pytest

from src.dl.labels import load_class_names
from src.rag.chunker import chunk_text, is_menu
from src.rag.config import CHROMA_DIR, SOURCES_CSV
from src.rag.generate import DISCLAIMER, generate_advice, sources_from_hits


def test_every_source_maps_to_a_real_class():
    classes = set(load_class_names())
    with open(SOURCES_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows
    for r in rows:
        assert r["disease_class"] in classes, r["doc_id"]
        assert r["region"] in {"India", "USA", "International"}
        assert r["url"].startswith("http")
    assert len({r["doc_id"] for r in rows}) == len(rows), "doc_id must be unique"


def test_chunks_respect_size_and_overlap():
    text = "\n".join(f"Sentence number {i} about leaf blight management." for i in range(200))
    chunks = chunk_text(text, size=100, overlap=20)
    assert len(chunks) > 5
    for _, c in chunks:
        assert len(c.split()) <= 150
    first, second = chunks[0][1].split(), chunks[1][1].split()
    assert first[-20:] == second[:20]


def test_menu_filter_keeps_pesticide_rows():
    assert is_menu("Home | About Us | Success Stories | Farmers Association | Contact")
    assert not is_menu("(Quadris F) | 6 fl oz | 4 | 0")


def test_no_hits_means_no_advice():
    assert generate_advice("Apple___Apple_scab", [])["mode"] == "no_knowledge"


def test_offline_mode_still_cites_and_warns(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    hits = [{"text": "Spray mancozeb 2 g/l.", "score": 0.9,
             "metadata": {"source_url": "u1", "source_title": "T", "publisher": "TNAU", "region": "India",
                          "section": "management"}}]
    out = generate_advice("Tomato___Late_blight", hits)
    assert out["mode"] == "extractive"
    assert DISCLAIMER in out["advice"]
    assert out["sources"] == sources_from_hits(hits) and out["sources"][0]["publisher"] == "TNAU"


def test_grounding_check_catches_invented_dose_and_missing_citation():
    from src.rag.generate import check_grounding
    hits = [{"text": "Spray mancozeb 2 g/l at 35 and 50 DAS.", "metadata": {}}]
    ok = check_grounding("- Spray mancozeb 2 g/l at 35 and 50 DAS. [1]", hits)
    assert ok == {"unsupported_numbers": [], "flagged_lines": [], "uncited_bullets": []}
    bad = check_grounding("- Spray mancozeb 3 g/l. [1]\n- Improve drainage.\n**When to get expert help:**\n- If unsure.", hits)
    assert bad["unsupported_numbers"] == ["3", "3g"]
    assert bad["flagged_lines"] == ["- Spray mancozeb 3 g/l. [1]"]
    assert bad["uncited_bullets"] == ["- Improve drainage."]


def test_grounding_reads_number_words_and_odd_citations():
    from src.rag.generate import check_grounding
    hits = [{"text": "Remove affected plant parts. Spray 200-250 ml/ha.", "metadata": {}}]
    assert check_grounding("- Rotate crops for two years. [1]", hits)["unsupported_numbers"] == ["2"]
    ok = check_grounding("- Spray 200-250 ml/ha【1†L5-L7】.", hits)
    assert ok["unsupported_numbers"] == [] and ok["uncited_bullets"] == []


def test_number_must_match_its_unit():
    from src.rag.generate import check_grounding
    hits = [{"text": "Infection at 10 degrees, RH 90%. Bordeaux mixture 1. Per cent.", "metadata": {}}]
    assert check_grounding("- Get help when loss exceeds 10 %. [1]", hits)["unsupported_numbers"] == ["10%"]
    assert check_grounding("- Spray Bordeaux mixture at 1 %. [1]", hits)["unsupported_numbers"] == []


def test_invented_cultural_tip_is_removed_but_sourced_one_stays():
    from src.rag.generate import remove_unsupported
    hits = [{"text": "Remove and destroy affected plant parts. Spray wettable sulphur 2 g/l.", "metadata": {}}]
    answer = ("**What to do now (cultural / non-chemical):**\n- Remove and destroy affected plant parts. [1]\n"
              "- Keep the crop well-watered and avoid excessive nitrogen. [1]\n\n"
              "**Chemical options:**\n- Spray wettable sulphur 2 g/l. [1]")
    text, removed = remove_unsupported(answer, hits)
    assert "affected plant parts" in text and "sulphur" in text and "nitrogen" not in text
    assert len(removed) == 1


def test_section_left_empty_says_not_found():
    from src.rag.generate import remove_unsupported
    hits = [{"text": "Spray wettable sulphur 2 g/l.", "metadata": {}}]
    answer = ("**What to do now (cultural / non-chemical):**\n- Monitor plants regularly for webbing. [1]\n\n"
              "**Chemical options:**\n- Spray wettable sulphur 2 g/l. [1]")
    text, removed = remove_unsupported(answer, hits)
    assert "Not found in the knowledge base." in text and len(removed) == 1


def test_retry_recovers_then_fallback(monkeypatch):
    import types
    import src.rag.generate as G
    hits = [{"text": "Spray mancozeb 2 g/l.", "score": 0.9,
             "metadata": {"source_url": "u", "source_title": "T", "publisher": "TNAU", "region": "India", "section": "m"}}]

    def fake_groq(answers):
        it = iter(answers)
        msg = lambda: types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=next(it)))])
        return lambda api_key: types.SimpleNamespace(chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=lambda **kw: msg())))

    monkeypatch.setenv("GROQ_API_KEY", "test")
    import groq
    monkeypatch.setattr(groq, "Groq", fake_groq(["- Spray mancozeb 5 g/l. [1]", "- Spray mancozeb 2 g/l. [1]"]))
    out = G.generate_advice("Tomato___Late_blight", hits)
    assert out["mode"] == "llm" and out["attempts"] == 2 and out["rejected"][0]["unsupported_numbers"] == ["5", "5g"]

    monkeypatch.setattr(groq, "Groq", fake_groq(["- 5 g/l [1]", "- 7 g/l [1]"]))
    out = G.generate_advice("Tomato___Late_blight", hits)
    assert out["mode"] == "extractive_fallback" and len(out["rejected"]) == 2


index_built =pytest.mark.skipif(not CHROMA_DIR.exists(), reason="run python -m src.rag.vector_store first")


@index_built
@pytest.mark.parametrize("label", ["Tomato___Late_blight", "Potato___Early_blight", "Corn_(maize)___Common_rust_"])
def test_retrieval_only_returns_the_asked_disease(label):
    from src.rag.retrieve import retrieve
    hits = retrieve(label)
    assert hits
    assert all(h["metadata"]["disease_class"] == label for h in hits)
    assert hits[0]["metadata"]["region"] == "India"


@index_built
def test_unsupported_disease_has_no_knowledge():
    from src.rag.retrieve import has_knowledge
    assert has_knowledge("Tomato___Late_blight")
    assert has_knowledge("Apple___Apple_scab")
    assert not has_knowledge("Tomato___healthy")
