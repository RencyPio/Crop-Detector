import re

from src.dl.labels import pretty_label

TELEGRAM_LIMIT = 4000
SMS_LIMIT = 480

HEADLINES = {
    "advise": "Crop alert",
    "healthy": "Leaf looks healthy",
    "uncertain_retake_photo": "Please retake the photo",
    "ambiguous_pair": "Two possible diseases",
}


def _plain(md: str) -> str:
    md = re.sub(r"\*\*(.+?)\*\*", r"\1", md)
    md = re.sub(r"(?<!\w)\*(.+?)\*(?!\w)", r"\1", md)
    return re.sub(r"\s*\[\d+\](\[\d+\])*", "", md)


def format_full(result: dict) -> str:
    parts = [f"🌱 {HEADLINES.get(result['decision'], 'Crop result')}", result["message"]]
    if result.get("advice"):
        parts.append(_plain(result["advice"]))
    if result.get("sources"):
        parts.append("Sources:\n" + "\n".join(f"- {s['publisher']}: {s['url']}" for s in result["sources"]))
    text = "\n\n".join(parts)
    return text if len(text) <= TELEGRAM_LIMIT else text[: TELEGRAM_LIMIT - 20] + "\n...(see app)"


def format_short(result: dict) -> str:
    top = result["predictions"][0]
    line = f"{HEADLINES.get(result['decision'], 'Crop result')}: "
    if result["decision"] == "advise":
        line += f"{pretty_label(top['label'])} ({top['prob']:.0%})."
        w = result.get("weather") or {}
        if w.get("modified_advice"):
            line += f" Rain likely ({w['rain_prob_next_6h']:.0%}) - DELAY spraying."
        elif w.get("available") is False:
            line += " Check forecast before spraying."
        line += " Full advice in the app. Follow the label and ask your agriculture officer."
    else:
        line += _plain(result["message"])
    return line if len(line) <= SMS_LIMIT else line[: SMS_LIMIT - 3] + "..."
