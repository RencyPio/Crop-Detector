from src.automation.formatter import format_full, format_short
from src.automation.senders import send_sms, send_telegram

SENDERS = {
    "telegram": lambda r: send_telegram(format_full(r)),
    "sms": lambda r: send_sms(format_short(r)),
}


def notify(result: dict, channels: list[str] | None, senders: dict | None = None) -> list[dict]:
    senders = senders or SENDERS
    out = []
    for ch in channels or []:
        if ch not in senders:
            out.append({"channel": ch, "status": "failed", "error": "unknown channel"})
            continue
        try:
            out.append(senders[ch](result))
        except Exception as e:
            out.append({"channel": ch, "status": "failed", "error": str(e)[:200]})
    return out
