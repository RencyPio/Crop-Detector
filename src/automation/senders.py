import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()
TIMEOUT = 15
RETRIES = 3


def _with_retry(send, channel: str) -> dict:
    last_error = None
    for attempt in range(1, RETRIES + 1):
        try:
            resp = send()
            if resp.status_code < 300:
                return {"channel": channel, "status": "sent", "attempts": attempt}
            last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
            if resp.status_code not in (429, 500, 502, 503, 504):
                break
        except requests.RequestException as e:
            last_error = str(e)[:200]
        if attempt < RETRIES:
            time.sleep(attempt)
    return {"channel": channel, "status": "failed", "attempts": attempt, "error": last_error}


def send_telegram(text: str, chat_id: str | None = None) -> dict:
    token, chat_id = os.getenv("TELEGRAM_BOT_TOKEN"), chat_id or os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return {"channel": "telegram", "status": "not_configured"}
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    return _with_retry(lambda: requests.post(url, data={"chat_id": chat_id, "text": text}, timeout=TIMEOUT),
                       "telegram")


def _twilio(to: str, from_: str, text: str, channel: str) -> dict:
    sid, auth = os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN")
    if not (sid and auth and from_ and to):
        return {"channel": channel, "status": "not_configured"}
    url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
    return _with_retry(lambda: requests.post(url, data={"To": to, "From": from_, "Body": text},
                                             auth=(sid, auth), timeout=TIMEOUT), channel)


def send_sms(text: str, to: str | None = None) -> dict:
    return _twilio(to or os.getenv("ALERT_SMS_TO"), os.getenv("TWILIO_FROM_NUMBER"), text, "sms")
