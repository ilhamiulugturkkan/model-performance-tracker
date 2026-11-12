import json
import os
import urllib.error
import urllib.request

TELEGRAM_URL = "https://api.telegram.org/bot{token}/sendMessage"
TELEGRAM_MAX_CHARS = 4000  # Telegram refuses messages longer than 4096


def send_telegram(
    text, token=None, chat_id=None, opener=urllib.request.urlopen, timeout=10
):
    # Needs TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in the environment (or pass them in).

    token = token or os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print(f"[alert NOT sent, Telegram is not configured] {text}")
        return False
    request = urllib.request.Request(
        TELEGRAM_URL.format(token=token),
        data=json.dumps({"chat_id": chat_id, "text": text[:TELEGRAM_MAX_CHARS]}).encode(
            "utf-8"
        ),
        headers={"Content-Type": "application/json"},
    )
    try:
        with opener(request, timeout=timeout) as response:
            return response.status == 200
    except (OSError, ValueError) as error:  # URLError is an OSError
        # Print the kind of error only: the message can contain the URL, and the URL contains the token.
        print(f"[alert NOT sent, {type(error).__name__}] {text}")
        return False
