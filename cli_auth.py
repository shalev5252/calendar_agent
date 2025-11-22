# cli_auth.py
from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Optional

from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

SCOPES = ["https://www.googleapis.com/auth/calendar"]

# תיק נפרד לטוקן מקומי, שלא יתנגש עם השרת
TOKENS_DIR = Path(".tokens")
LOCAL_TOKEN_PATH = TOKENS_DIR / "token_local.json"

# קובץ קרדנצ׳אלס מקומי ל־CLI (Client type: “Desktop” ב-Google Cloud)
# אם אין – ננסה ליפול־לאחור ל-credentials.json הרגיל (בתנאי שיש בו “installed”)
LOCAL_CREDENTIALS_CANDIDATES = [
    Path("credentials.local.json"),
    Path("credentials.json"),
]

def _find_local_credentials_file() -> Path:
    for p in LOCAL_CREDENTIALS_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Missing credentials.local.json / credentials.json for local CLI.\n"
        "Create an OAuth Client of type ‘Desktop App’ and download as credentials.local.json next to this file."
    )

def ensure_local_token() -> Credentials:
    TOKENS_DIR.mkdir(parents=True, exist_ok=True)
    creds: Optional[Credentials] = None

    if LOCAL_TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(LOCAL_TOKEN_PATH), SCOPES)

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        # ריענון שקט אם אפשר
        creds.refresh(Request())
        _save(creds)
        return creds

    # אין/לא תקין → הרצת InstalledAppFlow מקומית
    creds_file = _find_local_credentials_file()
    # חשוב: לקוח מסוג “Desktop” תומך ב־localhost אוטומטית
    flow = InstalledAppFlow.from_client_secrets_file(str(creds_file), SCOPES)
    try:
        creds = flow.run_local_server(port=0)  # פותח דפדפן
    except OSError:
        # fallback בלי פתיחת דפדפן (להדביק קוד ידנית במסוף)
        creds = flow.run_console()

    _save(creds)
    return creds

def _save(creds: Credentials) -> None:
    LOCAL_TOKEN_PATH.write_text(creds.to_json(), encoding="utf-8")

def get_calendar_service_local():
    """Calendar service מקומי לבדיקות CLI (לא משפיע על השרת)."""
    creds = ensure_local_token()
    return build("calendar", "v3", credentials=creds)
