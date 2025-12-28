# cli_auth.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# Full read/write access to Google Calendar for local CLI testing
SCOPES = ["https://www.googleapis.com/auth/calendar"]

# Store local CLI tokens separately so they don't collide with server tokens
TOKENS_DIR = Path(".tokens")
LOCAL_TOKEN_PATH = TOKENS_DIR / "token_local.json"

# Local OAuth client credentials for CLI (OAuth Client type: "Desktop App")
# Prefer credentials.local.json; fallback to credentials.json if it contains "installed"
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
        "Create an OAuth Client of type 'Desktop App' in Google Cloud Console and download it as "
        "credentials.local.json next to this file."
    )


def _save(creds: Credentials) -> None:
    TOKENS_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_TOKEN_PATH.write_text(creds.to_json(), encoding="utf-8")


def ensure_local_token() -> Credentials:
    """
    Ensures a valid local CLI OAuth token exists (stored under .tokens/token_local.json).
    Refreshes if expired and refresh_token exists; otherwise runs a local OAuth flow.
    """
    TOKENS_DIR.mkdir(parents=True, exist_ok=True)

    creds: Optional[Credentials] = None
    if LOCAL_TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(LOCAL_TOKEN_PATH), SCOPES)

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        _save(creds)
        return creds

    # No token / invalid token -> run InstalledAppFlow locally
    creds_file = _find_local_credentials_file()
    flow = InstalledAppFlow.from_client_secrets_file(str(creds_file), SCOPES)

    try:
        # Opens a browser and uses localhost redirect
        creds = flow.run_local_server(port=0)
    except OSError:
        # Fallback: manual copy/paste authorization code in terminal
        creds = flow.run_console()

    _save(creds)
    return creds


def get_calendar_service_local():
    """
    Returns a Calendar API service for local CLI testing (does not affect server auth).
    """
    creds = ensure_local_token()
    return build("calendar", "v3", credentials=creds)
