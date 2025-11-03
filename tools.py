# tools.py
import os
import json
import tempfile
from typing import Optional

from google_auth_oauthlib.flow import Flow, InstalledAppFlow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

# אם תרצה רענון אוטומטי לטוקן:
from google.auth.transport.requests import Request

SCOPES = ["https://www.googleapis.com/auth/calendar"]

# היכן לשמור/לקרוא את הטוקן בשרת
TOKEN_DIR = os.getenv("TOKEN_DIR", "/tmp")  # ב-Render מומלץ /data; ללוקאל /tmp זה אחלה
TOKEN_PATH = os.path.join(TOKEN_DIR, "token.json")

# שליטה בפולבק ללוקאל (לא חובה בשרת)
LOCAL_DEV = os.getenv("LOCAL_DEV", "0") == "1"


# -----------------------------
# עזר: יצירת קובץ סודות זמני מה-ENV (ל-Web OAuth)
# -----------------------------
def _client_secrets_file_from_env() -> str:
    secrets_json = os.getenv("GOOGLE_CLIENT_SECRETS_JSON")
    if not secrets_json:
        raise ValueError("Missing GOOGLE_CLIENT_SECRETS_JSON")

    # ודא שזה JSON תקין ושזה מסוג 'web'
    data = json.loads(secrets_json)
    if "web" not in data:
        raise ValueError("GOOGLE_CLIENT_SECRETS_JSON must contain a 'web' client (not 'installed').")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
        f.write(secrets_json.encode("utf-8"))
        return f.name


# -----------------------------
# שלב 1: יצירת קישור OAuth (לשרת)
# -----------------------------
def get_auth_url() -> str:
    redirect_uri = os.getenv("GOOGLE_REDIRECT_URI")
    if not redirect_uri:
        raise ValueError("Missing GOOGLE_REDIRECT_URI")

    client_file = _client_secrets_file_from_env()
    flow = Flow.from_client_secrets_file(client_file, scopes=SCOPES, redirect_uri=redirect_uri)
    auth_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    return auth_url


# -----------------------------
# שלב 2: המרת code לטוקן ושמירה ל-TOKEN_PATH (לשרת)
# -----------------------------
def exchange_code_for_token(code: str) -> str:
    os.makedirs(TOKEN_DIR, exist_ok=True)

    redirect_uri = os.getenv("GOOGLE_REDIRECT_URI")
    if not redirect_uri:
        raise ValueError("Missing GOOGLE_REDIRECT_URI")

    client_file = _client_secrets_file_from_env()
    flow = Flow.from_client_secrets_file(client_file, scopes=SCOPES, redirect_uri=redirect_uri)
    flow.fetch_token(code=code)

    creds = flow.credentials
    with open(TOKEN_PATH, "w", encoding="utf-8") as f:
        f.write(creds.to_json())
    return TOKEN_PATH


# -----------------------------
# טעינת אישורים קיימים (שרת)
# -----------------------------
def _load_creds_from_token_file() -> Optional[Credentials]:
    if os.path.exists(TOKEN_PATH):
        with open(TOKEN_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Credentials.from_authorized_user_info(data, SCOPES)
    return None


# -----------------------------
# שירות גוגל קלנדר – מאוחד לשרת/לוקאל
# -----------------------------
def get_calendar_service():
    """
    בשרת (Render): קורא token.json מ-TOKEN_DIR (נוצר ע"י /oauth2callback).
    בלוקאל (רק אם LOCAL_DEV=1): מבצע InstalledAppFlow מקובץ credentials.json ושומר token.json ל-TOKEN_DIR.
    """
    creds = _load_creds_from_token_file()

    if not creds:
        if LOCAL_DEV:
            # פולבק לפיתוח מקומי (רק אם הגדרת LOCAL_DEV=1)
            if not os.path.exists("credentials.json"):
                raise RuntimeError(
                    "credentials.json not found for local dev flow. "
                    "Either place it locally or use the web OAuth via /oauth2/start."
                )
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            # פתח דפדפן בלוקאל:
            creds = flow.run_local_server(port=0)
            os.makedirs(TOKEN_DIR, exist_ok=True)
            with open(TOKEN_PATH, "w", encoding="utf-8") as f:
                f.write(creds.to_json())
        else:
            # בשרת – אין טוקן => צריך קודם להשלים OAuth ב-/oauth2/start
            raise RuntimeError("No token found. Complete OAuth: GET /oauth2/start and finish the login.")

    # רענון אוטומטי אם צריך
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        # עדכון הקובץ לאחר רענון
        os.makedirs(TOKEN_DIR, exist_ok=True)
        with open(TOKEN_PATH, "w", encoding="utf-8") as f:
            f.write(creds.to_json())

    return build("calendar", "v3", credentials=creds)
