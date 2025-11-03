from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import os
import pickle
from dotenv import load_dotenv

# tools.py
from openai import OpenAI

# Initialize OpenAI client
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SCOPES = ['https://www.googleapis.com/auth/calendar']

def get_calendar_service():
    creds = None
    # Cache credentials to avoid re-auth every run
    if os.path.exists("token.pkl"):
        with open("token.pkl", "rb") as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
        try:
            creds = flow.run_local_server(port=0)  # dynamic port
        except OSError:
            print("Local server failed — trying fallback (headless browser)...")
            creds = flow.run_local_server(port=0, open_browser=False)

        with open("token.pkl", "wb") as token:
            pickle.dump(creds, token)

    return build("calendar", "v3", credentials=creds)

import json
import tempfile
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials

# ----------------------------------------------------
# OAuth URLs
# ----------------------------------------------------

def get_auth_url() -> str:
    """
    יוצר קישור OAuth של גוגל ומחזיר אותו למשתמש.
    """
    secrets_json = os.getenv("GOOGLE_CLIENT_SECRETS_JSON")
    if not secrets_json:
        raise ValueError("Missing GOOGLE_CLIENT_SECRETS_JSON")

    redirect_uri = os.getenv("GOOGLE_REDIRECT_URI")
    if not redirect_uri:
        raise ValueError("Missing GOOGLE_REDIRECT_URI")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
        f.write(secrets_json.encode())
        client_secrets_file = f.name

    flow = Flow.from_client_secrets_file(
        client_secrets_file,
        scopes=["https://www.googleapis.com/auth/calendar"],
        redirect_uri=redirect_uri,
    )

    auth_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent"
    )

    return auth_url


def exchange_code_for_token(code: str):
    """
    מקבל את הקוד שגוגל מחזירה וממיר אותו ל-token.json (נשמר בדיסק /data או בתיקייה מקומית)
    """
    secrets_json = os.getenv("GOOGLE_CLIENT_SECRETS_JSON")
    redirect_uri = os.getenv("GOOGLE_REDIRECT_URI")
    token_dir = os.getenv("TOKEN_DIR", "./data")

    os.makedirs(token_dir, exist_ok=True)

    if not secrets_json or not redirect_uri:
        raise ValueError("Missing OAuth environment variables")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
        f.write(secrets_json.encode())
        client_secrets_file = f.name

    flow = Flow.from_client_secrets_file(
        client_secrets_file,
        scopes=["https://www.googleapis.com/auth/calendar"],
        redirect_uri=redirect_uri,
    )

    flow.fetch_token(code=code)
    creds = flow.credentials

    token_path = os.path.join(token_dir, "token.json")
    with open(token_path, "w") as token_file:
        token_file.write(creds.to_json())

    return token_path


