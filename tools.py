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


