# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import io
from contextlib import redirect_stdout
import sys, os
import traceback

# Allow importing agent.py from project root (one level above /app)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import agent

from tools import (
    get_calendar_service,
    get_auth_url,
    exchange_code_for_token,
)

app = FastAPI(title="Google Calendar Agent API", version="1.0")

# CORS (adjust allow_origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# Schemas
# ----------------------------------------------------
class ParseRequest(BaseModel):
    prompt: str

class ParseResponse(BaseModel):
    ok: bool
    actions: List[Dict[str, Any]]

class ExecuteRequest(BaseModel):
    actions: List[Dict[str, Any]]
    original_prompt: str  # IMPORTANT: used for deterministic weekday fixing on the server

class ExecuteResponse(BaseModel):
    ok: bool
    executed: int
    logs: Optional[str] = None

class EventsQuery(BaseModel):
    from_datetime: str  # "YYYY-MM-DDTHH:MM:SS"
    to_datetime: str    # "YYYY-MM-DDTHH:MM:SS"
    time_zone: str = "Asia/Jerusalem"
    page_size: int = 50

class EventItem(BaseModel):
    id: str
    summary: Optional[str] = None
    start: Optional[Dict[str, Any]] = None
    end: Optional[Dict[str, Any]] = None
    recurringEventId: Optional[str] = None

class EventsResponse(BaseModel):
    ok: bool
    events: List[EventItem]

# ----------------------------------------------------
# Endpoints
# ----------------------------------------------------
@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/parse", response_model=ParseResponse)
def parse_prompt(req: ParseRequest):
    """
    Step 1: Parse the user prompt into a list of actions (NO execution).
    """
    try:
        actions = agent.plan_actions(req.prompt)
        return ParseResponse(ok=True, actions=actions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/execute", response_model=ExecuteResponse)
def execute_actions(req: ExecuteRequest):
    """
    Step 2: Execute actions. We ALSO receive the original_prompt so the server can
    deterministically fix weekday-based requests (next upcoming weekday) before execution.
    """
    def _unwrap_payload(a: dict) -> dict:
        # Flutter may wrap a payload object. Merge it into the top-level.
        if "payload" in a and isinstance(a["payload"], dict):
            merged = {"command": a.get("command")}
            merged.update(a["payload"])
            return merged
        return a

    normalized_actions = [_unwrap_payload(a) for a in req.actions]

    buf = io.StringIO()
    try:
        service = get_calendar_service()

        with redirect_stdout(buf):
            agent.execute_actions(
                normalized_actions,
                service=service,
                original_prompt=req.original_prompt,  # IMPORTANT
            )

        return ExecuteResponse(ok=True, executed=len(normalized_actions), logs=buf.getvalue())

    except Exception as e:
        return ExecuteResponse(ok=False, executed=0, logs=f"Error: {e}\n{buf.getvalue()}")


@app.post("/events", response_model=EventsResponse)
def list_events(req: EventsQuery):
    """
    Returns raw calendar events in a given local datetime range.
    The server converts local datetime to RFC3339 with correct offset (DST-aware).
    """
    try:
        service = get_calendar_service()
        time_min = agent._to_rfc3339_with_tz(req.from_datetime, req.time_zone)
        time_max = agent._to_rfc3339_with_tz(req.to_datetime, req.time_zone)

        result = service.events().list(
            calendarId="primary",
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy="startTime",
            maxResults=req.page_size,
        ).execute()

        items = result.get("items", [])
        events: List[Dict[str, Any]] = []
        for it in items:
            events.append({
                "id": it.get("id"),
                "summary": it.get("summary"),
                "start": it.get("start"),
                "end": it.get("end"),
                "recurringEventId": it.get("recurringEventId"),
            })

        return EventsResponse(ok=True, events=events)

    except Exception:
        return EventsResponse(ok=False, events=[])


# --- OAuth start: returns auth URL ---
@app.get("/oauth2/start")
def oauth2_start():
    try:
        url = get_auth_url()
        return {"ok": True, "auth_url": url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- OAuth callback: Google returns ?code= ---
@app.get("/oauth2callback")
def oauth2_callback(code: str | None = None):
    if not code:
        return HTMLResponse("<h3>Missing ?code</h3>", status_code=400)

    try:
        exchange_code_for_token(code)

        html = """
        <html>
        <body>
            <h3>Login completed. Returning to the app…</h3>
            <script>
                window.location.href = "myapp://oauth-complete";
            </script>
        </body>
        </html>
        """
        return HTMLResponse(html, status_code=200)

    except Exception as e:
        tb = traceback.format_exc()
        print("OAUTH ERROR:", e, tb)
        return HTMLResponse(f"<h3>OAuth error: {e}</h3><pre>{tb}</pre>", status_code=500)


@app.get("/auth/status")
def auth_status():
    """
    Checks whether a valid Google Calendar auth exists.
    Returns:
      { "ok": true } if token is valid
      { "ok": false } otherwise
    """
    try:
        service = get_calendar_service()  # raises if not authorized
        service.calendarList().list(maxResults=1).execute()
        return {"ok": True}
    except Exception:
        return {"ok": False}
