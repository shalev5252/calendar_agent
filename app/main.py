# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, List
import io
from contextlib import redirect_stdout

# ייבוא הקובץ agent.py שנמצא בתיקייה הראשית
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # מאפשר גישה לקובץ agent.py
import agent
from tools import get_calendar_service, get_auth_url, exchange_code_for_token  # ← חשוב


app = FastAPI(title="Google Calendar Agent API", version="1.0")

# הרשה קריאות מהאפליקציה (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# מודלים (Schemas)
# ----------------------------------------------------
class ParseRequest(BaseModel):
    prompt: str

class ParseResponse(BaseModel):
    ok: bool
    actions: List[Dict[str, Any]]

class ExecuteRequest(BaseModel):
    actions: List[Dict[str, Any]]

class ExecuteResponse(BaseModel):
    ok: bool
    executed: int
    logs: str | None = None


# ----------------------------------------------------
# Endpoints
# ----------------------------------------------------
@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/parse", response_model=ParseResponse)
def parse_prompt(req: ParseRequest):
    """
    שלב 1 – פירוק הפרומפט לרשימת פעולות בלבד (ללא ביצוע)
    """
    try:
        actions = agent.plan_actions(req.prompt)
        return ParseResponse(ok=True, actions=actions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/execute", response_model=ExecuteResponse)
def execute_actions(req: ExecuteRequest):
    import io
    from contextlib import redirect_stdout

    # פונקציה פנימית שמוציאה את ה-payload (אם קיים) לרמה העליונה
    def _unwrap_payload(a: dict) -> dict:
        """מאחד payload לרמה העליונה אם קיים."""
        if "payload" in a and isinstance(a["payload"], dict):
            merged = {"command": a.get("command")}
            merged.update(a["payload"])
            return merged
        return a

    # ננקה את כל האובייקטים כדי שיתאימו למה ש-agent מצפה
    normalized_actions = [_unwrap_payload(a) for a in req.actions]

    buf = io.StringIO()
    try:
        service = get_calendar_service()
        with redirect_stdout(buf):
            agent.execute_actions(normalized_actions, service=service)
        return ExecuteResponse(ok=True, executed=len(normalized_actions), logs=buf.getvalue())
    except Exception as e:
        return ExecuteResponse(ok=False, executed=0, logs=f"Error: {e}\n{buf.getvalue()}")

# ---- הוספה ל-Schemas (ליד שאר ה-Pydantic) ----
from typing import Optional

class EventsQuery(BaseModel):
    from_datetime: str  # "YYYY-MM-DDTHH:MM:SS"
    to_datetime: str    # "YYYY-MM-DDTHH:MM:SS"
    time_zone: str = "Asia/Jerusalem"
    page_size: int = 50

class EventItem(BaseModel):
    id: str
    summary: Optional[str] = None
    start: Dict[str, Any] | None = None
    end: Dict[str, Any] | None = None
    recurringEventId: Optional[str] = None

class EventsResponse(BaseModel):
    ok: bool
    events: List[EventItem]

# ---- הוסף את ה-endpoint עצמו ----
@app.post("/events", response_model=EventsResponse)
def list_events(req: EventsQuery):
    """
    מחזיר אירועים גולמיים מהיומן בטווח תאריכים נתון.
    השרת ממיר את ה-local datetime ל-RFC3339 עם offset נכון (כולל DST).
    """
    try:
        service = get_calendar_service()
        time_min = agent._to_rfc3339_with_tz(req.from_datetime, req.time_zone)
        time_max = agent._to_rfc3339_with_tz(req.to_datetime, req.time_zone)

        result = service.events().list(
            calendarId='primary',
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy='startTime',
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

    except Exception as e:
        # אפשר להחליף ל-HTTPException(500) אם תרצה לכפות קוד שגיאה
        return EventsResponse(ok=False, events=[])


# --- OAuth start: מחזיר קישור התחברות ---
@app.get("/oauth2/start")
def oauth2_start():
    try:
        url = get_auth_url()
        return {"ok": True, "auth_url": url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- OAuth callback: גוגל מחזירה ?code= ---
@app.get("/oauth2callback")
def oauth2_callback(code: str | None = None):
    if not code:
        return HTMLResponse("<h3>Missing ?code</h3>", status_code=400)
    try:
        exchange_code_for_token(code)
        return HTMLResponse("<h3>Authorization completed. You can close this tab.</h3>", status_code=200)
    except Exception as e:
        return HTMLResponse(f"<h3>OAuth error: {e}</h3>", status_code=500)
