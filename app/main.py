# app/main.py
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from typing import Any, Dict, List, Optional
import io
from contextlib import redirect_stdout
import traceback
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import agent
from tools import get_calendar_service, get_auth_url, exchange_code_for_token

app = FastAPI(title="Google Calendar Agent API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


def _unwrap_payload(a: dict) -> dict:
    if isinstance(a, dict) and "payload" in a and isinstance(a["payload"], dict):
        merged = {"command": a.get("command")}
        merged.update(a["payload"])
        return merged
    return a


@app.post("/parse")
def parse_prompt(payload: Dict[str, Any] = Body(...)):
    try:
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise HTTPException(status_code=400, detail="Missing 'prompt' string")

        actions = agent.plan_actions(prompt)
        # NOTE: actions here are NAIVE local times (no offsets) -> UI displays correctly
        return {"ok": True, "actions": actions}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/execute")
def execute_actions(payload: Dict[str, Any] = Body(...)):
    buf = io.StringIO()
    try:
        actions_raw = payload.get("actions")
        if not isinstance(actions_raw, list):
            raise HTTPException(status_code=400, detail="Missing 'actions' array")

        normalized_actions = [_unwrap_payload(a) for a in actions_raw if isinstance(a, dict)]

        service = get_calendar_service()
        with redirect_stdout(buf):
            agent.execute_actions(normalized_actions, service=service)

        return {"ok": True, "executed": len(normalized_actions), "logs": buf.getvalue()}

    except HTTPException as e:
        return {"ok": False, "executed": 0, "logs": f"Error: {e.detail}\n{buf.getvalue()}"}
    except Exception as e:
        return {"ok": False, "executed": 0, "logs": f"Error: {e}\n{buf.getvalue()}"}


@app.get("/oauth2/start")
def oauth2_start():
    try:
        url = get_auth_url()
        return {"ok": True, "auth_url": url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/oauth2callback")
def oauth2_callback(code: Optional[str] = None):
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
    try:
        service = get_calendar_service()
        service.calendarList().list(maxResults=1).execute()
        return {"ok": True}
    except Exception:
        return {"ok": False}
