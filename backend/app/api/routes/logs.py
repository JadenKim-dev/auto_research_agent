import json
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query

from ..schemas.logs import LogEntry

router = APIRouter(
    prefix="/api/logs",
    tags=["logs"],
    responses={404: {"description": "Not found"}},
)


@router.get("/sessions/{session_id}", response_model=List[LogEntry])
async def get_session_logs(
    session_id: str,
    execution_id: Optional[str] = Query(None, description="Filter by execution ID"),
):
    """Get all logs for a session, optionally filtered by execution ID."""
    session_dir = Path("logs") / "sessions" / session_id

    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    all_logs = []

    if execution_id:
        log_file_path = session_dir / f"execution_{execution_id}.json"
        if not log_file_path.exists():
            raise HTTPException(status_code=404, detail="Execution not found")

        with open(log_file_path) as log_file:
            logs = json.load(log_file)
            all_logs.extend(logs)
    else:
        for log_file_path in session_dir.glob("execution_*.json"):
            with open(log_file_path) as log_file:
                logs = json.load(log_file)
                all_logs.extend(logs)

    return [LogEntry(**log) for log in all_logs]


@router.get("/sessions", response_model=List[str])
async def list_sessions():
    """List all available session IDs."""
    logs_dir = Path("logs") / "sessions"

    if not logs_dir.exists():
        return []

    return [
        session_dir.name for session_dir in logs_dir.iterdir() if session_dir.is_dir()
    ]
