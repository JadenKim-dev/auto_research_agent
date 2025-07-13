from pydantic import BaseModel


class LogEntry(BaseModel):
    session_id: str
    execution_id: str
    timestamp: str
    step_number: int
    step_type: str
    content: dict
