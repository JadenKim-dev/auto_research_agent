import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, asdict
from langchain.schema import AgentAction, AgentFinish
from langchain.callbacks.base import BaseCallbackHandler


class StepType(str, Enum):
    ACTION = "action"
    OBSERVATION = "observation"
    FINAL_ANSWER = "final_answer"


@dataclass
class LogEntry:
    session_id: str
    execution_id: str
    timestamp: str
    step_number: int  # Sequential step number
    step_type: StepType  # Step type (action/observation/final_answer)
    content: Dict[str, Any]  # Detailed content data for each step


class TAOLogger(BaseCallbackHandler):
    def __init__(self, session_id: str, execution_id: Optional[str] = None):
        super().__init__()
        self.session_id = session_id
        self.execution_id = execution_id or str(uuid.uuid4())
        self.step_number = 0
        self.logs: List[LogEntry] = []
        self.thoughts: List[str] = []
        self.actions: List[Dict[str, Any]] = []
        self.observations: List[str] = []

    def _create_log_entry(
        self, step_type: StepType, content: Dict[str, Any]
    ) -> LogEntry:
        self.step_number += 1
        return LogEntry(
            session_id=self.session_id,
            execution_id=self.execution_id,
            timestamp=datetime.now().isoformat(),
            step_number=self.step_number,
            step_type=step_type,
            content=content,
        )

    def on_agent_action(self, action: AgentAction, **kwargs) -> None:
        self.thoughts.append(action.log)
        self.actions.append(
            {"tool": action.tool, "input": action.tool_input, "log": action.log}
        )
        log_entry = self._create_log_entry(
            StepType.ACTION,
            {
                "thought": action.log,
                "tool_name": action.tool,
                "tool_input": action.tool_input,
            },
        )
        self.logs.append(log_entry)

    def on_tool_end(self, output: str, **kwargs) -> None:
        self.observations.append(output)
        log_entry = self._create_log_entry(
            StepType.OBSERVATION, {"tool_output": output}
        )
        self.logs.append(log_entry)

    def on_agent_finish(self, finish: AgentFinish, **kwargs) -> None:
        self.thoughts.append(finish.log)
        log_entry = self._create_log_entry(
            StepType.FINAL_ANSWER, {"output": finish.return_values.get("output", "")}
        )
        self.logs.append(log_entry)
        self._save_logs()

    def _save_logs(self) -> None:
        log_dir = Path("logs") / "sessions" / self.session_id
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"execution_{self.execution_id}.json"
        with open(log_file, "w") as f:
            json.dump([asdict(log) for log in self.logs], f, indent=2)

    def get_logs(self) -> List[LogEntry]:
        return self.logs

    def get_trace(self) -> Dict[str, List]:
        return {
            "thoughts": self.thoughts,
            "actions": self.actions,
            "observations": self.observations,
        }
