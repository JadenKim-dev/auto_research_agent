from enum import Enum
from typing import Optional, Dict, Any
from datetime import datetime


class AgentState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    ERROR = "error"
    COMPLETED = "completed"


class AgentStateMachine:

    VALID_TRANSITIONS = {
        AgentState.IDLE: [AgentState.RUNNING, AgentState.ERROR],
        AgentState.RUNNING: [AgentState.COMPLETED, AgentState.ERROR],
        AgentState.ERROR: [AgentState.RUNNING, AgentState.IDLE],
        AgentState.COMPLETED: [AgentState.IDLE, AgentState.RUNNING, AgentState.ERROR],
    }

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.current_state = AgentState.IDLE
        self.error_context: Optional[Dict[str, Any]] = None
        self.state_history = []
        self.last_transition_time = datetime.now()

    def can_transition(self, target_state: AgentState) -> bool:
        return target_state in self.VALID_TRANSITIONS.get(self.current_state, [])

    def transition_to(
        self, new_state: AgentState, error_context: Optional[Dict[str, Any]] = None
    ) -> bool:
        if not self.can_transition(new_state):
            return False

        self.state_history.append(
            {
                "from_state": self.current_state.value,
                "to_state": new_state.value,
                "timestamp": datetime.now().isoformat(),
                "error_context": error_context,
            }
        )

        self.current_state = new_state
        self.last_transition_time = datetime.now()

        if new_state == AgentState.ERROR:
            self.error_context = error_context
        elif new_state in [AgentState.COMPLETED, AgentState.IDLE]:
            self.error_context = None

        return True

    def get_state(self) -> AgentState:
        return self.current_state

    def get_state_info(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "current_state": self.current_state.value,
            "error_context": self.error_context,
            "last_transition_time": self.last_transition_time.isoformat(),
            "state_history": self.state_history[-5:],  # Last 5 transitions
        }

    def reset(self):
        self.current_state = AgentState.IDLE
        self.error_context = None
        self.state_history = []
        self.last_transition_time = datetime.now()
