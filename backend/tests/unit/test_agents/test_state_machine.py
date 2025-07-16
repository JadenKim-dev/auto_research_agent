import pytest
from datetime import datetime
from app.agents.state_machine import AgentState, AgentStateMachine


class TestAgentStateMachine:

    def test_valid_transitions_from_idle(self):
        state_machine = AgentStateMachine("test-session")

        assert state_machine.can_transition(AgentState.RUNNING) is True
        assert state_machine.can_transition(AgentState.ERROR) is True
        assert state_machine.can_transition(AgentState.COMPLETED) is False
        assert state_machine.can_transition(AgentState.IDLE) is False

    def test_valid_transitions_from_running(self):
        state_machine = AgentStateMachine("test-session")
        state_machine.current_state = AgentState.RUNNING

        assert state_machine.can_transition(AgentState.COMPLETED) is True
        assert state_machine.can_transition(AgentState.ERROR) is True
        assert state_machine.can_transition(AgentState.IDLE) is False
        assert state_machine.can_transition(AgentState.RUNNING) is False

    def test_valid_transitions_from_error(self):
        state_machine = AgentStateMachine("test-session")
        state_machine.current_state = AgentState.ERROR

        assert state_machine.can_transition(AgentState.RUNNING) is True
        assert state_machine.can_transition(AgentState.IDLE) is True
        assert state_machine.can_transition(AgentState.ERROR) is False
        assert state_machine.can_transition(AgentState.COMPLETED) is False

    def test_valid_transitions_from_completed(self):
        state_machine = AgentStateMachine("test-session")
        state_machine.current_state = AgentState.COMPLETED

        assert state_machine.can_transition(AgentState.IDLE) is True
        assert state_machine.can_transition(AgentState.RUNNING) is True
        assert state_machine.can_transition(AgentState.ERROR) is True
        assert state_machine.can_transition(AgentState.COMPLETED) is False

    def test_successful_transition(self):
        state_machine = AgentStateMachine("test-session")

        result = state_machine.transition_to(AgentState.RUNNING)

        assert result is True
        assert state_machine.current_state == AgentState.RUNNING
        assert len(state_machine.state_history) == 1

        history_entry = state_machine.state_history[0]
        assert history_entry["from_state"] == "idle"
        assert history_entry["to_state"] == "running"
        assert "timestamp" in history_entry
        assert history_entry["error_context"] is None

    def test_failed_transition(self):
        state_machine = AgentStateMachine("test-session")

        result = state_machine.transition_to(AgentState.COMPLETED)

        assert result is False
        assert state_machine.current_state == AgentState.IDLE
        assert len(state_machine.state_history) == 0

    def test_error_context_handling(self):
        state_machine = AgentStateMachine("test-session")
        state_machine.transition_to(AgentState.RUNNING)

        error_context = {"error_type": "api_failure", "message": "Connection timeout"}
        result = state_machine.transition_to(AgentState.ERROR, error_context)

        assert result is True
        assert state_machine.current_state == AgentState.ERROR
        assert state_machine.error_context == error_context

        history_entry = state_machine.state_history[-1]
        assert history_entry["error_context"] == error_context

    def test_error_context_cleared_on_completion(self):
        state_machine = AgentStateMachine("test-session")
        state_machine.transition_to(AgentState.RUNNING)
        state_machine.transition_to(AgentState.ERROR, {"error": "test"})

        assert state_machine.error_context is not None

        state_machine.transition_to(AgentState.RUNNING)
        state_machine.transition_to(AgentState.COMPLETED)

        assert state_machine.error_context is None

    def test_error_context_cleared_on_idle(self):
        state_machine = AgentStateMachine("test-session")
        state_machine.transition_to(AgentState.RUNNING)
        state_machine.transition_to(AgentState.ERROR, {"error": "test"})

        assert state_machine.error_context is not None

        state_machine.transition_to(AgentState.IDLE)

        assert state_machine.error_context is None

    def test_get_state(self):
        state_machine = AgentStateMachine("test-session")

        assert state_machine.get_state() == AgentState.IDLE

        state_machine.transition_to(AgentState.RUNNING)
        assert state_machine.get_state() == AgentState.RUNNING

    def test_state_history_limit(self):
        state_machine = AgentStateMachine("test-session")

        # Create more than 5 transitions
        for i in range(7):
            if i % 2 == 0:
                state_machine.transition_to(AgentState.RUNNING)
            else:
                state_machine.transition_to(AgentState.COMPLETED)

        state_info = state_machine.get_state_info()
        assert len(state_info["state_history"]) == 5  # Only last 5 transitions

    def test_reset(self):
        state_machine = AgentStateMachine("test-session")
        state_machine.transition_to(AgentState.RUNNING)
        state_machine.transition_to(AgentState.ERROR, {"error": "test"})

        assert state_machine.current_state == AgentState.ERROR
        assert state_machine.error_context is not None
        assert len(state_machine.state_history) > 0

        state_machine.reset()

        assert state_machine.current_state == AgentState.IDLE
        assert state_machine.error_context is None
        assert state_machine.state_history == []
        assert isinstance(state_machine.last_transition_time, datetime)

    def test_full_workflow(self):
        state_machine = AgentStateMachine("workflow-test")

        # IDLE -> RUNNING
        assert state_machine.transition_to(AgentState.RUNNING) is True
        assert state_machine.get_state() == AgentState.RUNNING

        # RUNNING -> COMPLETED
        assert state_machine.transition_to(AgentState.COMPLETED) is True
        assert state_machine.get_state() == AgentState.COMPLETED

        # COMPLETED -> IDLE
        assert state_machine.transition_to(AgentState.IDLE) is True
        assert state_machine.get_state() == AgentState.IDLE

        # IDLE -> RUNNING -> ERROR -> RUNNING -> COMPLETED
        state_machine.transition_to(AgentState.RUNNING)
        error_ctx = {"type": "timeout"}
        state_machine.transition_to(AgentState.ERROR, error_ctx)
        assert state_machine.error_context == error_ctx

        state_machine.transition_to(AgentState.RUNNING)
        state_machine.transition_to(AgentState.COMPLETED)
        assert state_machine.error_context is None
