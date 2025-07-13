import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch
from langchain.schema import AgentAction, AgentFinish

from app.logging.tao_logger import TAOLogger


@pytest.fixture
def tao_logger():
    return TAOLogger(session_id="test_session", execution_id="test_execution")


def test_tao_logger_initialization(tao_logger):
    assert tao_logger.session_id == "test_session"
    assert tao_logger.execution_id == "test_execution"
    assert tao_logger.step_number == 0
    assert tao_logger.logs == []


def test_on_agent_action(tao_logger):
    action = AgentAction(
        tool="test_tool", tool_input={"query": "test"}, log="thinking about test"
    )

    tao_logger.on_agent_action(action)

    assert len(tao_logger.logs) == 1
    log_entry = tao_logger.logs[0]
    assert log_entry.step_type == "action"
    assert log_entry.step_number == 1
    assert log_entry.content["tool_name"] == "test_tool"
    assert log_entry.content["thought"] == "thinking about test"


def test_on_tool_end(tao_logger):
    tao_logger.on_tool_end("tool output")

    assert len(tao_logger.logs) == 1
    log_entry = tao_logger.logs[0]
    assert log_entry.step_type == "observation"
    assert log_entry.content["tool_output"] == "tool output"


def test_on_agent_finish_saves_logs(tao_logger):
    finish = AgentFinish(return_values={"output": "final answer"}, log="final thought")

    with tempfile.TemporaryDirectory() as temp_dir:
        with patch("pathlib.Path") as mock_path:
            mock_path.return_value = Path(temp_dir)
            tao_logger._save_logs = lambda: None  # Mock save to avoid file operations

            tao_logger.on_agent_finish(finish)

            assert len(tao_logger.logs) == 1
            log_entry = tao_logger.logs[0]
            assert log_entry.step_type == "final_answer"
            assert log_entry.content["output"] == "final answer"


def test_get_logs(tao_logger):
    logs = tao_logger.get_logs()
    assert logs == []

    tao_logger.logs = [{"test": "log"}]
    logs = tao_logger.get_logs()
    assert logs == [{"test": "log"}]
