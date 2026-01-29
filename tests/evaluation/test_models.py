"""Tests for evaluation models."""

from datetime import datetime, timezone

import pytest

from rhoai_mcp.evaluation.models import (
    EvaluationReport,
    EvaluationSession,
    ExpectedResult,
    ExpectedToolSequence,
    ParameterSpec,
    SessionStatus,
    ToolCall,
    TrajectorySpec,
    ValidationResult,
)


class TestToolCall:
    """Test ToolCall dataclass."""

    def test_create_tool_call(self) -> None:
        """Test creating a tool call."""
        call = ToolCall(
            tool_name="test_tool",
            arguments={"arg1": "value1"},
            result={"status": "ok"},
            duration_ms=100.0,
            success=True,
        )

        assert call.tool_name == "test_tool"
        assert call.arguments == {"arg1": "value1"}
        assert call.result == {"status": "ok"}
        assert call.duration_ms == 100.0
        assert call.success is True
        assert call.error is None
        assert call.call_id is not None

    def test_tool_call_with_error(self) -> None:
        """Test creating a failed tool call."""
        call = ToolCall(
            tool_name="test_tool",
            arguments={},
            result=None,
            duration_ms=50.0,
            success=False,
            error="Something went wrong",
        )

        assert call.success is False
        assert call.error == "Something went wrong"

    def test_model_dump(self) -> None:
        """Test serialization to dict."""
        call = ToolCall(
            tool_name="test_tool",
            arguments={"key": "value"},
            result="result",
            duration_ms=100.0,
            success=True,
        )

        data = call.model_dump()

        assert data["tool_name"] == "test_tool"
        assert data["arguments"] == {"key": "value"}
        assert data["result"] == "result"
        assert data["duration_ms"] == 100.0
        assert data["success"] is True
        assert "timestamp" in data
        assert "call_id" in data


class TestExpectedResult:
    """Test ExpectedResult dataclass."""

    def test_create_expected_result(self) -> None:
        """Test creating an expected result."""
        expected = ExpectedResult(
            tool_name="get_details",
            required_fields=["name", "status"],
            field_values={"status": "Active"},
            field_patterns={"name": r"^test-.*$"},
        )

        assert expected.tool_name == "get_details"
        assert expected.required_fields == ["name", "status"]
        assert expected.field_values == {"status": "Active"}
        assert expected.field_patterns == {"name": r"^test-.*$"}

    def test_model_dump(self) -> None:
        """Test serialization to dict."""
        expected = ExpectedResult(
            tool_name="test",
            required_fields=["field1"],
            custom_validator="my_validator",
        )

        data = expected.model_dump()

        assert data["tool_name"] == "test"
        assert data["required_fields"] == ["field1"]
        assert data["custom_validator"] == "my_validator"


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_passed_validation(self) -> None:
        """Test a passing validation result."""
        result = ValidationResult(
            passed=True,
            expected=None,
            actual_result={"status": "ok"},
        )

        assert result.passed is True
        assert result.failures == []

    def test_failed_validation(self) -> None:
        """Test a failing validation result."""
        result = ValidationResult(
            passed=False,
            expected=None,
            actual_result=None,
            failures=["Missing required field: name", "Value mismatch"],
        )

        assert result.passed is False
        assert len(result.failures) == 2


class TestExpectedToolSequence:
    """Test ExpectedToolSequence dataclass."""

    def test_create_sequence(self) -> None:
        """Test creating an expected tool sequence."""
        seq = ExpectedToolSequence(
            required_tools=["tool_a", "tool_b"],
            optional_tools=["tool_c"],
            forbidden_tools=["tool_x"],
            expected_order=["tool_a", "tool_b"],
        )

        assert seq.required_tools == ["tool_a", "tool_b"]
        assert seq.optional_tools == ["tool_c"]
        assert seq.forbidden_tools == ["tool_x"]
        assert seq.expected_order == ["tool_a", "tool_b"]


class TestTrajectorySpec:
    """Test TrajectorySpec dataclass."""

    def test_create_trajectory_spec(self) -> None:
        """Test creating a trajectory specification."""
        spec = TrajectorySpec(
            goal_description="Complete the task",
            optimal_trajectory=["step1", "step2", "step3"],
            acceptable_trajectories=[["step1", "step3"]],
            max_steps=10,
            required_checkpoints=["step1"],
        )

        assert spec.goal_description == "Complete the task"
        assert len(spec.optimal_trajectory) == 3
        assert spec.max_steps == 10


class TestParameterSpec:
    """Test ParameterSpec dataclass."""

    def test_create_parameter_spec(self) -> None:
        """Test creating a parameter specification."""
        spec = ParameterSpec(
            name="count",
            required=True,
            expected_type="int",
            min_value=1,
            max_value=100,
        )

        assert spec.name == "count"
        assert spec.required is True
        assert spec.expected_type == "int"
        assert spec.min_value == 1
        assert spec.max_value == 100


class TestEvaluationSession:
    """Test EvaluationSession dataclass."""

    def test_create_session(self) -> None:
        """Test creating an evaluation session."""
        session = EvaluationSession(
            session_id="test-123",
            name="Test Session",
            task_definition="Do something",
            expected_outcome="Something is done",
        )

        assert session.session_id == "test-123"
        assert session.name == "Test Session"
        assert session.status == SessionStatus.ACTIVE

    def test_session_metrics(self, sample_tool_calls) -> None:
        """Test session metric calculations."""
        session = EvaluationSession(
            session_id="test",
            name="Test",
            task_definition="Test",
            expected_outcome="Test",
            tool_calls=sample_tool_calls,
        )

        assert session.tool_count() == 4
        assert session.success_count() == 4
        assert session.error_count() == 0

    def test_get_tool_sequence(self, sample_tool_calls) -> None:
        """Test getting the tool sequence."""
        session = EvaluationSession(
            session_id="test",
            name="Test",
            task_definition="Test",
            expected_outcome="Test",
            tool_calls=sample_tool_calls,
        )

        seq = session.get_tool_sequence()

        assert seq == [
            "list_projects",
            "get_project_details",
            "create_workbench",
            "get_workbench_status",
        ]

    def test_model_dump(self) -> None:
        """Test serialization to dict."""
        session = EvaluationSession(
            session_id="test",
            name="Test",
            task_definition="Do it",
            expected_outcome="Done",
        )

        data = session.model_dump()

        assert data["session_id"] == "test"
        assert data["name"] == "Test"
        assert data["status"] == "active"


class TestEvaluationReport:
    """Test EvaluationReport dataclass."""

    def test_from_session(self, sample_session) -> None:
        """Test creating report from session."""
        report = EvaluationReport.from_session(sample_session)

        assert report.session_id == sample_session.session_id
        assert report.session_name == sample_session.name
        assert report.tool_count == 4
        assert report.success_count == 4
        assert report.error_count == 0
        assert report.success_rate == 100.0

    def test_model_dump(self, sample_session) -> None:
        """Test serialization to dict."""
        report = EvaluationReport.from_session(sample_session)
        data = report.model_dump()

        assert data["session_id"] == sample_session.session_id
        assert data["tool_count"] == 4
