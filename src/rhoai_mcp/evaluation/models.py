"""Data models for the evaluation harness.

This module defines the core data structures used for tracking
tool calls, evaluation sessions, and validation results.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SessionStatus(str, Enum):
    """Status of an evaluation session."""

    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ToolCall(BaseModel):
    """Record of a single tool invocation.

    Captures all relevant information about a tool call for
    later analysis and metrics calculation.
    """

    tool_name: str = Field(..., description="Name of the tool that was called")
    arguments: dict[str, Any] = Field(..., description="Arguments passed to the tool")
    result: Any = Field(..., description="Result returned by the tool (may be None on error)")
    duration_ms: float = Field(..., description="Execution time in milliseconds")
    success: bool = Field(..., description="True if the tool completed without exception")
    error: str | None = Field(None, description="Error message if the tool raised an exception")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the tool call occurred",
    )
    call_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this tool call",
    )

    model_config = {"arbitrary_types_allowed": True}

    def _serialize_result(self) -> Any:
        """Serialize result, handling non-JSON-serializable types."""
        if self.result is None:
            return None
        if isinstance(self.result, dict | list | str | int | float | bool):
            return self.result
        return str(self.result)


class ExpectedResult(BaseModel):
    """Expected result specification for validation.

    Defines what a tool result should look like for validation
    purposes. Supports multiple validation strategies.
    """

    tool_name: str = Field(..., description="Name of the tool this expectation applies to")
    required_fields: list[str] = Field(
        default_factory=list, description="Fields that must exist in the result"
    )
    field_values: dict[str, Any] = Field(
        default_factory=dict, description="Exact field value matches required"
    )
    field_patterns: dict[str, str] = Field(
        default_factory=dict, description="Regex patterns for field values"
    )
    custom_validator: str | None = Field(
        None, description="Name of custom validator function to use"
    )


class ValidationResult(BaseModel):
    """Result of validating a tool call against expectations.

    Contains detailed information about what passed and failed
    during validation.
    """

    passed: bool = Field(..., description="True if all validations passed")
    expected: ExpectedResult | None = Field(
        ..., description="The expected result that was validated against"
    )
    actual_result: Any = Field(..., description="The actual result that was validated")
    failures: list[str] = Field(
        default_factory=list, description="List of validation failure messages"
    )

    model_config = {"arbitrary_types_allowed": True}


class ExpectedToolSequence(BaseModel):
    """Define expected tool sequence for a task.

    Used for trajectory analysis to determine if the agent
    followed an optimal path to the goal.
    """

    required_tools: list[str] = Field(default_factory=list, description="Tools that MUST be called")
    optional_tools: list[str] = Field(default_factory=list, description="Tools that MAY be called")
    forbidden_tools: list[str] = Field(
        default_factory=list, description="Tools that should NOT be called"
    )
    expected_order: list[str] | None = Field(
        None, description="Expected call order (if order matters)"
    )


class TrajectorySpec(BaseModel):
    """Define expected trajectory for a task.

    Provides comprehensive trajectory expectations for
    multi-step trajectory evaluation.
    """

    goal_description: str = Field(..., description="Description of the goal to achieve")
    optimal_trajectory: list[str] = Field(
        default_factory=list, description="Optimal tool sequence to achieve the goal"
    )
    acceptable_trajectories: list[list[str]] = Field(
        default_factory=list, description="Alternative valid paths to the goal"
    )
    max_steps: int = Field(100, description="Maximum acceptable steps to complete the task")
    required_checkpoints: list[str] = Field(
        default_factory=list, description="Tools that must be called at some point"
    )


class ParameterSpec(BaseModel):
    """Expected parameter specification for precision analysis.

    Defines what a tool parameter should look like for
    parameter precision evaluation.
    """

    name: str = Field(..., description="Parameter name")
    required: bool = Field(True, description="Whether the parameter is required")
    expected_type: str = Field(
        "string",
        description="Expected type: 'string', 'int', 'float', 'bool', 'list', 'dict'",
    )
    pattern: str | None = Field(None, description="Regex pattern for string validation")
    min_value: float | None = Field(None, description="Minimum value for numeric types")
    max_value: float | None = Field(None, description="Maximum value for numeric types")
    allowed_values: list[Any] | None = Field(None, description="Enum of allowed values")


class EvaluationSession(BaseModel):
    """An evaluation session tracking tool calls and metrics.

    Represents a complete evaluation session from start to end,
    including all tool calls, expectations, and computed metrics.
    """

    session_id: str = Field(..., description="Unique identifier for this session")
    name: str = Field(..., description="Human-readable session name")
    task_definition: str = Field(..., description="Description of the task being evaluated")
    expected_outcome: str = Field(..., description="Description of the expected outcome")
    status: SessionStatus = Field(SessionStatus.ACTIVE, description="Current session status")
    tool_calls: list[ToolCall] = Field(
        default_factory=list, description="List of all tool calls in this session"
    )
    expected_results: list[ExpectedResult] = Field(
        default_factory=list, description="Expected results for validation"
    )
    expected_sequence: ExpectedToolSequence | None = Field(
        None, description="Expected tool sequence for trajectory analysis"
    )
    trajectory_spec: TrajectorySpec | None = Field(
        None, description="Trajectory specification for evaluation"
    )
    parameter_specs: dict[str, list[ParameterSpec]] = Field(
        default_factory=dict, description="Parameter specifications by tool name"
    )
    start_time: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the session started",
    )
    end_time: datetime | None = Field(None, description="When the session ended (if completed)")
    task_completed: bool = Field(False, description="Whether the task was successfully completed")
    notes: str = Field("", description="Additional notes about the session")

    def duration_seconds(self) -> float:
        """Calculate session duration in seconds."""
        end = self.end_time or datetime.now(timezone.utc)
        return (end - self.start_time).total_seconds()

    def tool_count(self) -> int:
        """Count total tool calls in this session."""
        return len(self.tool_calls)

    def success_count(self) -> int:
        """Count successful tool calls."""
        return sum(1 for tc in self.tool_calls if tc.success)

    def error_count(self) -> int:
        """Count failed tool calls."""
        return sum(1 for tc in self.tool_calls if not tc.success)

    def get_tool_sequence(self) -> list[str]:
        """Get the sequence of tool names called."""
        return [tc.tool_name for tc in self.tool_calls]


class EvaluationReport(BaseModel):
    """Summary report for an evaluation session.

    Contains computed metrics and analysis for a completed
    evaluation session.
    """

    session_id: str = Field(..., description="Session this report is for")
    session_name: str = Field(..., description="Human-readable session name")
    task_definition: str = Field(..., description="Description of the task that was evaluated")
    task_completed: bool = Field(..., description="Whether the task was successfully completed")
    duration_seconds: float = Field(..., description="Total session duration")
    tool_count: int = Field(..., description="Total number of tool calls")
    success_count: int = Field(..., description="Number of successful tool calls")
    error_count: int = Field(..., description="Number of failed tool calls")
    success_rate: float = Field(..., description="Percentage of successful tool calls")
    tool_sequence: list[str] = Field(..., description="Sequence of tools called")
    validation_results: list[ValidationResult] = Field(
        default_factory=list,
        description="Results of validating tool calls against expectations",
    )

    # Metrics summaries (populated after metrics calculation)
    stability_score: float | None = Field(None, description="Stability dimension score")
    performance_score: float | None = Field(None, description="Performance dimension score")
    tool_selection_score: float | None = Field(None, description="Tool selection dimension score")
    parameter_precision_score: float | None = Field(
        None, description="Parameter precision dimension score"
    )
    trajectory_score: float | None = Field(None, description="Trajectory dimension score")
    composite_score: float | None = Field(None, description="Composite overall score")
    grade: str | None = Field(None, description="Letter grade based on composite score")

    @classmethod
    def from_session(
        cls,
        session: EvaluationSession,
        validation_results: list[ValidationResult] | None = None,
    ) -> EvaluationReport:
        """Create a report from a session."""
        tool_count = session.tool_count()
        success_count = session.success_count()
        success_rate = (success_count / tool_count * 100) if tool_count > 0 else 0.0

        return cls(
            session_id=session.session_id,
            session_name=session.name,
            task_definition=session.task_definition,
            task_completed=session.task_completed,
            duration_seconds=session.duration_seconds(),
            tool_count=tool_count,
            success_count=success_count,
            error_count=session.error_count(),
            success_rate=success_rate,
            tool_sequence=session.get_tool_sequence(),
            validation_results=validation_results or [],
        )
