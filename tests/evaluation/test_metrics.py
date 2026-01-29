"""Tests for evaluation metrics."""

import pytest

from rhoai_mcp.evaluation.metrics import (
    ParameterPrecisionMetrics,
    PerformanceMetrics,
    StabilityMetrics,
    SuccessErrorMetrics,
    ToolSelectionMetrics,
    TrajectoryMetrics,
)
from rhoai_mcp.evaluation.models import (
    EvaluationSession,
    ExpectedToolSequence,
    ParameterSpec,
    SessionStatus,
    ToolCall,
    TrajectorySpec,
)


class TestStabilityMetrics:
    """Test StabilityMetrics calculation."""

    def test_from_empty_calls(self) -> None:
        """Test metrics from empty call list."""
        metrics = StabilityMetrics.from_tool_calls([])

        assert metrics.total_calls == 0
        assert metrics.successful_calls == 0
        assert metrics.stability_score == 1.0

    def test_from_all_successful_calls(self, sample_tool_calls) -> None:
        """Test metrics from all successful calls."""
        metrics = StabilityMetrics.from_tool_calls(sample_tool_calls)

        assert metrics.total_calls == 4
        assert metrics.successful_calls == 4
        assert metrics.stability_score == 1.0
        assert metrics.error_types == {}

    def test_from_calls_with_errors(self, sample_tool_calls_with_errors) -> None:
        """Test metrics from calls with errors."""
        metrics = StabilityMetrics.from_tool_calls(sample_tool_calls_with_errors)

        assert metrics.total_calls == 4
        assert metrics.successful_calls == 2
        assert metrics.stability_score == 0.5
        assert "not_found" in metrics.error_types

    def test_model_dump(self, sample_tool_calls) -> None:
        """Test serialization to dict."""
        metrics = StabilityMetrics.from_tool_calls(sample_tool_calls)
        data = metrics.model_dump()

        assert data["total_calls"] == 4
        assert data["stability_score"] == 1.0


class TestPerformanceMetrics:
    """Test PerformanceMetrics calculation."""

    def test_from_empty_calls(self) -> None:
        """Test metrics from empty call list."""
        metrics = PerformanceMetrics.from_tool_calls([], 1.0)

        assert metrics.latency_p50_ms == 0.0
        assert metrics.calls_per_second == 0.0

    def test_from_tool_calls(self, sample_tool_calls) -> None:
        """Test metrics from tool calls."""
        metrics = PerformanceMetrics.from_tool_calls(sample_tool_calls, 10.0)

        # Durations are 150, 200, 500, 100
        assert metrics.min_duration_ms == 100.0
        assert metrics.max_duration_ms == 500.0
        assert metrics.avg_duration_ms == pytest.approx(237.5)
        assert metrics.total_duration_ms == 950.0
        assert metrics.calls_per_second == 0.4  # 4 calls / 10 seconds

    def test_latency_percentiles(self) -> None:
        """Test latency percentile calculation."""
        calls = [
            ToolCall(
                tool_name="test",
                arguments={},
                result=None,
                duration_ms=float(i * 10),
                success=True,
            )
            for i in range(1, 101)  # 10, 20, ..., 1000
        ]

        metrics = PerformanceMetrics.from_tool_calls(calls, 100.0)

        # With 100 samples (10, 20, ..., 1000)
        # p50 index = int(100 * 0.50) = 50, sorted[50] = 510
        # p95 index = int(100 * 0.95) = 95, sorted[95] = 960
        # p99 index = int(100 * 0.99) = 99, sorted[99] = 1000
        assert metrics.latency_p50_ms == 510.0
        assert metrics.latency_p95_ms == 960.0
        assert metrics.latency_p99_ms == 1000.0


class TestToolSelectionMetrics:
    """Test ToolSelectionMetrics calculation."""

    def test_without_expectations(self, sample_session) -> None:
        """Test metrics without expected sequence."""
        metrics = ToolSelectionMetrics.from_session(sample_session)

        # No expectations means everything is fine
        assert metrics.required_coverage == 1.0
        assert metrics.forbidden_tool_violations == []

    def test_with_expectations(
        self, sample_session, sample_expected_sequence
    ) -> None:
        """Test metrics with expected sequence."""
        sample_session.expected_sequence = sample_expected_sequence
        metrics = ToolSelectionMetrics.from_session(sample_session)

        # Session has: list_projects, get_project_details, create_workbench, get_workbench_status
        # Required: list_projects, create_workbench (both called)
        assert metrics.required_tools_called == 2
        assert metrics.required_tools_total == 2
        assert metrics.required_coverage == 1.0
        assert metrics.forbidden_tool_violations == []

    def test_forbidden_violations(self, sample_session) -> None:
        """Test detection of forbidden tool calls."""
        sample_session.tool_calls.append(
            ToolCall(
                tool_name="delete_project",
                arguments={},
                result=None,
                duration_ms=100.0,
                success=True,
            )
        )
        sample_session.expected_sequence = ExpectedToolSequence(
            required_tools=[],
            forbidden_tools=["delete_project"],
        )

        metrics = ToolSelectionMetrics.from_session(sample_session)

        assert "delete_project" in metrics.forbidden_tool_violations


class TestSuccessErrorMetrics:
    """Test SuccessErrorMetrics calculation."""

    def test_from_empty_calls(self) -> None:
        """Test metrics from empty call list."""
        metrics = SuccessErrorMetrics.from_tool_calls([])

        assert metrics.success_count == 0
        assert metrics.error_count == 0
        assert metrics.success_rate == 1.0

    def test_from_successful_calls(self, sample_tool_calls) -> None:
        """Test metrics from successful calls."""
        metrics = SuccessErrorMetrics.from_tool_calls(sample_tool_calls)

        assert metrics.success_count == 4
        assert metrics.error_count == 0
        assert metrics.success_rate == 1.0

    def test_from_calls_with_errors(self, sample_tool_calls_with_errors) -> None:
        """Test metrics from calls with errors."""
        metrics = SuccessErrorMetrics.from_tool_calls(sample_tool_calls_with_errors)

        assert metrics.success_count == 2
        assert metrics.error_count == 2
        assert metrics.success_rate == 0.5

    def test_retry_detection(self) -> None:
        """Test retry detection."""
        calls = [
            ToolCall(
                tool_name="get_data",
                arguments={},
                result=None,
                duration_ms=100.0,
                success=False,
                error="Timeout",
            ),
            ToolCall(
                tool_name="get_data",
                arguments={},
                result={"data": "ok"},
                duration_ms=100.0,
                success=True,
            ),
        ]

        metrics = SuccessErrorMetrics.from_tool_calls(calls)

        assert metrics.retry_attempts == 1
        assert metrics.retry_success_rate == 1.0


class TestParameterPrecisionMetrics:
    """Test ParameterPrecisionMetrics calculation."""

    def test_without_specs(self, sample_session) -> None:
        """Test metrics without parameter specs."""
        metrics = ParameterPrecisionMetrics.from_session(sample_session)

        assert metrics.total_parameters == 0
        assert metrics.precision_score == 1.0

    def test_with_valid_parameters(self, sample_session) -> None:
        """Test metrics with valid parameters."""
        sample_session.parameter_specs = {
            "get_project_details": [
                ParameterSpec(
                    name="name",
                    required=True,
                    expected_type="string",
                )
            ]
        }

        metrics = ParameterPrecisionMetrics.from_session(sample_session)

        assert metrics.total_parameters == 1
        assert metrics.valid_parameters == 1
        assert metrics.precision_score == 1.0

    def test_with_invalid_parameters(self) -> None:
        """Test metrics with invalid parameters."""
        session = EvaluationSession(
            session_id="test",
            name="Test",
            task_definition="Test",
            expected_outcome="Test",
            tool_calls=[
                ToolCall(
                    tool_name="test_tool",
                    arguments={"count": "not_a_number"},  # Should be int
                    result=None,
                    duration_ms=100.0,
                    success=True,
                )
            ],
        )
        session.parameter_specs = {
            "test_tool": [
                ParameterSpec(
                    name="count",
                    required=True,
                    expected_type="int",
                )
            ]
        }

        metrics = ParameterPrecisionMetrics.from_session(session)

        assert metrics.total_parameters == 1
        assert metrics.valid_parameters == 0
        assert metrics.precision_score == 0.0
        assert len(metrics.type_errors) == 1


class TestTrajectoryMetrics:
    """Test TrajectoryMetrics calculation."""

    def test_without_spec(self, sample_session) -> None:
        """Test metrics without trajectory spec."""
        metrics = TrajectoryMetrics.from_session(sample_session)

        assert metrics.total_steps == 4
        assert metrics.efficiency_score == 1.0
        assert metrics.trajectory_similarity == 1.0

    def test_with_optimal_trajectory(
        self, sample_session, sample_trajectory_spec
    ) -> None:
        """Test metrics with optimal trajectory."""
        sample_session.trajectory_spec = sample_trajectory_spec
        sample_session.task_completed = True

        metrics = TrajectoryMetrics.from_session(sample_session)

        # Actual: list_projects, get_project_details, create_workbench, get_workbench_status
        # Optimal: same
        assert metrics.total_steps == 4
        assert metrics.optimal_steps == 4
        assert metrics.efficiency_score == 1.0
        assert metrics.trajectory_similarity == 1.0

    def test_checkpoint_coverage(self, sample_session, sample_trajectory_spec) -> None:
        """Test checkpoint coverage calculation."""
        sample_session.trajectory_spec = sample_trajectory_spec
        sample_session.task_completed = True

        metrics = TrajectoryMetrics.from_session(sample_session)

        # Required checkpoints: create_workbench, get_workbench_status (both called)
        assert metrics.checkpoints_hit == 2
        assert metrics.checkpoints_total == 2
        assert metrics.checkpoint_coverage == 1.0

    def test_efficiency_with_extra_steps(self) -> None:
        """Test efficiency when more steps than optimal."""
        session = EvaluationSession(
            session_id="test",
            name="Test",
            task_definition="Test",
            expected_outcome="Test",
            task_completed=True,
            tool_calls=[
                ToolCall(tool_name=f"step_{i}", arguments={}, result=None, duration_ms=10.0, success=True)
                for i in range(10)  # 10 steps
            ],
        )
        session.trajectory_spec = TrajectorySpec(
            goal_description="Test",
            optimal_trajectory=["step_0", "step_1"],  # Only 2 optimal steps
        )

        metrics = TrajectoryMetrics.from_session(session)

        assert metrics.total_steps == 10
        assert metrics.optimal_steps == 2
        assert metrics.efficiency_score == 0.2  # 2/10

    def test_backtracking_detection(self) -> None:
        """Test backtracking detection."""
        session = EvaluationSession(
            session_id="test",
            name="Test",
            task_definition="Test",
            expected_outcome="Test",
            tool_calls=[
                ToolCall(tool_name="step_a", arguments={}, result=None, duration_ms=10.0, success=True),
                ToolCall(tool_name="step_b", arguments={}, result=None, duration_ms=10.0, success=True),
                ToolCall(tool_name="step_c", arguments={}, result=None, duration_ms=10.0, success=True),
                ToolCall(tool_name="step_a", arguments={}, result=None, duration_ms=10.0, success=True),  # Backtrack
            ],
        )
        session.trajectory_spec = TrajectorySpec(goal_description="Test")

        metrics = TrajectoryMetrics.from_session(session)

        assert metrics.backtracking_count == 1
