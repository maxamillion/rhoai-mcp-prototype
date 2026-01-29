"""Tests for composite scoring."""

import pytest

from rhoai_mcp.evaluation.metrics import (
    ParameterPrecisionMetrics,
    PerformanceMetrics,
    StabilityMetrics,
    SuccessErrorMetrics,
    ToolSelectionMetrics,
    TrajectoryMetrics,
)
from rhoai_mcp.evaluation.models import EvaluationSession, SessionStatus, TrajectorySpec
from rhoai_mcp.evaluation.scoring import (
    DEFAULT_WEIGHTS,
    calculate_composite_score,
    calculate_score_from_session,
)


class TestCalculateCompositeScore:
    """Test calculate_composite_score function."""

    def test_perfect_score(self) -> None:
        """Test calculating a perfect score."""
        stability = StabilityMetrics(
            total_calls=10,
            successful_calls=10,
            stability_score=1.0,
        )
        performance = PerformanceMetrics(
            latency_p50_ms=100.0,
            latency_p95_ms=200.0,
            latency_p99_ms=300.0,
            avg_duration_ms=150.0,
            max_duration_ms=300.0,
            min_duration_ms=100.0,
            total_duration_ms=1500.0,
            calls_per_second=10.0,
        )
        tool_selection = ToolSelectionMetrics(
            required_tools_called=5,
            required_tools_total=5,
            required_coverage=1.0,
        )
        success_error = SuccessErrorMetrics(
            success_count=10,
            error_count=0,
            success_rate=1.0,
        )
        param_precision = ParameterPrecisionMetrics(
            total_parameters=20,
            valid_parameters=20,
            precision_score=1.0,
        )
        trajectory = TrajectoryMetrics(
            total_steps=5,
            optimal_steps=5,
            efficiency_score=1.0,
            goal_achieved=True,
            trajectory_similarity=1.0,
            checkpoint_coverage=1.0,
        )

        score = calculate_composite_score(
            stability=stability,
            performance=performance,
            tool_selection=tool_selection,
            success_error=success_error,
            parameter_precision=param_precision,
            trajectory=trajectory,
            baseline_latency_p95_ms=500.0,  # Performance is good (200 < 500)
        )

        assert score.stability_score == 1.0
        assert score.performance_score == 1.0  # Under baseline
        assert score.tool_selection_score == 1.0
        assert score.success_rate_score == 1.0
        assert score.parameter_precision_score == 1.0
        assert score.trajectory_score == 1.0
        assert score.overall_score == pytest.approx(1.0)
        assert score.grade == "A"

    def test_poor_score(self) -> None:
        """Test calculating a poor score."""
        stability = StabilityMetrics(
            total_calls=10,
            successful_calls=5,
            stability_score=0.5,
        )
        performance = PerformanceMetrics(
            latency_p50_ms=2000.0,
            latency_p95_ms=5000.0,  # 5x baseline
            latency_p99_ms=8000.0,
            avg_duration_ms=3000.0,
            max_duration_ms=8000.0,
            min_duration_ms=1000.0,
            total_duration_ms=30000.0,
            calls_per_second=1.0,
        )
        tool_selection = ToolSelectionMetrics(
            required_tools_called=2,
            required_tools_total=5,
            required_coverage=0.4,
        )
        success_error = SuccessErrorMetrics(
            success_count=5,
            error_count=5,
            success_rate=0.5,
        )
        param_precision = ParameterPrecisionMetrics(
            total_parameters=20,
            valid_parameters=10,
            precision_score=0.5,
        )
        trajectory = TrajectoryMetrics(
            total_steps=20,
            optimal_steps=5,
            efficiency_score=0.25,
            goal_achieved=False,
            trajectory_similarity=0.3,
            checkpoint_coverage=0.5,
        )

        score = calculate_composite_score(
            stability=stability,
            performance=performance,
            tool_selection=tool_selection,
            success_error=success_error,
            parameter_precision=param_precision,
            trajectory=trajectory,
            baseline_latency_p95_ms=1000.0,  # 5x over baseline
        )

        assert score.stability_score == 0.5
        assert score.performance_score == pytest.approx(0.2)  # 1/5
        assert score.success_rate_score == 0.5
        assert score.parameter_precision_score == 0.5
        assert score.overall_score < 0.6
        assert score.grade in ("D", "F")

    def test_custom_weights(self) -> None:
        """Test with custom weights."""
        # All metrics at 1.0 except stability at 0.0
        stability = StabilityMetrics(
            total_calls=10,
            successful_calls=0,
            stability_score=0.0,
        )
        performance = PerformanceMetrics(
            latency_p50_ms=100.0,
            latency_p95_ms=200.0,
            latency_p99_ms=300.0,
            avg_duration_ms=150.0,
            max_duration_ms=300.0,
            min_duration_ms=100.0,
            total_duration_ms=1500.0,
            calls_per_second=10.0,
        )
        tool_selection = ToolSelectionMetrics(
            required_tools_called=5,
            required_tools_total=5,
            required_coverage=1.0,
        )
        success_error = SuccessErrorMetrics(
            success_count=10,
            error_count=0,
            success_rate=1.0,
        )
        param_precision = ParameterPrecisionMetrics(
            total_parameters=20,
            valid_parameters=20,
            precision_score=1.0,
        )
        trajectory = TrajectoryMetrics(
            total_steps=5,
            optimal_steps=5,
            efficiency_score=1.0,
            goal_achieved=True,
            trajectory_similarity=1.0,
            checkpoint_coverage=1.0,
        )

        # Give stability all the weight
        heavy_stability_weights = {
            "stability": 1.0,
            "performance": 0.0,
            "tool_selection": 0.0,
            "success_rate": 0.0,
            "parameter_precision": 0.0,
            "trajectory": 0.0,
        }

        score = calculate_composite_score(
            stability=stability,
            performance=performance,
            tool_selection=tool_selection,
            success_error=success_error,
            parameter_precision=param_precision,
            trajectory=trajectory,
            weights=heavy_stability_weights,
        )

        assert score.overall_score == 0.0  # Only stability matters, and it's 0

    def test_model_dump(self) -> None:
        """Test score serialization."""
        stability = StabilityMetrics(
            total_calls=10,
            successful_calls=10,
            stability_score=1.0,
        )
        performance = PerformanceMetrics(
            latency_p50_ms=100.0,
            latency_p95_ms=200.0,
            latency_p99_ms=300.0,
            avg_duration_ms=150.0,
            max_duration_ms=300.0,
            min_duration_ms=100.0,
            total_duration_ms=1500.0,
            calls_per_second=10.0,
        )
        tool_selection = ToolSelectionMetrics(
            required_tools_called=5,
            required_tools_total=5,
            required_coverage=1.0,
        )
        success_error = SuccessErrorMetrics(
            success_count=10,
            error_count=0,
            success_rate=1.0,
        )
        param_precision = ParameterPrecisionMetrics(
            total_parameters=20,
            valid_parameters=20,
            precision_score=1.0,
        )
        trajectory = TrajectoryMetrics(
            total_steps=5,
            optimal_steps=5,
            efficiency_score=1.0,
            goal_achieved=True,
            trajectory_similarity=1.0,
            checkpoint_coverage=1.0,
        )

        score = calculate_composite_score(
            stability=stability,
            performance=performance,
            tool_selection=tool_selection,
            success_error=success_error,
            parameter_precision=param_precision,
            trajectory=trajectory,
        )

        data = score.model_dump()

        assert "scores" in data
        assert "overall_score" in data
        assert "grade" in data
        assert "metrics" in data
        assert "weights" in data


class TestCalculateScoreFromSession:
    """Test calculate_score_from_session function."""

    def test_from_session(self, sample_session) -> None:
        """Test calculating score from a session."""
        sample_session.task_completed = True

        score = calculate_score_from_session(sample_session)

        assert score.stability_score == 1.0  # All calls succeeded
        assert score.success_rate_score == 1.0
        assert score.grade is not None

    def test_from_session_with_trajectory_spec(self, sample_session, sample_trajectory_spec) -> None:
        """Test with trajectory specification."""
        sample_session.trajectory_spec = sample_trajectory_spec
        sample_session.task_completed = True

        score = calculate_score_from_session(sample_session)

        # The actual trajectory matches the optimal
        assert score.trajectory_score > 0.8


class TestGrading:
    """Test grade assignment."""

    def test_grade_a(self) -> None:
        """Test A grade for score >= 0.90."""
        from rhoai_mcp.evaluation.scoring import _score_to_grade

        assert _score_to_grade(0.95) == "A"
        assert _score_to_grade(0.90) == "A"

    def test_grade_b(self) -> None:
        """Test B grade for score >= 0.80."""
        from rhoai_mcp.evaluation.scoring import _score_to_grade

        assert _score_to_grade(0.85) == "B"
        assert _score_to_grade(0.80) == "B"

    def test_grade_c(self) -> None:
        """Test C grade for score >= 0.70."""
        from rhoai_mcp.evaluation.scoring import _score_to_grade

        assert _score_to_grade(0.75) == "C"
        assert _score_to_grade(0.70) == "C"

    def test_grade_d(self) -> None:
        """Test D grade for score >= 0.60."""
        from rhoai_mcp.evaluation.scoring import _score_to_grade

        assert _score_to_grade(0.65) == "D"
        assert _score_to_grade(0.60) == "D"

    def test_grade_f(self) -> None:
        """Test F grade for score < 0.60."""
        from rhoai_mcp.evaluation.scoring import _score_to_grade

        assert _score_to_grade(0.59) == "F"
        assert _score_to_grade(0.0) == "F"


class TestDefaultWeights:
    """Test default weights configuration."""

    def test_weights_sum_to_one(self) -> None:
        """Test that default weights sum to 1.0."""
        total = sum(DEFAULT_WEIGHTS.values())
        assert total == pytest.approx(1.0)

    def test_all_dimensions_have_weights(self) -> None:
        """Test all six dimensions have weights."""
        expected_keys = {
            "stability",
            "performance",
            "tool_selection",
            "success_rate",
            "parameter_precision",
            "trajectory",
        }
        assert set(DEFAULT_WEIGHTS.keys()) == expected_keys
