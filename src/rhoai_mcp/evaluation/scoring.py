"""Composite scoring for the evaluation harness.

This module provides functions to calculate composite evaluation
scores from individual metrics.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from rhoai_mcp.evaluation.metrics import (
    ParameterPrecisionMetrics,
    PerformanceMetrics,
    StabilityMetrics,
    SuccessErrorMetrics,
    ToolSelectionMetrics,
    TrajectoryMetrics,
)
from rhoai_mcp.evaluation.models import EvaluationSession


class CompositeEvaluationScore(BaseModel):
    """Overall evaluation combining all criteria.

    Provides individual dimension scores and a weighted composite
    score with a letter grade.
    """

    # Individual scores (0.0 - 1.0)
    stability_score: float = Field(..., description="Stability dimension score")
    performance_score: float = Field(..., description="Performance dimension score")
    tool_selection_score: float = Field(..., description="Tool selection dimension score")
    success_rate_score: float = Field(..., description="Success rate dimension score")
    parameter_precision_score: float = Field(..., description="Parameter precision dimension score")
    trajectory_score: float = Field(..., description="Trajectory dimension score")

    # Weighted composite
    overall_score: float = Field(..., description="Weighted composite score (0.0 - 1.0)")

    # Grade (A/B/C/D/F)
    grade: str = Field(..., description="Letter grade based on overall score")

    # Individual metrics for reference
    stability_metrics: StabilityMetrics | None = Field(
        None, description="Stability metrics details"
    )
    performance_metrics: PerformanceMetrics | None = Field(
        None, description="Performance metrics details"
    )
    tool_selection_metrics: ToolSelectionMetrics | None = Field(
        None, description="Tool selection metrics details"
    )
    success_error_metrics: SuccessErrorMetrics | None = Field(
        None, description="Success/error metrics details"
    )
    parameter_precision_metrics: ParameterPrecisionMetrics | None = Field(
        None, description="Parameter precision metrics details"
    )
    trajectory_metrics: TrajectoryMetrics | None = Field(
        None, description="Trajectory metrics details"
    )

    # Weights used
    weights: dict[str, float] = Field(default_factory=dict, description="Weights used for scoring")

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Override model_dump to provide structured output format."""
        # Ignore kwargs - we always return our custom structure
        _ = kwargs
        return {
            "scores": {
                "stability": self.stability_score,
                "performance": self.performance_score,
                "tool_selection": self.tool_selection_score,
                "success_rate": self.success_rate_score,
                "parameter_precision": self.parameter_precision_score,
                "trajectory": self.trajectory_score,
            },
            "overall_score": self.overall_score,
            "grade": self.grade,
            "weights": self.weights,
            "metrics": {
                "stability": (
                    self.stability_metrics.model_dump() if self.stability_metrics else None
                ),
                "performance": (
                    self.performance_metrics.model_dump() if self.performance_metrics else None
                ),
                "tool_selection": (
                    self.tool_selection_metrics.model_dump()
                    if self.tool_selection_metrics
                    else None
                ),
                "success_error": (
                    self.success_error_metrics.model_dump() if self.success_error_metrics else None
                ),
                "parameter_precision": (
                    self.parameter_precision_metrics.model_dump()
                    if self.parameter_precision_metrics
                    else None
                ),
                "trajectory": (
                    self.trajectory_metrics.model_dump() if self.trajectory_metrics else None
                ),
            },
        }


# Default weights for each evaluation dimension
DEFAULT_WEIGHTS: dict[str, float] = {
    "stability": 0.15,
    "performance": 0.10,
    "tool_selection": 0.20,
    "success_rate": 0.15,
    "parameter_precision": 0.15,
    "trajectory": 0.25,
}


def calculate_composite_score(
    stability: StabilityMetrics,
    performance: PerformanceMetrics,
    tool_selection: ToolSelectionMetrics,
    success_error: SuccessErrorMetrics,
    parameter_precision: ParameterPrecisionMetrics,
    trajectory: TrajectoryMetrics,
    weights: dict[str, float] | None = None,
    baseline_latency_p95_ms: float = 1000.0,
) -> CompositeEvaluationScore:
    """Calculate weighted composite evaluation score.

    Args:
        stability: Stability metrics.
        performance: Performance metrics.
        tool_selection: Tool selection metrics.
        success_error: Success/error metrics.
        parameter_precision: Parameter precision metrics.
        trajectory: Trajectory metrics.
        weights: Optional custom weights (must sum to 1.0).
        baseline_latency_p95_ms: Baseline p95 latency for performance scoring.

    Returns:
        CompositeEvaluationScore with all dimension scores and grade.
    """
    weights = weights or DEFAULT_WEIGHTS.copy()

    # Calculate individual dimension scores (all 0.0 - 1.0)
    stability_score = stability.stability_score
    performance_score = _calculate_performance_score(performance, baseline_latency_p95_ms)
    tool_selection_score = _calculate_tool_selection_score(tool_selection)
    success_rate_score = success_error.success_rate
    precision_score = parameter_precision.precision_score
    trajectory_score = _calculate_trajectory_score(trajectory)

    # Calculate weighted composite
    overall_score = (
        weights.get("stability", 0.15) * stability_score
        + weights.get("performance", 0.10) * performance_score
        + weights.get("tool_selection", 0.20) * tool_selection_score
        + weights.get("success_rate", 0.15) * success_rate_score
        + weights.get("parameter_precision", 0.15) * precision_score
        + weights.get("trajectory", 0.25) * trajectory_score
    )

    # Determine grade
    grade = _score_to_grade(overall_score)

    return CompositeEvaluationScore(
        stability_score=stability_score,
        performance_score=performance_score,
        tool_selection_score=tool_selection_score,
        success_rate_score=success_rate_score,
        parameter_precision_score=precision_score,
        trajectory_score=trajectory_score,
        overall_score=overall_score,
        grade=grade,
        stability_metrics=stability,
        performance_metrics=performance,
        tool_selection_metrics=tool_selection,
        success_error_metrics=success_error,
        parameter_precision_metrics=parameter_precision,
        trajectory_metrics=trajectory,
        weights=weights,
    )


def calculate_score_from_session(
    session: EvaluationSession,
    weights: dict[str, float] | None = None,
    baseline_latency_p95_ms: float = 1000.0,
) -> CompositeEvaluationScore:
    """Calculate composite score directly from a session.

    Args:
        session: The evaluation session to score.
        weights: Optional custom weights.
        baseline_latency_p95_ms: Baseline p95 latency for performance scoring.

    Returns:
        CompositeEvaluationScore for the session.
    """
    # Calculate all metrics from session
    stability = StabilityMetrics.from_tool_calls(session.tool_calls)
    performance = PerformanceMetrics.from_tool_calls(session.tool_calls, session.duration_seconds())
    tool_selection = ToolSelectionMetrics.from_session(session)
    success_error = SuccessErrorMetrics.from_tool_calls(session.tool_calls)
    parameter_precision = ParameterPrecisionMetrics.from_session(session)
    trajectory = TrajectoryMetrics.from_session(session)

    return calculate_composite_score(
        stability=stability,
        performance=performance,
        tool_selection=tool_selection,
        success_error=success_error,
        parameter_precision=parameter_precision,
        trajectory=trajectory,
        weights=weights,
        baseline_latency_p95_ms=baseline_latency_p95_ms,
    )


def _calculate_performance_score(metrics: PerformanceMetrics, baseline_p95_ms: float) -> float:
    """Calculate performance score based on latency vs baseline.

    Score is 1.0 if p95 latency is at or below baseline, decreasing
    as latency increases.
    """
    if metrics.latency_p95_ms <= 0:
        return 1.0
    if baseline_p95_ms <= 0:
        return 1.0

    # Score based on how close to baseline
    ratio = metrics.latency_p95_ms / baseline_p95_ms
    if ratio <= 1.0:
        return 1.0
    # Decay score as latency exceeds baseline
    # 2x baseline = 0.5, 3x baseline = 0.33, etc.
    return max(0.0, 1.0 / ratio)


def _calculate_tool_selection_score(metrics: ToolSelectionMetrics) -> float:
    """Calculate tool selection score from metrics.

    Considers required coverage, forbidden violations, and redundancy.
    """
    score = metrics.required_coverage

    # Penalize forbidden tool violations heavily
    if metrics.forbidden_tool_violations:
        violation_penalty = len(metrics.forbidden_tool_violations) * 0.2
        score -= violation_penalty

    # Minor penalty for order violations
    if metrics.order_violations > 0:
        score -= min(0.1, metrics.order_violations * 0.02)

    # Minor penalty for redundant calls
    if metrics.redundant_calls > 0:
        score -= min(0.1, metrics.redundant_calls * 0.01)

    return max(0.0, min(1.0, score))


def _calculate_trajectory_score(metrics: TrajectoryMetrics) -> float:
    """Calculate trajectory score from metrics.

    Combines efficiency, goal achievement, similarity, and checkpoints.
    """
    # Goal achievement is critical
    if not metrics.goal_achieved:
        # Still give partial credit based on progress
        return min(0.4, metrics.checkpoint_coverage * 0.4)

    # Weighted combination of trajectory factors
    score = (
        0.3 * metrics.efficiency_score
        + 0.3 * metrics.trajectory_similarity
        + 0.2 * metrics.checkpoint_coverage
        + 0.2  # Bonus for achieving goal
    )

    # Penalize unnecessary steps
    if metrics.unnecessary_steps:
        score -= min(0.1, len(metrics.unnecessary_steps) * 0.02)

    # Penalize backtracking
    if metrics.backtracking_count > 0:
        score -= min(0.1, metrics.backtracking_count * 0.02)

    return max(0.0, min(1.0, score))


def _score_to_grade(score: float) -> str:
    """Convert a 0.0-1.0 score to a letter grade."""
    if score >= 0.90:
        return "A"
    if score >= 0.80:
        return "B"
    if score >= 0.70:
        return "C"
    if score >= 0.60:
        return "D"
    return "F"
