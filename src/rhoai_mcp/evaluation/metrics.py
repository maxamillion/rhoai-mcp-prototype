"""Evaluation metrics for the evaluation harness.

This module defines the metrics models used for tracking
performance across six evaluation dimensions.
"""

from __future__ import annotations

import statistics
from typing import Any

from pydantic import BaseModel, Field

from rhoai_mcp.evaluation.models import (
    EvaluationSession,
    ExpectedToolSequence,
    ParameterSpec,
    ToolCall,
    TrajectorySpec,
)


class StabilityMetrics(BaseModel):
    """Measures whether tools work reliably across repeated invocations.

    Tracks success rates, error distributions, and consistency
    of repeated calls.
    """

    total_calls: int = Field(..., description="Total number of tool calls")
    successful_calls: int = Field(..., description="Number of successful calls")
    stability_score: float = Field(
        ..., description="Percentage of calls that complete without exceptions (0.0-1.0)"
    )
    error_types: dict[str, int] = Field(
        default_factory=dict, description="Distribution of error types encountered"
    )
    repeated_call_consistency: float = Field(
        1.0, description="Percentage of repeated calls with consistent results (0.0-1.0)"
    )

    @classmethod
    def from_tool_calls(cls, calls: list[ToolCall]) -> StabilityMetrics:
        """Calculate stability metrics from a list of tool calls."""
        if not calls:
            return cls(
                total_calls=0,
                successful_calls=0,
                stability_score=1.0,
            )

        total = len(calls)
        successful = sum(1 for c in calls if c.success)
        stability_score = successful / total if total > 0 else 1.0

        # Count error types
        error_types: dict[str, int] = {}
        for call in calls:
            if call.error:
                # Extract error type from error message
                error_type = _extract_error_type(call.error)
                error_types[error_type] = error_types.get(error_type, 0) + 1

        # Check consistency of repeated calls (same tool + same args)
        repeated_call_consistency = _calculate_repeated_call_consistency(calls)

        return cls(
            total_calls=total,
            successful_calls=successful,
            stability_score=stability_score,
            error_types=error_types,
            repeated_call_consistency=repeated_call_consistency,
        )


class PerformanceMetrics(BaseModel):
    """Measures latency and throughput characteristics.

    Tracks timing statistics across tool calls including
    percentiles and throughput.
    """

    latency_p50_ms: float = Field(..., description="50th percentile latency in milliseconds")
    latency_p95_ms: float = Field(..., description="95th percentile latency in milliseconds")
    latency_p99_ms: float = Field(..., description="99th percentile latency in milliseconds")
    avg_duration_ms: float = Field(..., description="Mean execution time in milliseconds")
    max_duration_ms: float = Field(..., description="Maximum execution time in milliseconds")
    min_duration_ms: float = Field(..., description="Minimum execution time in milliseconds")
    total_duration_ms: float = Field(..., description="Total execution time in milliseconds")
    calls_per_second: float = Field(
        ..., description="Throughput (tool_count / session_duration_seconds)"
    )

    @classmethod
    def from_tool_calls(
        cls, calls: list[ToolCall], session_duration_seconds: float
    ) -> PerformanceMetrics:
        """Calculate performance metrics from a list of tool calls."""
        if not calls:
            return cls(
                latency_p50_ms=0.0,
                latency_p95_ms=0.0,
                latency_p99_ms=0.0,
                avg_duration_ms=0.0,
                max_duration_ms=0.0,
                min_duration_ms=0.0,
                total_duration_ms=0.0,
                calls_per_second=0.0,
            )

        durations = [c.duration_ms for c in calls]
        sorted_durations = sorted(durations)

        # Calculate percentiles
        n = len(sorted_durations)
        p50_idx = int(n * 0.50)
        p95_idx = min(int(n * 0.95), n - 1)
        p99_idx = min(int(n * 0.99), n - 1)

        calls_per_second = (
            len(calls) / session_duration_seconds if session_duration_seconds > 0 else 0.0
        )

        return cls(
            latency_p50_ms=sorted_durations[p50_idx],
            latency_p95_ms=sorted_durations[p95_idx],
            latency_p99_ms=sorted_durations[p99_idx],
            avg_duration_ms=statistics.mean(durations),
            max_duration_ms=max(durations),
            min_duration_ms=min(durations),
            total_duration_ms=sum(durations),
            calls_per_second=calls_per_second,
        )


class ToolSelectionMetrics(BaseModel):
    """Measures whether the agent chose appropriate tools.

    Tracks coverage of required tools, violations of forbidden
    tools, and order compliance.
    """

    required_tools_called: int = Field(..., description="Number of required tools that were called")
    required_tools_total: int = Field(..., description="Total number of required tools")
    required_coverage: float = Field(
        ..., description="Percentage of required tools called (0.0-1.0)"
    )
    forbidden_tool_violations: list[str] = Field(
        default_factory=list, description="Forbidden tools that were called"
    )
    order_violations: int = Field(0, description="Number of out-of-order calls")
    redundant_calls: int = Field(0, description="Calls not in required or optional sets")

    @classmethod
    def from_session(
        cls, session: EvaluationSession, expected: ExpectedToolSequence | None = None
    ) -> ToolSelectionMetrics:
        """Calculate tool selection metrics from a session."""
        if expected is None:
            expected = session.expected_sequence

        if expected is None:
            return cls(
                required_tools_called=0,
                required_tools_total=0,
                required_coverage=1.0,
            )

        tool_sequence = session.get_tool_sequence()
        tools_called = set(tool_sequence)

        # Calculate required coverage
        required_set = set(expected.required_tools)
        called_required = tools_called.intersection(required_set)
        required_coverage = len(called_required) / len(required_set) if required_set else 1.0

        # Check forbidden violations
        forbidden_set = set(expected.forbidden_tools)
        violations = list(tools_called.intersection(forbidden_set))

        # Check order violations
        order_violations = 0
        if expected.expected_order:
            order_violations = _count_order_violations(tool_sequence, expected.expected_order)

        # Count redundant calls
        allowed_set = required_set.union(set(expected.optional_tools))
        redundant = sum(1 for t in tool_sequence if t not in allowed_set)

        return cls(
            required_tools_called=len(called_required),
            required_tools_total=len(required_set),
            required_coverage=required_coverage,
            forbidden_tool_violations=violations,
            order_violations=order_violations,
            redundant_calls=redundant,
        )


class SuccessErrorMetrics(BaseModel):
    """Measures overall reliability of tool execution.

    Tracks success/error counts, categorizes errors, and
    monitors retry behavior.
    """

    success_count: int = Field(..., description="Number of successful tool calls")
    error_count: int = Field(..., description="Number of failed tool calls")
    success_rate: float = Field(..., description="Percentage of successful calls (0.0-1.0)")
    error_categories: dict[str, int] = Field(
        default_factory=dict, description="Distribution of error categories"
    )
    retry_attempts: int = Field(0, description="Number of retries (same tool called after failure)")
    retry_success_rate: float = Field(0.0, description="Percentage of successful retries (0.0-1.0)")

    @classmethod
    def from_tool_calls(cls, calls: list[ToolCall]) -> SuccessErrorMetrics:
        """Calculate success/error metrics from a list of tool calls."""
        if not calls:
            return cls(
                success_count=0,
                error_count=0,
                success_rate=1.0,
            )

        success_count = sum(1 for c in calls if c.success)
        error_count = sum(1 for c in calls if not c.success)
        total = success_count + error_count
        success_rate = success_count / total if total > 0 else 1.0

        # Categorize errors
        error_categories: dict[str, int] = {}
        for call in calls:
            if call.error:
                category = _categorize_error(call.error)
                error_categories[category] = error_categories.get(category, 0) + 1

        # Detect retry attempts and success
        retry_attempts, retry_successes = _count_retries(calls)
        retry_success_rate = retry_successes / retry_attempts if retry_attempts > 0 else 0.0

        return cls(
            success_count=success_count,
            error_count=error_count,
            success_rate=success_rate,
            error_categories=error_categories,
            retry_attempts=retry_attempts,
            retry_success_rate=retry_success_rate,
        )


class ParameterPrecisionMetrics(BaseModel):
    """Measures whether tool parameters are correctly formed.

    Tracks schema compliance, type correctness, and value
    validity of tool parameters.
    """

    total_parameters: int = Field(..., description="Total number of parameters validated")
    valid_parameters: int = Field(..., description="Number of parameters that passed validation")
    precision_score: float = Field(..., description="Percentage of valid parameters (0.0-1.0)")
    type_errors: list[dict[str, Any]] = Field(
        default_factory=list, description="Parameters with wrong types"
    )
    missing_required: list[str] = Field(
        default_factory=list, description="Required parameters not provided"
    )
    out_of_range: list[dict[str, Any]] = Field(
        default_factory=list, description="Parameters outside expected range"
    )
    pattern_mismatches: list[dict[str, Any]] = Field(
        default_factory=list, description="Parameters not matching regex patterns"
    )

    @classmethod
    def from_session(cls, session: EvaluationSession) -> ParameterPrecisionMetrics:
        """Calculate parameter precision metrics from a session."""
        if not session.parameter_specs:
            return cls(
                total_parameters=0,
                valid_parameters=0,
                precision_score=1.0,
            )

        total_params = 0
        valid_params = 0
        type_errors: list[dict[str, Any]] = []
        missing_required: list[str] = []
        out_of_range: list[dict[str, Any]] = []
        pattern_mismatches: list[dict[str, Any]] = []

        for call in session.tool_calls:
            specs = session.parameter_specs.get(call.tool_name, [])
            for spec in specs:
                total_params += 1
                is_valid, error = _validate_parameter(spec, call.arguments.get(spec.name))
                if is_valid:
                    valid_params += 1
                elif error is not None:
                    if error["type"] == "type_error":
                        type_errors.append(error)
                    elif error["type"] == "missing":
                        missing_required.append(error["param"])
                    elif error["type"] == "range":
                        out_of_range.append(error)
                    elif error["type"] == "pattern":
                        pattern_mismatches.append(error)

        precision_score = valid_params / total_params if total_params > 0 else 1.0

        return cls(
            total_parameters=total_params,
            valid_parameters=valid_params,
            precision_score=precision_score,
            type_errors=type_errors,
            missing_required=missing_required,
            out_of_range=out_of_range,
            pattern_mismatches=pattern_mismatches,
        )


class TrajectoryMetrics(BaseModel):
    """Evaluates the entire sequence of tool calls as a trajectory.

    Tracks efficiency, goal achievement, and trajectory
    similarity to optimal paths.
    """

    total_steps: int = Field(..., description="Total number of steps taken")
    optimal_steps: int = Field(..., description="Number of steps in optimal trajectory")
    efficiency_score: float = Field(
        ..., description="Ratio of optimal to actual steps (capped at 1.0)"
    )
    goal_achieved: bool = Field(..., description="Whether the ultimate task was accomplished")
    trajectory_similarity: float = Field(
        ..., description="Similarity to optimal trajectory (0.0-1.0)"
    )
    checkpoints_hit: int = Field(0, description="Number of required checkpoints reached")
    checkpoints_total: int = Field(0, description="Total number of required checkpoints")
    checkpoint_coverage: float = Field(1.0, description="Percentage of checkpoints hit (0.0-1.0)")
    unnecessary_steps: list[str] = Field(
        default_factory=list, description="Steps not in any acceptable trajectory"
    )
    backtracking_count: int = Field(0, description="Times agent repeated earlier steps")

    @classmethod
    def from_session(
        cls, session: EvaluationSession, spec: TrajectorySpec | None = None
    ) -> TrajectoryMetrics:
        """Calculate trajectory metrics from a session."""
        if spec is None:
            spec = session.trajectory_spec

        tool_sequence = session.get_tool_sequence()
        total_steps = len(tool_sequence)

        if spec is None:
            return cls(
                total_steps=total_steps,
                optimal_steps=total_steps,
                efficiency_score=1.0,
                goal_achieved=session.task_completed,
                trajectory_similarity=1.0,
            )

        optimal_steps = len(spec.optimal_trajectory)
        efficiency_score = min(1.0, optimal_steps / total_steps) if total_steps > 0 else 1.0

        # Calculate trajectory similarity using Levenshtein-like metric
        trajectory_similarity = _calculate_trajectory_similarity(
            tool_sequence, spec.optimal_trajectory
        )

        # Check checkpoints
        checkpoints_total = len(spec.required_checkpoints)
        checkpoints_hit = sum(1 for cp in spec.required_checkpoints if cp in tool_sequence)
        checkpoint_coverage = checkpoints_hit / checkpoints_total if checkpoints_total > 0 else 1.0

        # Find unnecessary steps
        all_acceptable = set(spec.optimal_trajectory)
        for traj in spec.acceptable_trajectories:
            all_acceptable.update(traj)
        unnecessary = [t for t in tool_sequence if t not in all_acceptable]

        # Count backtracking
        backtracking_count = _count_backtracking(tool_sequence)

        return cls(
            total_steps=total_steps,
            optimal_steps=optimal_steps,
            efficiency_score=efficiency_score,
            goal_achieved=session.task_completed,
            trajectory_similarity=trajectory_similarity,
            checkpoints_hit=checkpoints_hit,
            checkpoints_total=checkpoints_total,
            checkpoint_coverage=checkpoint_coverage,
            unnecessary_steps=unnecessary,
            backtracking_count=backtracking_count,
        )


# Helper functions


def _extract_error_type(error: str) -> str:
    """Extract error type from error message."""
    error_lower = error.lower()
    if "timeout" in error_lower:
        return "timeout"
    if "connection" in error_lower or "network" in error_lower:
        return "network"
    if "auth" in error_lower or "permission" in error_lower or "forbidden" in error_lower:
        return "auth"
    if "not found" in error_lower or "404" in error_lower:
        return "not_found"
    if "validation" in error_lower or "invalid" in error_lower:
        return "validation"
    return "unknown"


def _categorize_error(error: str) -> str:
    """Categorize error into high-level categories."""
    return _extract_error_type(error)


def _calculate_repeated_call_consistency(calls: list[ToolCall]) -> float:
    """Calculate consistency of repeated calls with same tool and args."""
    # Group calls by (tool_name, args_hash)
    import json
    from hashlib import md5

    call_groups: dict[str, list[ToolCall]] = {}
    for call in calls:
        key = f"{call.tool_name}:{md5(json.dumps(call.arguments, sort_keys=True).encode()).hexdigest()}"
        if key not in call_groups:
            call_groups[key] = []
        call_groups[key].append(call)

    # Only consider groups with more than one call
    repeated_groups = [g for g in call_groups.values() if len(g) > 1]
    if not repeated_groups:
        return 1.0

    consistent_count = 0
    total_comparisons = 0

    for group in repeated_groups:
        # Check if all calls in the group have the same success status
        first_success = group[0].success
        for call in group[1:]:
            total_comparisons += 1
            if call.success == first_success:
                consistent_count += 1

    return consistent_count / total_comparisons if total_comparisons > 0 else 1.0


def _count_order_violations(actual: list[str], expected: list[str]) -> int:
    """Count how many tools are out of order compared to expected."""
    if not expected:
        return 0

    # Find positions of expected tools in actual sequence
    violations = 0
    expected_idx = 0

    for tool in actual:
        if expected_idx < len(expected) and tool == expected[expected_idx]:
            expected_idx += 1
        elif tool in expected:
            # Tool found but out of order
            violations += 1

    return violations


def _count_retries(calls: list[ToolCall]) -> tuple[int, int]:
    """Count retry attempts and successes.

    A retry is when the same tool is called again after a failure.
    """
    retries = 0
    successes = 0

    for i in range(1, len(calls)):
        prev_call = calls[i - 1]
        curr_call = calls[i]

        if not prev_call.success and curr_call.tool_name == prev_call.tool_name:
            retries += 1
            if curr_call.success:
                successes += 1

    return retries, successes


def _validate_parameter(spec: ParameterSpec, value: Any) -> tuple[bool, dict[str, Any] | None]:
    """Validate a parameter value against its spec."""
    import re

    # Check required
    if value is None:
        if spec.required:
            return False, {"type": "missing", "param": spec.name}
        return True, None

    # Check type
    type_map: dict[str, type | tuple[type, ...]] = {
        "string": str,
        "int": int,
        "float": (int, float),
        "bool": bool,
        "list": list,
        "dict": dict,
    }
    expected_type = type_map.get(spec.expected_type)
    if expected_type is not None and not isinstance(value, expected_type):
        return False, {
            "type": "type_error",
            "param": spec.name,
            "expected": spec.expected_type,
            "actual": type(value).__name__,
        }

    # Check pattern
    if spec.pattern and isinstance(value, str) and not re.match(spec.pattern, value):
        return False, {
            "type": "pattern",
            "param": spec.name,
            "pattern": spec.pattern,
            "value": value,
        }

    # Check range
    if isinstance(value, (int, float)):
        if spec.min_value is not None and value < spec.min_value:
            return False, {
                "type": "range",
                "param": spec.name,
                "min": spec.min_value,
                "value": value,
            }
        if spec.max_value is not None and value > spec.max_value:
            return False, {
                "type": "range",
                "param": spec.name,
                "max": spec.max_value,
                "value": value,
            }

    # Check allowed values
    if spec.allowed_values and value not in spec.allowed_values:
        return False, {
            "type": "allowed_values",
            "param": spec.name,
            "allowed": spec.allowed_values,
            "value": value,
        }

    return True, None


def _calculate_trajectory_similarity(actual: list[str], optimal: list[str]) -> float:
    """Calculate similarity between actual and optimal trajectories.

    Uses a normalized Levenshtein-like distance.
    """
    if not optimal:
        return 1.0
    if not actual:
        return 0.0

    # Simple edit distance
    m, n = len(actual), len(optimal)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if actual[i - 1] == optimal[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    distance = dp[m][n]
    max_len = max(m, n)
    similarity = 1.0 - (distance / max_len) if max_len > 0 else 1.0
    return max(0.0, similarity)


def _count_backtracking(sequence: list[str]) -> int:
    """Count how many times the agent repeated earlier steps."""
    seen: dict[str, int] = {}
    backtracking = 0

    for i, tool in enumerate(sequence):
        if tool in seen:
            # This tool was seen before, check if it's backtracking
            # (i.e., appearing again after other tools were called)
            last_seen = seen[tool]
            if i > last_seen + 1:
                backtracking += 1
        seen[tool] = i

    return backtracking
