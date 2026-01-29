"""MCP Tools for evaluation operations.

This module provides MCP tools for agents to manage evaluation
sessions and retrieve evaluation metrics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

from rhoai_mcp.evaluation.models import SessionStatus
from rhoai_mcp.evaluation.scoring import calculate_score_from_session
from rhoai_mcp.evaluation.session_manager import EvaluationSessionManager
from rhoai_mcp.evaluation.validation import ResultValidator

if TYPE_CHECKING:
    from rhoai_mcp.server import RHOAIServer


def register_tools(
    mcp: FastMCP,
    server: RHOAIServer,  # noqa: ARG001
    session_manager: EvaluationSessionManager,
    validator: ResultValidator,
) -> None:
    """Register evaluation tools with the MCP server.

    Args:
        mcp: The FastMCP server instance.
        server: The RHOAI server instance.
        session_manager: The evaluation session manager.
        validator: The result validator.
    """

    @mcp.tool()
    def eval_start_session(
        name: str,
        task_definition: str,
        expected_outcome: str,
    ) -> dict[str, Any]:
        """Start a new evaluation session.

        Begins tracking tool calls and metrics for evaluation purposes.
        Only one session can be active at a time.

        Args:
            name: Human-readable name for this evaluation session.
            task_definition: Description of the task being evaluated.
            expected_outcome: Description of what success looks like.

        Returns:
            Session information including the session ID.
        """
        try:
            session = session_manager.start_session(
                name=name,
                task_definition=task_definition,
                expected_outcome=expected_outcome,
            )
            return {
                "session_id": session.session_id,
                "name": session.name,
                "status": session.status.value,
                "started": session.start_time.isoformat(),
                "message": "Evaluation session started. Tool calls are now being tracked.",
            }
        except ValueError as e:
            return {"error": str(e)}

    @mcp.tool()
    def eval_end_session(
        session_id: str | None = None,
        task_completed: bool = False,
        notes: str = "",
    ) -> dict[str, Any]:
        """End an evaluation session.

        Stops tracking and finalizes the session. After ending,
        you can retrieve the evaluation report.

        Args:
            session_id: Session ID to end (defaults to active session).
            task_completed: Whether the task was successfully completed.
            notes: Optional notes about the session outcome.

        Returns:
            Session summary with basic metrics.
        """
        try:
            session = session_manager.end_session(
                session_id=session_id,
                task_completed=task_completed,
                notes=notes,
            )
            return {
                "session_id": session.session_id,
                "name": session.name,
                "status": session.status.value,
                "task_completed": session.task_completed,
                "duration_seconds": session.duration_seconds(),
                "tool_count": session.tool_count(),
                "success_count": session.success_count(),
                "error_count": session.error_count(),
                "message": "Evaluation session ended. Use eval_get_report for detailed metrics.",
            }
        except ValueError as e:
            return {"error": str(e)}

    @mcp.tool()
    def eval_cancel_session(session_id: str | None = None) -> dict[str, Any]:
        """Cancel an evaluation session.

        Cancels tracking without generating a final report.

        Args:
            session_id: Session ID to cancel (defaults to active session).

        Returns:
            Confirmation of cancellation.
        """
        try:
            session = session_manager.cancel_session(session_id)
            return {
                "session_id": session.session_id,
                "status": session.status.value,
                "message": "Evaluation session cancelled.",
            }
        except ValueError as e:
            return {"error": str(e)}

    @mcp.tool()
    def eval_get_report(session_id: str | None = None) -> dict[str, Any]:
        """Get the evaluation report for a session.

        Returns detailed metrics and analysis for the specified
        or active session.

        Args:
            session_id: Session ID to get report for (defaults to active/latest).

        Returns:
            Detailed evaluation report with metrics.
        """
        try:
            report = session_manager.generate_report(
                session_id=session_id,
                validator=validator,
            )
            return report.model_dump()
        except ValueError as e:
            return {"error": str(e)}

    @mcp.tool()
    def eval_get_composite_score(
        session_id: str | None = None,
        weights: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Get composite evaluation score for a session.

        Calculates weighted composite score across all six
        evaluation dimensions and returns a letter grade.

        Args:
            session_id: Session ID to score (defaults to active session).
            weights: Optional custom weights for dimensions.
                     Keys: stability, performance, tool_selection,
                           success_rate, parameter_precision, trajectory
                     Values should sum to 1.0.

        Returns:
            Composite score with individual dimension scores and grade.
        """
        sid = session_id or session_manager.active_session_id
        if not sid:
            return {"error": "No session specified"}

        session = session_manager.get_session(sid)
        if not session:
            return {"error": f"Session '{sid}' not found"}

        score = calculate_score_from_session(session, weights=weights)
        return score.model_dump()

    @mcp.tool()
    def eval_list_sessions(
        status: str | None = None,
    ) -> dict[str, Any]:
        """List all evaluation sessions.

        Returns summary information for all sessions, optionally
        filtered by status.

        Args:
            status: Filter by status: "active", "completed", or "cancelled".

        Returns:
            List of session summaries.
        """
        status_filter = None
        if status:
            try:
                status_filter = SessionStatus(status)
            except ValueError:
                return {"error": f"Invalid status: {status}"}

        sessions = session_manager.list_sessions(status=status_filter)
        return {
            "count": len(sessions),
            "sessions": [
                {
                    "session_id": s.session_id,
                    "name": s.name,
                    "status": s.status.value,
                    "task_definition": s.task_definition[:100] + "..."
                    if len(s.task_definition) > 100
                    else s.task_definition,
                    "tool_count": s.tool_count(),
                    "duration_seconds": s.duration_seconds(),
                }
                for s in sessions
            ],
        }

    @mcp.tool()
    def eval_get_session_status() -> dict[str, Any]:
        """Get the current evaluation session status.

        Returns information about the currently active session,
        or indicates if no session is active.

        Returns:
            Current session status and metrics.
        """
        return session_manager.get_current_session_status()

    @mcp.tool()
    def eval_add_expected_result(
        tool_name: str,
        required_fields: list[str] | None = None,
        field_values: dict[str, Any] | None = None,
        field_patterns: dict[str, str] | None = None,
        custom_validator: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Add an expected result for validation.

        Defines what a tool's result should look like. When the
        tool is called, its result will be validated against
        this expectation.

        Args:
            tool_name: Name of the tool this expectation applies to.
            required_fields: Field names that must exist in the result.
                            Supports dot notation for nested fields.
            field_values: Exact values that fields must have.
            field_patterns: Regex patterns that field values must match.
            custom_validator: Name of registered custom validator.
                            Built-in: "not_empty", "is_dict", "is_list",
                                      "no_error", "success_field"
            session_id: Session to add to (defaults to active session).

        Returns:
            Confirmation with the expected result specification.
        """
        try:
            expected = session_manager.add_expected_result(
                tool_name=tool_name,
                required_fields=required_fields,
                field_values=field_values,
                field_patterns=field_patterns,
                custom_validator=custom_validator,
                session_id=session_id,
            )
            return {
                "added": True,
                "expected_result": expected.model_dump(),
            }
        except ValueError as e:
            return {"error": str(e)}

    @mcp.tool()
    def eval_set_expected_trajectory(
        required_tools: list[str] | None = None,
        optional_tools: list[str] | None = None,
        forbidden_tools: list[str] | None = None,
        expected_order: list[str] | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Set the expected tool sequence for trajectory analysis.

        Defines which tools should be called (required), may be
        called (optional), or should not be called (forbidden)
        during the evaluation.

        Args:
            required_tools: Tools that MUST be called.
            optional_tools: Tools that MAY be called without penalty.
            forbidden_tools: Tools that should NOT be called.
            expected_order: Expected order of tool calls (if order matters).
            session_id: Session to update (defaults to active session).

        Returns:
            Confirmation with the trajectory specification.
        """
        try:
            sequence = session_manager.set_expected_trajectory(
                required_tools=required_tools,
                optional_tools=optional_tools,
                forbidden_tools=forbidden_tools,
                expected_order=expected_order,
                session_id=session_id,
            )
            return {
                "set": True,
                "expected_sequence": sequence.model_dump(),
            }
        except ValueError as e:
            return {"error": str(e)}

    @mcp.tool()
    def eval_set_trajectory_spec(
        goal_description: str,
        optimal_trajectory: list[str] | None = None,
        acceptable_trajectories: list[list[str]] | None = None,
        max_steps: int = 100,
        required_checkpoints: list[str] | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Set detailed trajectory specification for evaluation.

        Provides comprehensive trajectory expectations including
        optimal path, acceptable alternatives, and checkpoints.

        Args:
            goal_description: Description of the goal to achieve.
            optimal_trajectory: The optimal sequence of tool calls.
            acceptable_trajectories: Alternative valid tool sequences.
            max_steps: Maximum acceptable number of steps.
            required_checkpoints: Tools that must be called at some point.
            session_id: Session to update (defaults to active session).

        Returns:
            Confirmation with the trajectory specification.
        """
        try:
            spec = session_manager.set_trajectory_spec(
                goal_description=goal_description,
                optimal_trajectory=optimal_trajectory,
                acceptable_trajectories=acceptable_trajectories,
                max_steps=max_steps,
                required_checkpoints=required_checkpoints,
                session_id=session_id,
            )
            return {
                "set": True,
                "trajectory_spec": spec.model_dump(),
            }
        except ValueError as e:
            return {"error": str(e)}

    @mcp.tool()
    def eval_set_parameter_specs(
        tool_name: str,
        parameters: list[dict[str, Any]],
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Set parameter specifications for a tool.

        Defines expected parameters for precision analysis.
        When the tool is called, its parameters will be validated.

        Args:
            tool_name: Name of the tool these specs apply to.
            parameters: List of parameter specifications. Each spec can have:
                - name: Parameter name (required)
                - required: Whether parameter is required (default: true)
                - expected_type: Type ("string", "int", "float", "bool", "list", "dict")
                - pattern: Regex pattern for strings
                - min_value: Minimum for numbers
                - max_value: Maximum for numbers
                - allowed_values: List of allowed values
            session_id: Session to update (defaults to active session).

        Returns:
            Confirmation with the parameter specifications.
        """
        try:
            specs = session_manager.set_parameter_specs(
                tool_name=tool_name,
                parameters=parameters,
                session_id=session_id,
            )
            return {
                "set": True,
                "tool_name": tool_name,
                "parameter_count": len(specs),
                "parameters": [s.model_dump() for s in specs],
            }
        except ValueError as e:
            return {"error": str(e)}

    @mcp.tool()
    def eval_validate_session_results(session_id: str | None = None) -> dict[str, Any]:
        """Run validation on all tool calls in a session.

        Validates each tool call's result against any expected
        results that have been defined for that tool.

        Args:
            session_id: Session to validate (defaults to active session).

        Returns:
            Validation results for all tool calls.
        """
        sid = session_id or session_manager.active_session_id
        if not sid:
            return {"error": "No session specified"}

        session = session_manager.get_session(sid)
        if not session:
            return {"error": f"Session '{sid}' not found"}

        results = []
        for call in session.tool_calls:
            # Find matching expected result
            expected = next(
                (er for er in session.expected_results if er.tool_name == call.tool_name),
                None,
            )
            if expected:
                validation = validator.validate(call.result, expected)
                results.append(
                    {
                        "tool_name": call.tool_name,
                        "call_id": call.call_id,
                        "passed": validation.passed,
                        "failures": validation.failures,
                    }
                )

        passed_count = sum(1 for r in results if r["passed"])
        return {
            "session_id": sid,
            "validated_count": len(results),
            "passed_count": passed_count,
            "failed_count": len(results) - passed_count,
            "all_passed": passed_count == len(results),
            "results": results,
        }

    @mcp.tool()
    def eval_get_stability_metrics(session_id: str | None = None) -> dict[str, Any]:
        """Get stability metrics for a session.

        Returns metrics about tool reliability including success
        rate, error distribution, and call consistency.

        Args:
            session_id: Session to analyze (defaults to active session).

        Returns:
            Stability metrics.
        """
        from rhoai_mcp.evaluation.metrics import StabilityMetrics

        sid = session_id or session_manager.active_session_id
        if not sid:
            return {"error": "No session specified"}

        session = session_manager.get_session(sid)
        if not session:
            return {"error": f"Session '{sid}' not found"}

        metrics = StabilityMetrics.from_tool_calls(session.tool_calls)
        return {"session_id": sid, "metrics": metrics.model_dump()}

    @mcp.tool()
    def eval_get_performance_metrics(session_id: str | None = None) -> dict[str, Any]:
        """Get performance metrics for a session.

        Returns latency statistics and throughput information
        for tool calls in the session.

        Args:
            session_id: Session to analyze (defaults to active session).

        Returns:
            Performance metrics including latency percentiles.
        """
        from rhoai_mcp.evaluation.metrics import PerformanceMetrics

        sid = session_id or session_manager.active_session_id
        if not sid:
            return {"error": "No session specified"}

        session = session_manager.get_session(sid)
        if not session:
            return {"error": f"Session '{sid}' not found"}

        metrics = PerformanceMetrics.from_tool_calls(session.tool_calls, session.duration_seconds())
        return {"session_id": sid, "metrics": metrics.model_dump()}

    @mcp.tool()
    def eval_get_trajectory_analysis(session_id: str | None = None) -> dict[str, Any]:
        """Get trajectory analysis for a session.

        Evaluates the sequence of tool calls as a trajectory
        toward the goal, including efficiency and similarity
        to optimal paths.

        Args:
            session_id: Session to analyze (defaults to active session).

        Returns:
            Trajectory metrics and analysis.
        """
        from rhoai_mcp.evaluation.metrics import TrajectoryMetrics

        sid = session_id or session_manager.active_session_id
        if not sid:
            return {"error": "No session specified"}

        session = session_manager.get_session(sid)
        if not session:
            return {"error": f"Session '{sid}' not found"}

        metrics = TrajectoryMetrics.from_session(session)
        return {"session_id": sid, "metrics": metrics.model_dump()}
