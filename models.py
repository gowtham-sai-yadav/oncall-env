"""Pydantic models for OnCallEnv action and observation spaces."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


# ── Sub-models (data models, NOT agent actions) ───────────────────────────

class Alert(BaseModel):
    """A monitoring alert."""
    model_config = {"extra": "allow"}

    alert_id: str = Field(description="Unique alert identifier")
    severity: Literal["critical", "warning", "info"] = Field(description="Alert severity level")
    service: str = Field(description="Service that triggered the alert")
    message: str = Field(description="Alert message text")
    timestamp: str = Field(description="ISO 8601 timestamp")
    acknowledged: bool = Field(default=False, description="Whether the alert has been acknowledged")
    silenced: bool = Field(default=False, description="Whether the alert has been silenced")


class ServiceStatus(BaseModel):
    """Current status of a microservice."""
    model_config = {"extra": "allow"}

    name: str = Field(description="Service name")
    status: Literal["healthy", "degraded", "down", "unknown"] = Field(description="Current health status")
    latency_ms: float = Field(default=0.0, description="P99 latency in milliseconds")
    error_rate: float = Field(default=0.0, description="Error rate percentage")
    cpu_percent: float = Field(default=0.0, description="CPU utilization percentage")
    memory_percent: float = Field(default=0.0, description="Memory utilization percentage")
    version: str = Field(default="", description="Deployed version")


class LogEntry(BaseModel):
    """A single log line."""
    model_config = {"extra": "allow"}

    timestamp: str = Field(description="ISO 8601 timestamp")
    service: str = Field(description="Source service")
    level: Literal["DEBUG", "INFO", "WARN", "ERROR", "FATAL"] = Field(description="Log level")
    message: str = Field(description="Log message text")


class Deploy(BaseModel):
    """A recent deployment record."""
    model_config = {"extra": "allow"}

    service: str = Field(description="Deployed service name")
    version: str = Field(description="Deployed version")
    timestamp: str = Field(description="Deployment timestamp")
    deployer: str = Field(description="Who or what triggered the deploy")


class Event(BaseModel):
    """A timeline event (agent action or system event)."""
    model_config = {"extra": "allow"}

    timestamp: str = Field(description="ISO 8601 timestamp")
    event_type: str = Field(description="'agent_action' or 'system_event'")
    description: str = Field(description="Event description")


# ── Action ──────────────────────────────────────────────────────────────────

class OnCallAction(Action):
    """An action the on-call agent can take."""

    action_type: Literal[
        "query_logs",
        "check_metrics",
        "view_dependencies",
        "acknowledge_alert",
        "silence_alert",
        "restart_service",
        "scale_service",
        "rollback_deploy",
        "update_config",
        "set_severity",
        "write_summary",
        "escalate",
        "resolve_incident",
    ]
    params: dict[str, Any] = Field(default_factory=dict, description="Action parameters")


# ── Observation ─────────────────────────────────────────────────────────────

class OnCallObservation(Observation):
    """What the on-call agent sees after each step.

    Note: Internal environment state is stored as list[dict] and auto-coerced
    to typed models by Pydantic v2 at observation creation time. The grader
    operates on raw dicts and is NOT affected by these type annotations.
    """

    alerts: List[Alert] = Field(default_factory=list, description="Active monitoring alerts")
    services: List[ServiceStatus] = Field(default_factory=list, description="Current service statuses")
    recent_deployments: List[Deploy] = Field(default_factory=list, description="Recent deployment records")
    log_results: Optional[List[LogEntry]] = Field(default=None, description="Results from last query_logs action")
    metric_results: Optional[Dict[str, Any]] = Field(default=None, description="Results from last check_metrics action")
    dependency_graph: Optional[Dict[str, Any]] = Field(default=None, description="Results from last view_dependencies action")
    incident_timeline: List[Event] = Field(default_factory=list, description="Chronological event log")
    current_severity: Optional[str] = Field(default=None, description="Currently set incident severity")
    available_actions: List[str] = Field(default_factory=list, description="Valid action types")
    message: str = Field(default="", description="Human-readable feedback on last action")


# ── State ───────────────────────────────────────────────────────────────────

class OnCallState(State):
    """Internal environment state."""

    task_id: str = Field(default="", description="Current task identifier")
    scenario_id: str = Field(default="", description="Current scenario identifier")
    incident_resolved: bool = Field(default=False, description="Whether the incident has been resolved")
    actions_taken: List[Dict[str, Any]] = Field(default_factory=list, description="History of actions taken")
