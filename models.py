"""Pydantic models for OnCallEnv action and observation spaces."""

from __future__ import annotations

from typing import Any, Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


# ── Sub-models ──────────────────────────────────────────────────────────────

class Alert(Action):
    """A monitoring alert."""
    model_config = {"extra": "allow"}

    alert_id: str
    severity: Literal["critical", "warning", "info"]
    service: str
    message: str
    timestamp: str
    acknowledged: bool = False
    silenced: bool = False


class ServiceStatus(Action):
    """Current status of a microservice."""
    model_config = {"extra": "allow"}

    name: str
    status: Literal["healthy", "degraded", "down"]
    latency_ms: float = 0.0
    error_rate: float = 0.0
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    version: str = ""


class LogEntry(Action):
    """A single log line."""
    model_config = {"extra": "allow"}

    timestamp: str
    service: str
    level: Literal["DEBUG", "INFO", "WARN", "ERROR", "FATAL"]
    message: str


class Deploy(Action):
    """A recent deployment record."""
    model_config = {"extra": "allow"}

    service: str
    version: str
    timestamp: str
    deployer: str


class Event(Action):
    """A timeline event (agent action or system event)."""
    model_config = {"extra": "allow"}

    timestamp: str
    event_type: str  # "agent_action" | "system_event"
    description: str


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
    params: dict[str, Any] = Field(default_factory=dict)


# ── Observation ─────────────────────────────────────────────────────────────

class OnCallObservation(Observation):
    """What the on-call agent sees after each step."""

    alerts: list[dict] = Field(default_factory=list)
    services: list[dict] = Field(default_factory=list)
    recent_deployments: list[dict] = Field(default_factory=list)
    log_results: list[dict] | None = None
    metric_results: dict | None = None
    dependency_graph: dict | None = None
    incident_timeline: list[dict] = Field(default_factory=list)
    current_severity: str | None = None
    available_actions: list[str] = Field(default_factory=list)
    message: str = ""


# ── State ───────────────────────────────────────────────────────────────────

class OnCallState(State):
    """Internal environment state."""

    task_id: str = ""
    scenario_id: str = ""
    incident_resolved: bool = False
    actions_taken: list[dict] = Field(default_factory=list)
