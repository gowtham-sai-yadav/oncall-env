"""OnCallEnvironment -- core environment implementing reset/step/state."""

from __future__ import annotations

import copy
import random
import re
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

from oncall_env.models import OnCallAction, OnCallObservation, OnCallState
from oncall_env.server.scenario_loader import load_scenario_by_task
from oncall_env.server.graders import grade_episode
from oncall_env.server.rubric import OnCallRubric
from oncall_env.server.simulator import degrade_services, propagate_recovery


ALL_ACTIONS = [
    "query_logs", "check_metrics", "view_dependencies",
    "acknowledge_alert", "silence_alert",
    "restart_service", "scale_service", "rollback_deploy", "update_config",
    "set_severity", "write_summary", "escalate", "resolve_incident",
]

MAX_STEPS = 60
MAX_STRING_LEN = 5000


class OnCallEnvironment(Environment[OnCallAction, OnCallObservation, OnCallState]):
    """Simulates an on-call incident response scenario."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, **kwargs):
        self._rubric = OnCallRubric()
        self._rubric.set_env(self)
        super().__init__(rubric=self._rubric)
        self._state = OnCallState()
        self._scenario: dict[str, Any] = {}
        self._alerts: list[dict] = []
        self._services: list[dict] = []
        self._deployments: list[dict] = []
        self._timeline: list[dict] = []
        self._severity: str | None = None
        self._summary: str = ""
        self._escalated_to: str | None = None
        self._resolved: bool = False
        self._actions_taken: list[dict] = []
        self._investigated_services: set = set()
        self._service_alias_map: dict[str, str] = {}  # real_name -> anonymous label
        self._alias_to_real: dict[str, str] = {}  # anonymous label -> real_name

    # ── reset ───────────────────────────────────────────────────────────

    def reset(self, seed: int | None = None, episode_id: str | None = None, **kwargs) -> OnCallObservation:
        self._reset_rubric()

        task_id = kwargs.get("task_id", 1)
        scenario_idx = kwargs.get("scenario_idx", 0)

        self._scenario = load_scenario_by_task(task_id, scenario_idx)
        self._alerts = copy.deepcopy(self._scenario["initial_alerts"])
        self._services = copy.deepcopy(self._scenario["services"])
        self._deployments = copy.deepcopy(self._scenario.get("recent_deployments", []))

        # Shuffle lists so ordering doesn't leak the root cause
        rng = random.Random(seed if seed is not None else scenario_idx)
        rng.shuffle(self._alerts)
        rng.shuffle(self._services)
        self._timeline = []
        self._severity = None
        self._summary = ""
        self._escalated_to = None
        self._resolved = False
        self._actions_taken = []
        self._investigated_services = set()

        # Build anonymization map from ALL service names appearing in BOTH
        # the services list AND the alerts (some alerts reference services not
        # in the main services list, e.g., "monitoring")
        svc_names_from_services = {s["name"] for s in self._services}
        svc_names_from_alerts = {a["service"] for a in self._alerts}
        all_svc_names = list(svc_names_from_services | svc_names_from_alerts)
        rng.shuffle(all_svc_names)
        self._service_alias_map = {
            name: f"service-{chr(65 + i)}" for i, name in enumerate(all_svc_names)
        }
        self._alias_to_real = {v: k for k, v in self._service_alias_map.items()}

        eid = episode_id or str(uuid4())
        self._state = OnCallState(
            episode_id=eid,
            step_count=0,
            task_id=f"task{task_id}",
            scenario_id=self._scenario.get("incident_id", "unknown"),
        )

        self._add_event("system_event", "Incident opened. You are the on-call engineer.")

        # Anonymize the scenario description: replace real service names with aliases
        # and strip root cause type hints
        raw_desc = self._scenario.get("description", "New incident detected. Begin triage.")
        redacted_desc = self._redact_description(raw_desc)

        return self._make_observation(
            message=redacted_desc,
        )

    # ── alias resolution ─────────────────────────────────────────────

    def _resolve_service(self, name: str) -> str:
        """Resolve a service alias to its real name. Accepts both aliases and real names."""
        if not name:
            return name
        # Check if it's an alias (e.g., "service-B" -> "payment-api")
        if name in self._alias_to_real:
            return self._alias_to_real[name]
        # Already a real name, return as-is
        return name

    def _resolve_params(self, params: dict) -> dict:
        """Resolve any service aliases in action params to real service names.

        This ensures:
        - Handlers always work with real names (for looking up logs/metrics/deps)
        - _actions_taken records real names (for grader compatibility)
        - The model can use EITHER aliases or real names
        """
        resolved = dict(params)
        for key in ("service", "service_name"):
            if key in resolved and resolved[key]:
                resolved[key] = self._resolve_service(resolved[key])
        return resolved

    # ── step ────────────────────────────────────────────────────────────

    def step(self, action: OnCallAction, timeout_s: float | None = None, **kwargs) -> OnCallObservation:
        if self._resolved or self._state.step_count >= MAX_STEPS:
            reward = self._compute_final_reward()
            return self._make_observation(
                message="Incident already resolved or step limit reached.",
                done=True,
                reward=reward,
            )

        # Sanitize params: truncate extremely long strings
        sanitized_params = self._sanitize_params(action.params)
        # Resolve any service aliases to real names (so handlers and grader work correctly)
        sanitized_params = self._resolve_params(sanitized_params)

        self._state.step_count += 1
        self._actions_taken.append({"action_type": action.action_type, "params": sanitized_params, "step": self._state.step_count})
        self._state.actions_taken = self._actions_taken

        handler = getattr(self, f"_handle_{action.action_type}", None)
        if handler is None:
            return self._make_observation(
                message=f"Unknown action: '{action.action_type}'. Available actions: {', '.join(ALL_ACTIONS)}"
            )

        try:
            msg, extra = handler(sanitized_params)
        except Exception as e:
            return self._make_observation(
                message=f"Error executing {action.action_type}: {e}"
            )

        self._add_event("agent_action", f"{action.action_type}: {sanitized_params}")

        # Dynamic simulation: degrade unhealthy services each step
        deps = self._scenario.get("dependencies", {})
        degrade_services(self._services, deps)

        # Cascading recovery: if a remediation just fixed a service, propagate
        if action.action_type in ("restart_service", "scale_service", "rollback_deploy", "update_config"):
            target_svc = sanitized_params.get("service_name", "")
            svc_map = {s["name"]: s for s in self._services}
            if target_svc in svc_map and svc_map[target_svc]["status"] == "healthy":
                propagate_recovery(self._services, deps, target_svc)

        done = self._resolved or self._state.step_count >= MAX_STEPS
        reward = self._compute_final_reward() if done else self._compute_intermediate_reward()

        obs = self._make_observation(message=msg, done=done, reward=reward)
        # Merge any extra fields from handler
        for k, v in extra.items():
            if hasattr(obs, k):
                setattr(obs, k, v)

        # Accumulate step in rubric trajectory (for GRPO training compatibility)
        self._apply_rubric(action, obs)

        return obs

    # ── state property ──────────────────────────────────────────────────

    @property
    def state(self) -> OnCallState:
        return self._state

    # ── metadata ─────────────────────────────────────────────────────────

    def get_metadata(self):
        from openenv.core.env_server.types import EnvironmentMetadata
        return EnvironmentMetadata(
            name="OnCallEnv",
            description="Incident Response Command Center -- simulates production on-call engineering with alert triage, root cause diagnosis, remediation, and documentation across 4 difficulty levels.",
            version="0.2.0",
        )

    # ── Action Handlers ─────────────────────────────────────────────────

    def _handle_query_logs(self, params: dict) -> tuple[str, dict]:
        service = params.get("service", "")
        level = params.get("level", "")
        time_range = params.get("time_range", "")
        logs_db: dict = self._scenario.get("logs", {})

        if service:
            self._investigated_services.add(service)

        results = []
        for key, entries in logs_db.items():
            for entry in entries:
                match_svc = (not service) or entry.get("service", "").lower() == service.lower()
                match_lvl = (not level) or entry.get("level", "").upper() == level.upper()
                if match_svc and match_lvl:
                    results.append(entry)

        results = results[:50]  # cap results
        return (
            f"Found {len(results)} log entries for service='{service}' level='{level}'.",
            {"log_results": results},
        )

    def _handle_check_metrics(self, params: dict) -> tuple[str, dict]:
        service = params.get("service", "")
        metric_name = params.get("metric_name", "")
        metrics_db: dict = self._scenario.get("metrics", {})

        if service:
            self._investigated_services.add(service)

        key = f"{service}:{metric_name}" if service and metric_name else service or metric_name
        # Try exact match, then prefix match
        result = metrics_db.get(key)
        if result is None:
            for k, v in metrics_db.items():
                if service and service.lower() in k.lower():
                    if not metric_name or metric_name.lower() in k.lower():
                        result = v
                        break

        if result is None:
            # List available metrics for this service so the model can retry
            available = [
                k.split(":", 1)[1] if ":" in k else k
                for k in metrics_db.keys()
                if service and service.lower() in k.lower()
            ]
            if available:
                return (
                    f"No metric '{metric_name}' for service='{service}'. Available metrics: {', '.join(available)}",
                    {"metric_results": None},
                )
            return f"No metrics found for service='{service}'.", {"metric_results": None}
        return (
            f"Metrics for {service}/{metric_name}: returned {len(result) if isinstance(result, list) else 1} data points.",
            {"metric_results": {key: result}},
        )

    def _handle_view_dependencies(self, params: dict) -> tuple[str, dict]:
        service = params.get("service_name", "")
        deps: dict = self._scenario.get("dependencies", {})

        if service:
            self._investigated_services.add(service)
        svc_deps = deps.get(service)
        if svc_deps is None:
            return f"No dependency info for '{service}'.", {"dependency_graph": None}
        # Anonymize any internal service names in the dependency list
        # (external deps like databases keep their real names)
        anonymized_deps = []
        for dep in svc_deps:
            alias = self._service_alias_map.get(dep)
            if alias and dep not in self._investigated_services:
                anonymized_deps.append(alias)
            else:
                anonymized_deps.append(dep)
        return (
            f"Dependencies for {service}: {anonymized_deps}",
            {"dependency_graph": {service: anonymized_deps}},
        )

    def _handle_acknowledge_alert(self, params: dict) -> tuple[str, dict]:
        alert_id = params.get("alert_id", "")
        for alert in self._alerts:
            if alert["alert_id"] == alert_id:
                alert["acknowledged"] = True
                return f"Alert {alert_id} acknowledged.", {}
        return f"Alert {alert_id} not found.", {}

    def _handle_silence_alert(self, params: dict) -> tuple[str, dict]:
        alert_id = params.get("alert_id", "")
        for alert in self._alerts:
            if alert["alert_id"] == alert_id:
                alert["silenced"] = True
                return f"Alert {alert_id} silenced.", {}
        return f"Alert {alert_id} not found.", {}

    def _handle_restart_service(self, params: dict) -> tuple[str, dict]:
        svc_name = params.get("service_name", "")
        for svc in self._services:
            if svc["name"] == svc_name:
                remediation = self._scenario.get("valid_remediations", [])
                is_valid = any(
                    r.get("action") == "restart_service" and r.get("service") == svc_name
                    for r in remediation
                )
                if is_valid:
                    svc["status"] = "healthy"
                    svc["error_rate"] = 0.0
                    svc["latency_ms"] = max(svc.get("latency_ms", 50) * 0.3, 15)
                else:
                    # Restart doesn't fix the root cause but temporarily improves things
                    if svc["status"] == "down":
                        svc["status"] = "degraded"
                # Ambiguous response — model must verify via check_metrics/query_logs
                return f"Service {svc_name} restart initiated. Use check_metrics or query_logs to verify recovery.", {}
        return f"Service {svc_name} not found.", {}

    def _handle_scale_service(self, params: dict) -> tuple[str, dict]:
        svc_name = params.get("service_name", "")
        replicas = params.get("replicas", 1)
        for svc in self._services:
            if svc["name"] == svc_name:
                remediation = self._scenario.get("valid_remediations", [])
                is_valid = any(
                    r.get("action") == "scale_service" and r.get("service") == svc_name
                    for r in remediation
                )
                if is_valid:
                    svc["status"] = "healthy"
                    svc["error_rate"] = max(svc.get("error_rate", 0) * 0.2, 0)
                return f"Service {svc_name} scaled to {replicas} replicas. Use check_metrics to verify effect.", {}
        return f"Service {svc_name} not found.", {}

    def _handle_rollback_deploy(self, params: dict) -> tuple[str, dict]:
        svc_name = params.get("service_name", "")
        target_version = params.get("target_version", "")
        for svc in self._services:
            if svc["name"] == svc_name:
                remediation = self._scenario.get("valid_remediations", [])
                is_valid = any(
                    r.get("action") == "rollback_deploy" and r.get("service") == svc_name
                    for r in remediation
                )
                if is_valid:
                    svc["status"] = "healthy"
                    svc["error_rate"] = 0.0
                    svc["version"] = target_version
                return f"Service {svc_name} rollback to {target_version} initiated. Use check_metrics or query_logs to verify.", {}
        return f"Service {svc_name} not found.", {}

    def _handle_update_config(self, params: dict) -> tuple[str, dict]:
        svc_name = params.get("service_name", "")
        config_key = params.get("config_key", "")
        config_value = params.get("config_value", "")
        for svc in self._services:
            if svc["name"] == svc_name:
                remediation = self._scenario.get("valid_remediations", [])
                is_valid = any(
                    r.get("action") == "update_config"
                    and r.get("service") == svc_name
                    and r.get("config_key", "") == config_key
                    for r in remediation
                )
                if is_valid:
                    svc["status"] = "healthy"
                    svc["error_rate"] = 0.0
                return f"Config '{config_key}' updated to '{config_value}' on {svc_name}. Use check_metrics to verify effect.", {}
        return f"Service {svc_name} not found.", {}

    def _handle_set_severity(self, params: dict) -> tuple[str, dict]:
        level = params.get("level", "SEV3")
        self._severity = level
        return f"Incident severity set to {level}.", {}

    def _handle_write_summary(self, params: dict) -> tuple[str, dict]:
        text = params.get("text", "")
        if len(text) > MAX_STRING_LEN:
            text = text[:MAX_STRING_LEN]
        self._summary = text
        return "Incident summary recorded.", {}

    def _handle_escalate(self, params: dict) -> tuple[str, dict]:
        team = params.get("team", "")
        self._escalated_to = team
        return f"Escalated to {team}.", {}

    def _handle_resolve_incident(self, params: dict) -> tuple[str, dict]:
        note = params.get("resolution_note", "")
        self._resolved = True
        self._state.incident_resolved = True
        self._summary = self._summary or note
        return "Incident marked as resolved.", {}

    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _sanitize_params(params: dict) -> dict:
        """Truncate overly long string values in params to prevent memory issues."""
        sanitized = {}
        for k, v in params.items():
            if isinstance(v, str) and len(v) > MAX_STRING_LEN:
                sanitized[k] = v[:MAX_STRING_LEN]
            else:
                sanitized[k] = v
        return sanitized

    def _add_event(self, event_type: str, description: str):
        ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        self._timeline.append({
            "timestamp": ts,
            "event_type": event_type,
            "description": description,
        })

    def _compute_step_reward_signals(self) -> dict:
        """Compute intermediate reward signals for GRPO training.

        These signals are exposed in observation.metadata["reward_signals"]
        on every step, providing per-step feedback for multi-objective training.
        """
        scenario = self._scenario
        if not scenario:
            return {}

        # Triage progress: fraction of critical alerts acknowledged
        critical_alerts = [a for a in scenario.get("initial_alerts", []) if a.get("severity") == "critical"]
        acked = sum(1 for a in self._alerts if a.get("acknowledged"))
        triage_progress = acked / max(len(critical_alerts), 1)

        # Investigation depth: fraction of expected diagnostics completed
        expected_diags = scenario.get("expected_diagnostics", [])
        completed_diags = 0
        for d in expected_diags:
            if any(a["action_type"] == d.get("action_type") for a in self._actions_taken):
                completed_diags += 1
        investigation_depth = completed_diags / max(len(expected_diags), 1)

        # Premature action: penalty if agent remediated without investigating first
        has_investigation = any(
            a["action_type"] in ("query_logs", "check_metrics", "view_dependencies")
            for a in self._actions_taken
        )
        has_remediation = any(
            a["action_type"] in ("restart_service", "scale_service", "rollback_deploy", "update_config")
            for a in self._actions_taken
        )
        premature_action = 0.0 if (not has_remediation or has_investigation) else -0.5

        return {
            "oncall.triage_progress": round(triage_progress, 3),
            "oncall.investigation_depth": round(investigation_depth, 3),
            "oncall.premature_action": premature_action,
            "oncall.severity_set": 1.0 if self._severity is not None else 0.0,
            "oncall.summary_written": 1.0 if self._summary else 0.0,
            "oncall.resolved": 1.0 if self._resolved else 0.0,
        }

    def _redact_description(self, desc: str) -> str:
        """Redact the scenario description to remove service names and root cause hints.

        Replaces real service names with aliases and strips root cause type
        information (e.g., 'Connection Pool Exhaustion') so the agent
        can't identify the problem before investigating.
        """
        redacted = desc
        # Replace real service names with aliases
        for real_name, alias in self._service_alias_map.items():
            redacted = redacted.replace(real_name, alias)
        # Strip common root cause type patterns that give away the diagnosis
        root_cause_types = [
            "Connection Pool Exhaustion", "Memory Leak (OOM)", "Memory Leak",
            "Replication Lag", "Deadlock Storm", "CPU Spin Loop",
            "GC Pause Storm", "DNS Resolution Failure", "DNS Failure",
            "TLS Certificate Expiry", "Load Balancer Misconfiguration",
            "Bad Configuration Rollout", "Dependency Version Mismatch",
            "OOM", "Connection Pool", "Config Change",
        ]
        for rct in root_cause_types:
            redacted = redacted.replace(rct, "service degradation")
            redacted = redacted.replace(rct.lower(), "service degradation")
        return redacted

    def _redact_alert_messages(self, alerts: list[dict]) -> list[dict]:
        """Redact alerts: anonymize service names + strip metric values.

        The agent sees generic labels (service-A, service-B) instead of real
        service names. It must investigate to discover which real service
        corresponds to each alert.
        """
        redacted = []
        for alert in alerts:
            a = dict(alert)  # shallow copy
            msg = a.get("message", "")
            svc = a.get("service", "")

            # Anonymize the service field
            alias = self._service_alias_map.get(svc, svc)
            a["service"] = alias

            # Strip exact metric values from message
            msg = re.sub(r'\d+\.?\d*%', 'elevated', msg)
            msg = re.sub(r'\d+\.?\d*ms', 'high', msg)
            msg = re.sub(r'currently at \w+', 'currently elevated', msg)

            # Replace any real service names that appear in the message text
            for real_name, anon_label in self._service_alias_map.items():
                msg = msg.replace(real_name, anon_label)

            a["message"] = msg
            redacted.append(a)
        return redacted

    def _make_observation(
        self,
        message: str = "",
        done: bool = False,
        reward: float | None = None,
    ) -> OnCallObservation:
        # Partial observability: services show ALIASES until investigated
        observable_services = []
        for svc in self._services:
            real_name = svc["name"]
            alias = self._service_alias_map.get(real_name, real_name)
            if real_name in self._investigated_services:
                # Investigated: show REAL name + full details
                observable_services.append(svc)
            else:
                # Not investigated: show ALIAS + no details
                observable_services.append({"name": alias, "status": "unknown"})

        # Only show deployments for investigated services (with real names)
        observable_deployments = [
            d for d in self._deployments
            if d.get("service") in self._investigated_services
               or d.get("service_name") in self._investigated_services
        ]

        # Redact alert messages: anonymize service names + strip metric values
        redacted_alerts = self._redact_alert_messages(self._alerts)

        return OnCallObservation(
            alerts=redacted_alerts,
            services=observable_services,
            recent_deployments=observable_deployments,
            incident_timeline=self._timeline,
            current_severity=self._severity,
            available_actions=ALL_ACTIONS,
            message=message,
            done=done,
            reward=reward,
            metadata={
                "step_count": self._state.step_count,
                "task_id": self._state.task_id,
                "scenario_id": self._state.scenario_id,
                "reward_signals": self._compute_step_reward_signals(),
            },
        )

    def _compute_intermediate_reward(self) -> float:
        """Compute a progress-based reward for non-terminal steps.

        Aggregates the per-step reward signals into a single float so that
        the reward field provides varying signal across the trajectory,
        not just a sparse end-of-episode score.
        """
        signals = self._compute_step_reward_signals()
        reward = (
            signals.get("oncall.triage_progress", 0) * 0.25
            + signals.get("oncall.investigation_depth", 0) * 0.35
            + signals.get("oncall.severity_set", 0) * 0.15
            + signals.get("oncall.summary_written", 0) * 0.15
            + signals.get("oncall.resolved", 0) * 0.10
        )
        # Apply premature action penalty (scaled down for intermediate steps)
        premature = signals.get("oncall.premature_action", 0)
        if premature < 0:
            reward = max(reward + premature * 0.2, 0.0)
        return round(min(reward, 1.0), 4)

    def _compute_final_reward(self) -> float:
        return grade_episode(
            scenario=self._scenario,
            actions_taken=self._actions_taken,
            alerts_state=self._alerts,
            services_state=self._services,
            severity_set=self._severity,
            summary=self._summary,
            escalated_to=self._escalated_to,
            resolved=self._resolved,
        )
