"""Gradio UI for the OnCallEnv environment.

Provides an interactive interface for judges to explore scenarios,
take actions, and see environment state in real time.
"""

from __future__ import annotations

import json
from typing import Any

import gradio as gr

from oncall_env.models import OnCallAction
from oncall_env.server.environment import OnCallEnvironment
from oncall_env.server.scenario_loader import list_scenarios

# ── Helpers ────────────────────────────────────────────────────────────────

TASK_MAP = {
    1: ("Alert Triage", "task1_easy"),
    2: ("Root Cause Diagnosis", "task2_medium"),
    3: ("Full Incident Resolution", "task3_hard"),
    4: ("Cascading Failure with Red Herrings", "task4_expert"),
}

ACTION_PARAMS_HINT = {
    "query_logs": "service, level (optional), time_range (optional)",
    "check_metrics": "service, metric_name",
    "view_dependencies": "service_name",
    "acknowledge_alert": "alert_id",
    "silence_alert": "alert_id",
    "restart_service": "service_name",
    "scale_service": "service_name, replicas",
    "rollback_deploy": "service_name, target_version",
    "update_config": "service_name, config_key, config_value",
    "set_severity": "level (SEV1/SEV2/SEV3/SEV4)",
    "write_summary": "text",
    "escalate": "team",
    "resolve_incident": "resolution_note",
}


def _to_serializable(obj: Any) -> Any:
    """Convert Pydantic models / nested structures to JSON-safe dicts."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, list):
        return [_to_serializable(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    return obj


def _attr(obj, key, default=""):
    """Get attribute from a Pydantic model or dict."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _format_alerts(alerts: list) -> str:
    if not alerts:
        return "No active alerts."
    lines = []
    for a in alerts:
        sev = _attr(a, "severity", "?")
        svc = _attr(a, "service", "?")
        msg = _attr(a, "message", "")
        aid = _attr(a, "alert_id", "")
        ack = _attr(a, "acknowledged", False)
        sil = _attr(a, "silenced", False)
        status_tags = []
        if ack:
            status_tags.append("ACK")
        if sil:
            status_tags.append("SILENCED")
        tag = f" [{', '.join(status_tags)}]" if status_tags else ""
        lines.append(f"[{sev.upper()}] {svc} — {msg} (id: {aid}){tag}")
    return "\n".join(lines)


def _format_services(services: list) -> str:
    if not services:
        return "No service data."
    lines = []
    for s in services:
        name = _attr(s, "name", "?")
        status = _attr(s, "status", "?")
        err = _attr(s, "error_rate", None)
        lat = _attr(s, "latency_ms", None)
        detail = ""
        if status != "unknown" and err is not None:
            detail = f" | err={err}% lat={lat}ms"
        lines.append(f"  {name}: {status}{detail}")
    return "\n".join(lines)


def _format_timeline(timeline: list) -> str:
    if not timeline:
        return "No events yet."
    lines = []
    for e in timeline:
        ts = _attr(e, "timestamp", "")
        desc = _attr(e, "description", "")
        lines.append(f"  {ts}  {desc}")
    return "\n".join(lines)


def _format_reward_signals(metadata: dict) -> str:
    """Format the per-step reward signals for display."""
    signals = metadata.get("reward_signals", {}) if metadata else {}
    if not signals:
        return "No reward signals yet. Reset and take actions to see scoring."
    lines = []
    labels = {
        "oncall.triage_progress": "Triage Progress (alerts acknowledged)",
        "oncall.investigation_depth": "Investigation Depth (diagnostics done)",
        "oncall.premature_action": "Premature Action Penalty",
        "oncall.severity_set": "Severity Set",
        "oncall.summary_written": "Summary Written",
        "oncall.resolved": "Incident Resolved",
    }
    for key, val in signals.items():
        label = labels.get(key, key)
        bar_len = int(max(val, 0) * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        lines.append(f"  {label}: {val:>6.3f}  [{bar}]")
    return "\n".join(lines)


def _obs_to_display(obs) -> dict[str, str]:
    """Convert an observation to display strings for each UI component."""
    return {
        "message": obs.message or "",
        "alerts": _format_alerts(obs.alerts),
        "services": _format_services(obs.services),
        "timeline": _format_timeline(obs.incident_timeline),
        "severity": obs.current_severity or "Not set",
        "reward": str(obs.reward) if obs.reward is not None else "—",
        "done": "YES — episode complete" if obs.done else "No",
        "step": str(obs.metadata.get("step_count", 0)) if obs.metadata else "0",
        "logs": json.dumps(_to_serializable(obs.log_results), indent=2) if obs.log_results else "",
        "metrics": json.dumps(_to_serializable(obs.metric_results), indent=2) if obs.metric_results else "",
        "deps": json.dumps(_to_serializable(obs.dependency_graph), indent=2) if obs.dependency_graph else "",
        "reward_signals": _format_reward_signals(obs.metadata),
    }


# ── Session state ──────────────────────────────────────────────────────────

_envs: dict[str, OnCallEnvironment] = {}


def _get_env(session_id: str) -> OnCallEnvironment:
    if session_id not in _envs:
        _envs[session_id] = OnCallEnvironment()
    return _envs[session_id]


# ── Callbacks ──────────────────────────────────────────────────────────────

def reset_env(task_id: int, scenario_idx: int, session_state: dict):
    session_id = session_state.get("id", "default")
    env = _get_env(session_id)
    task_dir = TASK_MAP.get(task_id, (None, "task1_easy"))[1]
    num_scenarios = len(list_scenarios(task_dir))
    if scenario_idx >= num_scenarios:
        scenario_idx = 0

    obs = env.reset(task_id=task_id, scenario_idx=scenario_idx)
    d = _obs_to_display(obs)
    history = f"=== Reset: Task {task_id}, Scenario {scenario_idx} ===\n{d['message']}\n"
    return (
        d["message"],    # message
        d["alerts"],     # alerts
        d["services"],   # services
        d["timeline"],   # timeline
        d["severity"],   # severity
        d["reward"],     # reward
        d["done"],       # done
        d["step"],       # step
        "",              # logs
        "",              # metrics
        "",              # deps
        history,         # action log
        d["reward_signals"],  # reward breakdown
    )


def take_action(action_type: str, params_json: str, action_log: str, session_state: dict):
    session_id = session_state.get("id", "default")
    env = _get_env(session_id)

    try:
        params = json.loads(params_json) if params_json.strip() else {}
    except json.JSONDecodeError as e:
        return (
            f"Invalid JSON params: {e}", "", "", "", "", "", "", "",
            "", "", "", action_log + f"\n[ERROR] Invalid JSON: {e}\n", "",
        )

    action = OnCallAction(action_type=action_type, params=params)
    obs = env.step(action)
    d = _obs_to_display(obs)

    entry = f"\n> Step {d['step']}: {action_type}({json.dumps(params)})\n  {d['message']}\n  reward={d['reward']} done={d['done']}\n"
    new_log = action_log + entry

    return (
        d["message"],
        d["alerts"],
        d["services"],
        d["timeline"],
        d["severity"],
        d["reward"],
        d["done"],
        d["step"],
        d["logs"],
        d["metrics"],
        d["deps"],
        new_log,
        d["reward_signals"],
    )


def update_params_hint(action_type: str) -> str:
    hint = ACTION_PARAMS_HINT.get(action_type, "")
    if not hint:
        return "{}"
    # Build a template JSON
    keys = [k.strip().split(" ")[0] for k in hint.split(",")]
    template = {k: "" for k in keys}
    return json.dumps(template, indent=2)


# ── Build the Gradio app ──────────────────────────────────────────────────

def create_gradio_app() -> gr.Blocks:
    with gr.Blocks(
        title="OnCallEnv — Incident Response RL Environment",
    ) as demo:
        session_state = gr.State({"id": "gradio-default"})

        gr.Markdown("# OnCallEnv — Incident Response Command Center")
        gr.Markdown(
            "Interactive RL environment simulating production on-call engineering. "
            "Select a task & scenario, then take actions to triage, diagnose, and resolve incidents."
        )

        # ── Controls ───────────────────────────────────────────────────
        with gr.Row():
            task_dd = gr.Dropdown(
                choices=[(f"Task {k}: {v[0]}", k) for k, v in TASK_MAP.items()],
                value=1,
                label="Task",
                scale=2,
            )
            scenario_num = gr.Number(value=0, label="Scenario Index", precision=0, scale=1)
            reset_btn = gr.Button("Reset Environment", variant="primary", scale=1)

        # ── Status bar ─────────────────────────────────────────────────
        with gr.Row():
            step_display = gr.Textbox(label="Step", value="0", interactive=False, scale=1)
            severity_display = gr.Textbox(label="Severity", value="Not set", interactive=False, scale=1)
            reward_display = gr.Textbox(label="Reward", value="—", interactive=False, scale=1)
            done_display = gr.Textbox(label="Done?", value="No", interactive=False, scale=1)

        # ── Message ────────────────────────────────────────────────────
        message_display = gr.Textbox(label="Environment Message", lines=3, interactive=False)

        # ── Main panels ────────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=1):
                alerts_display = gr.Textbox(label="Active Alerts", lines=10, interactive=False)
                services_display = gr.Textbox(label="Services", lines=8, interactive=False)
            with gr.Column(scale=1):
                timeline_display = gr.Textbox(label="Incident Timeline", lines=10, interactive=False)
                with gr.Tabs():
                    with gr.Tab("Logs"):
                        logs_display = gr.Textbox(label="Log Results", lines=6, interactive=False)
                    with gr.Tab("Metrics"):
                        metrics_display = gr.Textbox(label="Metric Results", lines=6, interactive=False)
                    with gr.Tab("Dependencies"):
                        deps_display = gr.Textbox(label="Dependency Graph", lines=6, interactive=False)

        # ── Action input ───────────────────────────────────────────────
        gr.Markdown("### Take an Action")
        with gr.Row():
            action_dd = gr.Dropdown(
                choices=list(ACTION_PARAMS_HINT.keys()),
                value="query_logs",
                label="Action Type",
                scale=2,
            )
            params_input = gr.Textbox(
                label="Params (JSON)",
                value='{"service": ""}',
                lines=3,
                scale=3,
            )
            action_btn = gr.Button("Execute Action", variant="primary", scale=1)

        # ── Reward breakdown ───────────────────────────────────────────
        gr.Markdown("### Reward Signals (Grader Breakdown)")
        reward_signals_display = gr.Textbox(
            label="Reward Signals",
            value="No reward signals yet. Reset and take actions to see scoring.",
            lines=8,
            interactive=False,
        )

        # ── Action history ─────────────────────────────────────────────
        action_log = gr.Textbox(label="Action History", lines=10, interactive=False)

        # ── Wiring ─────────────────────────────────────────────────────
        all_outputs = [
            message_display, alerts_display, services_display, timeline_display,
            severity_display, reward_display, done_display, step_display,
            logs_display, metrics_display, deps_display, action_log,
            reward_signals_display,
        ]

        reset_btn.click(
            fn=reset_env,
            inputs=[task_dd, scenario_num, session_state],
            outputs=all_outputs,
        )

        action_btn.click(
            fn=take_action,
            inputs=[action_dd, params_input, action_log, session_state],
            outputs=all_outputs,
        )

        action_dd.change(
            fn=update_params_hint,
            inputs=[action_dd],
            outputs=[params_input],
        )

    return demo
