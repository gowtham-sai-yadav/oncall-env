"""Baseline inference script for OnCallEnv using OpenAI-compatible API."""

from __future__ import annotations

import asyncio
import json
import os
import sys

from oncall_env.client import OnCallEnvClient
from oncall_env.models import OnCallAction

SYSTEM_PROMPT = """You are an expert on-call engineer responding to a production incident.
You have access to these actions:
- query_logs: Query service logs. Params: service (str), level (optional: DEBUG/INFO/WARN/ERROR/FATAL), time_range (optional)
- check_metrics: Check service metrics. Params: service (str), metric_name (str)
- view_dependencies: View service dependency graph. Params: service_name (str)
- acknowledge_alert: Acknowledge an alert. Params: alert_id (str)
- silence_alert: Silence a non-actionable alert. Params: alert_id (str), duration (optional)
- restart_service: Restart a service. Params: service_name (str)
- scale_service: Scale a service. Params: service_name (str), replicas (int)
- rollback_deploy: Rollback a deployment. Params: service_name (str), target_version (str)
- update_config: Update service config. Params: service_name (str), config_key (str), config_value (str)
- set_severity: Set incident severity. Params: level (SEV1/SEV2/SEV3/SEV4)
- write_summary: Write incident summary. Params: text (str)
- escalate: Escalate to another team. Params: team (str)
- resolve_incident: Mark incident as resolved. Params: resolution_note (str)

Follow this incident response process strictly and efficiently (aim to finish within 10-15 actions):

1. TRIAGE (2-3 actions): Set severity. Acknowledge CRITICAL alerts. Silence INFO/noise alerts.
2. DIAGNOSE (3-5 actions): Query logs (ERROR level) and check metrics for degraded/down services. View dependencies to trace the failure chain. Focus on recently deployed services and downstream dependencies.
3. REMEDIATE (1-2 actions): Apply the fix — rollback the bad deployment, restart the root-cause service, or update the broken config. Target the ROOT CAUSE service, not symptoms.
4. DOCUMENT AND RESOLVE (2 actions — MANDATORY):
   a. write_summary: Write a detailed incident summary mentioning the root cause service, what failed, why, and what you did to fix it. Include technical keywords like "root cause", "rollback", "config", "deploy", etc.
   b. resolve_incident: You MUST call this as your final action to close the incident. Without this, the incident is not considered complete.

CRITICAL: You MUST always end with write_summary followed by resolve_incident. Never skip these steps. An incident without resolution is a failed incident.

Respond with exactly one JSON object per turn: {"action_type": "...", "params": {...}}
"""


def parse_action_from_llm(response_text: str) -> OnCallAction:
    """Parse LLM response into an OnCallAction."""
    text = response_text.strip()
    # Find JSON in response
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return OnCallAction(action_type="write_summary", params={"text": text})

    try:
        data = json.loads(text[start:end])
        return OnCallAction(
            action_type=data.get("action_type", "write_summary"),
            params=data.get("params", {}),
        )
    except json.JSONDecodeError:
        return OnCallAction(action_type="write_summary", params={"text": text})


def _alert_attr(a, key, default=""):
    """Access alert field as attribute or dict key (handles both typed models and raw dicts)."""
    return getattr(a, key, None) if not isinstance(a, dict) else a.get(key, default)


def format_observation(obs) -> str:
    """Format observation for LLM context."""
    parts = [f"Message: {obs.message}"]

    if obs.alerts:
        parts.append("\nActive Alerts:")
        for a in obs.alerts:
            ack = " [ACK]" if _alert_attr(a, "acknowledged") else ""
            sil = " [SILENCED]" if _alert_attr(a, "silenced") else ""
            sev = str(_alert_attr(a, "severity", "?")).upper()
            parts.append(f"  [{sev}] {_alert_attr(a, 'alert_id')}: {_alert_attr(a, 'service')} - {_alert_attr(a, 'message')}{ack}{sil}")

    if obs.services:
        parts.append("\nService Status:")
        for s in obs.services:
            name = _alert_attr(s, "name")
            status = _alert_attr(s, "status")
            latency = _alert_attr(s, "latency_ms", 0)
            errors = _alert_attr(s, "error_rate", 0)
            parts.append(f"  {name}: {status} (latency={latency}ms, errors={errors}%)")

    if obs.log_results:
        parts.append(f"\nLog Results ({len(obs.log_results)} entries):")
        for entry in obs.log_results[:10]:
            lvl = _alert_attr(entry, "level")
            ts = _alert_attr(entry, "timestamp")
            svc = _alert_attr(entry, "service")
            msg = _alert_attr(entry, "message")
            parts.append(f"  [{lvl}] {ts} {svc}: {msg}")

    if obs.metric_results:
        parts.append(f"\nMetric Results: {json.dumps(obs.metric_results, indent=2)}")

    if obs.dependency_graph:
        parts.append(f"\nDependency Graph: {json.dumps(obs.dependency_graph, indent=2)}")

    if obs.recent_deployments:
        parts.append("\nRecent Deployments:")
        for d in obs.recent_deployments:
            svc = _alert_attr(d, "service")
            ver = _alert_attr(d, "version")
            ts = _alert_attr(d, "timestamp")
            who = _alert_attr(d, "deployer")
            parts.append(f"  {svc} v{ver} at {ts} by {who}")

    return "\n".join(parts)


async def run_baseline(base_url: str, task_id: int = 1, scenario_idx: int = 0):
    """Run baseline agent against a task."""
    try:
        from openai import OpenAI
    except ImportError:
        print("openai package required. Install with: pip install openai")
        sys.exit(1)

    api_key = os.environ.get("HF_TOKEN", os.environ.get("OPENAI_API_KEY", ""))
    api_base = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("MODEL_NAME", "gpt-4o-mini")

    client_llm = OpenAI(api_key=api_key, base_url=api_base)

    env_client = OnCallEnvClient(base_url=base_url)
    await env_client.connect()

    try:
        result = await env_client.reset(task_id=task_id, scenario_idx=scenario_idx)
        obs = result.observation
        print(f"\n{'='*60}")
        print(f"Task {task_id} | Scenario {scenario_idx}")
        print(f"{'='*60}")
        print(format_observation(obs))

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_observation(obs)},
        ]

        step = 0
        while not obs.done and step < 50:
            step += 1
            response = client_llm.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                max_completion_tokens=500,
            )
            llm_text = response.choices[0].message.content or ""
            print(f"\n--- Step {step} ---")
            print(f"LLM: {llm_text}")

            action = parse_action_from_llm(llm_text)
            print(f"Action: {action.action_type} {action.params}")

            result = await env_client.step(action)
            obs = result.observation
            print(f"Reward: {obs.reward}, Done: {obs.done}")
            print(f"Env: {obs.message}")

            messages.append({"role": "assistant", "content": llm_text})
            messages.append({"role": "user", "content": format_observation(obs)})

        print(f"\n{'='*60}")
        print(f"Final Reward: {obs.reward}")
        print(f"Steps taken: {step}")
        print(f"{'='*60}")
        return obs.reward

    finally:
        await env_client.close()


async def main():
    base_url = os.environ.get("ONCALL_ENV_URL", "ws://localhost:8000")

    scores = {}
    for task_id in range(1, 5):
        try:
            reward = await run_baseline(base_url, task_id=task_id, scenario_idx=0)
            scores[f"task{task_id}"] = reward
        except Exception as e:
            print(f"Task {task_id} failed: {e}")
            scores[f"task{task_id}"] = None

    print("\n" + "=" * 60)
    print("BASELINE SCORES")
    print("=" * 60)
    for task, score in scores.items():
        print(f"  {task}: {score}")


if __name__ == "__main__":
    asyncio.run(main())
