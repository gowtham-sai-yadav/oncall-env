"""Baseline inference script for OnCallEnv using OpenAI-compatible API."""

from __future__ import annotations

import asyncio
import json
import os
import sys

from oncall_env.client import OnCallEnvClient
from oncall_env.models import OnCallAction

SYSTEM_PROMPT = """You are an on-call engineer responding to a production incident.

ENVIRONMENT RULES:
- Alerts reference anonymous service labels (service-A, service-B, etc.), NOT real service names.
- The services list also shows anonymous labels with "unknown" status.
- You can query_logs and check_metrics using EITHER the anonymous label OR the real service name.
- When you investigate a service, its REAL identity and status are revealed in the next observation.
- Deployments are hidden until you investigate the relevant service.
- Metric values in alerts are redacted. Use check_metrics for actual numbers.
- Logs contain raw technical output (stack traces, error codes, IP addresses) that you must interpret.

AVAILABLE ACTIONS:
- query_logs: Params: service (str), level (optional), time_range (optional)
- check_metrics: Params: service (str), metric_name (str)
- view_dependencies: Params: service_name (str)
- acknowledge_alert: Params: alert_id (str)
- silence_alert: Params: alert_id (str)
- restart_service: Params: service_name (str)
- scale_service: Params: service_name (str), replicas (int)
- rollback_deploy: Params: service_name (str), target_version (str)
- update_config: Params: service_name (str), config_key (str), config_value (str)
- set_severity: Params: level (SEV1/SEV2/SEV3/SEV4)
- write_summary: Params: text (str)
- escalate: Params: team (str)
- resolve_incident: Params: resolution_note (str)

PROCESS:
1. TRIAGE: Acknowledge CRITICAL alerts. Silence INFO alerts. Set severity.
2. INVESTIGATE: Query logs and metrics for services mentioned in alerts to discover their real identity and state for degraded/down services. View dependencies to understand the service graph.
3. DIAGNOSE: Based on evidence from logs and metrics, identify the root cause service and failure mechanism.
4. REMEDIATE: Fix the root cause (rollback, restart, or config change). Use the REAL service name discovered during investigation.
5.  DOCUMENT AND RESOLVE (MANDATORY):
   a. write_summary: Write a detailed incident summary mentioning the root cause service, what failed, why, and what you did to fix it. Include technical keywords like "root cause", "rollback", "config", "deploy", etc.
   b. resolve_incident: You MUST call this as your final action to close the incident. Without this, the incident is not considered complete.


CRITICAL: You MUST always end with write_summary followed by resolve_incident. Never skip these steps. An incident without resolution is a failed incident.

Respond with exactly one JSON object per turn: {"action_type": "...", "params": {...}}
"""


def parse_action_from_llm(response_text: str) -> OnCallAction:
    """Parse LLM response into an OnCallAction.

    Extracts the FIRST complete JSON object using bracket counting,
    so that multi-JSON responses (a known GPT-5.2 behavior) don't
    break parsing.
    """
    text = response_text.strip()
    start = text.find("{")
    if start == -1:
        return OnCallAction(action_type="write_summary", params={"text": text})

    # Find the end of the FIRST complete JSON object via bracket counting
    depth = 0
    in_string = False
    escape_next = False
    end = -1
    for i in range(start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end == -1:
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

        # Structured logging required by hackathon evaluator
        print(f"[START] task_id={task_id} scenario_idx={scenario_idx}")

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

            action = parse_action_from_llm(llm_text)
            params_json = json.dumps(action.params)
            print(f"[STEP] action={action.action_type} params={params_json}")

            result = await env_client.step(action)
            obs = result.observation

            messages.append({"role": "assistant", "content": llm_text})
            messages.append({"role": "user", "content": format_observation(obs)})

        print(f"[END] reward={obs.reward} steps={step}")
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

    print(f"[SUMMARY] scores={json.dumps(scores)}")


if __name__ == "__main__":
    asyncio.run(main())
