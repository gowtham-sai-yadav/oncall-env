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

Follow this incident response process:
1. TRIAGE: Review alerts, acknowledge critical ones, silence noise, set severity
2. DIAGNOSE: Query logs and metrics for affected services, check dependencies
3. IDENTIFY ROOT CAUSE: Determine the actual root cause (not just symptoms)
4. REMEDIATE: Apply the correct fix (restart, rollback, config change, etc.)
5. DOCUMENT: Write an incident summary and resolve

Respond with a JSON object: {"action_type": "...", "params": {...}}
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


def format_observation(obs) -> str:
    """Format observation for LLM context."""
    parts = [f"Message: {obs.message}"]

    if obs.alerts:
        parts.append("\nActive Alerts:")
        for a in obs.alerts:
            ack = " [ACK]" if a.get("acknowledged") else ""
            sil = " [SILENCED]" if a.get("silenced") else ""
            parts.append(f"  [{a.get('severity', '?').upper()}] {a.get('alert_id')}: {a.get('service')} - {a.get('message')}{ack}{sil}")

    if obs.services:
        parts.append("\nService Status:")
        for s in obs.services:
            parts.append(f"  {s.get('name')}: {s.get('status')} (latency={s.get('latency_ms')}ms, errors={s.get('error_rate')}%)")

    if obs.log_results:
        parts.append(f"\nLog Results ({len(obs.log_results)} entries):")
        for entry in obs.log_results[:10]:
            parts.append(f"  [{entry.get('level')}] {entry.get('timestamp')} {entry.get('service')}: {entry.get('message')}")

    if obs.metric_results:
        parts.append(f"\nMetric Results: {json.dumps(obs.metric_results, indent=2)}")

    if obs.dependency_graph:
        parts.append(f"\nDependency Graph: {json.dumps(obs.dependency_graph, indent=2)}")

    if obs.recent_deployments:
        parts.append("\nRecent Deployments:")
        for d in obs.recent_deployments:
            parts.append(f"  {d.get('service')} v{d.get('version')} at {d.get('timestamp')} by {d.get('deployer')}")

    return "\n".join(parts)


async def run_baseline(base_url: str, task_id: int = 1, scenario_idx: int = 0):
    """Run baseline agent against a task."""
    try:
        from openai import OpenAI
    except ImportError:
        print("openai package required. Install with: pip install openai")
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

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
        while not obs.done and step < 25:
            step += 1
            response = client_llm.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                max_tokens=500,
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
