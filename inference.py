"""
Inference Script for OnCallEnv
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI

from oncall_env.models import OnCallAction
from oncall_env.server.environment import OnCallEnvironment

API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = "oncall_env"
MAX_STEPS = 50
TEMPERATURE = 0.2
MAX_TOKENS = 500
SUCCESS_SCORE_THRESHOLD = 0.3

SYSTEM_PROMPT = textwrap.dedent("""
    You are an on-call engineer responding to a production incident.

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
    2. INVESTIGATE: Query logs and metrics for services mentioned in alerts to discover their real identity and state. View dependencies to understand the service graph.
    3. DIAGNOSE: Based on evidence from logs and metrics, identify the root cause service and failure mechanism.
    4. REMEDIATE: Fix the root cause (rollback, restart, or config change). Use the REAL service name discovered during investigation.
    5. DOCUMENT AND RESOLVE (MANDATORY):
       a. write_summary: Write a detailed incident summary mentioning the root cause service, what failed, why, and what you did to fix it.
       b. resolve_incident: You MUST call this as your final action to close the incident.

    CRITICAL: You MUST always end with write_summary followed by resolve_incident.

    Respond with exactly one JSON object per turn: {"action_type": "...", "params": {...}}
""").strip()


# ── Structured logging helpers ──────────────────────────────────────────────


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM response parsing ───────────────────────────────────────────────────


def parse_action_from_llm(response_text: str) -> OnCallAction:
    """Parse LLM response into an OnCallAction.

    Extracts the FIRST complete JSON object using bracket counting,
    so that multi-JSON responses don't break parsing.
    """
    text = response_text.strip()
    start = text.find("{")
    if start == -1:
        return OnCallAction(action_type="write_summary", params={"text": text})

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


# ── Observation formatting ──────────────────────────────────────────────────


def _attr(obj, key, default=""):
    """Access field as attribute or dict key."""
    return getattr(obj, key, None) if not isinstance(obj, dict) else obj.get(key, default)


def format_observation(obs) -> str:
    """Format observation for LLM context."""
    parts = [f"Message: {obs.message}"]

    if obs.alerts:
        parts.append("\nActive Alerts:")
        for a in obs.alerts:
            ack = " [ACK]" if _attr(a, "acknowledged") else ""
            sil = " [SILENCED]" if _attr(a, "silenced") else ""
            sev = str(_attr(a, "severity", "?")).upper()
            parts.append(f"  [{sev}] {_attr(a, 'alert_id')}: {_attr(a, 'service')} - {_attr(a, 'message')}{ack}{sil}")

    if obs.services:
        parts.append("\nService Status:")
        for s in obs.services:
            parts.append(f"  {_attr(s, 'name')}: {_attr(s, 'status')} (latency={_attr(s, 'latency_ms', 0)}ms, errors={_attr(s, 'error_rate', 0)}%)")

    if obs.log_results:
        parts.append(f"\nLog Results ({len(obs.log_results)} entries):")
        for entry in obs.log_results[:10]:
            parts.append(f"  [{_attr(entry, 'level')}] {_attr(entry, 'timestamp')} {_attr(entry, 'service')}: {_attr(entry, 'message')}")

    if obs.metric_results:
        parts.append(f"\nMetric Results: {json.dumps(obs.metric_results, indent=2)}")

    if obs.dependency_graph:
        parts.append(f"\nDependency Graph: {json.dumps(obs.dependency_graph, indent=2)}")

    if obs.recent_deployments:
        parts.append("\nRecent Deployments:")
        for d in obs.recent_deployments:
            parts.append(f"  {_attr(d, 'service')} v{_attr(d, 'version')} at {_attr(d, 'timestamp')} by {_attr(d, 'deployer')}")

    return "\n".join(parts)


# ── Error detection ─────────────────────────────────────────────────────────

_ERROR_PATTERNS = ("not found", "unknown action", "error executing")


def _extract_error(message: str) -> Optional[str]:
    """Return the message as an error string if it indicates a failed action."""
    msg_lower = message.lower()
    if any(pat in msg_lower for pat in _ERROR_PATTERNS):
        return message
    return None


# ── Episode runner ──────────────────────────────────────────────────────────


def run_episode(task_id: int = 1, scenario_idx: int = 0) -> float:
    """Run a single episode against the environment."""
    client_llm = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    task_name = f"task{task_id}_scenario{scenario_idx}"

    env = OnCallEnvironment()
    obs = env.reset(task_id=task_id, scenario_idx=scenario_idx)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_observation(obs)},
        ]

        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break

            response = client_llm.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            llm_text = (response.choices[0].message.content or "").strip()

            action = parse_action_from_llm(llm_text)
            action_str = f"{action.action_type}({json.dumps(action.params)})"

            obs = env.step(action)

            reward = obs.reward if obs.reward is not None else 0.0
            done = obs.done
            error = _extract_error(obs.message)

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            messages.append({"role": "assistant", "content": llm_text})
            messages.append({"role": "user", "content": format_observation(obs)})

            if done:
                break

        # Score is the final reward (grader returns 0-1)
        score = rewards[-1] if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", file=sys.stderr, flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    for task_id in range(1, 5):
        try:
            run_episode(task_id=task_id, scenario_idx=0)
        except Exception as e:
            print(f"[DEBUG] Task {task_id} failed: {e}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
