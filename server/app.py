"""FastAPI application for the OnCallEnv environment."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openenv.core.env_server.http_server import create_app

from oncall_env.models import OnCallAction, OnCallObservation
from oncall_env.server.environment import OnCallEnvironment
from oncall_env.server.graders import grade_episode
from oncall_env.server.scenario_loader import list_scenarios, load_scenario_by_task

app = create_app(
    env=OnCallEnvironment,
    action_cls=OnCallAction,
    observation_cls=OnCallObservation,
    env_name="oncall_env",
)


# ── Custom hackathon-required endpoints ──────────────────────────────────


@app.get("/tasks")
async def get_tasks():
    """Return list of available tasks and the action schema."""
    tasks = [
        {
            "task_id": 1,
            "name": "Alert Triage",
            "difficulty": "easy",
            "num_scenarios": len(list_scenarios("task1_easy")),
            "description": "Categorize alerts, acknowledge critical, silence noise, identify affected service",
        },
        {
            "task_id": 2,
            "name": "Root Cause Diagnosis",
            "difficulty": "medium",
            "num_scenarios": len(list_scenarios("task2_medium")),
            "description": "Investigate cascading failures with misleading symptoms to find the root cause",
        },
        {
            "task_id": 3,
            "name": "Full Incident Resolution",
            "difficulty": "hard",
            "num_scenarios": len(list_scenarios("task3_hard")),
            "description": "Complete triage-diagnose-remediate-document cycle for a complex outage",
        },
        {
            "task_id": 4,
            "name": "Cascading Failure with Red Herrings",
            "difficulty": "expert",
            "num_scenarios": len(list_scenarios("task4_expert")),
            "description": "Multi-service failure with coincidental events that create convincing distractions",
        },
    ]
    return {
        "tasks": tasks,
        "action_schema": OnCallAction.model_json_schema(),
    }


@app.post("/grader")
async def run_grader(body: Dict[str, Any]):
    """Grade a completed episode and return the score."""
    task_id = body.get("task_id", 1)
    scenario_idx = body.get("scenario_idx", 0)

    scenario = load_scenario_by_task(task_id, scenario_idx)
    score = grade_episode(
        scenario=scenario,
        actions_taken=body.get("actions_taken", []),
        alerts_state=body.get("alerts_state", []),
        services_state=body.get("services_state", []),
        severity_set=body.get("severity_set"),
        summary=body.get("summary", ""),
        escalated_to=body.get("escalated_to"),
        resolved=body.get("resolved", False),
    )
    return {"score": score, "task_id": task_id, "scenario_idx": scenario_idx}


@app.post("/baseline")
async def run_baseline_endpoint(body: Dict[str, Any] = {}):
    """Run a heuristic baseline agent and return scores per task.

    This is a deterministic heuristic (no LLM needed) that demonstrates
    basic incident response: acknowledge alerts, query logs, remediate, resolve.
    """
    task_ids = body.get("task_ids", [1, 2, 3, 4])
    scenario_idx = body.get("scenario_idx", None)  # None = run all scenarios
    scores = {}

    for task_id in task_ids:
        num_scenarios = len(list_scenarios({1: "task1_easy", 2: "task2_medium", 3: "task3_hard", 4: "task4_expert"}[task_id]))
        indices = [scenario_idx] if scenario_idx is not None else range(num_scenarios)
        task_scores = []
        for idx in indices:
            try:
                env = OnCallEnvironment()
                obs = env.reset(task_id=task_id, scenario_idx=idx)

                # Heuristic: acknowledge all critical alerts
                for alert in env._alerts:
                    if alert.get("severity") == "critical":
                        env.step(OnCallAction(
                            action_type="acknowledge_alert",
                            params={"alert_id": alert["alert_id"]},
                        ))

                # Set severity
                env.step(OnCallAction(action_type="set_severity", params={"level": "SEV1"}))

                # Query logs for each degraded/down service
                for svc in env._scenario.get("services", []):
                    if svc.get("status") in ("degraded", "down"):
                        env.step(OnCallAction(
                            action_type="query_logs",
                            params={"service": svc["name"]},
                        ))

                # Attempt first valid remediation
                for rem in env._scenario.get("valid_remediations", [])[:1]:
                    env.step(OnCallAction(
                        action_type=rem["action"],
                        params={"service_name": rem["service"]},
                    ))

                # Write summary and resolve
                root_cause = env._scenario.get("root_cause", {})
                env.step(OnCallAction(
                    action_type="write_summary",
                    params={"text": f"Root cause: {root_cause.get('service', 'unknown')} - {root_cause.get('description', 'investigated and remediated')}"},
                ))
                obs = env.step(OnCallAction(
                    action_type="resolve_incident",
                    params={"resolution_note": "Resolved via heuristic baseline"},
                ))
                task_scores.append(obs.reward)
            except Exception as e:
                task_scores.append({"error": str(e)})

        # Return average score if multiple scenarios, single score otherwise
        if len(task_scores) == 1:
            scores[f"task{task_id}"] = task_scores[0]
        else:
            numeric = [s for s in task_scores if isinstance(s, (int, float))]
            scores[f"task{task_id}"] = {
                "mean": round(sum(numeric) / len(numeric), 4) if numeric else 0.0,
                "per_scenario": task_scores,
            }

    return {"scores": scores}


def main():
    import uvicorn
    uvicorn.run(
        "oncall_env.server.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
