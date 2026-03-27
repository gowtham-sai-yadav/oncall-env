# OnCallEnv -- Incident Response Command Center

An OpenEnv RL environment that simulates production incident response. An AI agent acts as an **on-call engineer** -- triaging alerts, diagnosing root causes, remediating failures, and documenting incidents.

## Why Incident Response?

On-call engineering is a **universal real-world task** that every engineering organization performs daily. Unlike game environments, incident response requires:
- Multi-step reasoning over noisy, partial information
- Distinguishing signal from noise (red herrings, correlated-but-non-causal signals)
- Balancing speed vs. thoroughness
- Clear documentation and communication

Each scenario is a **digital twin** of a real production incident with planted root causes and deterministic grading.

## Quick Start

```bash
# Install
uv sync  # or: pip install -e .

# Run server
uvicorn oncall_env.server.app:app --host 0.0.0.0 --port 8000

# Run tests
pytest server/test_oncall.py -v

# Run baseline (requires OPENAI_API_KEY)
python baseline.py
```

## Docker

```bash
docker build -f server/Dockerfile -t oncall-env .
docker run -p 8000:8000 oncall-env
```

## Tasks (4 Difficulty Levels)

| Task | Difficulty | Description | Expected Baseline Score |
|------|-----------|-------------|------------------------|
| 1 | Easy | Alert Triage: categorize 7 alerts, acknowledge critical, silence noise | 0.6-0.8 |
| 2 | Medium | Root Cause Diagnosis: cascading failures with DB connection pool exhaustion | 0.3-0.5 |
| 3 | Hard | Full Incident Resolution: triage → diagnose → remediate → document (OOM from config change) | 0.1-0.3 |
| 4 | Expert | Cascading Failure with Red Herrings: corrupted DNS zone file + unrelated deployments | 0.05-0.15 |

## Action Space

| Action | Description | Parameters |
|--------|-------------|------------|
| `query_logs` | Query service logs | `service`, `level` (opt), `time_range` (opt) |
| `check_metrics` | Check service metrics | `service`, `metric_name` |
| `view_dependencies` | View service dependency graph | `service_name` |
| `acknowledge_alert` | Acknowledge an alert | `alert_id` |
| `silence_alert` | Silence a non-actionable alert | `alert_id` |
| `restart_service` | Restart a service | `service_name` |
| `scale_service` | Scale replicas | `service_name`, `replicas` |
| `rollback_deploy` | Rollback to previous version | `service_name`, `target_version` |
| `update_config` | Update service configuration | `service_name`, `config_key`, `config_value` |
| `set_severity` | Set incident severity | `level` (SEV1-SEV4) |
| `write_summary` | Write incident summary | `text` |
| `escalate` | Escalate to another team | `team` |
| `resolve_incident` | Mark incident resolved | `resolution_note` |

## Observation Space

Each step returns:
- **alerts**: List of active alerts with severity, service, message
- **services**: Current service status (healthy/degraded/down), latency, error rate
- **recent_deployments**: Recent deploys with version, timestamp, deployer
- **log_results**: Results from last `query_logs` action
- **metric_results**: Results from last `check_metrics` action
- **dependency_graph**: Results from last `view_dependencies` action
- **incident_timeline**: Chronological event log
- **current_severity**: Currently set severity level
- **message**: Human-readable feedback on last action
- **done**: Whether episode is complete
- **reward**: Final reward (0.0-1.0) when done

## Reward Function

```
reward = triage_accuracy    × 0.15
       + diagnostic_quality × 0.25
       + root_cause_correct × 0.25
       + remediation_quality× 0.20
       + efficiency         × 0.10
       + documentation      × 0.05
```

Each component independently scores 0.0-1.0. Even minimal triage yields partial reward (never sparse). Weights are adjusted per task via `grading_rubric` in scenario files.

## Example: Task 2 Walkthrough

```
Step 1: acknowledge_alert(alert-201)    → "Alert alert-201 acknowledged"
Step 2: set_severity(SEV1)              → "Severity set to SEV1"
Step 3: view_dependencies(order-service) → Shows: order-service → [inventory-service, payment-api, order-db]
Step 4: query_logs(inventory-service)   → Reveals: OOM killer, connection pool exhaustion
Step 5: check_metrics(inventory-service) → Memory: 55→99%, DB connections: 40→100 (saturated)
Step 6: rollback_deploy(inventory-service, v2.5.0) → "Service rolled back. Status healthy."
Step 7: write_summary("Root cause: inventory-service v2.5.1 memory leak...")
Step 8: resolve_incident("Rolled back inventory-service")

Final reward: ~0.75 (strong diagnostic + correct remediation + good documentation)
```

## Baseline Scores

| Task | Score | Notes |
|------|-------|-------|
| Task 1 (Easy) | ~0.65 | Good triage, partial diagnostics |
| Task 2 (Medium) | ~0.45 | Identifies symptoms, sometimes misses root cause |
| Task 3 (Hard) | ~0.25 | Struggles with multi-service correlation |
| Task 4 (Expert) | ~0.10 | Often distracted by red herrings |

## Architecture

```
oncall_env/
├── __init__.py              # Exports
├── models.py                # Action, Observation, State (Pydantic)
├── client.py                # EnvClient subclass
├── baseline.py              # Baseline inference script
├── openenv.yaml             # OpenEnv manifest
├── pyproject.toml           # Package config
├── scenarios/               # Pre-generated incident scenarios (JSON)
│   ├── task1_easy/          # 2 scenarios
│   ├── task2_medium/        # 2 scenarios
│   ├── task3_hard/          # 2 scenarios
│   └── task4_expert/        # 2 scenarios
└── server/
    ├── app.py               # FastAPI app (create_app)
    ├── environment.py       # Environment: reset/step/state
    ├── graders.py           # Deterministic grading (0.0-1.0)
    ├── scenario_loader.py   # Load scenario JSONs
    ├── test_oncall.py       # 22 tests
    └── Dockerfile           # Multi-stage build
```
