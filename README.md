---
title: OnCallEnv
emoji: "\U0001F6A8"
sdk: docker
app_port: 8000
tags:
  - openenv
  - reinforcement-learning
  - incident-response
---

# OnCallEnv -- Incident Response Command Center

An OpenEnv RL environment that simulates production incident response. An AI agent acts as an **on-call engineer** -- triaging alerts, diagnosing root causes through investigation, remediating failures, and documenting incidents.

**Key insight:** Frontier LLMs (GPT-5.2) act before investigating 97% of the time in incident response ([OpenSec, arXiv 2601.21083](https://arxiv.org/abs/2601.21083)). OnCallEnv is designed to test and train this calibration gap -- can an agent learn to *investigate before it acts*?

## Why Incident Response?

On-call engineering is a **universal real-world task** that every engineering organization performs daily. Unlike game environments, incident response requires:
- Multi-step reasoning over noisy, partial information
- Distinguishing signal from noise (red herrings, coincidental events)
- Investigation before action (the agent must discover what's wrong, not guess)
- Balancing speed vs. thoroughness under time pressure
- Clear documentation and communication

Each scenario is a **digital twin** of a real production incident with planted root causes and deterministic grading.

## Environment Design

OnCallEnv implements several mechanisms to ensure genuine difficulty:

- **Partial observability**: Service statuses are hidden ("unknown") until the agent actively investigates via `query_logs` or `check_metrics`
- **Anonymous service labels**: Alerts and service lists use anonymous identifiers (service-A, service-B). The agent must investigate to discover which real service each label represents
- **Clue-based logs**: Log messages contain raw technical output (stack traces, error codes, IP addresses) -- not pre-interpreted diagnoses. The agent must reason from evidence
- **Ambiguous remediation feedback**: Fix actions don't reveal whether they worked. The agent must verify via follow-up investigation
- **Dynamic degradation**: Unhealthy services get worse each step. Fixing the root cause triggers cascading recovery through the dependency graph
- **Per-step reward signals**: 6 intermediate signals in `observation.metadata["reward_signals"]` enable GRPO/RL training with dense feedback

## How It Works

```
                    THE RL LOOP
                    ==========

    +---------+     action      +--------------+     loads      +----------+
    |         | --------------> |              | <------------- | Scenario |
    |  Agent  |                 |  OnCallEnv   |                |  (JSON)  |
    |  (LLM)  | <------------- |  (Server)    |                +----------+
    |         |  observation    |              |
    +---------+  + reward       +------+-------+
                                       |
                                 +-----+------+
                                 |  Grader    |
                                 | (6 components)
                                 +------------+

    Agent Flow:
    1. See anonymized alerts (service-A has errors, service-B has latency...)
    2. Investigate: query_logs(service-A) -> discovers it's payment-api
    3. Read logs: HikariPool timeout, connection refused to 10.0.3.47:5432
    4. Interpret: "connection pool exhaustion on payment-api"
    5. Check dependencies + recent deploys (revealed after investigation)
    6. Remediate: rollback_deploy(payment-api, v2.4.0)
    7. Verify: check_metrics(payment-api, error_rate) -> confirm recovery
    8. Document + resolve
```

## Quick Start

```bash
# Install
uv sync  # or: pip install -e .

# Run server
uvicorn oncall_env.server.app:app --host 0.0.0.0 --port 8000

# Run tests
pytest server/test_oncall.py -v

# Run LLM baseline (requires API_BASE_URL, MODEL_NAME, HF_TOKEN)
python inference.py
```

## Docker

```bash
docker build -f server/Dockerfile -t oncall-env .
docker run -p 8000:8000 oncall-env
```

## Tasks (4 Difficulty Levels, 48 Scenarios)

| Task | Difficulty | Scenarios | Description |
|------|-----------|-----------|-------------|
| 1 | Easy | 12 | Alert triage + single-service diagnosis (e.g., connection pool exhaustion) |
| 2 | Medium | 12 | Cascading failure diagnosis across 2-3 services with dependency tracing |
| 3 | Hard | 12 | Full incident resolution with config-driven failures and misleading symptoms |
| 4 | Expert | 12 | Multi-service outage with red herrings (coincidental deploys, unrelated alerts) |

**Root cause types** (11 types across 48 scenarios): connection pool exhaustion, memory leak/OOM, replication lag, deadlock storm, CPU spin loop, GC pause storm, DNS resolution failure, TLS certificate expiry, load balancer misconfiguration, bad configuration rollout, dependency version mismatch.

## Action Space (13 Actions)

| Action | Category | Parameters |
|--------|----------|------------|
| `query_logs` | Investigation | `service`, `level` (opt), `time_range` (opt) |
| `check_metrics` | Investigation | `service`, `metric_name` |
| `view_dependencies` | Investigation | `service_name` |
| `acknowledge_alert` | Triage | `alert_id` |
| `silence_alert` | Triage | `alert_id` |
| `restart_service` | Remediation | `service_name` |
| `scale_service` | Remediation | `service_name`, `replicas` |
| `rollback_deploy` | Remediation | `service_name`, `target_version` |
| `update_config` | Remediation | `service_name`, `config_key`, `config_value` |
| `set_severity` | Documentation | `level` (SEV1-SEV4) |
| `write_summary` | Documentation | `text` |
| `escalate` | Communication | `team` |
| `resolve_incident` | Resolution | `resolution_note` |

## Observation Space

Each step returns:
- **alerts**: Active alerts with anonymous service labels, severity, and redacted messages (exact metric values hidden)
- **services**: Anonymous labels with "unknown" status until investigated; real names + full metrics revealed after investigation
- **recent_deployments**: Hidden until the relevant service is investigated
- **log_results**: Raw technical log entries (stack traces, error codes, IP addresses) from last `query_logs`
- **metric_results**: Time-series data from last `check_metrics` (available metric names listed on failed lookup)
- **dependency_graph**: Service dependency list from last `view_dependencies` (internal services anonymized until investigated)
- **incident_timeline**: Chronological event log of agent actions
- **current_severity**: Currently set severity level
- **message**: Feedback on last action (ambiguous for remediations -- no indication of success/failure)
- **done**: Whether episode is complete
- **reward**: Final reward (0.0-1.0) when done
- **metadata.reward_signals**: 6 per-step training signals (triage progress, investigation depth, premature action penalty, etc.)

## Reward Function

```
reward = triage       x weight    (alert ack/silence, severity)
       + diagnostic   x weight    (investigated expected services?)
       + root_cause   x weight    (correct service + keywords in summary, gated behind investigation)
       + remediation  x weight    (correct fix, EGAR bonus, blast radius penalty)
       + efficiency   x weight    (step count, investigation depth, duplicate penalty)
       + documentation x weight   (summary quality with scenario-specific keywords)
```

Weights shift per difficulty level:

| Component | Easy | Medium | Hard | Expert |
|-----------|------|--------|------|--------|
| Triage | 0.30 | 0.20 | 0.10 | 0.05 |
| Diagnostic | 0.20 | 0.25 | 0.25 | 0.20 |
| Root Cause | 0.15 | 0.20 | 0.30 | 0.35 |
| Remediation | 0.15 | 0.20 | 0.20 | 0.20 |
| Efficiency | 0.10 | 0.10 | 0.10 | 0.10 |
| Documentation | 0.10 | 0.05 | 0.05 | 0.10 |

**Key grading features:**
- **EGAR (Evidence-Gated Action Rate)**: Bonus for investigating a service BEFORE remediating it
- **Blast radius penalty**: Each wrong remediation target costs points; shotgunning restarts is heavily penalized
- **Investigation gate**: Root cause scoring requires actually investigating the root cause service (can't just guess from alert text)
- **Premature resolution penalty**: Resolving without any investigation is penalized
- **Anti-shotgun check**: Naming too many services as root cause in the summary reduces score

## Evaluation: GPT-5.2 Across 6 Rounds of Environment Hardening

We iteratively hardened the environment and re-evaluated GPT-5.2 (zero-shot) after each change. Each row shows what was added and how scores shifted.

### Round-by-Round Score Progression

| Round | What Changed | Task 1 | Task 2 | Task 3 | Task 4 |
|-------|-------------|--------|--------|--------|--------|
| **R1** | Baseline (full visibility, diagnostic logs, binary feedback) | **0.89** | 0.67 | **0.75** | **0.74** |
| **R2** | + Partial observability (services hidden until investigated) | 0.83 | 0.67 | 0.68 | 0.55 |
| **R3** | + Clue-based logs (raw stack traces, not diagnoses) | 0.83 | 0.60 | 0.68 | 0.64 |
| **R4** | + Service anonymization (alerts show service-A, not payment-api) | 0.83 | 0.63 | 0.48 | 0.40 |
| **R5** | + Description redaction + stricter grading (EGAR, blast radius) | 0.79 | 0.60 | 0.68 | 0.55 |
| **R6** | + Full alias system + ambiguous remediation feedback | **0.73** | **0.51** | **0.53** | **0.55** |
| | **Total drop** | **-0.16** | **-0.16** | **-0.22** | **-0.19** |

### What Each Round Exposed

**R1 → R2 (Partial observability):** Scores dropped 0.06-0.19. The model could no longer see which services were degraded from the first observation. Had to actively investigate. Biggest impact on Expert task (-0.19) where 9 services made blind investigation harder.

**R2 → R4 (Anonymization):** Task 3 dropped from 0.68 to 0.48, Task 4 from 0.55 to 0.40. Without real service names, the model couldn't jump straight to the root cause. Had to discover identities through investigation — adding 5-10 steps per episode.

**R5 → R6 (Ambiguous feedback):** With remediation responses no longer saying "Status healthy" vs "issues persist", the model lost its brute-force shortcut. Previously it could restart every service until one returned "healthy." Now it must verify via `check_metrics` after each remediation — costing extra steps and exposing restart-loop behavior.

### What GPT-5.2 Gets Right

- **Root cause identification**: 4/4 tasks — correctly identified payment-api pool exhaustion (T1), inventory-service memory leak (T2), order-service config change (T3), DNS zone corruption (T4)
- **Alias adaptation**: Learns the alias system within 1-2 queries per episode
- **Log interpretation**: Correctly interprets HikariPool timeouts, OOM stack traces, DNS SERVFAIL from raw log format
- **Dependency tracing**: Uses `view_dependencies` to trace failure chains (api-gateway → order-service → inventory-service)

### What GPT-5.2 Gets Wrong

| Failure Mode | Observed In | Example |
|---|---|---|
| **Restart loops** | All tasks | Restarted config-service 5x in T4, order-service 4x in T2, api-gateway 3x in T3 — each time getting ambiguous feedback, never changing strategy |
| **Symptom chasing** | T2, T3, T4 | After correctly fixing root cause, spent 15-27 additional steps trying to restart downstream services that can't recover on their own |
| **Can't break failed strategy** | T2, T4 | After 5+ failed restarts on the same service, continued restarting instead of trying escalation or a different approach |
| **Metric name guessing** | All tasks | Tried `db_pool_waiting`, `5xx_error_rate`, `container_memory_usage_bytes` — none matched actual metric keys. Required hint listing available metrics |
| **Premature remediation** | T1, T3 | Applied rollback within 2 steps of first log query, before checking metrics or dependencies |
| **Investigation-to-remediation ratio** | All tasks | Averaged 1.2:1 (environment rewards 2.0:1). Model remediates too eagerly relative to how much it investigates |

### The Calibration Gap

GPT-5.2 identifies root causes correctly in every task but cannot resolve incidents efficiently. This mirrors the finding from OpenSec (arXiv 2601.21083) where GPT-5.2 achieved 100% containment but 97% false positive rate — **the gap between knowing what's wrong and acting on it without collateral damage.**

This gap is precisely what RL training can close: teaching the model to investigate 3+ services before remediating, verify fixes via metrics instead of restarting blindly, and stop chasing downstream symptoms after the root cause is addressed.

## Per-Step Reward Signals (for RL Training)

Every observation includes `metadata.reward_signals` with 6 training signals:

| Signal | Description | Range |
|--------|-------------|-------|
| `oncall.triage_progress` | Fraction of critical alerts acknowledged | 0.0-1.0 |
| `oncall.investigation_depth` | Fraction of expected diagnostics completed | 0.0-1.0 |
| `oncall.premature_action` | Penalty if remediated before investigating | 0.0 or -0.5 |
| `oncall.severity_set` | Whether severity has been set | 0.0 or 1.0 |
| `oncall.summary_written` | Whether a summary has been written | 0.0 or 1.0 |
| `oncall.resolved` | Whether incident has been resolved | 0.0 or 1.0 |

These enable GRPO training with dense per-step feedback via TRL's `GRPOTrainer`.

## API Endpoints

### OpenEnv Standard (auto-generated)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start new episode (params: `task_id`, `scenario_idx`) |
| `/step` | POST | Execute an action |
| `/state` | GET | Query current episode state |
| `/health` | GET | Health check |
| `/schema` | GET | JSON schemas for Action/Observation/State |
| `/metadata` | GET | Environment info |
| `/docs` | GET | OpenAPI documentation |
| `/web` | GET | Interactive Gradio UI |
| `/ws` | WebSocket | Persistent session |

### Custom Hackathon Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/tasks` | GET | List all 4 tasks + action JSON schema |
| `/grader` | POST | Grade a completed episode |
| `/baseline` | POST | Run heuristic baseline agent, return scores |

## Training with GRPO (PyTorch + TRL)

OnCallEnv includes a GRPO training script that teaches a small LLM to become a better on-call engineer through reinforcement learning.

```bash
# Install training dependencies
pip install -e ".[train]"

# Run training (uses env directly, no server needed)
python train.py --model Qwen/Qwen2.5-0.5B-Instruct --episodes 200 --epochs 3

# Quick test (15-20 min on CPU)
python train.py --episodes 20 --epochs 1 --batch-size 2

# Eval-only (verify pipeline works, ~5 min)
python train.py --eval-only
```

**How it works:**
1. Generates training prompts by resetting the env with random tasks/scenarios
2. Model generates action completions for each prompt
3. Actions are executed in the environment, scored by the 6-component grader
4. GRPO uses reward comparisons across completions to update model weights
5. Per-step reward signals (`metadata.reward_signals`) provide dense feedback

Auto-detects CUDA GPU, Apple Silicon (MPS), or CPU.

## Architecture

```
oncall_env/
├── __init__.py              # Package exports
├── models.py                # Action, Observation, State (typed Pydantic models)
├── client.py                # EnvClient subclass (WebSocket)
├── inference.py             # LLM-based baseline agent (OpenAI-compatible)
├── train.py                 # GRPO training script (PyTorch + TRL)
├── generate_scenarios.py    # Scenario generator (48 scenarios, 11 root cause types)
├── openenv.yaml             # OpenEnv manifest
├── pyproject.toml           # Dependencies
├── scenarios/               # 48 pre-generated incident scenarios
│   ├── task1_easy/          # 12 scenarios
│   ├── task2_medium/        # 12 scenarios
│   ├── task3_hard/          # 12 scenarios
│   └── task4_expert/        # 12 scenarios
└── server/
    ├── app.py               # FastAPI app + custom endpoints (/tasks, /baseline, /grader)
    ├── environment.py       # Core environment: reset/step/state + alias system + simulation
    ├── graders.py           # 6-component deterministic grading + EGAR + blast radius
    ├── rubric.py            # TrajectoryRubric for GRPO training integration
    ├── simulator.py         # Dynamic degradation + cascading recovery engine
    ├── scenario_loader.py   # JSON scenario loading
    ├── test_oncall.py       # 43 tests (all passing)
    └── Dockerfile           # Multi-stage Docker build
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_BASE_URL` | LLM API endpoint | `https://api.openai.com/v1` |
| `MODEL_NAME` | Model to use for inference | `gpt-4o-mini` |
| `HF_TOKEN` | Hugging Face token (fallback API key) | - |
| `ONCALL_ENV_URL` | Environment WebSocket URL | `ws://localhost:8000` |

## Rubric System (GRPO Training)

OnCallEnv integrates with the OpenEnv `TrajectoryRubric` system:

```python
from oncall_env.client import OnCallEnvClient
from oncall_env.models import OnCallAction

async with OnCallEnvClient(base_url="wss://your-space.hf.space") as env:
    result = await env.reset(task_id=1, scenario_idx=0)
    
    # Agent investigates
    result = await env.step(OnCallAction(
        action_type="query_logs",
        params={"service": "service-A"}  # alias -- resolved internally
    ))
    
    # Per-step reward signals available for training
    signals = result.observation.metadata["reward_signals"]
    
    # After episode: compute per-step credit assignment
    step_rewards = env._rubric.compute_step_rewards()
```

## Support

- **Email**: help_openenvhackathon@scaler.com
- **Discord**: https://discord.gg/Dedhy5pkWD
- **OpenEnv Framework**: https://github.com/meta-pytorch/OpenEnv
- **OpenEnv Course**: https://github.com/raun/openenv-course
