# OnCallEnv — Complete Flaws, Gaps & What Must Be Done

> **Purpose:** Every single issue found across 7 research agents, 8 scenario files, full codebase audit, OpenEnv framework comparison, OpenSec reference study, and HF Spaces deployment research.
> **Team:** Diff Maker | **Deadline:** April 8, 11:59 PM IST | **Goal:** First Prize ($10,000)

---

## HOW THIS DOCUMENT IS ORGANIZED

1. **Spec Compliance Failures** — Things that will FAIL automated validation
2. **Framework Integration Gaps** — Things gold-standard envs do that we don't
3. **Simulation Realism Flaws** — Why the env feels fake, not real
4. **Scenario Design Problems** — Issues in the 8 scenario JSON files
5. **Grading System Weaknesses** — How the reward logic can be gamed or is wrong
6. **Code Quality Issues** — Bugs, type errors, inconsistencies
7. **Deployment Risks** — Things that will break on HF Spaces
8. **Strategic Gaps** — What separates us from first prize
9. **Lessons From OpenSec** — What the best incident-response env does differently
10. **Complete Fix Priority Matrix** — Everything ranked by effort and impact

---

## 1. SPEC COMPLIANCE FAILURES (Will Fail Automated Validation)

### 1.1 Missing `/tasks` Endpoint
The hackathon requires `GET /tasks` that returns the list of tasks and the action schema (fields required for a step action). `create_app()` does NOT auto-generate this. We must add it manually.

**What it should return:**
```json
{
  "tasks": [
    {"task_id": 1, "name": "Alert Triage", "difficulty": "easy", "description": "..."},
    {"task_id": 2, "name": "Root Cause Diagnosis", "difficulty": "medium", "description": "..."},
    {"task_id": 3, "name": "Full Incident Resolution", "difficulty": "hard", "description": "..."},
    {"task_id": 4, "name": "Cascading Failure with Red Herrings", "difficulty": "expert", "description": "..."}
  ],
  "action_schema": { ... OnCallAction JSON schema ... }
}
```

### 1.2 Missing `/baseline` Endpoint
Must trigger the inference script and return baseline scores for all tasks.

### 1.3 Missing `/grader` Endpoint
Must return the grader score after an episode is completed.

### 1.4 Wrong Filename: `baseline.py` Must Be `inference.py`
The hackathon explicitly requires `inference.py` in the root directory. Ours is called `baseline.py`.

### 1.5 Wrong Environment Variables
Hackathon requires:
- `API_BASE_URL` (we use `OPENAI_API_BASE`)
- `MODEL_NAME` (we use `OPENAI_MODEL`)
- `HF_TOKEN` (we don't use this at all)
Must use OpenAI Client for all LLM calls (we do, but with wrong var names).

### 1.6 HF Spaces README Frontmatter Missing
HF Spaces Docker deployments need YAML frontmatter in README.md:
```yaml
---
title: OnCallEnv
emoji: 🚨
sdk: docker
app_port: 8000
tags:
  - openenv
---
```
Our README has no frontmatter. Without `app_port: 8000`, HF Spaces defaults to 7860, and the Space will be stuck "Starting" forever.

### 1.7 Not Deployed to HF Spaces
The pre-submission check pings the HF Space URL. We haven't deployed yet.

---

## 2. FRAMEWORK INTEGRATION GAPS (Gold-Standard vs Our Code)

### 2.1 No Rubric System Integration
**Gold standard (chess_env, textarena_env):**
```python
class ChessEnvironment(Environment):
    def __init__(self):
        super().__init__(rubric=ChessWinLossRubric(gamma=0.99))
    def step(self, action):
        obs = ...
        reward = self._apply_rubric(action, obs)
```

**Our code:**
```python
class OnCallEnvironment(Environment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # No rubric!
    def step(self, action):
        reward = grade_episode(...)  # Standalone function, not framework Rubric
```

The GRPO training pipeline calls `env.rubric.named_rubrics()` to introspect reward components. Without Rubric integration, our environment cannot be used for RL training with the standard OpenEnv toolchain.

### 2.2 No Per-Step Reward Signals
**Gold standard:**
```python
observation.metadata["reward_signals"] = {
    "wordle.greens": 0.6,
    "wordle.yellows": 0.4,
    "wordle.correct": 0.0,
}
```

**Our code:** `reward = None` for all intermediate steps. Only a single scalar at episode end.

GRPO needs per-step signals to compute advantages. Without them, training gives zero gradient for all steps except the last.

**What we should provide on every step:**
```python
observation.metadata["reward_signals"] = {
    "oncall.triage_progress": 0.4,       # % critical alerts acknowledged
    "oncall.investigation_depth": 0.3,    # % expected diagnostics completed
    "oncall.premature_action": 0.0,       # penalty if remediated without investigating
    "oncall.correct_service_focus": 0.5,  # is agent looking at right service?
}
```

### 2.3 No `get_metadata()` Override
The `/metadata` endpoint returns generic info. Should return environment name, description, version, task count, author.

### 2.4 No GRPO Training Script
The ultimate proof the environment works for RL training. OpenEnv Module 5 shows GRPO on Wordle. We have nothing equivalent.

### 2.5 No Web Interface Enabled
The Gradio-based `/web` UI lets judges interactively play through incidents. Set `ENABLE_WEB_INTERFACE=true` in Docker or app.py. Free demo capability.

### 2.6 Zero PyTorch Usage
This is a **Meta PyTorch** hackathon. No `import torch` anywhere. Must add:
- PyTorch-based reward model or LLM judge
- Or GRPO training script using TRL (which uses PyTorch)
- Or TorchRL TensorDict observation encoding

---

## 3. SIMULATION REALISM FLAWS (Why It Feels Fake)

### 3.1 No Time-Based Degradation
Real incidents get WORSE over time. A DB connection pool leak doesn't stay at 12.3% error rate — it climbs to 50%, 80%, 100%. Our services are frozen at their initial JSON values forever.

**Fix:** Each `step()` should advance a simulated clock. Unhealthy services degrade further per tick:
```python
def _tick_degradation(self):
    for svc in self._services:
        if svc["status"] == "degraded":
            svc["error_rate"] = min(svc["error_rate"] * 1.15, 100.0)
            svc["latency_ms"] = svc["latency_ms"] * 1.1
```

### 3.2 No Cascading Recovery
When you fix the root cause, downstream services should RECOVER. Currently:
- Fix inventory-service (Task 2) → inventory-service becomes healthy
- order-service stays at 40% error rate FOREVER
- api-gateway stays at 25% error rate FOREVER

**Fix:** After valid remediation, propagate recovery through the dependency graph.

### 3.3 No New Alerts During Episode
Real incidents generate NEW alerts as the situation evolves. Our alert list is static from `reset()`.

**Fix:** On each tick, check if degraded services crossed thresholds → generate new alerts dynamically.

### 3.4 No Stochasticity / Seeded Randomness
Every run is 100% identical. The `seed` parameter is accepted but never used. `random` is never imported. This means:
- No generalization testing (agent can memorize 8 scenarios)
- No noise in metrics (unrealistic)
- No jitter in service behavior

**Fix:** Use `seed` to initialize a `random.Random` instance. Add noise to metrics, randomize non-critical alert timing.

### 3.5 Observation Leaks Full State Every Step
Every `step()` returns ALL alerts and ALL service statuses with full detail (latency, error_rate, cpu, memory). The agent has perfect observability without querying anything. This makes `check_metrics` and `query_logs` actions pointless — the agent already knows everything from the observation.

**Fix:** Return only alert summaries and basic service status (name + status). Detailed metrics, logs, and dependencies should ONLY be available through explicit diagnostic actions.

### 3.6 Remediation is Binary (Fully Fixes or Does Nothing)
Every valid remediation sets `status = "healthy"` and `error_rate = 0.0` instantly. In reality:
- A restart temporarily fixes a memory leak but it returns
- A rollback takes time to propagate
- A config change may require service restart to take effect

**Fix:** Add gradual recovery (error_rate decreases over 2-3 steps after remediation).

### 3.7 No Concept of "Wrong Action Consequences"
Restarting a healthy service has zero consequence. Scaling a service that doesn't need scaling has zero consequence. In production, unnecessary restarts cause brief downtime and dropped connections.

**Fix:** Penalize restarting healthy services (brief degradation). The grader already partially handles this but the environment state doesn't reflect it.

---

## 4. SCENARIO DESIGN PROBLEMS

### 4.1 RED HERRING Labels in Alert Messages (CRITICAL)
**task4_expert/scenario_001.json:**
```json
"message": "Model inference latency increased 3x (RED HERRING: scheduled model update)"
"message": "Search latency elevated to 600ms (RED HERRING: elasticsearch reindexing)"
```

**task4_expert/scenario_002.json:**
```json
"message": "External latency elevated to 850ms -- CDN cache purge in progress (RED HERRING: scheduled purge)"
"message": "CPU usage 91% on pipeline worker nodes -- batch ETL job running (RED HERRING: nightly batch)"
```

This completely defeats the purpose of expert difficulty. Remove ALL "(RED HERRING: ...)" text.

### 4.2 Scenario Descriptions Contain the Answer
**task3_hard/scenario_001.json:**
> "The root cause is subtle -- a recent config change combined with a traffic spike triggered a memory leak."

**task3_hard/scenario_002.json:**
> "a recent config change disabled cache eviction, leading to OOM and a cascade of dependent service failures"

**task4_expert/scenario_001.json:**
> "Several unrelated events coincide with the real failure, creating red herrings."

**task4_expert/scenario_002.json:**
> "The real failure is a bad intermediate CA certificate pushed to the Istio/Envoy service mesh at 21:00, silently breaking all mTLS connections between internal services."

The description is part of the initial observation! The agent literally reads the answer before investigating.

**Fix:** Descriptions should only describe SYMPTOMS: "Multiple services degraded. Error rates elevated across the platform. Investigate and resolve."

### 4.3 Only 2 Scenarios Per Task (8 Total)
OpenSec has 220 scenarios. Even moderate benchmarks have 50+. With 2 per difficulty, any agent can memorize them. There's no generalization test.

**Fix:** Generate at least 5 per task (20 total). Even better: build a scenario template system that generates unlimited variations.

### 4.4 Inconsistent Service Counts Across Scenarios
- Task 1 scenario_002: 3 services
- Task 4 scenario_002: 9 services

This is fine for difficulty variation, but the GRPO training pipeline may struggle with variable observation sizes.

### 4.5 Inconsistent Deploy Schema
task4_expert/scenario_002.json has a `notes` field in recent_deployments that no other scenario has. Inconsistent schema.

### 4.6 Root Cause Keywords Are Too Easy to Extract
All scenario JSON files have a `root_cause.keywords` array. Anyone reading the scenario files (including an LLM analyzing the repo) can extract these keywords and stuff them into summaries. The grader rewards keyword presence.

### 4.7 Some Scenarios Have Self-Resolving Issues
task1_easy/scenario_002 describes a Redis connection issue that "Self-recovered." But the environment has no concept of self-resolution — services don't improve on their own.

### 4.8 No Scenario Involves: Disk Full, Network Partition, Certificate Expiry (wait — task4_002 does), Rate Limiting, Third-Party API Failure, Kubernetes Pod Eviction
The incident variety is limited to: bad deploy, memory leak, connection pool exhaustion, config change, DNS corruption, cache OOM, Redis timeout, TLS cert failure. More variety would make the benchmark more robust.

---

## 5. GRADING SYSTEM WEAKNESSES

### 5.1 Root Cause Grading is Keyword Stuffing
`_grade_root_cause()` uses `if kw in summary_lower`. An agent can write: "memory leak connection pool oom database deployment v2.5.1 inventory dns zone corrupted" and score 1.0 without understanding anything.

**Fix:** Use LLM-as-judge (OpenEnv has `LLMJudge` rubric built-in) for summary evaluation. Or require structured root cause format: `{"service": "X", "cause": "Y", "evidence": "Z"}`.

### 5.2 No "Investigation Before Remediation" Metric
The grader doesn't track whether the agent investigated before acting. An agent that immediately restarts every service scores the same on the remediation component as one that carefully diagnosed first.

OpenSec's EGAR (Evidence-Gated Action Rate) is exactly this metric: "Did you fetch evidence about the target before acting on it?"

**Fix:** Add an EGAR-style component:
```python
def _grade_investigation_before_action(scenario, actions):
    """Score: did agent query logs/metrics for a service BEFORE restarting/rolling back that service?"""
    remediation_actions = [a for a in actions if a["action_type"] in REMEDIATION_TYPES]
    gated_count = 0
    for ra in remediation_actions:
        target_svc = ra["params"].get("service_name", "")
        # Check if any prior action queried this service
        prior = [a for a in actions if a["step"] < ra["step"]]
        investigated = any(
            a["action_type"] in ("query_logs", "check_metrics") and
            a["params"].get("service", "") == target_svc
            for a in prior
        )
        if investigated:
            gated_count += 1
    return gated_count / max(len(remediation_actions), 1)
```

### 5.3 No Blast Radius Tracking
The grader penalizes restarting healthy services (-0.1 per instance) but doesn't track the overall "blast radius" — how many wrong actions were taken vs correct ones. OpenSec measures this as `FP_actions / correct_actions`.

### 5.4 No Premature Resolution Penalty
An agent that calls `resolve_incident` in step 1 without any investigation gets:
- triage: 0 (nothing acknowledged)
- diagnostic: 0 (nothing queried)
- root_cause: 0 (no summary)
- remediation: 0.2 (bonus for resolving)
- efficiency: 1.0 (only 1 step!)
- documentation: 0.0

Total: ~0.12. The efficiency score rewards premature resolution. This is wrong.

**Fix:** Efficiency should require minimum useful actions (at least 1 diagnostic + 1 remediation) to score above 0.

### 5.5 Efficiency Scoring is Too Coarse
Four buckets: <=10 (1.0), <=15 (0.7), <=20 (0.4), >20 (0.2). An agent taking 10 redundant log queries scores the same as 10 precise, targeted actions. No distinction between useful and wasted actions.

**Fix:** Track "useful actions" (actions that discovered new information or changed state) vs "wasted actions" (duplicate queries, actions on wrong services).

### 5.6 Grading Weights Don't Validate Summing to 1.0
Each scenario provides custom rubric weights. While all 8 current scenarios sum to 1.0, there's no validation. A malformed scenario could produce max reward < 1.0.

### 5.7 Triage Scoring Uses Fractional `total_checks`
`total_checks += 0.5` for info alerts makes the denominator unpredictable and the scoring non-intuitive. Should use integer check counts with weighted scoring.

---

## 6. CODE QUALITY ISSUES

### 6.1 Sub-Models Inherit from `Action` Instead of `BaseModel`
`Alert`, `ServiceStatus`, `LogEntry`, `Deploy`, `Event` all inherit from `Action`. These are data models, not agent actions. `Action` has `extra="forbid"` and a `metadata` field. Sub-models override with `extra="allow"`, contradicting the parent.

**Fix:** Change to `class Alert(BaseModel):` with `ConfigDict(extra="forbid")`.

### 6.2 Typed Sub-Models Are Defined But Never Used
`OnCallObservation` uses `list[dict]` for alerts, services, deployments. The typed models `Alert`, `ServiceStatus`, etc. are exported in `__init__.py` but never used in environment.py or graders.py.

**Fix:** Change observation types to `List[Alert]`, `List[ServiceStatus]`, etc.

### 6.3 `check_metrics` Ignores `metric_name` Parameter
The fuzzy matching logic returns the first service prefix match, ignoring `metric_name`. If you query `check_metrics(service="order-service", metric_name="error_rate")`, you might get `order-service:memory_mb` depending on dict iteration order.

**Fix:** Build the exact key `f"{service}:{metric_name}"` and try exact match first. Only fall back to prefix if exact fails.

### 6.4 Inconsistent Parameter Naming: `service` vs `service_name`
- `query_logs`, `check_metrics`: use `params.get("service")`
- `restart_service`, `scale_service`, `rollback_deploy`, `update_config`, `view_dependencies`: use `params.get("service_name")`

Confusing for agents. Should standardize.

### 6.5 `scale_service` Ignores `replicas` Parameter
The agent specifies `replicas=10` but the environment never uses this value. The service either gets fixed or doesn't based on `valid_remediations`.

### 6.6 `query_logs` Ignores `time_range` Parameter
Accepted but never used for filtering. Agent gets same results regardless.

### 6.7 Baseline Uses Sync OpenAI in Async Function
`run_baseline` is `async def` but calls `client_llm.chat.completions.create()` synchronously, blocking the event loop.

### 6.8 `_compute_final_reward()` Recalculated on Every Post-Done Step
After resolution, every subsequent `step()` recomputes the reward. Should cache.

### 6.9 `_handle_restart_service` Reduces Latency with Magic Formula
`max(svc.get("latency_ms", 50) * 0.3, 15)` — arbitrary. Only restart does this; rollback and update_config don't touch latency. Inconsistent.

### 6.10 `OnCallState` Missing from `__init__.py` Exports
`__init__.py` exports `OnCallAction`, `OnCallObservation`, `Alert`, `ServiceStatus`, `LogEntry`, `Deploy`, `Event` but NOT `OnCallState`. The client imports it directly from `models.py`.

### 6.11 No `pyproject.toml` Test Dependencies
No `[project.optional-dependencies]` section with `pytest`. Test suite can't be run without manually installing pytest.

### 6.12 No Input Validation on Action Params
Action handlers use `params.get("service", "")` with empty string default. No validation that required params are present. Agent sending `restart_service` with empty params gets "Service not found" instead of "Missing required parameter: service_name".

---

## 7. DEPLOYMENT RISKS

### 7.1 HF Spaces Port Mismatch Risk
Our Dockerfile uses port 8000. HF Spaces defaults to 7860. Without `app_port: 8000` in README frontmatter, the Space will never start.

### 7.2 Base Image Availability
`ghcr.io/meta-pytorch/openenv-base:latest` is publicly available (confirmed via echo-env Space), but if it changes or gets pulled, our build breaks.

### 7.3 WebSocket Must Use `wss://` on HF Spaces
HF Spaces proxies via HTTPS. Plain `ws://` will fail with 404. Our baseline uses `ws://localhost:8000` which is fine locally but client connections from outside must use `wss://`.

### 7.4 Free Tier Sleep Behavior
Spaces sleep after inactivity. Cold start can take minutes. The automated validator may timeout if the Space is sleeping.

**Fix:** Use `startup_duration_timeout: 10m` in README frontmatter. Consider paying for persistent hardware during judging window.

### 7.5 Outbound Networking Limited
HF Spaces only allows outbound on ports 80, 443, 8080. If baseline calls an LLM API on a non-standard port, it will be blocked.

### 7.6 Docker Health Check Uses `curl`
Our Dockerfile uses `curl -f http://localhost:8000/health`. Some base images don't have curl. The gold standard uses Python: `python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"`.

### 7.7 No `.dockerignore`
Without `.dockerignore`, the Docker build copies `.git/`, `__pycache__/`, `uv.lock`, `*.egg-info/` etc. into the image. Wastes space and slows build.

---

## 8. STRATEGIC GAPS (What Separates Us From First Prize)

### 8.1 No Unique Insight / Novel Metric
OpenSec introduced EGAR (Evidence-Gated Action Rate) — a genuinely new metric that revealed frontier models have 90-97% false positive rates in incident response. This finding alone made the project noteworthy.

**What we need:** A novel metric or insight. Suggestion: **"Diagnostic Depth Before Remediation" (DDBR)** — measures how many diagnostic steps the agent takes before its first remediation action. Reveals whether agents investigate or guess.

### 8.2 No Published Dataset
OpenSec published 220 scenarios + 320 baseline traces on HuggingFace Datasets. This makes the benchmark reusable by the community.

**What we should do:** Publish our scenarios as a HuggingFace Dataset. Even 20 scenarios with ground truth is a contribution.

### 8.3 No Research Framing
OpenSec has an arXiv paper. Obviously we can't write one in time, but our README should frame the environment as addressing a research question: "Can LLMs learn to investigate before acting in production incident response?"

### 8.4 No Baseline Comparison Across Models
OpenSec benchmarked GPT-5.2, Sonnet 4.5, Gemini 3, DeepSeek 3.2. Showing how different models perform on our env demonstrates its discriminative power.

**What we should show:**
```
| Model         | Task 1 | Task 2 | Task 3 | Task 4 |
|---------------|--------|--------|--------|--------|
| Random        | 0.05   | 0.02   | 0.01   | 0.01   |
| Heuristic     | 0.45   | 0.25   | 0.10   | 0.05   |
| GPT-4o-mini   | 0.65   | 0.45   | 0.25   | 0.10   |
```

### 8.5 No Interactive Demo / Playground
OpenSec has a "playground" for interactive trace visualization. We should enable the Gradio web interface (`ENABLE_WEB_INTERFACE=true`).

### 8.6 No Architecture Diagram
The README describes the microservice topology in text. A visual diagram (ASCII or image) would make it immediately comprehensible.

---

## 9. LESSONS FROM OPENSEC (The Gold Standard for This Domain)

### What OpenSec Does That We Don't

| Feature | OpenSec | OnCallEnv | Gap |
|---|---|---|---|
| **Dynamic simulation** | Attacker state machine advances in real-time | Static JSON lookup | Critical |
| **Partial observability** | Must query to discover evidence | Full state leaked in every observation | Critical |
| **Novel metric (EGAR)** | "Did you investigate before acting?" | No equivalent | High |
| **Blast radius tracking** | FP_actions / correct_actions | Only -0.1 penalty per wrong restart | High |
| **Scenario count** | 220 (160 train + 60 eval) | 8 | High |
| **Step budget pressure** | Fixed budget forces prioritization | MAX_STEPS=30 exists but no temporal pressure | Medium |
| **Adversarial evidence** | Prompt injection in logs/alerts | Red herrings (but labeled!) | Medium |
| **Deterministic replay** | Cached via (scenario_id, step, state, action_hash) | Deterministic by being static | Low |
| **Execution-based scoring** | Scores tool calls that change world state | Scores everything including text | Medium |
| **Published dataset** | 220 scenarios on HuggingFace | Not published | Medium |
| **Research paper** | arXiv 2601.21083 | None | Low |

### Key Insight From OpenSec Paper

> "Traditional benchmarks conflate action *execution* with *correct* execution. A model that acts on every scenario gets high containment rates but would be catastrophic in production."

This is directly applicable to our env. An agent that restarts every service scores non-zero on remediation. We need to heavily penalize false positive remediations and reward investigation.

---

## 10. COMPLETE FIX PRIORITY MATRIX

### Tier 1: MUST DO (Automated Validation Will Fail Without These)
| # | Fix | Effort | File(s) |
|---|---|---|---|
| 1 | Add `/tasks` endpoint | 1 hr | server/app.py |
| 2 | Add `/baseline` endpoint | 1.5 hr | server/app.py |
| 3 | Add `/grader` endpoint | 1 hr | server/app.py |
| 4 | Rename `baseline.py` → `inference.py` | 10 min | root |
| 5 | Fix env vars: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` | 30 min | inference.py |
| 6 | Add HF Spaces README frontmatter (`app_port: 8000`) | 10 min | README.md |
| 7 | Deploy to HF Spaces | 1 hr | - |
| 8 | Remove "(RED HERRING: ...)" from alert messages | 20 min | scenarios/*.json |
| 9 | Remove answers from scenario descriptions | 30 min | scenarios/*.json |
| 10 | Add `.dockerignore` | 5 min | root |

**Total: ~6 hours**

### Tier 2: MUST DO (Framework Compliance + Realism)
| # | Fix | Effort | File(s) |
|---|---|---|---|
| 11 | Fix sub-model inheritance (Action → BaseModel) | 1 hr | models.py |
| 12 | Use typed models in observation (list[dict] → List[Alert]) | 1.5 hr | models.py, environment.py |
| 13 | Add per-step reward signals in metadata | 3 hr | environment.py, graders.py |
| 14 | Implement OnCallRubric (TrajectoryRubric) | 3 hr | new: server/rubric.py |
| 15 | Add time-based degradation (services get worse per step) | 3 hr | environment.py |
| 16 | Add cascading recovery (fix root cause → downstream recovers) | 2 hr | environment.py |
| 17 | Fix observation to not leak full state | 2 hr | environment.py |
| 18 | Fix `check_metrics` to respect `metric_name` | 30 min | environment.py |
| 19 | Standardize parameter names (service vs service_name) | 1 hr | environment.py, models.py |
| 20 | Export `OnCallState` from `__init__.py` | 5 min | __init__.py |

**Total: ~17 hours**

### Tier 3: SHOULD DO (Competitive Advantage)
| # | Fix | Effort | File(s) |
|---|---|---|---|
| 21 | Add PyTorch GRPO training script | 6 hr | new: train.py |
| 22 | Generate more scenarios (5 per task = 20 total) | 4 hr | scenarios/*.json |
| 23 | Add EGAR-style "investigation before remediation" metric | 2 hr | graders.py |
| 24 | Add blast radius tracking | 1 hr | graders.py |
| 25 | Add premature resolution penalty | 30 min | graders.py |
| 26 | Enable web interface | 30 min | Dockerfile / app.py |
| 27 | Add architecture diagram to README | 1 hr | README.md |
| 28 | Add baseline comparison table (random vs heuristic vs LLM) | 3 hr | README.md, scripts |
| 29 | Switch Docker healthcheck to Python-based | 10 min | Dockerfile |
| 30 | Override `get_metadata()` | 30 min | environment.py |

**Total: ~19 hours**

### Tier 4: NICE TO HAVE (First Prize Polish)
| # | Fix | Effort | File(s) |
|---|---|---|---|
| 31 | Build scenario generator (template-based) | 5 hr | new: scenario_generator.py |
| 32 | Add dynamic alert generation during episodes | 3 hr | environment.py |
| 33 | Add gradual recovery modeling (not instant fix) | 2 hr | environment.py |
| 34 | Add seeded randomness / noise to metrics | 2 hr | environment.py |
| 35 | Publish scenarios as HuggingFace Dataset | 1 hr | - |
| 36 | Add LLM-as-judge for summary evaluation | 3 hr | graders.py |
| 37 | Add "wrong action consequences" (restart healthy svc = brief downtime) | 2 hr | environment.py |
| 38 | Add structured root cause format requirement | 1 hr | graders.py |
| 39 | Validate grading weights sum to 1.0 | 15 min | graders.py |
| 40 | Add test dependencies to pyproject.toml | 10 min | pyproject.toml |

**Total: ~19 hours**

---

## EFFORT SUMMARY

| Tier | Items | Total Effort | Status |
|---|---|---|---|
| Tier 1 (Validation) | 10 | ~6 hours | MUST DO FIRST |
| Tier 2 (Framework) | 10 | ~17 hours | MUST DO |
| Tier 3 (Competitive) | 10 | ~19 hours | SHOULD DO |
| Tier 4 (Polish) | 10 | ~19 hours | IF TIME |
| **TOTAL** | **40** | **~61 hours** | |

With a 3-person team and ~10 days until deadline, that's ~2 hours per person per day to complete Tier 1+2+3. Tier 4 is stretch.

**Recommended approach:**
- **Days 1-2:** Tier 1 (pass validation) + Deploy to HF Spaces
- **Days 3-5:** Tier 2 (framework compliance + simulation realism)
- **Days 6-8:** Tier 3 (competitive advantage)
- **Days 9-10:** Tier 4 polish + final testing

---

## WHAT WE GET RIGHT (Don't Break These)

1. Domain choice — incident response is genuinely real-world, universally relatable
2. Rich action space — 13 action types across 5 categories
3. Sophisticated grading — 6 components with per-task weight tuning
4. 4 difficulty levels (exceeds the 3-task minimum)
5. Second scenarios are well-crafted (task3_002 cache eviction, task4_002 TLS cert rotation are excellent)
6. 30/30 tests passing
7. Deterministic episodes
8. LLM-based baseline agent
9. Clean OpenEnv project structure
10. Edge case handling (truncation, invalid inputs, post-resolution)

---

*Compiled from 7 parallel research agents analyzing: OpenEnv framework source, OpenEnv course, Meta judging criteria, full codebase audit, reference environment comparison, OpenSec paper analysis, and HF Spaces deployment requirements.*

*Team Diff Maker — March 27, 2026*
