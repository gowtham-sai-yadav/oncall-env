# OnCallEnv - Brutally Honest Critical Analysis

> **Goal:** Win First Prize ($10,000) at Meta PyTorch OpenEnv Hackathon
> **Current State:** Functional but NOT first-prize quality. Major gaps identified.
> **Team:** Diff Maker | **Deadline:** April 8, 11:59 PM IST

---

## THE VERDICT

Your oncall-env has solid domain logic and good scenario design. But compared to gold-standard OpenEnv environments (chess_env, textarena_env), it's missing critical framework integration, has zero PyTorch involvement, and operates as a **static JSON quiz** rather than a **living simulation**. Without fixes, this is a middle-of-the-pack submission.

---

## SECTION 1: CRITICAL ISSUES (Fix These or Lose)

### CRIT-1: ZERO PYTORCH USAGE -- This is a PyTorch Hackathon

**The problem:** There is not a single `import torch` anywhere in the codebase. This is a **Meta PyTorch** hackathon. The judges are PyTorch engineers. The OpenEnv framework integrates with TRL (which uses PyTorch for GRPO training). Submitting an environment with zero PyTorch is like entering a cooking competition without using the required ingredient.

**What to do:**
- Add a PyTorch-based learned reward model or LLM-as-judge grader
- Add a GRPO training script showing how to train an LLM on your environment using TRL + PyTorch
- Use TorchRL's `TensorDict` for observation encoding (optional but shows depth)
- Even a simple PyTorch reward model that scores agent summaries would count

**Impact:** Judges will literally search for `torch` imports. Without it, you cannot win.

---

### CRIT-2: Environment is STATIC -- No Dynamic Simulation

**The problem:** Your environment is a glorified JSON lookup engine. When the agent takes actions:
- Services don't degrade over time if the agent is slow
- No cascading recovery when root cause is fixed (fix inventory-service, but order-service stays degraded)
- No new alerts fire during the episode
- No stochasticity -- every run is identical
- Metrics are frozen arrays from JSON, not live time series

**Why this matters:** A real production incident is a living, breathing thing. Services get WORSE if you don't act. Fixing the root cause causes downstream services to RECOVER. New alerts appear as the situation evolves. Your environment simulates none of this.

**What Meta engineers think:** Meta's ARE (Agents Research Environments) paper explicitly calls for "dynamic environments" and "grounded situations that mirror actual real-world challenges." A static JSON quiz does not qualify.

**What to do:**
- Add a tick/clock system: every step advances simulated time by 30s
- Services degrade over time (error_rate increases, latency grows) if not remediated
- Fixing root cause propagates recovery through dependency graph
- New alerts fire as conditions worsen
- Add noise/jitter to metrics (seeded randomness for reproducibility)

---

### CRIT-3: No Rubric System Integration -- Framework Mismatch

**The problem:** Gold-standard OpenEnv environments (chess_env, textarena_env) use the framework's Rubric system:

```python
# How gold-standard envs work:
class ChessEnvironment(Environment):
    def __init__(self):
        super().__init__(rubric=ChessWinLossRubric(gamma=0.99))

    def step(self, action):
        obs = ...
        reward = self._apply_rubric(action, obs)  # Framework method
        obs.reward = reward
        return obs
```

**Your env does this instead:**

```python
# How oncall-env works (WRONG):
class OnCallEnvironment(Environment):
    def __init__(self):
        super().__init__()  # No rubric!

    def step(self, action):
        ...
        reward = grade_episode(...)  # Standalone function, not a Rubric
        return obs
```

**Why this matters:**
- The GRPO training pipeline calls `env.rubric.named_rubrics()` to introspect reward components
- `compute_step_rewards()` on TrajectoryRubric provides per-step credit assignment
- Without the Rubric, your env can't be used for RL training -- only evaluation
- This is like building a car without connecting the engine to the wheels

**What to do:**
- Create `OnCallRubric(TrajectoryRubric)` in a new `rubric.py` file
- Implement `score_trajectory()` and `intermediate_reward()`
- Pass it to `super().__init__(rubric=OnCallRubric())`
- Use `self._apply_rubric(action, obs)` in `step()`

---

### CRIT-4: No Per-Step Reward Signals -- GRPO Can't Train

**The problem:** Your environment only provides a single scalar reward at episode end. Gold-standard environments provide multi-dimensional reward signals on EVERY step:

```python
# TextArena Wordle (gold standard):
observation.metadata["reward_signals"] = {
    "wordle.greens": 0.6,     # Correct-position letters
    "wordle.yellows": 0.4,    # Correct letters wrong position
    "wordle.repetitions": 1.0, # No repeated guesses
    "wordle.correct": 0.0,     # Haven't won yet
}
```

**Your env:** `reward = None` for all intermediate steps, only a single float at the end.

**Why this matters:** GRPO (the training algorithm in the OpenEnv tutorial) NEEDS per-step reward signals to compute advantages. Without them, training gives zero gradient for all steps except the last one. The entire point of the environment is to train agents -- if it can't provide training signals, it's an evaluation harness at best.

**What to do:**
- On every step, compute and expose partial reward signals in `observation.metadata["reward_signals"]`:
  - `oncall.triage_progress` -- how many critical alerts acknowledged so far
  - `oncall.investigation_quality` -- did the agent query logs/metrics before acting?
  - `oncall.premature_action_penalty` -- did the agent remediate without investigating first?
  - `oncall.correct_service_targeted` -- is the agent looking at the right service?
- Expose the 6 grading component scores as separate signals at episode end

---

### CRIT-5: RED HERRINGS are LITERALLY LABELED "RED HERRING"

**The problem:** In `scenarios/task4_expert/scenario_001.json`, alert messages contain:

```json
"message": "Model inference latency increased 3x (RED HERRING: scheduled model update)"
"message": "Search latency elevated to 600ms (RED HERRING: elasticsearch reindexing)"
```

This literally tells the agent which alerts to ignore. The expert task is supposed to be hard because of misleading signals -- but you've labeled them.

**Similarly, scenario descriptions contain answers:**
- Task 3: "The root cause is subtle -- a recent config change combined with a traffic spike triggered a memory leak"
- Task 4: "Several unrelated events coincide with the real failure, creating red herrings"

**What to do:**
- Remove ALL "(RED HERRING: ...)" labels from alert messages
- Remove root cause hints from descriptions -- describe only the symptoms
- The description should say "Multiple services are degraded. Investigate and resolve." -- nothing more

---

### CRIT-6: Sub-models Inherit from Action Instead of BaseModel

**The problem:** `Alert`, `ServiceStatus`, `LogEntry`, `Deploy`, `Event` all inherit from `Action`:

```python
class Alert(Action):  # WRONG -- Alert is data, not an agent action
```

The `Action` base class has `model_config = ConfigDict(extra="forbid")` and includes a `metadata` field. Your sub-models override this with `extra="allow"`, contradicting the parent. More importantly, these are DATA models, not actions the agent takes.

**What to do:**

```python
from pydantic import BaseModel

class Alert(BaseModel):  # CORRECT
    model_config = ConfigDict(extra="forbid")
    alert_id: str
    severity: Literal["critical", "warning", "info"]
    ...
```

Also: `OnCallObservation` uses `list[dict]` for alerts/services/etc. The typed sub-models are defined but NEVER USED. Change to `alerts: List[Alert]`, `services: List[ServiceStatus]`, etc.

---

## SECTION 2: HIGH-PRIORITY ISSUES (Fix to Win)

### HIGH-1: Remediation Logic is Unrealistic -- No Cascading Recovery

When you fix the root cause service, downstream services should RECOVER. Currently:
- Fix inventory-service in Task 2 -> inventory-service becomes "healthy"
- But order-service stays "degraded" with 40% error rate FOREVER
- api-gateway stays "degraded" with 25% error rate FOREVER

**What to do:** After any valid remediation, propagate recovery through the dependency graph:
```python
def _propagate_recovery(self):
    """When a root cause is fixed, downstream services recover."""
    # For each service, check if all its dependencies are healthy
    # If so, gradually improve its metrics
```

### HIGH-2: Grader is Trivially Gameable

`_grade_root_cause` uses substring matching: `if kw in summary_lower`. An agent could write a summary stuffed with keywords without understanding anything. The keywords are in the scenario JSON (readable by anyone).

**What to do:**
- Use an LLM-as-judge for summary evaluation (OpenEnv has `LLMJudge` rubric built-in)
- Require the summary to be coherent, not just keyword-stuffed
- Weight the summary evaluation more on structure: "What happened? Why? What was done? What prevented recurrence?"

### HIGH-3: Only 2 Scenarios Per Task (8 Total) -- Way Too Thin

Gold-standard benchmarks have hundreds of scenarios. With 2 per difficulty, an agent can memorize them. There's no generalization test.

**What to do:**
- Generate at least 5 scenarios per difficulty level (20 total minimum)
- Even better: create a scenario GENERATOR that produces random incidents from templates
- Scenario variety: DB failures, network partitions, bad deploys, config errors, certificate expiry, DNS issues, disk full, memory leaks, CPU spikes, dependency version conflicts

### HIGH-4: Observation Leaks Full State Every Step

Every step returns ALL alerts and ALL service statuses. The agent has perfect observability without querying anything. This makes `check_metrics` and `query_logs` actions pointless -- the agent already knows everything.

**What to do:**
- Return only alerts and basic service status in every observation
- Make detailed metrics/logs/dependencies ONLY available through explicit actions
- This forces the agent to INVESTIGATE, which is the whole point

### HIGH-5: Missing Required Endpoints (/tasks, /baseline, /grader)

The hackathon dashboard specifies three custom endpoints that are NOT auto-generated by `create_app()`:
- `/tasks` -- Return list of tasks and the action schema
- `/baseline` -- Trigger inference script and return baseline scores
- `/grader` -- Return grader score after an episode

These are hackathon-specific requirements, not OpenEnv framework features.

**What to do:** Add these as custom FastAPI routes in `app.py`:
```python
@app.get("/tasks")
async def list_tasks(): ...

@app.post("/baseline")
async def run_baseline(): ...

@app.post("/grader")
async def grade(): ...
```

### HIGH-6: Baseline Uses Wrong Environment Variables

The hackathon requires:
- `API_BASE_URL` (not `OPENAI_API_BASE`)
- `MODEL_NAME` (not `OPENAI_MODEL`)
- `HF_TOKEN`
- Must use OpenAI Client for LLM calls
- Must be named `inference.py` (not `baseline.py`)

Your baseline.py uses `OPENAI_API_KEY`, `OPENAI_API_BASE`, `OPENAI_MODEL`.

**What to do:** Rename to `inference.py`, use the required env vars.

### HIGH-7: Inference Must Run in < 20 Minutes on 2 vCPU / 8GB RAM

Your baseline runs 4 tasks with up to 25 LLM calls each = 100 LLM API calls. If the API is slow (500ms-2s per call), that's 50-200 seconds. Should be fine, but:
- Task 4 with its 10 alerts and 9 services produces very large observation strings
- Each LLM call includes the full conversation history
- By step 20, the context window could be enormous

**What to do:** Truncate conversation history to last N messages, or summarize periodically.

---

## SECTION 3: MEDIUM-PRIORITY ISSUES (Polish for Victory)

### MED-1: No `get_metadata()` Override

The `/metadata` endpoint returns generic info. Override to return:
```python
def get_metadata(self) -> EnvironmentMetadata:
    return EnvironmentMetadata(
        name="oncall_env",
        description="Incident Response Command Center - simulates production incidents for training on-call AI agents",
        version="0.1.0",
        ...
    )
```

### MED-2: Inconsistent Parameter Naming

Some actions use `service`, others use `service_name`:
- `query_logs` -> `params.service`
- `restart_service` -> `params.service_name`

This is confusing for agents. Standardize to one.

### MED-3: `check_metrics` Ignores metric_name Parameter

The metric lookup does prefix matching on service name but ignores `metric_name`. If you query `check_metrics(service="order-service", metric_name="error_rate")`, you might get `order-service:memory_mb` instead.

### MED-4: Baseline Uses Synchronous OpenAI in Async Function

`run_baseline` is `async def` but calls synchronous `client_llm.chat.completions.create()`, blocking the event loop. Use `AsyncOpenAI` instead.

### MED-5: No GRPO Training Script

The ultimate proof that your environment works is showing an LLM actually getting BETTER at incident response through training. Even a simple script showing 100 training episodes with improving scores would be hugely impressive.

### MED-6: `_handle_scale_service` Ignores the `replicas` Parameter

The agent can specify `replicas=10` but the environment never uses this value. The service either gets fixed or doesn't based on `valid_remediations`.

### MED-7: Reward Caching After Resolution

`_compute_final_reward()` is recalculated every time `step()` is called after resolution. Cache the reward.

---

## SECTION 4: STRATEGIC RECOMMENDATIONS (What Winners Do)

### STRAT-1: Add a GRPO Training Demo

The single most impressive thing you can show is: "Here's an LLM that gets better at incident response after training on our environment." The OpenEnv Module 5 shows GRPO training on Wordle. Replicate this for oncall-env.

This requires:
1. Per-step reward signals (CRIT-4)
2. Rubric integration (CRIT-3)
3. A training script using TRL GRPOTrainer
4. Before/after scores showing improvement

### STRAT-2: Enable the Web Interface

Set `ENABLE_WEB_INTERFACE=true` in your environment. This gives judges a Gradio-based UI where they can interactively play through incidents. Very impressive for demos.

### STRAT-3: Exceptional README

The hackathon uses LLM scoring on documentation. Your README should:
- Start with a compelling pitch (not setup instructions)
- Include a diagram of the microservice architecture
- Show a complete walkthrough with screenshots
- Explain the reward function design philosophy
- Link to the OpenEnv framework
- Include baseline results with interpretation

### STRAT-4: Add Scenario Generation

Instead of 8 hand-crafted scenarios, build a scenario generator that can produce unlimited incidents from templates. This shows engineering depth and makes the environment truly useful for RL training.

### STRAT-5: Show Baseline Score Progression

In your README, show how different strategies perform:
```
| Strategy        | Task 1 | Task 2 | Task 3 | Task 4 |
|-----------------|--------|--------|--------|--------|
| Random          | 0.05   | 0.02   | 0.01   | 0.01   |
| Simple Heuristic| 0.45   | 0.25   | 0.10   | 0.05   |
| GPT-4o-mini     | 0.65   | 0.45   | 0.25   | 0.10   |
| GRPO-trained    | 0.85   | 0.70   | 0.50   | 0.30   |
```

This demonstrates the environment has a meaningful learning curve.

---

## SECTION 5: COMPLETE PRIORITY LIST

### Must Do (Days 1-2) -- Without these, you lose

| # | Issue | Effort | Impact |
|---|---|---|---|
| 1 | Remove RED HERRING labels from alerts & answers from descriptions | 30 min | Fixes broken difficulty |
| 2 | Fix sub-model inheritance (Action -> BaseModel) | 1 hr | Spec compliance |
| 3 | Add /tasks, /baseline, /grader endpoints | 2 hr | Pass pre-submission check |
| 4 | Rename baseline.py -> inference.py, fix env vars | 30 min | Pass pre-submission check |
| 5 | Add per-step reward signals in observation.metadata | 3 hr | Training integration |
| 6 | Implement OnCallRubric (TrajectoryRubric subclass) | 3 hr | Framework compliance |
| 7 | Add dynamic simulation (time degradation, cascading recovery) | 4 hr | Realism |

### Should Do (Days 3-4) -- These make you competitive

| # | Issue | Effort | Impact |
|---|---|---|---|
| 8 | Add PyTorch usage (reward model or GRPO training script) | 4 hr | PyTorch hackathon credibility |
| 9 | Generate more scenarios (5+ per task) | 3 hr | Benchmark depth |
| 10 | Fix observation to not leak full state | 2 hr | Realistic investigation |
| 11 | Fix check_metrics to use metric_name properly | 1 hr | Action correctness |
| 12 | Fix cascading recovery propagation | 2 hr | Realistic remediation |
| 13 | Standardize parameter names (service vs service_name) | 1 hr | API consistency |
| 14 | Enable web interface | 30 min | Demo capability |

### Nice to Have (Day 5+) -- These win first prize

| # | Issue | Effort | Impact |
|---|---|---|---|
| 15 | GRPO training script with before/after results | 6 hr | Ultimate differentiator |
| 16 | Scenario generator (template-based infinite scenarios) | 4 hr | Engineering depth |
| 17 | LLM-as-judge for summary evaluation | 3 hr | Grading sophistication |
| 18 | Multiple baseline strategies with comparison table | 2 hr | README quality |
| 19 | Architecture diagram in README | 1 hr | Presentation |
| 20 | Override get_metadata() | 30 min | Polish |

---

## SECTION 6: WHAT THE CURRENT CODE GETS RIGHT

To be fair, here's what's already strong:

1. **Domain choice is excellent** -- incident response is genuinely real-world, not a toy
2. **13 action types across 5 categories** -- rich, realistic action space
3. **6-component grading with per-task weight tuning** -- sophisticated evaluation
4. **4 difficulty levels exceeding the 3-task minimum** -- shows ambition
5. **Red herrings and misleading signals** (concept is right, execution needs fix)
6. **30/30 tests passing** -- solid test coverage
7. **Deterministic episodes** -- reproducible results
8. **Clean project structure** following OpenEnv conventions
9. **LLM-based baseline agent** -- ready to produce scores
10. **Edge case handling** -- truncation, invalid inputs, post-resolution actions

The foundation is solid. The issues are about framework integration, dynamic simulation, and PyTorch connection -- all fixable.

---

## SECTION 7: HACKATHON-SPECIFIC CHECKLIST

From research, the exact automated validation checks are:

| Check | Our Status | Action Needed |
|---|---|---|
| HF Space returns HTTP 200 | NOT DEPLOYED | Deploy to HF Spaces |
| HF Space responds to `reset()` | Should work | Test after deployment |
| `openenv.yaml` validates | PASS | - |
| Typed models exist | PASS (but with inheritance bug) | Fix sub-model inheritance |
| `step()`/`reset()`/`state()` endpoints work | PASS | - |
| Dockerfile builds | UNTESTED on HF | Test Docker build |
| `inference.py` completes without error | WRONG FILENAME | Rename + fix env vars |
| `inference.py` runs in < 20 min | LIKELY PASS | Verify |
| 3+ task graders exist | PASS (have 4) | - |
| Graders produce scores 0.0-1.0 | PASS | - |
| README with action/observation spaces | PASS | Polish |
| `/tasks` endpoint | MISSING | Add |
| `/baseline` endpoint | MISSING | Add |
| `/grader` endpoint | MISSING | Add |

---

## SECTION 8: KEY RESEARCH FINDINGS

### What Meta Engineers Look For (from ARE paper):
- Dynamic environments that evolve over time
- Multi-step reasoning (10+ steps)
- Deterministic verifiers checking agent write operations against ground truth
- Failure mode surfacing -- tasks that expose interesting agent failures

### What Wins Hackathons (from Devpost judges):
- "Focus on a single feature your application does extremely well"
- Presentation and storytelling matter as much as code
- Complete, polished submissions beat ambitious but half-built ones
- Address EVERY criterion in the rubric -- missing one hurts disproportionately

### OpenSec Environment (Similar Domain, Well-Received):
- Dual-control simulation (defender vs attacker state machine)
- SQLite-backed evidence with deliberate noise
- Step budget forcing investigation-vs-action trade-offs
- 6 distinct calibration metrics
- Web-based trace visualization

### Prize Information:
- 1st place: $10,000
- 2nd place: $4,550
- Total pool: $30,000
- Winners get direct interviews with Meta & Hugging Face AI teams
- Code reviewed by Meta engineers
- Round 2 (Finale): 48-hour in-person hackathon in Bangalore, April 25-26

---

*Analysis generated from 5 parallel research agents covering: OpenEnv framework source, OpenEnv course, Meta judging criteria, codebase audit, and reference environment comparison.*
*Team Diff Maker, March 2026*
