# OnCallEnv — Complete Implementation Plan (Revised)

> **Goal:** Fix all 40 identified issues to win First Prize ($10,000) at Meta PyTorch OpenEnv Hackathon
> **Codebase:** `/Users/devmhrn/Desktop/WorkSpace/RL_ENV/oncall-env/`
> **Analysis docs:** `/Users/devmhrn/Desktop/WorkSpace/RL_ENV/oncall-env/docs/`
> **Deadline:** April 8, 11:59 PM IST

---

## Context

The oncall-env is an OpenEnv RL environment simulating production incident response. It has solid domain design (13 actions, 4 difficulty levels, 8 scenarios, 6-component grading, 30/30 tests passing) but critical gaps in: automated validation compliance, OpenEnv framework integration, simulation realism, and PyTorch usage. This plan addresses all 40 issues across 8 sequential phases, ordered to minimize breakage and keep tests green at every checkpoint.

### Verified Framework Facts
- `TrajectoryRubric` exists in installed `openenv-core` (**confirmed**)
- `Environment.__init__` signature: `(self, transform=None, rubric=None)` (**confirmed**)
- `_apply_rubric(action, obs)` and `_reset_rubric()` both exist (**confirmed**)
- `create_app()` returns `FastAPI` instance — custom routes can be added after (**confirmed**)
- `create_app(env=...)` takes a `Callable[[], Environment]` factory, not an instance (**confirmed**)

---

## Phase 0: Zero-Risk Housekeeping (~2 hours)

*No code logic changes. File renames, text edits, config additions.*

| Step | What | File(s) | Risk |
|---|---|---|---|
| 0.1 | Rename `baseline.py` → `inference.py`, update README references | `baseline.py`, `README.md` | None |
| 0.2 | Fix env vars: `OPENAI_API_BASE` → `API_BASE_URL`, `OPENAI_MODEL` → `MODEL_NAME`, add `HF_TOKEN` support | `inference.py` lines 102-104 | None |
| 0.3 | Remove `(RED HERRING: ...)` text from alert messages | `scenarios/task4_expert/scenario_001.json` lines 38,56; `scenario_002.json` lines 47,56 | None |
| 0.4 | Rewrite scenario descriptions to show SYMPTOMS only (remove root cause answers) | `scenarios/task3_hard/scenario_001.json`, `scenario_002.json`; `scenarios/task4_expert/scenario_002.json` | None |
| 0.5 | Export `OnCallState` from `__init__.py` | `__init__.py` line 3 + `__all__` | None |
| 0.6 | Create `.dockerignore` (exclude .git, __pycache__, *.egg-info, .venv, docs/) | NEW `.dockerignore` | None |
| 0.7 | Add `description` and `version` fields to `openenv.yaml` | `openenv.yaml` | None |

**Verify:** `pytest server/test_oncall.py -v` → 30/30 passing

---

## Phase 1: Model Layer Fixes (~4 hours)

*Fix Pydantic model inheritance and typing. Prerequisite for observation changes.*

### CRITICAL DESIGN DECISION: Internal state stays `list[dict]`, only observation type changes

The environment internally stores `self._alerts`, `self._services`, `self._deployments`, `self._timeline` as `list[dict]` (copied from scenario JSON). Handlers mutate these dicts directly (e.g., `alert["acknowledged"] = True`). The grader `grade_episode()` receives these raw dicts and uses `.get()` access throughout.

**We MUST NOT change the internal state type.** Only the `OnCallObservation` class gets typed fields. Pydantic v2 auto-coerces dicts to typed models at observation creation time. This means:
- `self._alerts` stays `list[dict]` — handlers and graders work unchanged
- `OnCallObservation.alerts` becomes `List[Alert]` — Pydantic coerces automatically
- `inference.py` `format_observation()` receives typed objects, needs attribute access
- `graders.py` is NOT affected (it receives raw dicts from `self._alerts`, not from observation)

| Step | What | File(s) | Risk |
|---|---|---|---|
| 1.1 | Change `Alert`, `ServiceStatus`, `LogEntry`, `Deploy`, `Event` to inherit from `BaseModel` instead of `Action`. Keep `model_config = {"extra": "allow"}` on each. | `models.py` lines 13-65 | Medium — verify sub-models still accept scenario JSON extra fields |
| 1.2 | Type observation fields: `alerts: List[Alert]`, `services: List[ServiceStatus]`, `recent_deployments: List[Deploy]`, `log_results: Optional[List[LogEntry]]`, `incident_timeline: List[Event]`. Add `Field(description=...)` to all fields. | `models.py` lines 96-106 | HIGH — see dependencies below |
| 1.3 | Update `inference.py` `format_observation()` to use attribute access: `a.severity` not `a.get("severity")`, `a.acknowledged` not `a.get("acknowledged")`, `s.name` not `s.get("name")`, etc. | `inference.py` lines 60-91 | Medium — must match all field names |
| 1.4 | Fix `client.py` `_parse_result()` to avoid double-setting `reward`/`done`: extract them from payload FIRST, then remove from obs_data before constructing `OnCallObservation` | `client.py` lines 19-29 | Low |
| 1.5 | Fix `check_metrics` to properly use `metric_name` in fuzzy fallback matching: `if not metric_name or metric_name.lower() in k.lower()` | `server/environment.py` lines 158-160 | Low |

**What we DON'T touch:** `server/graders.py` (receives raw dicts from `self._alerts`/`self._services`, not from observation), `server/environment.py` internal state types, handler mutation code.

**Verify:** `pytest server/test_oncall.py -v` → 30/30 passing. `python -c "from oncall_env.models import Alert; Alert(alert_id='x', severity='info', service='s', message='m', timestamp='t')"` works.

---

## Phase 2: Rubric Integration + Per-Step Rewards (~6 hours)

*Add the OpenEnv rubric system and intermediate reward signals.*

### Timing clarification
The rubric is created in `__init__` but `score_trajectory()` only runs when `obs.done=True`. By that time `reset()` has populated `self._scenario`, `self._alerts`, etc. No chicken-and-egg problem — the rubric stores a reference to `self` (the env) and reads its fields at scoring time, not at init time.

| Step | What | File(s) | Risk |
|---|---|---|---|
| 2.1 | Create `OnCallRubric(TrajectoryRubric)` with `score_trajectory()` calling existing `grade_episode()`, and `compute_step_rewards()` for uniform credit assignment. Rubric holds a reference to the environment instance (set via `set_env()`). | NEW `server/rubric.py` | Low — new file |
| 2.2 | Integrate rubric into environment: create rubric in `__init__`, call `rubric.set_env(self)`, pass `rubric=` to `super().__init__()`. In `reset()`: call `self._reset_rubric()`. In `step()`: call `self._apply_rubric(action, obs)` after computing observation. **Keep existing `_compute_final_reward()` as the authoritative reward source** — the rubric mirrors it for framework compatibility but doesn't replace the direct grading path. | `server/environment.py` lines 33-34 (`__init__`), line 49 (`reset`), lines 110-112 (`step`) | Medium |
| 2.3 | Add per-step reward signals in `observation.metadata["reward_signals"]` on every step. New method `_compute_step_reward_signals()` returns intermediate progress metrics. | `server/environment.py` `_make_observation()` (line 319) | Low — metadata is a dict, adding keys is safe |
| 2.4 | **Write tests** for new reward signals: verify `obs.metadata["reward_signals"]` exists after step, verify rubric produces same score as direct grading | `server/test_oncall.py` (append new tests) | None |

**Per-step reward signals (computed on every step):**
```python
def _compute_step_reward_signals(self) -> dict:
    scenario = self._scenario
    critical_alerts = [a for a in scenario.get("initial_alerts", []) if a.get("severity") == "critical"]
    acked = sum(1 for a in self._alerts if a.get("acknowledged"))
    expected_diags = scenario.get("expected_diagnostics", [])
    completed_diags = sum(1 for d in expected_diags
        if any(a["action_type"] == d.get("action_type") for a in self._actions_taken))
    has_investigation = any(a["action_type"] in ("query_logs", "check_metrics", "view_dependencies")
        for a in self._actions_taken)
    has_remediation = any(a["action_type"] in ("restart_service", "scale_service", "rollback_deploy", "update_config")
        for a in self._actions_taken)

    return {
        "oncall.triage_progress": acked / max(len(critical_alerts), 1),
        "oncall.investigation_depth": completed_diags / max(len(expected_diags), 1),
        "oncall.premature_action": 0.0 if (not has_remediation or has_investigation) else -0.5,
        "oncall.summary_written": 1.0 if self._summary else 0.0,
        "oncall.resolved": 1.0 if self._resolved else 0.0,
    }
```

**Verify:** `pytest server/test_oncall.py -v` → 30+ tests passing (existing 30 + new tests).

---

## Phase 3: Dynamic Simulation Engine (~5 hours)

*Transform from static JSON quiz to living simulation.*

### Design: degradation changes `error_rate` and `latency_ms` but also triggers status transitions

| Step | What | File(s) | Risk |
|---|---|---|---|
| 3.1 | Create simulation engine with two functions: `degrade_services()` — unhealthy services get 1.1x worse each step, with status transitions (`degraded` → `down` when `error_rate > 80`); `propagate_recovery()` — fixing root cause propagates recovery to downstream services via dependency graph (not instant: sets to `degraded` with reduced error rate, not straight to `healthy`) | NEW `server/simulator.py` | Low — new file |
| 3.2 | Integrate into `step()`: call `degrade_services()` after every handler execution, call `propagate_recovery()` after successful remediation that sets a service to `healthy` | `server/environment.py` step() between handler call and `_make_observation` | MEDIUM-HIGH — see risk analysis below |
| 3.3 | **Write tests** for simulation: verify services degrade over steps, verify recovery propagates | `server/test_oncall.py` (append) | None |

**Risk analysis for Phase 3.2:**
- `test_full_episode_task1` (11 steps): payment-api starts degraded (12.3% error). After restart at step 9, it becomes healthy. order-service starts degraded (15% error). Over 9 steps before fix: `15 * 1.1^9 = 35.4%`. Status stays "degraded" (below 80% threshold). Grader checks `all_healthy` on services_state — order-service is still degraded, so `all_healthy = False`. This is ALREADY the case in the static env. **Test should still pass `reward > 0.5`.**
- `test_full_episode_task2` (11 steps): inventory-service starts degraded (5% error). After rollback at step 9: `5 * 1.1^9 = 11.8%`. Stays degraded. With `propagate_recovery`, order-service and payment-api (which depend on inventory-service) would recover. This could actually INCREASE the reward. **Test should still pass `reward > 0.4`.**
- `test_determinism`: Both envs degrade identically (deterministic). **Passes.**
- `test_grader_*`: Call `grade_episode()` directly, bypassing environment. **Not affected.**

**Fallback:** If tests fail, reduce degradation factor to 1.05x or make degradation configurable via a class attribute.

**NOTE: Partial observability is MOVED to Phase 8.** It requires updating the baseline agent, system prompt, and documented scores. Too risky to do before deployment.

**Verify:** `pytest server/test_oncall.py -v` → all tests passing (30 existing + new simulation tests).

---

## Phase 4: API Endpoints + Grader Improvements (~4 hours)

*Add hackathon-required endpoints and improve grading sophistication.*

| Step | What | File(s) | Risk |
|---|---|---|---|
| 4.1 | Add `GET /tasks` — returns task list + `OnCallAction.model_json_schema()` | `server/app.py` (add after `create_app()`) | Low |
| 4.2 | Add `POST /grader` — accepts episode data, returns score via `grade_episode()` | `server/app.py` | Low |
| 4.3 | Add `POST /baseline` — creates `OnCallEnvironment()` inline, runs a simple heuristic agent (acknowledge critical alerts → query logs for degraded services → restart root cause service → write summary → resolve), returns scores per task. Does NOT call LLM — pure heuristic so it works without API keys. | `server/app.py` | Low |
| 4.4 | Add EGAR metric (Evidence-Gated Action Rate): for each remediation action, check if agent queried logs/metrics for that SPECIFIC service beforehand. Store as supplementary metric in grade_episode return, NOT changing the main 6-component reward formula. | `server/graders.py` new function `_compute_egar()` | Low |
| 4.5 | Add premature resolution penalty in `_grade_remediation()`: if `resolved=True` but zero investigation actions (`query_logs`, `check_metrics`, `view_dependencies`) taken, apply -0.3. Verify: `test_environment_step_resolve_incident` resolves immediately → reward drops but still >= 0.0 (grader clamps). `test_grader_empty_episode` has `resolved=False` → not affected. `test_grader_varying_scores` r3 has investigation → not penalized. | `server/graders.py` `_grade_remediation()` around line 178 | Low-Medium |
| 4.6 | Add blast radius tracking: count services targeted by wrong remediation actions vs correct ones. Return as supplementary metric alongside EGAR. | `server/graders.py` new function | Low |
| 4.7 | **Write tests** for new endpoints: GET /tasks returns schema, POST /grader returns score, POST /baseline returns scores dict | `server/test_oncall.py` (append) | None |

**Verify:** `pytest` passes. Start server locally, `curl localhost:8000/tasks`, `curl -X POST localhost:8000/grader -H 'Content-Type: application/json' -d '{...}'`, `curl -X POST localhost:8000/baseline`.

---

## Phase 5: Docker + Metadata + Web Interface (~2 hours)

| Step | What | File(s) | Risk |
|---|---|---|---|
| 5.1 | Override `get_metadata()` — return name, description, version, task count | `server/environment.py` new method | None |
| 5.2 | Switch healthcheck from `curl` to Python-based: `python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"` | `server/Dockerfile` line 34-35 | Low |
| 5.3 | Add `ENV ENABLE_WEB_INTERFACE=true` to Dockerfile. Check if gradio is needed as explicit dependency (openenv-core may bundle it). If needed, add to pyproject.toml. | `server/Dockerfile`, `pyproject.toml` | Medium — test Docker build locally |
| 5.4 | Add `app_port: 8000` frontmatter to README for HF Spaces (do this here, before deployment) | `README.md` (add YAML block at top) | None |

**Verify:** `docker build -f server/Dockerfile -t oncall-env .` succeeds. `docker run -p 8000:8000 oncall-env` starts. `curl localhost:8000/health` and `curl localhost:8000/web` both work.

---

## Phase 6: Scenarios + README + HF Deployment (~12-16 hours)

*Honest time estimate: each scenario is 100-200 lines of realistic JSON. Writing one good scenario takes 30-60 minutes.*

| Step | What | File(s) | Effort |
|---|---|---|---|
| 6.1 | Generate **8 new scenarios** (2 more per task = 16 total). Reduced from 12 to stay realistic. Themes: **easy** — cert expiry warning, disk space alerts; **medium** — message queue backlog, network partition; **hard** — database migration gone wrong, auto-scaler feedback loop; **expert** — Kubernetes node pressure with pod eviction, CDN origin failover with DNS TTL confusion. | NEW `scenarios/task*/scenario_003-004.json` | 6-8 hrs |
| 6.2 | Add ASCII architecture diagram to README showing microservice topology + agent interaction flow | `README.md` | 1 hr |
| 6.3 | Add baseline comparison table — run heuristic baseline on all scenarios, record scores. Format: random vs heuristic vs LLM (documented, not necessarily measured for LLM). | `README.md` | 1 hr |
| 6.4 | Polish README: compelling intro paragraph, reward formula explanation, EGAR metric description, link to OpenEnv framework | `README.md` | 1 hr |
| 6.5 | Deploy to HF Spaces. Create Space with `sdk: docker`, set secrets (`HF_TOKEN`, `API_BASE_URL`, `MODEL_NAME`). Test all endpoints. | HF Hub | 2-3 hrs (including debugging) |

**Verify:** HF Space URL returns 200 on `/health`. `/tasks` returns task list. `/web` shows Gradio UI. `inference.py` can connect to the deployed Space.

---

## Phase 7: PyTorch Integration (~6 hours)

*This is a PROOF OF CONCEPT. GRPO training requires GPU (A100 40GB minimum). The hackathon eval runs on 2 vCPU / 8GB RAM. The training script demonstrates PyTorch integration but cannot be run during automated evaluation.*

| Step | What | File(s) | Risk |
|---|---|---|---|
| 7.1 | Add `[project.optional-dependencies]` with test, train, inference groups. Train group includes `torch>=2.0`, `trl>=0.7.0`, `transformers>=4.35.0`. These are NOT in the main dependencies (won't affect server Docker build). | `pyproject.toml` | None |
| 7.2 | Create GRPO training script using TRL's `environment_factory` pattern. Define `OnCallToolEnv` class with tool methods mapping to environment actions. Define per-component reward functions reading from `observation.metadata["reward_signals"]`. Include `--dry-run` flag for import-only verification without GPU. | NEW `train.py` | Low — standalone |
| 7.3 | Add training section to README documenting: what GRPO is, how to run training, expected results, hardware requirements | `README.md` | None |

**TRL environment_factory pattern (verified from docs):**
```python
class OnCallToolEnv:
    """Tool-calling environment wrapper for TRL GRPO training."""
    def __init__(self):
        self.env = OnCallEnvironment()
        self.reward = 0.0

    def reset(self, **kwargs) -> str | None:
        obs = self.env.reset(task_id=1, scenario_idx=0)
        return format_observation_as_text(obs)

    def query_logs(self, service: str) -> str:
        """Query service logs. Args: service name."""
        obs = self.env.step(OnCallAction(action_type="query_logs", params={"service": service}))
        self.reward = obs.reward or 0.0
        return obs.message

    def restart_service(self, service_name: str) -> str:
        """Restart a service. Args: service name."""
        obs = self.env.step(OnCallAction(action_type="restart_service", params={"service_name": service_name}))
        self.reward = obs.reward or 0.0
        return obs.message
    # ... one method per action type

def reward_func(environments, **kwargs) -> list[float]:
    return [env.reward for env in environments]

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-1B-Instruct",
    reward_funcs=reward_func,
    environment_factory=OnCallToolEnv,
    args=GRPOConfig(max_completion_length=2048, num_generations=4, ...),
    train_dataset=dataset,
)
```

**Verify:** `python train.py --dry-run` imports successfully (no GPU needed). With GPU: training runs for 1 step without error.

---

## Phase 8: Final Polish (if time permits, ~10 hours)

| Step | What | Priority | Effort |
|---|---|---|---|
| 8.1 | **Partial observability** — hide detailed metrics for un-investigated services. Requires updating baseline agent system prompt and re-measuring baseline scores. MOVED HERE from Phase 3 because it changes agent behavior. | High | 3 hrs |
| 8.2 | Scenario generator — template-based, generates variations at `reset()` time using seed | High | 4 hrs |
| 8.3 | Dynamic alert generation — new alerts fire as services degrade during episode | Medium | 2 hrs |
| 8.4 | Seeded randomness — use `reset(seed=...)` to add noise to metrics/latencies | Medium | 1 hr |
| 8.5 | Gradual recovery — `status = "recovering"` instead of instant healthy | Medium | 1 hr |
| 8.6 | Wrong action consequences — restart healthy svc = brief degradation (2-3 steps) | Medium | 1 hr |
| 8.7 | Validate grading weights sum to 1.0 | Low | 15 min |
| 8.8 | Add pyproject.toml test dependencies | Low | 10 min |
| 8.9 | LLM-as-judge for summary evaluation | Low | 3 hrs |
| 8.10 | Publish scenarios as HuggingFace Dataset | Low | 1 hr |

---

## Critical Files to Modify

| File | Phases | Key Changes |
|---|---|---|
| `models.py` | 1 | BaseModel inheritance for sub-models, typed observation fields, Field descriptions |
| `server/environment.py` | 2, 3, 5 | Rubric integration, simulation hooks, per-step rewards, get_metadata() |
| `server/app.py` | 4 | /tasks, /baseline, /grader endpoints |
| `server/graders.py` | 4 | EGAR metric, blast radius, premature resolution penalty (supplementary — main formula unchanged) |
| `inference.py` (renamed) | 0, 1 | Env vars fix, attribute access for typed models |
| `client.py` | 1 | Fix _parse_result double-set of reward/done |
| `server/Dockerfile` | 5 | Python healthcheck, ENABLE_WEB_INTERFACE |
| `README.md` | 5, 6 | HF frontmatter, architecture diagram, comparison table, training docs |
| `openenv.yaml` | 0 | Description, version |
| `pyproject.toml` | 5, 7 | Gradio dep (if needed), optional dep groups |
| `__init__.py` | 0 | Export OnCallState |
| `server/test_oncall.py` | 2, 3, 4 | New tests for reward signals, simulation, endpoints |

## New Files to Create

| File | Phase | Purpose |
|---|---|---|
| `server/rubric.py` | 2 | OnCallRubric(TrajectoryRubric) |
| `server/simulator.py` | 3 | Dynamic degradation + cascading recovery |
| `train.py` | 7 | GRPO training script (PyTorch proof-of-concept) |
| `.dockerignore` | 0 | Exclude build artifacts |
| 8 new scenario JSONs | 6 | 2 additional scenarios per task |

---

## Honest Time Estimates

| Phase | Effort (1 person) | Parallelizable? |
|---|---|---|
| Phase 0: Housekeeping | 2 hrs | Yes — split across team |
| Phase 1: Models | 4 hrs | No — must be sequential |
| Phase 2: Rubric | 6 hrs | No — depends on Phase 1 |
| Phase 3: Simulation | 5 hrs | No — depends on Phase 2 |
| Phase 4: Endpoints | 4 hrs | Yes — independent of Phase 3 |
| Phase 5: Docker | 2 hrs | Yes — independent of Phase 4 |
| Phase 6: Scenarios + Deploy | 12-16 hrs | Scenarios parallelizable, deploy sequential |
| Phase 7: PyTorch | 6 hrs | Yes — standalone |
| Phase 8: Polish | 10 hrs | Yes — each item independent |
| **TOTAL** | **~51-55 hrs** | |

**With 3 people, 11 days:** ~17-18 hrs/person = ~1.6 hrs/day/person. Feasible but tight.

## Team Work Distribution (3 people, ~11 days)

| Person | Phases | Focus | Est. Hours |
|---|---|---|---|
| **A (Backend)** | 1, 2, 3 | Models, rubric, simulation engine | ~15 hrs |
| **B (API/Deploy)** | 0, 4, 5, 7 | Endpoints, Docker, HF deploy, PyTorch | ~18 hrs |
| **C (Content)** | 0.3-0.4, 6, 8 | Scenarios, README, polish | ~18 hrs |

**Critical path:** Phase 0 → Phase 1 → Phase 2 → Phase 3 (Person A, sequential)
**Parallel:** Person B does Phase 0 + Phase 4 while A does Phase 1-2. Person C starts scenarios (Phase 6.1) immediately.

**Recommended timeline:**
- **Days 1-2:** Phase 0 (all) + Phase 1 (A) + Phase 4 (B) + start scenarios (C)
- **Days 3-4:** Phase 2 (A) + Phase 5 (B) + scenarios (C)
- **Days 5-6:** Phase 3 (A) + Phase 6.5 deploy (B) + README (C)
- **Days 7-8:** Phase 7 (B) + Phase 8 (A+C)
- **Days 9-10:** Integration testing, bug fixes, final deployment
- **Day 11:** Buffer for unexpected issues

## Risk Mitigation

- **Phase 1.2 (typed observation fields)** is highest risk. Fallback: keep `list[dict]` in observation, only change sub-model inheritance (Phase 1.1 alone is still valuable).
- **Phase 3.2 (dynamic simulation)** may affect test reward values. Start with 1.1x degradation. If tests fail, reduce to 1.05x. If still failing, make degradation configurable and disable for tests.
- **Phase 5.3 (web interface)** depends on gradio being available. If not in openenv-core, add it explicitly. If Docker build fails, skip web interface (it's not required by hackathon).
- **Phase 6.5 (HF deploy)** — attempt on Day 5 with partial changes to shake out Docker/build issues early. Don't wait until everything is done.
- **Run `pytest server/test_oncall.py -v` after EVERY step.** Never proceed with failing tests.

## Verification (End-to-End)

After all phases:
1. `pytest server/test_oncall.py -v` → all tests pass (30 existing + new)
2. `docker build -f server/Dockerfile -t oncall-env .` → builds successfully
3. `docker run -p 8000:8000 oncall-env` → starts, health check passes
4. `curl http://localhost:8000/health` → `{"status": "healthy"}`
5. `curl http://localhost:8000/tasks` → returns 4 tasks + action schema
6. `curl -X POST http://localhost:8000/reset` with `{"task_id": 1}` → returns observation
7. `curl -X POST http://localhost:8000/grader` with episode data → returns score
8. `curl -X POST http://localhost:8000/baseline` → returns scores for all tasks
9. `curl http://localhost:8000/web` → Gradio UI loads (if enabled)
10. HF Space URL responds to all above
11. `python inference.py` → completes in < 20 min, prints scores for all 4 tasks
12. `python train.py --dry-run` → imports successfully, prints training config

---

## Review Corrections Applied

This revision addresses the following issues found in the plan review:

1. **Phase 1.2 blast radius** — Explicitly documented that internal state stays `list[dict]`, graders are NOT affected, only observation type and inference.py format_observation change.
2. **Phase 3.3 partial observability moved to Phase 8** — It breaks the baseline agent and requires re-measuring documented scores. Too risky before deployment.
3. **Phase 6.1 time estimate corrected** — From "~6 hours" to "12-16 hours" for the full phase. Scenario count reduced from 12 to 8 (2 new per task).
4. **Phase 7 honesty** — Explicitly marked as proof-of-concept. Cannot run during hackathon eval (no GPU). Added `--dry-run` flag.
5. **Missing test steps added** — Phases 2, 3, 4 now include explicit test-writing steps (2.4, 3.3, 4.7).
6. **Client _parse_result fix added** — Phase 1.4 addresses the double-set of reward/done.
7. **Simulation status transitions added** — Phase 3.1 includes `degraded → down` threshold at error_rate > 80.
8. **Phase 2 timing clarification** — Rubric references env fields at scoring time (during step when done), not at init time. No chicken-and-egg issue.
9. **Timeline made realistic** — Per-person hours calculated, daily commitment estimated, recommended day-by-day schedule added.
10. **HF frontmatter moved earlier** — To Phase 5.4 (before deployment) instead of Phase 6.2.
