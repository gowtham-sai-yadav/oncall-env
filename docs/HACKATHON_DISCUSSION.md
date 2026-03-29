# Meta PyTorch OpenEnv Hackathon — Complete Discussion & Plan

> **Team:** Diff Maker | **Lead:** Debashis Maharana | **Date:** March 2026

---

## Table of Contents

1. [Hackathon Overview](#1-hackathon-overview)
2. [Team Details](#2-team-details)
3. [Timeline](#3-timeline)
4. [Problem Statement & Requirements](#4-problem-statement--requirements)
5. [OpenEnv Specification](#5-openenv-specification)
6. [Idea Brainstorming](#6-idea-brainstorming)
7. [Deep Dive: CodingRL](#7-deep-dive-codingrl--swe-agent-environment)
8. [Deep Dive: MLRL](#8-deep-dive-mlrl--the-automl-agent)
9. [Deep Dive: InfraRL](#9-deep-dive-infrarl--production-incident-commander)
10. [Comparison & Recommendation](#10-comparison--recommendation)
11. [Resources](#11-resources)

---

## 1. Hackathon Overview

**Name:** Meta PyTorch OpenEnv Hackathon
**Hosted by:** Scaler School of Technology
**Dashboard URL:** `https://www.scaler.com/school-of-technology/meta-pytorch-hackathon/dashboard`

**What it is:** Build a complete, real-world **OpenEnv environment** that an AI agent can learn from through the standard `step()` / `reset()` / `state()` API.

**What OpenEnv is:** A framework/spec by Meta PyTorch for building RL environments as web services. Environments are deployed as Dockerized FastAPI apps on Hugging Face Spaces. Agents interact via HTTP/WebSocket endpoints.

---

## 2. Team Details

| Member | Email | Role | Status |
|---|---|---|---|
| **Debashis Maharana** | debashismaharana7854@gmail.com | Team Lead | Confirmed |
| **Gowtham Sai G** | gowthamyadav023@gmail.com | Member | Accepted |
| **Harsh Kumar** | harshkumar3446@gmail.com | Member | Accepted |

**Team Name:** Diff Maker
**Team Status:** Permanently locked. No changes allowed.

---

## 3. Timeline

| Phase | Dates | Status |
|---|---|---|
| Registration | 14th March – 3rd April | Done |
| Declaration (Team/Solo) | Before Round 1 | Done |
| Prepare (Course) | Now – 25th March | Done |
| **Round 1** | **25th March – 5th April** | **ACTIVE** |
| Submission Window Opens | **28th March** | |
| **Round 1 Deadline** | **7th April, 11:59 PM** | |
| Results | 10th April | |
| Finale | 25th–26th April | |

---

## 4. Problem Statement & Requirements

### The Task

> Build a complete, real-world OpenEnv environment that an AI agent can learn from through the standard `step()` / `reset()` / `state()` API.

### 7 Key Requirements

1. **Real-world task** — Must simulate a real-world task (not games or toys)
2. **Full OpenEnv spec** — Typed models, `step()`/`reset()`/`state()`, `openenv.yaml`
3. **3+ tasks with graders** — Easy → Medium → Hard, scores 0.0–1.0
4. **Meaningful reward function** — Partial progress signals (not binary pass/fail)
5. **Baseline inference script** — Reproducible scores
6. **Deploy to HF Spaces** — Working Dockerfile
7. **README** — Environment description, action/observation spaces, setup instructions

### Required Endpoints

| Endpoint | Purpose |
|---|---|
| `/baseline` | Trigger inference script, return baseline scores for all 3 tasks |
| `/grader` | Return grader score after an episode completes |
| `/tasks` | Return list of tasks and action schema (fields needed for a step) |

### Pre-Submission Checklist (Fail = Disqualified)

1. **HF Space deploys** — Automated ping must return 200 and respond to `reset()`
2. **OpenEnv spec compliance** — Validate `openenv.yaml`, typed models, `step()`/`reset()`/`state()` endpoints
3. **Dockerfile builds** — Automated docker build on submitted repo
4. **Baseline reproduces** — Inference script completes without error, produces scores
5. **3+ tasks with graders** — Enumerate tasks, run graders, verify scores in 0.0–1.0

### Evaluation Criteria

| Criterion | What They Check |
|---|---|
| **Runtime correctness** | Runs without errors |
| **Interface compliance** | Follows OpenEnv standard |
| **Task design** | Clear, realistic, testable |
| **Grading logic** | Reward system makes sense |

---

## 5. OpenEnv Specification

### 5.1 Project Structure

```
my_env/
├── models.py              # Action, Observation, State (Pydantic models)
├── client.py              # EnvClient subclass (WebSocket communication)
├── openenv.yaml           # Manifest file
├── pyproject.toml         # Package metadata
├── __init__.py
└── server/
    ├── environment.py     # Environment class: reset() / step() / state()
    ├── app.py             # FastAPI app via create_app()
    ├── Dockerfile
    ├── requirements.txt
    └── __init__.py
```

### 5.2 openenv.yaml

```yaml
spec_version: 1
name: my_env
type: space
runtime: fastapi
app: server.app:app
port: 8000
```

### 5.3 Typed Models (models.py)

Every environment defines 3 Pydantic models inheriting from OpenEnv base types:

```python
from openenv.core.env_server import Action, Observation, State

class MyAction(Action):
    # Action has: metadata: Dict[str, Any] = {}
    # Add your custom fields here
    pass

class MyObservation(Observation):
    # Observation has: done: bool = False, reward: Optional[float] = None, metadata: Dict = {}
    # Add your custom fields here
    pass

class MyState(State):
    # State has: episode_id: Optional[str] = None, step_count: int = 0
    # Add your custom fields here
    pass
```

### 5.4 Environment Logic (server/environment.py)

```python
from openenv.core.env_server import Environment

class MyEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._state = MyState()

    def reset(self, seed=None, episode_id=None, **kwargs) -> MyObservation:
        # Initialize new episode
        # Return first observation
        ...

    def step(self, action: MyAction, timeout_s=None, **kwargs) -> MyObservation:
        # Process action, update internal state
        # Return observation with done=True/False, reward=float
        ...

    @property
    def state(self) -> MyState:
        return self._state
```

### 5.5 App Setup (server/app.py)

```python
from openenv.core.env_server import create_app
from .environment import MyEnvironment
from ..models import MyAction, MyObservation

app = create_app(
    MyEnvironment,       # Pass the CLASS, not instance
    MyAction,
    MyObservation,
    env_name="my_env",
)
```

This auto-registers: `/ws`, `/reset`, `/step`, `/state`, `/health`, `/schema`, `/metadata`, `/web`, `/docs`

### 5.6 Client (client.py)

```python
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

class MyEnv(EnvClient[MyAction, MyObservation, MyState]):
    def _step_payload(self, action: MyAction) -> dict:
        return action.model_dump()

    def _parse_result(self, payload: dict) -> StepResult:
        obs = MyObservation(**payload.get("observation", {}))
        return StepResult(observation=obs, reward=payload.get("reward"), done=payload.get("done", False))

    def _parse_state(self, payload: dict) -> MyState:
        return MyState(**payload)
```

### 5.7 Usage Pattern

```python
# Async
async with MyEnv(base_url="https://...") as env:
    result = await env.reset()
    result = await env.step(MyAction(field="value"))

# Sync
with MyEnv(base_url="https://...").sync() as env:
    result = env.reset()
    result = env.step(MyAction(field="value"))
```

### 5.8 Baseline Inference Script Pattern

```python
env = MyEnv(base_url="https://...").sync()
result = env.reset()

for turn in range(max_turns):
    if result.done:
        break
    # Agent decides action (LLM call or heuristic)
    action = decide_action(result.observation)
    result = env.step(action)

final_score = result.observation.reward
```

### 5.9 Rubric / Grading System

OpenEnv uses a Rubric system for reward computation:

```python
from openenv.core.rubrics import Rubric

class MyRubric(Rubric):
    def forward(self, action, observation) -> float:
        # Return reward 0.0 - 1.0
        ...
```

Rubric types:
- **Rubric** — Per-step reward
- **TrajectoryRubric** — Episode-level scoring (accumulates steps, scores at end)
- **LLMJudge** — LLM-based grading with template prompts

### 5.10 Dockerfile

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 5.11 Existing Environments in OpenEnv

The `meta-pytorch/OpenEnv` repo has 29 environments including:

| Existing Env | Notes |
|---|---|
| `coding_env` | Basic sandboxed Python execution (just runs code) |
| `finrl_env` | Algorithmic trading |
| `chess_env` | Chess with configurable opponents |
| `openspiel_env` | 6 DeepMind games |
| `textarena_env` | Text games including Wordle |
| `browsergym_env` | Browser automation |

**Key insight:** `coding_env` and `finrl_env` already exist. Building something novel (MLRL, InfraRL) would stand out more.

---

## 6. Idea Brainstorming

### Initial Ideas Considered

| Idea | Domain | Verdict |
|---|---|---|
| Healthcare Clinical Decision | Medical | Good but domain expertise needed |
| Agriculture Crop Management | AgriTech | Clean physics model, feasible |
| Energy Grid Balancing | Energy | Good partial rewards |
| Personal Finance Advisor | Finance | Straightforward simulation |
| Disaster Response | Emergency | Engaging, visual |
| Drug Discovery | BioTech | Complex, needs domain knowledge |

### Shortlisted Ideas (Thinking Like a Meta AGI Engineer)

| Idea | Why It Matters for AGI |
|---|---|
| **CodingRL** | #1 AI benchmark (SWE-Bench). Every lab invests here. |
| **MLRL** | "Agent that trains agents" — recursive self-improvement |
| **InfraRL** | Every tech company's dream — automated ops |
| CodeReviewRL | Practical PR reviewer agent |
| DataScienceRL | End-to-end analysis automation |
| FinRL | Well-known but already exists in OpenEnv |

**Final 3 candidates: CodingRL, MLRL, InfraRL**

---

## 7. Deep Dive: CodingRL — SWE-Agent Environment

> An agent that navigates codebases, finds bugs, and fixes them. This is THE frontier of AI.

### 7.1 Concept

The environment presents a buggy Python project with failing tests. The agent must:
1. Read files to understand the codebase
2. Search for relevant code
3. Identify the root cause
4. Edit files to fix the bug
5. Run tests to verify the fix

### 7.2 Models

```python
class CodingAction(Action):
    command: str                  # "read_file", "edit_file", "run_tests", "search", "submit"
    path: Optional[str]          # file path
    content: Optional[str]       # edit content / search query
    line_start: Optional[int]    # for targeted edits
    line_end: Optional[int]

class CodingObservation(Observation):
    # done: bool, reward: float inherited
    output: str                  # file content / test output / search results
    current_file: Optional[str]
    test_results: Optional[dict] # {passed: int, failed: int, errors: list}
    file_tree: List[str]         # repo structure
    step_count: int
    message: str

class CodingState(State):
    # episode_id, step_count inherited
    bug_id: str
    repo_name: str
    files_modified: List[str]
    tests_run: int
    correct_file_found: bool
    correct_function_found: bool
    patch_applied: bool
    tests_passing: bool
```

### 7.3 Environment Logic

```
reset(task_id):
    1. Load a pre-built buggy Python project (stored as in-memory dict)
    2. Each "bug" = repo snapshot + failing tests + known fix
    3. Return Observation(file_tree, bug_description, failing_test_output)

step(action):
    "read_file"   → Return file contents from in-memory filesystem
    "search"      → Grep across all files, return matches
    "edit_file"   → Apply edit to in-memory filesystem
    "run_tests"   → Execute pytest via subprocess, return results
    "submit"      → Run full test suite, grade the patch, done=True
```

### 7.4 Bug Repository (Pre-built)

Bugs are hand-crafted small Python projects stored as dictionaries:

```python
BUGS = {
    "easy_off_by_one": {
        "files": {
            "calculator.py": "def average(nums):\n    return sum(nums) / (len(nums) - 1)\n",
            "test_calculator.py": "def test_average():\n    assert average([1,2,3]) == 2.0\n",
        },
        "description": "test_average fails: division gives wrong result",
        "fix_file": "calculator.py",
        "fix_line": 2,
        "fixed_content": "    return sum(nums) / len(nums)\n",
    },
    ...
}
```

### 7.5 Three Tasks + Graders

| Task | Difficulty | Bug Type | Example |
|---|---|---|---|
| **Task 1** | Easy | Single-file logic bug, clear traceback | Off-by-one in `average()`, 1 failing test |
| **Task 2** | Medium | Multi-file bug, misleading error | Import error masks a type bug in `utils.py`, traceback points to `main.py` |
| **Task 3** | Hard | Subtle edge case across 3 files | Empty list handling causes cascade, multiple failing tests |

**Grader Logic:**

```python
def grade_task(task_id, state):
    score = 0.0
    if state.correct_file_found:
        score += 0.2       # Found the right file
    if state.correct_function_found:
        score += 0.2       # Identified the right function
    if state.patch_applied:
        score += 0.2       # Made an edit that compiles
    if state.tests_passing:
        score += 0.4       # All tests pass
    return score            # 0.0 - 1.0
```

### 7.6 Project Structure

```
coding_rl/
├── models.py
├── client.py
├── openenv.yaml
├── server/
│   ├── environment.py       # Main reset/step/state
│   ├── bug_repository.py    # 3+ pre-made buggy projects (as dicts)
│   ├── sandbox.py           # In-memory filesystem + subprocess test runner
│   ├── grader.py            # Scoring logic per task
│   ├── app.py               # create_app()
│   ├── Dockerfile
│   └── requirements.txt
├── baseline.py              # Baseline inference script
└── README.md
```

### 7.7 Pros & Cons

| Pros | Cons |
|---|---|
| Directly on the AGI frontier (SWE-Bench) | Need to craft good bug scenarios |
| Very clear grading (tests pass/fail) | Subprocess execution adds complexity |
| Natural partial rewards | Basic `coding_env` already exists in OpenEnv |
| Judges will immediately understand | Need sandboxing for security |

---

## 8. Deep Dive: MLRL — The AutoML Agent

> An agent that explores data, selects models, tunes hyperparameters, and trains ML models. The "agent that trains agents."

### 8.1 Concept

The environment presents a dataset + ML task. The agent must:
1. Explore the data (stats, distributions, correlations)
2. Preprocess (handle missing values, normalize, encode)
3. Select a model (logistic regression, random forest, XGBoost, etc.)
4. Set hyperparameters
5. Train and evaluate
6. Submit the best model

### 8.2 Models

```python
class MLAction(Action):
    command: str                    # "explore_data", "preprocess", "select_model",
                                   # "set_params", "train", "evaluate", "submit"
    # For explore_data:
    column: Optional[str]          # specific column to inspect
    # For preprocess:
    strategy: Optional[str]        # "drop_na", "fill_mean", "fill_median",
                                   # "normalize", "encode_categorical", "smote"
    target_column: Optional[str]
    # For select_model:
    model_type: Optional[str]      # "logistic_regression", "random_forest",
                                   # "xgboost", "svm", "neural_net"
    # For set_params:
    params: Optional[Dict[str, Any]]  # {"n_estimators": 100, "max_depth": 5}

class MLObservation(Observation):
    # done: bool, reward: float inherited
    message: str
    data_summary: Optional[Dict]         # {shape, columns, dtypes, missing_pct, class_dist}
    column_stats: Optional[Dict]         # {mean, std, min, max, unique, nulls, histogram}
    preprocessing_log: List[str]         # actions taken so far
    model_selected: Optional[str]
    training_metrics: Optional[Dict]     # {train_acc, val_acc, train_loss, val_loss, time}
    evaluation_metrics: Optional[Dict]   # {test_acc, precision, recall, f1, confusion_matrix}
    compute_budget_remaining: float      # 0.0 - 1.0

class MLState(State):
    # episode_id, step_count inherited
    task_id: str
    dataset_name: str
    target_metric: str             # "accuracy", "f1", "rmse"
    target_threshold: float        # score to beat
    compute_used: float
    best_score: float
    current_model: Optional[str]
    preprocessing_applied: List[str]
```

### 8.3 Environment Logic

```
reset(task_id):
    1. Load a pre-built dataset (small CSV, embedded in Docker image)
    2. Split into train/val/test (fixed seed for reproducibility)
    3. Initialize compute budget = 1.0
    4. Return: data_summary (shape, columns, types, target variable, missing %)

step(action):
    "explore_data"  → Return column stats, correlations, distributions
                      Cost: 0.01 compute budget
    "preprocess"    → Apply transformation to internal dataframe
                      (drop_na, fill_mean, normalize, encode_categorical, smote)
                      Return: updated data_summary + log
                      Cost: 0.02 compute budget
    "select_model"  → Set current model type (stores choice, no compute)
    "set_params"    → Configure hyperparameters (stores params, no compute)
    "train"         → Train sklearn model on processed data
                      Return: train_acc, val_acc, training_time
                      Cost: varies (LR=0.05, RF=0.10, XGB=0.15, SVM=0.10, NN=0.25)
    "evaluate"      → Run current model on held-out test set
                      Return: test_acc, precision, recall, f1, confusion_matrix
                      Cost: 0.02
    "submit"        → Final evaluation, compute final score
                      done=True, reward = grader_score
```

### 8.4 Datasets (Pre-built, Embedded)

| Dataset | Rows | Features | Task | Difficulty |
|---|---|---|---|---|
| `easy_clean.csv` | 500 | 10 numeric, 2 categorical | Binary classification | No missing data, balanced classes, clear patterns |
| `medium_messy.csv` | 1000 | 15 mixed | Binary classification | 30% missing, 10:1 class imbalance, noisy features |
| `hard_shift.csv` | 2000 | 20 mixed | Multi-class (5) | Distribution shift between train/test, tight compute budget |

Datasets can be generated synthetically using `sklearn.datasets.make_classification()` with controlled properties.

### 8.5 Three Tasks + Graders

**Task 1: Easy — Clean Dataset**
```
Dataset: 500 rows, no missing values, balanced binary classification
Target metric: Accuracy
Threshold: 0.85
Grader: reward = min(test_accuracy / 0.85, 1.0)
```

**Task 2: Medium — Messy Dataset**
```
Dataset: 1000 rows, 30% missing, 10:1 class imbalance, noisy features
Target metric: F1 Score
Threshold: 0.70
Grader: reward = 0.7 * min(f1 / 0.70, 1.0) + 0.3 * compute_budget_remaining
```
Agent must handle missing values + imbalance. Penalized for wasting compute.

**Task 3: Hard — Distribution Shift**
```
Dataset: 2000 rows, multi-class, train/test distribution differs
Target metric: Macro F1
Threshold: 0.65
Compute budget: Only allows ~3 training runs
Grader: reward = 0.5 * min(macro_f1 / 0.65, 1.0)
                + 0.25 * compute_budget_remaining
                + 0.25 * (1.0 if generalization_gap < 0.05 else 0.0)
```
Agent must be smart about model/param choices (can't brute force).

### 8.6 ML Engine (Wraps scikit-learn)

```python
# ml_engine.py — All training happens in-process, small data = fast

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
# Optional: from xgboost import XGBClassifier

MODELS = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "svm": SVC,
    "neural_net": MLPClassifier,
}

COMPUTE_COSTS = {
    "logistic_regression": 0.05,
    "random_forest": 0.10,
    "xgboost": 0.15,
    "svm": 0.10,
    "neural_net": 0.25,
}

def train_model(model_type, params, X_train, y_train, X_val, y_val):
    model_cls = MODELS[model_type]
    model = model_cls(**params)
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    val_acc = model.score(X_val, y_val)
    return model, {"train_acc": train_acc, "val_acc": val_acc}

def evaluate_model(model, X_test, y_test):
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    preds = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "f1": f1_score(y_test, preds, average="macro"),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
    }
```

### 8.7 Project Structure

```
ml_rl/
├── models.py
├── client.py
├── openenv.yaml
├── server/
│   ├── environment.py       # Main reset/step/state
│   ├── ml_engine.py         # Wraps sklearn: train, evaluate, preprocess
│   ├── datasets/            # 3 pre-built CSV datasets (small, embedded)
│   │   ├── easy_clean.csv
│   │   ├── medium_messy.csv
│   │   └── hard_shift.csv
│   ├── dataset_generator.py # Script to generate datasets with controlled properties
│   ├── grader.py            # Scoring per task
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
├── baseline.py              # Baseline: explore → preprocess → train RF → evaluate
└── README.md
```

### 8.8 Baseline Inference Script

```python
# baseline.py — Simple heuristic agent
env = MLEnv(base_url="...").sync()

for task_id in ["easy", "medium", "hard"]:
    result = env.reset(task_id=task_id)

    # 1. Explore data
    result = env.step(MLAction(command="explore_data"))

    # 2. Basic preprocessing
    result = env.step(MLAction(command="preprocess", strategy="fill_mean"))
    result = env.step(MLAction(command="preprocess", strategy="normalize"))

    # 3. Select model & train
    result = env.step(MLAction(command="select_model", model_type="random_forest"))
    result = env.step(MLAction(command="set_params", params={"n_estimators": 100}))
    result = env.step(MLAction(command="train"))

    # 4. Evaluate & submit
    result = env.step(MLAction(command="evaluate"))
    result = env.step(MLAction(command="submit"))

    print(f"Task {task_id}: score = {result.observation.reward}")
```

### 8.9 Pros & Cons

| Pros | Cons |
|---|---|
| **Fully novel** — nothing like it in OpenEnv | Needs scikit-learn dependency |
| Fastest to build (~2-3 days) | Less "visual" than CodingRL |
| Cleanest grading (accuracy IS the score) | |
| "Agent that trains agents" — incredible pitch | |
| Minimal dependencies (sklearn + pandas) | |
| Training takes < 1 second (small datasets) | |
| Very natural partial rewards | |
| Docker image stays small | |

---

## 9. Deep Dive: InfraRL — Production Incident Commander

> An agent that diagnoses and resolves production incidents in a simulated microservice architecture.

### 9.1 Concept

The environment simulates a microservice system experiencing failures. The agent must:
1. Check metrics/logs to understand what's happening
2. Diagnose the root cause (not just symptoms)
3. Take remediation actions (rollback, scale, restart, failover)
4. Verify all services are healthy
5. Resolve the incident

### 9.2 Models

```python
class InfraAction(Action):
    command: str                # "check_metrics", "check_logs", "rollback",
                               # "scale", "restart", "failover_db",
                               # "update_config", "page_team", "resolve"
    service: Optional[str]     # "api-gateway", "user-service", "db-primary",
                               # "cache", "queue", "worker"
    params: Optional[Dict]     # {"replicas": 5}, {"version": "v2.3"}

class ServiceStatus(BaseModel):
    name: str
    status: str                # "healthy", "degraded", "down"
    error_rate: float          # 0.0 - 1.0
    latency_p99_ms: float
    cpu_percent: float
    memory_percent: float
    replicas: int
    version: str

class InfraObservation(Observation):
    # done: bool, reward: float inherited
    timestamp: str                          # simulated time (e.g., "14:32:30")
    services: Dict[str, ServiceStatus]      # health of each service
    alerts: List[str]                       # active alerts
    logs: Optional[List[str]]               # if check_logs was called
    metrics: Optional[Dict]                 # if check_metrics was called
    incident_duration_s: int                # time since incident started
    message: str

class InfraState(State):
    # episode_id, step_count inherited
    incident_id: str
    incident_type: str
    root_cause: str
    root_cause_service: str
    services: Dict[str, dict]
    time_elapsed_s: int
    sla_violated: bool
    actions_taken: List[str]
    correct_diagnosis: bool
    incident_resolved: bool
```

### 9.3 Microservice Architecture (Simulated)

```
                    ┌─────────────┐
                    │ api-gateway  │
                    └──────┬───────┘
                    ┌──────┴───────┐
              ┌─────┴─────┐  ┌─────┴──────┐
              │user-service│  │order-service│
              └─────┬──────┘  └──┬─────┬───┘
                    │           │     │
              ┌─────┴─────┐  ┌─┴──┐ ┌┴────┐
              │ db-primary │  │cache│ │queue│
              └────────────┘  └────┘ └──┬──┘
                                       │
                                   ┌───┴───┐
                                   │ worker │
                                   └───────┘
```

Each service is a state machine with: `(status, error_rate, latency, cpu, memory, replicas, version)`

**Cascading effects:** If `db-primary` slows down → `user-service` latency rises → `api-gateway` error rate increases → alerts fire

### 9.4 Environment Logic

```
reset(task_id):
    1. Initialize all 6 services as "healthy"
    2. Load a pre-scripted incident scenario
    3. Inject the failure (e.g., bad deploy on user-service)
    4. Propagate cascading effects
    5. Return: initial alerts + service statuses

step(action):
    # Advance simulated time by 30s per step
    # Propagate cascading effects through dependency graph

    "check_metrics" → Return CPU/memory/error_rate/latency for specified service
    "check_logs"    → Return simulated log lines (contain clues about root cause)
    "rollback"      → If correct service + version: error_rate drops
                      If wrong service: no effect
    "scale"         → Increase replicas → reduces latency if overloaded
    "restart"       → Fixes crashed service (but takes 60s = 2 steps)
    "failover_db"   → Switches to replica, fixes DB issues
    "update_config" → Change config (e.g., connection pool size)
    "resolve"       → Agent declares incident resolved
                      Grader checks: root cause fixed? All services healthy?
                      done=True, reward=score
```

### 9.5 Incident Scenarios (Pre-scripted)

**Incident 1: Bad Deploy (Easy)**
```
What happens: user-service v2.5 deployed with a bug → error_rate jumps to 0.8
Cascading: api-gateway starts returning 502s
Root cause: user-service bad deploy
Fix: rollback user-service to v2.4
Red herring: api-gateway looks sick but it's a symptom
```

**Incident 2: DB Connection Exhaustion (Medium)**
```
What happens: db-primary connection pool fills up → queries timeout
Cascading: user-service → slow responses → api-gateway → high latency → timeouts
Root cause: db-primary connection pool exhaustion
Fix: update_config db-primary connection_pool_size OR failover_db
Red herring: api-gateway and user-service show high latency (symptoms, not cause)
```

**Incident 3: Multi-Failure Cascading (Hard)**
```
What happens simultaneously:
  1. Bad deploy on order-service (new version has memory leak)
  2. Traffic spike → api-gateway overloaded
  3. Cache eviction storm → all requests hit db-primary
Root cause: 3 concurrent issues
Fix: rollback order-service + scale api-gateway + restart cache
Agent must identify and fix ALL three
```

### 9.6 Log Generator

```python
# log_generator.py — Template-based realistic logs

LOG_TEMPLATES = {
    "bad_deploy": [
        "[ERROR] {service} - NullPointerException in handleRequest()",
        "[WARN] {service} - Response time exceeded 5000ms",
        "[ERROR] {service} - Connection refused to downstream service",
        "[INFO] {service} - Version v{version} deployed at {time}",  # CLUE
    ],
    "db_exhaustion": [
        "[ERROR] {service} - Cannot acquire connection from pool",     # CLUE
        "[WARN] {service} - Active connections: {count}/100",          # CLUE
        "[ERROR] {service} - Query timeout after 30000ms",
        "[INFO] {service} - Retrying connection attempt {n}/3",
    ],
}
```

### 9.7 Three Tasks + Graders

| Task | Grader Logic |
|---|---|
| **Task 1 (Easy)** | `0.3` checked logs of correct service + `0.3` identified root cause service + `0.4` rollback fixed it. Time bonus: +0.1 if < 5 steps (capped at 1.0) |
| **Task 2 (Medium)** | `0.2` didn't just restart api-gateway (avoided red herring) + `0.3` found DB root cause + `0.2` applied correct fix + `0.3` all services recover |
| **Task 3 (Hard)** | `0.1` per correct root cause identified (×3 = 0.3) + `0.15` per correct remediation (×3 = 0.45) + `0.25` all services healthy at resolution |

### 9.8 Project Structure

```
infra_rl/
├── models.py
├── client.py
├── openenv.yaml
├── server/
│   ├── environment.py          # Main reset/step/state
│   ├── simulator.py            # Microservice graph + cascading effects engine
│   ├── incidents/              # Pre-scripted failure scenarios
│   │   ├── easy_bad_deploy.py
│   │   ├── medium_db_cascade.py
│   │   └── hard_multi_failure.py
│   ├── log_generator.py        # Generates realistic log lines with clues
│   ├── grader.py               # Scoring logic per task
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
├── baseline.py
└── README.md
```

### 9.9 Pros & Cons

| Pros | Cons |
|---|---|
| **Fully novel** — nothing like it in OpenEnv | Cascading simulation needs careful tuning |
| Every tech company wants this | Grading logic more subjective than MLRL |
| Very engaging (detective work!) | More code to write (simulator engine) |
| Pure Python — zero external dependencies | |
| Smallest Docker image | |
| Natural partial rewards (each correct diagnosis) | |

---

## 10. Comparison & Recommendation

### Side-by-Side

| Criterion | CodingRL | MLRL | InfraRL |
|---|---|---|---|
| **Novelty in OpenEnv** | Low (coding_env exists) | **Fully novel** | **Fully novel** |
| **Build time estimate** | 3–4 days | **2–3 days** | 3–4 days |
| **Dependencies** | Python + pytest | Python + sklearn + pandas | **Pure Python** |
| **Docker image size** | Small | Small (~50MB for sklearn) | **Smallest** |
| **AGI relevance** | Highest (SWE-Bench) | Very high (meta-learning) | High (ops automation) |
| **Judge "wow" factor** | "Everyone does this" | **"Agent trains agents!"** | "Every company needs this" |
| **Grading clarity** | Very clear (tests pass/fail) | **Clearest (accuracy = score)** | Good (services healthy) |
| **Partial rewards** | Natural | **Very natural** | Natural |
| **Complexity to build** | Medium (sandboxing) | **Low** | Medium (simulation) |
| **Pitch in 1 sentence** | "Agent debugs code like a senior dev" | **"Agent that builds ML models"** | "Agent that resolves production incidents" |

### Recommendation

**Primary: MLRL (The AutoML Agent)**

Reasons:
1. **Fastest to build** — sklearn handles all ML, we just wrap it
2. **Cleanest grading** — model accuracy IS the score, no ambiguity
3. **Most novel** — nothing like it exists in OpenEnv
4. **Best pitch** — "An agent that trains agents" is unforgettable
5. **Minimal dependencies** — sklearn + pandas, Docker stays lean
6. **Natural fit for OpenEnv** — explore → preprocess → train → evaluate maps perfectly to step()

**Backup: InfraRL** if we want something with zero ML dependencies and a more "detective story" feel.

**Not recommended: CodingRL** for this hackathon — basic `coding_env` already exists, and subprocess sandboxing adds complexity without enough differentiation time.

---

## 11. Resources

### Hackathon
- **Dashboard:** `https://www.scaler.com/school-of-technology/meta-pytorch-hackathon/dashboard`
- **Support Email:** help_openenvhackathon@scaler.com
- **Discord:** https://discord.gg/Dedhy5pkWD

### OpenEnv
- **Course Repo:** https://github.com/raun/openenv-course
- **Framework Repo:** https://github.com/meta-pytorch/OpenEnv
- **Module 4 (Building Environments):** Most important for Round 1

### Preparatory Course (~3.5 hours)
| Module | Topic | Duration | Priority |
|---|---|---|---|
| Module 1 | Why OpenEnv? | 45 min | Essential |
| Module 2 | Using Existing Environments | 50 min | Essential |
| Module 3 | Deploying Environments | 45 min | Essential |
| Module 4 | Building Your Own Environment | 60 min | **Most Important** |

### Key Files to Study
- `src/openenv/core/env_server/http_server.py` — HTTP endpoint registration
- `src/openenv/core/env_server/types.py` — Base Action/Observation/State types
- `src/openenv/core/env_server/interfaces.py` — Environment interface
- `src/openenv/core/rubrics/` — Grading system
- `src/openenv/cli/templates/` — Scaffold templates
- Any environment in `/envs/` — Reference implementations

---

## Next Steps

1. **Decide** which environment to build (recommendation: MLRL)
2. **Complete Module 4** of the prep course if not done
3. **Scaffold** the project using `openenv init`
4. **Implement** models → environment → grader → baseline → Dockerfile
5. **Test locally** with `uv run server`
6. **Deploy** to Hugging Face Spaces
7. **Run validator** before submitting
8. **Submit** before 7th April 11:59 PM

---

*Document generated from hackathon discussion — Team Diff Maker, March 2026*
