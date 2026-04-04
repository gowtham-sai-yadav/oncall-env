# OnCallEnv — How Everything Works

> A complete technical walkthrough of the system: data flow, internal parts, dependencies, and how every piece connects.

---

## 1. The Big Picture

OnCallEnv is an RL (Reinforcement Learning) environment. It simulates a production incident. An AI agent plays the role of an on-call engineer — investigating alerts, querying logs, finding the root cause, fixing services, and documenting the incident.

```
                         THE RL LOOP
                         ==========

    +--------+    action     +-------------+
    |        | ------------> |             |
    |  Agent |               |  OnCallEnv  |
    |  (LLM) | <------------ |  (Server)   |
    |        |  observation  |             |
    +--------+   + reward    +-------------+
                                   |
                              loads from
                                   |
                            +------+------+
                            |  Scenario   |
                            |   (JSON)    |
                            +-------------+
```

**One episode = one incident.** The agent takes actions, gets observations back, and at the end receives a reward score (0.0 to 1.0).

---

## 2. File Structure — What Each File Does

```
oncall-env/
|
|-- models.py                    [DATA TYPES]
|   Defines what Actions, Observations, and State look like.
|   Think of it as the "contract" between agent and environment.
|
|-- server/
|   |-- app.py                   [WEB SERVER]
|   |   Creates the FastAPI app. Exposes HTTP/WebSocket endpoints.
|   |   The front door — everything enters through here.
|   |
|   |-- environment.py           [BRAIN]
|   |   The core logic. Implements reset(), step(), state().
|   |   Contains all 13 action handlers.
|   |   This is where the simulation lives.
|   |
|   |-- graders.py               [JUDGE]
|   |   Computes the final reward (0.0-1.0).
|   |   6 components: triage, diagnostic, root_cause,
|   |   remediation, efficiency, documentation.
|   |
|   |-- rubric.py                [TRAINING BRIDGE]
|   |   Wraps graders.py into OpenEnv's Rubric system.
|   |   Needed so GRPO training can read per-step rewards.
|   |
|   |-- simulator.py             [PHYSICS ENGINE]
|   |   Makes the simulation dynamic:
|   |   - Services degrade over time if not fixed
|   |   - Fixing root cause heals downstream services
|   |
|   |-- scenario_loader.py       [FILE READER]
|   |   Loads scenario JSON files from the scenarios/ directory.
|   |
|   |-- Dockerfile               [CONTAINER]
|   |   Packages everything into a Docker image for deployment.
|   |
|   `-- test_oncall.py           [TESTS]
|       40 tests covering every component.
|
|-- scenarios/                   [INCIDENT DATA]
|   |-- task1_easy/              2 scenarios (alert triage)
|   |-- task2_medium/            2 scenarios (root cause diagnosis)
|   |-- task3_hard/              2 scenarios (full incident resolution)
|   `-- task4_expert/            2 scenarios (cascading + red herrings)
|
|-- client.py                    [REMOTE CLIENT]
|   For connecting to the server over WebSocket.
|   Used by inference.py and training scripts.
|
|-- inference.py                 [BASELINE AGENT]
|   An LLM-based agent that plays through incidents.
|   Uses OpenAI API to decide actions.
|
|-- openenv.yaml                 [MANIFEST]
|   Tells the OpenEnv framework about this environment.
|
`-- pyproject.toml               [PACKAGE CONFIG]
    Dependencies, entry points, package structure.
```

---

## 3. How a Single Episode Works (Step by Step)

### Phase A: Reset — Starting a New Incident

```
Agent calls: reset(task_id=2, scenario_idx=0)
                    |
                    v
        +-------------------+
        | scenario_loader   |
        | loads JSON file:  |
        | task2_medium/     |
        | scenario_001.json |
        +--------+----------+
                 |
                 v
        +-------------------+
        | environment.py    |
        | reset():          |
        |  1. Clear rubric  |    <-- _reset_rubric()
        |  2. Load scenario |    <-- load_scenario_by_task(2, 0)
        |  3. Deep copy:    |
        |     - alerts      |    <-- from scenario["initial_alerts"]
        |     - services    |    <-- from scenario["services"]
        |     - deployments |    <-- from scenario["recent_deployments"]
        |  4. Clear state   |
        |  5. Create episode|
        +--------+----------+
                 |
                 v
        Returns: OnCallObservation
          - alerts: [7 active alerts]
          - services: [6 services with status]
          - deployments: [recent deploys]
          - message: "Multiple services degraded..."
          - done: false
          - reward: null
          - metadata.reward_signals: {all zeros}
```

### Phase B: Step — Agent Takes an Action

```
Agent calls: step(action_type="query_logs", params={"service": "inventory-service"})
                    |
                    v
        +----------------------------------------------+
        | environment.py step():                       |
        |                                              |
        |  1. Check if already done (resolved or       |
        |     step_count >= 30)                        |
        |                                              |
        |  2. Sanitize params (truncate long strings)  |
        |                                              |
        |  3. Increment step_count                     |
        |                                              |
        |  4. Record action in actions_taken list      |
        |                                              |
        |  5. Find handler: _handle_query_logs()  -----+---> Handler executes
        |                                              |     Returns (message, extra_data)
        |  6. Add to timeline                          |
        |                                              |
        |  7. SIMULATE: degrade_services()  -----------+---> simulator.py
        |     (unhealthy services get 1.1x worse)      |     error_rate *= 1.1
        |                                              |     latency *= 1.1
        |  8. If remediation action AND service fixed: |     degraded->down if >80%
        |     propagate_recovery()  -------------------+---> downstream services recover
        |                                              |
        |  9. Check if done (resolved or step limit)   |
        |                                              |
        | 10. If done: compute final reward  ----------+---> graders.py grade_episode()
        |     If not done: reward = null               |
        |                                              |
        | 11. Build observation  ----------------------+---> _make_observation()
        |     (includes reward_signals in metadata)    |     _compute_step_reward_signals()
        |                                              |
        | 12. Apply rubric  ---------------------------+---> rubric.py
        |     (accumulates trajectory for training)    |     TrajectoryRubric.forward()
        |                                              |
        +----------------------------------------------+
                    |
                    v
        Returns: OnCallObservation
          - alerts: [updated alert states]
          - services: [updated service states - degraded further]
          - log_results: [matching log entries]  <-- from handler
          - message: "Found 5 log entries..."
          - done: false
          - reward: null
          - metadata.reward_signals: {
              oncall.triage_progress: 0.0,
              oncall.investigation_depth: 0.25,  <-- queried 1 of 4 expected
              oncall.premature_action: 0.0,
              oncall.severity_set: 0.0,
              oncall.summary_written: 0.0,
              oncall.resolved: 0.0
            }
```

### Phase C: Episode Ends — Agent Resolves

```
Agent calls: step(action_type="resolve_incident", params={"resolution_note": "Fixed"})
                    |
                    v
        +----------------------------------------------+
        | environment.py step():                       |
        |                                              |
        |  Handler: _handle_resolve_incident()         |
        |    -> self._resolved = True                  |
        |                                              |
        |  done = True (self._resolved is True)        |
        |                                              |
        |  Compute final reward:                       |
        |    grade_episode() -----+                    |
        +-------------------------+--------------------+
                                  |
                                  v
        +----------------------------------------------+
        | graders.py grade_episode():                  |
        |                                              |
        |  Reads weights from scenario grading_rubric  |
        |                                              |
        |  Computes 6 components:                      |
        |                                              |
        |  1. TRIAGE (0.0-1.0)                         |
        |     - Were critical alerts acknowledged?     |
        |     - Was severity set correctly?            |
        |     - Were info alerts silenced?             |
        |     - Penalty for silencing critical alerts  |
        |                                              |
        |  2. DIAGNOSTIC (0.0-1.0)                     |
        |     - Did agent query expected logs/metrics? |
        |     - Partial credit for right action type   |
        |                                              |
        |  3. ROOT CAUSE (0.0-1.0)                     |
        |     - Does summary mention root cause svc?   |
        |     - Does summary contain keywords?         |
        |     - Did remediation target right service?  |
        |                                              |
        |  4. REMEDIATION (0.0-1.0)                    |
        |     - Was a valid fix applied?               |
        |     - Are services healthy now?              |
        |     - Was incident resolved?                 |
        |     - Penalty: restarting healthy services   |
        |     - Penalty: resolving without investigate |
        |                                              |
        |  5. EFFICIENCY (0.0-1.0)                     |
        |     <=10 steps: 1.0                          |
        |     11-15: 0.7, 16-20: 0.4, >20: 0.2       |
        |                                              |
        |  6. DOCUMENTATION (0.0-1.0)                  |
        |     - Summary word count                     |
        |     - Technical keywords present             |
        |                                              |
        |  Final = weighted sum (weights from scenario)|
        |  Clamped to [0.0, 1.0]                       |
        +----------------------------------------------+
                    |
                    v
        Returns: OnCallObservation
          - done: true
          - reward: 0.72  (example)
          - metadata.reward_signals: {all updated}
```

---

## 4. The 13 Actions — What Each One Does Internally

```
+---------------------------+----------+------------------------------------------+
| Action                    | Category | What Happens Inside                      |
+---------------------------+----------+------------------------------------------+
| query_logs                | INVEST.  | Searches scenario["logs"] by service &   |
|                           |          | level. Returns matching log entries.      |
+---------------------------+----------+------------------------------------------+
| check_metrics             | INVEST.  | Looks up scenario["metrics"] by          |
|                           |          | "service:metric_name". Returns time      |
|                           |          | series data.                             |
+---------------------------+----------+------------------------------------------+
| view_dependencies         | INVEST.  | Returns scenario["dependencies"] for     |
|                           |          | the given service (who it depends on).   |
+---------------------------+----------+------------------------------------------+
| acknowledge_alert         | TRIAGE   | Sets alert["acknowledged"] = true.       |
|                           |          | Good practice for critical alerts.       |
+---------------------------+----------+------------------------------------------+
| silence_alert             | TRIAGE   | Sets alert["silenced"] = true.           |
|                           |          | Good for info alerts. BAD for critical.  |
+---------------------------+----------+------------------------------------------+
| restart_service           | FIX      | If in valid_remediations: sets service   |
|                           |          | to healthy, error_rate=0. If not valid:  |
|                           |          | partial improvement (down->degraded).    |
+---------------------------+----------+------------------------------------------+
| scale_service             | FIX      | If valid: sets healthy, error_rate*0.2.  |
|                           |          | If not valid: no significant change.     |
+---------------------------+----------+------------------------------------------+
| rollback_deploy           | FIX      | If valid: sets healthy, error_rate=0,    |
|                           |          | updates version. If not: "issues         |
|                           |          | persist."                                |
+---------------------------+----------+------------------------------------------+
| update_config             | FIX      | If valid (matches action+service+key):   |
|                           |          | sets healthy, error_rate=0.              |
+---------------------------+----------+------------------------------------------+
| set_severity              | DOC      | Stores severity level (SEV1-SEV4).       |
|                           |          | Compared against expected_severity.      |
+---------------------------+----------+------------------------------------------+
| write_summary             | DOC      | Stores incident summary text.            |
|                           |          | Graded on length + keywords.             |
+---------------------------+----------+------------------------------------------+
| escalate                  | COMM     | Records which team was escalated to.     |
+---------------------------+----------+------------------------------------------+
| resolve_incident          | CLOSE    | Sets _resolved=true. Triggers final      |
|                           |          | reward computation. Episode ends.        |
+---------------------------+----------+------------------------------------------+
```

---

## 5. The Scenario JSON — What's Inside

Each scenario JSON is a self-contained incident definition. Here's the structure:

```
scenario_001.json
|
|-- incident_id: "INC-20260326-001"          Unique ID
|-- difficulty: "easy"                       easy/medium/hard/expert
|-- description: "Multiple alerts..."        What agent sees first
|-- expected_severity: "SEV2"                Correct severity to set
|
|-- initial_alerts: [                        THE ALERTS (what fires)
|     { alert_id, severity, service,
|       message, timestamp,
|       acknowledged: false,
|       silenced: false }
|     ...5-10 alerts
|   ]
|
|-- services: [                              THE SERVICES (what's running)
|     { name, status, latency_ms,
|       error_rate, cpu_percent,
|       memory_percent, version }
|     ...3-9 services
|   ]
|
|-- recent_deployments: [                    RECENT DEPLOYS (clues)
|     { service, version, timestamp,
|       deployer }
|   ]
|
|-- logs: {                                  LOGS DATABASE
|     "payment-api": [                       (queried by agent)
|       { timestamp, service, level,
|         message }                          <-- these contain CLUES
|     ],
|     "order-service": [...],
|     ...
|   }
|
|-- metrics: {                               METRICS DATABASE
|     "payment-api:error_rate":              (queried by agent)
|       [0.1, 0.2, 5.8, 12.3],             <-- time series
|     "payment-api:latency_p99":
|       [150, 450, 2800, 3450],
|     ...
|   }
|
|-- dependencies: {                          SERVICE DEPENDENCY GRAPH
|     "payment-api":                         (queried by agent)
|       ["payments-db", "stripe-gateway"],
|     "order-service":
|       ["payment-api", "inventory-service"],
|     ...
|   }
|
|-- root_cause: {                            THE ANSWER (used by grader)
|     service: "payment-api",
|     description: "DB connection pool...",
|     keywords: ["connection pool",
|       "payment", "database", "leak"]
|   }
|
|-- valid_remediations: [                    CORRECT FIXES (used by grader)
|     { action: "restart_service",
|       service: "payment-api" },
|     { action: "rollback_deploy",
|       service: "payment-api" }
|   ]
|
|-- expected_diagnostics: [                  EXPECTED INVESTIGATION STEPS
|     { action_type: "query_logs",           (used by grader)
|       params: {service: "payment-api"} },
|     { action_type: "check_metrics",
|       params: {service: "payment-api"} }
|   ]
|
|-- grading_rubric: {                        WEIGHT CONFIGURATION
|     triage_weight: 0.30,                   (differs per difficulty)
|     diagnostic_weight: 0.20,
|     root_cause_weight: 0.15,
|     remediation_weight: 0.15,
|     efficiency_weight: 0.10,
|     documentation_weight: 0.10
|   }
```

**The agent NEVER sees:** `root_cause`, `valid_remediations`, `expected_diagnostics`, `grading_rubric`. These are only used by the grader.

**The agent DOES see:** `initial_alerts`, `services`, `recent_deployments`, `description`. And can query for: `logs`, `metrics`, `dependencies`.

---

## 6. The Dynamic Simulation — What Happens Each Step

```
        BEFORE step()          DURING step()           AFTER step()
        ============          =============           ============

    payment-api:            Handler runs:           simulator.py runs:
      status: degraded        (agent queries         degrade_services():
      error: 12.3%            logs, etc.)              payment-api:
      latency: 3450ms                                    error: 12.3 * 1.1 = 13.5%
                                                         latency: 3450 * 1.1 = 3795ms

    order-service:          (nothing changes          order-service:
      status: degraded        for order-service        error: 15.0 * 1.1 = 16.5%
      error: 15.0%            from this action)        latency: 890 * 1.1 = 979ms

    user-service:                                     user-service:
      status: healthy         (healthy services          (no change - healthy)
      error: 0.1%             don't degrade)

    IF agent fixes payment-api (valid remediation):

    payment-api:            _handle_restart_service:  propagate_recovery():
      status: degraded  -->   status: healthy     -->   check: does order-service
      error: 13.5%            error: 0.0%               depend on payment-api?
                                                        YES -> but also depends on
                                                        inventory-service (still sick)
                                                        -> NO recovery (not all deps ok)
```

**Key rule:** A downstream service only recovers if ALL its dependencies are healthy, not just one.

---

## 7. The Grading System — How Reward Is Computed

```
    FINAL REWARD = weighted sum of 6 components
    ============

    +-----------------+   Each scores 0.0 to 1.0
    |                 |   Weights come from scenario JSON
    |   TRIAGE        |   (differ per difficulty level)
    |   weight: 0.15  |
    +-----------------+        Easy tasks: triage_weight = 0.30
            +                  Hard tasks: root_cause_weight = 0.30
    +-----------------+        Expert tasks: root_cause_weight = 0.35
    |   DIAGNOSTIC    |
    |   weight: 0.25  |
    +-----------------+
            +
    +-----------------+
    |   ROOT CAUSE    |
    |   weight: 0.25  |
    +-----------------+
            +                  Example:
    +-----------------+        triage=0.8, diagnostic=0.6, root_cause=0.7
    |   REMEDIATION   |        remediation=0.9, efficiency=1.0, documentation=0.5
    |   weight: 0.20  |
    +-----------------+        reward = 0.8*0.15 + 0.6*0.25 + 0.7*0.25
            +                         + 0.9*0.20 + 1.0*0.10 + 0.5*0.05
    +-----------------+                = 0.12 + 0.15 + 0.175
    |   EFFICIENCY    |                + 0.18 + 0.10 + 0.025
    |   weight: 0.10  |                = 0.75
    +-----------------+
            +
    +-----------------+
    |   DOCUMENTATION |
    |   weight: 0.05  |
    +-----------------+
            =
    +-----------------+
    |  REWARD: 0.75   |   clamped to [0.0, 1.0]
    +-----------------+
```

---

## 8. The Rubric System — Bridge to PyTorch Training

```
    WITHOUT RUBRIC (evaluation only):

    step() -> grade_episode() -> single reward at episode end
              (only when done=True)


    WITH RUBRIC (training compatible):

    step() -> _apply_rubric(action, obs)
                    |
                    v
              TrajectoryRubric.forward()
                    |
                    +-- Stores (action, obs) in trajectory list
                    |
                    +-- If obs.done=True:
                    |     calls score_trajectory()
                    |       -> delegates to grade_episode()
                    |       -> returns final score
                    |
                    +-- If obs.done=False:
                          returns 0.0 (intermediate_reward)


    AFTER EPISODE (for GRPO training):

    rubric.compute_step_rewards()
      -> [R/N, R/N, R/N, ..., R/N]   (uniform credit assignment)
         where R = final score, N = number of steps

    rubric.trajectory
      -> [(action1, obs1), (action2, obs2), ...]
```

**Why this matters:** The GRPO training algorithm (from TRL/PyTorch) needs per-step rewards to compute gradients. Without the rubric, it only gets a reward at the very last step — making training extremely slow. The rubric distributes credit to every step.

---

## 9. Per-Step Reward Signals — What They Mean

Every observation includes `metadata.reward_signals` — a dict of intermediate progress metrics:

```
metadata.reward_signals = {

    "oncall.triage_progress": 0.667
    |   How many critical alerts have been acknowledged?
    |   = acknowledged_count / total_critical_alerts
    |   Goes from 0.0 to 1.0 as agent acks alerts.

    "oncall.investigation_depth": 0.5
    |   How many expected diagnostic steps completed?
    |   = completed_diagnostics / expected_diagnostics
    |   Goes from 0.0 to 1.0 as agent queries logs/metrics.

    "oncall.premature_action": 0.0
    |   Did agent remediate WITHOUT investigating first?
    |   0.0 = good (investigated first, or hasn't remediated yet)
    |   -0.5 = bad (remediated without any investigation)

    "oncall.severity_set": 1.0
    |   Has agent set a severity level?
    |   0.0 = not set, 1.0 = set

    "oncall.summary_written": 0.0
    |   Has agent written an incident summary?
    |   0.0 = no, 1.0 = yes

    "oncall.resolved": 0.0
    |   Has agent resolved the incident?
    |   0.0 = no, 1.0 = yes
}
```

These are NOT part of the final reward. They exist so a training algorithm can give the agent feedback DURING the episode, not just at the end.

---

## 10. The HTTP Endpoints — How External Systems Interact

```
    STANDARD OPENENV ENDPOINTS (auto-generated by create_app):

    POST /reset           Start new episode
    POST /step            Execute an action
    GET  /state           Query current state
    GET  /health          Health check -> {"status": "healthy"}
    GET  /schema          JSON schemas for Action/Observation/State
    GET  /metadata        Environment info (name, version, etc.)
    GET  /docs            Swagger/OpenAPI documentation
    WS   /ws              WebSocket for persistent sessions
    GET  /web             Gradio interactive UI (if enabled)


    CUSTOM HACKATHON ENDPOINTS (added in app.py):

    GET  /tasks           List all 4 tasks + action schema
    POST /grader          Grade a completed episode
    POST /baseline        Run heuristic baseline, return scores
```

---

## 11. The Dependency Graph — How Files Connect

```
                    +----------------+
                    |  openenv-core  |   External framework
                    | (pip package)  |   Provides: Environment, Action,
                    +-------+--------+   Observation, State, create_app,
                            |            TrajectoryRubric, EnvClient
            +---------------+----------------+
            |               |                |
      +-----+-----+  +-----+------+  +------+------+
      | models.py  |  | client.py  |  | app.py      |
      | (types)    |  | (remote    |  | (FastAPI    |
      |            |  |  access)   |  |  server)    |
      +-----+------+  +------------+  +------+------+
            |                                |
            |          +---------------------+
            |          |
      +-----+----------+------+
      |  environment.py        |     THE CORE
      |  (reset/step/state)    |
      +---+------+------+-----+
          |      |      |
    +-----+  +---+--+  ++--------+
    |        |      |  |         |
+---+----+ +-+------+-+ +-------+-+  +----------+
|rubric  | |graders.py| |simulator| |scenario   |
|.py     | |(reward   | |.py      | |_loader.py |
|(bridge | | logic)   | |(physics)| |(file I/O) |
| to     | |          | |         | |           |
| GRPO)  | +----------+ +---------+ +-----+-----+
+--------+                               |
                                   +------+------+
                                   | scenarios/  |
                                   | *.json      |
                                   | (8 files)   |
                                   +-------------+
```

---

## 12. Data Types — What Flows Between Components

```
    AGENT SENDS:
    +------------------+
    | OnCallAction     |
    |   action_type:   |  "query_logs", "restart_service", etc.
    |     Literal[13]  |
    |   params:        |  {"service": "payment-api", ...}
    |     dict         |
    +------------------+


    ENVIRONMENT RETURNS:
    +-------------------------+
    | OnCallObservation       |
    |   alerts:               |  List[Alert]         - current alert states
    |   services:             |  List[ServiceStatus] - current service health
    |   recent_deployments:   |  List[Deploy]        - deploy history
    |   log_results:          |  List[LogEntry]|None - from query_logs
    |   metric_results:       |  dict|None           - from check_metrics
    |   dependency_graph:     |  dict|None           - from view_dependencies
    |   incident_timeline:    |  List[Event]         - action history
    |   current_severity:     |  str|None            - set by agent
    |   available_actions:    |  List[str]           - 13 action types
    |   message:              |  str                 - feedback text
    |   done:                 |  bool                - episode over?
    |   reward:               |  float|None          - only when done=True
    |   metadata:             |  dict                - contains reward_signals
    +-------------------------+


    ENVIRONMENT STATE (internal):
    +-------------------------+
    | OnCallState             |
    |   episode_id:           |  str     - unique episode ID
    |   step_count:           |  int     - current step number
    |   task_id:              |  str     - "task1", "task2", etc.
    |   scenario_id:          |  str     - "INC-20260326-001"
    |   incident_resolved:    |  bool    - has agent resolved?
    |   actions_taken:        |  list    - full action history
    +-------------------------+


    INTERNAL (never sent to agent):
    +-------------------------+
    | self._scenario          |  The full JSON dict
    | self._alerts            |  list[dict] - mutable alert state
    | self._services          |  list[dict] - mutable service state
    | self._severity          |  str|None
    | self._summary           |  str
    | self._escalated_to      |  str|None
    | self._resolved          |  bool
    | self._actions_taken     |  list[dict] - action log with step numbers
    +-------------------------+
```

**Important design note:** Internal state is `list[dict]` (raw Python dicts). The observation type uses `List[Alert]`, `List[ServiceStatus]`, etc. Pydantic v2 automatically converts dicts to typed models when building the observation. The grader receives raw dicts (not typed models) — this is intentional to keep the grader simple and avoid breakage.

---

## 13. A Complete Episode Walkthrough (Task 2, Medium)

```
SCENARIO: inventory-service v2.5.1 memory leak -> inventory-db OOM-killed -> cascade

STEP  ACTION                              WHAT HAPPENS                              SIGNALS
----  ------                              ------------                              -------
 0    reset(task_id=2)                    Load scenario. 6 alerts fire.             triage: 0.0
                                          4 services degraded, 1 down.              invest: 0.0

 1    acknowledge_alert("alert-201")      api-gateway alert ack'd                   triage: 0.33
                                          (it's critical)                           invest: 0.0

 2    acknowledge_alert("alert-202")      order-service alert ack'd                 triage: 0.67
                                                                                    invest: 0.0

 3    set_severity("SEV1")                Severity stored                           severity: 1.0

 4    view_dependencies("order-service")  Returns: [inventory-service,              invest: 0.25
                                          payment-api, order-db]
                                          CLUE: order-service depends on
                                          inventory-service!

 5    query_logs("order-service")         Returns: "Inventory check failed          invest: 0.5
                                          for order: upstream 503"
                                          CLUE: inventory-service is the problem

 6    query_logs("inventory-service")     Returns: "OOM killer invoked",            invest: 0.75
                                          "max_connections reached",
                                          "memory exceeded limit"
                                          ROOT CAUSE FOUND!

 7    check_metrics("inventory-service")  Returns: memory 55->99%,                  invest: 1.0
                                          db_connections 40->100 (saturated)

 8    rollback_deploy("inventory-service", Inventory-service -> healthy              premature: 0.0
       "v2.5.0")                          error_rate -> 0.0                         (investigated first!)
                                          THEN: propagate_recovery runs
                                          order-service starts recovering

 9    write_summary("Root cause:          Summary stored                            summary: 1.0
       inventory-service v2.5.1 memory
       leak in DB connection pool...")

10    resolve_incident("Rolled back")     _resolved = True                          resolved: 1.0
                                          DONE! grade_episode() runs

FINAL REWARD COMPUTATION:
  triage:        0.67 * 0.10 = 0.067   (ack'd 2/3 critical, set severity)
  diagnostic:    0.85 * 0.30 = 0.255   (queried 3/4 expected diagnostics)
  root_cause:    0.90 * 0.30 = 0.270   (summary mentions inventory, keywords match)
  remediation:   0.80 * 0.15 = 0.120   (valid rollback, resolved, not all healthy)
  efficiency:    1.00 * 0.10 = 0.100   (10 steps <= 10, perfect)
  documentation: 0.80 * 0.05 = 0.040   (30+ words, tech keywords present)
                                -----
  TOTAL REWARD:                0.852    (clamped to 1.0 max)
```

---

## 14. Supplementary Metrics — EGAR and Blast Radius

These are computed alongside the main reward but DON'T affect it. They exist for analysis.

```
EGAR (Evidence-Gated Action Rate):
  "For each remediation, did you investigate that SPECIFIC service first?"

  Example:
    Step 6: query_logs(inventory-service)     <-- investigation
    Step 8: rollback_deploy(inventory-service) <-- remediation

    EGAR = 1/1 = 1.0 (investigated before every remediation)

  Bad example:
    Step 1: restart_service(payment-api)       <-- remediation with NO investigation
    Step 2: restart_service(order-service)     <-- another blind restart

    EGAR = 0/2 = 0.0 (never investigated before acting)


BLAST RADIUS:
  "What fraction of your remediations targeted the WRONG service?"

  Example:
    Valid remediations: [{restart, inventory-service}]
    Agent did: restart(inventory-service)
    Blast radius = 0/1 = 0.0 (perfect)

  Bad example:
    Agent did: restart(payment-api), restart(order-service), restart(inventory-service)
    Blast radius = 2/3 = 0.67 (2 out of 3 were wrong targets)
```

---

## 15. How Training Would Work (Phase 7 — Future)

```
                 GRPO TRAINING LOOP
                 ==================

    +------------+     prompts     +-------------------+
    | Training   | --------------> |  LLM (e.g.        |
    | Pipeline   |                 |  Qwen 1B)         |
    | (TRL)      | <-------------- |                   |
    |            |   completions   +-------------------+
    +-----+------+
          |
          | completions are actions
          |
    +-----v------+
    | OnCallEnv  |     step() returns observation
    | (our env)  |     + reward_signals in metadata
    +-----+------+
          |
          | reward signals
          |
    +-----v------+
    | Reward      |     Reads metadata.reward_signals
    | Functions   |     Returns per-signal float
    +-----+------+
          |
          | multi-dimensional rewards
          |
    +-----v------+
    | GRPO Loss   |     Computes advantages from rewards
    | + Optimizer |     Updates model weights
    +-------------+

    After N training epochs:
      LLM gets BETTER at incident response
      (investigates before acting, targets right services, etc.)
```

This requires PyTorch + GPU. The `train.py` script we'll create is a proof-of-concept showing this is possible with our environment.

---

## 16. Quick Reference — What Happens When

| Event | What Gets Called | Key Files |
|---|---|---|
| Agent connects via WebSocket | `create_app()` creates new `OnCallEnvironment()` per connection | app.py |
| Agent calls `reset(task_id=2)` | `environment.reset()` -> `scenario_loader.load_scenario_by_task(2,0)` | environment.py, scenario_loader.py |
| Agent sends action | `environment.step()` -> handler -> simulator -> grader (if done) -> rubric | environment.py, simulator.py, graders.py, rubric.py |
| Agent queries `/tasks` | Custom FastAPI endpoint returns task list + action schema | app.py |
| Agent queries `/baseline` | Runs heuristic agent inline, returns scores | app.py |
| Episode ends | `grade_episode()` computes 6-component reward. `rubric.compute_step_rewards()` distributes credit. | graders.py, rubric.py |
| Training pipeline calls env | `rubric.trajectory` provides full episode data. `compute_step_rewards()` gives per-step credits. | rubric.py |

---

*This document reflects the codebase after Phases 0-5. 40 tests passing.*
