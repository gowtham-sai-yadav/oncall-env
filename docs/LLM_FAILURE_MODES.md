# How Frontier LLMs Fail at Incident Response — Research Findings

> Research from 5 parallel agents covering: GPT-5 failure modes, tool-calling errors, incident response AI, cognitive blind spots, and multi-turn context failures. All findings backed by published research.

---

## The 10 Failure Modes That GPT-5.2 Cannot Overcome

### 1. ACTS BEFORE INVESTIGATING (The #1 Problem)

**What happens:** The model takes remediation actions (restart, rollback) BEFORE gathering sufficient evidence about what's actually wrong.

**The data:**
- OpenSec benchmark: GPT-5.2 executes containment within **6.95 steps** on average — acting before evidence is gathered
- GPT-5.2 has **97% false positive rate** on containment actions (OpenSec, arXiv 2601.21083)
- Gemini 3: also 97% FP rate. DeepSeek 3.2: 90%. Even Claude Sonnet 4.5: 72%
- Adding an explicit instruction "investigate first" improved one model's accuracy by **+34.58 percentage points** (KAMI benchmark) — proving the default behavior is to act without investigating

**Why it matters for us:** Our GPT-5.2 run showed this: Task 1, step 10 — it rolled back payment-api after only querying logs once. It got lucky because the answer was obvious. On harder tasks, this premature action pattern would cause wrong remediations.

**How to exploit this:**
- Hide detailed metrics from observation (force investigation)
- Make the "obvious" service NOT the root cause
- Have multiple recently-deployed services so the model can't just pattern-match "recent deploy = root cause"

---

### 2. ANCHORING ON FIRST EVIDENCE (Can't Let Go of Initial Hypothesis)

**What happens:** The model forms a hypothesis from the FIRST thing it sees and then seeks confirming evidence, ignoring contradicting evidence.

**The data:**
- In 93.2% of cases, model outputs shifted in the direction of the provided anchor (Springer, 2025)
- STRONGER models showed GREATER anchoring: GPT-4/4o produced fully anchored responses in 72% of cases vs 25% for GPT-3.5
- Chain-of-Thought, "ignore the hint" instructions, and reflection prompting were **all ineffective** at reducing anchoring
- Security code review: when vulnerable code was framed as "secure" via PR metadata, GPT-4o-mini's detection dropped from **97.2% to 3.6%** (-93.5 percentage points)

**Why it matters for us:** In our Task 4 output, the model saw `internal-dns: degraded` in the first observation and immediately investigated DNS. It happened to be right. But if we put a MORE visibly broken service first (that's actually a symptom, not the cause), the model would anchor on it and chase the wrong path.

**How to exploit this:**
- Put the ROOT CAUSE service at the BOTTOM of the service list, with moderate degradation
- Put a SYMPTOM service at the TOP with severe degradation
- Make the first log entries from the wrong service
- The model will anchor on whatever it sees first

---

### 3. CONFUSES CORRELATION WITH CAUSATION

**What happens:** "Service X was deployed recently AND Service X is broken" → model concludes the deploy caused the problem. But sometimes Service X is broken because of Service Y, and the deploy is coincidental.

**The data:**
- Only 30% of standard LLMs achieved ideal causal structures in reasoning (arXiv, 2025)
- GPT-4 achieved only F1=29.08 on the Corr2Cause benchmark — described as "causal parroting"
- On logic tasks, Chain-of-Thought causality was only **0.3%** — meaning CoT had almost zero causal influence on answers

**Why it matters for us:** In EVERY task, the model did: "see degraded service → check recent deploys → rollback." This worked because our scenarios have a direct deploy→failure correlation. A harder scenario would have:
- Service A deployed recently (but deploy is fine)
- Service B is the actual root cause (no recent deploy, just a config drift or cert expiry)
- Service A is degraded because it depends on Service B

**How to exploit this:**
- Decouple deploy timing from root cause
- Have the root cause be something that WASN'T recently deployed (cert expiry, config drift, resource exhaustion over time)
- Have a recently deployed service that's a victim, not the cause

---

### 4. CANNOT SAY "I DON'T HAVE ENOUGH INFORMATION"

**What happens:** The model NEVER pauses to acknowledge uncertainty. It always produces a confident diagnosis, even when evidence is insufficient.

**The data:**
- GPT-4 assigned 10/10 confidence to **87% of its responses** including wrong ones
- Reasoning fine-tuning degrades abstention ability by **24% on average** (AbstentionBench, 2025)
- Claude at 90%+ stated confidence had actual accuracy of only 70% (+24.6% overconfidence gap)
- Belief updates are **5.5x larger than optimal** — models overreact to weak evidence and underreact to strong evidence equally

**Why it matters for us:** Our GPT-5.2 run never once said "I need more information." It always produced a confident next action. On Task 2, it dumped ALL actions in one response — a sign of overconfidence ("I already know the answer, let me do everything at once").

**How to exploit this:**
- Create scenarios where the first 3-4 diagnostic queries return AMBIGUOUS results
- The agent should need 5+ investigation steps before having enough evidence to diagnose
- Penalize remediation attempts made before a minimum number of diagnostic steps
- Add scenarios where the correct answer is "escalate" (not enough info to fix)

---

### 5. LOST IN THE MIDDLE — MISSES CRITICAL DETAILS IN LONG OBSERVATIONS

**What happens:** When observations are long (many alerts, many services, many log entries), the model misses critical details that appear in the middle of the text.

**The data:**
- Models show 20-30% accuracy drop for information positioned in the middle of context (Liu et al., 2023)
- With 5+ needles distributed across 100K tokens, retrieval drops to 40-60%
- In SWE-bench, agents read entire files but missed the relevant function or error
- Accuracy drops from ~95% with 3 items to ~70% with 20+ items on triage-like tasks

**Why it matters for us:** Our Task 4 observation has 10 alerts and 9 services. The model processed it well — but only because the critical clue (internal-dns: degraded) was fairly visible. If we buried the root cause signal among 15-20 services and 12-15 alerts, the model would likely miss it.

**How to exploit this:**
- Increase the number of services to 12-15 per scenario (not 5-7)
- Increase alerts to 12-15 per scenario
- Put the root cause service in the MIDDLE of the list, not near the top
- Add many "noise" services that are healthy but present
- Make log entries for the root cause service SHORT and buried among longer, more dramatic logs from symptom services

---

### 6. GETS STUCK IN LOOPS (Repeating Failed Actions)

**What happens:** After a remediation returns "issues persist," the model tries the SAME type of action again on a different service, or the same service again.

**The data:**
- 20-40% of failed agent trajectories involve repetitive loops (SWE-bench analyses)
- Claude Code sub-agent consumed 27 million tokens in an infinite loop over 4.6 hours
- LangGraph agent processed 2,847 iterations at $400+ cost for a $5 task
- In WebArena, agents entered repetitive click loops ~25% of the time

**Why it matters for us:** Our GPT-5.2 Task 4 output shows EXACTLY this:
```
Step 11: restart_service config-service     → "issues persist"
Step 13: restart_service internal-dns       → already healthy, wasteful
Step 14: restart_service service-mesh       → "issues persist"
Step 15: restart_service config-service     → SAME SERVICE AGAIN → "issues persist"
Step 22: restart_service order-service      → "issues persist"
Step 23: restart_service payment-api        → "issues persist"
Step 25: restart_service user-service       → "issues persist"
Step 27: restart_service api-gateway        → "issues persist"
```

17 of 29 steps were restart attempts that all failed. The model couldn't break out of the "try restarting things" loop.

**How to exploit this:**
- Our current penalty for this is too mild. Steps 11-27 scored 0.74 overall — should be much lower
- Add a "diminishing returns" penalty: each restart of a different service without new investigation in between gets progressively penalized
- Add a "loop detection" penalty: restarting the same service twice = heavy penalty

---

### 7. FAILS TO SYNTHESIZE INFORMATION FROM MULTIPLE STEPS

**What happens:** The model gathers clues from logs, metrics, and dependencies across multiple steps — but fails to connect them into a coherent diagnosis.

**The data:**
- 15-25% compositionality gap between sub-question accuracy and multi-hop accuracy (Press et al., 2023)
- ~40% of HotpotQA failures were attributed to incorrect synthesis rather than incorrect retrieval
- Accuracy drops logarithmically with the number of facts that need to be combined (significant drops after 5-7 facts)
- Multi-stage relay planning: 90.7% accuracy collapsed to 22.5% over 5 hops

**Why it matters for us:** Our current scenarios have simple causal chains:
```
DNS corrupted → services can't resolve → services fail
```

A harder scenario would require connecting 3-4 separate clues:
```
Clue 1 (logs): "connection timeout to 10.0.3.47:5432"
Clue 2 (metrics): memory usage climbing on service-X
Clue 3 (dependencies): service-X depends on shared-db
Clue 4 (deploy): shared-db had a config change 2 hours ago
```

The agent needs to synthesize: timeout → memory → shared dependency → config change = root cause is the shared-db config change causing memory leak in service-X causing timeouts.

**How to exploit this:**
- Create scenarios where the root cause requires connecting 3+ separate investigation results
- No single log entry or metric should reveal the root cause alone
- The "story" only makes sense when you combine logs from service A + metrics from service B + dependencies from service C

---

### 8. HALLUCINATES WHEN TOOL RETURNS NO RESULTS

**What happens:** When `check_metrics` returns "No metrics found" or `query_logs` returns empty results, the model either ignores this signal or invents data to fill the gap.

**The data:**
- When data is missing, models substitute plausible alternatives rather than returning null or asking for clarification (KAMI v0.1)
- DeepSeek V3.1 autonomously substituted a similar company name without instruction
- "Over-helpfulness under uncertainty" is one of four primary failure archetypes
- Models fail to refuse infeasible tasks: best model achieves only 53.9% refusal rate

**Why it matters for us:** In our GPT-5.2 run:
- Step 8 (Task 1): `check_metrics(payment-api, db_connection_pool_in_use)` → "No metrics found" — the model ignored this and proceeded correctly. But it didn't use the ABSENCE of data as information.
- Step 17 (Task 4): `check_metrics(internal-dns, dns_servfail_rate)` → "No metrics found" — again ignored.

**How to exploit this:**
- Create scenarios where the ABSENCE of metrics is the clue (e.g., "metrics collector is down → that's why we can't see metrics for the root cause service")
- Return misleading partial data instead of "no data found" — let the model try to reason with incomplete/incorrect information
- Add scenarios where the correct action is "escalate" because the available tools can't diagnose the problem

---

### 9. VULNERABLE TO RED HERRINGS (Context Pollution)

**What happens:** Distractor information in tool outputs causes incorrect reasoning. Models feel compelled to explain ALL data, including irrelevant noise.

**The data:**
- When irrelevant distractor tool was added: Qwen3-8B hallucination jumped from 5.4% to 56.8% — reasoning AMPLIFIED hallucination
- Both Granite 4 Small (36.7% failure) and DeepSeek V3.1 (33.3% failure) fixated on distractor tables
- Enhanced reasoning makes distractor vulnerability WORSE, not better
- Adding irrelevant context to math problems caused 5-20% accuracy drop even for GPT-4

**Why it matters for us:** Our current red herrings are too obviously benign:
- recommendation-engine: `healthy (latency=300ms, errors=1.0%)` — obviously fine
- search-api: `healthy (latency=600ms, errors=0.5%)` — obviously fine

A real red herring should be GENUINELY CONFUSING — a service that LOOKS broken, has alarming logs, was recently deployed, AND happens to be unrelated to the actual incident.

**How to exploit this:**
- Make red herring services show as `degraded` with 10-20% error rates (not "healthy")
- Give red herring services alarming ERROR log entries
- Give red herring services recent deployments
- Make the red herring incident plausible (e.g., "cache eviction rate increased" when the actual problem is DNS)
- The only way to distinguish: check dependencies and realize the red herring has NO dependency on the actual root cause

---

### 10. COLLAPSES UNDER CONFLICTING INFORMATION

**What happens:** When presented with evidence that contradicts the model's current hypothesis, it either ignores the contradiction or collapses to random-chance accuracy.

**The data:**
- Counterfactual reasoning with conflicting info: accuracy collapses to ~50% (random chance) (arXiv, 2025)
- Models default to parametric knowledge when presented conflicting premises ("context-ignoring")
- Sycophancy in 58% of interactions — when users suggest alternative causes, models flip to agree
- LLM accuracy drops up to -54% under prompt perturbations

**Why it matters for us:** Our current scenarios have CONSISTENT evidence — all logs, metrics, and dependencies point to the same root cause. There's no conflicting signal. A harder scenario would have:
- Log from service A says: "timeout connecting to service B" (suggests B is the problem)
- Log from service B says: "healthy, serving requests normally" (contradicts above)
- The actual issue: network partition between A and B (neither service is broken individually)

**How to exploit this:**
- Create scenarios with genuinely contradictory signals
- Service A reports B is down. Service B reports itself as healthy. The truth: intermittent network issue.
- Metrics show improvement (error rate dropping) but logs show new errors (a different failure mode replacing the first)
- Recent deploy IMPROVED things, but an older config change is causing the real problem

---

## Summary: The 10 Failure Modes Ranked by Exploitability

| # | Failure Mode | GPT-5.2 Vulnerability | How Hard to Exploit | Impact on Score |
|---|---|---|---|---|
| 1 | Acts before investigating | 97% FP rate in OpenSec | Easy — hide metrics from observation | -0.20 to -0.30 |
| 2 | Anchoring on first evidence | 93% shift in anchor direction | Easy — order services strategically | -0.10 to -0.15 |
| 3 | Correlation ≠ causation | Only 30% achieve causal reasoning | Medium — decouple deploy from cause | -0.15 to -0.20 |
| 4 | Can't say "I don't know" | 87% rate as max confidence | Medium — ambiguous early evidence | -0.10 to -0.15 |
| 5 | Lost in the middle | 20-30% drop for mid-positioned info | Easy — more services, bury the clue | -0.10 to -0.15 |
| 6 | Gets stuck in loops | 20-40% of failed trajectories | Already happening — tighten grading | -0.10 to -0.15 |
| 7 | Can't synthesize multi-hop | 90.7% → 22.5% over 5 hops | Hard — requires scenario redesign | -0.15 to -0.25 |
| 8 | Hallucinates on missing data | 53.9% fail to refuse infeasible | Medium — add "no data" scenarios | -0.05 to -0.10 |
| 9 | Vulnerable to red herrings | Up to 56.8% hallucination with distractors | Medium — make red herrings degraded | -0.10 to -0.20 |
| 10 | Collapses under conflicting info | Drops to ~50% (random) | Hard — requires contradictory scenarios | -0.15 to -0.25 |

---

## What This Means For Our Environment

If we design scenarios that systematically target these 10 failure modes, GPT-5.2 should score:

```
Current scores (too easy):
  Easy: 0.89, Medium: 0.19 (parser bug), Hard: 0.75, Expert: 0.74

Target scores (after exploiting failure modes):
  Easy: 0.45-0.55    (partial observability + anchoring traps)
  Medium: 0.25-0.35  (correlation traps + ambiguous evidence)
  Hard: 0.10-0.20    (multi-hop synthesis + conflicting signals)
  Expert: 0.03-0.10  (all 10 failure modes combined)
```

This creates a MUCH more compelling RL training environment — there's genuine room for improvement through training, and the environment tests real cognitive weaknesses, not just pattern matching.

---

## Key Sources

- OpenSec (arXiv 2601.21083) — 97% FP rate for GPT-5.2 on incident response
- OpenRCA (ICLR 2025) — Best LLM at 11.34% on real root cause analysis
- KAMI v0.1 (arXiv 2512.07497) — 4 failure archetypes in agentic scenarios
- MAST Taxonomy (arXiv 2503.13657) — 14 failure modes across 1,642 traces
- AbstentionBench (2025) — Reasoning degrades abstention by 24%
- The Reasoning Trap (arXiv 2510.22977) — Reasoning amplifies distractor hallucination
- Kim et al. (arXiv 2602.09937) — >66% hallucination rate in cloud RCA
- Liu et al. (2023) — Lost in the middle, 20-30% accuracy drop
- Press et al. (2023) — 15-25% compositionality gap
- Springer (2025) — 93.2% anchoring in LLM outputs

---

*Compiled from 5 parallel research agents. April 2026.*
