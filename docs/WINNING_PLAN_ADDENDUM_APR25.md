# Addendum to WINNING_PLAN.md — April 25, 2026

Written 2026-04-25 ~00:55 IST. Today's research confirms yesterday's pivot call. Four sharpenings follow.

---

## Sharpening 1 — Reframe as "RLVE over DPI", not "BharatBench"

**Why:** HuggingFace just shipped TRL v1 (2026-03-27) with OpenEnv as first-class. They're amplifying the RLVE paper (arxiv 2511.07317) and published an "Ecom-RLVE" blog as the template. RLVE = Reinforcement Learning with Verifiable **Environments** — the distinguishing idea is procedurally-generated tasks with 12-axis adaptive difficulty, not a fixed scenario set.

**What this changes:**
- Working title: **"BharatStack-RLVE"** (or keep "BharatBench" if you prefer, but subtitle with "An RLVE for DPI").
- Paper-style framing: *"Adaptive Verifiable Environments for Multilingual Agents on India's Digital Public Infrastructure."*
- MVP: instead of 60 static scenarios, use a **procedural generator with 5-6 difficulty axes** (service count, language code-switch density, ambiguity level, step budget, red-herring presence, memory-requirement depth). Same total scenario count, but presented as procedurally adaptive.
- Storytelling hook: explicit nod to the RLVE paper and Ecom-RLVE blog in README. Judges from HF will notice.

**Cost:** negligible — procedural generator already in scope. Reframing is mostly README/pitch language.

---

## Sharpening 2 — Cut to 3 DPI-core services for MVP, not 5

**Why:** The current plan lists UPI + IRCTC + Swiggy + Ola + DigiLocker. Swiggy/Ola are private consumer apps; the story "India's Digital Public Infrastructure" lands harder with UPI + DigiLocker + IRCTC (all sovereign/public-sector). DPI is a globally recognized term (Bloomberg/Economist coverage). 5 services in 36 hours is scope debt.

**What this changes:**
- MVP services: **UPI + DigiLocker + IRCTC**.
- Stretch services: Swiggy/Zomato, Ola, ONDC seller.
- Moonshot: Bhashini (real Gov-of-India language API, public) for an authentic hook — even if we only stub it.

**Cost saving:** ~6–8 hours in mock-service engineering. Time redirected to scenario diversity + reward-audit + demo polish.

---

## Sharpening 3 — Make "adversarial reward audit" a headline section

**Why:** Two independent April 2026 signals:
1. Berkeley RDI published "8 agent benchmarks can be exploited to near-100% without solving the task" — reward hacking is the hottest topic in benchmark/env design right now.
2. Adarsh Shirawalmath's AQI paper (EMNLP 2025) is specifically about diagnosing hidden misalignment. He will scrutinize our reward design for gameability.

**What this changes:**
- New section in README titled **"Adversarial Reward Audit"**.
- Content: a page showing 4–5 gaming strategies we tried, why they fail, what the penalty is.
  - Example: "What if the agent just spams `check_balance` to look busy?" → efficiency penalty + no-op detection → reward 0.0.
  - Example: "What if the agent invents a fake train number?" → strict schema + known-entity check → reward -1.0.
  - Example: "What if the agent just parrots the user's message as final reply?" → intent fulfillment = 0.0 (state-diff based, not text-based).
- Takes 1 hour to write, huge signal to Adarsh + Sanyam + the Red Hat trio.

**Cost:** 1 hour. Disproportionately high payoff.

---

## Sharpening 4 — Include self-curriculum as explicit stretch goal (not moonshot)

**Why:** Today's arxiv signal strongly confirms the zeitgeist around **self-play curricula** — Agent0 (2511.16043), SPELL (2509.23863), Multi-Agent Evolve (2510.23595). Soumik Rakshit (ml-intern), Sanyam Bhutani (AMD Synthetic Data Challenge), and Adithya Kolavi (AAAI multi-agent workshop paper) all have direct skin in this game. A self-curriculum component adds *Theme 4 coverage* and an obvious paper extension.

**What this changes:**
- Promote self-curriculum from "moonshot" to **"stretch goal with a concrete scope"**:
  - Add a `MetaCurriculumAgent` that, given a model's recent success rate, proposes a new scenario at the frontier (slightly harder than current pass rate).
  - Generator takes 6 difficulty axes as input, outputs a new procedurally-generated scenario.
  - Saves as a JSON file on the Space; judges can watch the curriculum evolve.
- Goes in only if MVP (intent/grader/training/deploy) is green by 26 Apr 10:00 IST.
- Estimated 4 hours of work. Adds one more reward curve to the plot (learner-vs-fixed-curriculum baseline).

**Cost:** 4 hours. Only attempt after MVP ships.

---

## Net effect on 44-hour plan

| Phase | Before | After sharpening |
|---|---|---|
| Env + Mocks (Phase 1) | 5 services, 6h | **3 services, 4h** |
| Graders + Rubric (Phase 2) | Same 6h | Same 6h + 1h adversarial audit writeup |
| Train + Deploy (Phase 3) | Same 6h | Same |
| Train + Stretch (Phase 4) | Memory | Memory + self-curriculum, MVP-permitting |
| Eval + Polish (Phase 5) | Same 6h | Same |

**Time freed: ~3-4 hours on Day 1, which should be re-invested in scenario diversity and pitch rehearsal.**

---

## Updated success theorem

Submission wins 1st if all of these are true:

- [x] Domain is Indian DPI — empty lane, no competitor
- [x] Framing is RLVE, not fixed benchmark — matches HF template
- [x] Reward is deterministic-first, LLM-judge capped at 10% — Adarsh-proof
- [x] Adversarial reward audit section in README — Adarsh-signal
- [x] Multilingual (Hinglish + Tamil + Kannada) — Kolavi + Shirawalmath signal
- [x] Llama/Qwen post-trained end-to-end with reward curves committed — Sanyam template
- [x] HF Space deployed + Gradio UI + 2-min demo video — table stakes
- [x] Self-curriculum as stretch — Soumik signal, Theme 4 coverage
- [x] Story in non-technical English — 30% weight

---

## Decision checkpoint — what I need from Gowtham now

1. **Go / no-go on pivot** to BharatStack-RLVE (or BharatBench, name TBD).
2. **3 vs 5 services for MVP** — my recommendation: 3 (UPI + DigiLocker + IRCTC).
3. **Languages** — Hinglish required. Second language: Tamil or Kannada? (Kolavi's NetraEmbed covers 22 languages but he's Kannadiga; Shirawalmath built Kannada Llama. Tamil hits Sarvam indirectly.)
4. **Base model** — Qwen2.5-1.5B-Instruct (current plan) vs Qwen2.5-3B-Instruct (more capacity, slower training)?
5. **Repo name** — `bharat-bench`, `bharat-stack-rlve`, or other?

Reply with answers to these 5 and I start Phase 0 immediately (repo init + CLAUDE.md + mock-service skeleton).
