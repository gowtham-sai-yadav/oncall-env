# SUMMARY — Meta PyTorch OpenEnv Hackathon Grand Finale

> **For Team Diff Maker. Target: 1st prize ($7,500). Generated 2026-04-24 from official docs + web research.**

---

## 1. The Brute Facts

| | |
|---|---|
| **Event** | Meta PyTorch OpenEnv Hackathon × Scaler School of Technology |
| **Scale** | First-of-its-kind in India at this scale. 800+ Round 1 submissions |
| **Finale** | 25 April 2026 (8am) → 26 April 2026 (8pm). On-site, Scaler School of Technology, Electronic City, Bangalore |
| **Duration** | ~36 hours onsite + HF compute credits given onsite only |
| **Results** | NOT declared onsite. Hybrid evaluation: automated screening + 20–30 min dedicated domain-expert review per top team |
| **Submissions** | One per team. URL locked at deadline. Post-deadline commits ignored |
| **Prize pool** | $30,000 total across 15 teams. **1st $7,500** / 2nd $5,000 / 3rd $3,500 / 4–8 $2K / 9–15 $650 |
| **Bonus** | Top teams get interview opportunity with Meta + HF AI teams |

---

## 2. What You're Being Judged On (CRITICAL — DO NOT GET WRONG)

Judging is 100% driven by this four-axis rubric. Everything else is decoration.

| Weight | Criterion | What it means |
|---|---|---|
| **40%** | **Environment Innovation** | Novel, creative, genuinely challenging. Domain underexplored in RL/LLM training. Could a researcher publish a paper on this? |
| **30%** | **Storytelling & Presentation** | Clear problem + environment + agent behavior explanation. Demo engaging to non-technical audience |
| **20%** | **Showing Improvement in Rewards** | Reward curves, before/after behavior, comparison against a baseline. Observable evidence the agent actually learned |
| **10%** | **Reward & Training Pipeline** | Coherent reward logic. Pipeline produces meaningful improvement |

**Judge direct quote:** "A messy but ambitious environment with real training evidence beats a polished but boring one."

**Judges have explicitly said:** They've seen enough chess, snake, tic-tac-toe, grid-world, and generic incident-response. They want **fresh, underexplored domains**.

---

## 3. Non-Negotiable Minimum Submission Requirements

Missing any of these = serious disadvantage (effectively disqualified for 1st place):

1. Uses **OpenEnv latest release** (`openenv-core`)
2. Working **training script via Unsloth OR HuggingFace TRL**, ideally a Colab notebook judges can re-run
3. **Loss + reward plots from an actual training run**, committed as .png in the repo
4. One of: **mini-blog on HuggingFace** OR **<2 min YouTube video** OR **slide deck**. Linked from README
5. Env pushed to a **HuggingFace Space** (this is the submission URL)
6. **README** that motivates the problem, explains how env works, shows results, links to all external materials
7. `openenv.yaml` manifest valid
8. No reserved tool names (reset/step/state/close) for MCP tools
9. Follow Gym-style API (reset/step/state)
10. Proper `Environment` / `MCPEnvironment` base class inheritance

---

## 4. The 11 Judges — Who to Impress

### Biggest signal from cross-judge analysis: 5 of 11 judges have explicit India bias
These 5 give a "free multiplier" on an India-context / Indic-language / Indian-stack environment:

- **Aashay Sachdeva** (Sarvam founding team) — led Sarvam-M post-training with SFT+**RLVR**. Excites at: Indic sovereignty, verifiable rewards, reasoning mode
- **Adithya S Kolavi** (CognitiveLab / MSR Fellow / HF) — built NetraEmbed (22 Indian languages). Won Meta LLaMA Impact Grant. Heavy India bias. Multi-agent background (AAAI 2025)
- **Adarsh Shirawalmath** (Tensoic / HF) — Kannada Llama creator. Wrote EMNLP 2025 **AQI paper** on detecting game-able alignment. Will see through cheap reward design
- **Nilesh Pandey** (Meta Mumbai) — GenAI Partner Engineer. Production-shaped envs with Llama compatibility
- **Deepa Dhevannan** (GenAI Solutions Architect) — enterprise/healthcare. Clear ROI storytelling

### Systems-realism judges — punish toy gridworlds
- **Ayush Satyam** (Red Hat PyTorch/vLLM) — distributed systems rigor
- **Parshant Sharma** (Red Hat PyTorch Engineering) — observability, cloud-native ops, fault-tolerance
- **Arkadip Maitra** (Red Hat) — PyTorch Distributed, agents for humanitarian use

### Innovation / PyTorch-native judges
- **Sanyam Bhutani** (Meta) — TorchForge/Weaver author. Wants Llama post-trained end-to-end with verifiable rewards. Publicly called out judge fatigue on gym clones
- **Soumik Rakshit** (Zomato, ex-HF) — built HF `ml-intern` (self-improving ML engineer agent). Excites at agentic loops, torch.compile perf, memory-augmented agents
- **Yash Khare** (Meta) — low public footprint, multimodal/Llama background

### Cross-judge themes a winner must hit
1. **PyTorch/OpenEnv-native + RLVR training end-to-end** (8/11 judges)
2. **Production-systems realism, not toy gridworlds**
3. **Alignment/evaluation rigor — reward cannot be gameable**
4. **India-context angle = free multiplier** (5 judges)
5. **Self-improving / agentic loops** over static benchmarks

---

## 5. Competitive Crowding — Where OnCallEnv Stands

Scanned ~200 public "OpenEnv Hackathon" GitHub repos, all HF OpenEnv Spaces + community org.

### Current OnCallEnv position: 25-way dogfight
Direct competitors already public:
- **Harikishanth/Incident-Triage-Environment** (52⭐)
- **bvsbharat/OpenOfficeRL** (17⭐)
- **sid-rp/kube-sre-gym** (12⭐)
- **ankan00V/Incident-Commander** (11⭐)
- **Stack Doctor** (HF Spaces, SF hackathon submission)
- 20+ more clones tagged with "sre-incident-env", "firewatch", "pagersim", etc.

Innovation ceiling on incident-response is capped. Judges will tune out.

### Empty lanes (thin or single-repo):
- **Indian govt/services personal-assistant** — 1 repo only
- **Self-improvement / auto-curriculum** — thinnest theme overall
- **Reversibility / irreversible-action detection** — 1 weak repo
- **Scalable oversight / overseer-vs-responder** — 2–3 shallow repos
- **Multi-agent auction/market-clearing w/ asymmetric private info** — 1 repo
- **Shared-state personal tasks (household, family planning)** — zero

### HF signals what it's pushing (blog posts Jan–April 2026)
- **Ecom-RLVE** blog — multi-turn verifiable RL with 12-axis adaptive difficulty curriculum. **THIS IS THE TEMPLATE HF IS SIGNALING.**
- ben burtenshaw "Scaling OpenEnv" — 256 sessions/core via WebSockets
- TRL v1 (Mar 27 2026) — OpenEnv first-class with stateless-tools vs stateful-environments modes
- AI Trends 2026 — test-time reasoning + reflective agents + process reward models

### Verified research gold (arxiv IDs confirmed)
- **RLVE (2511.07317)** — 400-env gym with dynamic difficulty adaptation
- **Agentic Memory (2601.01885)** — memory ops as tool actions, step-wise GRPO. **Perfect hackathon fit.**
- Reasoning Gym (2505.24760) — NeurIPS 2025 Spotlight
- AgentPRM (2511.08325) — step-wise progress PRMs
- WebAgent-R1 (2505.16421) — Qwen-3B 6.1%→33.9% via multi-turn RL

---

## 6. What the Existing OnCallEnv Gets Right (and Where It Fails 40% Innovation)

### Reusable assets (do NOT rebuild from scratch):
- ~4000 LOC OpenEnv-compliant Python
- Docker + `openenv.yaml` + HF Space-ready structure
- Pydantic Action/Observation/State models
- FastAPI client-server split
- Multi-component grader architecture (6 components, weight-tuneable per scenario)
- Scenario loader + 48 scenarios across 4 difficulty tiers
- GRPO training script scaffolding (`train.py`, 553 LOC)
- Baseline eval outputs from GPT-4o, GPT-4o-mini, Sonnet, Opus, Gemini 2.5 Flash

### Fatal weaknesses for 1st prize:
- Domain (incident response) is **25-way crowded** — 40% Innovation capped
- No training evidence yet (no reward curves committed)
- Not deployed to HF Space yet
- Zero memory / self-improvement code
- No India-context angle
- No MCP integration (Meta's new direction)

---

## 7. Hackathon Stack — What Judges Expect

| Layer | Tool |
|---|---|
| Environment interface | **OpenEnv** (latest) |
| Training algorithm | **TRL GRPOTrainer** (or PPO) |
| Efficiency | **Unsloth** (4-bit QLoRA; rollout perf matters — rollouts dominate runtime) |
| Base model | Llama-3.2-1B/3B-Instruct OR Qwen2.5-0.5B/1.5B/3B-Instruct |
| Reward type | **RLVR** (verifiable, programmatic). LLM-as-judge only as one of several signals |
| Deployment | HuggingFace Spaces (Docker container, port 8000) |
| Monitoring | Weights & Biases run link in README; reward curves committed as .png |
| Save pattern | Proper merged-save path for QLoRA (do NOT upcast-then-merge) |

---

## 8. Common Pitfalls to Avoid (from official Meta guide + FAQs)

1. Task so hard the model never succeeds — zero reward → no learning
2. Single reward function — easy to hack. Use **multiple independent verifiers**
3. Not checking for reward hacking. Sample generations during training
4. Training before environment is stable
5. Relying on average reward; ignoring component metrics
6. Forgetting timeouts / sandbox limits
7. LLM-as-judge only — gets gamed
8. Over-shaping dense rewards — creates local optima
9. Static task difficulty — learning signal collapses at both extremes
10. Narrow environment set — narrow competence. Use procedural generation

---

## 9. Key Resources (official)

- OpenEnv core: https://github.com/meta-pytorch/OpenEnv
- OpenEnv docs: https://meta-pytorch.org/OpenEnv/
- HF OpenEnv hub: https://huggingface.co/openenv
- TRL OpenEnv integration: https://huggingface.co/docs/trl/openenv
- Course site: https://openenv-india-apr-2026.lovable.app
- Discord: discord.gg/Dedhy5pkWD
- Scaler dashboard: https://www.scaler.com/school-of-technology/meta-pytorch-hackathon

---

## 10. Strategic Conclusion

**Locked direction: Viveka — Reversibility Gym + Confidence Calibration (surgical combination).**

After comparing 8 candidate directions (OnCallEnv, BharatBench, CurriculumGym, Reversibility Gym, Adversarial Code, ExploreEnv, SpiralGuard, CalibrationGym + amalgams), the highest-EV path is Viveka with a confidence-calibration component bolted into the reward (NOT a 50/50 amalgam — surgical addition only).

**What Viveka is:** an OpenEnv RL environment that teaches agents to (1) predict reversibility BEFORE acting, (2) emit calibrated confidence scores via proper scoring rules, (3) ask the user when uncertain or facing irreversible actions. Scenarios drawn from Indian daily-life and enterprise contexts (UPI, IRCTC, filesystem, cloud console, messaging). 6-component reward, 60+ scenarios, single agent + GRPO + Unsloth + Qwen2.5-1.5B-Instruct.

**Why it wins:**
- Innovation 40%: reversibility-as-RL is genuinely fresh (1 weak GitHub competitor). Calibration via proper scoring rule is paper-worthy. Adarsh Shirawalmath's EMNLP 2025 AQI paper is the only adjacent published research.
- Storytelling 30%: "AI nearly sends ₹50,000 to wrong UPI — catches itself" lands in 5 seconds. Reliability diagram is a famous, beautiful plot. Two hero visuals.
- Reward 20%: binary ground truth on reversibility + Brier on confidence = mathematically clean, un-game-able curves. Single agent = clean training.
- Pipeline 10%: standard TRL GRPO + Unsloth, proven recipe.

**Theme coverage:** Theme 3.1 (World Modeling Professional) + Theme 5 (Wild Card "Impress Us") + partial Theme 3.2 (personalized via Indian scenarios) + partial Theme 2 (long-horizon multi-step).

**Judge coverage:** 11/11 hit. Direct hits on Adarsh (AQI), Aashay (RLVR), Red Hat trio (production safety), India-biased 5 (Indian scenarios). Multiple judges actively championing.

**Estimated 1st-prize probability: ~36%**, highest-EV path evaluated.

See **WINNING_PLAN.md** for the 44-hour execution timeline.
