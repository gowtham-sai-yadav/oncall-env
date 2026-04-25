# Competitive Crowding Map — OpenEnv Hackathon

Generated 2026-04-24 from web research on GitHub + HF. Used for strategy.

## HF OpenEnv Hub (Official, Meta-maintained)

29 official envs under meta-pytorch/OpenEnv:
- **Games (crowded/judge-fatigue):** atari, chess, connect4, grid_world, maze, snake, openspiel, textarena (Wordle/Sudoku)
- **Code:** coding, repl, git, julia, tbench2, kernrl
- **Browser/web:** browsergym, openapp, websearch
- **Classical sim:** carla, unity, sumo_rl, dm_control
- **Domain:** reasoning_gym, finqa, finrl, calendar, chat, echo, dipg_safety, wildfire

**Community org (huggingface.co/openenv-community) — 38 spaces:** mostly 0 likes. Notable: kube-sre-gym (3), medagentbench, staffing-agency, game-2048, compute_market, EHRGym, chessecon.

**Anchor/top: KernelRL (Meta flagship — train LLMs to write faster CUDA/Triton). This is the "vibe" Meta wants.**

## Round-1 Landscape — GitHub Tagged "OpenEnv Hackathon" (~200 repos scanned)

### EXTREMELY CROWDED (10+ repos each)

- **SRE / Incident Response / On-call — 25+ repos**
  - Harikishanth/Incident-Triage-Environment (**52 stars**)
  - bvsbharat/OpenOfficeRL (17 stars)
  - sid-rp/kube-sre-gym (12 stars)
  - ankan00V/Incident-Commander (11 stars)
  - jbarnes850/opensec-env
  - devhemanthac-commits, mayur-dhavan, supriyo84374, mdnowroz13, dakshdoesdev, m-zest, agp-369, GazalThakur, PiyushSPatil, hitendras510, firewatch, arjunsvdc24, pagersim, 8+ more "sre-incident-env"
  - **Verdict for OnCallEnv: capped innovation ceiling. Judge will tune out on seeing yet another "on-call log diagnosis" demo.**

- **Email triage / assistant — 15+**
- **Code review / SQL / debugging — 15+** (coding_env and git_env are already official)
- **Customer support / ticket triage — 10+**
- **Traffic / SUMO / smart-city — 8+** (sumo_rl is official)
- **Data cleaning / pandas pipelines — 6+**
- **Finance / trading / underwriting — 8+** (finqa/finrl official)

### THIN — THEMATIC GAPS TO EXPLOIT

- **Multi-agent negotiation with budget constraints:** only abhinavgautam01/GPU_Budget_Negotiation_Arena + AyushbhaiPatel/openenv-boardroom
- **Scalable oversight / overseer-vs-responder:** anikasoni/oversight-arena, MrEinsteinE/sentinel-openenv, Arun-Sanjay/Red-Button. Genuinely thin
- **Adversarial self-improvement arenas:** mkstudioslearncbse-art/code-review-arena (BugInjector vs CodeReviewer), ADITYAGABA1322/sentinel-env. Very thin
- **Reversibility / irreversible-action detection:** sidd707/my-openenv. Single repo
- **Calibration / honesty:** Rushhaabhhh/HONEST-RL-Calibrator. Rare
- **Indian govt/services workflows:** vishalvardhan24816/Govt-Services-Navigator. **ONE repo. Massive gap.**
- **Motorsport strategy, materials/crystal design, antibiotic resistance:** single-repo each

## Theme-by-Theme Crowding

| Theme | Crowding | Whitespace |
|---|---|---|
| 1. Multi-Agent | ~15 shallow repos | auction/market-clearing w/ asymmetric private info; coalition-formation w/ partial comm |
| 2. Long-Horizon | crowded via SRE/email/support | multi-day plans persisting across resets; deadline-aware task switching |
| 3.1 World Modeling Pro | heavily crowded (SRE, SQL, finance, coding) | must out-polish Harikishanth 52⭐ to win |
| **3.2 World Modeling Personal** | **15 email envs, 1 calendar, nothing shared-state** | **shared household, taste elicitation, personal-finance life-events, Indic services** |
| **4. Self-Improvement** | **THINNEST theme** | auto-curriculum generator, adversarial self-play, meta-RL |
| 5. Wild Card | single-repo turf | judges explicitly want this |

## Direct Competitors to OnCallEnv
- Harikishanth/Incident-Triage-Environment (52⭐)
- bvsbharat/OpenOfficeRL (17⭐)
- sid-rp/kube-sre-gym (12⭐)
- ankan00V/Incident-Commander (11⭐)
- Stack Doctor (on HF Spaces, SF hackathon)
- **25+ clones total**

## HF Blog Signals (what HF is pushing April 2026)
- ben burtenshaw "Scaling OpenEnv" Jan 2026 — WebSockets, 256 sessions/core, infra scalability
- TRL v1 Mar 27 2026 — OpenEnv first-class (stateless tools vs stateful environments)
- **Ecom-RLVE blog post — 12-axis adaptive difficulty curriculum, multi-turn e-commerce, verifiable checks. This is the template HF signals.**
- AI Trends 2026 post — test-time reasoning + reflective agents + process reward models

## Meta Signal
- Sanyam Bhutani: "PyTorch-native large-scale RL first-class workflow"
- Meta's headline env: KernelRL. Verifiable, measurable improvement, researcher could publish the reward design.
- MCPEnvironment (MCP-native envs) is the new abstraction — MCP tool discovery, FastMCP 2.x/3.x, persistent sessions.

## Verified Research Threads (arxiv IDs confirmed via fetch)

- **RLVE: Adaptive Verifiable Environments** — arxiv 2511.07317 (Nov 2025). 400-env RLVE-Gym, dynamic difficulty, +3.37% avg. **Blueprint for paper-worthy reward design.**
- **Agentic Memory (AgeMem)** — arxiv 2601.01885 (Jan 2026). Memory ops (store/retrieve/update/summarize/discard) as tool actions, trained with step-wise GRPO. **Perfect hackathon fit.**
- **Reasoning Gym** — arxiv 2505.24760 (NeurIPS 2025 Spotlight)
- **AgentPRM** — arxiv 2511.08325 (step-wise promise/progress PRMs, 3B beats GPT-4o on ALFWorld)
- **WebAgent-R1** — arxiv 2505.16421 (end-to-end multi-turn RL, Qwen-3B 6.1→33.9%)

## Bottom Line

OnCallEnv places team in 25-way dogfight with existing 17–52⭐ competitors.
Innovation = 40% of score.
**Pivot to an empty lane that also hits India + verifiable + multi-turn. Best target: BharatOps — Indic personal assistant over real Indian service APIs (UPI/IRCTC/Swiggy/DigiLocker/Ola). Hits Themes 3.2 + 2 + 4. Hits 5/11 judges directly (Kolavi, Shirawalmath, Sachdeva, Pandey, Dhevannan).**
