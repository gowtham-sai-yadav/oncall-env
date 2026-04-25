# Judge Research — Who's Evaluating Us

Generated 2026-04-24 from web research. Used for strategy.

## Per-Judge Signals

### 1. Sanyam Bhutani — Meta Partner Engineer
- Core contributor to **TorchForge, OpenEnv, Llama-Cookbook, Synthetic-Data-Kit**
- Ran AMD Synthetic Data AI Agents Challenge 2025 + NeurIPS 2025 OpenEnv RL demo
- Public evidence: [PyTorch blog TorchForge+Weaver](https://pytorch.org/blog/supercharging-llms-scalable-rl-with-torchforge-and-weaver/)
- **Excites at:** Novel RL envs, end-to-end Llama training with verifiable rewards (RLVR), Weaver-style weak verifiers, clean pseudocode-like RL loops
- **Red flag:** Has publicly said judges have seen too much chess/snake/gridworld. Will discount "Gym classic clones"

### 2. Yash Khare — Meta Partner Engineer
- Low public footprint, Meta-Llama contributor (khare19yash)
- IIIT-H MS grad, multimodal/CV background
- **Excites at:** Multimodal + Llama plays, clean PyTorch-native code

### 3. Nilesh Pandey — Meta Partner Engineering (GenAI, Mumbai)
- Partner engineer on Llama ecosystem
- **Excites at:** Production-shaped envs, practical agentic use cases, Llama compatibility, **India context** (Mumbai-based)

### 4. Adithya S Kolavi — CognitiveLab founder / MSR Fellow / listed as HF
- Dec 2025 released [NetraEmbed](https://www.cognitivelab.in/blog/introducing-netraembed) — SoTA multilingual multimodal retrieval, 22 Indian languages, +152% over baselines
- Won **Meta LLaMA Impact Grant** (only India team)
- AAAI 2025 multi-agent workshop paper on visual content generation
- **Excites at:** Indic + multimodal + agentic systems, VLMs, retrieval. **Heavy India bias**

### 5. Adarsh Shirawalmath — Tensoic / HF
- Creator of **Kannada Llama**
- Co-author EMNLP 2025 "Alignment Quality Index (AQI)" — diagnoses hidden misalignment via latent geometry (DPO/GRPO/RLHF tested)
- **Excites at:** Alignment, evaluation, Indic LLMs, jailbreak/alignment-faking detection
- **Red flag:** Will instantly see through envs whose reward is game-able. His paper is literally about that

### 6. Arkadip Maitra — Red Hat Associate ML Engineer
- Focus: "Sign language generation and recognition. PyTorch Distributed. Agents for humanitarian applications"
- Prior: agents at Rezolve AI
- **Excites at:** PyTorch-distributed envs, agentic humanitarian/social-impact, accessibility

### 7. Aashay Sachdeva — Sarvam Founding Team (ML)
- Led post-training of **Sarvam-M (24B hybrid reasoning)** using SFT + **RLVR** + inference opts
- "We want to do cool things in ML for the sake of it"
- Former MPL real-time ML systems background
- **Excites at:** RLVR, verifiable rewards, reasoning/thinking-mode post-training, **Indic sovereignty**, production ML systems

### 8. Deepa Dhevannan — Gen AI Solutions Architect (Prana Life Sciences)
- Healthcare/life-sciences AI architect
- **Excites at:** Enterprise-grade, domain-grounded environments, clear ROI storytelling, solution architecture cleanliness

### 9. Soumik Rakshit — Zomato / AthenaAgent, ex-HF
- Lead dev of HF **ml-intern** — autonomous ML-engineer agent that reads papers, trains, iterates. Shipped +22 GPQA, +60% HealthBench
- PyTorch EU 2026 talk on torch.compile + Diffusers
- Currently researching post-training for diffusion + multimodal
- **Excites at:** Self-improving agents, autonomous research loops, post-training pipelines, torch.compile performance

### 10. Ayush Satyam — Red Hat Systems ML
- "Making PyTorch and vLLM better @RedHat"
- Blog posts on Raft, memory systems — distributed systems generalist
- **Excites at:** Systems-rigorous envs — real distributed/networking/DB mechanics, not toy abstractions

### 11. Parshant Sharma — Red Hat Associate ML (PyTorch Engineering)
- "Fault-tolerant systems, observability, scaling cloud-native platforms"
- 4 Scopus ML papers
- **Excites at:** Reliability, observability, cloud-native ops framed as RL. Directly aligned with SRE themes

---

## Cross-Judge Themes (5 dimensions a winning submission must hit)

1. **PyTorch/OpenEnv-native + RLVR post-training end-to-end** (8/11 judges)
   - Real Llama/Qwen trained with verifiable rewards, reward curves shown
   - Weaver-style weak verifiers beat LLM-judge-only rewards

2. **Production-systems realism, NOT toy gridworlds**
   - Red Hat trio (Ayush + Parshant + Arkadip) + Deepa punish gym clones
   - Sanyam has said this publicly
   - Systems fidelity = real APIs, real failure modes, real distributed state

3. **Alignment / evaluation rigor — reward cannot be gameable**
   - Adarsh's AQI paper will scrutinize
   - Need composable rubrics, adversarial probes, baseline-vs-trained plots

4. **India-context or sovereign-AI angle — free multiplier**
   - 5/11 judges with strong India bias: Adithya, Adarsh, Aashay, Nilesh, Deepa
   - Indian enterprise/stack (UPI, Aadhaar, DigiLocker, Jio network), Indic languages

5. **Self-improving / agentic loops over static benchmarks**
   - Soumik (ml-intern), Arkadip (agents), Sanyam (agent hackathons), Adithya (multi-agent AAAI)
   - Zeitgeist: agents that generate own curricula, not agents that pass fixed tests

---

## HF OpenEnv Hub — Existing Environments (crowding map, from direct fetch)

**Official meta-pytorch/OpenEnv envs (29)**: atari, browsergym, calendar, carla, chat, chess, coding, connect4, dipg_safety, dm_control, echo, finqa, finrl, git, grid_world, julia, kernrl, maze, openapp, openspiel, reasoning_gym, repl, snake, sumo_rl, tbench2, textarena, unity, websearch, wildfire

**HF OpenEnv org Spaces (13)**: Chat, BrowserGym, TB2, OpenSpiel, Coding, Atari, Wordle/Sudoku (TextArena), Echo, REPL, HarFeast, Vc Gemini, **Stack Doctor**, Football Play-Calling

**Direct competitors to OnCallEnv:**
- **Stack Doctor** (HF Spaces, Agentic RL Hackathon SF 2026) — incident response angle, single-agent
- **Kube SRE Gym** (HF Spaces, external) — Kubernetes SRE
- Commercial cousins: Rootly, Resolve AI, Cleric, incident.io Rover (not OpenEnv envs)

**Empty/underexplored categories:**
- No multi-agent incident response (war-room coordination)
- No Indic-language / Indian-stack scenarios
- No self-improving SRE with persistent runbook memory
- No adversarial red-team + blue-team SRE
- Personal tasks (Theme 3.2) — only `calendar` env exists
