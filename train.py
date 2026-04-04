"""GRPO training script for OnCallEnv using PyTorch + TRL.

Trains a small language model to become a better on-call engineer by
playing episodes in the OnCallEnv environment and learning from rewards.

Usage:
    # Install dependencies first:
    pip install torch trl transformers accelerate datasets

    # Start the environment server in another terminal:
    python -m oncall_env.server.app

    # Run training:
    python train.py

    # Or with a specific model:
    python train.py --model Qwen/Qwen2.5-0.5B-Instruct --episodes 200
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from oncall_env.models import OnCallAction
from server.environment import OnCallEnvironment

# ── System prompt (same as inference.py) ─────────────────────────────────

SYSTEM_PROMPT = """You are an on-call engineer responding to a production incident.

ENVIRONMENT RULES:
- Alerts reference anonymous service labels (service-A, service-B, etc.).
- Services show "unknown" status until you investigate them.
- Use query_logs or check_metrics to discover a service's real identity.
- Deployments are hidden until you investigate the relevant service.

AVAILABLE ACTIONS:
- query_logs: Params: service (str), level (optional)
- check_metrics: Params: service (str), metric_name (str)
- view_dependencies: Params: service_name (str)
- acknowledge_alert: Params: alert_id (str)
- silence_alert: Params: alert_id (str)
- restart_service: Params: service_name (str)
- scale_service: Params: service_name (str), replicas (int)
- rollback_deploy: Params: service_name (str), target_version (str)
- update_config: Params: service_name (str), config_key (str), config_value (str)
- set_severity: Params: level (SEV1/SEV2/SEV3/SEV4)
- write_summary: Params: text (str)
- escalate: Params: team (str)
- resolve_incident: Params: resolution_note (str)

PROCESS:
1. TRIAGE: Acknowledge CRITICAL alerts. Set severity.
2. INVESTIGATE: query_logs and check_metrics for anonymous services.
3. DIAGNOSE: Identify root cause from evidence.
4. REMEDIATE: Fix the root cause (rollback, restart, or config change).
5. DOCUMENT: write_summary then resolve_incident (MANDATORY).

Respond with exactly one JSON object: {"action_type": "...", "params": {...}}"""


# ── Helpers ──────────────────────────────────────────────────────────────


def _alert_attr(a: Any, key: str, default: str = "") -> Any:
    """Access alert field as attribute or dict key."""
    return getattr(a, key, None) if not isinstance(a, dict) else a.get(key, default)


def format_observation(obs: Any) -> str:
    """Format an observation into a text prompt for the model."""
    parts = [f"Message: {obs.message}"]

    if obs.alerts:
        parts.append("\nActive Alerts:")
        for a in obs.alerts:
            ack = " [ACK]" if _alert_attr(a, "acknowledged") else ""
            sil = " [SILENCED]" if _alert_attr(a, "silenced") else ""
            sev = str(_alert_attr(a, "severity", "?")).upper()
            parts.append(
                f"  [{sev}] {_alert_attr(a, 'alert_id')}: "
                f"{_alert_attr(a, 'service')} - {_alert_attr(a, 'message')}{ack}{sil}"
            )

    if obs.services:
        parts.append("\nService Status:")
        for s in obs.services:
            name = _alert_attr(s, "name")
            status = _alert_attr(s, "status")
            latency = _alert_attr(s, "latency_ms", 0)
            errors = _alert_attr(s, "error_rate", 0)
            parts.append(f"  {name}: {status} (latency={latency}ms, errors={errors}%)")

    if obs.log_results:
        parts.append(f"\nLog Results ({len(obs.log_results)} entries):")
        for entry in obs.log_results[:10]:
            parts.append(
                f"  [{_alert_attr(entry, 'level')}] {_alert_attr(entry, 'timestamp')} "
                f"{_alert_attr(entry, 'service')}: {_alert_attr(entry, 'message')}"
            )

    if obs.metric_results:
        parts.append(f"\nMetric Results: {json.dumps(obs.metric_results, indent=2)}")

    if obs.dependency_graph:
        parts.append(f"\nDependency Graph: {json.dumps(obs.dependency_graph, indent=2)}")

    if obs.recent_deployments:
        parts.append("\nRecent Deployments:")
        for d in obs.recent_deployments:
            parts.append(
                f"  {_alert_attr(d, 'service')} v{_alert_attr(d, 'version')} "
                f"at {_alert_attr(d, 'timestamp')} by {_alert_attr(d, 'deployer')}"
            )

    return "\n".join(parts)


def parse_action(text: str) -> OnCallAction:
    """Extract the first JSON object from model output and parse as action."""
    start = text.find("{")
    if start == -1:
        return OnCallAction(action_type="resolve_incident", params={"resolution_note": text})

    depth, in_str, esc, end = 0, False, False, -1
    for i in range(start, len(text)):
        ch = text[i]
        if esc:
            esc = False
            continue
        if ch == "\\" and in_str:
            esc = True
            continue
        if ch == '"' and not esc:
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end == -1:
        return OnCallAction(action_type="resolve_incident", params={"resolution_note": text})

    try:
        data = json.loads(text[start:end])
        return OnCallAction(
            action_type=data.get("action_type", "resolve_incident"),
            params=data.get("params", {}),
        )
    except json.JSONDecodeError:
        return OnCallAction(action_type="resolve_incident", params={"resolution_note": text})


# ── Episode runner ───────────────────────────────────────────────────────


def run_episode(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    task_id: int,
    scenario_idx: int,
    device: torch.device,
    max_steps: int = 25,
) -> tuple[float, list[dict]]:
    """Run a single episode and return (reward, trajectory).

    The trajectory is a list of {prompt, completion, reward} dicts
    for each step -- used by GRPO to compute per-step advantages.
    """
    env = OnCallEnvironment()
    obs = env.reset(task_id=task_id, scenario_idx=scenario_idx)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_observation(obs)},
    ]

    trajectory = []
    step = 0

    while not obs.done and step < max_steps:
        step += 1

        # Format conversation for the model
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        completion = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        action = parse_action(completion)
        obs = env.step(action)

        trajectory.append({
            "prompt": prompt_text,
            "completion": completion,
            "action": action.action_type,
        })

        messages.append({"role": "assistant", "content": completion})
        messages.append({"role": "user", "content": format_observation(obs)})

    # Final reward from grader
    reward = obs.reward if obs.reward is not None else env._compute_final_reward()
    return reward, trajectory


# ── GRPO reward function ─────────────────────────────────────────────────


def build_reward_fn(device: torch.device):
    """Build the reward function that GRPO calls during training.

    For each prompt (initial observation), the model generates a completion
    (action plan). We run that completion through the environment and return
    the grader's reward.
    """

    def reward_fn(completions: list[str], prompts: list[str] | None = None, **kwargs) -> list[float]:
        """Score a batch of model completions by running them in the env.

        Each completion is treated as a single-step action. The reward
        is the intermediate step reward signal for that action.
        """
        rewards = []
        for i, completion in enumerate(completions):
            try:
                action = parse_action(completion)
                # Use metadata embedded in the prompt to determine task/scenario
                meta = kwargs.get("metadata", [{}] * len(completions))
                task_id = meta[i].get("task_id", 1) if i < len(meta) else 1
                scenario_idx = meta[i].get("scenario_idx", 0) if i < len(meta) else 0

                env = OnCallEnvironment()
                obs = env.reset(task_id=task_id, scenario_idx=scenario_idx)

                # Replay any prior actions stored in metadata
                prior_actions = meta[i].get("prior_actions", []) if i < len(meta) else []
                for pa in prior_actions:
                    obs = env.step(OnCallAction(
                        action_type=pa["action_type"],
                        params=pa.get("params", {}),
                    ))

                # Execute the new action
                obs = env.step(action)

                if obs.done and obs.reward is not None:
                    rewards.append(obs.reward)
                else:
                    # Use intermediate reward signals as shaped reward
                    signals = obs.metadata.get("reward_signals", {})
                    shaped = (
                        signals.get("oncall.triage_progress", 0.0) * 0.2
                        + signals.get("oncall.investigation_depth", 0.0) * 0.3
                        + signals.get("oncall.severity_set", 0.0) * 0.1
                        + signals.get("oncall.premature_action", 0.0) * 0.2
                    )
                    rewards.append(shaped)

            except Exception:
                rewards.append(0.0)

        return rewards

    return reward_fn


# ── Dataset generation ───────────────────────────────────────────────────


def generate_training_prompts(
    tokenizer: AutoTokenizer,
    num_prompts: int = 200,
) -> Dataset:
    """Generate training prompts by resetting the environment with different scenarios.

    Each prompt is an initial observation that the model must respond to
    with an appropriate incident response action.
    """
    prompts = []

    for i in range(num_prompts):
        task_id = random.choice([1, 2, 3, 4])
        # Each task has 12 scenarios
        scenario_idx = random.randint(0, 11)

        env = OnCallEnvironment()
        obs = env.reset(task_id=task_id, scenario_idx=scenario_idx)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_observation(obs)},
        ]

        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        prompts.append({
            "prompt": prompt_text,
            "task_id": task_id,
            "scenario_idx": scenario_idx,
        })

    return Dataset.from_list(prompts)


# ── Evaluation ───────────────────────────────────────────────────────────


def evaluate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    num_episodes: int = 4,
) -> dict[str, float]:
    """Evaluate the model on one scenario per task and return scores."""
    scores = {}
    for task_id in range(1, 5):
        reward, trajectory = run_episode(
            model, tokenizer,
            task_id=task_id,
            scenario_idx=0,
            device=device,
        )
        scores[f"task{task_id}"] = reward
        print(f"  task{task_id}: {reward:.4f} ({len(trajectory)} steps)")
    return scores


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Train an on-call agent with GRPO")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="HuggingFace model ID (default: Qwen/Qwen2.5-0.5B-Instruct)")
    parser.add_argument("--episodes", type=int, default=200,
                        help="Number of training prompts to generate")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--output-dir", default="./oncall-agent-grpo",
                        help="Directory to save trained model")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only run evaluation, skip training")
    args = parser.parse_args()

    # ── Device setup ─────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU (training will be slow)")

    # ── Load model ───────────────────────────────────────────────────
    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32 if device.type == "cpu" else torch.float16,
        trust_remote_code=True,
    ).to(device)

    # ── Evaluate before training ─────────────────────────────────────
    print("\n--- Pre-training evaluation ---")
    pre_scores = evaluate(model, tokenizer, device)

    if args.eval_only:
        return

    # ── Generate training dataset ────────────────────────────────────
    print(f"\nGenerating {args.episodes} training prompts...")
    dataset = generate_training_prompts(tokenizer, num_prompts=args.episodes)
    print(f"Dataset size: {len(dataset)} prompts across 4 tasks")

    # ── Configure GRPO trainer ───────────────────────────────────────
    print("\nConfiguring GRPO trainer...")
    training_config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=50,
        max_completion_length=200,
        num_generations=4,          # number of completions per prompt for GRPO
        temperature=0.7,
        log_with=None,              # disable wandb/tensorboard
        report_to="none",
    )

    reward_fn = build_reward_fn(device)

    trainer = GRPOTrainer(
        model=model,
        args=training_config,
        tokenizer=tokenizer,
        train_dataset=dataset,
        reward_funcs=reward_fn,
    )

    # ── Train ────────────────────────────────────────────────────────
    print("\n--- Starting GRPO training ---")
    print(f"  Model:      {args.model}")
    print(f"  Episodes:   {args.episodes}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR:         {args.lr}")
    print(f"  Device:     {device}")
    print()

    trainer.train()

    # ── Save trained model ───────────────────────────────────────────
    print(f"\nSaving trained model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # ── Evaluate after training ──────────────────────────────────────
    print("\n--- Post-training evaluation ---")
    post_scores = evaluate(model, tokenizer, device)

    # ── Report ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)
    print(f"{'Task':<10} {'Before':>10} {'After':>10} {'Delta':>10}")
    print("-" * 40)
    for task in ["task1", "task2", "task3", "task4"]:
        before = pre_scores.get(task, 0.0)
        after = post_scores.get(task, 0.0)
        delta = after - before
        sign = "+" if delta >= 0 else ""
        print(f"{task:<10} {before:>10.4f} {after:>10.4f} {sign}{delta:>9.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
