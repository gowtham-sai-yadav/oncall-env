"""GRPO training script for OnCallEnv using PyTorch + TRL.

Trains a small language model to become a better on-call engineer by
playing episodes in the OnCallEnv environment and learning from rewards.

Uses TRL's `environment_factory` pattern where each action becomes a
callable tool. The GRPOTrainer handles the multi-turn episode loop
automatically.

Usage:
    # Install training dependencies:
    pip install torch trl transformers accelerate datasets

    # Dry run (verify imports, no GPU needed):
    python train.py --dry-run

    # Evaluate a model without training:
    python train.py --eval-only

    # Full training (requires GPU):
    python train.py --model Qwen/Qwen2.5-0.5B-Instruct --episodes 200

    # With a larger model:
    python train.py --model Qwen/Qwen3-1.7B --episodes 500 --epochs 3
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from typing import Any

# ── Dry-run check (before heavy imports) ────────────────────────────────

def check_imports():
    """Verify all required packages are importable."""
    missing = []
    for pkg in ["torch", "transformers", "trl", "datasets"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    # Check oncall_env
    try:
        from oncall_env.server.environment import OnCallEnvironment
        from oncall_env.models import OnCallAction
        env = OnCallEnvironment()
        obs = env.reset(task_id=1, scenario_idx=0)
        obs = env.step(OnCallAction(action_type="resolve_incident", params={"resolution_note": "test"}))
        assert obs.done and 0.0 <= obs.reward <= 1.0
        print("oncall_env: OK (environment loads, step/reset/reward work)")
    except Exception as e:
        print(f"oncall_env: FAILED ({e})")
        return False
    return True


# ── OnCallToolEnv: TRL environment_factory class ───────────────────────

class OnCallToolEnv:
    """Tool-calling environment wrapper for TRL GRPOTrainer.

    Each public method (except reset) becomes a callable tool that the
    model can invoke during training episodes. TRL automatically discovers
    tools from method signatures and docstrings.

    The trainer creates one instance per generation, calls reset() to get
    the initial observation, then lets the model call tool methods until
    the episode ends.
    """

    def __init__(self):
        from oncall_env.server.environment import OnCallEnvironment
        self.env = OnCallEnvironment()
        self.reward = 0.0
        self._done = False
        # Pick a random task/scenario for this episode
        self._task_id = random.choice([1, 2, 3, 4])
        self._scenario_idx = random.randint(0, 11)

    def reset(self, **kwargs) -> str | None:
        """Reset the environment and return the initial observation.

        Returns:
            Initial incident description with alerts and service status,
            or None if no initial context is needed.
        """
        self.reward = 0.0
        self._done = False
        obs = self.env.reset(
            task_id=self._task_id,
            scenario_idx=self._scenario_idx,
        )
        return _format_obs(obs)

    def query_logs(self, service: str, level: str = "ERROR") -> str:
        """Query service logs to investigate issues.

        Args:
            service: Service name or alias (e.g., 'service-A' or 'payment-api')
            level: Log level filter — DEBUG, INFO, WARN, ERROR, or FATAL

        Returns:
            Log entries matching the query, revealing the service's real identity.
        """
        return self._step("query_logs", {"service": service, "level": level})

    def check_metrics(self, service: str, metric_name: str) -> str:
        """Check service metrics. Lists available metrics if name doesn't match.

        Args:
            service: Service name or alias
            metric_name: Metric to query (e.g., 'error_rate', 'latency_p99', 'db_connections')

        Returns:
            Time-series metric data, or list of available metrics if name not found.
        """
        return self._step("check_metrics", {"service": service, "metric_name": metric_name})

    def view_dependencies(self, service_name: str) -> str:
        """View the dependency graph for a service.

        Args:
            service_name: Service name or alias to check dependencies for

        Returns:
            List of upstream dependencies (databases, other services).
        """
        return self._step("view_dependencies", {"service_name": service_name})

    def acknowledge_alert(self, alert_id: str) -> str:
        """Acknowledge a critical alert to indicate it's being handled.

        Args:
            alert_id: The alert identifier (e.g., 'alert-001')

        Returns:
            Confirmation that the alert was acknowledged.
        """
        return self._step("acknowledge_alert", {"alert_id": alert_id})

    def silence_alert(self, alert_id: str) -> str:
        """Silence a non-actionable alert (INFO/noise).

        Args:
            alert_id: The alert identifier to silence

        Returns:
            Confirmation that the alert was silenced.
        """
        return self._step("silence_alert", {"alert_id": alert_id})

    def restart_service(self, service_name: str) -> str:
        """Restart a service. Verify recovery with check_metrics afterward.

        Args:
            service_name: Real service name (discovered through investigation)

        Returns:
            Confirmation that restart was initiated. Use check_metrics to verify.
        """
        return self._step("restart_service", {"service_name": service_name})

    def rollback_deploy(self, service_name: str, target_version: str) -> str:
        """Rollback a service to a previous version.

        Args:
            service_name: Real service name to rollback
            target_version: Version to rollback to (e.g., 'v2.4.0')

        Returns:
            Confirmation that rollback was initiated. Use check_metrics to verify.
        """
        return self._step("rollback_deploy", {"service_name": service_name, "target_version": target_version})

    def update_config(self, service_name: str, config_key: str, config_value: str) -> str:
        """Update a service configuration parameter.

        Args:
            service_name: Real service name
            config_key: Configuration key to update
            config_value: New value for the configuration key

        Returns:
            Confirmation that config was updated. Use check_metrics to verify.
        """
        return self._step("update_config", {"service_name": service_name, "config_key": config_key, "config_value": config_value})

    def set_severity(self, level: str) -> str:
        """Set the incident severity level.

        Args:
            level: Severity level — SEV1, SEV2, SEV3, or SEV4

        Returns:
            Confirmation of severity setting.
        """
        return self._step("set_severity", {"level": level})

    def write_summary(self, text: str) -> str:
        """Write an incident summary documenting root cause and remediation.

        Args:
            text: Detailed incident summary including root cause, evidence, and fix applied

        Returns:
            Confirmation that summary was recorded.
        """
        return self._step("write_summary", {"text": text})

    def escalate(self, team: str) -> str:
        """Escalate the incident to another team.

        Args:
            team: Team name to escalate to (e.g., 'database-team', 'infra-team')

        Returns:
            Confirmation of escalation.
        """
        return self._step("escalate", {"team": team})

    def resolve_incident(self, resolution_note: str) -> str:
        """Mark the incident as resolved. This MUST be the final action.

        Args:
            resolution_note: Brief note describing the resolution

        Returns:
            Confirmation and final reward score.
        """
        return self._step("resolve_incident", {"resolution_note": resolution_note})

    # ── Internal helper ─────────────────────────────────────────────

    def _step(self, action_type: str, params: dict) -> str:
        """Execute an action and return formatted observation."""
        from oncall_env.models import OnCallAction
        if self._done:
            return "Incident already resolved."
        obs = self.env.step(OnCallAction(action_type=action_type, params=params))
        if obs.done and obs.reward is not None:
            self.reward = obs.reward
            self._done = True
        else:
            # Use shaped reward from intermediate signals
            signals = obs.metadata.get("reward_signals", {})
            self.reward = (
                signals.get("oncall.triage_progress", 0.0) * 0.2
                + signals.get("oncall.investigation_depth", 0.0) * 0.3
                + signals.get("oncall.severity_set", 0.0) * 0.1
                + (1.0 + signals.get("oncall.premature_action", 0.0)) * 0.2
            )
        return _format_obs(obs)


# ── Helpers ─────────────────────────────────────────────────────────────

def _attr(obj: Any, key: str, default: str = "") -> Any:
    """Access field as attribute or dict key."""
    return getattr(obj, key, None) if not isinstance(obj, dict) else obj.get(key, default)


def _format_obs(obs: Any) -> str:
    """Format observation as text for the model."""
    parts = [obs.message]
    if obs.alerts:
        parts.append("\nAlerts:")
        for a in obs.alerts:
            sev = str(_attr(a, "severity", "?")).upper()
            parts.append(f"  [{sev}] {_attr(a, 'alert_id')}: {_attr(a, 'service')} - {_attr(a, 'message')}")
    if obs.services:
        parts.append("\nServices:")
        for s in obs.services:
            parts.append(f"  {_attr(s, 'name')}: {_attr(s, 'status')}")
    if obs.log_results:
        parts.append(f"\nLogs ({len(obs.log_results)} entries):")
        for e in obs.log_results[:8]:
            parts.append(f"  [{_attr(e, 'level')}] {_attr(e, 'service')}: {_attr(e, 'message')[:120]}")
    if obs.metric_results:
        parts.append(f"\nMetrics: {json.dumps(obs.metric_results, indent=2)}")
    if obs.dependency_graph:
        parts.append(f"\nDependencies: {json.dumps(obs.dependency_graph)}")
    if obs.recent_deployments:
        parts.append("\nDeploys:")
        for d in obs.recent_deployments:
            parts.append(f"  {_attr(d, 'service')} v{_attr(d, 'version')} at {_attr(d, 'timestamp')}")
    return "\n".join(parts)


# ── Reward function ─────────────────────────────────────────────────────

def oncall_reward(environments: list[OnCallToolEnv], **kwargs) -> list[float]:
    """Extract reward from each environment instance.

    Called by GRPOTrainer after each episode completes. Returns the
    grader's reward for completed episodes, or shaped intermediate
    reward for ongoing episodes.
    """
    return [env.reward for env in environments]


# ── Evaluation (standalone, not part of training loop) ──────────────────

def evaluate_model(model_name_or_path: str, num_tasks: int = 4) -> dict[str, float]:
    """Evaluate a model by running one episode per task.

    Uses the model's generate() directly (not through TRL).
    This works without GPU for small models on CPU/MPS.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from oncall_env.server.environment import OnCallEnvironment
    from oncall_env.models import OnCallAction

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float32 if device.type == "cpu" else torch.float16,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    system_prompt = "You are an on-call engineer. Respond with one JSON action: {\"action_type\": \"...\", \"params\": {...}}"

    scores = {}
    for task_id in range(1, num_tasks + 1):
        env = OnCallEnvironment()
        obs = env.reset(task_id=task_id, scenario_idx=0)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _format_obs(obs)},
        ]

        for step in range(25):
            if obs.done:
                break
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)

            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)

            completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

            # Parse first JSON object
            action = _parse_action(completion)
            obs = env.step(action)

            messages.append({"role": "assistant", "content": completion})
            messages.append({"role": "user", "content": _format_obs(obs)})

            # Truncate history to prevent context overflow
            if len(messages) > 20:
                messages = messages[:2] + messages[-16:]

        reward = obs.reward if obs.reward is not None else env._compute_final_reward()
        scores[f"task{task_id}"] = reward
        print(f"  task{task_id}: {reward:.4f} ({step + 1} steps)")

    return scores


def _parse_action(text: str):
    """Extract first JSON object from text and parse as OnCallAction."""
    from oncall_env.models import OnCallAction
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
        return OnCallAction(action_type=data.get("action_type", "resolve_incident"), params=data.get("params", {}))
    except json.JSONDecodeError:
        return OnCallAction(action_type="resolve_incident", params={"resolution_note": text})


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train an on-call agent with GRPO")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="HuggingFace model ID")
    parser.add_argument("--episodes", type=int, default=200, help="Number of training prompts")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--num-generations", type=int, default=4, help="GRPO group size")
    parser.add_argument("--output-dir", default="./oncall-agent-grpo", help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Verify imports only, no GPU needed")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate, skip training")
    args = parser.parse_args()

    # ── Dry run ─────────────────────────────────────────────────────
    if args.dry_run:
        print("Dry run: checking imports and environment...")
        ok = check_imports()
        if ok:
            import torch
            print(f"torch: {torch.__version__} (CUDA: {torch.cuda.is_available()}, MPS: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()})")
            import trl
            print(f"trl: {trl.__version__}")
            print("\nAll checks passed. Ready for training with --model and GPU.")
        else:
            print("\nSome checks failed. Fix issues above before training.")
            sys.exit(1)
        return

    # ── Eval only ───────────────────────────────────────────────────
    if args.eval_only:
        print(f"Evaluating {args.model}...")
        scores = evaluate_model(args.model)
        print("\n" + "=" * 40)
        for task, score in scores.items():
            print(f"  {task}: {score:.4f}")
        return

    # ── Full training ───────────────────────────────────────────────
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer
    from oncall_env.server.environment import OnCallEnvironment

    device_str = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device_str}")

    # Generate training prompts
    # NOTE: Prompts contain ONLY the system message. The scenario-specific
    # observation comes from OnCallToolEnv.reset() which TRL calls automatically.
    # This ensures the dataset prompt and environment scenario always match.
    print(f"\nGenerating {args.episodes} training prompts...")

    system_msg = (
        "You are an on-call engineer responding to a production incident. "
        "Use the available tools to investigate, diagnose, and resolve the incident. "
        "Start by acknowledging critical alerts, then investigate services to discover "
        "their real identities and diagnose the root cause."
    )
    prompts = [
        {
            "prompt": [{"role": "system", "content": system_msg}],
        }
        for _ in range(args.episodes)
    ]

    dataset = Dataset.from_list(prompts)
    print(f"Dataset: {len(dataset)} prompts (scenario assigned by environment at reset)")

    # Evaluate before training
    print("\n--- Pre-training evaluation ---")
    pre_scores = evaluate_model(args.model)

    # Configure and run GRPO
    print("\n--- Starting GRPO training ---")
    config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_generations=args.num_generations,
        max_completion_length=256,
        max_prompt_length=2048,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=50,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=args.model,
        args=config,
        train_dataset=dataset,
        reward_funcs=oncall_reward,
        environment_factory=OnCallToolEnv,
    )

    trainer.train()

    # Save
    print(f"\nSaving to {args.output_dir}")
    trainer.save_model(args.output_dir)

    # Evaluate after training
    print("\n--- Post-training evaluation ---")
    post_scores = evaluate_model(args.output_dir)

    # Report
    print("\n" + "=" * 50)
    print("TRAINING RESULTS")
    print("=" * 50)
    print(f"{'Task':<10} {'Before':>10} {'After':>10} {'Delta':>10}")
    print("-" * 40)
    for task in ["task1", "task2", "task3", "task4"]:
        before = pre_scores.get(task, 0.0)
        after = post_scores.get(task, 0.0)
        delta = after - before
        print(f"{task:<10} {before:>10.4f} {after:>10.4f} {'+' if delta >= 0 else ''}{delta:>9.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
