"""Load and validate scenario JSON files."""

from __future__ import annotations

import importlib.resources
import json
from pathlib import Path
from typing import Any


def _find_scenarios_dir() -> Path:
    """Find scenarios directory, whether running from source or installed package."""
    # First try relative to this file (source layout)
    source_dir = Path(__file__).resolve().parent.parent / "scenarios"
    if source_dir.exists() and any(source_dir.iterdir()):
        return source_dir
    # Fallback: use importlib.resources for installed package
    try:
        pkg = importlib.resources.files("oncall_env.scenarios")
        pkg_path = Path(str(pkg))
        if pkg_path.exists():
            return pkg_path
    except (ImportError, TypeError):
        pass
    return source_dir  # return original even if missing, let caller handle error


SCENARIOS_DIR = _find_scenarios_dir()


def list_scenarios(task_dir: str) -> list[Path]:
    """Return sorted list of scenario JSON files for a task directory."""
    d = SCENARIOS_DIR / task_dir
    if not d.exists():
        return []
    return sorted(d.glob("scenario_*.json"))


def load_scenario(task_dir: str, scenario_name: str = "scenario_001") -> dict[str, Any]:
    """Load a single scenario JSON."""
    path = SCENARIOS_DIR / task_dir / f"{scenario_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Scenario not found: {path}")
    with open(path) as f:
        return json.load(f)


def load_scenario_by_task(task_id: int, scenario_idx: int = 0) -> dict[str, Any]:
    """Load scenario by task number (1-4) and scenario index."""
    task_map = {1: "task1_easy", 2: "task2_medium", 3: "task3_hard", 4: "task4_expert"}
    task_dir = task_map.get(task_id)
    if task_dir is None:
        raise ValueError(f"Invalid task_id: {task_id}. Must be 1-4.")
    scenarios = list_scenarios(task_dir)
    if not scenarios:
        raise FileNotFoundError(f"No scenarios found for {task_dir}")
    if scenario_idx >= len(scenarios):
        raise IndexError(f"Scenario index {scenario_idx} out of range (have {len(scenarios)})")
    with open(scenarios[scenario_idx]) as f:
        return json.load(f)
