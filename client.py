"""OnCallEnv client for interacting with the environment server."""

from __future__ import annotations

from typing import Any

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from oncall_env.models import OnCallAction, OnCallObservation, OnCallState


class OnCallEnvClient(EnvClient[OnCallAction, OnCallObservation, OnCallState]):
    """Client for the OnCallEnv environment."""

    def _step_payload(self, action: OnCallAction) -> dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[OnCallObservation]:
        # Server sends {"observation": {...}, "reward": ..., "done": ...}
        obs_data = payload.get("observation", payload)
        reward = payload.get("reward")
        done = payload.get("done", False)
        # Avoid double-setting reward/done if already in obs_data
        obs_data_clean = {k: v for k, v in obs_data.items() if k not in ("reward", "done")}
        obs = OnCallObservation(**obs_data_clean, reward=reward, done=done)
        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
        )

    def _parse_state(self, payload: dict[str, Any]) -> OnCallState:
        return OnCallState(**payload)
