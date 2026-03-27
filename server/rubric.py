"""OnCallRubric -- trajectory-based rubric for on-call incident response episodes."""

from __future__ import annotations

from typing import Any, List, Tuple

from openenv.core.rubrics.trajectory import TrajectoryRubric

from oncall_env.server.graders import grade_episode


class OnCallRubric(TrajectoryRubric):
    """Trajectory rubric for on-call incident response.

    Accumulates (action, observation) pairs during an episode, then scores
    the complete trajectory using the existing grade_episode() function when
    observation.done=True.

    The rubric holds a reference to the environment instance (set via set_env())
    so it can access scenario data and episode state at scoring time.
    """

    def __init__(self):
        super().__init__(intermediate_reward=0.0)
        self._env_ref = None

    def set_env(self, env: Any) -> None:
        """Store a reference to the environment for scoring access."""
        self._env_ref = env

    def score_trajectory(self, trajectory: List[Tuple[Any, Any]]) -> float:
        """Compute final score using the existing grade_episode grader.

        Called by TrajectoryRubric.forward() when observation.done=True.
        Delegates to grade_episode() which computes the 6-component weighted reward.
        """
        if self._env_ref is None:
            return 0.0
        env = self._env_ref
        return grade_episode(
            scenario=env._scenario,
            actions_taken=env._actions_taken,
            alerts_state=env._alerts,
            services_state=env._services,
            severity_set=env._severity,
            summary=env._summary,
            escalated_to=env._escalated_to,
            resolved=env._resolved,
        )

    def compute_step_rewards(self) -> List[float]:
        """Distribute final score uniformly across all steps.

        For GRPO training, each step gets equal credit: R_final / T.
        This is a simple credit assignment strategy; can be upgraded to
        exponential discounting or investigation-weighted assignment later.
        """
        if not self._trajectory:
            return []
        final_score = self.score_trajectory(self._trajectory)
        n = len(self._trajectory)
        return [final_score / n] * n

    def reset(self) -> None:
        """Clear trajectory for new episode."""
        super().reset()
