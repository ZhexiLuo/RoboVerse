"""Base class for GPT-generated tasks."""

from __future__ import annotations

import torch

from metasim.example.example_pack.tasks.checkers import EmptyChecker
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.base import BaseTaskEnv
from metasim.utils.demo_util import get_traj
from metasim.utils.state import TensorState


class GptBaseTask(BaseTaskEnv):
    """GPT task base class, loads init_state from PKL."""

    scenario = None
    max_episode_steps = 250
    task_desc = None
    checker = EmptyChecker()
    traj_filepath = None

    def __init__(self, scenario: ScenarioCfg, device: str | torch.device | None = None) -> None:
        super().__init__(scenario, device)

    def _terminated(self, states: TensorState) -> torch.Tensor:
        return self.checker.check(self.handler, states)

    def reset(self, states=None, env_ids=None):
        states = super().reset(states, env_ids)
        self.checker.reset(self.handler, env_ids=env_ids)
        return states

    def _get_initial_states(self) -> list[dict] | None:
        """Load initial states from PKL trajectory file."""
        initial_states, _, _ = get_traj(self.traj_filepath, self.scenario.robots[0], self.handler)
        if len(initial_states) < self.num_envs:
            k = self.num_envs // len(initial_states)
            initial_states = initial_states * k + initial_states[: self.num_envs % len(initial_states)]
        self._initial_states = initial_states[: self.num_envs]
        return self._initial_states
