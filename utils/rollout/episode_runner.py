"""Episode execution shared by the evaluation runner and the debug loop."""

import logging
from dataclasses import dataclass, replace
from typing import Callable, Dict, Optional

from core.env import BaseEnvironment
from core.policy import BasePolicy
from core.steering import BaseSteering
from core.types import Observation

logger = logging.getLogger(__name__)


@dataclass
class EpisodeResult:
    """Result of a single episode rollout."""

    success: bool          # Whether the task oracle reported success
    episode_length: int    # Number of policy calls (each executes one action chunk)
    episode_reward: float  # Total reward
    info: Dict             # Final info dict from the environment


class EpisodeRunner:
    """Runs one policy rollout: policy call, environment step, optional callback."""

    def __init__(
        self,
        env: BaseEnvironment,
        policy: BasePolicy,
        steering: Optional[BaseSteering] = None,
        max_steps: int = 1000,
    ):
        self.env = env
        self.policy = policy
        self.steering = steering
        self.max_steps = max_steps

    def run_episode(
        self,
        initial_obs: Optional[Observation] = None,
        reset_env: bool = True,
        reset_policy: bool = True,
        step_callback: Optional[Callable] = None,
        policy_instruction: Optional[str] = None,
    ) -> EpisodeResult:
        """Run a single episode.

        Args:
            initial_obs: Observation from a reset the caller already did
                (required when `reset_env` is False).
            reset_env: Whether to reset the environment here.
            reset_policy: Whether to clear the policy's history buffers.
            step_callback: Called after each policy call as
                callback(timestep, obs, action, reward, done, info), where `obs`
                is the observation the policy acted on.
            policy_instruction: Optional policy-only text, applied on every step.
                Environment, oracle, and steering callbacks keep the original text.
        """
        if reset_env:
            obs = self.env.reset()
        else:
            obs = initial_obs
            if obs is None:
                raise ValueError("Must provide initial_obs if reset_env=False")

        if reset_policy:
            self.policy.reset()

        done = False
        timestep = 0
        episode_reward = 0.0
        final_info: Dict = {}

        while not done and timestep < self.max_steps:
            policy_obs = (
                replace(obs, instruction=policy_instruction)
                if policy_instruction is not None else obs
            )
            action = self.policy.forward(policy_obs, steering=self.steering)

            next_obs, reward, done, info = self.env.step(action)
            episode_reward += reward

            if step_callback is not None:
                step_callback(timestep, obs, action, reward, done, info)

            obs = next_obs
            timestep += 1
            final_info = info

        success = final_info.get("success", False)
        logger.debug(
            f"Episode complete: {'SUCCESS' if success else 'FAILED'} | "
            f"Steps: {timestep} | Reward: {episode_reward:.2f}"
        )
        return EpisodeResult(
            success=success,
            episode_length=timestep,
            episode_reward=episode_reward,
            info=final_info,
        )
