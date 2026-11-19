"""Component factories, and a single-task rollout loop for debugging.

The factories are the dispatch points for environments, policies and steering
methods; `scripts/run_evaluation.py` builds every condition through them.

    python scripts/run_experiment.py env.task=open_drawer policy=diffuser_actor_overloaded
    python scripts/run_experiment.py env.task=open_drawer policy=diffuser_actor_overloaded steering=voxposer
    python scripts/run_experiment.py env.task=open_drawer policy=pi05_guided   # needs the pi0.5 server

Benchmark numbers come from `scripts/run_overloaded_evaluation.py`, which fixes
the starting scenes and seeds; this loop resets to the task's preset scene.
"""

import logging
import random
import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.env import BaseEnvironment
from core.policy import BasePolicy
from core.steering import BaseSteering

logger = logging.getLogger(__name__)


def instantiate_env(cfg: DictConfig) -> BaseEnvironment:
    """Factory function to instantiate environment based on config."""
    env_name = cfg.env.name
    if env_name == "calvin":
        from envs.calvin import CalvinEnvironment
        return CalvinEnvironment(OmegaConf.to_container(cfg.env, resolve=True))
    raise ValueError(f"Unknown environment: {env_name}")


def instantiate_policy(cfg: DictConfig) -> BasePolicy:
    """Factory function to instantiate policy based on config."""
    policy_name = cfg.policy.name
    if policy_name == "diffuser_actor":
        from policies.diffuser_actor import build_diffuser_actor_policy
        policy = build_diffuser_actor_policy(OmegaConf.to_container(cfg.policy, resolve=True))
    elif policy_name == "pi05_guided":
        from policies.pi05_guided import GuidedPi05Policy
        policy = GuidedPi05Policy(OmegaConf.to_container(cfg.policy, resolve=True))
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    # Load checkpoint if specified
    if hasattr(cfg.policy, "ckpt_path") and cfg.policy.ckpt_path:
        policy.load_checkpoint(cfg.policy.ckpt_path)

    return policy


def instantiate_steering(cfg: DictConfig) -> BaseSteering | None:
    """Factory function to instantiate steering module based on config."""
    steering_name = cfg.steering.name
    if steering_name == "none" or steering_name is None:
        return None
    if steering_name == "voxposer":
        from steering.voxposer_steering import VoxPoserSteering
        return VoxPoserSteering(OmegaConf.to_container(cfg.steering, resolve=True))
    raise ValueError(f"Unknown steering: {steering_name}")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Roll out `cfg.num_episodes` episodes of `cfg.env.task` and log the success rate."""
    import torch

    from scripts.run_evaluation import make_step_callback, setup_voxposer_episode, wire_steering
    from utils.rollout import EpisodeRunner

    env = instantiate_env(cfg)
    env._provide_pcd_images = True
    policy = instantiate_policy(cfg)
    steering = instantiate_steering(cfg)
    if steering is not None:
        wire_steering(steering, policy, cfg)
    logger.info(f"Task: {env._task_name} -> '{env.task_description}'")

    runner = EpisodeRunner(env=env, policy=policy, steering=steering, max_steps=cfg.env.max_steps * 2)
    successes = 0
    for episode in range(cfg.num_episodes):
        seed = cfg.seed + episode
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        obs = env.reset()
        if steering is not None:
            setup_voxposer_episode(steering, env, cfg)
            # Plan generation must not alter the policy's noise seed.
            torch.manual_seed(seed)
        result = runner.run_episode(
            initial_obs=obs, reset_env=False, reset_policy=True,
            step_callback=make_step_callback(steering, env),
        )
        successes += int(result.success)
        logger.info(f"Episode {episode + 1}/{cfg.num_episodes}: "
                    f"{'success' if result.success else 'failure'} after {result.episode_length} policy calls")
    env.close()
    logger.info(f"Success rate: {successes}/{cfg.num_episodes}")


if __name__ == "__main__":
    main()
