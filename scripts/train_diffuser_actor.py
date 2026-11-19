#!/usr/bin/env python3
"""Training script for 3D Diffuser Actor on CALVIN dataset.

Runs single-process, or distributed when launched by torchrun (RANK/WORLD_SIZE)
or under SLURM (SLURM_PROCID/SLURM_NTASKS); see DiffuserActorTrainingWorkspace.

Usage:
    # Single GPU (CALVIN_3DA_DATASET_PATH: the packaged dataset root)
    python scripts/train_diffuser_actor.py training=diffuser_actor_calvin

    # Multi-GPU on one node
    torchrun --nproc_per_node=4 scripts/train_diffuser_actor.py training=diffuser_actor_overloaded

    # Override specific parameters
    python scripts/train_diffuser_actor.py training=diffuser_actor_calvin training.batch_size=8 training.train_iters=100000
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import hydra
from omegaconf import DictConfig, OmegaConf
import logging

from training.policies.diffuser_actor.trainer import DiffuserActorTrainingWorkspace

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Main training entry point."""

    # Ensure training config is provided
    if cfg.training is None:
        raise ValueError(
            "Training configuration not specified. "
            "Use: python scripts/train_diffuser_actor.py training=diffuser_actor_calvin"
        )

    # Log configuration (only on main process)
    rank = int(os.environ.get('RANK', 0))
    if rank == 0:
        logger.info("=" * 80)
        logger.info("3D Diffuser Actor Training Configuration:")
        logger.info("=" * 80)
        logger.info(OmegaConf.to_yaml(cfg.training))
        logger.info("=" * 80)

    # Initialize and run training workspace
    workspace = DiffuserActorTrainingWorkspace(cfg.training)
    workspace.train()


if __name__ == "__main__":
    main()
