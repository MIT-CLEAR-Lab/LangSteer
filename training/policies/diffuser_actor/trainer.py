"""DiffuserActor Training Workspace with SLURM/DDP Support.

Iteration-based training for 3D Diffuser Actor on CALVIN.
Adapted from 3d_diffuser_actor/engine.py + main_trajectory_calvin.py.
"""

import math
import os
import logging
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from omegaconf import DictConfig, OmegaConf
from tqdm import trange

from policies.diffuser_actor_components import DiffuserActor
from training.policies.diffuser_actor.dataset import CalvinDataset, traj_collate_fn
from training.policies.diffuser_actor.instruction_pools import (
    OverloadedInstructions,
    load_instruction_pool,
)
from training.common.checkpoint_util import TopKCheckpointManager
from training.common.ema_model import EMAModel

logger = logging.getLogger(__name__)

# Validation metric that selects best.pth and ranks the retained periodic
# checkpoints: the fraction of predicted waypoints within 1 cm. Higher is better.
SELECTION_METRIC = "val/traj_pos_acc_001"


def check_dataset_paths(dataset_cfg) -> None:
    """Fail with a setup error unless the dataset paths resolve to existing directories.

    The training configs read the dataset location from environment variables
    (`CALVIN_3DA_DATASET_PATH`, optionally `CALVIN_3DA_INSTRUCTIONS_PATH`). Without this
    check an unset variable surfaces as an OmegaConf interpolation error, and a wrong
    path as an empty dataset that fails later with a division by zero.
    """
    from omegaconf.errors import InterpolationResolutionError

    for key in ("train_path", "val_path", "instructions_path"):
        try:
            value = dataset_cfg[key]
        except InterpolationResolutionError as e:
            raise RuntimeError(
                f"dataset.{key} could not be resolved ({str(e).splitlines()[0]}). Set "
                "CALVIN_3DA_DATASET_PATH to the packaged CALVIN dataset root (the "
                "directory holding training/ and validation/), and optionally "
                "CALVIN_3DA_INSTRUCTIONS_PATH to the directory holding the instruction "
                f"pickles, or override dataset.{key} on the command line."
            ) from None
        if not Path(value).expanduser().is_dir():
            raise FileNotFoundError(
                f"dataset.{key} is {value!r}, which is not a directory. Check "
                "CALVIN_3DA_DATASET_PATH / CALVIN_3DA_INSTRUCTIONS_PATH."
            )


class DiffuserActorTrainingWorkspace:
    """
    Iteration-based training workspace for DiffuserActor on CALVIN.

    Key features:
    - Iteration-based (200K steps)
    - Loss computed inside model.forward() (returns scalar)
    - Separate weight decay groups (no decay for bias/LayerNorm)
    - DDP with find_unused_parameters=True (frozen backbone)
    - Validation runs inference and computes trajectory metrics
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        check_dataset_paths(cfg.dataset)

        # Setup distributed training
        self._setup_distributed()
        self._set_seed(cfg.seed)

        # Device
        if self.is_distributed:
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device(cfg.device)

        # Build model
        self.model = self._build_model(cfg)
        self.model.to(self.device)

        if self.is_distributed:
            find_unused = cfg.get("ddp_find_unused_parameters", True)
            if not find_unused:
                self._freeze_structurally_unused_params()
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                broadcast_buffers=False,
                find_unused_parameters=find_unused,
            )

        # EMA model
        self.ema_model = None
        if cfg.get("use_ema", False):
            self.ema_model = EMAModel(
                model=self.model.module if self.is_distributed else self.model,
                power=cfg.get("ema_power", 0.75),
                update_after_step=cfg.get("ema_update_after_step", 0),
                inv_gamma=cfg.get("ema_inv_gamma", 1.0),
                max_value=cfg.get("ema_max_value", 0.9999),
            )

        # Optimizer with separate weight decay groups
        self.optimizer = self._build_optimizer(
            self.model.module if self.is_distributed else self.model, cfg
        )

        # Cosine LR schedule with linear warmup
        self.scheduler = self._build_scheduler(self.optimizer, cfg)

        # Training state
        self.global_step = 0
        self.best_metric = None  # best SELECTION_METRIC so far (higher is better)
        self._retained_state = None  # TopKCheckpointManager state restored on resume

        # Logging
        self.use_wandb = cfg.get("use_wandb", True) and self.is_main_process
        if self.use_wandb:
            import wandb
            wandb.init(
                project=cfg.get("wandb_project", "langsteer_diffuser_actor"),
                name=cfg.get("experiment_name", "diffuser_actor_calvin"),
                config=OmegaConf.to_container(cfg, resolve=True),
                resume="allow",
            )

        if self.is_main_process:
            logger.info("DiffuserActorTrainingWorkspace initialized")

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _setup_distributed(self):
        """Setup distributed training (SLURM-aware)."""
        if "SLURM_PROCID" in os.environ:
            self.rank = int(os.environ["SLURM_PROCID"])
            self.local_rank = int(os.environ["SLURM_LOCALID"])
            self.world_size = int(os.environ["SLURM_NTASKS"])
            if "MASTER_ADDR" not in os.environ:
                os.environ["MASTER_ADDR"] = os.environ.get(
                    "SLURM_LAUNCH_NODE_IPADDR", "localhost"
                )
            if "MASTER_PORT" not in os.environ:
                os.environ["MASTER_PORT"] = "29500"
            self.is_distributed = True
            dist.init_process_group(
                backend="nccl", init_method="env://",
                world_size=self.world_size, rank=self.rank,
            )
        elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            self.rank = int(os.environ["RANK"])
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.is_distributed = True
            dist.init_process_group(backend="nccl", init_method="env://")
        else:
            self.rank = 0
            self.local_rank = 0
            self.world_size = 1
            self.is_distributed = False

        self.is_main_process = self.rank == 0

        if self.is_distributed:
            torch.cuda.set_device(self.local_rank)
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

    # Trainable parameters that never receive a gradient in this architecture, verified
    # stable across batches (identical set over 5 different samples) and therefore
    # structural rather than data-dependent:
    #   - feature_pyramid levels torchvision builds but the model never consumes
    #   - goal_gripper_embed: no goal conditioning in this setup
    #   - traj_lang_attention ffn_12/norm_122: the reverse-direction FFN of a
    #     bidirectional attention block. Upstream 3DDA behaviour, present in the
    #     reference implementation too -- the forward language path IS live (8 of 14
    #     language tensors receive gradients).
    # Freezing them is mathematically a no-op: a parameter with no gradient never
    # changes. It exists only so DDP can drop find_unused_parameters, which costs an
    # autograd-graph traversal on every backward pass.
    STRUCTURALLY_UNUSED_PARAMS = (
        "encoder.feature_pyramid.inner_blocks.0.",
        "encoder.feature_pyramid.inner_blocks.1.",
        "encoder.feature_pyramid.layer_blocks.0.",
        "encoder.feature_pyramid.layer_blocks.1.",
        "encoder.feature_pyramid.layer_blocks.3.",
        "encoder.feature_pyramid.layer_blocks.4.",
        "encoder.goal_gripper_embed",
        "prediction_head.traj_lang_attention.0.layers.0.ffn_12.",
        "prediction_head.traj_lang_attention.0.layers.0.norm_122.",
    )

    def _freeze_structurally_unused_params(self) -> None:
        """Freeze the never-gradient parameters so DDP can skip unused-param search.

        If the architecture changes and this list drifts, DDP raises on the first
        backward with a clear message naming the offending parameters -- a loud
        failure, not silent divergence. Re-derive by running one backward and
        collecting `p.grad is None` over trainable parameters.
        """
        frozen = []
        for name, p in self.model.named_parameters():
            if p.requires_grad and name.startswith(self.STRUCTURALLY_UNUSED_PARAMS):
                p.requires_grad = False
                frozen.append(name)
        if self.is_main_process:
            logger.info(
                f"Froze {len(frozen)} structurally-unused params for "
                f"find_unused_parameters=False"
            )

    def _set_seed(self, seed: int):
        seed = seed + self.rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _build_model(self, cfg: DictConfig) -> DiffuserActor:
        """Build DiffuserActor model from config."""
        policy_cfg = cfg.policy
        gripper_loc_bounds = policy_cfg.get(
            "gripper_loc_bounds", [[-2, -2, -2], [2, 2, 2]]
        )
        if isinstance(gripper_loc_bounds, str):
            from training.policies.diffuser_actor.preprocessing.calvin_utils import (
                get_gripper_loc_bounds,
            )
            gripper_loc_bounds = get_gripper_loc_bounds(gripper_loc_bounds)

        model = DiffuserActor(
            backbone=policy_cfg.get("backbone", "clip"),
            image_size=tuple(policy_cfg.get("image_size", [256, 256])),
            embedding_dim=policy_cfg.get("embedding_dim", 120),
            num_vis_ins_attn_layers=policy_cfg.get("num_vis_ins_attn_layers", 2),
            fps_subsampling_factor=policy_cfg.get("fps_subsampling_factor", 5),
            gripper_loc_bounds=gripper_loc_bounds,
            rotation_parametrization=policy_cfg.get("rotation_parametrization", "6D"),
            quaternion_format=policy_cfg.get("quaternion_format", "xyzw"),
            diffusion_timesteps=policy_cfg.get("diffusion_timesteps", 100),
            nhist=policy_cfg.get("nhist", 3),
            relative=policy_cfg.get("relative", True),
            lang_enhanced=policy_cfg.get("lang_enhanced", True),
        )

        if self.is_main_process:
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Model parameters: {n_params:,}")

        return model

    def _build_optimizer(self, model: nn.Module, cfg: DictConfig):
        """Build AdamW optimizer with separate weight decay groups."""
        optimizer_grouped_parameters = [
            {"params": [], "weight_decay": 0.0, "lr": cfg.lr},
            {"params": [], "weight_decay": cfg.get("wd", 5e-3), "lr": cfg.lr},
        ]
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in no_decay):
                optimizer_grouped_parameters[0]["params"].append(param)
            else:
                optimizer_grouped_parameters[1]["params"].append(param)
        return optim.AdamW(optimizer_grouped_parameters)

    def _build_scheduler(self, optimizer: optim.Optimizer, cfg: DictConfig):
        """Cosine LR schedule with linear warmup.

        Warms up linearly from 0 → lr over lr_warmup_steps steps, then decays
        via cosine to lr * 0.01 by the end of training.
        """
        warmup_steps = cfg.get("lr_warmup_steps", 2000)
        # scheduler.step() fires once per optimizer step, i.e. once every
        # accumulate_grad_batches micro-steps -- while train_iters counts micro-steps.
        # Dividing keeps the cosine spanning the whole run; without it the schedule
        # only advances 1/N of the way and the LR never decays to lr*min_lr_ratio.
        total_steps = cfg.train_iters // max(1, cfg.get("accumulate_grad_batches", 1))
        min_lr_ratio = 0.01  # final lr = lr * min_lr_ratio

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine_val = 0.5 * (1.0 + math.cos(math.pi * progress))
            return cosine_val * (1.0 - min_lr_ratio) + min_lr_ratio

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _get_datasets(self, cfg: DictConfig):
        """Initialize train and validation datasets."""
        dataset_cfg = cfg.dataset

        # Precomputed CLIP instruction embeddings (.pkl).
        overloaded = dataset_cfg.get("use_overloaded_instructions", False)
        # Fail fast rather than silently training an "overloaded" run whose
        # instructions never vary because the legacy artifact was loaded.
        ambiguous_prob = dataset_cfg.get("ambiguous_prob", 0.0)
        if ambiguous_prob > 0 and not overloaded:
            raise ValueError(
                f"ambiguous_prob={ambiguous_prob} requires "
                "use_overloaded_instructions=true; the legacy flat artifact has "
                "no ambiguous tier to draw from."
            )
        train_instructions = self._load_instructions(
            dataset_cfg.instructions_path, "training", overloaded
        )
        val_instructions = self._load_instructions(
            dataset_cfg.instructions_path, "validation", overloaded
        )
        if overloaded and not isinstance(train_instructions, OverloadedInstructions):
            raise ValueError(
                f"{dataset_cfg.instructions_path}/training_overloaded.pkl is not "
                "an overloaded artifact (no 'strings' key). Regenerate it with "
                "preprocess_overloaded_instructions.py."
            )

        # Load from config if specified, otherwise default to all 4 CALVIN scenes
        taskvar_cfg = dataset_cfg.get("taskvar", None)
        if taskvar_cfg is not None:
            taskvar = [tuple(tv) for tv in taskvar_cfg]
        else:
            taskvar = [("A", 0), ("B", 0), ("C", 0), ("D", 0)]

        image_rescale = tuple(
            float(x) for x in str(dataset_cfg.get("image_rescale", "0.75,1.25")).split(",")
        )

        train_dataset = CalvinDataset(
            root=dataset_cfg.train_path,
            instructions=train_instructions,
            taskvar=taskvar,
            max_episode_length=dataset_cfg.get("max_episode_length", 5),
            cache_size=dataset_cfg.get("cache_size", 100),
            max_episodes_per_task=dataset_cfg.get("max_episodes_per_task", -1),
            num_iters=cfg.train_iters,
            cameras=("front", "wrist"),
            training=True,
            image_rescale=image_rescale,
            return_low_lvl_trajectory=True,
            dense_interpolation=dataset_cfg.get("dense_interpolation", True),
            interpolation_length=dataset_cfg.get("interpolation_length", 100),
            relative_action=dataset_cfg.get("relative_action", True),
            ambiguous_prob=dataset_cfg.get("ambiguous_prob", 0.0),
        )
        val_dataset = CalvinDataset(
            root=dataset_cfg.val_path,
            instructions=val_instructions,
            taskvar=taskvar,
            max_episode_length=dataset_cfg.get("max_episode_length", 5),
            cache_size=dataset_cfg.get("cache_size_val", 100),
            max_episodes_per_task=dataset_cfg.get("max_episodes_per_task", -1),
            cameras=("front", "wrist"),
            training=False,
            image_rescale=image_rescale,
            return_low_lvl_trajectory=True,
            dense_interpolation=dataset_cfg.get("dense_interpolation", True),
            interpolation_length=dataset_cfg.get("interpolation_length", 100),
            relative_action=dataset_cfg.get("relative_action", True),
        )
        return train_dataset, val_dataset

    @staticmethod
    def _load_instructions(instructions_path, split, overloaded=False):
        """Load precomputed CLIP instruction embeddings.

        `overloaded` selects {split}_overloaded.pkl (deduplicated embeddings plus
        per-window specific/ambiguous pools) over the legacy flat {split}.pkl. It is
        opt-in rather than auto-detected: both artifacts commonly live in the same
        directory, and silently preferring the overloaded one would switch an
        existing config from CALVIN's single baked-in string to paraphrase sampling.
        """
        if overloaded:
            path = Path(f"{instructions_path}/{split}_overloaded.pkl")
            if not path.exists():
                raise FileNotFoundError(
                    f"use_overloaded_instructions=true but {path} is missing. "
                    "Generate it with preprocess_overloaded_instructions.py."
                )
            instructions = load_instruction_pool(path)
            logger.info(
                f"Loaded overloaded instructions from {path}: "
                f"{len(instructions.strings)} unique strings over "
                f"{len(instructions)} annotations"
            )
            return instructions

        path = f"{instructions_path}/{split}.pkl"
        instructions = load_instruction_pool(path)
        logger.info(f"Loaded {len(instructions)} flat instruction embeddings from {path}")
        return instructions

    def _get_loaders(self, train_dataset, val_dataset, cfg):
        """Build data loaders with distributed samplers.

        Returns (train_loader, val_loader, train_sampler).  The sampler is
        returned so the training loop can call set_epoch() on each restart,
        which re-shuffles the data ordering and avoids the sampler being
        frozen at epoch 0 for the entire 600K-step run.
        """
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(0)

        train_sampler = DistributedSampler(train_dataset) if self.is_distributed else None
        val_sampler = DistributedSampler(val_dataset, shuffle=True) if self.is_distributed else None

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=(train_sampler is None),
            num_workers=cfg.get("num_workers", 1),
            worker_init_fn=seed_worker,
            collate_fn=traj_collate_fn,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True,
            generator=g,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.get("batch_size_val", 4),
            shuffle=(val_sampler is None),
            num_workers=0,
            worker_init_fn=seed_worker,
            collate_fn=traj_collate_fn,
            pin_memory=True,
            sampler=val_sampler,
            drop_last=False,
            generator=g,
        )
        return train_loader, val_loader, train_sampler

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self):
        """Main iteration-based training loop."""
        cfg = self.cfg

        # Setup datasets and loaders
        train_dataset, val_dataset = self._get_datasets(cfg)
        train_loader, val_loader, train_sampler = self._get_loaders(train_dataset, val_dataset, cfg)

        # Resume from checkpoint — uses resume_checkpoint_path from config.
        # Starting from scratch unless explicitly set; no stale checkpoint is
        # ever loaded automatically.
        resume_path = cfg.get("resume_checkpoint_path", None)
        if cfg.get("resume", False):
            if not resume_path:
                raise ValueError("resume=true needs resume_checkpoint_path.")
            self._load_checkpoint(resume_path)
            if self.is_main_process:
                logger.info(f"Resuming from checkpoint: {resume_path} (step {self.global_step})")
        else:
            if resume_path and self.is_main_process:
                logger.warning(
                    f"resume_checkpoint_path={resume_path} is ignored because resume=false"
                )
            if self.is_main_process:
                logger.info("Starting training from scratch (no checkpoint loaded)")

        # Retention of the periodic checkpoints (last.pth and best.pth are always
        # written). save_top_k < 0 keeps every periodic checkpoint.
        topk_manager = None
        if self.is_main_process:
            checkpoint_dir = Path(cfg.get("checkpoint_dir", "checkpoints/diffuser_actor"))
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            topk_manager = TopKCheckpointManager(
                save_dir=checkpoint_dir, k=cfg.get("save_top_k", -1), mode="max"
            )
            if self._retained_state is not None:
                topk_manager.load_state_dict(self._retained_state)

        # Training loop (iteration-based)
        iter_loader = iter(train_loader)
        loader_epoch = 0  # tracks DataLoader restarts for set_epoch shuffling
        self.model.train()
        nhist = cfg.policy.get("nhist", 3)
        accumulate_grad = cfg.get("accumulate_grad_batches", 1)
        grad_clip = cfg.get("grad_clip", 1.0)

        if self.is_main_process:
            logger.info(
                f"Starting training for {cfg.train_iters} iterations "
                f"(lr={cfg.lr}, warmup={cfg.get('lr_warmup_steps', 2000)}, "
                f"grad_clip={grad_clip}, wd={cfg.get('wd', 5e-3)})"
            )

        for step_id in trange(self.global_step, cfg.train_iters, disable=not self.is_main_process):
            # Get next batch; call set_epoch on each DataLoader restart so the
            # DistributedSampler re-shuffles with a new seed each pass.
            try:
                sample = next(iter_loader)
            except StopIteration:
                loader_epoch += 1
                if train_sampler is not None:
                    train_sampler.set_epoch(loader_epoch)
                iter_loader = iter(train_loader)
                sample = next(iter_loader)

            # Zero gradients
            if step_id % accumulate_grad == 0:
                self.optimizer.zero_grad()

            # Prepare trajectory (skip first frame — it's the current gripper pose)
            sample["trajectory"] = sample["trajectory"][:, 1:]
            sample["trajectory_mask"] = sample["trajectory_mask"][:, 1:]

            # Select gripper history
            curr_gripper = sample["curr_gripper_history"][:, -nhist:]

            # Forward pass (loss computed inside model).
            loss = self.model(
                sample["trajectory"].to(self.device),
                sample["trajectory_mask"].to(self.device),
                sample["rgbs"].to(self.device),
                sample["pcds"].to(self.device),
                curr_gripper.to(self.device),
                instruction=sample["instr"].to(self.device),
            )

            # Backward pass. Divide by the accumulation count so the accumulated
            # gradient is a MEAN over the effective batch rather than a sum. Without
            # this, accumulate_grad_batches=N yields gradients N times too large --
            # which Adam mostly absorbs but grad_clip does not, so clipping would fire
            # far more aggressively and accumulation would NOT be equivalent to a real
            # batch of batch_size*N. loss.item() below stays unscaled for logging.
            (loss / accumulate_grad).backward()

            if step_id % accumulate_grad == accumulate_grad - 1:
                # Gradient clipping before the optimizer step
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)

                self.optimizer.step()
                self.scheduler.step()

                # EMA update
                if self.ema_model is not None:
                    model_ref = self.model.module if self.is_distributed else self.model
                    self.ema_model.step(model_ref)

            self.global_step = step_id + 1

            # Per-step loss logging (frequent)
            if self.is_main_process and self.use_wandb and (step_id + 1) % cfg.get("log_freq", 10) == 0:
                import wandb
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "lr": self.scheduler.get_last_lr()[0],
                    },
                    step=step_id,
                )

            # Val checkpoint (infrequent)
            if self.is_main_process and (step_id + 1) % cfg.get("val_freq", 500) == 0:
                # Validation
                self.model.eval()
                val_metrics = self._evaluate(val_loader, step_id)
                self.model.train()

                # Save checkpoint
                self._save_checkpoint(
                    topk_manager, step_id, val_metrics.get(SELECTION_METRIC, None)
                )

                if self.use_wandb:
                    import wandb
                    wandb.log(val_metrics, step=step_id)

                logger.info(
                    f"Step {step_id} | train_loss={loss.item():.4f} | lr={self.scheduler.get_last_lr()[0]:.2e} | "
                    + " | ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
                )

        # Cleanup
        if self.is_distributed:
            dist.destroy_process_group()
        if self.is_main_process and self.use_wandb:
            import wandb
            wandb.finish()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _evaluate(self, loader, step_id, max_iters=20):
        """Run evaluation and compute trajectory metrics."""
        values = {}
        device = self.device
        nhist = self.cfg.policy.get("nhist", 3)

        for i, sample in enumerate(loader):
            if i >= max_iters:
                break

            sample["trajectory"] = sample["trajectory"][:, 1:]
            sample["trajectory_mask"] = sample["trajectory_mask"][:, 1:]
            curr_gripper = sample["curr_gripper_history"][:, -nhist:]

            # Run inference
            pred = self.model(
                sample["trajectory"].to(device),
                sample["trajectory_mask"].to(device),
                sample["rgbs"].to(device),
                sample["pcds"].to(device),
                curr_gripper.to(device),
                instruction=sample["instr"].to(device),
                run_inference=True,
            )

            gt = sample["trajectory"].to(device)
            metrics = self._compute_metrics(pred, gt)
            for k, v in metrics.items():
                key = f"val/{k}"
                if key not in values:
                    values[key] = []
                values[key].append(v.item())

        # Average metrics
        values = {k: np.mean(v) for k, v in values.items()}

        if self.use_wandb:
            import wandb
            wandb.log(values, step=step_id)

        return values

    @staticmethod
    def _compute_metrics(pred, gt):
        """Compute trajectory prediction metrics."""
        # Position L2
        pos_l2 = ((pred[..., :3] - gt[..., :3]) ** 2).sum(-1).sqrt()

        # Symmetric quaternion L1
        quat_l1 = (pred[..., 3:7] - gt[..., 3:7]).abs().sum(-1)
        quat_l1_neg = (pred[..., 3:7] + gt[..., 3:7]).abs().sum(-1)
        select_mask = (quat_l1 < quat_l1_neg).float()
        quat_l1 = select_mask * quat_l1 + (1 - select_mask) * quat_l1_neg

        # Gripper accuracy
        if pred.shape[-1] > 7 and gt.shape[-1] > 7:
            openess = ((pred[..., 7:] >= 0.5) == (gt[..., 7:] > 0.0)).float()
            gripper_acc = openess.mean()
        else:
            gripper_acc = torch.tensor(0.0)

        return {
            "traj_pos_l2": pos_l2.mean(),
            "traj_pos_acc_001": (pos_l2 < 0.01).float().mean(),
            "traj_rot_l1": quat_l1.mean(),
            "traj_rot_acc_0025": (quat_l1 < 0.025).float().mean(),
            "traj_gripper_acc": gripper_acc,
        }

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, topk_manager, step_id, metric):
        """Save last.pth, best.pth when `metric` improves, and a retained periodic one.

        `metric` is the SELECTION_METRIC of this validation (higher is better). The
        best-metric bookkeeping is updated first, so last.pth carries the same value
        as best.pth.
        """
        if not self.is_main_process:
            return

        is_best = metric is not None and (
            self.best_metric is None or metric >= self.best_metric
        )
        if is_best:
            self.best_metric = float(metric)
        periodic_name = f"{step_id:07d}.pth"
        keep_periodic = topk_manager.admit(periodic_name, metric)

        model_ref = self.model.module if self.is_distributed else self.model
        checkpoint = {
            "weight": model_ref.state_dict() if not self.is_distributed
                      else self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "iter": step_id + 1,
            "selection_metric": SELECTION_METRIC,
            "metric": None if metric is None else float(metric),
            "best_metric": self.best_metric,
            "retained_checkpoints": topk_manager.state_dict(),
            # The policy settings that define the architecture (and so the
            # state-dict keys and shapes) this checkpoint needs.
            "arch_config": {
                key: OmegaConf.to_container(value) if OmegaConf.is_config(value) else value
                for key, value in self.cfg.policy.items()
            },
        }
        if self.ema_model is not None:
            checkpoint["ema_weight"] = self.ema_model.averaged_model.state_dict()
            checkpoint["ema_optimization_step"] = self.ema_model.optimization_step

        checkpoint_dir = Path(self.cfg.get("checkpoint_dir", "checkpoints/diffuser_actor"))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save latest
        torch.save(checkpoint, checkpoint_dir / "last.pth")

        # Save best
        if is_best:
            torch.save(checkpoint, checkpoint_dir / "best.pth")
            logger.info(
                f"New best checkpoint at step {step_id} ({SELECTION_METRIC}={metric:.4f})"
            )

        # Save periodic
        if keep_periodic:
            torch.save(checkpoint, checkpoint_dir / periodic_name)

    def _load_checkpoint(self, checkpoint_path):
        """Restore the training state from a checkpoint written by `_save_checkpoint`.

        Restored: model weights, EMA weights and the EMA update counter, optimizer
        state, the learning-rate schedule position, the step counter, the best
        selection metric and the retained-checkpoint set. Not restored: Python,
        NumPy and torch RNG streams, data-loader worker state and the sampler epoch.
        The data order after a resume therefore differs from an uninterrupted run;
        a resumed run is a continuation, not a bit-identical replay.
        """
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"resume_checkpoint_path does not exist: {checkpoint_path}")

        logger.info(f"Loading checkpoint: {checkpoint_path}")
        # The trainer's own checkpoints hold NumPy scalars in their metadata.
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Distributed runs save the DDP state dict ("module." keys); accept either
        # form whatever the current world size.
        model_ref = self.model.module if self.is_distributed else self.model
        weights = {k.removeprefix("module."): v for k, v in ckpt["weight"].items()}
        model_ref.load_state_dict(weights)

        if "optimizer" not in ckpt:
            raise ValueError(f"{checkpoint_path} has no optimizer state; it cannot be resumed.")
        self.optimizer.load_state_dict(ckpt["optimizer"])
        # The schedule is recomputed from the configured lr below, so a checkpoint
        # written under a different lr config does not carry its lr over.
        for pg in self.optimizer.param_groups:
            pg["lr"] = self.cfg.lr
        self.global_step = ckpt.get("iter", 0)

        # The scheduler and the EMA advance once per optimizer update, which is
        # every accumulate_grad_batches micro-steps; global_step counts micro-steps.
        accumulate = max(1, self.cfg.get("accumulate_grad_batches", 1))
        updates = self.global_step // accumulate
        if self.global_step % accumulate:
            logger.warning(
                f"Checkpoint step {self.global_step} is not a multiple of "
                f"accumulate_grad_batches={accumulate}; the partly accumulated "
                "gradient of that update is lost."
            )
        with warnings.catch_warnings():
            # Replaying the schedule calls scheduler.step() without optimizer.step().
            warnings.simplefilter("ignore", UserWarning)
            for _ in range(updates):
                self.scheduler.step()

        if self.ema_model is not None:
            if "ema_weight" not in ckpt:
                raise ValueError(
                    f"use_ema=true but {checkpoint_path} has no EMA weights; resuming "
                    "would restart the average from the resumed weights."
                )
            self.ema_model.averaged_model.load_state_dict(ckpt["ema_weight"])
            # Older checkpoints did not store the counter; it equals the number of
            # optimizer updates, since the EMA steps once per update.
            self.ema_model.optimization_step = int(ckpt.get("ema_optimization_step", updates))
        # Checkpoints written before the selection fix stored `best_loss`, the
        # *lowest* accuracy seen; it is not a usable baseline and is ignored.
        self.best_metric = ckpt.get("best_metric", None)
        self._retained_state = ckpt.get("retained_checkpoints", None)

        logger.info(f"Resumed from step {self.global_step}")
