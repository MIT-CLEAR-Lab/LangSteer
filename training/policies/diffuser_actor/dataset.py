"""Dataset classes for 3D Diffuser Actor training on CALVIN.

Merged from:
    - 3d_diffuser_actor/datasets/dataset_engine.py (episode discovery, caching and
      length, now `EpisodeDataset`; its RLBench sample construction is not carried)
    - 3d_diffuser_actor/datasets/dataset_calvin.py (CalvinDataset)
    - 3d_diffuser_actor/datasets/utils.py (loader, Resize, TrajectoryInterpolator)
"""

import pickle
from collections import defaultdict, Counter
import itertools
import math
import random
from pathlib import Path
from pickle import UnpicklingError
from time import time

import einops
import numpy as np
from scipy.interpolate import CubicSpline, interp1d
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_f

from policies.diffuser_actor_components.rotation_utils import normalise_quat
from training.policies.diffuser_actor.instruction_pools import OverloadedInstructions
from training.policies.diffuser_actor.preprocessing.calvin_utils import (
    to_relative_action,
    convert_rotation,
)


# ---------------------------------------------------------------------------
# File loaders
# ---------------------------------------------------------------------------

def loader(file):
    """Load episode data from .npy, .dat (blosc), or .pkl files."""
    if str(file).endswith(".npy"):
        try:
            content = np.load(file, allow_pickle=True)
            return content
        except UnpicklingError as e:
            print(f"Can't load {file}: {e}")
    elif str(file).endswith(".dat"):
        try:
            with open(file, "rb") as f:
                import blosc
                content = pickle.loads(blosc.decompress(f.read()))
            return content
        except UnpicklingError as e:
            print(f"Can't load {file}: {e}")
    elif str(file).endswith(".pkl"):
        try:
            with open(file, 'rb') as f:
                content = pickle.load(f)
            return content
        except UnpicklingError as e:
            print(f"Can't load {file}: {e}")
    return None


# ---------------------------------------------------------------------------
# Augmentation and interpolation utilities
# ---------------------------------------------------------------------------

class Resize:
    """Resize and pad/crop the image and aligned point cloud."""

    def __init__(self, scales):
        self.scales = scales

    def __call__(self, **kwargs):
        """Accept tensors as T, N, C, H, W."""
        keys = list(kwargs.keys())
        if len(keys) == 0:
            raise RuntimeError("No args")

        sc = np.random.uniform(*self.scales)

        t, n, c, raw_h, raw_w = kwargs[keys[0]].shape
        kwargs = {n: arg.flatten(0, 1) for n, arg in kwargs.items()}
        resized_size = [int(raw_h * sc), int(raw_w * sc)]

        kwargs = {
            n: transforms_f.resize(
                arg, resized_size, transforms.InterpolationMode.NEAREST
            )
            for n, arg in kwargs.items()
        }

        if raw_h > resized_size[0] or raw_w > resized_size[1]:
            right_pad = max(raw_w - resized_size[1], 0)
            bottom_pad = max(raw_h - resized_size[0], 0)
            kwargs = {
                n: transforms_f.pad(
                    arg, padding=[0, 0, right_pad, bottom_pad],
                    padding_mode="reflect",
                )
                for n, arg in kwargs.items()
            }

        i, j, h, w = transforms.RandomCrop.get_params(
            kwargs[keys[0]], output_size=(raw_h, raw_w)
        )
        kwargs = {
            n: transforms_f.crop(arg, i, j, h, w) for n, arg in kwargs.items()
        }

        kwargs = {
            n: einops.rearrange(arg, "(t n) c h w -> t n c h w", t=t)
            for n, arg in kwargs.items()
        }

        return kwargs


class TrajectoryInterpolator:
    """Interpolate a trajectory to have fixed length."""

    def __init__(self, use=False, interpolation_length=50):
        self._use = use
        self._interpolation_length = interpolation_length

    def __call__(self, trajectory):
        if not self._use:
            return trajectory
        trajectory = trajectory.numpy()
        old_num_steps = len(trajectory)
        old_steps = np.linspace(0, 1, old_num_steps)
        new_steps = np.linspace(0, 1, self._interpolation_length)

        resampled = np.empty((self._interpolation_length, trajectory.shape[1]))
        for i in range(trajectory.shape[1]):
            if i == (trajectory.shape[1] - 1):  # gripper opening
                interpolator = interp1d(old_steps, trajectory[:, i])
            else:
                interpolator = CubicSpline(old_steps, trajectory[:, i])
            resampled[:, i] = interpolator(new_steps)

        resampled = torch.tensor(resampled)
        if trajectory.shape[1] == 8:
            resampled[:, 3:7] = normalise_quat(resampled[:, 3:7])
        return resampled


# ---------------------------------------------------------------------------
# Episode-file dataset base (loading, caching, length; from dataset_engine.py)
# ---------------------------------------------------------------------------

class EpisodeDataset(Dataset):
    """Episodes stored one per file under `<root>/<task>+<variation>/`.

    Discovers the episode files, reads them through an optional in-memory cache,
    and reports `num_iters` as the length when given (iteration-based training).
    Subclasses build the samples in `__getitem__`.
    """

    def __init__(
        self,
        root,
        taskvar,
        max_episode_length=5,
        cache_size=0,
        max_episodes_per_task=100,
        num_iters=None,
        cameras=("front", "wrist"),
        training=True,
        image_rescale=(1.0, 1.0),
        return_low_lvl_trajectory=False,
        dense_interpolation=False,
        interpolation_length=100,
        relative_action=True,
    ):
        self._cache = {}
        self._cache_size = cache_size
        self._cameras = cameras
        self._max_episode_length = max_episode_length
        self._num_iters = num_iters
        self._training = training
        self._taskvar = taskvar
        self._return_low_lvl_trajectory = return_low_lvl_trajectory
        if isinstance(root, (Path, str)):
            root = [Path(root)]
        self._root = [Path(r).expanduser() for r in root]
        self._relative_action = relative_action

        if return_low_lvl_trajectory:
            assert dense_interpolation
            self._interpolate_traj = TrajectoryInterpolator(
                use=dense_interpolation,
                interpolation_length=interpolation_length,
            )

        self._num_vars = Counter()
        for root, (task, var) in itertools.product(self._root, taskvar):
            data_dir = root / f"{task}+{var}"
            if data_dir.is_dir():
                self._num_vars[task] += 1

        if self._training:
            self._resize = Resize(scales=image_rescale)

        episodes_by_task = defaultdict(list)
        for root, (task, var) in itertools.product(self._root, taskvar):
            data_dir = root / f"{task}+{var}"
            if not data_dir.is_dir():
                print(f"Can't find dataset folder {data_dir}")
                continue
            npy_episodes = [(task, var, ep) for ep in data_dir.glob("*.npy")]
            dat_episodes = [(task, var, ep) for ep in data_dir.glob("*.dat")]
            pkl_episodes = [(task, var, ep) for ep in data_dir.glob("*.pkl")]
            episodes = npy_episodes + dat_episodes + pkl_episodes
            if max_episodes_per_task > -1:
                episodes = episodes[
                    :max_episodes_per_task // self._num_vars[task] + 1
                ]
            if len(episodes) == 0:
                print(f"Can't find episodes at folder {data_dir}")
                continue
            episodes_by_task[task] += episodes

        self._episodes = []
        self._num_episodes = 0
        for task, eps in episodes_by_task.items():
            if len(eps) > max_episodes_per_task and max_episodes_per_task > -1:
                eps = random.sample(eps, max_episodes_per_task)
            self._episodes += eps
            self._num_episodes += len(eps)

        print(f"Created dataset from {root} with {self._num_episodes}")

    def read_from_cache(self, args):
        if self._cache_size == 0:
            return loader(args)
        if args in self._cache:
            return self._cache[args]
        value = loader(args)
        if len(self._cache) == self._cache_size:
            key = list(self._cache.keys())[int(time()) % self._cache_size]
            del self._cache[key]
        if len(self._cache) < self._cache_size:
            self._cache[args] = value
        return value

    @staticmethod
    def _unnormalize_rgb(rgb):
        return rgb / 2 + 0.5

    def __getitem__(self, episode_id):
        raise NotImplementedError

    def __len__(self):
        if self._num_iters is not None:
            return self._num_iters
        return self._num_episodes


# ---------------------------------------------------------------------------
# CALVIN-specific dataset (from dataset_calvin.py)
# ---------------------------------------------------------------------------

class CalvinDataset(EpisodeDataset):
    """CALVIN dataset for 3D Diffuser Actor training.

    `instructions` holds the precomputed CLIP embeddings, indexed by the
    annotation id stored in each episode file: the flat list from
    preprocess_instructions.py, or an `OverloadedInstructions` pool, from which
    each training sample draws (ambiguous with probability `ambiguous_prob` when
    the window's task has an ambiguous tier). The other arguments are those of
    `EpisodeDataset`.
    """

    def __init__(self, root, instructions, taskvar, ambiguous_prob=0.0, **kwargs):
        if instructions is None:
            raise ValueError(
                "CalvinDataset needs instruction embeddings (the flat {split}.pkl or "
                "{split}_overloaded.pkl); the model is language-conditioned."
            )
        super().__init__(root, taskvar, **kwargs)
        self._instructions = instructions
        # p(ambiguous | the window's task has an ambiguous tier). Only consulted when
        # `instructions` is an OverloadedInstructions pool; 0.0 reproduces the legacy
        # one-instruction-per-window behaviour exactly.
        self._ambiguous_prob = ambiguous_prob

    def __getitem__(self, episode_id):
        episode_id %= self._num_episodes
        task, variation, file = self._episodes[episode_id]
        episode = self.read_from_cache(file)
        if episode is None:
            return None

        chunk = random.randint(
            0, math.ceil(len(episode[0]) / self._max_episode_length) - 1
        )
        frame_ids = episode[0][
            chunk * self._max_episode_length:
            (chunk + 1) * self._max_episode_length
        ]
        states = torch.stack([
            episode[1][i] if isinstance(episode[1][i], torch.Tensor)
            else torch.from_numpy(episode[1][i])
            for i in frame_ids
        ])
        if episode[3]:
            cameras = list(episode[3][0].keys())
            assert all(c in cameras for c in self._cameras)
            index = torch.tensor([cameras.index(c) for c in self._cameras])
            states = states[:, index]

        # Split RGB and XYZ, crop to 160x160
        rgbs = states[:, :, 0, :, 20:180, 20:180]
        pcds = states[:, :, 1, :, 20:180, 20:180]
        rgbs = self._unnormalize_rgb(rgbs)

        action = torch.cat([episode[2][i] for i in frame_ids])

        # Instruction embeddings (CALVIN uses flat array, not task-indexed)
        instr_ind = episode[6][0]
        if isinstance(self._instructions, OverloadedInstructions):
            # Overloaded mode: draw from this window's specific/ambiguous pools.
            # Validation stays deterministic and specific-only so its loss remains
            # comparable to a baseline trained without the ambiguous tier.
            instr = self._instructions.draw(
                instr_ind,
                ambiguous_prob=self._ambiguous_prob,
                deterministic=not self._training,
            ).repeat(len(rgbs), 1, 1)
        else:
            instr = torch.as_tensor(self._instructions[instr_ind])
            instr = instr.repeat(len(rgbs), 1, 1)

        gripper = torch.cat([episode[4][i] for i in frame_ids])

        # gripper history
        if len(episode) > 7:
            gripper_history = torch.cat([
                episode[7][i] for i in frame_ids
            ], dim=0)
        else:
            gripper_history = torch.stack([
                torch.cat([episode[4][max(0, i-2)] for i in frame_ids]),
                torch.cat([episode[4][max(0, i-1)] for i in frame_ids]),
                gripper
            ], dim=1)

        # Low-level trajectory
        traj, traj_lens = None, 0
        if self._return_low_lvl_trajectory:
            if len(episode) > 5:
                traj_items = [
                    self._interpolate_traj(episode[5][i]) for i in frame_ids
                ]
            else:
                traj_items = [
                    self._interpolate_traj(
                        torch.cat([episode[4][i], episode[2][i]], dim=0)
                    ) for i in frame_ids
                ]
            max_l = max(len(item) for item in traj_items)
            traj = torch.zeros(
                len(traj_items), max_l, traj_items[0].shape[-1]
            )
            traj_lens = torch.as_tensor([len(item) for item in traj_items])
            for i, item in enumerate(traj_items):
                traj[i, :len(item)] = item
            traj_mask = torch.zeros(traj.shape[:-1])
            for i, len_ in enumerate(traj_lens.long()):
                traj_mask[i, len_:] = 1

        if self._training:
            if traj is not None:
                for t, tlen in enumerate(traj_lens):
                    traj[t, tlen:] = 0
            modals = self._resize(rgbs=rgbs, pcds=pcds)
            rgbs = modals["rgbs"]
            pcds = modals["pcds"]

        # Compute relative action
        if self._relative_action and traj is not None:
            rel_traj = torch.zeros_like(traj)
            for i in range(traj.shape[0]):
                for j in range(traj.shape[1]):
                    rel_traj[i, j] = torch.as_tensor(to_relative_action(
                        traj[i, j].numpy(), traj[i, 0].numpy(), clip=False
                    ))
            traj = rel_traj

        # Convert Euler angles to quaternion
        action = torch.cat([
            action[..., :3],
            torch.as_tensor(convert_rotation(action[..., 3:6])),
            action[..., 6:]
        ], dim=-1)
        gripper = torch.cat([
            gripper[..., :3],
            torch.as_tensor(convert_rotation(gripper[..., 3:6])),
            gripper[..., 6:]
        ], dim=-1)
        gripper_history = torch.cat([
            gripper_history[..., :3],
            torch.as_tensor(convert_rotation(gripper_history[..., 3:6])),
            gripper_history[..., 6:]
        ], dim=-1)
        if traj is not None:
            traj = torch.cat([
                traj[..., :3],
                torch.as_tensor(convert_rotation(traj[..., 3:6])),
                traj[..., 6:]
            ], dim=-1)

        ret_dict = {
            "task": [task for _ in frame_ids],
            "rgbs": rgbs,
            "pcds": pcds,
            "action": action,
            "instr": instr,
            "curr_gripper": gripper,
            "curr_gripper_history": gripper_history,
        }
        if self._return_low_lvl_trajectory:
            ret_dict.update({
                "trajectory": traj,
                "trajectory_mask": traj_mask.bool(),
            })
        return ret_dict


# ---------------------------------------------------------------------------
# Collation function
# ---------------------------------------------------------------------------

def traj_collate_fn(batch):
    """Custom collation for variable-length trajectories.

    `trajectory_mask` stays bool; everything else is float.
    """
    keys = [
        "trajectory", "trajectory_mask",
        "rgbs", "pcds",
        "curr_gripper", "curr_gripper_history", "action", "instr"
    ]
    def _cast(t, key):
        if key == "trajectory_mask":
            return t
        return t.float()
    ret_dict = {
        key: torch.cat([_cast(item[key], key) for item in batch])
        for key in keys
    }
    ret_dict["task"] = []
    for item in batch:
        ret_dict["task"] += item['task']
    return ret_dict
