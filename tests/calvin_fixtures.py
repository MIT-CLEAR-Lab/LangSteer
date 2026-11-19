"""Small synthetic stand-ins for the packaged CALVIN data and instruction pools.

The packaged episodes follow package_calvin.py's `state_dict` layout:
[frame_ids, rgb_pcd (T, ncam, 2, 3, 200, 200), actions [(1, 7)], camera_dicts,
grippers [(1, 7)], trajectories [(L_i, 7)], annotation_ids]. Poses are
[pos(3), euler_XYZ(3), gripper(1)], as CALVIN stores them.
"""

import pickle
from pathlib import Path

import numpy as np
import torch

from training.policies.diffuser_actor.instruction_pools import OverloadedInstructions

EMBED_SHAPE = (53, 512)


def _pose(rng):
    return np.concatenate([
        rng.uniform(-0.2, 0.2, 3),
        rng.uniform(-0.5, 0.5, 3) + np.array([3.1, 0.0, 1.5]),
        [rng.choice([-1.0, 1.0])],
    ]).astype(np.float32)


def write_episode(path, annotation_id, n_frames=4, seed=0):
    import blosc

    rng = np.random.default_rng(seed)
    rgb_pcd = torch.as_tensor(
        rng.uniform(-1, 1, (n_frames, 2, 2, 3, 200, 200)), dtype=torch.float32
    )
    grippers = [torch.as_tensor(_pose(rng)).view(1, -1) for _ in range(n_frames)]
    actions = [torch.as_tensor(_pose(rng)).view(1, -1) for _ in range(n_frames)]
    trajectories = [
        torch.stack([torch.as_tensor(_pose(rng)) for _ in range(int(rng.integers(3, 7)))])
        for _ in range(n_frames)
    ]
    episode = [
        list(range(n_frames)),
        rgb_pcd,
        actions,
        [{"front": (0, 0), "wrist": (0, 0)}],
        grippers,
        trajectories,
        [annotation_id] * n_frames,
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blosc.compress(pickle.dumps(episode)))


def write_packaged_split(root, annotation_ids, scene="D"):
    """One episode per annotation id under <root>/<scene>+0/ann_<id>.dat."""
    for i, annotation_id in enumerate(annotation_ids):
        write_episode(Path(root) / f"{scene}+0" / f"ann_{annotation_id}.dat",
                      annotation_id, seed=i)


def flat_instructions(n_annotations, seed=0):
    """The legacy flat artifact: one (1, 53, 512) embedding per annotation."""
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(1, *EMBED_SHAPE, generator=g).numpy() for _ in range(n_annotations)]


def overloaded_pool(n_annotations, with_ambiguous=(), seed=0):
    """An OverloadedInstructions pool with three specific strings per annotation and,
    for the ids in `with_ambiguous`, two shared ambiguous templates."""
    strings, specific, ambiguous, classes = [], {}, {}, {}
    for a in range(n_annotations):
        specific[a] = []
        for k in range(3):
            specific[a].append(len(strings))
            strings.append(f"specific {k} of annotation {a}.")
    templates = [len(strings), len(strings) + 1]
    strings += ["pick up the block.", "take the block."]
    for a in range(n_annotations):
        ambiguous[a] = list(templates) if a in with_ambiguous else []
        classes[a] = "lift_block" if a in with_ambiguous else None
    g = torch.Generator().manual_seed(seed)
    return OverloadedInstructions({
        "strings": strings,
        "embeddings": torch.randn(len(strings), *EMBED_SHAPE, generator=g),
        "specific_ids": specific,
        "ambiguous_ids": ambiguous,
        "classes": classes,
    })
