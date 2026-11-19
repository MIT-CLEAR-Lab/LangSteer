"""The STRUCTURALLY_UNUSED_PARAMS list must exactly match reality.

If it drifts, find_unused_parameters=False makes DDP raise on the first backward.
This test catches the drift locally instead. It runs one forward/backward of the
full model on a synthetic packaged episode, on the GPU when there is one and on
the CPU otherwise. It needs the CLIP RN50 weights, which clip.load() downloads to
~/.cache/clip on first use; it is skipped when they are not there.
"""
import types
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize

from training.policies.diffuser_actor.trainer import DiffuserActorTrainingWorkspace as W

pytestmark = pytest.mark.skipif(
    not (Path.home() / ".cache" / "clip" / "RN50.pt").exists(),
    reason="needs the CLIP RN50 weights in ~/.cache/clip",
)


@pytest.fixture(scope="module")
def unused_names(tmp_path_factory):
    """Names of trainable params that receive no gradient, from a real backward."""
    import random

    from tests.calvin_fixtures import overloaded_pool, write_packaged_split
    from training.policies.diffuser_actor.dataset import CalvinDataset

    root = tmp_path_factory.mktemp("packaged") / "training"
    write_packaged_split(root, [0, 1])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="config",
                      overrides=["training=diffuser_actor_overloaded"]).training
    model = W._build_model(types.SimpleNamespace(is_main_process=False), cfg).to(device)
    ds = CalvinDataset(
        root=root, instructions=overloaded_pool(2, with_ambiguous=(0,)),
        taskvar=[("D", 0)], max_episode_length=30, cache_size=0, max_episodes_per_task=2,
        cameras=("front", "wrist"), training=True, image_rescale=(1.0, 1.0),
        return_low_lvl_trajectory=True, dense_interpolation=True,
        interpolation_length=20, ambiguous_prob=0.3,
    )
    random.seed(0)
    s = ds[0]
    nhist = cfg.policy.get("nhist", 3)
    loss = model(s["trajectory"][:, 1:].to(device), s["trajectory_mask"][:, 1:].to(device),
                 s["rgbs"].to(device), s["pcds"].to(device),
                 s["curr_gripper_history"][:, -nhist:].to(device),
                 instruction=s["instr"].to(device))
    loss.backward()
    return {n for n, p in model.named_parameters() if p.requires_grad and p.grad is None}


def test_freeze_list_covers_every_unused_param(unused_names):
    """Anything unused but NOT frozen makes DDP raise with find_unused_parameters=False."""
    uncovered = {n for n in unused_names
                 if not n.startswith(W.STRUCTURALLY_UNUSED_PARAMS)}
    assert not uncovered, f"unused but not in the freeze list: {sorted(uncovered)}"


def test_freeze_list_freezes_nothing_that_is_used(unused_names):
    """The dangerous direction: freezing a param that DOES train would break training."""
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="config",
                      overrides=["training=diffuser_actor_overloaded"]).training
    model = W._build_model(types.SimpleNamespace(is_main_process=False), cfg)
    would_freeze = {n for n, p in model.named_parameters()
                    if p.requires_grad and n.startswith(W.STRUCTURALLY_UNUSED_PARAMS)}
    wrongly = would_freeze - unused_names
    assert not wrongly, f"would freeze params that receive gradients: {sorted(wrongly)}"


def test_language_path_is_live(unused_names):
    """Guard the premise: the forward language cross-attention must train."""
    lang_unused = {n for n in unused_names if "lang" in n}
    assert all("ffn_12" in n or "norm_122" in n for n in lang_unused), (
        f"unexpected dead language params: {sorted(lang_unused)}"
    )
