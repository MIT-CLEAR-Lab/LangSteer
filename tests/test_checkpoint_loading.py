"""3DDA policy checkpoint loading is strict unless a config opts out."""

import pytest
import torch
from omegaconf import OmegaConf

from policies.diffuser_actor_base import DiffuserActorBasePolicy


def _policy(cfg):
    policy = DiffuserActorBasePolicy.__new__(DiffuserActorBasePolicy)
    policy.cfg = OmegaConf.create(cfg)
    policy._device = torch.device("cpu")
    policy._model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    return policy


def test_matching_checkpoint_loads_with_ddp_prefix(tmp_path):
    source = _policy({})
    weights = {f"module.{k}": v + 1 for k, v in source._model.state_dict().items()}
    torch.save({"weight": weights}, tmp_path / "ok.pth")
    target = _policy({})
    target.load_checkpoint(str(tmp_path / "ok.pth"))
    for k, v in target._model.state_dict().items():
        assert torch.equal(v, weights[f"module.{k}"])


def test_incompatible_checkpoint_is_rejected_by_default(tmp_path):
    state = _policy({})._model.state_dict()
    state.pop("1.bias")
    torch.save({"weight": state}, tmp_path / "missing.pth")
    with pytest.raises(RuntimeError, match="Missing key"):
        _policy({}).load_checkpoint(str(tmp_path / "missing.pth"))
    # Explicit opt-out still loads what matches.
    _policy({"strict_checkpoint": False}).load_checkpoint(str(tmp_path / "missing.pth"))


@pytest.mark.parametrize("name", ["diffuser_actor", "diffuser_actor_overloaded"])
def test_public_policy_configs_load_strictly(name):
    from hydra import compose, initialize

    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="config", overrides=[f"policy={name}"])
    assert cfg.policy.strict_checkpoint is True
