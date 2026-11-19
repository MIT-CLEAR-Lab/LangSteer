"""Training configuration and workspace behaviour that needs no dataset or GPU."""

import pytest
from hydra import compose, initialize

from training.policies.diffuser_actor.trainer import check_dataset_paths

TRAINING_CONFIGS = ["diffuser_actor_calvin", "diffuser_actor_overloaded"]


def _training_cfg(group):
    with initialize(version_base=None, config_path="../conf"):
        return compose(config_name="config", overrides=[f"training={group}"]).training


@pytest.mark.parametrize("group", TRAINING_CONFIGS)
def test_missing_dataset_variable_is_a_setup_error(group, monkeypatch):
    monkeypatch.delenv("CALVIN_3DA_DATASET_PATH", raising=False)
    monkeypatch.delenv("CALVIN_3DA_INSTRUCTIONS_PATH", raising=False)
    cfg = _training_cfg(group)
    with pytest.raises(RuntimeError, match="CALVIN_3DA_DATASET_PATH"):
        check_dataset_paths(cfg.dataset)


@pytest.mark.parametrize("group", TRAINING_CONFIGS)
def test_dataset_paths_follow_the_environment(group, monkeypatch, tmp_path):
    (tmp_path / "training").mkdir()
    (tmp_path / "validation").mkdir()
    monkeypatch.setenv("CALVIN_3DA_DATASET_PATH", str(tmp_path))
    monkeypatch.delenv("CALVIN_3DA_INSTRUCTIONS_PATH", raising=False)
    cfg = _training_cfg(group)
    check_dataset_paths(cfg.dataset)
    assert cfg.dataset.train_path == f"{tmp_path}/training"
    assert cfg.dataset.val_path == f"{tmp_path}/validation"
    assert cfg.dataset.instructions_path == str(tmp_path)

    monkeypatch.setenv("CALVIN_3DA_INSTRUCTIONS_PATH", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="instructions_path"):
        check_dataset_paths(cfg.dataset)


# ---------------------------------------------------------------------------
# Checkpoint selection and retention
# ---------------------------------------------------------------------------

def _workspace(tmp_path, save_top_k=-1):
    """A workspace with a tiny model, enough for checkpoint save/load."""
    import torch
    from omegaconf import OmegaConf

    from training.policies.diffuser_actor.trainer import DiffuserActorTrainingWorkspace as W

    ws = W.__new__(W)
    ws.cfg = OmegaConf.create({
        "checkpoint_dir": str(tmp_path), "save_top_k": save_top_k, "lr": 1e-3,
        "train_iters": 100, "lr_warmup_steps": 2, "policy": {},
    })
    ws.is_main_process = True
    ws.is_distributed = False
    ws.model = torch.nn.Linear(2, 2)
    ws.optimizer = torch.optim.AdamW(ws.model.parameters(), lr=1e-3)
    ws.scheduler = ws._build_scheduler(ws.optimizer, ws.cfg)
    ws.ema_model = None
    ws.global_step = 0
    ws.best_metric = None
    ws._retained_state = None
    return ws


def _manager(ws):
    from training.common.checkpoint_util import TopKCheckpointManager

    return TopKCheckpointManager(ws.cfg.checkpoint_dir, k=ws.cfg.save_top_k, mode="max")


def test_best_checkpoint_maximises_position_accuracy(tmp_path):
    import torch

    ws = _workspace(tmp_path)
    manager = _manager(ws)
    for step, acc in [(9, 0.1), (19, 0.5), (29, 0.9), (39, 0.3)]:
        ws._save_checkpoint(manager, step, acc)
    best = torch.load(tmp_path / "best.pth", weights_only=False)
    last = torch.load(tmp_path / "last.pth", weights_only=False)
    assert best["iter"] == 30 and best["metric"] == pytest.approx(0.9)
    assert last["iter"] == 40 and last["best_metric"] == pytest.approx(0.9)
    assert sorted(p.name for p in tmp_path.glob("00*.pth")) == [
        "0000009.pth", "0000019.pth", "0000029.pth", "0000039.pth"
    ]


def test_save_top_k_retains_the_best_periodic_checkpoints(tmp_path):
    import torch

    ws = _workspace(tmp_path, save_top_k=2)
    manager = _manager(ws)
    for step, acc in [(9, 0.1), (19, 0.5), (29, 0.9), (39, 0.3), (49, 0.7)]:
        ws._save_checkpoint(manager, step, acc)
    assert sorted(p.name for p in tmp_path.glob("00*.pth")) == ["0000029.pth", "0000049.pth"]

    # The retained set travels in the checkpoint and survives a resume.
    resumed = _workspace(tmp_path, save_top_k=2)
    resumed._retained_state = torch.load(tmp_path / "last.pth", weights_only=False)[
        "retained_checkpoints"
    ]
    manager = _manager(resumed)
    manager.load_state_dict(resumed._retained_state)
    resumed.best_metric = 0.9
    resumed._save_checkpoint(manager, 59, 0.8)
    assert sorted(p.name for p in tmp_path.glob("00*.pth")) == ["0000029.pth", "0000059.pth"]


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

def _train_updates(ws, micro_steps, accumulate):
    """The optimizer/scheduler/EMA cadence of DiffuserActorTrainingWorkspace.train()."""
    import torch

    for step_id in range(micro_steps):
        if step_id % accumulate == 0:
            ws.optimizer.zero_grad()
        (ws.model(torch.ones(1, 2)).sum() / accumulate).backward()
        if step_id % accumulate == accumulate - 1:
            ws.optimizer.step()
            ws.scheduler.step()
            ws.ema_model.step(ws.model)
        ws.global_step = step_id + 1


def test_resume_restores_schedule_and_ema_with_accumulation(tmp_path):
    import torch

    from training.common.ema_model import EMAModel

    def make():
        torch.manual_seed(0)
        ws = _workspace(tmp_path)
        ws.cfg.accumulate_grad_batches = 4
        ws.cfg.lr_warmup_steps = 10
        ws.ema_model = EMAModel(model=ws.model, power=0.75)
        return ws

    original = make()
    _train_updates(original, micro_steps=16, accumulate=4)
    original._save_checkpoint(_manager(original), 15, 0.5)
    assert original.ema_model.optimization_step == 4

    resumed = make()
    resumed._load_checkpoint(tmp_path / "last.pth")
    assert resumed.global_step == 16
    assert resumed.scheduler.last_epoch == 4
    assert resumed.scheduler.get_last_lr() == pytest.approx(original.scheduler.get_last_lr())
    assert [g["lr"] for g in resumed.optimizer.param_groups] == pytest.approx(
        [g["lr"] for g in original.optimizer.param_groups]
    )
    assert resumed.ema_model.optimization_step == 4
    for a, b in zip(resumed.ema_model.averaged_model.parameters(),
                    original.ema_model.averaged_model.parameters()):
        assert torch.equal(a, b)
    for a, b in zip(resumed.model.parameters(), original.model.parameters()):
        assert torch.equal(a, b)
    assert resumed.best_metric == pytest.approx(0.5)


def test_resume_from_a_missing_checkpoint_fails(tmp_path):
    ws = _workspace(tmp_path)
    with pytest.raises(FileNotFoundError):
        ws._load_checkpoint(tmp_path / "missing.pth")


def test_resume_without_ema_weights_fails_when_ema_is_on(tmp_path):
    import torch

    from training.common.ema_model import EMAModel

    ws = _workspace(tmp_path)
    ws._save_checkpoint(_manager(ws), 9, 0.1)  # no EMA in this run
    resumed = _workspace(tmp_path)
    resumed.ema_model = EMAModel(model=resumed.model)
    with pytest.raises(ValueError, match="EMA"):
        resumed._load_checkpoint(tmp_path / "last.pth")
    assert torch.load(tmp_path / "last.pth", weights_only=False)["iter"] == 10
