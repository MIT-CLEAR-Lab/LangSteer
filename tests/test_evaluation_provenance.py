"""Session provenance of evaluation runs (code, prompts, layout, dependency versions)."""
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core.types import Observation
from tests.frozen_prompts import PROMPT_SHA256
from utils import evaluation_config as provenance

ROOT = Path(__file__).resolve().parents[1]


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def test_source_identity_tracks_revision_and_uncommitted_changes(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "a.py").write_text("x = 1\n")
    _git(tmp_path, "add", "a.py")
    _git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "c")
    clean = provenance.source_identity(tmp_path)
    assert len(clean["revision"]) == 40 and clean["dirty"] is False and clean["dirty_sha256"] is None

    (tmp_path / "a.py").write_text("x = 2\n")
    modified = provenance.source_identity(tmp_path)
    assert modified["revision"] == clean["revision"] and modified["dirty"] is True
    (tmp_path / "b.py").write_text("y = 1\n")
    untracked = provenance.source_identity(tmp_path)
    assert untracked["dirty_sha256"] != modified["dirty_sha256"]
    assert provenance.source_identity(tmp_path / "missing")["revision"] is None


def test_prompt_hashes_follow_the_prompt_directory_in_effect(tmp_path, monkeypatch):
    monkeypatch.delenv("LANGSTEER_PROMPT_DIR", raising=False)
    default = provenance.steering_prompt_hashes()
    assert default["prompt_dir_overridden"] is False
    assert default["sha256"] == {f"calvin/{name}": digest for name, digest in PROMPT_SHA256.items()}

    variant = tmp_path / "prompts"
    shutil.copytree(ROOT / "voxposer" / "prompts", variant)
    with (variant / "calvin" / "composer_prompt.txt").open("a") as f:
        f.write("\n# variant\n")
    monkeypatch.setenv("LANGSTEER_PROMPT_DIR", str(variant))
    overridden = provenance.steering_prompt_hashes()
    assert overridden["prompt_dir_overridden"] is True
    changed = {k for k in default["sha256"] if default["sha256"][k] != overridden["sha256"][k]}
    assert changed == {"calvin/composer_prompt.txt"}


def test_package_versions_and_launch_layout(monkeypatch):
    versions = provenance.package_versions()
    assert versions["python"] and versions["numpy"] == np.__version__
    monkeypatch.delenv(provenance.LAUNCH_ENV, raising=False)
    assert provenance.launch_layout() is None
    monkeypatch.setenv(provenance.LAUNCH_ENV, json.dumps({"workers": 5}))
    assert provenance.launch_layout() == {"workers": 5}


def test_only_changed_steering_prompts_refuse_a_resume():
    base = {"source": {"revision": "a"}, "packages": {"torch": "1"}, "process_plan": [],
            "launch": None, "steering_prompts": {"sha256": {"calvin/x.txt": "1"}}}
    assert provenance.check_session_provenance([], base, "r.json") == []
    moved = {**base, "source": {"revision": "b"}, "packages": {"torch": "2"}}
    assert provenance.check_session_provenance([base], moved, "r.json") == ["source", "packages"]
    reprompted = {**base, "steering_prompts": {"sha256": {"calvin/x.txt": "2"}}}
    with pytest.raises(ValueError, match="steering prompts changed"):
        provenance.check_session_provenance([base], reprompted, "r.json")


def test_launcher_passes_its_layout_to_every_axis(monkeypatch, tmp_path):
    from scripts import run_overloaded_evaluation as launcher

    captured = {}
    monkeypatch.setattr(launcher, "execute_commands", lambda commands, env, workers: captured.update(env=env))
    monkeypatch.setattr(launcher, "summarize", lambda output_dir: None)
    monkeypatch.setattr("sys.argv", ["x", "matrix", "--dataset-path", str(tmp_path), "--output-dir",
                                     str(tmp_path / "out"), "--axes", "P0", "P3", "--dry-run"])
    launcher.main()
    layout = json.loads(captured["env"][provenance.LAUNCH_ENV])
    assert layout == {"backbone": "3dda", "phase": "matrix", "workers": 1,
                      "axes": ["P0", "P3"], "conditions": launcher.CONDITIONS}


def _run(tmp_path, monkeypatch, *, steered=True):
    """One condition through run_condition with every simulator part faked."""
    from omegaconf import OmegaConf
    from scripts import run_evaluation as evaluation

    task = "lift_red_block_table"
    def obs():
        return Observation({}, np.zeros(1), np.zeros(7), "lift the red block")
    env = SimpleNamespace(
        _provide_pcd_images=True, _current_step=0, task_description="lift the red block",
        _task_name=task, set_task=lambda t: None, reset=lambda **kw: obs(), close=lambda: None,
        get_scene_state=lambda: {"robot_obs": np.zeros(1), "scene_obs": np.zeros(1)},
        step=lambda action: (obs(), 0, True, {"success": True}))
    policy = SimpleNamespace(reset=lambda: None, forward=lambda o, steering=None: None)
    steering = SimpleNamespace(_value_map=object(), setup_episode=lambda *a, **kw: None)
    monkeypatch.setattr(evaluation, "instantiate_env", lambda cfg: env)
    monkeypatch.setattr(evaluation, "instantiate_policy", lambda cfg: policy)
    monkeypatch.setattr(evaluation, "instantiate_steering", lambda cfg: steering if steered else None)
    artifact = tmp_path / "artifact"
    artifact.write_text("input")
    cfg = OmegaConf.create({
        "env": {"instruction_level": "specific", "lang_ann_path": str(artifact)},
        "policy": {"ckpt_path": str(artifact)},
        "steering": {"name": "voxposer" if steered else "none"},
    })
    plan = [("overloaded_base", [task]), ("overloaded_base_steered", [task])]
    return evaluation.run_condition(
        {"condition_name": "test", "tasks": [task]}, cfg, {task: [(np.zeros(1), np.zeros(1))]},
        {task: [7]}, 1, 360, 42, tmp_path / "overloaded_base_steered.json",
        config_name="overloaded_base_steered", process_plan=plan)


def test_sessions_are_recorded_and_older_directories_still_resume(tmp_path, monkeypatch):
    monkeypatch.delenv("LANGSTEER_PROMPT_DIR", raising=False)
    first = _run(tmp_path, monkeypatch)
    (session,) = first["provenance"]
    assert session["condition"] == "overloaded_base_steered"
    assert [p["condition"] for p in session["process_plan"]] == ["overloaded_base", "overloaded_base_steered"]
    assert session["source"]["revision"] and session["packages"]["torch"]
    assert session["steering_prompts"]["sha256"]["calvin/composer_prompt.txt"] == PROMPT_SHA256["composer_prompt.txt"]
    assert session["completed_episodes_at_start"] == {"lift_red_block_table": 0}

    # A directory written before provenance existed: same run_metadata, no sessions.
    path = tmp_path / "overloaded_base_steered.json"
    stored = json.loads(path.read_text())
    run_metadata = stored["run_metadata"]
    del stored["provenance"]
    path.write_text(json.dumps(stored))
    resumed = _run(tmp_path, monkeypatch)
    assert resumed["run_metadata"] == run_metadata
    assert len(resumed["provenance"]) == 1
    assert resumed["provenance"][0]["completed_episodes_at_start"] == {"lift_red_block_table": 1}
    assert len(resumed["tasks"]["lift_red_block_table"]["episodes"]) == 1

    again = _run(tmp_path, monkeypatch)
    assert len(again["provenance"]) == 2


def test_resume_with_other_steering_prompts_is_refused(tmp_path, monkeypatch):
    monkeypatch.delenv("LANGSTEER_PROMPT_DIR", raising=False)
    _run(tmp_path, monkeypatch)
    variant = tmp_path / "prompts"
    shutil.copytree(ROOT / "voxposer" / "prompts", variant)
    (variant / "calvin" / "get_affordance_map_prompt.txt").write_text("changed")
    monkeypatch.setenv("LANGSTEER_PROMPT_DIR", str(variant))
    with pytest.raises(ValueError, match="steering prompts changed"):
        _run(tmp_path, monkeypatch)
