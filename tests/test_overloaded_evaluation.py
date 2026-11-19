from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from core.types import Observation
from utils.evaluation_config import load_evaluation_config, validate_resume
from utils.language_distillation import LanguageDistiller
from utils.rollout import EpisodeRunner
from voxposer.llm_cache import DiskCache
from voxposer.lmp import LLMBackend

ROOT = Path(__file__).resolve().parents[1]


def test_four_conditions_share_model_tasks_and_steering_settings():
    names = ["overloaded_base", "overloaded_distilled", "overloaded_base_steered", "overloaded_distilled_steered"]
    configs = [load_evaluation_config(ROOT / "conf", n) for n in names]
    assert all(c["policy"] == configs[0]["policy"] for c in configs)
    assert all(c["tasks"] == configs[0]["tasks"] and len(c["tasks"]) == 34 for c in configs)
    assert [c["steering"] for c in configs] == ["none", "none", "voxposer", "voxposer"]
    assert [bool(c.get("distillation")) for c in configs] == [False, True, False, True]
    assert configs[2]["steering_overrides"] == configs[3]["steering_overrides"]
    assert configs[1]["distillation"] == configs[3]["distillation"]
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
        cfg = compose(config_name="config", overrides=["policy=diffuser_actor_overloaded"])
    assert cfg.policy.text_max_length == 53
    assert cfg.policy.strict_checkpoint
    assert cfg.env.instruction_level == "specific"


def test_policy_text_changes_on_every_step_without_changing_env_or_callbacks():
    original = "please lift the red block from the table"
    observations = [Observation({}, np.zeros(1), np.zeros(7), original) for _ in range(3)]
    seen, callbacks = [], []
    steering = object()
    class Env:
        step_count = 0
        def step(self, action):
            self.step_count += 1
            return observations[self.step_count], 0, self.step_count == 2, {"success": True}
    class Policy:
        def reset(self):
            pass
        def forward(self, obs, steering=None):
            seen.append((obs.instruction, steering))
            return None
    result = EpisodeRunner(Env(), Policy(), steering).run_episode(
        initial_obs=observations[0], reset_env=False,
        policy_instruction="pick up the block",
        step_callback=lambda t, obs, *args: callbacks.append(obs.instruction),
    )
    assert result.success
    assert seen == [("pick up the block", steering)] * 2
    assert callbacks == [original] * 2
    assert all(obs.instruction == original for obs in observations)


def test_distillation_receives_only_instruction_and_prompt():
    requests = []
    def generate(prompt, stop):
        requests.append(prompt)
        return '{"instruction": "pick up the block."}'
    distiller = LanguageDistiller(SimpleNamespace(generate=generate), prompt="Generic rules")
    assert distiller.distill("lift the blue object") == "pick up the block"
    assert requests == ['Generic rules\n# Query:\n{"instruction": "lift the blue object"}']


@pytest.mark.parametrize("response", ['{}', '{"instruction": ""}', '{"instruction": 3}',
    '{"instruction": "lift; push"}', '{"instruction":"lift","task":"secret"}', 'not json'])
def test_invalid_distillation_fails_instead_of_falling_back(response):
    distiller = LanguageDistiller(SimpleNamespace(generate=lambda *a, **kw: response))
    with pytest.raises(ValueError, match="Invalid distillation"):
        distiller.distill("input")


def test_backend_custom_prompt_is_cached_separately_and_does_not_request_code(tmp_path):
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"instruction":"push the block"}'))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    cache = DiskCache(str(tmp_path))
    for system in ["JSON only", "JSON only", "Other rules"]:
        backend = LLMBackend("openai", "gpt-5.4-mini", 0, 512, cache, system_prompt=system)
        backend._client = client
        backend.generate("Rewrite\n# Query:\npush it", stop=[])
    assert len(calls) == 2
    assert calls[0]["messages"][0]["content"] == "JSON only"
    assert "context code" not in calls[0]["messages"][1]["content"]


def test_resume_rejects_changed_checkpoint_or_prompt():
    previous = {"checkpoint_sha256": "abc", "prompt": "v1"}
    validate_resume(previous, dict(previous), "results.json")
    for key in previous:
        with pytest.raises(ValueError, match="Cannot resume"):
            validate_resume(previous, {**previous, key: "different"}, "results.json")


@pytest.mark.parametrize("distilled,steered", [(False, False), (True, False), (False, True), (True, True)])
def test_evaluation_wires_original_steering_and_policy_text(tmp_path, monkeypatch, distilled, steered):
    from omegaconf import OmegaConf
    from scripts import run_evaluation as evaluation

    original = "pick up the red block from the table"
    policy_inputs, steering_inputs = [], []
    def obs():
        return Observation({}, np.zeros(1), np.zeros(7), original)
    class Env:
        _provide_pcd_images = True
        _current_step = 0
        task_description = original
        _task_name = "lift_red_block_table"
        def set_task(self, task):
            self._task_name = task
        def reset(self, **kwargs):
            return obs()
        def get_scene_state(self):
            return {"robot_obs": np.zeros(1), "scene_obs": np.zeros(1)}
        def step(self, action):
            self._current_step += 1
            return obs(), 0, True, {"success": True}
        def close(self):
            pass
    class Policy:
        def reset(self):
            pass
        def forward(self, observation, steering=None):
            policy_inputs.append(observation.instruction)
            return None
    class Steering:
        _value_map = object()
        def setup_episode(self, task, instruction, **kwargs):
            steering_inputs.append(instruction)
    env = Env()
    monkeypatch.setattr(evaluation, "instantiate_env", lambda cfg: env)
    monkeypatch.setattr(evaluation, "instantiate_policy", lambda cfg: Policy())
    monkeypatch.setattr(evaluation, "instantiate_steering", lambda cfg: Steering() if steered else None)
    backend = SimpleNamespace(generate=lambda *args, **kw: '{"instruction":"pick up the block"}')
    monkeypatch.setattr(evaluation.LanguageDistiller, "from_config", lambda cfg: LanguageDistiller(backend))
    artifact = tmp_path / "artifact"
    artifact.write_text("test input")
    cfg = OmegaConf.create({
        "env": {"instruction_level": "specific", "lang_ann_path": str(artifact)},
        "policy": {"ckpt_path": str(artifact)},
        "steering": {"name": "voxposer" if steered else "none"},
    })
    eval_cfg = {"condition_name": "test", "tasks": [env._task_name]}
    if distilled:
        eval_cfg["distillation"] = {"model": "fake"}
    result = evaluation.run_condition(
        eval_cfg, cfg, {env._task_name: [(np.zeros(1), np.zeros(1))]},
        {env._task_name: [123]}, 1, 360, 42, tmp_path / "result.json",
    )
    expected = "pick up the block" if distilled else original
    assert policy_inputs == [expected]
    assert steering_inputs == ([original] if steered else [])
    task = result["tasks"][env._task_name]
    assert task["instruction"] == original
    assert task["policy_instruction"] == expected
    assert task["episodes"][0]["success"]
    assert task["episodes"][0]["source_episode_id"] == 123
    assert "mode" not in task["episodes"][0]
    assert result["run_metadata"]["instruction_level"] == "specific"


def test_missing_steering_map_aborts_instead_of_running_unsteered():
    from scripts.run_evaluation import setup_voxposer_episode

    env = SimpleNamespace(
        task_description='lift the blue block lying in the drawer',
        _task_name='lift_blue_block_drawer',
        get_scene_state=lambda: {'robot_obs': np.zeros(15), 'scene_obs': np.zeros(24)},
    )
    steering = SimpleNamespace(_value_map=None, setup_episode=lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match='Aborting to avoid recording an unsteered rollout'):
        setup_voxposer_episode(steering, env)
