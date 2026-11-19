"""The prompts and steering configuration the steered conditions rely on."""
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from utils.evaluation_config import load_evaluation_config

ROOT = Path(__file__).resolve().parents[1]
STEERED = ['overloaded_base_steered', 'overloaded_distilled_steered',
           'pi05_overloaded_base_steered', 'pi05_overloaded_distilled_steered']

def resolved(condition, extra=()):
    eval_cfg = load_evaluation_config(ROOT/'conf', condition)
    overrides = [f"policy={eval_cfg['policy_config']}", f"steering={eval_cfg['steering']}"]
    overrides += [f'policy.{k}={v}' for k, v in eval_cfg['policy'].items()]
    overrides += [f'steering.{k}={v}' for k, v in eval_cfg.get('steering_overrides', {}).items()]
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(ROOT/'conf'), version_base=None):
        cfg = compose(config_name='config', overrides=overrides+list(extra))
    return eval_cfg, OmegaConf.to_container(cfg, resolve=True)


def test_prompts_are_the_frozen_set():
    # The LLM cache key is prompt text plus model, so editing a prompt re-buys every
    # call and changes the plans the reference was run with.
    from tests.frozen_prompts import PROMPT_SHA256 as frozen
    directory = ROOT/'voxposer/prompts/calvin'
    assert {p.name for p in directory.glob('*.txt')} == set(frozen)
    for name, digest in frozen.items():
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == digest, (
            f'{name} differs from the frozen prompt set: cached LLM plans no longer apply')


@pytest.mark.parametrize('condition', STEERED)
def test_steered_conditions_share_one_configuration(condition):
    _, cfg = resolved(condition)
    steering = cfg['steering']
    assert steering['llm_model'] == 'gpt-5.4'
    assert steering['guidance_strength'] == 1 and steering['guidance_strength_rot'] == 2.0
    assert 'scene_grounding' not in steering                     # plans are text-only
    assert steering['loop_back_on_last_stage'] is True
    assert steering['use_distance_scaling'] is True and steering['rot_hemisphere_fix'] is True
    assert 'guidance_mode' not in steering and 'rot_guidance_method' not in steering   # epsilon + SLERP only


def test_pi05_conditions_use_the_flow_settings():
    # docs/guides/pi05_steering.md explains each setting.
    for condition in ('pi05_overloaded_base_steered', 'pi05_overloaded_distilled_steered'):
        _, cfg = resolved(condition)
        steering = cfg['steering']
        assert steering['flow_gain'] == 4.0 and steering['flow_time_floor'] == 0.5
        assert steering['flow_rotation_time_floor'] == 0.0
        assert steering['flow_rotation_when'] == 'holding' and steering['flow_rotation_yield_deg'] == 0.0
        assert steering['flow_rotation_gain'] == 25.0 and steering['flow_step_bound'] == 'result'
        assert steering['flow_override_gain'] == 8.0 and steering['flow_override_when'] == 'approach'
        assert steering['flow_override_reach_cm'] == 5.0 and steering['flow_override_progress_cm'] == 2.0
        assert steering['flow_override_near_cm'] == 4.0 and steering['flow_override_full_cm'] == 12.0
        # the pull fades near the stage target exactly as it does for 3DDA
        assert (steering['distance_full'], steering['distance_near'], steering['distance_floor']) == (0.12, 0.04, 0.1)


def test_time_based_settings_mean_the_same_duration_for_both_policies():
    from steering.stage_manager import StageManager
    for condition, steps_per_call, dwell_calls in [('overloaded_base_steered', 20, 4), ('pi05_overloaded_base_steered', 5, 16)]:
        _, cfg = resolved(condition)
        assert cfg['steering']['steps_per_policy_call'] == steps_per_call
        manager = StageManager(cfg['steering'], device='cpu', map_size=100,
                               workspace_min=np.zeros(3), workspace_max=np.ones(3))
        assert manager._loop_back_dwell_steps == dwell_calls       # 80 control steps either way
        assert cfg['steering']['reference_horizon'] == 20


def test_world_step_is_the_diffuser_actor_normalisation():
    # The DDPM branch gets its world-space step from policy.gripper_loc_bounds; the flow
    # branch reads steering.world_step_bounds. They must be the same numbers.
    _, dda = resolved('overloaded_base_steered')
    _, pi05 = resolved('pi05_overloaded_base_steered')
    assert dda['steering']['world_step_bounds'] == dda['policy']['gripper_loc_bounds']
    assert pi05['steering']['world_step_bounds'] == dda['policy']['gripper_loc_bounds']
    assert dda['steering']['gripper_loc_bounds'] == dda['policy']['gripper_loc_bounds']


def test_diffuser_actor_runs_relative_with_the_overloaded_checkpoint_settings():
    _, cfg = resolved('overloaded_base')
    policy = cfg['policy']
    assert policy['relative'] is True and policy['pred_horizon'] == 20 and policy['executed_steps'] == 20
    assert policy['text_max_length'] == 53 and policy['strict_checkpoint'] is True
    assert policy['ckpt_path'].endswith('last.pth')     # best.pth is step 5,000: its selector minimised accuracy


def test_rotation_target_of_a_plan_is_the_turn_itself():
    # 3DDA predicts rotations relative to the current gripper pose, so a stage's rotation
    # target must be the turn (left is +90 about z), never an absolute orientation.
    from steering.stage_spec import normalize_rot_target
    from voxposer.calvin_interface import CalvinLMPInterface
    left = CalvinLMPInterface.rotation_about_axis(None, 'z', 90)
    np.testing.assert_allclose(left @ np.array([1., 0., 0.]), [0., 1., 0.], atol=1e-6)     # counter-clockwise
    six = normalize_rot_target(left, idx=0)
    np.testing.assert_allclose(six, np.concatenate([left[:, 0], left[:, 1]]), atol=1e-6)
    prompt = (ROOT/'voxposer/prompts/calvin/composer_prompt.txt').read_text()
    assert "rot_target = rotation_about_axis('z', 90)" in prompt and "rotation_about_axis('z', -90)" in prompt


def test_evaluation_aborts_when_no_steering_map_is_produced():
    from scripts.run_evaluation import setup_voxposer_episode
    state = {'robot_obs': np.zeros(15), 'scene_obs': np.zeros(24)}
    env = SimpleNamespace(get_scene_state=lambda: state, task_description='do something', _task_name='open_drawer')
    steering = SimpleNamespace(setup_episode=lambda *a, **k: None, _value_map=None)
    with pytest.raises(RuntimeError, match='no steering map'):
        setup_voxposer_episode(steering, env, None)


def test_p4_uses_the_validated_scene_subset():
    import json
    data = json.loads((ROOT/'conf/evaluation/p4_valid_indices.json').read_text())
    assert data['seed'] == 42 and data['num_episodes'] == 30
    tasks = load_evaluation_config(ROOT/'conf', 'overloaded_base')['tasks']
    assert set(data['valid_indices']) == set(tasks) and len(tasks) == 34
    assert all(0 < len(v) <= 30 and all(0 <= i < 30 for i in v) for v in data['valid_indices'].values())


def test_every_condition_runs_the_same_34_tasks():
    tasks = load_evaluation_config(ROOT/'conf', 'overloaded_base')['tasks']
    for condition in STEERED + ['overloaded_distilled', 'pi05_overloaded_base', 'pi05_overloaded_distilled']:
        assert load_evaluation_config(ROOT/'conf', condition)['tasks'] == tasks
