from pathlib import Path
from types import SimpleNamespace

import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest

from core.types import Observation
from pi05_guidance.payload import export_guidance, horizon_ramp, relative_euler, rotation_matrix_from_6d
from utils.evaluation_config import load_evaluation_config

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['pi05_overloaded_base', 'pi05_overloaded_distilled', 'pi05_overloaded_base_steered', 'pi05_overloaded_distilled_steered']


def test_matrix_matches_four_original_conditions():
    original = [load_evaluation_config(ROOT/'conf', name.replace('pi05_', '')) for name in NAMES]
    new = [load_evaluation_config(ROOT/'conf', name) for name in NAMES]
    for old, current in zip(original, new):
        assert old['tasks'] == current['tasks'] and len(current['tasks']) == 34
        assert old['steering'] == current['steering']
        assert old.get('distillation') == current.get('distillation')
        assert current['policy_config'] == 'pi05_guided'
    assert all(c['policy'] == new[0]['policy'] for c in new)
    assert new[2]['steering_overrides'] == new[3]['steering_overrides']
    with initialize_config_dir(config_dir=str(ROOT/'conf'), version_base=None):
        cfg = compose(config_name='config', overrides=['policy=pi05_guided', 'steering=voxposer', 'steering.device=cpu'])
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert resolved['steering']['horizon'] == 10
    assert resolved['steering']['gripper_loc_bounds'] is None
    # pi0.5 executes 5 of its 10 predicted actions per call
    assert resolved['policy']['replan_steps'] == 5 and resolved['steering']['steps_per_policy_call'] == 5


def test_existing_matrix_launcher_selects_pi05_for_every_axis(tmp_path):
    from scripts.run_overloaded_evaluation import build_commands
    args = SimpleNamespace(backbone='pi05', phase='matrix', axes=['P0', 'P1', 'P2', 'P3', 'P4'],
        conditions=NAMES, dataset_path=tmp_path, output_dir=tmp_path, num_episodes=30, seed=42,
        max_steps=360, trials_per_task=30, tasks=None, p4_valid_indices=None, dry_run=False,
        override=[])
    commands = build_commands(args)
    assert len(commands) == 5
    for command in commands:
        assert all(name in command for name in NAMES)
        assert command[command.index('--target-trials-per-task')+1] == '30'


def test_axis_workers_use_disjoint_outputs_and_preserve_commands(tmp_path, monkeypatch):
    from scripts import run_overloaded_evaluation as launcher
    import threading
    barrier = threading.Barrier(2)
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        barrier.wait(timeout=5)
    monkeypatch.setattr(launcher.subprocess, 'run', run)
    commands = [['python', 'evaluate.py', '--output-dir', str(tmp_path/axis)] for axis in ['P0', 'P1']]
    launcher.execute_commands(commands, env={}, workers=2)
    assert sorted(calls) == sorted(commands)
    assert (tmp_path/'P0/worker.log').exists() and (tmp_path/'P1/worker.log').exists()


def test_shared_cache_replacement_never_exposes_partial_pickle(tmp_path):
    from voxposer.llm_cache import DiskCache
    from concurrent.futures import ThreadPoolExecutor
    import pickle
    cache = DiskCache(str(tmp_path))
    key = {'prompt': 'same shared request'}
    cache[key] = 'initial'
    filename = tmp_path/cache._generate_filename(key)
    def writer():
        for i in range(20):
            cache[key] = str(i)*10000
    def reader():
        for _ in range(200):
            with filename.open('rb') as f:
                saved_key, value = pickle.load(f)
            assert saved_key == key and isinstance(value, str)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda f:f(), [writer, reader]))


def test_controller_target_and_scale_match_calvin_accumulation():
    from calvin_env.robot.robot import Robot
    anchor = np.array([.1, -.2, .5, 3., .1, .3])
    robot = SimpleNamespace(use_target_pose=True, target_pos=anchor[:3].copy(), target_orn=anchor[3:].copy(),
                            max_rel_pos=.02, max_rel_orn=.05, magic_scaling_factor_pos=1., magic_scaling_factor_orn=1.)
    actions = np.random.default_rng(42).uniform(-1, 1, (10, 7))
    actual = []
    for action in actions:
        position, rotation, _ = Robot.relative_to_absolute(robot, action)
        actual.append(np.concatenate([position, rotation]))
    expected = anchor+np.cumsum(actions[:, :6]*np.array([.02]*3+[.05]*3), axis=0)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def _steering(stage, **cfg):
    base = {'world_step_bounds': [[-0.3021, -0.2569, -0.2156], [0.3613, 0.3377, 0.1837]], 'reference_horizon': 20}
    return SimpleNamespace(cfg={**base, **cfg}, guidance_strength=1., _stage_manager=SimpleNamespace(current=lambda: stage))


def _stage(**fields):
    value_map = SimpleNamespace(_cost_map=np.zeros((3, 3, 3)), workspace_bounds_min=-np.ones(3), workspace_bounds_max=np.ones(3))
    return SimpleNamespace(**{**dict(value_map=value_map, stage_target_world=None, steps_in_stage=0, rotation_target_6d=None), **fields})


def _observation(anchor=(.1, .2, .3, .4, .5, .6)):
    return Observation({}, np.zeros(15), np.zeros(7), 'test', action_context={
        'protocol': 'calvin_target_relative_v1', 'use_target_pose': True, 'anchor': np.asarray(anchor), 'scale': np.array([.02]*3+[.05]*3)})


def test_payload_uses_controller_anchor_and_world_step():
    obs = _observation()
    payload = export_guidance(_steering(_stage()), obs, 10)
    np.testing.assert_allclose(payload['anchor'], obs.action_context['anchor'])
    np.testing.assert_allclose(payload['controller_scale'], [.02]*3+[.05]*3)
    # strength 1 in world units: the squared half-range of the 3DDA position normalisation
    np.testing.assert_allclose(payload['position_gain'], [0.3317**2, 0.2973**2, 0.19965**2], rtol=1e-4)
    assert payload['rotation_strength'] == 0 and not payload['rotation_target_euler'].any()
    obs.action_context = None
    with pytest.raises(ValueError, match='controller target'):
        export_guidance(_steering(_stage()), obs, 10)


def test_rotation_target_is_exported_as_the_turn_itself():
    from voxposer.calvin_interface import CalvinLMPInterface
    for angle in (90, -90):
        turn = CalvinLMPInterface.rotation_about_axis(None, 'z', angle)      # what a plan emits
        six = np.concatenate([turn[:, 0], turn[:, 1]])                        # ortho6d: first two columns
        np.testing.assert_allclose(rotation_matrix_from_6d(six), turn, atol=1e-6)
        payload = export_guidance(_steering(_stage(rotation_target_6d=six), guidance_strength_rot=2.0), _observation(), 10)
        # a relative Euler change about z, never an absolute orientation
        np.testing.assert_allclose(payload['rotation_target_euler'], [0, 0, np.radians(angle)], atol=1e-6)
        assert payload['rotation_strength'] == 2.0


def test_relative_euler_inverts_the_diffuser_actor_rotation_encoding():
    import torch
    from training.policies.diffuser_actor.preprocessing.pytorch3d_transforms import euler_angles_to_matrix
    angles = np.random.default_rng(0).uniform(-1.2, 1.2, (50, 3)).astype(np.float32)
    matrices = euler_angles_to_matrix(torch.from_numpy(angles), 'XYZ').numpy()
    np.testing.assert_allclose([relative_euler(m) for m in matrices], angles, atol=1e-5)


def test_rotation_ramp_is_a_function_of_control_steps_ahead():
    # 3DDA (20 waypoints) keeps its ramp; pi0.5 (10 waypoints) reads the first half of the same ramp
    reference = 0.6*(np.arange(20)/19)**2
    np.testing.assert_allclose(horizon_ramp(20, 20, 0.0, 0.6), reference, atol=1e-7)
    np.testing.assert_allclose(horizon_ramp(10, 20, 0.0, 0.6), reference[:10], atol=1e-7)
    assert horizon_ramp(30, 20, 0.0, 0.6).max() == pytest.approx(0.6)


def test_distance_decay_scales_the_position_strength():
    stage = _stage(stage_target_world=np.array([0., 0., 0.]))
    decay = dict(use_distance_scaling=True, distance_full=.12, distance_near=.04, distance_floor=.1)
    def strength(distance):
        obs = Observation({}, np.array([distance, 0, 0]+[0]*12, dtype=np.float32), np.zeros(7), 'test',
                          action_context=_observation().action_context)
        return float(export_guidance(_steering(stage, **decay), obs, 10)['position_strength'])

    assert strength(.30) == pytest.approx(1.0) and strength(.02) == pytest.approx(0.1)
    assert strength(.08) == pytest.approx(0.55)


def test_flow_settings_reach_the_payload():
    payload = export_guidance(_steering(_stage()), _observation(), 10)
    assert payload['position_strength'] == 1.0 and payload['time_floor'] == 0 and payload['rotation_time_floor'] == 0
    payload = export_guidance(_steering(_stage(), flow_gain=4.0, flow_time_floor=0.5, flow_rotation_time_floor=0.3), _observation(), 10)
    assert payload['position_strength'] == 4.0
    assert payload['time_floor'] == pytest.approx(0.5) and payload['rotation_time_floor'] == pytest.approx(0.3)


def test_rotation_can_be_limited_to_a_held_object():
    from voxposer.calvin_interface import CalvinLMPInterface
    turn = CalvinLMPInterface.rotation_about_axis(None, 'z', 90)
    stage = _stage(rotation_target_6d=np.concatenate([turn[:, 0], turn[:, 1]]))

    def strength(width, **cfg):
        obs = Observation({}, np.zeros(15), np.array([0, 0, 0, 0, 0, 0, width]), 'test', action_context=_observation().action_context)
        return float(export_guidance(_steering(stage, guidance_strength_rot=2.0, **cfg), obs, 10)['rotation_strength'])

    assert strength(0.08) == 2.0                                            # default: always
    holding = dict(flow_rotation_when='holding', grasp_min_width=0.02, grasp_max_width=0.07)
    assert strength(0.08, **holding) == 0.0                                 # open
    assert strength(0.00, **holding) == 0.0                                 # closed on nothing
    assert strength(0.045, **holding) == 2.0                                # closed on a block


def test_override_is_off_by_default_and_released_near_the_stage_target():
    payload = export_guidance(_steering(_stage(stage_target_world=np.zeros(3))), _observation(), 10)
    assert payload['override_strength'] == 0 and payload['bound_result'] == 0
    cfg = dict(flow_override_gain=8.0, flow_override_near_cm=4.0, flow_override_full_cm=12.0,
               flow_override_progress_cm=2.0, flow_step_bound='result')

    def exported(distance):
        obs = Observation({}, np.array([distance, 0, 0]+[0]*12, dtype=np.float32), np.zeros(7), 'test',
                          action_context=_observation().action_context)
        return export_guidance(_steering(_stage(stage_target_world=np.zeros(3)), **cfg), obs, 10)

    assert float(exported(.30)['override_strength']) == pytest.approx(8.0)     # far: full
    assert float(exported(.08)['override_strength']) == pytest.approx(4.0)     # halfway through the release
    assert float(exported(.02)['override_strength']) == 0.0                    # near the target: the policy's own
    assert float(exported(.30)['override_progress']) == pytest.approx(0.02) and exported(.30)['bound_result'] == 1


def test_rotation_gain_scales_the_rotation_term():
    from voxposer.calvin_interface import CalvinLMPInterface
    turn = CalvinLMPInterface.rotation_about_axis(None, 'z', -90)
    stage = _stage(rotation_target_6d=np.concatenate([turn[:, 0], turn[:, 1]]))
    payload = export_guidance(_steering(stage, guidance_strength_rot=2.0, flow_rotation_gain=25.0), _observation(), 10)
    assert float(payload['rotation_strength']) == pytest.approx(50.0)


def test_override_can_be_limited_to_the_approach_to_a_grasp():
    cfg = dict(flow_override_gain=8.0, flow_override_when='approach', grasp_min_width=0.02, grasp_max_width=0.07)

    def strength(primitive, width):
        obs = Observation({}, np.array([.3, 0, 0]+[0]*12, dtype=np.float32), np.array([0, 0, 0, 0, 0, 0, width]), 'test',
                          action_context=_observation().action_context)
        stage = _stage(stage_target_world=np.zeros(3), primitive=primitive)
        return float(export_guidance(_steering(stage, **cfg), obs, 10)['override_strength'])

    assert strength('grasp', 0.08) == pytest.approx(8.0)     # approaching with an open gripper
    assert strength('grasp', 0.045) == 0.0                   # closed on an object
    for primitive in ('push', 'place', 'rotate', 'pull'):
        assert strength(primitive, 0.08) == 0.0
