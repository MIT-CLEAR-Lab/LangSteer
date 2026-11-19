"""Capture a real CALVIN observation and controller state for serving checks."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from hydra import compose, initialize_config_dir

from scripts.run_evaluation import presample_starting_conditions
from scripts.run_experiment import instantiate_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    args = parser.parse_args()
    task = 'open_drawer'
    states, ids = presample_starting_conditions(args.dataset, 'validation', [task], 1, 42)
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1]/'conf'), version_base=None):
        cfg = compose(config_name='config', overrides=['policy=pi05_guided', 'env.provide_pcd_images=true'])
    cfg.env.dataset_path = str(args.dataset)
    env = instantiate_env(cfg)
    try:
        env.set_task(task)
        robot_obs, scene_obs = states[task][0]
        obs = env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        from policies.pi05_guided import GuidedPi05Policy
        context = obs.action_context
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.output, image=GuidedPi05Policy._to_uint8_hwc(obs.rgb['static']),
                 wrist_image=GuidedPi05Policy._to_uint8_hwc(obs.rgb['gripper_native']), state=obs.proprio,
                 prompt=np.asarray(obs.instruction), anchor=context['anchor'], controller_scale=context['scale'],
                 source_episode_id=np.asarray(ids[task][0]))
        print('Saved real CALVIN validation fixture:', args.output, 'controller scale:', context['scale'])
    finally:
        env.close()


if __name__ == '__main__':
    main()
