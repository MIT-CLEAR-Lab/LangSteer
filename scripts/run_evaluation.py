"""Multi-task CALVIN benchmark evaluation.

Runs evaluation conditions across tasks, recording per-episode data
(success, steps, reward). Results JSON is updated after every episode
for crash safety. Supports resuming interrupted runs and extending
with additional episodes.

Usage (scripts/run_overloaded_evaluation.py is the usual way to launch this):
    python scripts/run_evaluation.py --evaluation overloaded_base overloaded_base_steered --output-dir <dir>

    # Resume an interrupted run (picks up where it left off): repeat the command
    # with the same --output-dir.
"""

import argparse
import json
import logging
import os
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_experiment import instantiate_env, instantiate_policy, instantiate_steering
from utils.rollout import EpisodeRunner
from voxposer.calvin_interface import ObjectResolutionError
from voxposer.lmp import VocabValidationError
from utils.evaluation_config import (
    check_session_provenance,
    file_sha256,
    load_evaluation_config,
    session_provenance,
    validate_resume,
)
from utils.language_distillation import LanguageDistiller, PROMPT_PATH

logger = logging.getLogger(__name__)


class _HardFailResult:
    """Stand-in for `EpisodeResult` when an episode hard-fails before/during
    `run_episode`. Used by the runner's per-episode catch to record a failed
    episode without taking down the whole run.
    """

    success = False
    episode_length = 0
    episode_reward = 0.0


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-task CALVIN evaluation")
    parser.add_argument(
        "--evaluation", nargs="+", required=True,
        help="Evaluation condition config(s) from conf/evaluation/ (e.g. overloaded_base overloaded_base_steered)",
    )
    parser.add_argument(
        "--num-episodes", type=int, default=25,
        help="Starting-condition pool size per task. For P4 this must match the "
             "pool size used when labeling (p4_valid_indices.json's num_episodes), "
             "because the saved valid indices map into that exact presample; the "
             "filter then narrows to the few valid scenes per task.",
    )
    parser.add_argument(
        "--tries-per-episode", type=int, default=1,
        help="Re-roll each starting condition this many times with different "
             "model seeds (same scene, different diffusion noise). Total rollouts "
             "per task = (post-filter conditions) * tries_per_episode. Ordering: "
             "all tries of cond 0, then all tries of cond 1, ...",
    )
    parser.add_argument(
        "--target-trials-per-task", type=int, default=None,
        help="Alternative to --tries-per-episode: keep cycling through the valid "
             "starting conditions, reseeding the model on each pass, until each "
             "task has exactly this many rollouts. Ordering: pass 0 covers cond "
             "0..N-1, then pass 1, etc. — every scene gets equal coverage even if "
             "the loop is interrupted. Mutually exclusive with --tries-per-episode.",
    )
    parser.add_argument("--max-steps", type=int, default=360, help="Max env steps per episode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for starting conditions")
    parser.add_argument("--dataset-path", type=Path, help="CALVIN dataset root (containing validation/)")
    parser.add_argument("--p4-valid-indices", type=Path, default=None)
    parser.add_argument("--override", action="append", default=[], help="Hydra override; repeat as needed")
    parser.add_argument("--dry-run", action="store_true", help="Resolve configs and verify inputs without loading models")
    parser.add_argument("--prepare-language", action="store_true", help="Cache and export conditioning text without loading the environment or policy")
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory. If it contains existing results, the run resumes/extends.",
    )
    parser.add_argument(
        "--tasks", type=str, default=None,
        help="Path to a JSON file with a 'tasks' list. Overrides (and reorders) "
             "the task list from each --evaluation config. Tasks not present in "
             "an eval condition's own task list are silently skipped for that "
             "condition. Useful for running a curated subset (e.g. skip rotations).",
    )
    parser.add_argument(
        "--perturbation-axis", type=str, default=None, choices=["P1", "P2", "P3", "P4"],
        help="Use perturbed instructions from perturbed_language_annotations.json at this "
             "axis (P1..P4). P4 additionally filters starting conditions to indices listed "
             "in conf/evaluation/p4_valid_indices.json.",
    )
    parser.add_argument(
        "--instruction-level", type=str, default="specific",
        choices=["specific", "ambiguous"],
        help="Instruction specificity. 'ambiguous' conditions on the low-specificity "
             "L0 template of each task's ambiguity class ('pick up the block'), resets "
             "to the class eval condition, and scores success against the class's "
             "feasible_modes. Tasks with no ambiguous tier are skipped.",
    )
    return parser.parse_args()


def deterministic_hash(s: str) -> int:
    """Deterministic 32-bit hash from a string."""
    import hashlib
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % (2**31)


def configure_determinism():
    """Opt-in (LANGSTEER_DETERMINISTIC=1): restrict cuDNN to deterministic algorithms."""
    if os.environ.get("LANGSTEER_DETERMINISTIC") == "1":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info("cuDNN restricted to deterministic algorithms")


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Starting-condition sampling (deterministic, extensible)
# ---------------------------------------------------------------------------

def presample_starting_conditions(
    dataset_path: Path,
    split: str,
    task_list: List[str],
    num_episodes: int,
    seed: int,
) -> Tuple[Dict[str, List[Tuple[np.ndarray, np.ndarray]]], Dict[str, List[int]]]:
    """Pre-sample N (robot_obs, scene_obs) pairs per task from dataset episodes.

    The RNG is seeded once, then tasks are sampled in sorted order so that
    requesting more episodes later extends the sequence deterministically
    (the first K samples are always the same regardless of N >= K).
    """
    ann_path = dataset_path / split / "lang_annotations" / "auto_lang_ann.npy"
    ann = np.load(str(ann_path), allow_pickle=True).item()
    tasks = ann["language"]["task"]
    start_end = ann["info"]["indx"]

    task_episode_ids: Dict[str, List[int]] = {}
    for i, task_name in enumerate(tasks):
        task_episode_ids.setdefault(task_name, []).append(start_end[i][0])

    starting_conditions: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
    sampled_episode_ids: Dict[str, List[int]] = {}

    for task_name in sorted(task_list):
        ep_ids = task_episode_ids.get(task_name, [])
        if not ep_ids:
            logger.warning(f"No dataset episodes for task '{task_name}', skipping")
            starting_conditions[task_name] = []
            sampled_episode_ids[task_name] = []
            continue

        # Per-task RNG seeded from (global seed, task_name) — independent of task order
        rng = random.Random(seed + deterministic_hash(task_name))
        sampled_ids = [rng.choice(ep_ids) for _ in range(num_episodes)]
        sampled_episode_ids[task_name] = sampled_ids

        conditions = []
        for ep_id in sampled_ids:
            ep_path = dataset_path / split / f"episode_{ep_id:07d}.npz"
            data = np.load(str(ep_path))
            conditions.append((
                data["robot_obs"].astype(np.float32),
                data["scene_obs"].astype(np.float32),
            ))
        starting_conditions[task_name] = conditions

    logger.info(f"Pre-sampled {num_episodes} starting conditions for {len(task_list)} tasks")
    return starting_conditions, sampled_episode_ids


# ---------------------------------------------------------------------------
# Results file I/O (incremental, resumable)
# ---------------------------------------------------------------------------

def load_results(results_path: Path) -> Dict:
    """Load existing results JSON, or return empty structure."""
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    return {}


def save_results(results_path: Path, results: Dict):
    """Atomically write results JSON (write tmp then rename)."""
    tmp_path = results_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(results, f, indent=2)
    tmp_path.rename(results_path)


def get_completed_episodes(results: Dict, task_name: str) -> int:
    """Return how many episodes are already recorded for a task."""
    return len(results.get("tasks", {}).get(task_name, {}).get("episodes", []))


# ---------------------------------------------------------------------------
# Steering helpers
# ---------------------------------------------------------------------------

def wire_steering(steering, policy, cfg):
    """Connect the steering module to the policy's sampler."""
    if hasattr(policy, "bind_steering"):
        policy.bind_steering(steering)
        logger.info("Using policy-native steering instead of DDPM schedulers")
        return
    if hasattr(steering, "set_position_scheduler"):
        steering.set_position_scheduler(policy._model.position_noise_scheduler)
        logger.info("Set position scheduler for steering")
    if hasattr(steering, "set_rotation_scheduler"):
        steering.set_rotation_scheduler(policy._model.rotation_noise_scheduler)
        logger.info("Set rotation scheduler for steering")


def setup_voxposer_episode(steering, env, hydra_cfg=None):
    """Setup VoxPoser steering for a new episode (after env.reset)."""
    state = env.get_scene_state()
    vp_instruction = env.task_description

    steering.setup_episode(
        env._task_name,
        instruction=vp_instruction,
        robot_obs=state["robot_obs"],
        scene_obs=state["scene_obs"],
        fixture_positions=state.get("fixture_positions"),
        block_aabbs=state.get("block_aabbs"),
    )
    if steering._value_map is not None:
        logger.info(f"VoxPoser value maps generated for: '{vp_instruction}'")
    else:
        raise RuntimeError(
            f"VoxPoser produced no steering map for {vp_instruction!r}. "
            "Aborting to avoid recording an unsteered rollout as a steered result. "
            "Check the preceding LLM/API errors."
        )


def make_step_callback(steering, env):
    """Callback run after every policy call (one executed action chunk).

    Advances the steering clock, refreshes the active cost map from the current
    scene and checks stage transitions. `obs` is the observation the policy
    acted on, as passed by `EpisodeRunner`.
    """
    def step_callback(timestep, obs, action, reward, done, info):
        if steering is not None and hasattr(steering, "increment_step"):
            steering.increment_step()
        if steering is not None and hasattr(steering, "refresh_costmap"):
            state = env.get_scene_state()
            steering.refresh_costmap(
                state["robot_obs"], state["scene_obs"],
                fixture_positions=state.get("fixture_positions"),
                block_aabbs=state.get("block_aabbs"),
            )
        if steering is not None and hasattr(steering, "check_stage_transition"):
            steering.check_stage_transition(
                obs.ee_pose[:3], float(obs.ee_pose[6])
            )
    return step_callback


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------

def run_condition(
    eval_cfg: Dict,
    hydra_cfg,
    starting_conditions: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
    sampled_episode_ids: Dict[str, List[int]],
    num_episodes: int,
    max_steps: int,
    base_seed: int,
    results_path: Path,
    perturbation_axis: str = None,
    p4_valid_indices: Dict[str, List[int]] = None,
    tries_per_episode: int = 1,
    target_trials_per_task: Optional[int] = None,
    config_name: Optional[str] = None,
    process_plan: Optional[List[Tuple[str, List[str]]]] = None,
):
    """Run evaluation for a single condition, saving after every episode.

    `process_plan` is the ordered (condition, tasks) list this process runs; it is
    recorded with the session provenance (see utils/evaluation_config.py).
    """
    condition_name = eval_cfg["condition_name"]
    task_list = eval_cfg["tasks"]

    # Load existing results (for resume)
    results = load_results(results_path)
    run_metadata = {
        "seed": base_seed, "max_steps": max_steps,
        "perturbation_axis": perturbation_axis,
        "instruction_level": hydra_cfg.env.instruction_level,
        "config": OmegaConf.to_container(hydra_cfg, resolve=True),
        "checkpoint_sha256": file_sha256(hydra_cfg.policy.ckpt_path),
        "distillation": eval_cfg.get("distillation"),
        "distillation_prompt_sha256": file_sha256(PROMPT_PATH) if eval_cfg.get("distillation") else None,
        "annotation_sha256": file_sha256(hydra_cfg.env.lang_ann_path),
        "perturbations_sha256": file_sha256(hydra_cfg.env.perturbed_ann_path) if perturbation_axis else None,
        "p4_valid_indices": p4_valid_indices,
    }
    if results:
        if "run_metadata" not in results:
            raise ValueError(f"{results_path} has no provenance; use a new output directory")
        validate_resume(results["run_metadata"], run_metadata, results_path)
    if not results:
        results = {
            "condition_name": condition_name,
            "seed": base_seed,
            "max_steps": max_steps,
            "perturbation_axis": perturbation_axis,
            "tries_per_episode": tries_per_episode,
            "target_trials_per_task": target_trials_per_task,
            "tasks": {},
            "run_metadata": run_metadata,
        }
    else:
        # Resume must reuse the same loop mode so the (condition_idx,
        # attempt_idx) layout of the flat episode list stays consistent.
        prev_tries = results.get("tries_per_episode", 1)
        prev_target = results.get("target_trials_per_task")
        if prev_tries != tries_per_episode or prev_target != target_trials_per_task:
            logger.error(
                f"Existing results in {results_path} were produced with "
                f"tries_per_episode={prev_tries}, target_trials_per_task={prev_target}, "
                f"but this run requested tries_per_episode={tries_per_episode}, "
                f"target_trials_per_task={target_trials_per_task}. Resume would "
                f"shuffle the condition/attempt mapping. Start a new --output-dir."
            )
            sys.exit(1)

    # Session provenance: recorded, not part of the resume identity, except that
    # changed steering prompts refuse the resume.
    config_name = config_name or results_path.stem
    session = session_provenance(
        process_plan=process_plan or [(config_name, task_list)],
        condition=config_name,
        steered=hydra_cfg.steering.name != "none",
        completed_episodes={t: get_completed_episodes(results, t) for t in task_list},
    )
    changed = check_session_provenance(results.get("provenance", []), session, results_path)
    if changed:
        logger.warning(f"Resuming {results_path} with changed {', '.join(changed)} (recorded)")
    results.setdefault("provenance", []).append(session)
    save_results(results_path, results)

    logger.info(f"\n{'='*70}")
    logger.info(f"CONDITION: {condition_name}")
    logger.info(f"{'='*70}")

    # Instantiate components
    env = instantiate_env(hydra_cfg)
    policy = instantiate_policy(hydra_cfg)
    steering = instantiate_steering(hydra_cfg)

    if not env._provide_pcd_images:
        env._provide_pcd_images = True

    if steering is not None:
        wire_steering(steering, policy, hydra_cfg)

    runner = EpisodeRunner(
        env=env, policy=policy, steering=steering, max_steps=max_steps * 2,
    )
    distiller = (
        LanguageDistiller.from_config(eval_cfg["distillation"])
        if eval_cfg.get("distillation") else None
    )

    for task_name in task_list:
        env.set_task(task_name)
        env._max_steps = max_steps

        conditions = starting_conditions.get(task_name, [])
        ep_ids = sampled_episode_ids.get(task_name, [])

        # P4: filter to starts manually verified as unambiguous for this task.
        # Tasks without an entry in the valid-indices map are unambiguous and run as-is.
        if perturbation_axis == "P4" and p4_valid_indices and task_name in p4_valid_indices:
            valid = [i for i in p4_valid_indices[task_name] if 0 <= i < len(conditions)]
            conditions = [conditions[i] for i in valid]
            ep_ids = [ep_ids[i] for i in valid]
            logger.info(f"  [P4] Filtered {task_name} to {len(valid)} valid starting conditions")

        n_available = min(num_episodes, len(conditions))
        if target_trials_per_task is not None:
            # Cycling mode: pass 0 covers cond 0..N-1, pass 1 covers cond 0..N-1, ...
            total_attempts = target_trials_per_task
        else:
            total_attempts = n_available * tries_per_episode

        # Initialize task entry if missing
        if task_name not in results["tasks"]:
            results["tasks"][task_name] = {
                "instruction": env.task_description,
                "episodes": [],
            }

        completed = get_completed_episodes(results, task_name)

        if completed >= total_attempts:
            logger.info(f"\n  {task_name}: {completed}/{total_attempts} already done, skipping")
            continue

        if n_available == 0:
            logger.warning(f"  No starting conditions for {task_name}, skipping")
            continue

        policy_instruction = env.task_description
        if distiller is not None:
            policy_instruction = distiller.distill(env.task_description)
        recorded_text = results["tasks"][task_name].get("policy_instruction")
        if recorded_text is not None and recorded_text != policy_instruction:
            raise ValueError(f"Distillation changed for {task_name}; use the original cache or a new output directory")
        results["tasks"][task_name]["policy_instruction"] = policy_instruction
        logger.info("Policy conditioning: %r", policy_instruction)
        # Loading CLIP can consume RNG on the first call. Warm text embeddings
        # outside the seeded rollout so cached/fresh runs share diffusion noise.
        if hasattr(policy, "_get_instruction_embedding"):
            policy._get_instruction_embedding(policy_instruction)

        logger.info(f"\n  Task: {task_name} ({env.task_description})")
        if target_trials_per_task is not None:
            logger.info(
                f"    Cycling {n_available} conditions to hit {total_attempts} "
                f"trials; resuming from trial {completed + 1}/{total_attempts}"
            )
        elif tries_per_episode > 1:
            logger.info(
                f"    {n_available} conditions × {tries_per_episode} tries = "
                f"{total_attempts} rollouts; resuming from rollout "
                f"{completed + 1}/{total_attempts}"
            )
        else:
            logger.info(f"    Resuming from episode {completed + 1}/{total_attempts}")

        for flat_idx in range(completed, total_attempts):
            if target_trials_per_task is not None:
                # Cycling: (cond=0, att=0), (cond=1, att=0), ..., (cond=0, att=1), ...
                condition_idx = flat_idx % n_available
                attempt_idx = flat_idx // n_available
            else:
                # Sequential: (cond=0, att=0..T-1), (cond=1, att=0..T-1), ...
                condition_idx = flat_idx // tries_per_episode
                attempt_idx = flat_idx % tries_per_episode

            robot_obs, scene_obs = conditions[condition_idx]
            source_episode_id = (
                int(ep_ids[condition_idx]) if condition_idx < len(ep_ids) else None
            )

            # Reproducible diffusion noise. attempt 0 keeps the original
            # (no-tries) seed so single-try runs are bit-identical to before.
            ep_seed = base_seed + deterministic_hash(task_name) + condition_idx
            if attempt_idx > 0:
                ep_seed += attempt_idx * 1_000_000
            set_seed(ep_seed)

            obs = env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

            step_callback = make_step_callback(steering, env)

            # Catch hard-fail exceptions from steering (vocab linter exhaustion,
            # object-name resolution exhaustion) at the episode boundary so a
            # single broken task doesn't take down the whole run. The block
            # covers `setup_voxposer_episode` (composer may raise at stage
            # activation), the step callback (refresh_costmap re-evaluates),
            # and the rollout itself. Each hard fail is logged + counted as a
            # failed episode; the loop continues to the next ep.
            hard_fail_reason: Optional[str] = None
            try:
                # Setup steering — composer + stage 0 activation may raise.
                if steering is not None and hasattr(steering, "setup_episode"):
                    if hydra_cfg.steering.name == "voxposer":
                        setup_voxposer_episode(steering, env, hydra_cfg)

                # LLM/steering initialization must not alter the policy noise seed.
                set_seed(ep_seed)
                result = runner.run_episode(
                    initial_obs=obs, reset_env=False, reset_policy=True,
                    step_callback=step_callback,
                    policy_instruction=policy_instruction,
                )
            except VocabValidationError as e:
                logger.error(
                    f"    Rollout {flat_idx + 1}/{total_attempts} hard-failed: "
                    f"VocabValidationError (composer exhausted re-prompts). {e}"
                )
                result = _HardFailResult()
                hard_fail_reason = "vocab_validation_exhausted"
            except ObjectResolutionError as e:
                logger.error(
                    f"    Rollout {flat_idx + 1}/{total_attempts} hard-failed: "
                    f"ObjectResolutionError (composer emitted unresolvable "
                    f"object name). {e}"
                )
                result = _HardFailResult()
                hard_fail_reason = "object_resolution_exhausted"

            # Record episode data
            ep_record = {
                "episode_idx": flat_idx,
                "condition_idx": condition_idx,
                "attempt_idx": attempt_idx,
                "source_episode_id": source_episode_id,
                "seed": ep_seed,
                "policy_instruction": policy_instruction,
                "success": result.success,
                "steps": result.episode_length,
                "env_steps": int(env._current_step),
                "reward": float(result.episode_reward),
            }
            if hasattr(policy, "get_diagnostics"):
                ep_record["policy_diagnostics"] = policy.get_diagnostics()
            # Under ambiguous conditioning success is class-level, so record which
            # mode the policy actually took -- that histogram is the evidence the
            # policy is genuinely multimodal rather than collapsed onto one member.
            mode = getattr(result, "info", {}).get("mode")
            if mode is not None and hydra_cfg.env.instruction_level == "ambiguous":
                ep_record["mode"] = mode
                ep_record["achieved_tasks"] = result.info.get("achieved_tasks", [])
            if hard_fail_reason is not None:
                ep_record["hard_fail_reason"] = hard_fail_reason
            results["tasks"][task_name]["episodes"].append(ep_record)

            # Save immediately (crash-safe)
            save_results(results_path, results)

            attempt_tag = (
                f" cond {condition_idx + 1}/{n_available} try {attempt_idx + 1}/{tries_per_episode}"
                if tries_per_episode > 1 else ""
            )
            if result.success:
                logger.info(
                    f"    ✓ Rollout {flat_idx+1}/{total_attempts}{attempt_tag} SUCCEEDED "
                    f"(steps={result.episode_length}, reward={result.episode_reward:.2f})"
                )
            else:
                logger.info(
                    f"    ✗ Rollout {flat_idx+1}/{total_attempts}{attempt_tag} FAILED "
                    f"(steps={result.episode_length}, reward={result.episode_reward:.2f})"
                )

        # Log task summary so far
        eps = results["tasks"][task_name]["episodes"]
        n_success = sum(1 for e in eps if e["success"])
        avg_steps = np.mean([e["steps"] for e in eps if e["success"]]) if n_success else float("nan")
        logger.info(
            f"    -> {task_name}: {n_success}/{len(eps)} succeeded, "
            f"avg steps to success: {avg_steps:.0f}"
        )

    env.close()

    # Print final summary
    print_summary(results)
    return results


def print_summary(results: Dict):
    """Print a per-task summary table."""
    logger.info(f"\n{'='*80}")
    logger.info(f"{'Task':<35} {'Success':>10} {'Rate':>8} {'Avg Steps (succ)':>18}")
    logger.info(f"{'-'*80}")

    total_success = 0
    total_episodes = 0

    for task_name, task_data in results["tasks"].items():
        eps = task_data["episodes"]
        n = len(eps)
        n_success = sum(1 for e in eps if e["success"])
        rate = n_success / n if n else 0.0
        succ_steps = [e["steps"] for e in eps if e["success"]]
        avg_steps = np.mean(succ_steps) if succ_steps else float("nan")

        total_success += n_success
        total_episodes += n

        logger.info(f"  {task_name:<33} {n_success:>3}/{n:<4}  {rate:>7.1%}  {avg_steps:>16.0f}")

    overall_rate = total_success / total_episodes if total_episodes else 0.0
    logger.info(f"{'-'*80}")
    logger.info(f"  {'OVERALL':<33} {total_success:>3}/{total_episodes:<4}  {overall_rate:>7.1%}")
    logger.info(f"{'='*80}")

    print_mode_histogram(results)


def print_mode_histogram(results: Dict):
    """Per-task distribution over the modes actually taken, with normalized entropy.

    Only meaningful under ambiguous conditioning, where several tasks count as
    success. Entropy is normalized by log(number of distinct modes observed for the
    task) so tasks with differently sized ambiguity classes are comparable: 1.0 is a
    uniform spread over the observed modes, 0.0 means the policy always did the same
    thing. A collapsed histogram is the signal that the ambiguous mix ratio is too
    low, not that the eval is broken.
    """
    per_task = {}
    for task_name, task_data in results["tasks"].items():
        modes = [e["mode"] for e in task_data["episodes"] if e.get("mode")]
        if modes:
            per_task[task_name] = Counter(modes)
    if not per_task:
        return

    logger.info(f"\n{'='*80}")
    logger.info("Mode distribution under ambiguous conditioning")
    logger.info(f"{'-'*80}")
    for task_name, counts in per_task.items():
        total = sum(counts.values())
        probs = [c / total for c in counts.values()]
        entropy = -sum(p * np.log(p) for p in probs)
        ceiling = np.log(len(counts))
        normalized = entropy / ceiling if ceiling > 0 else 0.0
        spread = ", ".join(
            f"{mode}={count}" for mode, count in counts.most_common()
        )
        logger.info(
            f"  {task_name:<33} n={total:<4} modes={len(counts)} "
            f"H_norm={normalized:.2f}  [{spread}]"
        )
    logger.info(f"{'='*80}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()
    configure_determinism()
    if min(args.num_episodes, args.tries_per_episode, args.max_steps) < 1:
        raise ValueError("Episode, try, and step counts must be positive")
    if args.target_trials_per_task is not None and args.target_trials_per_task < 1:
        raise ValueError("Target trials must be positive")

    if args.target_trials_per_task is not None and args.tries_per_episode != 1:
        logger.error(
            "--target-trials-per-task and --tries-per-episode are mutually "
            "exclusive (they impose different iteration orderings). Pick one."
        )
        sys.exit(1)

    # Output directory (axis-prefixed when using a perturbed run so axes don't collide)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        dir_name = f"{args.perturbation_axis}_{timestamp}" if args.perturbation_axis else timestamp
        output_dir = Path("outputs/evaluation") / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load evaluation configs
    conf_dir = Path(__file__).parent.parent / "conf"
    eval_configs = []
    for name in args.evaluation:
        eval_path = conf_dir / "evaluation" / f"{name}.yaml"
        if not eval_path.exists():
            logger.error(f"Evaluation config not found: {eval_path}")
            sys.exit(1)
        eval_cfg = load_evaluation_config(conf_dir, name)
        if eval_cfg.get("distillation") and args.instruction_level != "specific":
            raise ValueError("Distilled evaluation requires specific task scoring; do not use --instruction-level ambiguous")
        eval_configs.append((name, eval_cfg))

    # Optional curated task list / ordering. Filters each condition's task list
    # to the intersection (preserving the JSON's order), so a single run can
    # skip e.g. all rotations while still using the per-condition configs.
    if args.tasks is not None:
        tasks_path = Path(args.tasks)
        if not tasks_path.exists():
            logger.error(f"--tasks file not found: {tasks_path}")
            sys.exit(1)
        with open(tasks_path) as f:
            tasks_data = json.load(f)
        ordered_tasks = tasks_data["tasks"] if isinstance(tasks_data, dict) else tasks_data
        for _, eval_cfg in eval_configs:
            eval_tasks = set(eval_cfg["tasks"])
            eval_cfg["tasks"] = [t for t in ordered_tasks if t in eval_tasks]
        logger.info(
            f"Task list overridden from {tasks_path}: "
            f"{len(ordered_tasks)} tasks requested"
        )

    # All tasks (union across configs, preserving order from first)
    all_tasks = list(dict.fromkeys(t for _, cfg in eval_configs for t in cfg["tasks"]))
    if not all_tasks:
        raise ValueError("No evaluation tasks selected")
    if args.target_trials_per_task is not None:
        suffix = (
            " (cycles over P4-valid scenes per task)"
            if args.perturbation_axis == "P4" else ""
        )
        logger.info(
            f"Evaluating {len(all_tasks)} tasks, pool={args.num_episodes}, "
            f"cycling to {args.target_trials_per_task} trials/task{suffix}"
        )
    elif args.tries_per_episode > 1:
        suffix = (
            " (P4 filter will further narrow the per-task condition count)"
            if args.perturbation_axis == "P4" else ""
        )
        logger.info(
            f"Evaluating {len(all_tasks)} tasks, pool={args.num_episodes} "
            f"× {args.tries_per_episode} tries/condition{suffix}"
        )
    else:
        logger.info(f"Evaluating {len(all_tasks)} tasks, {args.num_episodes} episodes each")

    # Dataset path from base Hydra config
    abs_conf_dir = str(conf_dir.resolve())
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=abs_conf_dir, version_base=None):
        base_cfg = compose(config_name="config", overrides=[
            value for value in args.override if value.startswith(("env.", "env="))
        ])
    if args.dataset_path:
        base_cfg.env.dataset_path = str(args.dataset_path.resolve())
    dataset_path = Path(base_cfg.env.dataset_path)
    split = base_cfg.env.split

    # Pre-sample starting conditions (shared across all conditions)
    starting_conditions, sampled_episode_ids = presample_starting_conditions(
        dataset_path, split, all_tasks, args.num_episodes, args.seed,
    )

    # P4: load manually verified valid starting indices. These were picked against
    # the same (seed, num_episodes) used here, so the indices map 1:1 into the
    # presampled lists. Mismatches are fatal — the filter would otherwise pick the
    # wrong starts.
    p4_valid_indices = None
    if args.perturbation_axis == "P4":
        p4_path = args.p4_valid_indices or conf_dir / "evaluation" / "p4_valid_indices.json"
        if not p4_path.exists():
            logger.error(
                f"--perturbation-axis P4 requires {p4_path}, the validated P4 starting states "
                f"(shipped as conf/evaluation/p4_valid_indices.json; --p4-valid-indices selects another file)."
            )
            sys.exit(1)
        with open(p4_path) as f:
            p4_data = json.load(f)
        if p4_data.get("annotation_sha256") and p4_data["annotation_sha256"] != file_sha256(base_cfg.env.lang_ann_path):
            raise ValueError("P4 labels were made with a different annotation index; use the labeled dataset or relabel")
        if p4_data.get("seed") != args.seed or p4_data.get("num_episodes") != args.num_episodes:
            logger.error(
                f"P4 valid-indices JSON was built with seed={p4_data.get('seed')}, "
                f"num_episodes={p4_data.get('num_episodes')}, but this run uses "
                f"seed={args.seed}, num_episodes={args.num_episodes}. Indices would "
                f"point at different starting conditions. Re-run the labeler or "
                f"match the seed/num-episodes."
            )
            sys.exit(1)
        p4_valid_indices = p4_data["valid_indices"]
        logger.info(f"Loaded P4 valid indices for {len(p4_valid_indices)} tasks")

    # Save/update sampled episode IDs
    ep_ids_path = output_dir / "sampled_episodes.json"
    if ep_ids_path.exists():
        previous = json.loads(ep_ids_path.read_text())
        if previous["seed"] != args.seed:
            raise ValueError("Cannot resume with a different scene seed")
        if args.target_trials_per_task is not None and previous["num_episodes"] != args.num_episodes:
            raise ValueError("Cannot change the scene pool while cycling trials; use a new output directory")
        for task, ids in previous["episode_ids"].items():
            if task in sampled_episode_ids:
                current = [int(x) for x in sampled_episode_ids[task]]
                if current[:len(ids)] != ids:
                    raise ValueError(f"Starting scenes changed for {task}; use a new output directory")
    if ep_ids_path.exists():
        saved_ids = previous["episode_ids"]
    else:
        saved_ids = {}
    saved_ids.update({k: [int(x) for x in v] for k, v in sampled_episode_ids.items()})
    with open(ep_ids_path, "w") as f:
        json.dump({
            "seed": args.seed,
            "num_episodes": args.num_episodes,
            "episode_ids": saved_ids,
        }, f, indent=2)

    # Run each condition
    for config_name, eval_cfg in eval_configs:
        overrides = [
            f"steering={eval_cfg['steering']}",
            "env.provide_pcd_images=true",
            "env.use_gui=false",
            f"env.max_steps={args.max_steps}",
        ]
        if "policy_config" in eval_cfg:
            overrides.insert(0, f"policy={eval_cfg['policy_config']}")
        for key, value in eval_cfg["policy"].items():
            overrides.append(f"policy.{key}={value}")
        for key, value in eval_cfg.get("steering_overrides", {}).items():
            overrides.append(f"steering.{key}={value}")
        if args.perturbation_axis is not None:
            overrides += [
                "env.perturbed_ann_path=perturbed_language_annotations.json",
                f"env.perturbation_axis={args.perturbation_axis}",
            ]
        if args.instruction_level != "specific":
            overrides.append(f"env.instruction_level={args.instruction_level}")
        overrides.extend(args.override)

        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=abs_conf_dir, version_base=None):
            hydra_cfg = compose(config_name="config", overrides=overrides)
        hydra_cfg.env.dataset_path = str(dataset_path)
        hydra_cfg.env.lang_ann_path = base_cfg.env.lang_ann_path
        if eval_cfg.get("distillation") and hydra_cfg.env.instruction_level != "specific":
            raise ValueError("Distillation must retain specific task scoring")
        if args.dry_run:
            if not Path(hydra_cfg.policy.ckpt_path).is_file():
                raise FileNotFoundError(hydra_cfg.policy.ckpt_path)
            logger.info("%s: checkpoint=%s, text_tokens=%s, steering=%s, distilled=%s, tasks=%d",
                        config_name, hydra_cfg.policy.ckpt_path,
                        hydra_cfg.policy.get("text_max_length"), hydra_cfg.steering.name,
                        bool(eval_cfg.get("distillation")), len(eval_cfg["tasks"]))
            continue
        if args.prepare_language:
            from envs.calvin_utils.language_ann import load_language_annotations, load_perturbed_annotations
            instructions = (
                load_perturbed_annotations(hydra_cfg.env.perturbed_ann_path, args.perturbation_axis)
                if args.perturbation_axis else load_language_annotations(hydra_cfg.env.lang_ann_path)
            )
            distiller = LanguageDistiller.from_config(eval_cfg["distillation"]) if eval_cfg.get("distillation") else None
            texts = []
            for task in eval_cfg["tasks"]:
                original = instructions[task]
                distilled = distiller.distill(original) if distiller else original
                texts.append({"task": task, "original": original, "policy_instruction": distilled})
                logger.info("%s: %r -> %r", task, original, distilled)
                save_results(output_dir / f"{config_name}.conditioning.json", {
                    "condition_name": eval_cfg["condition_name"], "distillation": eval_cfg.get("distillation"),
                    "prompt_sha256": file_sha256(PROMPT_PATH) if distiller else None,
                    "instructions": texts,
                })
            continue

        # Each condition gets its own results file (resumable independently)
        results_path = output_dir / f"{config_name}.json"

        run_condition(
            eval_cfg=eval_cfg,
            hydra_cfg=hydra_cfg,
            starting_conditions=starting_conditions,
            sampled_episode_ids=sampled_episode_ids,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            base_seed=args.seed,
            results_path=results_path,
            perturbation_axis=args.perturbation_axis,
            p4_valid_indices=p4_valid_indices,
            tries_per_episode=args.tries_per_episode,
            target_trials_per_task=args.target_trials_per_task,
            config_name=config_name,
            process_plan=[(name, cfg["tasks"]) for name, cfg in eval_configs],
        )

    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
