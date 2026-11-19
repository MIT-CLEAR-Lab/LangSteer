"""Entry point for the CALVIN evaluation matrix of the overloaded checkpoints.

Four conditions (base, distilled, base + steering, distilled + steering) on
instruction axes P0-P4, for 3D Diffuser Actor (`--backbone 3dda`, default) or
pi0.5 (`--backbone pi05`, needs the pi0.5 server: see pi05_guidance/server.py).

    python scripts/run_overloaded_evaluation.py matrix --dataset-path <task_D_D> --output-dir <out>
    python scripts/run_overloaded_evaluation.py summary --output-dir <out>

Each axis writes `<out>/<axis>/<condition>.json`, saved after every episode and
resumed by repeating the command. LLM outputs are cached under `<out>/cache`;
copy or link an existing cache there to reuse exactly the same plans.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.evaluation_config import LAUNCH_ENV  # noqa: E402
CONDITIONS = ["overloaded_base", "overloaded_distilled",
              "overloaded_base_steered", "overloaded_distilled_steered"]
PI05_CONDITIONS = ["pi05_" + name for name in CONDITIONS]


def build_commands(args):
    pi05 = getattr(args, "backbone", "3dda") == "pi05"
    prefix = "pi05_" if pi05 else ""
    axes = ["P0"] if args.phase == "sanity" else args.axes
    conditions = [prefix + "overloaded_base"] if args.phase == "sanity" else args.conditions
    if args.phase == "prepare":
        conditions = [prefix + "overloaded_distilled"]
    commands = []
    for axis in axes:
        command = [sys.executable, str(ROOT / "scripts/run_evaluation.py"),
                   "--evaluation", *conditions, "--dataset-path", str(args.dataset_path.resolve()),
                   "--num-episodes", str(args.num_episodes), "--seed", str(args.seed),
                   "--max-steps", str(args.max_steps),
                   "--output-dir", str(args.output_dir.resolve() / axis)]
        if axis != "P0":
            command += ["--perturbation-axis", axis]
        if args.phase == "matrix":
            command += ["--target-trials-per-task", str(args.trials_per_task)]
        if args.tasks:
            command += ["--tasks", str(args.tasks.resolve())]
        if args.p4_valid_indices:
            command += ["--p4-valid-indices", str(args.p4_valid_indices.resolve())]
        if args.dry_run:
            command += ["--dry-run"]
        if args.phase == "prepare":
            command += ["--prepare-language"]
        for override in args.override:
            command += ["--override", override]
        commands.append(command)
    return commands


def summarize(output_dir):
    print("axis,condition,successes,trials,micro_success,macro_success,hard_failures")
    for path in sorted(output_dir.glob("P*/*.json")):
        data = json.loads(path.read_text())
        if "tasks" not in data:
            continue
        groups = [t["episodes"] for t in data["tasks"].values() if t["episodes"]]
        episodes = [e for group in groups for e in group]
        successes = sum(bool(e["success"]) for e in episodes)
        macro = sum(sum(bool(e["success"]) for e in g) / len(g) for g in groups) / len(groups) if groups else 0
        micro = successes / len(episodes) if episodes else 0
        failures = sum(bool(e.get("hard_fail_reason")) for e in episodes)
        print(f"{path.parent.name},{path.stem},{successes},{len(episodes)},{micro:.4f},{macro:.4f},{failures}")


def execute_commands(commands, *, env, workers=1):
    """Axes have disjoint result paths; seeded π0.5 serving serializes inference."""
    def execute(command):
        print(shlex.join(command), flush=True)
        if workers == 1:
            subprocess.run(command, cwd=ROOT, env=env, check=True)
        else:
            directory = Path(command[command.index('--output-dir')+1])
            directory.mkdir(parents=True, exist_ok=True)
            with (directory/'worker.log').open('a') as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(execute, commands))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["sanity", "prepare", "matrix", "summary"])
    parser.add_argument("--backbone", choices=["3dda", "pi05"], default="3dda")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent axis simulators sharing a seeded π0.5 server")
    parser.add_argument("--dataset-path", type=Path, default=os.environ.get("CALVIN_DATASET_PATH"))
    parser.add_argument("--checkpoint", type=Path, help="3DDA weight file, or the π0.5 COMPLETED.json manifest")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--axes", nargs="+", choices=["P0", "P1", "P2", "P3", "P4"], default=["P0", "P1", "P2", "P3", "P4"])
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS + PI05_CONDITIONS)
    parser.add_argument("--num-episodes", type=int, default=None, help="Scene pool; default sanity=1, matrix=30 (P4 labels)")
    parser.add_argument("--trials-per-task", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=360)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tasks", type=Path)
    parser.add_argument("--p4-valid-indices", type=Path)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or (args.workers > 1 and args.backbone != 'pi05'):
        parser.error("Multiple workers require the seeded π0.5 server")
    expected_conditions = PI05_CONDITIONS if args.backbone == "pi05" else CONDITIONS
    if args.conditions is None:
        args.conditions = expected_conditions
    if not set(args.conditions).issubset(expected_conditions):
        parser.error("Conditions must use the selected backbone")
    if args.checkpoint is None:
        args.checkpoint = ROOT / ("checkpoints/pi05_calvin_overloaded/200000/COMPLETED.json" if args.backbone == "pi05" else "checkpoints/overloaded/last.pth")
    if args.phase == "summary":
        summarize(args.output_dir)
        return
    if not args.dataset_path:
        parser.error("Provide --dataset-path or CALVIN_DATASET_PATH")
    args.dataset_path = Path(args.dataset_path)
    if args.num_episodes is None:
        args.num_episodes = 1 if args.phase == "sanity" else 30
    if min(args.num_episodes, args.trials_per_task, args.max_steps) < 1:
        parser.error("Episode/trial/step counts must be positive")
    if args.phase == "sanity" and args.tasks is None:
        args.tasks = ROOT / "conf/evaluation/overloaded_sanity_tasks.json"
    env = os.environ.copy()
    env["OVERLOADED_CHECKPOINT"] = str(args.checkpoint.resolve())
    if args.backbone == "pi05":
        env["PI05_OVERLOADED_CHECKPOINT"] = str(args.checkpoint.resolve())
    # All axes/conditions share caches. Carry this directory to another machine
    # to reuse precisely the same LLM outputs.
    env["LANGSTEER_DISTILL_CACHE"] = str(args.output_dir.resolve() / "cache/distillation")
    env["LANGSTEER_STEERING_CACHE"] = str(args.output_dir.resolve() / "cache/steering")
    # Recorded in each condition's session provenance (utils/evaluation_config.py).
    env[LAUNCH_ENV] = json.dumps({
        "backbone": args.backbone, "phase": args.phase, "workers": args.workers,
        "axes": ["P0"] if args.phase == "sanity" else list(args.axes),
        "conditions": list(args.conditions),
    })
    execute_commands(build_commands(args), env=env, workers=args.workers)
    summarize(args.output_dir.resolve())


if __name__ == "__main__":
    main()
