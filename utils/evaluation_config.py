"""Small evaluation config inheritance and immutable run metadata helpers."""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]

# Packages whose versions are recorded with every evaluation session.
KEY_PACKAGES = (
    "torch", "torchvision", "numpy", "scipy", "diffusers", "transformers", "openai",
    "anthropic", "hydra-core", "omegaconf", "pybullet", "calvin-env", "clip", "openpi-client",
)
# Environment variable through which run_overloaded_evaluation.py passes its launch
# layout (backbone, worker count, axes, conditions) to each run_evaluation.py process.
LAUNCH_ENV = "LANGSTEER_LAUNCH"


def load_evaluation_config(conf_dir, name, parents=()):
    if name in parents:
        raise ValueError(f"Evaluation config inheritance cycle: {parents + (name,)}")
    cfg = OmegaConf.load(Path(conf_dir) / "evaluation" / f"{name}.yaml")
    parent = cfg.pop("extends", None)
    if parent:
        cfg = OmegaConf.merge(
            load_evaluation_config(conf_dir, parent, parents + (name,)), cfg
        )
    return OmegaConf.to_container(cfg, resolve=True)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_resume(previous, current, path):
    if previous != current:
        changed = [key for key in set(previous) | set(current)
                   if previous.get(key) != current.get(key)]
        raise ValueError(
            f"Cannot resume {path}: changed {', '.join(sorted(changed))}. "
            "Use a new output directory."
        )


# ---------------------------------------------------------------------------
# Session provenance
#
# `run_metadata` (above) is the run's identity: a resume must match it exactly.
# Provenance is recorded per session (each start or resume of a condition) in a
# separate `provenance` list, so that code revisions, dependency versions and the
# process layout are on record without making every commit a reason to refuse a
# resume. Only a change of the steering prompts refuses a resume, since the prompt
# text is part of the LLM cache key and so decides the plans.
# ---------------------------------------------------------------------------

def _git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=True).stdout


def source_identity(root=REPO_ROOT):
    """Code revision of `root`, whether the source differs from it, and a hash of the
    difference (tracked changes plus untracked, non-ignored files)."""
    try:
        revision = _git(root, "rev-parse", "HEAD").decode().strip()
        diff = _git(root, "diff", "HEAD", "--binary")
        untracked = _git(root, "ls-files", "--others", "--exclude-standard", "-z").split(b"\0")
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None, "dirty_sha256": None}
    digest = hashlib.sha256(diff)
    dirty = bool(diff)
    for name in sorted(n for n in untracked if n):
        dirty = True
        digest.update(b"\0untracked\0" + name + b"\0")
        path = Path(root) / name.decode()
        if path.is_file():
            digest.update(path.read_bytes())
    return {"revision": revision, "dirty": dirty, "dirty_sha256": digest.hexdigest() if dirty else None}


def steering_prompt_hashes():
    """SHA-256 of every prompt the steering LMPs can load, from the prompt root in
    effect (LANGSTEER_PROMPT_DIR or voxposer/prompts)."""
    from voxposer.utils import prompt_root

    root = Path(prompt_root())
    return {
        "prompt_dir_overridden": bool(os.environ.get("LANGSTEER_PROMPT_DIR")),
        "sha256": {
            str(path.relative_to(root)): file_sha256(path)
            for path in sorted(root.rglob("*.txt"))
        },
    }


def package_versions(packages=KEY_PACKAGES):
    versions = {"python": platform.python_version()}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def launch_layout():
    """The launch layout passed by run_overloaded_evaluation.py, or None when this
    process was started directly."""
    raw = os.environ.get(LAUNCH_ENV)
    return json.loads(raw) if raw else None


def session_provenance(*, process_plan, condition, steered, completed_episodes):
    """Provenance of one evaluation session of one condition.

    `process_plan` is the ordered list of (condition, [tasks]) this process runs;
    per-episode results depend on what ran earlier in the same process, so the
    order is part of the record. `completed_episodes` maps each task to the number
    of episodes already recorded when the session started (non-zero on resume).
    """
    return {
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source_identity(),
        "process_plan": [{"condition": name, "tasks": list(tasks)} for name, tasks in process_plan],
        "condition": condition,
        "launch": launch_layout(),
        "steering_prompts": steering_prompt_hashes() if steered else None,
        "packages": package_versions(),
        "completed_episodes_at_start": dict(completed_episodes),
    }


def check_session_provenance(previous_sessions, current, path):
    """Refuse a resume whose steering prompts differ from the last session's.

    Returns the names of the other recorded fields that changed since the last
    session (code, dependencies, layout), for the caller to log.
    """
    if not previous_sessions:
        return []
    last = previous_sessions[-1]
    before = (last.get("steering_prompts") or {}).get("sha256")
    after = (current.get("steering_prompts") or {}).get("sha256")
    if before is not None and after is not None and before != after:
        changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        raise ValueError(
            f"Cannot resume {path}: steering prompts changed ({', '.join(changed)}). "
            "Use a new output directory."
        )
    return [key for key in ("source", "packages", "process_plan", "launch")
            if last.get(key) != current.get(key)]
