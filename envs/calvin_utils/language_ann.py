"""Language annotation loading for CALVIN dataset.

Loads language annotations from auto_lang_ann.npy file and creates
task name to instruction mappings.
"""

import json
import os
import logging
from typing import Dict
import numpy as np

logger = logging.getLogger(__name__)


def load_language_annotations(ann_path: str) -> Dict[str, str]:
    """
    Load CALVIN language annotations and create task -> instruction mapping.

    The CALVIN dataset provides language annotations in the format:
    {
        'language': {
            'ann': [list of instruction strings],
            'task': [list of task IDs],
            'emb': [precomputed embeddings]
        },
        'info': {
            'indx': [start/end indices for sequences]
        }
    }

    Args:
        ann_path: Path to auto_lang_ann.npy file

    Returns:
        Dictionary mapping task names/IDs to language instructions
        Empty dict if file not found
    """
    if ann_path is None or not os.path.exists(ann_path):
        logger.warning(f"Language annotation file not found: {ann_path}")
        logger.info("Falling back to task names as instructions")
        return {}

    try:
        ann_data = np.load(ann_path, allow_pickle=True).item()
        logger.info(f"Loaded language annotations from: {ann_path}")

        # Extract language and task information
        if 'language' not in ann_data:
            logger.error("Invalid annotation file format: missing 'language' key")
            return {}

        language_data = ann_data['language']
        if 'ann' not in language_data or 'task' not in language_data:
            logger.error("Invalid annotation file format: missing 'ann' or 'task' keys")
            return {}

        annotations = language_data['ann']
        task_ids = language_data['task']

        # Create task -> instruction mapping
        # Use first annotation for each unique task
        task_to_instruction = {}
        for task_id, instruction in zip(task_ids, annotations):
            task_key = str(task_id)  # Ensure string key
            if task_key not in task_to_instruction:
                task_to_instruction[task_key] = instruction
                logger.debug(f"Task '{task_key}': {instruction}")

        logger.info(f"Loaded {len(task_to_instruction)} unique task annotations")
        return task_to_instruction

    except Exception as e:
        logger.error(f"Error loading language annotations: {e}")
        return {}


def get_instruction_for_task(task_name: str, task_instructions: Dict[str, str]) -> str:
    """
    Get language instruction for a specific task.

    Args:
        task_name: Task name/ID
        task_instructions: Dictionary of task -> instruction mappings

    Returns:
        Language instruction string, or task_name if not found
    """
    # Try direct lookup
    if task_name in task_instructions:
        return task_instructions[task_name]

    # Try case-insensitive lookup
    task_name_lower = task_name.lower()
    for key, instruction in task_instructions.items():
        if key.lower() == task_name_lower:
            return instruction

    # Fallback to task name
    logger.warning(f"No annotation found for task '{task_name}', using task name as instruction")
    return task_name


def load_perturbed_annotations(path: str, axis: str) -> Dict[str, str]:
    """Load task -> instruction map from a perturbed annotations JSON for a given axis.

    The JSON is expected to map each task name to a dict with keys P1..P4, e.g.:
        {"open_drawer": {"P1": "pull the drawer", "P2": ..., "P3": ..., "P4": ...}, ...}

    Raises FileNotFoundError if path is missing and ValueError if any task lacks the
    requested axis — callers should fail fast rather than silently fall back.
    """
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Perturbed annotations file not found: {path}")

    with open(path) as f:
        data = json.load(f)

    missing = [t for t, axes in data.items() if axis not in axes]
    if missing:
        raise ValueError(
            f"Perturbed annotations at {path} missing axis '{axis}' for tasks: {missing}"
        )

    task_to_instruction = {task: axes[axis] for task, axes in data.items()}
    logger.info(
        f"Loaded perturbed annotations from {path} (axis={axis}, {len(task_to_instruction)} tasks)"
    )
    return task_to_instruction


def load_overloaded_annotations(path: str) -> Dict:
    """Load and validate the ambiguity-class table.

    Returns the parsed overloaded_language_annotations.json. Raises if the file is
    missing or if its `classes` and `specific_only` groups do not partition the task
    set, since a silently incomplete table would mis-score an ambiguous sweep.
    """
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Overloaded annotations file not found: {path}")

    with open(path) as f:
        data = json.load(f)

    covered = [t for cls in data["classes"].values() for t in cls["tasks"]]
    everything = covered + list(data["specific_only"])
    duplicates = {t for t in everything if everything.count(t) > 1}
    if duplicates:
        raise ValueError(
            f"Overloaded annotations at {path} assign these tasks to more than one "
            f"group: {sorted(duplicates)}"
        )

    logger.info(
        f"Loaded {len(data['classes'])} ambiguity classes covering {len(covered)} tasks "
        f"from {path} ({len(data['specific_only'])} specific-only)"
    )
    return data


def load_ambiguous_annotations(path: str) -> Dict[str, str]:
    """Load task -> low-specificity (L0) instruction for the tasks that have one.

    Only the tasks belonging to an ambiguity class appear in the result. The
    specific-only tasks are deliberately absent so that
    `get_ambiguous_instruction_for_task` raises on them rather than quietly falling
    back to a specific instruction, which would make an ambiguous sweep silently
    score some tasks under the wrong conditioning.
    """
    data = load_overloaded_annotations(path)
    return {
        task: cls["ambiguous"][0]
        for cls in data["classes"].values()
        for task in cls["tasks"]
    }


def get_ambiguous_instruction_for_task(
    task_name: str, task_instructions: Dict[str, str]
) -> str:
    """Look up a task's L0 instruction, raising if the task has no ambiguous tier."""
    if task_name not in task_instructions:
        raise KeyError(
            f"Task '{task_name}' has no ambiguous instruction — it is specific-only "
            "(its siblings are not executable from the same scene state). Evaluate it "
            "under specific conditioning instead."
        )
    return task_instructions[task_name]


def get_ambiguity_class(task_name: str, overloaded: Dict) -> Dict:
    """Return the ambiguity class dict a task belongs to, or None if specific-only.

    Class-level success and mode entropy must be scored against `feasible_modes`,
    not `tasks`: per-class eval conditions cannot make every member simultaneously
    executable, so `tasks` overstates the reachable mode count.
    """
    for name, cls in overloaded["classes"].items():
        if task_name in cls["tasks"]:
            return {"name": name, **cls}
    return None
