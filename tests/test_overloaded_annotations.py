"""Validation for the overloaded / specific language annotation files.

These files are hand-authored, so the invariants the training and eval pipelines
rely on are asserted here rather than discovered during a 600K-iteration run.
"""

import json
from pathlib import Path

import pytest

from envs.calvin_utils.task_configs import TASK_INITIAL_CONDITIONS

REPO_ROOT = Path(__file__).resolve().parent.parent
OVERLOADED = REPO_ROOT / "overloaded_language_annotations.json"
SPECIFIC = REPO_ROOT / "calvin_language_annotations.json"

COLOR_WORDS = ("red", "blue", "pink")
# `grasp` is its own primitive in action_primitive_object_annotations.json, so an
# L0 template must never use it as a stand-in for lift/pick-up.
FORBIDDEN_L0_VERBS = ("grasp",)


@pytest.fixture(scope="module")
def overloaded():
    return json.loads(OVERLOADED.read_text())


@pytest.fixture(scope="module")
def specific():
    return json.loads(SPECIFIC.read_text())


def test_specific_annotations_are_strict_json(specific):
    assert len(specific) == 34
    for task, strings in specific.items():
        assert strings, task
        assert all(isinstance(s, str) and s.strip() for s in strings), task
        assert len(set(strings)) == len(strings), f"{task} has duplicate strings"


def test_partition_covers_all_34_tasks_exactly_once(overloaded):
    covered = [t for c in overloaded["classes"].values() for t in c["tasks"]]
    specific_only = list(overloaded["specific_only"])
    everything = covered + specific_only

    assert len(everything) == len(set(everything)), "a task appears in more than one group"
    assert set(everything) == set(TASK_INITIAL_CONDITIONS), (
        "class table and TASK_INITIAL_CONDITIONS disagree: "
        f"extra={sorted(set(everything) - set(TASK_INITIAL_CONDITIONS))} "
        f"missing={sorted(set(TASK_INITIAL_CONDITIONS) - set(everything))}"
    )
    assert len(everything) == 34


def test_every_task_has_a_specific_pool(overloaded, specific):
    for task in TASK_INITIAL_CONDITIONS:
        assert task in specific, f"{task} has no specific instruction pool"


def test_l0_templates_are_low_specificity(overloaded):
    for name, cls in overloaded["classes"].items():
        assert cls["ambiguous"], name
        for template in cls["ambiguous"]:
            words = template.lower().split()
            assert not any(c in words for c in COLOR_WORDS), (
                f"{name}: L0 template {template!r} names a color"
            )
            assert not any(v in words for v in FORBIDDEN_L0_VERBS), (
                f"{name}: L0 template {template!r} crosses a primitive boundary"
            )


def test_l0_templates_are_unique_across_classes(overloaded):
    """A template shared by two classes would make the ambiguity class unrecoverable."""
    seen = {}
    for name, cls in overloaded["classes"].items():
        for template in cls["ambiguous"]:
            assert template not in seen, (
                f"{template!r} is shared by {seen[template]} and {name}"
            )
            seen[template] = name


def test_feasible_modes_are_class_members(overloaded):
    for name, cls in overloaded["classes"].items():
        assert cls["feasible_modes"], name
        assert set(cls["feasible_modes"]) <= set(cls["tasks"]), (
            f"{name}: feasible_modes contains a non-member"
        )


def test_eval_conditions_match_the_scene_schema(overloaded):
    schema = set(TASK_INITIAL_CONDITIONS["open_drawer"])
    for name, cls in overloaded["classes"].items():
        assert set(cls["eval_condition"]) == schema, (
            f"{name}: eval_condition keys do not match TASK_INITIAL_CONDITIONS"
        )


def test_specific_only_tasks_document_why(overloaded):
    for task, reason in overloaded["specific_only"].items():
        assert isinstance(reason, str) and reason.strip(), task
