"""Eval-side ambiguity wiring: instruction lookup, success set, reset condition.

These exercise CalvinEnvironment's new methods directly against a stub, so they run
without booting PyBullet.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from envs.calvin import CalvinEnvironment
from envs.calvin_utils.language_ann import (
    get_ambiguity_class,
    get_ambiguous_instruction_for_task,
    load_ambiguous_annotations,
    load_overloaded_annotations,
)
from envs.calvin_utils.task_configs import TASK_INITIAL_CONDITIONS

OVERLOADED = str(Path(__file__).resolve().parent.parent / "overloaded_language_annotations.json")


@pytest.fixture(scope="module")
def overloaded():
    return load_overloaded_annotations(OVERLOADED)


@pytest.fixture(scope="module")
def ambiguous_map():
    return load_ambiguous_annotations(OVERLOADED)


def make_env(task, level, overloaded=None, ambiguous_map=None):
    env = SimpleNamespace(
        _task_name=task,
        _instruction_level=level,
        _overloaded_ann=overloaded if level == "ambiguous" else None,
        _task_instructions=ambiguous_map,
        _task_oracle=SimpleNamespace(tasks=set(TASK_INITIAL_CONDITIONS)),
    )
    env._ambiguity_class = CalvinEnvironment._lookup_ambiguity_class(env, task)
    return env


def test_ambiguous_lookup_shares_one_template_across_the_class(ambiguous_map):
    lifts = ["lift_red_block_table", "lift_blue_block_slider", "lift_pink_block_drawer"]
    templates = {get_ambiguous_instruction_for_task(t, ambiguous_map) for t in lifts}
    assert templates == {"pick up the block"}


def test_specific_conditioning_scores_only_the_task(overloaded, ambiguous_map):
    env = make_env("lift_red_block_table", "specific", overloaded, ambiguous_map)
    assert CalvinEnvironment._success_task_set(env) == {"lift_red_block_table"}


def test_ambiguous_conditioning_scores_the_whole_class(overloaded, ambiguous_map):
    """Full class, not feasible_modes: eval resets to real scenes whose reachable
    members vary, and the oracle only reports what was actually completed."""
    env = make_env("lift_red_block_table", "ambiguous", overloaded, ambiguous_map)
    assert CalvinEnvironment._success_task_set(env) == set(
        get_ambiguity_class("lift_red_block_table", overloaded)["tasks"]
    )
    assert len(CalvinEnvironment._success_task_set(env)) == 9


def test_ambiguous_success_set_never_exceeds_the_class(overloaded, ambiguous_map):
    for name, cls in overloaded["classes"].items():
        for task in cls["tasks"]:
            env = make_env(task, "ambiguous", overloaded, ambiguous_map)
            targets = CalvinEnvironment._success_task_set(env)
            assert targets, f"{name}/{task} has an empty success set"
            assert targets <= set(cls["tasks"]), f"{name}/{task} scores outside its class"


def test_specific_only_task_falls_back_to_itself(overloaded, ambiguous_map):
    """No class -> the success set is the single task, even if level says ambiguous."""
    env = make_env("open_drawer", "ambiguous", overloaded, ambiguous_map)
    assert env._ambiguity_class is None
    assert CalvinEnvironment._success_task_set(env) == {"open_drawer"}


def test_class_eval_condition_makes_every_feasible_mode_reachable(overloaded):
    """The whole point of a per-class condition: no member is state-excluded."""
    light_off = get_ambiguity_class("turn_off_led", overloaded)
    assert light_off["eval_condition"]["led"] == 1
    assert light_off["eval_condition"]["lightbulb"] == 1

    # Per-task conditions do NOT satisfy that -- which is why the class one exists.
    assert TASK_INITIAL_CONDITIONS["turn_off_led"]["lightbulb"] == 0
    assert TASK_INITIAL_CONDITIONS["turn_off_lightbulb"]["led"] == 0


def test_lift_feasible_modes_exclude_unreachable_members(overloaded):
    """Blocks can only be on the table or in the slider, never in the drawer."""
    lift = get_ambiguity_class("lift_red_block_table", overloaded)
    assert len(lift["tasks"]) == 9
    assert all(t.endswith("_table") for t in lift["feasible_modes"])
    assert all(
        lift["eval_condition"][f"{c}_block"] == "table" for c in ("red", "blue", "pink")
    )
