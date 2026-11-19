"""Perception helpers that generated programs use for positional instructions."""

import numpy as np
import pytest

from voxposer.calvin_interface import CalvinLMPInterface

ON_TABLE = (0.10, -0.10, 0.46)
IN_CABINET = (0.05, 0.08, 0.461)  # table height, but inside the cabinet cavity
IN_DRAWER = (0.15, -0.30, 0.362)
GRIPPER_OPEN, GRIPPER_CLOSED = 0.08, 0.03


def _interface(red, blue, pink, ee=(0.0, -0.2, 0.60), gripper_width=GRIPPER_OPEN):
    robot_obs = np.zeros(15, dtype=np.float32)
    robot_obs[:3] = ee
    robot_obs[6] = gripper_width
    scene_obs = np.zeros(24, dtype=np.float32)
    scene_obs[6:9], scene_obs[12:15], scene_obs[18:21] = red, blue, pink
    interface = CalvinLMPInterface({})
    interface.update_state(robot_obs, scene_obs)
    return interface


def _names(blocks):
    return [b.name for b in blocks]


def test_each_place_returns_only_the_blocks_lying_there():
    interface = _interface(red=ON_TABLE, blue=IN_CABINET, pink=IN_DRAWER)

    assert _names(interface.blocks("table")) == ["red block"]
    assert _names(interface.blocks("cabinet")) == ["blue block"]
    assert _names(interface.blocks("drawer")) == ["pink block"]
    assert _names(interface.blocks("gripper")) == []
    assert _names(interface.blocks()) == ["red block", "blue block", "pink block"]


def test_a_cabinet_block_is_not_on_the_table_although_it_is_at_table_height():
    interface = _interface(red=IN_CABINET, blue=ON_TABLE, pink=IN_DRAWER)

    assert abs(IN_CABINET[2] - ON_TABLE[2]) < 0.01
    assert _names(interface.blocks("table")) == ["blue block"]


def test_a_block_grasped_but_not_yet_lifted_is_held_and_still_on_the_table():
    interface = _interface(
        red=ON_TABLE,
        blue=IN_CABINET,
        pink=IN_DRAWER,
        ee=ON_TABLE,
        gripper_width=GRIPPER_CLOSED,
    )

    assert _names(interface.blocks("gripper")) == ["red block"]
    assert _names(interface.blocks("table")) == ["red block"]


def test_a_block_carried_above_the_table_is_held_and_in_no_other_place():
    carried = (ON_TABLE[0], ON_TABLE[1], ON_TABLE[2] + 0.08)
    interface = _interface(
        red=carried,
        blue=IN_CABINET,
        pink=IN_DRAWER,
        ee=carried,
        gripper_width=GRIPPER_CLOSED,
    )

    assert _names(interface.blocks("gripper")) == ["red block"]
    assert _names(interface.blocks("table")) == []


def test_a_block_stacked_on_another_is_in_no_place():
    stacked = (ON_TABLE[0], ON_TABLE[1], ON_TABLE[2] + 0.05)
    interface = _interface(red=ON_TABLE, blue=stacked, pink=IN_DRAWER)

    assert _names(interface.blocks("table")) == ["red block"]
    assert "blue block" in _names(interface.blocks())
    assert all(
        "blue block" not in _names(interface.blocks(p))
        for p in ("table", "drawer", "cabinet")
    )


def test_an_unknown_place_is_rejected():
    interface = _interface(red=ON_TABLE, blue=IN_CABINET, pink=IN_DRAWER)

    with pytest.raises(ValueError):
        interface.blocks("shelf")


def test_toward_and_away_from_give_the_sideways_direction_relative_to_a_landmark():
    interface = _interface(
        red=(0.10, -0.10, 0.46), blue=(-0.20, -0.10, 0.46), pink=IN_DRAWER
    )

    # the light switch is at the right edge of the workspace, the button left of centre
    assert interface.toward("red block", "light switch") == "right"
    assert interface.away_from("red block", "light switch") == "left"
    assert interface.toward("red block", "button") == "left"
    assert interface.toward("blue block", "button") == "right"


def test_inward_is_toward_the_middle_of_the_table():
    interface = _interface(
        red=(0.25, -0.10, 0.46), blue=(-0.25, -0.10, 0.46), pink=IN_DRAWER
    )

    assert interface.toward("red block", "table") == "left"
    assert interface.toward("blue block", "table") == "right"
