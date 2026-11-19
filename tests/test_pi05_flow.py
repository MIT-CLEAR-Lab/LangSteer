"""Flow guidance math. Run in the OpenPI environment with JAX_PLATFORMS=cpu."""
import numpy as np
import pytest

jax = pytest.importorskip('jax')
import jax.numpy as jnp  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from pi05_guidance.flow import (actions_for_shift, clean_action_shift, relative_path,  # noqa: E402
                                relative_rotation_matrix, waypoint_shift)
from pi05_guidance.server import validate_guidance  # noqa: E402

H = 10


def guidance(**overrides):
    # cost = |x - 0.5|: slope 1 per metre, lowest at x = 0.5
    axis = np.linspace(-1, 1, 41, dtype=np.float32)
    grid = np.broadcast_to(np.abs(axis-.5)[:, None, None], (41, 41, 41)).copy()
    g = dict(cost_map=jnp.asarray(grid), workspace_min=jnp.full(3, -1.), workspace_max=jnp.ones(3),
             anchor=jnp.zeros(6), controller_scale=jnp.array([.02]*3+[.05]*3),
             action_scale=jnp.array([.7, .6, .5, .43, .45, 1.]), action_offset=jnp.zeros(6),
             position_strength=jnp.asarray(1.), position_gain=jnp.array([.11, .088, .04]),
             rotation_strength=jnp.asarray(0.), rotation_target_euler=jnp.zeros(3),
             rotation_ramp=jnp.asarray(.6*(np.arange(H)/19)**2), rotation_alignment=jnp.array([.5, .1, .05, 1.]),
             timestep_scaling=jnp.asarray(True), min_timestep_scale=jnp.asarray(.1),
             time_floor=jnp.asarray(0.), rotation_time_floor=jnp.asarray(0.),
             rotation_yield=jnp.asarray(0.),
             override_strength=jnp.asarray(0.), override_progress=jnp.asarray(.02), override_reach=jnp.asarray(0.),
             bound_result=jnp.asarray(0.))
    g.update({k: jnp.asarray(v) for k, v in overrides.items()})
    return g


def test_last_step_moves_every_waypoint_by_strength_times_gain_times_slope():
    shift = waypoint_shift(jnp.zeros((1, H, 32)), 0.1, -0.1, guidance())
    # ramp at the last step is min_timestep_scale = 0.1; gain 0.11 m^2; slope 1 per metre, toward +x
    np.testing.assert_allclose(shift[0, :, 0], 0.1*0.11, rtol=1e-4)
    np.testing.assert_allclose(shift[0, :, 1:], 0, atol=1e-7)


def test_ramp_uses_the_noise_level_after_the_step():
    first = waypoint_shift(jnp.zeros((1, H, 32)), 1.0, -0.1, guidance())
    np.testing.assert_allclose(first[0, 0, 0], (0.1+0.9*0.9)*0.11, rtol=1e-4)


def test_time_floor_guides_only_the_steps_that_start_above_it():
    clean = jnp.zeros((1, H, 32))
    early = guidance(time_floor=.5, rotation_time_floor=.5, rotation_strength=2., rotation_target_euler=[0., 0., 1.])
    for time in (1.0, 0.8, 0.6000001, 0.5999999):
        assert float(jnp.abs(waypoint_shift(clean, time, -0.1, early)).sum()) > 0
    for time in (0.5000001, 0.4999999, 0.3, 0.1):
        np.testing.assert_array_equal(waypoint_shift(clean, time, -0.1, early), 0)
    # the two windows are independent
    split = guidance(time_floor=.5, rotation_strength=2., rotation_target_euler=[0., 0., 1.])
    late = waypoint_shift(clean, 0.3, -0.1, split)
    np.testing.assert_array_equal(late[..., :3], 0)
    assert float(jnp.abs(late[..., 3:]).sum()) > 0


def test_rotation_yields_to_a_plan_that_turns_the_other_way():
    quarter = [0., 0., np.pi/2]
    g = guidance(position_strength=0., rotation_strength=2., rotation_target_euler=quarter, rotation_yield=np.radians(5.))
    scale = float(g['action_scale'][5]*g['controller_scale'][5])          # radians per normalized action unit

    def plan(total_deg):                                                   # a chunk that turns `total_deg` about z in all
        return jnp.zeros((1, H, 32)).at[..., 5].set(np.radians(total_deg)/H/scale)

    with_plan = waypoint_shift(plan(+30), 0.5, -0.1, g)
    still = waypoint_shift(plan(0), 0.5, -0.1, g)
    slightly_against = waypoint_shift(plan(-3), 0.5, -0.1, g)
    against = waypoint_shift(plan(-30), 0.5, -0.1, g)
    assert float(with_plan[..., 5].sum()) > 0 and float(still[..., 5].sum()) > 0 and float(slightly_against[..., 5].sum()) > 0
    np.testing.assert_array_equal(against, 0)
    # without the setting the rotation term opposes the plan
    assert float(waypoint_shift(plan(-30), 0.5, -0.1, guidance(position_strength=0., rotation_strength=2.,
                                                                rotation_target_euler=quarter))[..., 5].sum()) > 0


def test_shift_is_spread_over_the_steps_and_preserved_on_average():
    g = guidance()
    clean = jnp.zeros((1, H, 32))
    change, shift = clean_action_shift(clean, 0.1, -0.1, g)
    moved = relative_path(clean+change, g)-relative_path(clean, g)
    ratio = np.asarray(moved[0, :, 0]/shift[0, :, 0])
    np.testing.assert_allclose(ratio.mean(), 1.0, rtol=1e-4)     # same displacement on average
    assert np.all(np.diff(ratio) > 0) and ratio[0] < 0.3           # no single step carries it
    assert float(jnp.abs(change[0, 0, 0])) == pytest.approx(float(jnp.abs(change[0, :, 0]).max()))
    np.testing.assert_array_equal(change[..., 6:], 0)              # gripper and padding untouched
    np.testing.assert_array_equal(change[..., 3:6], 0)             # no rotation target, no rotation change


def test_a_step_is_changed_by_at_most_one_full_scale_controller_step():
    g = guidance(position_strength=1e4)
    steps = actions_for_shift(waypoint_shift(jnp.zeros((1, H, 32)), 1.0, -0.1, g), g)
    assert float(jnp.linalg.norm(steps[..., :3], axis=-1).max()) <= 0.02*(1+1e-5)


def _plan(step_m, axis=0):
    """A chunk whose every step moves `step_m` metres along one axis."""
    g = guidance()
    scale = float(g['action_scale'][axis]*g['controller_scale'][axis])
    return jnp.zeros((1, H, 32)).at[..., axis].set(step_m/scale)


def test_override_pulls_only_a_plan_that_does_not_head_down_the_cost_map():
    # cost falls toward +x. The early-step pull is off at t = 0.1; only the override can act.
    g = guidance(time_floor=.5, override_strength=8.)
    toward = waypoint_shift(_plan(+0.005), 0.1, -0.1, g)       # 5 cm of progress over the chunk
    np.testing.assert_array_equal(toward, 0)
    for step in (0.0, -0.005):                                   # hovering, retreating: full override
        away = waypoint_shift(_plan(step), 0.1, -0.1, g)
        np.testing.assert_allclose(away[0, :, 0], 8*0.1*0.11, rtol=1e-4)
    half = waypoint_shift(_plan(+0.001), 0.1, -0.1, g)          # 1 cm of the 2 cm asked: half the override
    np.testing.assert_allclose(half[0, :, 0], 0.5*8*0.1*0.11, rtol=1e-3)
    sideways = waypoint_shift(_plan(0.005, axis=1), 0.1, -0.1, g)
    np.testing.assert_allclose(sideways[0, :, 0], 8*0.1*0.11, rtol=1e-4)
    # off by default, and it adds to the early-step pull where both act
    np.testing.assert_array_equal(waypoint_shift(_plan(-0.005), 0.1, -0.1, guidance(time_floor=.5)), 0)
    both = waypoint_shift(_plan(-0.005), 1.0, -0.1, g)
    np.testing.assert_allclose(both[0, 0, 0], (1+8)*(0.1+0.9*0.9)*0.11, rtol=1e-4)


def test_override_leaves_a_path_that_reaches_the_target_alone():
    # cost = |x - 0.5|: a chunk at x = 0.44 that moves sideways comes within 6 cm of the target
    near = guidance(time_floor=.5, override_strength=8., override_reach=.07, anchor=[.44, 0, 0, 0, 0, 0])
    sideways = _plan(0.005, axis=1)
    np.testing.assert_array_equal(waypoint_shift(sideways, 0.1, -0.1, near), 0)
    far = guidance(time_floor=.5, override_strength=8., override_reach=.07, anchor=[.2, 0, 0, 0, 0, 0])
    assert float(waypoint_shift(sideways, 0.1, -0.1, far)[0, 0, 0]) > 0       # 30 cm away and not closing: override


def test_result_bound_reverses_a_step_that_goes_the_other_way():
    quarter = [0., 0., np.pi/2]
    strong = dict(position_strength=0., rotation_strength=1e4, rotation_target_euler=quarter)
    scale = guidance()['action_scale'][5]*guidance()['controller_scale'][5]
    against = jnp.zeros((1, H, 32)).at[..., 5].set(-0.045/scale)        # turning right at 0.9 of full scale
    by_change, _ = clean_action_shift(against, 0.1, -0.1, guidance(**strong))
    by_result, _ = clean_action_shift(against, 0.1, -0.1, guidance(bound_result=1., **strong))
    guided = lambda change: np.asarray((against+change)[0, :5, 5]*scale)   # noqa: E731
    np.testing.assert_allclose(guided(by_change), 0.005, atol=1e-5)       # cancelled: one step of change
    np.testing.assert_allclose(guided(by_result), 0.05, atol=1e-5)        # reversed: a full step the planned way
    # a step that already goes the planned way is not made longer than one controller step
    along = jnp.zeros((1, H, 32)).at[..., 5].set(0.045/scale)
    change, _ = clean_action_shift(along, 0.1, -0.1, guidance(bound_result=1., **strong))
    np.testing.assert_allclose(np.asarray((along+change)[0, :5, 5]*scale), 0.05, atol=1e-5)
    # nothing wanted, nothing changed, also for a policy step longer than the bound
    long_step = jnp.zeros((1, H, 32)).at[..., 0].set(3.0)
    change, _ = clean_action_shift(long_step, 0.1, -0.1, guidance(bound_result=1., position_strength=0.))
    np.testing.assert_array_equal(change, 0)


def test_zero_strength_is_exactly_zero():
    change, _ = clean_action_shift(jnp.full((1, H, 32), .1), 0.5, -0.1, guidance(position_strength=0.))
    np.testing.assert_array_equal(change, 0)


def test_rotation_convention_matches_the_diffuser_actor_encoding():
    angles = np.array([[.3, -.2, .7], [3., .1, -2.5], [0., 0., np.pi/2]], dtype=np.float32)
    np.testing.assert_allclose(relative_rotation_matrix(jnp.asarray(angles)),
                               Rotation.from_euler('XYZ', angles).as_matrix(), atol=1e-6)
    np.testing.assert_allclose(relative_rotation_matrix(jnp.array([0., 0., np.pi/2])),
                               Rotation.from_euler('z', 90, degrees=True).as_matrix(), atol=1e-6)


@pytest.mark.parametrize('turn', [np.pi/2, -np.pi/2])
def test_rotation_target_is_a_relative_turn_about_z(turn):
    g = guidance(position_strength=0., rotation_strength=2., rotation_target_euler=[0., 0., turn])
    clean = jnp.zeros((1, H, 32))
    shift = waypoint_shift(clean, 0.1, -0.1, g)
    # yaw only, in the direction of the turn, growing along the horizon with the ramp
    np.testing.assert_array_equal(shift[..., :5], 0)
    yaw = np.asarray(shift[0, :, 5])
    assert np.all(np.sign(yaw[1:]) == np.sign(turn)) and np.all(np.diff(np.abs(yaw)) > 0)
    np.testing.assert_allclose(yaw[-1], 2.*0.1*float(g['rotation_ramp'][-1])*turn, rtol=1e-4)
    change, _ = clean_action_shift(clean, 0.1, -0.1, g)
    assert np.all(np.sign(np.asarray(change[0, :, 5])) == np.sign(turn))
    np.testing.assert_array_equal(change[..., :5], 0)
    np.testing.assert_array_equal(change[..., 6:], 0)


def test_rotation_fades_once_the_turn_is_done():
    g = guidance(position_strength=0., rotation_strength=2., rotation_target_euler=[0., 0., .4])
    # every step already turns so that the middle waypoint has reached the target
    done = jnp.zeros((1, H, 32)).at[0, :, 5].set(.4/((H//2+1)*.05*1.))
    far = waypoint_shift(jnp.zeros((1, H, 32)), 0.1, -0.1, g)
    near = waypoint_shift(done, 0.1, -0.1, g)
    assert abs(float(near[0, 2, 5])) < 0.2*abs(float(far[0, 2, 5]))


def test_inverse_normalization_matches_openpi_transform():
    from openpi.transforms import NormStats, Unnormalize
    q01, q99 = np.arange(7, dtype=np.float32)*.01-.9, np.arange(7, dtype=np.float32)*.02+1.1
    stats = NormStats(mean=np.zeros(7), std=np.ones(7), q01=q01, q99=q99)
    action = np.random.default_rng(42).normal(size=(4, 7)).astype(np.float32)
    expected = Unnormalize({'actions': stats}, use_quantiles=True)({'actions': action})['actions']
    g = guidance(action_scale=(q99[:6]-q01[:6]+1e-6)/2, action_offset=q01[:6]+(q99[:6]-q01[:6]+1e-6)/2)
    expected_path = np.cumsum(expected[:, :6]*np.asarray(g['controller_scale']), axis=0)
    np.testing.assert_allclose(relative_path(jnp.asarray(action), g), expected_path, atol=1e-6)


def test_payload_validation():
    g = {k: np.asarray(v) for k, v in guidance().items() if k not in ('action_scale', 'action_offset')}
    g['protocol'] = 'calvin_waypoint_shift_v1'
    validate_guidance(g, H)
    with pytest.raises(ValueError, match='world/controller'):
        validate_guidance({**g, 'controller_scale': np.zeros(6)}, H)
    with pytest.raises(ValueError, match='nonnegative'):
        validate_guidance({**g, 'position_strength': np.float32(-1)}, H)
    with pytest.raises(ValueError, match='protocol'):
        validate_guidance({**g, 'protocol': 'calvin_spatial_flow_v1'}, H)
    with pytest.raises(ValueError, match='fields'):
        validate_guidance({k: v for k, v in g.items() if k != 'position_gain'}, H)
