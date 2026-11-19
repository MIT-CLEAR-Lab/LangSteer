"""Spatial guidance for OpenPI's rectified-flow action sampler.

The guidance used with 3D Diffuser Actor adds sqrt(abar)/sqrt(1-abar) * g to the
predicted noise, which is the same as moving the clean-trajectory estimate by
-eta_t * g at every denoising step. This module applies that update to a flow
sampler. OpenPI uses x_t = (1-t)*action + t*noise and integrates t from 1 to 0,
so the clean estimate is x_t - t*v, and moving it by d while x_t stays fixed
means velocity -= d/t.

The move is defined in world units, so a strength means the same thing for every
policy: a predicted waypoint is displaced by -eta_t * position_gain * dC/dp
metres, and a wrist rotation by eta_t * ramp_j * (remaining turn) radians.
pi0.5 predicts relative controller steps, so the displacement wanted for the
predicted path is carried back to the actions through the cumulative-sum map
(its transpose, normalised: see `actions_for_shift`).

Two settings adapt the update to a policy whose plan barely varies between
noise samples (pi0.5). `time_floor` limits the pull to the early Euler steps, so
it chooses between the policy's own modes instead of pushing on a settled plan.
`rotation_yield` switches the rotation term off while the plan turns the other
way. Both are off at 0; conf/steering/voxposer.yaml holds the values in use.

Two further settings give the guidance authority over a policy that disagrees
with the plan (off at 0). `override_strength` adds a pull on
every Euler step, weighted by how little the policy's own predicted path
descends the cost map, so it is zero for a plan that already heads to the stage
target. `bound_result` bounds the guided controller step instead of the change
made to it, so a step that goes the other way can be reversed, not only
cancelled.

Only the six pose dimensions are changed. Gripper and padded action dimensions
are untouched, no network gradient is taken, and no weight is modified.
This module imports no LangSteer or PyTorch code; it runs in the OpenPI venv.
"""
import einops
import jax
import jax.numpy as jnp

from openpi.models import model as model_lib
from openpi.models.pi0 import make_attn_mask

PROTOCOL = "calvin_waypoint_shift_v1"


def interpolate_cost(grid, points, lower, upper):
    """Trilinear voxel lookup; endpoints match the map's world bounds."""
    size = jnp.asarray(grid.shape, dtype=points.dtype)
    coords = jnp.clip((points-lower)/(upper-lower)*(size-1), 0, size-1)
    lo = jnp.minimum(jnp.floor(coords).astype(jnp.int32), jnp.asarray(grid.shape)-2)
    frac = coords-lo
    result = jnp.zeros(points.shape[:-1], dtype=points.dtype)
    for x in (0, 1):
        for y in (0, 1):
            for z in (0, 1):
                corner = jnp.array([x, y, z])
                weight = jnp.prod(jnp.where(corner, frac, 1-frac), axis=-1)
                idx = lo+corner
                result += weight*grid[idx[..., 0], idx[..., 1], idx[..., 2]]
    return result


def relative_rotation_matrix(euler):
    """Rx @ Ry @ Rz: how 3D Diffuser Actor encodes a CALVIN Euler triple."""
    a, b, c = jnp.moveaxis(euler, -1, 0)
    ca, sa, cb, sb, cc, sc = jnp.cos(a), jnp.sin(a), jnp.cos(b), jnp.sin(b), jnp.cos(c), jnp.sin(c)
    return jnp.stack([
        jnp.stack([cb*cc, -cb*sc, sb], -1),
        jnp.stack([ca*sc+sa*sb*cc, ca*cc-sa*sb*sc, -sa*cb], -1),
        jnp.stack([sa*sc-ca*sb*cc, sa*cc+ca*sb*sc, ca*cb], -1),
    ], -2)


def wrap_angle(angle):
    return (angle+jnp.pi) % (2*jnp.pi)-jnp.pi


def relative_path(actions, guidance):
    """Cumulative controller displacement after each action: metres and radians.

    CALVIN adds each relative action, scaled, to the controller's target pose
    (positions and Euler angles alike), so the path is a cumulative sum.
    """
    physical = (actions[..., :6]*guidance['action_scale']+guidance['action_offset'])*guidance['controller_scale']
    return jnp.cumsum(physical, axis=-2)


def guided_step(time, dt, time_floor):
    """1 for an Euler step that starts above `time_floor`, 0 for a later one.

    Time runs from 1 (noise) to 0 (action). With `time_floor` 0 every step is
    guided. With a floor, only the early, noisy steps are: the policy's own
    remaining steps then integrate from the shifted sample, so the shift survives
    in proportion to how undecided the policy is (it picks between the policy's
    own modes) instead of being added to whatever the policy predicts.
    """
    return (time > time_floor-dt/4).astype(jnp.float32)


def plan_disagreement(points, guidance):
    """1 where the predicted path does not descend the cost map, 0 where it does. Shape (B,).

    The descent is the cost at the controller's current target minus the cost at
    the end of the predicted path, divided by the slope at the current target:
    metres of progress toward the stage target that the policy's own plan makes.
    A plan that advances `override_progress` metres or more counts as agreeing.
    """
    def cost(p):
        return interpolate_cost(guidance['cost_map'], p, guidance['workspace_min'], guidance['workspace_max'])

    start = jnp.clip(guidance['anchor'][:3], guidance['workspace_min'], guidance['workspace_max'])
    slope = jnp.linalg.norm(jax.grad(cost)(start))
    progress = (cost(start)-cost(points[..., -1, :]))/jnp.maximum(slope, 1e-6)
    heading_elsewhere = jnp.clip(1-progress/jnp.maximum(guidance['override_progress'], 1e-6), 0, 1)
    # A path that already comes within `override_reach` of the stage target is the policy's own
    # approach to the right place (from the side, for a push): leave it alone. The cost maps are
    # distance cones, so cost divided by slope is the distance to the target. Off at 0.
    reach = guidance['override_reach']
    closest = jnp.min(cost(points), axis=-1)/jnp.maximum(slope, 1e-6)
    away = jnp.where(reach > 0, jnp.clip((closest-reach)/jnp.maximum(0.5*reach, 1e-6), 0, 1), 1.0)
    return heading_elsewhere*away


def waypoint_shift(clean, time, dt, guidance):
    """Displacement wanted for every predicted waypoint, shape (B, H, 6)."""
    path = relative_path(clean, guidance)
    points = jnp.clip(guidance['anchor'][:3]+path[..., :3], guidance['workspace_min'], guidance['workspace_max'])
    gradient = jax.grad(lambda p: interpolate_cost(
        guidance['cost_map'], p, guidance['workspace_min'], guidance['workspace_max']).sum())(points)
    # Same ramp as the DDPM branch. Its argument is the noise level the sample has
    # once this step is taken, so the last step uses min_timestep_scale.
    noise_after = jnp.maximum(time+dt, 0.0)
    ramp = jnp.where(guidance['timestep_scaling'],
                     guidance['min_timestep_scale']+(1-guidance['min_timestep_scale'])*noise_after, 1.0)
    strength = (guidance['position_strength']*guided_step(time, dt, guidance['time_floor'])
                + guidance['override_strength']*plan_disagreement(points, guidance))
    position = -strength[..., None, None]*ramp*guidance['position_gain']*gradient

    # The rotation target is the turn itself, relative to the pose at this policy
    # call, as an Euler change (see pi05_guidance/payload.py).
    turned = path[..., 3:6]
    remaining = wrap_angle(guidance['rotation_target_euler']-turned)
    middle = turned[..., turned.shape[-2]//2, :]
    distance = jnp.linalg.norm(
        relative_rotation_matrix(middle)-relative_rotation_matrix(guidance['rotation_target_euler']), axis=(-2, -1))
    full, near, floor, enabled = guidance['rotation_alignment']
    alignment = floor+(1-floor)*jnp.clip((distance-near)/jnp.maximum(full-near, 1e-6), 0, 1)
    alignment = jnp.where(enabled, alignment, 1.0)[..., None, None]
    # Yield to a policy that is turning the other way. `rotation_yield` is how far
    # (radians, over the whole predicted chunk, about the target's axis) the plan may
    # turn against the target before the rotation term is switched off; 0 never yields.
    target = guidance['rotation_target_euler']
    against = -jnp.sum(turned[..., -1, :]*target, axis=-1)/jnp.maximum(jnp.linalg.norm(target), 1e-6)
    keeps = jnp.where(guidance['rotation_yield'] > 0, against < guidance['rotation_yield'], True)
    keeps = keeps.astype(jnp.float32)[..., None, None]
    rotation = (guidance['rotation_strength']*ramp*guided_step(time, dt, guidance['rotation_time_floor'])
                * keeps*alignment*guidance['rotation_ramp'][:, None]*remaining)
    return jnp.concatenate([position, rotation], axis=-1)


def actions_for_shift(shift, guidance, own=None):
    """Change of the relative controller steps that carries a waypoint displacement.

    Step k moves every waypoint from k onwards, so the gradient of the summed
    waypoint cost with respect to step k is the sum of the displacements wanted
    for waypoints k..H. That sum is divided by (H+1)(2H+1)/6, the factor by which
    the cumulative-sum map amplifies a displacement common to all waypoints, so
    that such a displacement moves the path by the same amount on average.
    No single step carries the whole correction, and each step is changed by at
    most one full-scale controller step. With `bound_result` (and the policy's
    own controller steps given as `own`) the bound is on the guided step instead:
    it stays within one full-scale controller step, or within the length of the
    policy's own step if that is longer, so a step in the opposite direction can
    be reversed.
    """
    horizon = shift.shape[-2]
    pulled = jnp.flip(jnp.cumsum(jnp.flip(shift, axis=-2), axis=-2), axis=-2)/((horizon+1)*(2*horizon+1)/6)
    limit = guidance['controller_scale']

    def bounded(part, cap):
        norm = jnp.linalg.norm(part, axis=-1, keepdims=True)
        return part*jnp.minimum(1.0, cap/jnp.maximum(norm, 1e-12))

    def kept(part, mine, cap):
        cap = jnp.maximum(cap, jnp.linalg.norm(mine, axis=-1, keepdims=True))
        guided = mine+part
        norm = jnp.linalg.norm(guided, axis=-1, keepdims=True)
        change = guided*jnp.minimum(1.0, cap/jnp.maximum(norm, 1e-12))-mine
        return jnp.where(jnp.linalg.norm(part, axis=-1, keepdims=True) > 0, change, 0.0)

    by_change = jnp.concatenate([bounded(pulled[..., :3], limit[0]), bounded(pulled[..., 3:6], limit[3])], axis=-1)
    if own is None:
        return by_change
    by_result = jnp.concatenate([kept(pulled[..., :3], own[..., :3], limit[0]),
                                 kept(pulled[..., 3:6], own[..., 3:6], limit[3])], axis=-1)
    return jnp.where(guidance['bound_result'] > 0, by_result, by_change)


def clean_action_shift(clean, time, dt, guidance):
    """Shift of the clean normalized actions, and the waypoint displacement behind it."""
    shift = waypoint_shift(clean, time, dt, guidance)
    own = (clean[..., :6]*guidance['action_scale']+guidance['action_offset'])*guidance['controller_scale']
    steps = actions_for_shift(shift, guidance, own)
    normalized = steps/(guidance['action_scale']*guidance['controller_scale'])
    return jnp.zeros_like(clean).at[..., :6].set(normalized), shift


def sample_actions_guided(self, rng, observation, *, guidance, num_steps=10, noise=None):
    """OpenPI Pi0.sample_actions with the clean estimate shifted in every Euler step.

    Prefix caching, masks, time convention and base velocity are copied from
    OpenPI commit 15a9616a00943ada6c20a0f158e3adb39df2ccac (Apache-2.0, see
    THIRD_PARTY_NOTICES.md) and modified: the guidance correction is added to
    the velocity in every Euler step, and steering diagnostics are returned.
    The server keeps the native sampler for all unsteered and zero-strength
    requests.
    """
    observation = model_lib.preprocess_observation(None, observation, train=False)
    dt = -1.0/num_steps
    batch_size = observation.state.shape[0]
    if noise is None:
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1)-1
    _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

    def step(carry):
        x_t, time, moved, turned, active = carry
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, jnp.broadcast_to(time, batch_size))
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn = einops.repeat(prefix_mask, 'b p -> b s p', s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn, suffix_attn_mask], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None]+jnp.cumsum(suffix_mask, axis=-1)-1
        (_, suffix_out), _ = self.PaliGemma.llm([None, suffix_tokens], mask=full_attn_mask,
            positions=positions, kv_cache=kv_cache, adarms_cond=[None, adarms_cond])
        velocity = self.action_out_proj(suffix_out[:, -self.action_horizon:])
        change, shift = clean_action_shift(x_t-time*velocity, time, dt, guidance)
        correction = -change/jnp.maximum(time, 1e-6)
        position = jnp.linalg.norm(shift[..., :3], axis=-1).mean()
        rotation = jnp.linalg.norm(shift[..., 3:6], axis=-1).mean()
        return (x_t+dt*(velocity+correction), time+dt, moved+position, turned+rotation,
                active+(jnp.linalg.norm(change) > 0).astype(jnp.int32))

    zero = jnp.asarray(0.0)
    actions, _, moved, turned, active = jax.lax.while_loop(
        lambda carry: carry[1] >= -dt/2, step, (noise, jnp.asarray(1.0), zero, zero, jnp.asarray(0, dtype=jnp.int32)))
    return actions, {'waypoint_shift_m_sum': moved, 'waypoint_turn_rad_sum': turned, 'nonzero_guidance_steps': active}


def sample_actions_batched(self, rng, observation, *, guidance, noise, num_steps=10):
    """`sample_actions_guided` for a batch of independent requests.

    Every leaf of `guidance` and `noise` carries a leading batch dimension: each
    request has its own cost map, controller state, strengths and noise. A request
    without steering is a row of zero strengths, for which the shift is exactly zero.
    Returns the actions and per-request diagnostics.
    """
    observation = model_lib.preprocess_observation(None, observation, train=False)
    dt = -1.0/num_steps
    batch_size = observation.state.shape[0]
    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1)-1
    _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

    def shift_one(clean, time, row):
        change, shift = clean_action_shift(clean[None], time, dt, row)
        return change[0], shift[0]

    def velocity_at(x_t, time):
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, jnp.broadcast_to(time, batch_size))
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn = einops.repeat(prefix_mask, 'b p -> b s p', s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn, suffix_attn_mask], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None]+jnp.cumsum(suffix_mask, axis=-1)-1
        (_, suffix_out), _ = self.PaliGemma.llm([None, suffix_tokens], mask=full_attn_mask,
            positions=positions, kv_cache=kv_cache, adarms_cond=[None, adarms_cond])
        velocity = self.action_out_proj(suffix_out[:, -self.action_horizon:])
        change, shift = jax.vmap(lambda c, row: shift_one(c, time, row))(x_t-time*velocity, guidance)
        return velocity, change, shift

    def step(carry):
        x_t, time, moved, turned, active = carry
        velocity, change, shift = velocity_at(x_t, time)
        correction = -change/jnp.maximum(time, 1e-6)
        position = jnp.linalg.norm(shift[..., :3], axis=-1).mean(axis=-1)
        rotation = jnp.linalg.norm(shift[..., 3:6], axis=-1).mean(axis=-1)
        acted = (jnp.linalg.norm(change.reshape(batch_size, -1), axis=-1) > 0).astype(jnp.int32)
        return x_t+dt*(velocity+correction), time+dt, moved+position, turned+rotation, active+acted

    zeros = jnp.zeros(batch_size)
    actions, _, moved, turned, active = jax.lax.while_loop(
        lambda carry: carry[1] >= -dt/2, step,
        (noise, jnp.asarray(1.0), zeros, zeros, jnp.zeros(batch_size, dtype=jnp.int32)))
    return actions, {'waypoint_shift_m_sum': moved, 'waypoint_turn_rad_sum': turned, 'nonzero_guidance_steps': active}
