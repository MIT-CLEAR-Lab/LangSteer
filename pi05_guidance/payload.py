"""Export the active LangSteer stage for the JAX pi0.5 sampler (client side).

Everything the server needs to guide one action chunk: the stage's cost map,
the controller state the relative actions accumulate on, and the strengths.
The adaptive scalers that depend on the scene (distance decay, step decay) are
evaluated here, exactly as the DDPM branch evaluates them once per policy call.
"""
import numpy as np

PROTOCOL = 'calvin_waypoint_shift_v1'


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def relative_euler(matrix):
    """Euler triple (a, b, c) with matrix = Rx(a) @ Ry(b) @ Rz(c).

    3D Diffuser Actor encodes a CALVIN relative Euler step this way, and a
    stage's rotation target is the turn itself (`rotation_about_axis('z', 90)`
    is a quarter turn to the left), never an absolute orientation. So the target
    becomes the Euler change to add to the pose at this policy call.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    b = np.arcsin(np.clip(matrix[0, 2], -1.0, 1.0))
    a = np.arctan2(-matrix[1, 2], matrix[2, 2])
    c = np.arctan2(-matrix[0, 1], matrix[0, 0])
    return np.array([a, b, c], dtype=np.float32)


def rotation_matrix_from_6d(six):
    """First two matrix columns, Gram-Schmidt orthonormalised (ortho6d convention)."""
    six = _array(six).reshape(6)
    first = six[:3]/max(np.linalg.norm(six[:3]), 1e-8)
    third = np.cross(first, six[3:])
    third /= max(np.linalg.norm(third), 1e-8)
    second = np.cross(third, first)
    return np.stack([first, second, third], axis=-1)


def horizon_ramp(horizon, reference_horizon, floor, maximum):
    """Rotation ramp for a policy that predicts `horizon` control steps.

    The ramp is a function of how many control steps ahead a waypoint lies:
    floor at the first step, `maximum` at step `reference_horizon`, quadratic in
    between. A shorter-horizon policy reads the first part of the same ramp
    instead of compressing all of it into its own horizon.
    """
    ahead = np.minimum(np.arange(horizon)/max(reference_horizon-1, 1), 1.0)
    return np.asarray(floor+(maximum-floor)*ahead**2, dtype=np.float32)


def export_guidance(steering, obs, horizon):
    context = obs.action_context
    if not context or context.get('protocol') != 'calvin_target_relative_v1' or not context.get('use_target_pose'):
        raise ValueError('π0.5 steering requires the current CALVIN controller target and cumulative relative actions')
    stage = steering._stage_manager.current()
    if stage.value_map is None or stage.value_map._cost_map is None:
        raise RuntimeError('The steered condition has no active spatial cost map')
    cfg = steering.cfg
    position_scale = 1.0
    # The pull that takes over from a policy heading elsewhere (flow_override_gain) is
    # released between flow_override_full_cm and flow_override_near_cm from the stage target.
    override_scale = 1.0
    if stage.stage_target_world is not None:
        distance = np.linalg.norm(np.asarray(obs.proprio[:3])-stage.stage_target_world)
        near, full = float(cfg.get('flow_override_near_cm', 4.0))/100, float(cfg.get('flow_override_full_cm', 12.0))/100
        override_scale = float(np.clip((distance-near)/max(full-near, 1e-6), 0, 1))
    if cfg.get('flow_override_when', 'always') == 'approach':
        # Only while approaching an object to grasp it: a grasp stage, gripper not yet closed on an object.
        width = float(obs.ee_pose[6])
        holding = float(cfg.get('grasp_min_width', 0.02)) < width < float(cfg.get('grasp_max_width', 0.07))
        if stage.primitive != 'grasp' or holding:
            override_scale = 0.0
    if cfg.get('use_distance_scaling', False) and stage.stage_target_world is not None:
        distance = np.linalg.norm(np.asarray(obs.proprio[:3])-stage.stage_target_world)
        full, near, floor = (float(cfg.get(k, default)) for k, default in [('distance_full', .12), ('distance_near', .04), ('distance_floor', .1)])
        position_scale *= floor+(1-floor)*float(np.clip((distance-near)/max(full-near, 1e-6), 0, 1))
    if cfg.get('use_step_scaling', False):
        full, decay, floor = float(cfg.get('step_full', 0)), float(cfg.get('step_decay', 80)), float(cfg.get('step_floor', .05))
        position_scale *= 1-(1-floor)*float(np.clip((stage.steps_in_stage-full)/max(decay, 1), 0, 1))
    turn = np.zeros(3, dtype=np.float32)
    rotation_strength = 0.0
    if stage.rotation_target_6d is not None:
        turn = relative_euler(rotation_matrix_from_6d(stage.rotation_target_6d))
        rotation_strength = float(cfg.get('guidance_strength_rot', steering.guidance_strength))*float(cfg.get('flow_rotation_gain', 1.0))
        if cfg.get('flow_rotation_when', 'always') == 'holding':
            # Turn only what is held: the same width window the grasp gate uses for "closed on an object".
            width = float(obs.ee_pose[6])
            if not float(cfg.get('grasp_min_width', 0.02)) < width < float(cfg.get('grasp_max_width', 0.07)):
                rotation_strength = 0.0
    bounds = np.asarray(cfg['world_step_bounds'], dtype=np.float32)
    value_map = stage.value_map
    return dict(
        protocol=PROTOCOL, cost_map=_array(value_map._cost_map),
        workspace_min=_array(value_map.workspace_bounds_min), workspace_max=_array(value_map.workspace_bounds_max),
        anchor=_array(context['anchor']), controller_scale=_array(context['scale']),
        position_strength=np.float32(steering.guidance_strength*position_scale*float(cfg.get('flow_gain', 1.0))),
        # What strength 1 means in world units: see conf/steering/voxposer.yaml.
        position_gain=((bounds[1]-bounds[0])/2)**2,
        rotation_strength=np.float32(rotation_strength), rotation_target_euler=turn,
        rotation_ramp=horizon_ramp(horizon, int(cfg['reference_horizon']),
                                   float(cfg.get('rot_horizon_floor', 0)), float(cfg.get('rot_horizon_alpha_max', .6))),
        rotation_alignment=np.asarray([cfg.get('rot_align_full', .5), cfg.get('rot_align_near', .1),
                                       cfg.get('rot_align_floor', .05), cfg.get('use_rot_alignment_scaling', True)], dtype=np.float32),
        timestep_scaling=np.bool_(cfg.get('use_timestep_scaling', True)),
        min_timestep_scale=np.float32(cfg.get('min_timestep_scale', .1)),
        # guide only the Euler steps that start above this time (0: every step)
        time_floor=np.float32(cfg.get('flow_time_floor', 0.0)),
        rotation_time_floor=np.float32(cfg.get('flow_rotation_time_floor', 0.0)),
        # switch the rotation term off when the plan turns this far against the target (0: never)
        rotation_yield=np.float32(np.radians(float(cfg.get('flow_rotation_yield_deg', 0.0)))),
        # pull on every step in proportion to how little the policy's own path descends the cost map
        override_strength=np.float32(steering.guidance_strength*override_scale*float(cfg.get('flow_override_gain', 0.0))),
        override_progress=np.float32(float(cfg.get('flow_override_progress_cm', 2.0))/100),
        # a predicted path that comes this close to the stage target is the policy's own approach (0: off)
        override_reach=np.float32(float(cfg.get('flow_override_reach_cm', 0.0))/100),
        # bound the guided controller step (result) instead of the change made to it
        bound_result=np.float32(cfg.get('flow_step_bound', 'change') == 'result'))
