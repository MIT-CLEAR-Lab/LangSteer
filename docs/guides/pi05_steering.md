# Steering pi0.5

pi0.5 (OpenPI) samples an action chunk with a rectified-flow sampler. LangSteer steers it
with the same LLM-written cost maps and the same stage logic as 3D Diffuser Actor (3DDA).
This guide explains how the 3DDA guidance carries over to a flow sampler, and what the
`steering.flow_*` settings in `conf/steering/voxposer.yaml` do. Those settings are the
only part of the steering configuration that is specific to pi0.5. Strengths, distance
decay, stage logic, prompts and model are shared with 3DDA, and nothing is set per task
or per axis.

## 1. The update

3DDA's guidance adds `sqrt(abar)/sqrt(1-abar) * g` to the predicted noise. This moves the
clean-trajectory estimate by `-eta_t * g` while the noisy sample stays fixed, with
`eta_t = strength * timestep ramp * adaptive scalers`. A rectified flow has the same
update. With `x_t = (1-t)*action + t*noise` (time runs from 1, noise, to 0, action), the
clean estimate is `x_t - t*v`. Moving it by `d` while `x_t` stays fixed means `v -= d/t`.
No gradient is taken through the network: the cost gradient is evaluated at the clean
estimate, as in 3DDA.

`pi05_guidance/flow.py` applies the update with three conversions, so that the guidance is
the same rather than retuned:

- **World units.** A predicted waypoint is moved by `-eta_t * position_gain * dC/dp`
  metres. `position_gain` is the squared half-range of the 3DDA position normalisation
  (`steering.world_step_bounds`), which is what strength 1 means for 3DDA, written in
  metres.
- **Through the action map.** pi0.5 predicts relative controller steps; waypoint `j` is
  the sum of steps `1..j`. The displacement wanted for the waypoints is carried back to
  the steps with the transpose of that map. Step `k` receives the sum over waypoints
  `k..H`, divided by `(H+1)(2H+1)/6`, so that a displacement common to all waypoints
  moves the path by that amount on average. Each step is changed by at most one
  full-scale controller step.
- **Time in control steps.** 3DDA executes 20 control steps per policy call and pi0.5
  executes 5. Settings that describe a duration (the rotation ramp, the loop-back dwell)
  are in control steps (`steering.steps_per_policy_call`, `steering.reference_horizon`).

Rotation targets are relative turns (`rotation_about_axis('z', 90)` is a quarter turn to
the left). They are exported as relative Euler changes in the convention 3DDA uses for a
CALVIN Euler triple (`pi05_guidance/payload.py`).

## 2. Why the flow sampler needs its own schedule

pi0.5's plan hardly depends on the sampler's noise: plans drawn from different noise for
the same observation differ by a few millimetres. Guidance applied on every sampler step
therefore acts as a fixed push added to the policy's plan, because the shift made on the
last steps goes straight into the action. Guidance applied only on the early, noisy
steps is integrated by the policy's own remaining steps. It survives where the policy is
undecided and largely disappears where the policy has settled on a plan.

The settings:

| Setting | Effect |
|---|---|
| `flow_time_floor` | Position pull only on the Euler steps that start above this time (0: every step). |
| `flow_gain` | Multiplies the position strength on those steps. |
| `flow_rotation_time_floor` | The same window for the rotation term. |
| `flow_rotation_when` | `holding`: rotation term only while the gripper is closed on an object (the grasp gate's width window); `always`. |
| `flow_rotation_yield_deg` | Rotation term off while the predicted chunk turns more than this against the target (0: never). |
| `flow_rotation_gain` | Multiplies the rotation strength in the flow sampler. |
| `flow_override_gain` | A pull on every Euler step, weighted by how little the policy's own predicted path moves toward the stage target. It is zero for a path that advances `flow_override_progress_cm` or more, and full for a path that hovers or heads elsewhere (0: off). |
| `flow_override_reach_cm` | A predicted path that already comes within this distance of the stage target is the policy's own approach to the right place (for example a push from the side) and is left alone (0: off). |
| `flow_override_when` | `approach`: the override acts only in a grasp stage and only until the gripper closes on an object, so it supplies which object to take and leaves grasping, carrying, pushing and turning to the policy. `always`. |
| `flow_override_near_cm`, `flow_override_full_cm` | The override fades out as the gripper comes within these distances of the stage target, so that the policy makes the final approach and the grasp itself. |
| `flow_step_bound` | `change`: each controller step is changed by at most one full-scale step. `result`: the guided step is at most one full-scale step (or the policy's own step if longer), so a step heading the other way can be reversed, not only cancelled. |

The values in use are those in `conf/steering/voxposer.yaml`.

## 3. Serving

The model runs in the OpenPI environment behind `python -m pi05_guidance.server` (see the
README for the command). The LangSteer client (`policies/pi05_guided.py`) sends the
observation, a per-request noise seed and, in steered conditions, the active stage's
guidance payload. The client checks that the server holds the checkpoint it expects.

- Requests without guidance, or with all strengths zero, go through OpenPI's native
  sampler unchanged.
- At start-up the server checks that a repeated seed gives bit-identical actions, that
  zero guidance gives exactly the native actions, and that guidance moves the path in
  the expected direction (`--validate-fixture`; the fixture comes from
  `scripts/prepare_pi05_guidance_fixture.py`).
- `--max-batch N` serves several evaluation workers' requests in one model call. Each
  request keeps its own seed, cost map and strengths. Batched actions agree with the
  one-at-a-time path to within about 0.005 action units, so episodes are not
  bit-identical to it. The server compiles every batch size at start-up.

## 4. Tests

- `tests/test_pi05_flow.py` (OpenPI environment, `JAX_PLATFORMS=cpu`): the update's
  magnitude in world units, the timestep ramp, the time windows, the action-map
  pull-back and its per-step bound, rotation conventions, yielding, the override and the
  result bound, and payload validation.
- `tests/test_pi05_guided_integration.py` (LangSteer environment): what the client
  exports for a stage (anchor, scales, decay, rotation target, flow settings).
