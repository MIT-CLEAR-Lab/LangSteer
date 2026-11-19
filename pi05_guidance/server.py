"""Serve the unchanged pi0.5 overloaded checkpoint with optional spatial guidance.

Runs in the OpenPI venv (JAX and PyTorch cannot share a process):

    python -m pi05_guidance.server --checkpoint <dir> --training-config-dir training/common \
        --metadata-out <json> --validate-fixture <npz> --port 8003

Every request carries its own noise seed. Requests without guidance, or with
zero strengths, go through OpenPI's native sampler unchanged.
"""
import argparse
import asyncio
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
from types import MethodType

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as model_lib
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.shared import nnx_utils
from openpi.training import checkpoints

from pi05_guidance.flow import PROTOCOL, sample_actions_batched, sample_actions_guided


def validate_guidance(payload, horizon):
    payload = dict(payload)
    if payload.pop('protocol', None) != PROTOCOL:
        raise ValueError('Unsupported spatial-guidance protocol')
    shapes = {'workspace_min': (3,), 'workspace_max': (3,), 'anchor': (6,), 'controller_scale': (6,),
              'position_gain': (3,), 'rotation_target_euler': (3,), 'rotation_ramp': (horizon,), 'rotation_alignment': (4,)}
    scalar_keys = ['position_strength', 'rotation_strength', 'timestep_scaling', 'min_timestep_scale',
                   'time_floor', 'rotation_time_floor', 'rotation_yield',
                   'override_strength', 'override_progress', 'override_reach', 'bound_result']
    if set(payload) != set(shapes) | set(scalar_keys) | {'cost_map'}:
        raise ValueError('Missing or unexpected spatial-guidance fields')
    payload = {k: np.asarray(v, dtype=np.float32) for k, v in payload.items()}
    for key, shape in shapes.items():
        if payload[key].shape != shape:
            raise ValueError(f'Invalid {key} shape')
    for key in scalar_keys:
        if payload[key].shape != ():
            raise ValueError(f'Invalid scalar {key}')
    grid = payload['cost_map']
    if grid.ndim != 3 or min(grid.shape) < 2 or max(grid.shape) > 128:
        raise ValueError('Invalid spatial cost volume')
    if not all(np.isfinite(v).all() for v in payload.values()):
        raise ValueError('Non-finite spatial-guidance payload')
    if not np.all(payload['workspace_max'] > payload['workspace_min']) or not np.all(payload['controller_scale'] > 0):
        raise ValueError('Invalid world/controller scales')
    if np.any(payload['position_gain'] < 0) or payload['position_strength'] < 0 or payload['rotation_strength'] < 0:
        raise ValueError('Guidance strengths and gains must be nonnegative')
    if not 0 <= payload['min_timestep_scale'] <= 1 or not 0 <= payload['time_floor'] < 1 \
            or not 0 <= payload['rotation_time_floor'] < 1 or payload['rotation_yield'] < 0 \
            or payload['override_strength'] < 0 or payload['override_progress'] < 0 or payload['override_reach'] < 0:
        raise ValueError('Invalid guidance schedule')
    return payload


ZERO_DIAGNOSTICS = {'waypoint_shift_m_sum': 0.0, 'waypoint_turn_rad_sum': 0.0, 'nonzero_guidance_steps': 0}


class GuidedServerPolicy:
    def __init__(self, native, norm_stats, use_quantiles, metadata):
        self.native = native
        self.metadata = metadata
        self.guided_sample = nnx_utils.module_jit(MethodType(sample_actions_guided, native._model))
        stats = norm_stats['actions']
        if use_quantiles:
            self.scale = (np.asarray(stats.q99[:6])-np.asarray(stats.q01[:6])+1e-6)/2
            self.offset = np.asarray(stats.q01[:6])+self.scale
        else:
            self.scale, self.offset = np.asarray(stats.std[:6])+1e-6, np.asarray(stats.mean[:6])
        self.scale, self.offset = self.scale.astype(np.float32), self.offset.astype(np.float32)
        if not np.isfinite(self.scale).all() or not np.all(self.scale > 0) or not np.isfinite(self.offset).all():
            raise ValueError('Invalid checkpoint normalization statistics')

    def infer(self, obs):
        obs = dict(obs)
        seed = int(obs.pop('_sample_seed'))
        if not 0 <= seed < 2**32:
            raise ValueError('Invalid uint32 inference seed')
        payload = obs.pop('_steering', None)
        guidance = validate_guidance(payload, self.native._model.action_horizon) if payload is not None else None
        enabled = guidance is not None and any(guidance[k] > 0 for k in ('position_strength', 'rotation_strength', 'override_strength'))
        self.native._rng = jax.random.PRNGKey(np.uint32(seed))
        if not enabled:
            # Exact native path, including the same RNG split and transforms.
            result = self.native.infer(obs)
            diagnostics = dict(ZERO_DIAGNOSTICS)
        else:
            started = time.monotonic()
            inputs = self.native._input_transform(jax.tree.map(lambda x: x, obs))
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            _, sample_rng = jax.random.split(self.native._rng)
            guidance.update(action_scale=self.scale, action_offset=self.offset)
            actions, diagnostics = self.guided_sample(sample_rng, model_lib.Observation.from_dict(inputs),
                guidance=jax.tree.map(jnp.asarray, guidance), **self.native._sample_kwargs)
            actions = np.asarray(actions[0])
            result = self.native._output_transform({'state': np.asarray(inputs['state'][0]), 'actions': actions})
            diagnostics = {k: np.asarray(v).item() for k, v in diagnostics.items()}
            result['policy_timing'] = {'infer_ms': (time.monotonic()-started)*1000}
        if not np.isfinite(result['actions']).all() or not all(np.isfinite(v) for v in diagnostics.values()):
            raise RuntimeError('Non-finite model action or steering correction')
        result['steering_info'] = dict(protocol=PROTOCOL, requested=payload is not None, enabled=bool(enabled), **diagnostics)
        return result


class BatchedServerPolicy:
    """Serves several pending requests in one model call.

    Each request keeps its own noise seed, cost map and strengths, so a request's
    result does not depend on what it is batched with beyond floating-point
    rounding. It is not bit-identical to the sequential path; use that path
    (no --max-batch) when a run has to reproduce stored episodes exactly.
    """

    SIZES = (1, 2, 4, 8, 16)

    def __init__(self, sequential, max_batch):
        self.sequential = sequential
        self.native = sequential.native
        self.max_batch = max_batch
        self.sampler = nnx_utils.module_jit(MethodType(sample_actions_batched, self.native._model))
        self.horizon = self.native._model.action_horizon
        self.action_dim = self.native._model.action_dim
        self._off = None
        self.map_size = 100   # voxels per side of the cost maps the client sends

    def _row(self, guidance):
        """One request's guidance as arrays; no steering is a row of zero strengths."""
        if guidance is not None:
            return dict(guidance, action_scale=self.sequential.scale, action_offset=self.sequential.offset)
        if self._off is None:
            def zeros(*shape):
                return np.zeros(shape, dtype=np.float32)

            def scalar(value):
                return np.asarray(value, dtype=np.float32)

            # Zero strengths switch every term off; the rest only has to be well formed.
            self._off = dict(
                cost_map=zeros(self.map_size, self.map_size, self.map_size), workspace_min=zeros(3),
                workspace_max=np.ones(3, dtype=np.float32), anchor=zeros(6), controller_scale=np.ones(6, dtype=np.float32),
                position_gain=zeros(3), rotation_target_euler=zeros(3), rotation_ramp=zeros(self.horizon),
                rotation_alignment=zeros(4), position_strength=scalar(0), rotation_strength=scalar(0),
                timestep_scaling=scalar(0), min_timestep_scale=scalar(0),
                time_floor=scalar(0), rotation_time_floor=scalar(0), rotation_yield=scalar(0),
                override_strength=scalar(0), override_progress=scalar(0), override_reach=scalar(0), bound_result=scalar(0),
                action_scale=self.sequential.scale, action_offset=self.sequential.offset)
        return self._off

    def infer_batch(self, requests):
        prepared = []
        for obs in requests:
            obs = dict(obs)
            seed = int(obs.pop('_sample_seed'))
            if not 0 <= seed < 2**32:
                raise ValueError('Invalid uint32 inference seed')
            payload = obs.pop('_steering', None)
            guidance = validate_guidance(payload, self.horizon) if payload is not None else None
            inputs = self.native._input_transform(jax.tree.map(lambda x: x, obs))
            prepared.append((seed, payload is not None, guidance, inputs))
        rows = [self._row(item[2]) for item in prepared]
        size = next(s for s in self.SIZES if s >= len(prepared))
        order = list(range(len(prepared)))+[len(prepared)-1]*(size-len(prepared))
        inputs = jax.tree.map(lambda *xs: jnp.asarray(np.stack(xs)), *[prepared[i][3] for i in order])
        noise = jnp.stack([jax.random.normal(jax.random.split(jax.random.PRNGKey(np.uint32(prepared[i][0])))[1],
                                             (1, self.horizon, self.action_dim))[0] for i in order])
        guidance = {k: jnp.asarray(np.stack([rows[i][k] for i in order])) for k in rows[0]}
        actions, diagnostics = self.sampler(None, model_lib.Observation.from_dict(inputs), guidance=guidance,
                                            noise=noise, **self.native._sample_kwargs)
        actions = np.asarray(actions)
        diagnostics = {k: np.asarray(v) for k, v in diagnostics.items()}
        results = []
        for i, (seed, requested, guidance_i, inputs_i) in enumerate(prepared):
            result = self.native._output_transform({'state': np.asarray(inputs_i['state']), 'actions': actions[i]})
            info = {k: v[i].item() for k, v in diagnostics.items()}
            if not np.isfinite(result['actions']).all() or not all(np.isfinite(v) for v in info.values()):
                raise RuntimeError('Non-finite model action or steering correction')
            enabled = guidance_i is not None and any(guidance_i[k] > 0 for k in ('position_strength', 'rotation_strength', 'override_strength'))
            result['steering_info'] = dict(protocol=PROTOCOL, requested=requested, enabled=bool(enabled), **info)
            results.append(result)
        return results


class BatchingServer:
    """Websocket server (OpenPI client protocol) that groups pending requests into one model call."""

    def __init__(self, policy, host, port, metadata, max_wait_s=0.004):
        self.policy, self.host, self.port, self.metadata, self.max_wait_s = policy, host, port, metadata, max_wait_s

    def serve_forever(self):
        asyncio.run(self.run())

    async def run(self):
        import websockets.asyncio.server as ws_server
        self.queue = asyncio.Queue()
        worker = asyncio.create_task(self._batch_loop())
        async with ws_server.serve(self._handler, self.host, self.port, compression=None, max_size=None) as server:
            logging.info('batching server listening on %s:%s (max batch %d)', self.host, self.port, self.policy.max_batch)
            await server.serve_forever()
        worker.cancel()

    async def _handler(self, websocket):
        import traceback

        import websockets
        from openpi_client import msgpack_numpy
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self.metadata))
        loop = asyncio.get_running_loop()
        while True:
            try:
                obs = msgpack_numpy.unpackb(await websocket.recv())
                future = loop.create_future()
                await self.queue.put((obs, future))
                await websocket.send(packer.pack(await future))
            except websockets.ConnectionClosed:
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(code=websockets.frames.CloseCode.INTERNAL_ERROR, reason='Internal server error')
                raise

    async def _batch_loop(self):
        loop = asyncio.get_running_loop()
        while True:
            items = [await self.queue.get()]
            await asyncio.sleep(self.max_wait_s)       # let requests that are already on the wire arrive
            while len(items) < self.policy.max_batch and not self.queue.empty():
                items.append(self.queue.get_nowait())
            try:
                results = await loop.run_in_executor(None, self.policy.infer_batch, [obs for obs, _ in items])
                for (_, future), result in zip(items, results):
                    if not future.done():
                        future.set_result(result)
            except Exception as error:   # one bad request fails its batch, not the server
                logging.exception('batch failed')
                for _, future in items:
                    if not future.done():
                        future.set_exception(error)


def validate_batched_policy(batched, fixture, output):
    """The batched path must agree with the sequential one on the same requests."""
    data = np.load(fixture, allow_pickle=False)
    horizon = batched.horizon
    grid = np.broadcast_to(np.linspace(0, 1, 100, dtype=np.float32)[:, None, None], (100, 100, 100)).copy()
    guidance = dict(protocol=PROTOCOL, cost_map=grid, workspace_min=np.array([-1, -1, 0]), workspace_max=np.array([1, 1, 2]),
        anchor=data['anchor'], controller_scale=data['controller_scale'], position_strength=1., rotation_strength=2.,
        position_gain=np.array([.11, .088, .04]), rotation_target_euler=np.array([0., 0., np.pi/2]),
        rotation_ramp=np.linspace(0, 1, horizon)**2*.6, rotation_alignment=np.array([.5, .1, .05, 1]),
        timestep_scaling=True, min_timestep_scale=.1, time_floor=0., rotation_time_floor=0., rotation_yield=0.,
        override_strength=0., override_progress=0., override_reach=0., bound_result=0.)
    def packet(seed, steered):
        base = {'observation/image': data['image'], 'observation/wrist_image': data['wrist_image'],
                'observation/state': data['state'], 'prompt': str(data['prompt']), '_sample_seed': seed}
        return {**base, '_steering': guidance} if steered else base
    requests = [packet(11, True), packet(12, False), packet(13, True), packet(11, False), packet(14, True)]
    reference = [batched.sequential.infer(dict(r))['actions'] for r in requests]
    together = [r['actions'] for r in batched.infer_batch([dict(r) for r in requests])]
    alone = [batched.infer_batch([dict(r)])[0]['actions'] for r in requests[:2]]
    worst = max(float(np.abs(a-b).max()) for a, b in zip(reference, together))
    worst_alone = max(float(np.abs(a-b).max()) for a, b in zip(reference[:2], alone))
    # The model runs in half precision, so two compilations of the same sampler differ by a few
    # thousandths of an action unit (0.1 mm per step). A logic error would differ by tenths.
    if worst > 2e-2 or worst_alone > 2e-2:
        raise AssertionError(f'Batched and sequential actions differ by {worst:.2e} / {worst_alone:.2e}')
    # Compile every batch size the server can form now, so that a first batch of a new size does not block
    # the event loop long enough for clients' keepalive pings to time out.
    compiled = {}
    for size in [s for s in batched.SIZES if s <= batched.max_batch]:
        start = time.time()
        batched.infer_batch([dict(packet(20+i, i % 2 == 0)) for i in range(size)])
        compiled[size] = round(time.time()-start, 1)
    result = dict(batched_vs_sequential_max_abs=worst, single_vs_sequential_max_abs=worst_alone, max_batch=batched.max_batch,
                  warmup_seconds=compiled)
    Path(output).write_text(json.dumps(result, indent=2)+'\n')
    logging.info('Batched policy validation passed: %s', result)


def validate_loaded_policy(policy, fixture, output):
    """Real-checkpoint reproducibility and steering check before serving."""
    data = np.load(fixture, allow_pickle=False)
    packet = {'observation/image': data['image'], 'observation/wrist_image': data['wrist_image'],
              'observation/state': data['state'], 'prompt': str(data['prompt']), '_sample_seed': 12345}
    horizon = policy.native._model.action_horizon
    # Cost rises along +x, so guidance must move the path toward -x; the rotation
    # target is a quarter turn to the left, so the wrist must turn that way.
    grid = np.broadcast_to(np.linspace(0, 1, 100, dtype=np.float32)[:, None, None], (100, 100, 100)).copy()
    guidance = dict(protocol=PROTOCOL, cost_map=grid, workspace_min=np.array([-1, -1, 0]), workspace_max=np.array([1, 1, 2]),
        anchor=data['anchor'], controller_scale=data['controller_scale'], position_strength=0., rotation_strength=0.,
        position_gain=np.array([.11, .088, .04]), rotation_target_euler=np.array([0., 0., np.pi/2]),
        rotation_ramp=np.linspace(0, 1, horizon)**2*.6, rotation_alignment=np.array([.5, .1, .05, 1]),
        timestep_scaling=True, min_timestep_scale=.1, time_floor=0., rotation_time_floor=0., rotation_yield=0.,
        override_strength=0., override_progress=0., override_reach=0., bound_result=0.)
    native = policy.infer(packet)['actions']
    repeated = policy.infer(packet)['actions']
    zero = policy.infer({**packet, '_steering': guidance})['actions']
    if not np.array_equal(native, repeated) or not np.array_equal(native, zero):
        raise AssertionError('Native, repeated-seed, and zero-guidance actions differ')
    guided = policy.infer({**packet, '_steering': {**guidance, 'position_strength': 1., 'rotation_strength': 2.}})
    moved = np.cumsum((guided['actions'][:, :6]-native[:, :6])*np.asarray(data['controller_scale']), axis=0)
    if guided['steering_info']['nonzero_guidance_steps'] < 1 or not moved[-1, 0] < 0 or not moved[-1, 5] > 0:
        raise AssertionError('Guidance did not move the path down the cost slope and toward the turn')
    # Guidance never edits the gripper dimension, but later sampling steps see a
    # shifted pose, so the gripper output can move a little; report by how much.
    gripper = guided['actions'][:, 6]-native[:, 6]
    result = dict(checkpoint_manifest_sha256=policy.metadata['checkpoint_manifest_sha256'],
        native_repeat_bitwise_equal=True, zero_guidance_bitwise_equal=True,
        nonzero_guidance=guided['steering_info'], path_end_shift_m=moved[-1, :3].tolist(),
        path_end_turn_deg=float(np.degrees(moved[-1, 5])),
        gripper_max_abs_change=float(np.abs(gripper).max()),
        gripper_command_flips=int(np.sum(np.sign(guided['actions'][:, 6]) != np.sign(native[:, 6]))))
    Path(output).write_text(json.dumps(result, indent=2)+'\n')
    logging.info('Loaded policy validation passed: %s', result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--training-config-dir', type=Path, required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8003)
    parser.add_argument('--metadata-out', type=Path, required=True)
    parser.add_argument('--validate-fixture', type=Path, required=True)
    parser.add_argument('--max-batch', type=int, default=0,
                        help='Group up to this many pending requests into one model call (0: one request at a time, exact)')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)
    if not (args.checkpoint/'DOWNLOAD_VERIFIED.json').is_file():
        raise RuntimeError('Download verification marker is missing')
    sys.path.insert(0, str(args.training_config_dir))
    from pi05_calvin_config import PI05_CALVIN_CONFIGS
    config = next(c for c in PI05_CALVIN_CONFIGS if c.name == 'pi05_calvin_overloaded')
    native = policy_config.create_trained_policy(config, args.checkpoint)
    data_config = config.data.create(config.assets_dirs, config.model)
    stats = checkpoints.load_norm_stats(args.checkpoint/'assets', data_config.asset_id)
    metadata = dict(checkpoint_manifest_sha256=hashlib.sha256((args.checkpoint/'COMPLETED.json').read_bytes()).hexdigest(),
        optimizer_step=200000, config_name=config.name, rng_protocol='uint32_per_request_v1', guidance_protocol=PROTOCOL,
        action_horizon=config.model.action_horizon, action_dim=7,
        implementation_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), Path(__file__).with_name('flow.py')]})
    policy = GuidedServerPolicy(native, stats, data_config.use_quantile_norm, metadata)
    validate_loaded_policy(policy, args.validate_fixture, args.metadata_out.with_name('loaded_policy_validation.json'))
    args.metadata_out.write_text(json.dumps(metadata, indent=2)+'\n')
    if args.max_batch > 1:
        batched = BatchedServerPolicy(policy, args.max_batch)
        validate_batched_policy(batched, args.validate_fixture, args.metadata_out.with_name('batched_policy_validation.json'))
        metadata['batched'] = args.max_batch
        args.metadata_out.write_text(json.dumps(metadata, indent=2)+'\n')
        BatchingServer(batched, args.host, args.port, metadata).serve_forever()
        return
    websocket_policy_server.WebsocketPolicyServer(policy, host=args.host, port=args.port, metadata=metadata).serve_forever()


if __name__ == '__main__':
    main()
