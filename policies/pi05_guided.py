"""pi0.5 (OpenPI) policy: a numpy-only websocket client with optional steering.

The pi0.5 JAX model cannot share a process with LangSteer's PyTorch stack, so it
runs in the OpenPI venv behind `pi05_guidance.server`. This class packs the
LangSteer `Observation` into the server's observation dict, sends a per-request
noise seed, exports the active steering stage when there is one, and returns a
receding-horizon `Action` of RELATIVE CALVIN actions.

Server observation contract (keys read by OpenPI `LiberoInputs`):
    observation/image        uint8 (H,W,3)   static camera  (native 200x200)
    observation/wrist_image  uint8 (H,W,3)   gripper camera (native 84x84)
    observation/state        float32 (15,)   CALVIN robot_obs
    prompt                   str             language instruction
The server returns `(action_horizon, 7)` relative actions
`[dx, dy, dz, d_euler_x, d_euler_y, d_euler_z, gripper]`; the first
`replan_steps` rows are executed before the next request.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from core.policy import BasePolicy
from core.steering import BaseSteering
from core.types import Action, Observation
from pi05_guidance.payload import PROTOCOL, export_guidance

logger = logging.getLogger(__name__)


class GuidedPi05Policy(BasePolicy):
    """Seeded pi0.5 client; steering is applied inside the server's sampler."""

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self._host = cfg.get("host", "127.0.0.1")
        self._port = int(cfg.get("port", 8003))
        self._action_horizon = int(cfg.get("action_horizon", 10))
        self._replan_steps = int(cfg.get("replan_steps", 5))
        self._static_key = cfg.get("static_key", "static")
        self._gripper_key = cfg.get("gripper_key", "gripper_native")
        self._prompt_override = cfg.get("prompt", None)
        # The checkpoint's completion manifest identifies the weights the server must hold.
        self._expected_manifest = hashlib.sha256(Path(cfg["ckpt_path"]).read_bytes()).hexdigest()
        self._episode_seed: Optional[int] = None
        self._request_index = 0
        self._diagnostics: dict = {}
        self._client: Any = None
        self._connect()

    def _connect(self) -> None:
        # openpi_client is numpy/msgpack/websockets only (no jax, no torch).
        from openpi_client import websocket_client_policy

        logger.info(f"[pi05] connecting to {self._host}:{self._port} (retries until the server is reachable)")
        self._client = websocket_client_policy.WebsocketClientPolicy(host=self._host, port=self._port)
        metadata = self._client.get_server_metadata()
        if metadata.get("checkpoint_manifest_sha256") != self._expected_manifest:
            raise RuntimeError("π0.5 server checkpoint identity differs from this evaluation")
        if metadata.get("rng_protocol") != "uint32_per_request_v1" or metadata.get("guidance_protocol") != PROTOCOL:
            raise RuntimeError("π0.5 server does not support seeded spatial flow guidance")

    def load_checkpoint(self, path: str) -> None:
        # Weights live on the server; `ckpt_path` names the manifest checked in `_connect`.
        logger.info(f"[pi05] weights are served remotely; manifest={path!r}")

    def bind_steering(self, steering: BaseSteering) -> None:
        from steering.voxposer_steering import VoxPoserSteering

        if not isinstance(steering, VoxPoserSteering):
            raise TypeError("π0.5 currently supports LangSteer spatial fields (voxposer)")

    def reset(self) -> None:
        self._client.reset()  # server-side no-op
        self._episode_seed = int(np.random.randint(0, 2**32, dtype=np.uint32))
        self._request_index = 0
        self._diagnostics = dict(requests=0, steered_requests=0)

    @staticmethod
    def _to_uint8_hwc(img: np.ndarray) -> np.ndarray:
        """Coerce an image to contiguous uint8 HWC.

        The evaluation path delivers RGB as float32 in [0, 1]; convert back to
        uint8. Defensively transpose a CHW array to HWC.
        """
        a = np.asarray(img)
        if a.dtype != np.uint8:
            a = np.clip(np.round(a * 255.0), 0, 255).astype(np.uint8)
        if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[2] not in (1, 3):
            a = np.transpose(a, (1, 2, 0))
        return np.ascontiguousarray(a)

    def _build_obs(self, obs: Observation) -> dict:
        for key in (self._static_key, self._gripper_key):
            if key not in obs.rgb:
                raise KeyError(f"[pi05] missing obs.rgb['{key}']; available keys: {list(obs.rgb)}")
        return {
            "observation/image": self._to_uint8_hwc(obs.rgb[self._static_key]),         # 200x200x3
            "observation/wrist_image": self._to_uint8_hwc(obs.rgb[self._gripper_key]),  # 84x84x3
            "observation/state": np.asarray(obs.proprio, dtype=np.float32),             # (15,)
            "prompt": str(self._prompt_override or obs.instruction),
        }

    def forward(self, obs: Observation, steering: Optional[BaseSteering] = None) -> Action:
        if self._episode_seed is None:
            raise RuntimeError("Reset the π0.5 policy before inference")
        packet = self._build_obs(obs)
        packet["_sample_seed"] = (self._episode_seed + 0x9E3779B9 * self._request_index) & 0xFFFFFFFF
        if steering is not None:
            self.bind_steering(steering)
            steering.set_current_gripper_pos(np.asarray(obs.proprio[:3]))
            packet["_steering"] = export_guidance(steering, obs, self._action_horizon)
        result = self._client.infer(packet)
        self._request_index += 1
        info = result.get("steering_info", {})
        if steering is not None and (info.get("protocol") != PROTOCOL or not info.get("requested")):
            raise RuntimeError("The model server did not acknowledge spatial guidance")
        chunk = np.asarray(result["actions"], dtype=np.float32)
        if chunk.shape != (self._action_horizon, 7) or not np.isfinite(chunk).all():
            raise ValueError(f"Invalid π0.5 action chunk: shape={chunk.shape}")
        self._diagnostics["requests"] += 1
        self._diagnostics["steered_requests"] += int(steering is not None)
        for key, value in info.items():
            if key not in ("protocol", "requested", "enabled"):
                self._diagnostics[key] = self._diagnostics.get(key, 0) + value
        # Receding horizon: execute the first `replan_steps` relative rows, then re-infer.
        rows = chunk[: self._replan_steps].copy()
        return Action(trajectory=rows, gripper=float(rows[0, 6]), relative=True)

    def get_diagnostics(self) -> dict:
        return dict(self._diagnostics)
