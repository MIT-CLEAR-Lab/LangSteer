"""Retention of periodic training checkpoints."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


class TopKCheckpointManager:
    """Keep the `k` periodic checkpoints with the best monitored value.

    `k < 0` keeps every checkpoint and `k == 0` keeps none. With `mode='max'` a
    higher value is better. A candidate replaces the worst retained checkpoint only
    when it is strictly better; the displaced file is deleted. Candidates without a
    value are kept only when everything is kept.

    The retained set is plain data (`state_dict` / `load_state_dict`) so that it can
    travel inside the training checkpoint and survive a resume.
    """

    def __init__(self, save_dir, k: int, mode: str = "max"):
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")
        self.save_dir = Path(save_dir)
        self.k = int(k)
        self.mode = mode
        self.retained: dict[str, float] = {}  # file name -> monitored value

    def _better(self, a: float, b: float) -> bool:
        return a > b if self.mode == "max" else a < b

    def admit(self, name: str, value: Optional[float]) -> bool:
        """Decide whether the checkpoint `name` should be written.

        Returns True when the caller should save it. When it displaces a retained
        checkpoint, that file is removed from `save_dir` here.
        """
        if self.k < 0:
            return True
        if self.k == 0 or value is None:
            return False
        value = float(value)
        if len(self.retained) < self.k:
            self.retained[name] = value
            return True
        worst_name = min(
            self.retained,
            key=lambda n: self.retained[n] if self.mode == "max" else -self.retained[n],
        )
        if not self._better(value, self.retained[worst_name]):
            return False
        del self.retained[worst_name]
        worst_path = self.save_dir / worst_name
        if worst_path.exists():
            os.remove(worst_path)
        self.retained[name] = value
        return True

    def state_dict(self) -> dict:
        return {"k": self.k, "mode": self.mode, "retained": dict(self.retained)}

    def load_state_dict(self, state: dict) -> None:
        self.retained = {
            name: float(value)
            for name, value in state.get("retained", {}).items()
            if (self.save_dir / name).exists()
        }
