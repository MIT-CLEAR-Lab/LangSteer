"""Text-only policy conditioning; deliberately independent of task/annotation tables."""

import json
from pathlib import Path


PROMPT_PATH = Path(__file__).resolve().parents[1] / "conf/prompts/distill_language.txt"


class LanguageDistiller:
    def __init__(self, backend, prompt=None):
        self.backend = backend
        self.prompt = prompt if prompt is not None else PROMPT_PATH.read_text()

    @classmethod
    def from_config(cls, cfg):
        from voxposer.llm_cache import DiskCache
        from voxposer.lmp import LLMBackend

        return cls(LLMBackend(
            provider=cfg["provider"], model=cfg["model"],
            temperature=0, max_tokens=cfg.get("max_tokens", 512),
            cache=DiskCache(cfg["cache_dir"]),
            system_prompt="You simplify robot instructions. Follow the supplied rewriting rules and return only JSON.",
        ))

    def distill(self, instruction: str) -> str:
        # Only the incoming instruction crosses this boundary: no task IDs, scene
        # state, annotation files, candidate vocabulary, or success feedback.
        prompt = self.prompt + "\n# Query:\n" + json.dumps({"instruction": instruction})
        response = self.backend.generate(prompt, stop=[])
        try:
            payload = json.loads(response)
            text = payload["instruction"]
            if set(payload) != {"instruction"} or not isinstance(text, str):
                raise ValueError("Expected exactly one string field: instruction")
            text = text.strip().rstrip(".").strip()
            if not text or len(text.split()) > 20 or any(c in text for c in '\n\r;.!?'):
                raise ValueError("Expected one short imperative (at most 20 words)")
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"Invalid distillation response: {response!r}") from exc
        return text
