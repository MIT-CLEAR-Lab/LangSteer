"""Instruction pools for overloaded language conditioning.

The legacy artifact from preprocess_instructions.py is a flat list holding one CLIP
embedding per annotated window. The overloaded artifact from
preprocess_overloaded_instructions.py instead holds deduplicated embeddings plus, for
each window, a pool of *specific* instruction indices and a pool of *ambiguous* ones.
Drawing from the ambiguous pool is what pairs one low-specificity template with every
trajectory in its ambiguity class.

`load_instruction_pool` dispatches on the pickle's shape so the trainer keeps working
with artifacts produced before this existed.
"""

import pickle
import random

import torch


class OverloadedInstructions:
    """Deduplicated instruction embeddings with per-window specific/ambiguous pools."""

    def __init__(self, payload):
        self.strings = payload["strings"]
        self.embeddings = payload["embeddings"]      # (n_unique, max_length, 512)
        self.specific_ids = payload["specific_ids"]  # {annotation_id: [str_idx, ...]}
        self.ambiguous_ids = payload["ambiguous_ids"]
        self.classes = payload["classes"]
        self.meta = payload.get("meta", {})

        if not isinstance(self.embeddings, torch.Tensor):
            self.embeddings = torch.stack([torch.as_tensor(e) for e in self.embeddings])

    def __len__(self):
        return len(self.specific_ids)

    def has_ambiguous(self, annotation_id):
        return bool(self.ambiguous_ids.get(int(annotation_id)))

    def class_of(self, annotation_id):
        return self.classes.get(int(annotation_id))

    def draw(self, annotation_id, ambiguous_prob=0.0, deterministic=False):
        """Return one (max_length, 512) embedding for this window.

        With `deterministic` the first specific instruction is returned every time,
        which keeps validation loss comparable across runs and against the baseline.
        Otherwise a pool is chosen -- ambiguous with probability `ambiguous_prob`,
        conditioned on the window's task actually having an ambiguous tier -- and one
        instruction is drawn uniformly from it.
        """
        annotation_id = int(annotation_id)
        specific = self.specific_ids[annotation_id]
        if deterministic:
            return self.embeddings[specific[0]]

        ambiguous = self.ambiguous_ids.get(annotation_id)
        if ambiguous and random.random() < ambiguous_prob:
            return self.embeddings[random.choice(ambiguous)]
        return self.embeddings[random.choice(specific)]

    def text(self, index):
        return self.strings[index]


def load_instruction_pool(path):
    """Load either artifact format. Returns OverloadedInstructions or a plain list."""
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "strings" in payload:
        return OverloadedInstructions(payload)
    return payload["embeddings"]
