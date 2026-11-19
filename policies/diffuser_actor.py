"""Construction of the 3D Diffuser Actor policy from a config dict.

The model is conditioned on CLIP text features of the instruction. The split into
`DiffuserActorBasePolicy` (shared plumbing) and `LanguageDiffuserActorPolicy`
(the conditioning hooks) keeps observation and action handling separate from how
the instruction is encoded.
"""

from __future__ import annotations

from typing import Any

from policies.diffuser_actor_base import DiffuserActorBasePolicy


def build_diffuser_actor_policy(cfg: Any) -> DiffuserActorBasePolicy:
    """Return the language-conditioned 3D Diffuser Actor policy."""
    from policies.diffuser_actor_language import LanguageDiffuserActorPolicy

    return LanguageDiffuserActorPolicy(cfg)


# `DiffuserActorPolicy(cfg)` works as if it were the class.
DiffuserActorPolicy = build_diffuser_actor_policy
