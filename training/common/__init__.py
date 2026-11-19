"""Utilities shared by the training code.

3D Diffuser Actor training uses `checkpoint_util` and `ema_model`.
`pi05_calvin_config` is the OpenPI training config of the pi0.5 checkpoint; the pi0.5
server imports it inside the OpenPI environment, so nothing is imported eagerly here.
"""
