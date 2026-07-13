#!/usr/bin/env python3
"""Truncation-aware godot_rl bridge (#12): consume the wire's additive `truncated` field.

The Godot side (NcnnSync) sends `truncated` alongside `done` since #12 — `done` keeps meaning
terminated-OR-truncated so stock godot_rl 0.8.2 (which reads only obs/reward/done/info and has a
`TODO update API to term, trunc` in step_recv) is untouched. This module is OUR consumer:
`TruncationAwareGodotEnv` overrides `step_recv` to return the real Gymnasium (terminated,
truncated) split instead of upstream's done-duplication. Used by our own adapters
(GodotParallelEnv for PettingZoo/RLlib-multi-policy, GodotRLlibEnv) — trainers going through
stock godot_rl wrappers behave exactly as before.

Fallback when the wire has no `truncated` (a pre-#12 scene): every episode end counts as
TERMINAL and truncated stays False — strictly more useful than upstream's duplication (which
marks every end as both), and it matches what our adapters assumed before this module existed.

Upstream status (checked 2026-07-13): no godot_rl release since 0.8.2 (PyPI 2025-02-25); main
(207b6f4, 2026-06-13) still collapses both flags into `done`; no upstream PR implements the
split. When upstream lands it, this subclass either becomes a no-op or gets retired.

Heavy imports are lazy-ish: godot_rl/numpy are needed to *use* the class, but the pure
`split_done_truncated` helper imports nothing (unit-tested bare, #141).
"""
from __future__ import annotations

try:
    from godot_rl.core.godot_env import GodotEnv
    _HAVE_GODOT_RL = True
except Exception:  # noqa: BLE001  (keep the pure helper importable without the dep)
    GodotEnv = object
    _HAVE_GODOT_RL = False


def split_done_truncated(done, truncated):
    """Wire flags -> Gymnasium (terminated, truncated) lists.

    `done[i]` is terminated-OR-truncated (the 0.8.2 wire invariant); `truncated` is the
    additive #12 field or None when the scene predates it (-> all ends terminal)."""
    done = [bool(d) for d in done]
    if truncated is None:
        return done, [False] * len(done)
    truncated = [bool(t) for t in truncated]
    terminated = [d and not t for d, t in zip(done, truncated)]
    return terminated, truncated


if _HAVE_GODOT_RL:
    class TruncationAwareGodotEnv(GodotEnv):
        """GodotEnv whose step_recv returns the REAL (terminated, truncated) split (#12)."""

        def step_recv(self):
            response = self._get_json_dict()
            response["obs"] = self._process_obs(response["obs"])
            # Kept for backward compatibility if the plugin doesn't send info (as upstream).
            default_info = [{}] * len(response["done"])
            terminated, truncated = split_done_truncated(
                response["done"], response.get("truncated"))
            return (
                response["obs"],
                response["reward"],
                terminated,
                truncated,
                response.get("info", default_info),
            )
