#!/usr/bin/env python3
"""JAX batch backend for the chase twin (#361, follow-up to #37).

The NumPy twin (chase_twin_env.py) deleted the socket + engine; this ports its step/obs/reward
to JAX so a whole BATCH of envs advances as one jit-compiled kernel (`batched_step`) — the path
to very large batch sizes (and GPU) that SubprocVecEnv can't reach. Numerical parity with the
NumPy twin is the transfer-critical invariant (unit-tested: same obs encoding, same
ACTION_REPEAT sub-frame cadence at DT=1/60, same catch/rebase/pending-bonus semantics), so a
policy trained here deploys into the REAL Godot engine exactly like the NumPy-trained one.

Randomness note: target relocation draws from a per-env JAX PRNG key, so trajectories are NOT
stream-identical to the NumPy twin (they don't need to be — transfer needs identical dynamics
given state, not identical randomness).

Honest scope (from the #361 design): on CPU at small batch this is comparable to the NumPy twin;
the win appears at large batch (and on GPU, where the same code runs unchanged). The trainer
(train_chase_jax.py) reports measured steps/s so the claim stays honest.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

# The single source of the matching math/constants is the NumPy twin.
from chase_twin_env import (  # noqa: F401  (re-exported for the trainer)
    ACTION_REPEAT, ARENA_H, ARENA_W, DT, MAX_DIST, MAX_STEPS, MOVE_SPEED, N_ACTIONS,
    OBS_DIM, STEP_PENALTY, TOUCH_BONUS, TOUCH_RADIUS,
)

# Discrete action -> velocity table (row = action index), matching action_to_velocity.
_VELOCITIES = jnp.array([
    [0.0, 0.0],
    [0.0, -MOVE_SPEED],
    [0.0, MOVE_SPEED],
    [-MOVE_SPEED, 0.0],
    [MOVE_SPEED, 0.0],
])


class TwinState(NamedTuple):
    """Batched env state: leading axis = env index."""

    agent: jnp.ndarray          # (N, 2)
    target: jnp.ndarray         # (N, 2)
    prev_dist: jnp.ndarray      # (N,)
    pending_bonus: jnp.ndarray  # (N,)
    catches: jnp.ndarray        # (N,) int32
    key: jnp.ndarray            # (N, 2) per-env PRNG keys


def compute_obs(agent: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """The 5-dim chase obs for ONE (agent, target) pair — mirrors chase_twin_env.compute_obs."""
    rel = target - agent
    dist = jnp.hypot(rel[0], rel[1])
    safe = jnp.where(dist > 0.0, dist, 1.0)
    dirv = jnp.where(dist > 0.0, rel / safe, jnp.zeros(2))
    return jnp.array([
        (agent[0] / ARENA_W - 0.5) * 2.0,
        (agent[1] / ARENA_H - 0.5) * 2.0,
        dirv[0],
        dirv[1],
        jnp.clip(dist / MAX_DIST, 0.0, 1.0),
    ], dtype=jnp.float32)


_batched_obs = jax.vmap(compute_obs)


def _random_pos(key: jnp.ndarray) -> jnp.ndarray:
    return jax.random.uniform(key, (2,)) * jnp.array([ARENA_W, ARENA_H])


def make_state(agent: jnp.ndarray, target: jnp.ndarray, key: jnp.ndarray) -> TwinState:
    """State from explicit (N,2) positions (tests / custom resets)."""
    n = agent.shape[0]
    prev = jnp.hypot(target[:, 0] - agent[:, 0], target[:, 1] - agent[:, 1])
    return TwinState(agent=jnp.asarray(agent, jnp.float32), target=jnp.asarray(target, jnp.float32),
                     prev_dist=prev.astype(jnp.float32),
                     pending_bonus=jnp.zeros(n, jnp.float32),
                     catches=jnp.zeros(n, jnp.int32),
                     key=jax.random.split(key, n))


def reset_batch(n: int, key: jnp.ndarray) -> TwinState:
    ka, kt, ks = jax.random.split(key, 3)
    agent = jax.random.uniform(ka, (n, 2)) * jnp.array([ARENA_W, ARENA_H])
    target = jax.random.uniform(kt, (n, 2)) * jnp.array([ARENA_W, ARENA_H])
    return make_state(agent, target, ks)


def _step_one(agent, target, prev_dist, pending, catches, key, action):
    """One env's full env-step (ACTION_REPEAT sub-frames) — the NumPy twin's loop, jax-traced."""
    vel = _VELOCITIES[action]
    reward = 0.0

    def sub_frame(_, carry):
        agent, target, prev_dist, pending, catches, key, reward = carry
        agent = jnp.clip(agent + vel * DT, jnp.zeros(2), jnp.array([ARENA_W, ARENA_H]))
        cur = jnp.hypot(target[0] - agent[0], target[1] - agent[1])
        reward = reward + (prev_dist - cur) / MAX_DIST - STEP_PENALTY + pending
        pending = 0.0
        prev_dist = cur
        caught = cur < TOUCH_RADIUS
        key, sub = jax.random.split(key)
        new_target = _random_pos(sub)
        target = jnp.where(caught, new_target, target)
        new_prev = jnp.hypot(target[0] - agent[0], target[1] - agent[1])
        prev_dist = jnp.where(caught, new_prev, prev_dist)
        pending = jnp.where(caught, TOUCH_BONUS, pending)
        catches = catches + caught.astype(jnp.int32)
        return agent, target, prev_dist, pending, catches, key, reward

    carry = (agent, target, prev_dist, pending, catches, key, reward)
    agent, target, prev_dist, pending, catches, key, reward = jax.lax.fori_loop(
        0, ACTION_REPEAT, sub_frame, carry)
    return agent, target, prev_dist, pending, catches, key, reward


_vmapped_step = jax.vmap(_step_one)


@jax.jit
def batched_step(state: TwinState, actions: jnp.ndarray):
    """Advance ALL envs one decision (ACTION_REPEAT sub-frames) as a single jit kernel.
    Returns (new_state, obs (N, 5), reward (N,), catches (N,))."""
    agent, target, prev_dist, pending, catches, key, reward = _vmapped_step(
        state.agent, state.target, state.prev_dist, state.pending_bonus,
        state.catches, state.key, actions)
    new_state = TwinState(agent=agent, target=target, prev_dist=prev_dist,
                          pending_bonus=pending, catches=catches, key=key)
    return new_state, _batched_obs(agent, target), reward, catches
