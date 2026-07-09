#!/usr/bin/env python3
"""A pure-NumPy Gymnasium "twin" of the Godot chase env (#37, backlog item 31).

Reimplements chase's kinematics + obs + reward EXACTLY as the Godot side (chase_game.gd /
chase_agent.gd / chase_obs.gd / run_speed.gd), so a policy trained here — with no Godot and no
socket, at far higher throughput — deploys back into Godot via ncnn and still catches. The obs
encoding and the per-step displacement are the transfer-critical invariants; they mirror the GDScript
byte-for-byte (see the unit tests). Reward shaping is faithful but only affects training.

Cadence note: Godot's run_speed scales physics-ticks AND time_scale together, so _physics_process
delta stays 1/60 at any speedup. One env step = action_repeat (8) sub-frames — train and deploy both
use 8 — each moving vel/60 = 5 px, i.e. 40 px/step, matching touch_radius = 40.

NumPy (not JAX): the win is deleting the socket + engine, not GPU kernels — chase's step is a few
float ops, so SB3 vectorization over many in-process copies already runs orders of magnitude faster
than the bridge with zero new heavy deps. A JAX batch backend is a noted future extension.
"""
from __future__ import annotations

import math

import numpy as np

try:  # gymnasium is in .venv-train; keep the pure helpers importable even if it is absent.
    import gymnasium as gym
    from gymnasium import spaces
    _HAVE_GYM = True
except Exception:  # noqa: BLE001
    gym = None
    spaces = None
    _HAVE_GYM = False

# --- Constants mirrored from the Godot chase env (single source of the matching math) ---
ARENA_W = 1000.0
ARENA_H = 600.0
MOVE_SPEED = 300.0        # chase_game.move_speed
TOUCH_RADIUS = 40.0       # chase_game.touch_radius
DT = 1.0 / 60.0           # _physics_process delta (speedup keeps this at 1/60)
ACTION_REPEAT = 8         # sub-frames per decision (train + deploy)
STEP_PENALTY = 0.001      # chase_agent.step_penalty (per sub-frame)
TOUCH_BONUS = 1.0         # chase_agent.touch_bonus
MAX_STEPS = 1000          # NcnnAIController2D.reset_after (episode horizon, in decisions)
N_ACTIONS = 5
OBS_DIM = 5
MAX_DIST = math.hypot(ARENA_W, ARENA_H)   # ‖arena‖, the obs/reward distance scale


def compute_obs(agent: np.ndarray, target: np.ndarray) -> np.ndarray:
    """The 5-dim chase obs, byte-identical to ChaseObs.compute_obs (agent/target are (x, y))."""
    rel = target - agent
    dist = float(np.hypot(rel[0], rel[1]))
    if dist > 0.0:
        dir_x, dir_y = rel[0] / dist, rel[1] / dist
    else:
        dir_x, dir_y = 0.0, 0.0
    return np.array([
        (agent[0] / ARENA_W - 0.5) * 2.0,
        (agent[1] / ARENA_H - 0.5) * 2.0,
        dir_x,
        dir_y,
        min(max(dist / MAX_DIST, 0.0), 1.0),
    ], dtype=np.float32)


def action_to_velocity(idx: int, speed: float = MOVE_SPEED) -> np.ndarray:
    """Discrete action -> velocity, matching ChaseObs.action_index_to_velocity (0 = noop)."""
    if idx == 1:
        return np.array([0.0, -speed])
    if idx == 2:
        return np.array([0.0, speed])
    if idx == 3:
        return np.array([-speed, 0.0])
    if idx == 4:
        return np.array([speed, 0.0])
    return np.array([0.0, 0.0])


def _clamp_to_bounds(pos: np.ndarray) -> np.ndarray:
    return np.array([min(max(pos[0], 0.0), ARENA_W), min(max(pos[1], 0.0), ARENA_H)])


if _HAVE_GYM:
    class ChaseTwinEnv(gym.Env):
        """Godot-chase twin as a Gymnasium env. SB3 vectorizes many copies in-process (no socket)."""

        metadata = {"render_modes": []}

        def __init__(self, seed: int | None = None):
            super().__init__()
            self.action_space = spaces.Discrete(N_ACTIONS)
            self.observation_space = spaces.Box(
                low=np.array([-1.0, -1.0, -1.0, -1.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
            self._agent = np.zeros(2)
            self._target = np.zeros(2)
            self._prev_dist = 0.0
            self._pending_bonus = 0.0
            self._steps = 0
            self.catches = 0
            self._seed = seed

        def _random_pos(self) -> np.ndarray:
            return np.array([self.np_random.uniform(0.0, ARENA_W),
                             self.np_random.uniform(0.0, ARENA_H)])

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed if seed is not None else self._seed)
            self._seed = None  # only honor the ctor seed on the first reset
            self._agent = self._random_pos()
            self._target = self._random_pos()
            self._prev_dist = float(np.hypot(*(self._target - self._agent)))
            self._pending_bonus = 0.0
            self._steps = 0
            self.catches = 0
            return compute_obs(self._agent, self._target), {}

        def step(self, action):
            vel = action_to_velocity(int(action))
            reward = 0.0
            for _ in range(ACTION_REPEAT):
                self._agent = _clamp_to_bounds(self._agent + vel * DT)
                cur = float(np.hypot(*(self._target - self._agent)))
                # Progress + step penalty + any bonus queued by the previous sub-frame's catch.
                reward += (self._prev_dist - cur) / MAX_DIST - STEP_PENALTY + self._pending_bonus
                self._pending_bonus = 0.0
                self._prev_dist = cur
                if cur < TOUCH_RADIUS:
                    self.catches += 1
                    self._target = self._random_pos()
                    self._prev_dist = float(np.hypot(*(self._target - self._agent)))  # rebase baseline
                    self._pending_bonus = TOUCH_BONUS  # lands next sub-frame (matches Godot)
            self._steps += 1
            truncated = self._steps >= MAX_STEPS
            return (compute_obs(self._agent, self._target), reward, False, truncated,
                    {"catches": self.catches})
