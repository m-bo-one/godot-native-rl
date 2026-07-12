#!/usr/bin/env python3
"""chase_rays: the chase twin + ANALYTIC ray-vs-AABB observations (#364, follow-up to #37).

Extends the pure-NumPy chase twin with the missing axis the issue asks for: a raycast-sensor
observation reproduced analytically. Four fixed 100x100 AABB obstacles block the agent
(code-side minimal-penetration resolution — mirrored exactly by chase_rays_game.gd, which
does the same in GDScript instead of engine physics), and an 8-ray surround RaycastSensor2D
(cone 315deg, length 300, closeness encoding) observes them. The Godot side uses the REAL
physics RaycastSensor2D against StaticBody2D boxes; the committed golden fixture
(test/fixtures/chase_rays_golden_obs.json) pins engine-physics == this module's slab-method
analytic distances (Python exact; GDScript within 2e-3).

Obs = ChaseObs 5 floats + 8 closenesses = 13. Everything else (actions, cadence, reward)
is inherited from chase_twin_env — the proven transfer-critical invariants.
"""
from __future__ import annotations

import math

import numpy as np

from chase_twin_env import (  # noqa: F401  (re-exported constants used by the trainer)
    ACTION_REPEAT, ARENA_H, ARENA_W, DT, MAX_DIST, MAX_STEPS, MOVE_SPEED, N_ACTIONS,
    STEP_PENALTY, TOUCH_BONUS, TOUCH_RADIUS, action_to_velocity, compute_obs,
)

try:
    import gymnasium as gym
    from gymnasium import spaces
    _HAVE_GYM = True
except Exception:  # noqa: BLE001
    gym = None
    spaces = None
    _HAVE_GYM = False

# --- Fixed geometry (mirrored by chase_rays_game.gd; a unit test on each side pins it) ---
# AABBs as (min_x, min_y, max_x, max_y): 100x100 boxes at centers (300,200) (700,200) (300,400) (700,400).
OBSTACLES = (
    (250.0, 150.0, 350.0, 250.0),
    (650.0, 150.0, 750.0, 250.0),
    (250.0, 350.0, 350.0, 450.0),
    (650.0, 350.0, 750.0, 450.0),
)
N_RAYS = 8
CONE_DEGREES = 315.0
RAY_LENGTH = 300.0
OBS_DIM = 5 + N_RAYS


def ray_directions() -> list:
    """The RaycastSensor2D fan (RaycastMath.ray_directions_2d(8, 315, 0)): endpoints inclusive
    from -cone/2 to +cone/2 around +X — 45 degrees apart, full surround, no duplicate."""
    cone = math.radians(CONE_DEGREES)
    start = -cone / 2.0
    step = cone / float(N_RAYS - 1)
    return [(math.cos(start + step * i), math.sin(start + step * i)) for i in range(N_RAYS)]


def ray_aabb_distance(origin, direction, aabb) -> float:
    """Slab-method ray-vs-AABB: hit distance (>= 0) along the unit `direction`, or -1.0 for a
    miss / a box entirely behind the origin. An origin inside the box reads 0.0 (the physics
    ray reports a hit at the origin)."""
    ox, oy = origin
    dx, dy = direction
    min_x, min_y, max_x, max_y = aabb
    t_near = -math.inf
    t_far = math.inf
    for o, d, lo, hi in ((ox, dx, min_x, max_x), (oy, dy, min_y, max_y)):
        if abs(d) < 1e-12:
            if o < lo or o > hi:
                return -1.0
            continue
        t1 = (lo - o) / d
        t2 = (hi - o) / d
        if t1 > t2:
            t1, t2 = t2, t1
        t_near = max(t_near, t1)
        t_far = min(t_far, t2)
        if t_near > t_far:
            return -1.0
    if t_far < 0.0:
        return -1.0
    return max(t_near, 0.0)


def closeness(distance: float, ray_length: float = RAY_LENGTH) -> float:
    """RaycastMath.closeness: miss (negative) -> 0.0, hit -> 1 - clamp(d / L)."""
    if distance < 0.0:
        return 0.0
    return min(max(1.0 - distance / ray_length, 0.0), 1.0)


def ray_closenesses(agent) -> list:
    """The 8-ray obs block at `agent`: nearest obstacle hit per ray, closeness-encoded."""
    out = []
    for d in ray_directions():
        best = -1.0
        for box in OBSTACLES:
            hit = ray_aabb_distance(agent, d, box)
            if hit >= 0.0 and (best < 0.0 or hit < best):
                best = hit
        out.append(closeness(best))
    return out


def inside_any_obstacle(pos) -> bool:
    x, y = pos
    return any(min_x < x < max_x and min_y < y < max_y for min_x, min_y, max_x, max_y in OBSTACLES)


def resolve_aabb(pos, aabb):
    """Push `pos` out of `aabb` along the minimal-penetration axis (no-op outside). The blocking
    primitive — chase_rays_game.gd implements the identical rule, so twin and engine agree."""
    x, y = float(pos[0]), float(pos[1])
    min_x, min_y, max_x, max_y = aabb
    if not (min_x < x < max_x and min_y < y < max_y):
        return np.array([x, y])
    push_left = x - min_x
    push_right = max_x - x
    push_up = y - min_y
    push_down = max_y - y
    m = min(push_left, push_right, push_up, push_down)
    if m == push_left:
        return np.array([min_x, y])
    if m == push_right:
        return np.array([max_x, y])
    if m == push_up:
        return np.array([x, min_y])
    return np.array([x, max_y])


def resolve_obstacles(pos):
    """Apply resolve_aabb over the fixed obstacle list in order (both sides use this order)."""
    p = np.asarray(pos, dtype=float)
    for box in OBSTACLES:
        p = resolve_aabb(p, box)
    return p


def _clamp_to_bounds(pos):
    return np.array([min(max(pos[0], 0.0), ARENA_W), min(max(pos[1], 0.0), ARENA_H)])


def full_obs(agent, target) -> "np.ndarray":
    return np.concatenate([
        compute_obs(np.asarray(agent), np.asarray(target)),
        np.array(ray_closenesses((float(agent[0]), float(agent[1]))), dtype=np.float32),
    ]).astype(np.float32)


if _HAVE_GYM:
    class ChaseRaysTwinEnv(gym.Env):
        """Chase + solid AABB obstacles + analytic 8-ray obs, as a Gymnasium env (#364)."""

        metadata = {"render_modes": []}

        def __init__(self, seed: int | None = None):
            super().__init__()
            self.action_space = spaces.Discrete(N_ACTIONS)
            low = np.array([-1.0, -1.0, -1.0, -1.0, 0.0] + [0.0] * N_RAYS, dtype=np.float32)
            high = np.array([1.0, 1.0, 1.0, 1.0, 1.0] + [1.0] * N_RAYS, dtype=np.float32)
            self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
            self._agent = np.zeros(2)
            self._target = np.zeros(2)
            self._prev_dist = 0.0
            self._pending_bonus = 0.0
            self._steps = 0
            self.catches = 0
            self._seed = seed

        def _random_pos_outside(self):
            while True:
                p = np.array([self.np_random.uniform(0.0, ARENA_W),
                              self.np_random.uniform(0.0, ARENA_H)])
                if not inside_any_obstacle((p[0], p[1])):
                    return p

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed if seed is not None else self._seed)
            self._seed = None
            self._agent = self._random_pos_outside()
            self._target = self._random_pos_outside()
            self._prev_dist = float(np.hypot(*(self._target - self._agent)))
            self._pending_bonus = 0.0
            self._steps = 0
            self.catches = 0
            return full_obs(self._agent, self._target), {}

        def step(self, action):
            vel = action_to_velocity(int(action))
            reward = 0.0
            for _ in range(ACTION_REPEAT):
                self._agent = resolve_obstacles(_clamp_to_bounds(self._agent + vel * DT))
                cur = float(np.hypot(*(self._target - self._agent)))
                reward += (self._prev_dist - cur) / MAX_DIST - STEP_PENALTY + self._pending_bonus
                self._pending_bonus = 0.0
                self._prev_dist = cur
                if cur < TOUCH_RADIUS:
                    self.catches += 1
                    self._target = self._random_pos_outside()
                    self._prev_dist = float(np.hypot(*(self._target - self._agent)))
                    self._pending_bonus = TOUCH_BONUS
            self._steps += 1
            truncated = self._steps >= MAX_STEPS
            return (full_obs(self._agent, self._target), reward, False, truncated,
                    {"catches": self.catches})
