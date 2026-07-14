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

from chase_twin_env import (
    ARENA_H, ARENA_W, N_ACTIONS, _clamp_to_bounds, compute_obs,
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
    ray reports a hit at the origin).

    #374 edge: an origin EXACTLY on a face is ill-defined and NOT part of the parity contract.
    Here it reads hit-at-0 for every direction (t_far becomes -0.0, which `t_far < 0.0` treats as
    in front), so all 8 rays encode closeness 1.0; Godot's intersect_ray from a point coincident
    with a collider boundary is direction/face-dependent instead. Parity holds arbitrarily close to
    a face (see the near-wall golden cases) but not AT it — the game keeps the agent OUTSIDE boxes
    (resolve_aabb pins it on the boundary at worst), and this coincident-origin state is transient
    and unobserved by the reward, so the tiny obs discrepancy there does not affect training."""
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


# The fan is constant — computed once, not per env step (the twin's whole point is throughput).
_RAY_DIRS = tuple(ray_directions())


def ray_closenesses(agent) -> list:
    """The 8-ray obs block at `agent`: nearest obstacle hit per ray, closeness-encoded."""
    out = []
    for d in _RAY_DIRS:
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
        return pos  # no-op path (the common case): no allocation
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


def full_obs(agent, target) -> "np.ndarray":
    return np.concatenate([
        compute_obs(np.asarray(agent), np.asarray(target)),
        np.array(ray_closenesses((float(agent[0]), float(agent[1]))), dtype=np.float32),
    ]).astype(np.float32)


if _HAVE_GYM:
    from chase_twin_env import ChaseTwinEnv

    class ChaseRaysTwinEnv(ChaseTwinEnv):
        """Chase + solid AABB obstacles + analytic 8-ray obs (#364). Subclasses the proven
        chase twin: only the movement rule (_move: clamp then block), the spawn rule
        (_random_pos: rejection-sampled outside the boxes) and the obs (_obs: base 5 + rays)
        differ — the transfer-critical sub-frame reward/rebase machinery stays single-source
        in ChaseTwinEnv.step."""

        def __init__(self, seed: int | None = None):
            super().__init__(seed=seed)
            low = np.array([-1.0, -1.0, -1.0, -1.0, 0.0] + [0.0] * N_RAYS, dtype=np.float32)
            high = np.array([1.0, 1.0, 1.0, 1.0, 1.0] + [1.0] * N_RAYS, dtype=np.float32)
            self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        def _move(self, pos):
            return resolve_obstacles(_clamp_to_bounds(pos))

        def _random_pos(self):
            while True:
                p = super()._random_pos()
                if not inside_any_obstacle((p[0], p[1])):
                    return p

        def _obs(self):
            return full_obs(self._agent, self._target)
