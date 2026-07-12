"""Unit tests for the chase_rays NumPy twin (#364) — the analytic ray-vs-AABB axis.

Pure math (ray distances, closeness encoding, AABB blocking) runs bare; the Gymnasium env cases
are dep-probe-guarded. The golden-fixture test pins the SAME values the GDScript physics-parity
test (test/unit/test_chase_rays.gd) asserts against the real RaycastSensor2D."""
import json
import math
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

try:
    import numpy as np  # noqa: F401
    HAVE_NP = True
except ImportError:
    HAVE_NP = False
if HAVE_NP:
    import chase_rays_twin_env as cr

try:
    import gymnasium  # noqa: F401
    HAVE_GYM = HAVE_NP
except ImportError:
    HAVE_GYM = False

needs_np = unittest.skipUnless(HAVE_NP, "numpy not installed")
needs_gym = unittest.skipUnless(HAVE_GYM, "numpy/gymnasium not installed")

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "chase_rays_golden_obs.json"


@needs_np
class TestRayAabbDistance(unittest.TestCase):
    # Obstacle 0 is the AABB [250,350]x[150,250] (center (300,200), 100x100).
    def test_straight_hit(self):
        # From (100, 200) firing +X: hits the left face at x=250 -> distance 150.
        d = cr.ray_aabb_distance((100.0, 200.0), (1.0, 0.0), (250.0, 150.0, 350.0, 250.0))
        self.assertAlmostEqual(d, 150.0, places=6)

    def test_miss_parallel(self):
        # Firing +X from y=100 passes under the box (box spans y 150..250).
        d = cr.ray_aabb_distance((100.0, 100.0), (1.0, 0.0), (250.0, 150.0, 350.0, 250.0))
        self.assertEqual(d, -1.0)

    def test_behind_is_a_miss(self):
        # Box is behind the ray origin.
        d = cr.ray_aabb_distance((500.0, 200.0), (1.0, 0.0), (250.0, 150.0, 350.0, 250.0))
        self.assertEqual(d, -1.0)

    def test_diagonal_hit(self):
        # From (200, 100) toward (1,1)/sqrt2: enters where both slabs overlap.
        inv = 1.0 / math.sqrt(2.0)
        d = cr.ray_aabb_distance((200.0, 100.0), (inv, inv), (250.0, 150.0, 350.0, 250.0))
        # x=250 at t=50*sqrt2 ~ 70.71 (y there = 150 — exactly the corner).
        self.assertAlmostEqual(d, 50.0 * math.sqrt(2.0), places=4)

    def test_origin_inside_reads_zero(self):
        d = cr.ray_aabb_distance((300.0, 200.0), (1.0, 0.0), (250.0, 150.0, 350.0, 250.0))
        self.assertEqual(d, 0.0)


@needs_np
class TestRayCloseness(unittest.TestCase):
    def test_closeness_encoding_matches_raycast_math(self):
        # RaycastMath.closeness: miss -> 0, hit -> 1 - clamp(d / L).
        self.assertEqual(cr.closeness(-1.0), 0.0)
        self.assertAlmostEqual(cr.closeness(150.0), 1.0 - 150.0 / cr.RAY_LENGTH, places=6)
        self.assertEqual(cr.closeness(cr.RAY_LENGTH * 2), 0.0)

    def test_ray_directions_match_raycast_math_fan(self):
        # RaycastMath.ray_directions_2d(8, 315, 0): start -157.5deg, step 45deg.
        dirs = cr.ray_directions()
        self.assertEqual(len(dirs), cr.N_RAYS)
        first = math.degrees(math.atan2(dirs[0][1], dirs[0][0]))
        self.assertAlmostEqual(first, -157.5, places=4)
        second = math.degrees(math.atan2(dirs[1][1], dirs[1][0]))
        self.assertAlmostEqual(second, -112.5, places=4)

    def test_golden_fixture_reproduced_exactly(self):
        # The committed fixture is the parity contract with the GDScript physics test.
        cases = json.loads(FIXTURE.read_text())["cases"]
        self.assertGreaterEqual(len(cases), 4)
        for case in cases:
            got = cr.ray_closenesses((case["agent"][0], case["agent"][1]))
            for i, (g, want) in enumerate(zip(got, case["rays"])):
                self.assertAlmostEqual(g, want, places=6,
                                       msg=f"agent {case['agent']} ray {i}")


@needs_np
class TestResolveAabb(unittest.TestCase):
    def test_pushes_out_min_axis(self):
        # 10 px into the left face, 40 from top -> min penetration is x.
        p = cr.resolve_aabb((260.0, 190.0), (250.0, 150.0, 350.0, 250.0))
        self.assertEqual(tuple(p), (250.0, 190.0))

    def test_pushes_out_y_when_shallower(self):
        p = cr.resolve_aabb((300.0, 245.0), (250.0, 150.0, 350.0, 250.0))
        self.assertEqual(tuple(p), (300.0, 250.0))

    def test_outside_unchanged(self):
        p = cr.resolve_aabb((100.0, 100.0), (250.0, 150.0, 350.0, 250.0))
        self.assertEqual(tuple(p), (100.0, 100.0))


@needs_gym
class TestChaseRaysTwinEnv(unittest.TestCase):
    def test_obs_is_13_dim_base_plus_rays(self):
        env = cr.ChaseRaysTwinEnv(seed=3)
        obs, _ = env.reset()
        self.assertEqual(obs.shape, (13,))
        self.assertEqual(env.observation_space.shape, (13,))

    def test_agent_blocked_by_obstacle(self):
        env = cr.ChaseRaysTwinEnv(seed=3)
        env.reset()
        # Place the agent just left of obstacle 0 and drive right for many steps: it must
        # never enter the box (x stays <= 250 while y is within the box's span).
        env._agent = np.array([240.0, 200.0])
        env._target = np.array([900.0, 500.0])  # far away, no catch
        env._prev_dist = float(np.hypot(*(env._target - env._agent)))
        for _ in range(5):
            env.step(4)  # +X
        self.assertLessEqual(env._agent[0], 250.0 + 1e-9)

    def test_spawns_never_inside_obstacles(self):
        env = cr.ChaseRaysTwinEnv(seed=7)
        for _ in range(50):
            env.reset()
            for pos in (env._agent, env._target):
                self.assertFalse(cr.inside_any_obstacle((pos[0], pos[1])),
                                 f"spawn {pos} inside an obstacle")


if __name__ == "__main__":
    unittest.main()
