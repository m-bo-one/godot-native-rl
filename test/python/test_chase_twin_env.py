"""Tests for the NumPy chase twin (#37): the obs/action/dynamics must match the Godot chase env
(chase_obs.gd / chase_game.gd / chase_agent.gd), else a policy trained on the twin won't deploy
back into Godot. Expected values are computed from the GDScript formulas independently (not from
the module's own helper), so the test pins the twin to Godot, not to itself."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import chase_twin_env as tw  # noqa: E402


class TestObs(unittest.TestCase):
    def test_center_on_target_is_all_zero(self):
        obs = tw.compute_obs(np.array([500.0, 300.0]), np.array([500.0, 300.0]))
        self.assertTrue(np.allclose(obs, [0, 0, 0, 0, 0]), obs)

    def test_corner_to_corner_matches_godot_formula(self):
        agent, target = np.array([0.0, 0.0]), np.array([1000.0, 600.0])
        obs = tw.compute_obs(agent, target)
        # Independently: normalized pos = (0/1000-0.5)*2 = -1, likewise y; dir = normalize(1000,600).
        dist = math.hypot(1000.0, 600.0)
        exp = [-1.0, -1.0, 1000.0 / dist, 600.0 / dist, 1.0]
        self.assertTrue(np.allclose(obs, exp, atol=1e-6), (obs, exp))

    def test_obs_dtype_and_shape(self):
        obs = tw.compute_obs(np.array([100.0, 200.0]), np.array([300.0, 400.0]))
        self.assertEqual(obs.shape, (5,))
        self.assertEqual(obs.dtype, np.float32)


class TestAction(unittest.TestCase):
    def test_velocity_table_matches_godot(self):
        s = tw.MOVE_SPEED
        self.assertTrue(np.array_equal(tw.action_to_velocity(0), [0, 0]))
        self.assertTrue(np.array_equal(tw.action_to_velocity(1), [0, -s]))
        self.assertTrue(np.array_equal(tw.action_to_velocity(2), [0, s]))
        self.assertTrue(np.array_equal(tw.action_to_velocity(3), [-s, 0]))
        self.assertTrue(np.array_equal(tw.action_to_velocity(4), [s, 0]))


@unittest.skipUnless(tw._HAVE_GYM, "gymnasium not installed")
class TestEnvDynamics(unittest.TestCase):
    def _env(self):
        env = tw.ChaseTwinEnv()
        env.reset(seed=0)
        return env

    def test_step_moves_40px_in_axis(self):
        env = self._env()
        env._agent = np.array([500.0, 300.0])
        env._target = np.array([900.0, 300.0])   # far away: no catch this step
        env._prev_dist = 400.0
        obs, reward, term, trunc, info = env.step(4)  # +x for 8 sub-frames * 5px = 40px
        self.assertAlmostEqual(env._agent[0], 540.0, places=5)
        self.assertAlmostEqual(env._agent[1], 300.0, places=5)
        self.assertFalse(term)
        self.assertEqual(info["catches"], 0)
        self.assertGreater(reward, 0.0)  # moved 40px closer -> positive progress

    def test_clamp_keeps_agent_in_bounds(self):
        env = self._env()
        env._agent = np.array([990.0, 300.0])
        env._target = np.array([10.0, 300.0])
        env._prev_dist = 980.0
        env.step(4)  # push +x past the wall
        self.assertLessEqual(env._agent[0], tw.ARENA_W)

    def test_catch_relocates_and_awards_bonus(self):
        env = self._env()
        env._agent = np.array([500.0, 300.0])
        env._target = np.array([505.0, 300.0])   # within touch_radius -> catch on first sub-frame
        env._prev_dist = 5.0
        obs, reward, term, trunc, info = env.step(0)  # even noop catches when already inside radius
        self.assertGreaterEqual(info["catches"], 1)
        self.assertGreater(reward, 0.9)  # the +1 catch bonus dominates

    def test_episode_truncates_at_max_steps(self):
        env = self._env()
        trunc = False
        for i in range(tw.MAX_STEPS):
            _, _, term, trunc, _ = env.step(0)
            if i < tw.MAX_STEPS - 1:
                self.assertFalse(trunc, "truncated early at step %d" % i)
        self.assertTrue(trunc, "did not truncate at MAX_STEPS")

    def test_seed_is_reproducible(self):
        a = tw.ChaseTwinEnv()
        obs_a, _ = a.reset(seed=123)
        b = tw.ChaseTwinEnv()
        obs_b, _ = b.reset(seed=123)
        self.assertTrue(np.allclose(obs_a, obs_b))


@unittest.skipUnless(tw._HAVE_GYM, "gymnasium not installed")
class TestGymApi(unittest.TestCase):
    def test_reset_and_step_contract(self):
        env = tw.ChaseTwinEnv()
        obs, info = env.reset(seed=1)
        self.assertEqual(obs.shape, (tw.OBS_DIM,))
        self.assertTrue(env.observation_space.contains(obs), obs)
        self.assertIsInstance(info, dict)
        out = env.step(env.action_space.sample())
        self.assertEqual(len(out), 5)
        obs2, reward, term, trunc, info2 = out
        self.assertTrue(env.observation_space.contains(obs2), obs2)
        self.assertIsInstance(float(reward), float)

    def test_passes_gymnasium_env_checker(self):
        from gymnasium.utils.env_checker import check_env
        check_env(tw.ChaseTwinEnv(), skip_render_check=True)


if __name__ == "__main__":
    unittest.main()
