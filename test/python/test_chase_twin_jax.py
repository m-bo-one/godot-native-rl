"""Unit tests for the JAX chase twin (#361) — numerical parity with the NumPy twin.

The transfer-critical invariant: the JAX step/obs must match chase_twin_env's math exactly
(same obs encoding, same ACTION_REPEAT sub-frame cadence, same catch/rebase semantics), or a
policy trained on the JAX batch env would not deploy. All cases are dep-probe-guarded on jax."""
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

try:
    import numpy as np
    import jax
    import jax.numpy as jnp  # noqa: F401
    HAVE_JAX = True
except ImportError:
    HAVE_JAX = False
if HAVE_JAX:
    import chase_twin_env as npt
    import chase_twin_jax as jxt

needs_jax = unittest.skipUnless(HAVE_JAX, "jax not installed")


@needs_jax
class TestObsParity(unittest.TestCase):
    def test_compute_obs_matches_numpy_twin(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            agent = rng.uniform([0, 0], [npt.ARENA_W, npt.ARENA_H])
            target = rng.uniform([0, 0], [npt.ARENA_W, npt.ARENA_H])
            want = npt.compute_obs(agent, target)
            got = np.asarray(jxt.compute_obs(jnp.asarray(agent), jnp.asarray(target)))
            np.testing.assert_allclose(got, want, atol=1e-6)

    def test_zero_distance_dir_is_zero(self):
        agent = jnp.asarray([500.0, 300.0])
        obs = np.asarray(jxt.compute_obs(agent, agent))
        self.assertEqual(obs[2], 0.0)
        self.assertEqual(obs[3], 0.0)


@needs_jax
class TestStepParity(unittest.TestCase):
    def test_deterministic_step_matches_numpy_twin(self):
        # Drive both twins from identical states with a scripted action sequence that causes NO
        # catch (catches consume different RNG streams — semantics are tested separately).
        env = npt.ChaseTwinEnv(seed=5)
        env.reset()
        env._agent = np.array([100.0, 100.0])
        env._target = np.array([900.0, 500.0])
        env._prev_dist = float(np.hypot(*(env._target - env._agent)))
        env._pending_bonus = 0.0

        state = jxt.make_state(jnp.asarray([[100.0, 100.0]]), jnp.asarray([[900.0, 500.0]]),
                               jax.random.PRNGKey(0))
        actions = [4, 4, 2, 1, 3, 0, 4, 2]
        for a in actions:
            _, np_reward, _, _, _ = env.step(a)
            state, jx_obs, jx_reward, _ = jxt.batched_step(state, jnp.asarray([a]))
            np.testing.assert_allclose(float(jx_reward[0]), np_reward, atol=1e-5)
            np.testing.assert_allclose(np.asarray(state.agent[0]), env._agent, atol=1e-4)
        # Final obs parity too.
        want = npt.compute_obs(env._agent, env._target)
        np.testing.assert_allclose(np.asarray(jx_obs[0]), want, atol=1e-5)

    def test_catch_relocates_and_pays_bonus_next_subframe(self):
        # Start one touch away: the catch must relocate the target (inside the arena) and the
        # +1 bonus must land (folded into the same env-step's remaining sub-frames).
        state = jxt.make_state(jnp.asarray([[500.0, 300.0]]), jnp.asarray([[530.0, 300.0]]),
                               jax.random.PRNGKey(1))
        state, _, reward, catches = jxt.batched_step(state, jnp.asarray([4]))  # move +X
        self.assertEqual(int(catches[0]), 1)
        self.assertGreater(float(reward[0]), 0.5)  # the touch bonus dominates
        t = np.asarray(state.target[0])
        self.assertTrue(0.0 <= t[0] <= npt.ARENA_W and 0.0 <= t[1] <= npt.ARENA_H)

    def test_batched_over_many_envs(self):
        n = 32
        key = jax.random.PRNGKey(2)
        state = jxt.reset_batch(n, key)
        for _ in range(3):
            key, sub = jax.random.split(key)
            acts = jax.random.randint(sub, (n,), 0, npt.N_ACTIONS)
            state, obs, reward, _ = jxt.batched_step(state, acts)
        self.assertEqual(obs.shape, (n, npt.OBS_DIM))
        self.assertEqual(reward.shape, (n,))
        agents = np.asarray(state.agent)
        self.assertTrue((agents[:, 0] >= 0).all() and (agents[:, 0] <= npt.ARENA_W).all())


try:
    import torch  # noqa: F401
    HAVE_TORCH_TOO = HAVE_JAX
except ImportError:
    HAVE_TORCH_TOO = False

needs_jax_torch = unittest.skipUnless(HAVE_TORCH_TOO, "jax+torch not installed")


@needs_jax_torch
class TestActorExportParity(unittest.TestCase):
    def test_flax_params_to_torch_actor_matches(self):
        # The deploy seam: the trained flax actor and the exported torch module must emit the
        # SAME logits, or the ncnn net would not be the trained policy.
        import torch
        import train_chase_jax as tj

        actor = tj.Actor(n_actions=npt.N_ACTIONS)
        params = actor.init(jax.random.PRNGKey(0), jnp.zeros((1, npt.OBS_DIM)))
        torch_actor = tj.params_to_torch_actor(params, npt.OBS_DIM, npt.N_ACTIONS)
        x = np.random.default_rng(0).normal(size=(6, npt.OBS_DIM)).astype(np.float32)
        want = np.asarray(actor.apply(params, jnp.asarray(x)))
        with torch.no_grad():
            got = torch_actor(torch.from_numpy(x)).numpy()
        np.testing.assert_allclose(got, want, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
