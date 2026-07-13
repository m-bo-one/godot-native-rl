"""Unit tests for the pure helpers of scripts/train_skrl.py (#25, SKRL backend).

The module must import without skrl/torch/gymnasium (lazy heavy imports); the net builders
need torch, so those cases are dep-probe-guarded (#141/#148)."""
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import train_skrl as ts  # noqa: E402

try:
    import torch  # noqa: F401, E402
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

needs_torch = unittest.skipUnless(HAVE_TORCH, "torch not installed")


class TestParseArgs(unittest.TestCase):
    def test_defaults(self):
        cfg = ts.parse_args([])
        self.assertEqual(cfg.base_port, 11008)
        self.assertEqual(cfg.timesteps, 200_000)
        self.assertEqual(cfg.out, "models/chase_skrl_policy.pt")
        self.assertEqual(cfg.rollouts, 512)

    def test_overrides(self):
        cfg = ts.parse_args(["--timesteps", "3000", "--rollouts", "64", "--out", "/tmp/x.pt"])
        self.assertEqual(cfg.timesteps, 3000)
        self.assertEqual(cfg.rollouts, 64)
        self.assertEqual(cfg.out, "/tmp/x.pt")


class TestPPOCfgKwargs(unittest.TestCase):
    def test_maps_config_to_skrl_names(self):
        cfg = ts.parse_args(["--rollouts", "128"])
        kw = ts.ppo_cfg_kwargs(cfg)
        # skrl PPO_CFG field names (introspected against skrl 2.1).
        self.assertEqual(kw["rollouts"], 128)
        self.assertEqual(kw["discount_factor"], 0.99)
        self.assertEqual(kw["learning_rate"], 2.5e-4)
        self.assertGreater(kw["entropy_loss_scale"], 0.0)
        self.assertGreaterEqual(kw["mini_batches"], 1)


class TestNets(unittest.TestCase):
    @needs_torch
    def test_actor_net_shapes(self):
        import torch as t
        net = ts.actor_net(5, 5)
        out = net(t.zeros(3, 5))
        self.assertEqual(tuple(out.shape), (3, 5))

    @needs_torch
    def test_value_net_shapes(self):
        import torch as t
        net = ts.value_net(5)
        out = net(t.zeros(3, 5))
        self.assertEqual(tuple(out.shape), (3, 1))


if __name__ == "__main__":
    unittest.main()
