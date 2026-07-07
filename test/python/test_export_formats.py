import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from export_formats import resolve_formats  # pure, no torch

try:
    import numpy as np  # noqa: F401
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    _HAS_SB3 = True
except Exception:
    _HAS_SB3 = False

if _HAS_SB3:
    from export_formats import export_policy

    # godot_rl obs convention: a Dict {"obs": Box} -> MultiInputPolicy. godot_rl's
    # export_model_as_onnx is written for this dict space, so the test env must match it.
    class _DummyDictEnv(gym.Env):
        def __init__(self):
            self.observation_space = spaces.Dict({"obs": spaces.Box(-1, 1, (5,), "float32")})
            self.action_space = spaces.MultiDiscrete([3])

        def reset(self, *, seed=None, options=None):
            return self.observation_space.sample(), {}

        def step(self, a):
            return self.observation_space.sample(), 0.0, False, False, {}


class TestResolveFormats(unittest.TestCase):
    def test_onnx(self):
        self.assertEqual(resolve_formats("onnx"), ("onnx",))

    def test_torchscript(self):
        self.assertEqual(resolve_formats("torchscript"), ("torchscript",))

    def test_both(self):
        self.assertEqual(resolve_formats("both"), ("onnx", "torchscript"))

    def test_invalid_raises(self):
        with self.assertRaises(ValueError):
            resolve_formats("tensorflow")


@unittest.skipUnless(_HAS_SB3, "SB3/torch not available")
class TestExportPolicy(unittest.TestCase):
    def _model(self):
        return PPO("MultiInputPolicy", _DummyDictEnv(), n_steps=8, batch_size=8)

    def test_torchscript_writes_pt_and_sidecar(self):
        with tempfile.TemporaryDirectory() as td:
            stem = str(Path(td) / "policy")
            written = export_policy(self._model(), stem, "torchscript")
            self.assertTrue((Path(td) / "policy.pt").is_file())
            self.assertTrue((Path(td) / "policy.pt.shape.json").is_file())
            self.assertFalse((Path(td) / "policy.onnx").is_file())
            self.assertIn(Path(td) / "policy.pt", written)

    def test_both_writes_onnx_and_pt(self):
        with tempfile.TemporaryDirectory() as td:
            stem = str(Path(td) / "policy")
            export_policy(self._model(), stem, "both")
            self.assertTrue((Path(td) / "policy.onnx").is_file())
            self.assertTrue((Path(td) / "policy.pt").is_file())
            self.assertTrue((Path(td) / "policy.pt.shape.json").is_file())


if __name__ == "__main__":
    unittest.main()
