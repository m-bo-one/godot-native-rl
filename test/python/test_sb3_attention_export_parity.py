import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

try:
    import numpy as np
    import torch
    import ncnn  # noqa: F401
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    _HAS = True
except Exception:
    _HAS = False

if _HAS:
    from attention_features_extractor import AttentionFeaturesExtractor
    from train_sorter_sb3 import export_sb3_sorter_policy

    class _DummyDictEnv(gym.Env):
        def __init__(self):
            self.observation_space = spaces.Dict({"obs": spaces.Box(-10, 10, (30,), np.float32)})
            self.action_space = spaces.MultiDiscrete([5])

        def reset(self, *, seed=None, options=None):
            return {"obs": np.zeros(30, np.float32)}, {}

        def step(self, a):
            return {"obs": np.zeros(30, np.float32)}, 0.0, False, False, {}


@unittest.skipUnless(_HAS, "torch/ncnn/SB3 not available")
class TestSB3AttentionExportParity(unittest.TestCase):
    def test_sb3_export_matches_ncnn(self):
        import ncnn
        torch.manual_seed(3)
        model = PPO("MultiInputPolicy", _DummyDictEnv(), n_steps=8, batch_size=8, seed=3,
                    policy_kwargs=dict(features_extractor_class=AttentionFeaturesExtractor,
                                       features_extractor_kwargs=dict(embed_dim=16, num_heads=2),
                                       net_arch=[]))
        policy = model.policy.eval()
        with tempfile.TemporaryDirectory() as td:
            param, binp = export_sb3_sorter_policy(model, td, "spike_sb3")
            net = ncnn.Net()
            net.load_param(str(param))
            net.load_model(str(binp))
            for flags in ([1] * 6, [1, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]):
                ent = torch.randn(1, 24)
                flat = torch.cat([ent, torch.tensor([flags], dtype=torch.float32)], dim=1)
                with torch.no_grad():
                    feats = policy.features_extractor({"obs": flat})
                    ref = policy.action_net(feats)[0].tolist()
                ex = net.create_extractor()
                ex.input("in0", ncnn.Mat(flat[0].numpy().copy()))
                _, out = ex.extract("out0")
                got = [out[i] for i in range(5)]
                for a, b in zip(got, ref):
                    self.assertAlmostEqual(a, b, delta=1e-2)
                self.assertEqual(max(range(5), key=lambda i: got[i]),
                                 max(range(5), key=lambda i: ref[i]))


if __name__ == "__main__":
    unittest.main()
