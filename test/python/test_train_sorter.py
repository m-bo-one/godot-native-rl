import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

if _HAS_TORCH:
    from train_sorter_cleanrl import build_sorter_agent


@unittest.skipUnless(_HAS_TORCH, "torch not available")
class TestSorterAgent(unittest.TestCase):
    def test_agent_shapes(self):
        agent = build_sorter_agent(obs_dim=30, n_act=5, embed_dim=16, num_heads=2)
        obs = torch.zeros(4, 30)
        logits = agent.logits(obs)
        value = agent.value(obs)
        self.assertEqual(tuple(logits.shape), (4, 5))
        self.assertEqual(tuple(value.shape), (4,))
        self.assertEqual(agent.encoder.n * agent.encoder.f + agent.encoder.n, 30)

    def test_obs_dim_mismatch_raises(self):
        with self.assertRaises(AssertionError):
            build_sorter_agent(obs_dim=31, n_act=5, embed_dim=16, num_heads=2)


if __name__ == "__main__":
    unittest.main()
