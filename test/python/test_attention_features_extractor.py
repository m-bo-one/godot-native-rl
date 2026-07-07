import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

try:
    import numpy as np
    import torch
    from gymnasium import spaces
    _HAS = True
except Exception:
    _HAS = False

if _HAS:
    from attention_features_extractor import AttentionFeaturesExtractor
    from attention_encoder import AttentionEncoder


@unittest.skipUnless(_HAS, "torch/gymnasium not available")
class TestAttentionFeaturesExtractor(unittest.TestCase):
    def _space(self, width=30):
        return spaces.Dict({"obs": spaces.Box(-10, 10, (width,), np.float32)})

    def test_shapes_and_encoder(self):
        ext = AttentionFeaturesExtractor(self._space(30), embed_dim=16, num_heads=2)
        self.assertEqual(ext.features_dim, 16)
        out = ext({"obs": torch.zeros(4, 30)})
        self.assertEqual(tuple(out.shape), (4, 16))
        self.assertIsInstance(ext.encoder, AttentionEncoder)
        self.assertEqual(ext.encoder.n * ext.encoder.f + ext.encoder.n, 30)

    def test_plain_box_space(self):
        ext = AttentionFeaturesExtractor(spaces.Box(-10, 10, (30,), np.float32),
                                         embed_dim=16, num_heads=2)
        out = ext(torch.zeros(2, 30))
        self.assertEqual(tuple(out.shape), (2, 16))

    def test_width_mismatch_raises(self):
        with self.assertRaises(AssertionError):
            AttentionFeaturesExtractor(self._space(31), embed_dim=16, num_heads=2)


if __name__ == "__main__":
    unittest.main()
