"""Unit tests for the quadruped trainer's pure warm-start weight surgery (#252).

remap_widen_input copies a source policy's weights into a fresh (wider-input) policy: matching
layers are copied verbatim, the input layer is widened by keeping the source columns and zeroing the
new ones (so a hurdles net warm-started from the walk net initially ignores the 6 hurdle rays and
behaves exactly like the walk net). Torch is only needed for the tensor ops, so it's guarded."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

try:
    import torch
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestRemapWidenInput(unittest.TestCase):
    def test_surgery(self):
        from train_quadruped import remap_widen_input

        # Source: 29-dim input (walk). Target: 35-dim input (hurdles), same 64-wide hidden + heads.
        src = {
            "mlp_extractor.policy_net.0.weight": torch.arange(64 * 29).float().reshape(64, 29),
            "mlp_extractor.policy_net.0.bias": torch.arange(64).float(),
            "action_net.weight": torch.ones(8, 64),
            "stale_key": torch.zeros(3),  # present in src, absent in tgt -> ignored
        }
        tgt = {
            "mlp_extractor.policy_net.0.weight": torch.full((64, 35), -1.0),
            "mlp_extractor.policy_net.0.bias": torch.full((64,), -1.0),
            "action_net.weight": torch.full((8, 64), -1.0),
            "fresh_head.weight": torch.full((2, 64), 7.0),  # only in tgt -> kept as-is
        }

        out = remap_widen_input(src, tgt)

        # Same-shape layers copied verbatim.
        self.assertTrue(torch.equal(out["mlp_extractor.policy_net.0.bias"], src["mlp_extractor.policy_net.0.bias"]))
        self.assertTrue(torch.equal(out["action_net.weight"], src["action_net.weight"]))
        # Target-only layer kept (not clobbered).
        self.assertTrue(torch.equal(out["fresh_head.weight"], tgt["fresh_head.weight"]))
        # Widened input layer: first 29 cols == source, last 6 == 0.
        w = out["mlp_extractor.policy_net.0.weight"]
        self.assertEqual(tuple(w.shape), (64, 35))
        self.assertTrue(torch.equal(w[:, :29], src["mlp_extractor.policy_net.0.weight"]))
        self.assertTrue(torch.equal(w[:, 29:], torch.zeros(64, 6)))
        # Keys are exactly the target's (no stale source keys leaked in).
        self.assertEqual(set(out.keys()), set(tgt.keys()))


if __name__ == "__main__":
    unittest.main()
