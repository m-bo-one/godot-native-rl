import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import make_synthetic_attention as msa  # the ncnn-verified replica + LCG

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

# Imported unconditionally when torch is present, so a missing/broken attention_encoder is a real
# ERROR (not a silent skip). attention_encoder imports torch, so only import it when torch exists.
if _HAS_TORCH:
    from attention_encoder import AttentionEncoder, NEG_MASK


@unittest.skipUnless(_HAS_TORCH, "torch not available")
class TestAttentionEncoder(unittest.TestCase):
    def _load_encoder(self, weights, N, F, D, H):
        enc = AttentionEncoder(N, F, D, H)
        with torch.no_grad():
            enc.emb.weight.copy_(torch.tensor(weights["emb_w"]).reshape(D, F))
            enc.emb.bias.copy_(torch.tensor(weights["emb_b"]))
            for name in ("q", "k", "v", "out"):
                getattr(enc, name).weight.copy_(torch.tensor(weights[name + "_w"]).reshape(D, D))
                getattr(enc, name).bias.copy_(torch.tensor(weights[name + "_b"]))
        return enc.eval()

    def test_matches_ncnn_replica(self):
        # Reproduce write_encoder_fixture's weight draw exactly (seed 20260705, same order).
        gen = msa.lcg(20260705)
        D, F, N, H = msa.D, msa.F, msa.N, msa.HEADS
        emb_w, emb_b = msa.take(gen, D * F), msa.take(gen, D)
        qw, qb = msa.take(gen, D * D), msa.take(gen, D)
        kw, kb = msa.take(gen, D * D), msa.take(gen, D)
        vw, vb = msa.take(gen, D * D), msa.take(gen, D)
        ow, ob = msa.take(gen, D * D), msa.take(gen, D)
        weights = {"emb_w": emb_w, "emb_b": emb_b, "q_w": qw, "q_b": qb, "k_w": kw, "k_b": kb,
                   "v_w": vw, "v_b": vb, "out_w": ow, "out_b": ob}
        enc = self._load_encoder(weights, N, F, D, H)

        def flat_obs(ent_rows, flags):
            padded = ent_rows + [[0.0] * F] * (N - len(ent_rows))
            return [v for row in padded for v in row] + flags

        real2 = [msa.take(gen, F), msa.take(gen, F)]
        cases = [
            ("all_present", real2 + [msa.take(gen, F)], [1.0, 1.0, 1.0]),
            ("two_present", real2, [1.0, 1.0, 0.0]),
            ("two_present_junk", real2 + [[9.0, -9.0]], [1.0, 1.0, 0.0]),
            ("none_present", [msa.take(gen, F)], [0.0, 0.0, 0.0]),
        ]
        for name, rows, flags in cases:
            flat = flat_obs(rows, flags)
            expected = msa.encoder_forward(flat, emb_w, emb_b, qw, qb, kw, kb, vw, vb, ow, ob)
            got = enc(torch.tensor([flat], dtype=torch.float32))[0].tolist()
            for a, b in zip(got, expected):
                self.assertAlmostEqual(a, b, places=5, msg="%s mismatch" % name)

    def test_mask_invariance_and_zeros(self):
        torch.manual_seed(0)
        N, F, D, H = 4, 3, 8, 2
        enc = AttentionEncoder(N, F, D, H).eval()
        # two real entities + two padded slots; scramble padded contents -> identical output
        base = torch.randn(1, N * F + N)
        base[0, N * F:] = torch.tensor([1.0, 1.0, 0.0, 0.0])
        a = enc(base.clone())
        scrambled = base.clone()
        scrambled[0, 2 * F:4 * F] = torch.randn(2 * F)
        b = enc(scrambled)
        self.assertTrue(torch.allclose(a, b, atol=1e-5))
        # fully masked -> exact zeros, no NaN (NEG_MASK finite)
        empty = torch.randn(1, N * F + N)
        empty[0, N * F:] = 0.0
        z = enc(empty)
        self.assertFalse(torch.isnan(z).any())
        self.assertTrue(torch.allclose(z, torch.zeros_like(z), atol=1e-6))

    def test_neg_mask_is_fp16_safe(self):
        self.assertEqual(NEG_MASK, -6e4)


if __name__ == "__main__":
    unittest.main()
