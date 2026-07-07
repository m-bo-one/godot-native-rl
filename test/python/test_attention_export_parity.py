import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

try:
    import ncnn  # noqa: F401
    _HAS_NCNN = True
except Exception:
    _HAS_NCNN = False

if _HAS_TORCH and _HAS_NCNN:
    from attention_encoder import AttentionEncoder
    from spike_attention_ncnn import encoder_weights_dict, export_encoder_policy  # noqa: F401


@unittest.skipUnless(_HAS_TORCH and _HAS_NCNN, "torch/ncnn not available")
class TestAttentionExportParity(unittest.TestCase):
    def test_torch_vs_ncnn_parity_with_head(self):
        import ncnn
        torch.manual_seed(1)
        N, F, D, H, A = 6, 4, 16, 2, 5
        enc = AttentionEncoder(N, F, D, H).eval()
        head = torch.nn.Linear(D, A)
        with torch.no_grad():
            torch.nn.init.normal_(head.weight, std=0.3)
        with tempfile.TemporaryDirectory() as td:
            param, binp = export_encoder_policy(
                enc, head.weight.detach(), head.bias.detach(), td, "spike")
            net = ncnn.Net()
            net.load_param(str(param))
            net.load_model(str(binp))
            for flags in ([1, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]):
                ent = torch.randn(1, N * F)
                flat = torch.cat([ent, torch.tensor([flags], dtype=torch.float32)], dim=1)
                with torch.no_grad():
                    ref = head(enc(flat))[0].tolist()
                ex = net.create_extractor()
                ex.input("in0", ncnn.Mat(flat[0].numpy().copy()))
                _, out = ex.extract("out0")
                got = [out[i] for i in range(A)]
                for a, b in zip(got, ref):
                    self.assertAlmostEqual(a, b, delta=1e-2)
                self.assertEqual(max(range(A), key=lambda i: got[i]),
                                 max(range(A), key=lambda i: ref[i]), "argmax parity")


if __name__ == "__main__":
    unittest.main()
