#!/usr/bin/env python3
"""M2 export-parity gate + pnnx-emission probe (#258).

Primary deploy path (approved design): the direct hand-written exporter
``export_statedict_to_ncnn.attention_policy_layers`` — already production and byte-pinned to
ncnn-verified fixtures (#307). This script exports a torch ``AttentionEncoder`` + linear actor
head through it and asserts torch-vs-ncnn parity, then ADDITIONALLY probes whether pnnx can emit
the graph from the traced module (logged for the design note, NOT a gate).

Run: .venv-train/bin/python scripts/spike_attention_ncnn.py
"""
from __future__ import annotations

from pathlib import Path

import export_statedict_to_ncnn as sd


def encoder_weights_dict(enc) -> dict:
    """Map an AttentionEncoder's parameters to the attention_policy_layers weight-key contract.

    torch nn.Linear.weight is [out, in] row-major = the exporter's expected layout, so a flat
    ``.tolist()`` is copied verbatim (no transpose).
    """
    def flat(t):
        return t.detach().cpu().flatten().tolist()
    return {
        "emb_w": flat(enc.emb.weight), "emb_b": flat(enc.emb.bias),
        "q_w": flat(enc.q.weight), "q_b": flat(enc.q.bias),
        "k_w": flat(enc.k.weight), "k_b": flat(enc.k.bias),
        "v_w": flat(enc.v.weight), "v_b": flat(enc.v.bias),
        "out_w": flat(enc.out.weight), "out_b": flat(enc.out.bias),
    }


def export_encoder_policy(enc, head_weight, head_bias, outdir, stem="sorter_attention"):
    """Write ``<outdir>/<stem>.ncnn.{param,bin}`` for enc + a single linear actor head.

    ``head_weight`` is [n_act, embed_dim], ``head_bias`` is [n_act]. Returns (param, bin) paths.
    """
    weights = encoder_weights_dict(enc)
    weights["head0_w"] = head_weight.detach().cpu().flatten().tolist()
    weights["head0_b"] = head_bias.detach().cpu().flatten().tolist()
    layers = sd.attention_policy_layers(enc.n, enc.f, enc.d, enc.h, weights,
                                        head_dims=[len(weights["head0_b"])])
    outdir = Path(outdir)
    param = outdir / ("%s.ncnn.param" % stem)
    binp = outdir / ("%s.ncnn.bin" % stem)
    param.write_text(sd.ncnn_param_text(layers))
    binp.write_bytes(sd.ncnn_bin_bytes(layers))
    return param, binp


def _probe_pnnx(enc) -> str:
    """Trace the encoder and try pnnx; return a status string (logged, NOT a gate)."""
    try:
        import subprocess
        import tempfile

        import torch
        obs_dim = enc.n * enc.f + enc.n
        with tempfile.TemporaryDirectory() as td:
            pt = Path(td) / "enc.pt"
            traced = torch.jit.trace(enc.eval(), torch.randn(1, obs_dim))
            traced.save(str(pt))
            # Absolute path: the subprocess runs with cwd=td, so a relative pnnx path would miss.
            pnnx = (Path(__file__).resolve().parents[1] / ".venv" / "bin" / "pnnx")
            if not pnnx.exists():
                return "pnnx-absent"
            r = subprocess.run([str(pnnx), str(pt), "inputshape=[1,%d]" % obs_dim],
                               cwd=td, capture_output=True, text=True, timeout=180)
            param = Path(td) / "enc.ncnn.param"
            if r.returncode != 0 or not param.exists():
                return "pnnx-failed(rc=%s)" % r.returncode
            text = param.read_text()
            return "pnnx-emitted-MHA" if "MultiHeadAttention" in text else "pnnx-decomposed"
    except Exception as e:  # noqa: BLE001
        return "pnnx-error:%s" % type(e).__name__


def main() -> None:
    import tempfile

    import torch

    from attention_encoder import AttentionEncoder

    N, F, D, H, A = 6, 4, 16, 2, 5
    enc = AttentionEncoder(N, F, D, H).eval()
    head = torch.nn.Linear(D, A)
    with tempfile.TemporaryDirectory() as td:
        param, binp = export_encoder_policy(enc, head.weight, head.bias, td, "spike")
        import ncnn
        net = ncnn.Net()
        net.load_param(str(param))
        net.load_model(str(binp))
        worst = 0.0
        for flags in ([1] * 6, [1, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]):
            flat = torch.cat([torch.randn(1, N * F),
                              torch.tensor([flags], dtype=torch.float32)], dim=1)
            with torch.no_grad():
                ref = head(enc(flat))[0].tolist()
            ex = net.create_extractor()
            ex.input("in0", ncnn.Mat(flat[0].numpy().copy()))
            _, out = ex.extract("out0")
            worst = max(worst, max(abs(out[i] - ref[i]) for i in range(A)))
        print("DIRECT-EXPORT parity worst abs err: %.2e (gate: <1e-2)" % worst)
        print("PNNX-EMISSION probe:", _probe_pnnx(enc))
        assert worst < 1e-2, "direct export parity failed"
        print("SPIKE OK: direct hand-written export is the M2 deploy path.")


if __name__ == "__main__":
    main()
