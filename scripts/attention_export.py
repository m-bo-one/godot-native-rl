#!/usr/bin/env python3
"""Production ncnn export for attention-encoder policies (#46 M2/M3/M4).

The deploy path for every attention-encoder trainer (CleanRL, SB3, BC) goes through here: it maps
a torch ``AttentionEncoder`` + a single linear actor head onto the direct
``export_statedict_to_ncnn.attention_policy_layers`` writer (NOT ONNX/pnnx — pnnx decomposes the
hand-built attention; the direct path is byte-pinned to the #307 ncnn-verified fixtures). Split out
of the former ``spike_attention_ncnn`` module so shipped deploy code no longer lives in a file whose
role is a one-off parity probe.
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
