#!/usr/bin/env python3
"""Torch attention encoder for variable-length entity observations (#46/#258).

Numerically identical to the ncnn-verified pure-Python replica ``encoder_forward`` in
``scripts/make_synthetic_attention.py``, so a trained model exports byte-for-byte through
``scripts/export_statedict_to_ncnn.py::attention_policy_layers``. Export-safe primitives only:
one masked multi-head self-attention block over an entity embedding, then a masked mean-pool.

Load-bearing invariants (diverge from these and train/deploy silently disagree):
  * NEG_MASK is -6e4 (fp16-finite). -1e9 overflows fp16 -> -inf -> NaN on a fully-masked row on
    ARM's default fp16 path (#322).
  * Per-head scale is 1/sqrt(embed_dim / num_heads) (ncnn omits param 6 -> that default).
  * No residual, no LayerNorm; ReLU only after the embedding.
  * Linear weights use torch's [out, in] row-major layout, copied verbatim by the exporter.
"""
from __future__ import annotations

import math

import torch
from torch import nn

NEG_MASK = -6e4  # matches export_statedict_to_ncnn.FP16_SAFE_NEG_MASK


class AttentionEncoder(nn.Module):
    def __init__(self, n_entities: int, feat: int, embed_dim: int, num_heads: int,
                 global_dim: int = 0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert global_dim == 0, "global block not supported by the direct exporter (M2)"
        self.n = int(n_entities)
        self.f = int(feat)
        self.d = int(embed_dim)
        self.h = int(num_heads)
        self.dh = self.d // self.h
        self.scale = 1.0 / math.sqrt(self.dh)
        self.emb = nn.Linear(self.f, self.d)
        self.q = nn.Linear(self.d, self.d)
        self.k = nn.Linear(self.d, self.d)
        self.v = nn.Linear(self.d, self.d)
        self.out = nn.Linear(self.d, self.d)

    def forward(self, flat: torch.Tensor) -> torch.Tensor:
        b = flat.shape[0]
        nf = self.n * self.f
        ents = flat[:, :nf].reshape(b, self.n, self.f)                       # (B,N,F) row-major
        flags = flat[:, nf:nf + self.n]                                      # (B,N)
        emb = torch.relu(self.emb(ents))                                     # (B,N,D)
        q = self.q(emb).reshape(b, self.n, self.h, self.dh).transpose(1, 2)  # (B,H,N,d)
        k = self.k(emb).reshape(b, self.n, self.h, self.dh).transpose(1, 2)
        v = self.v(emb).reshape(b, self.n, self.h, self.dh).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale           # (B,H,N,N)
        addmask = (1.0 - flags) * NEG_MASK                                   # (B,N) on key dim j
        scores = scores + addmask[:, None, None, :]
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v).transpose(1, 2).reshape(b, self.n, self.d)  # (B,N,D)
        o = self.out(ctx)                                                     # (B,N,D)
        denom = torch.clamp(flags.sum(dim=1, keepdim=True), min=1.0)          # (B,1)
        pooled = (flags[:, :, None] * o).sum(dim=1) / denom                   # (B,D)
        return pooled
