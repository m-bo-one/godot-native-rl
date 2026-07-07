# Attention Encoder M2 (torch side) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Ship the torch-side attention encoder, a CleanRL Sorter trainer, and an export-parity spike that plugs into the already-proven ncnn deploy contract (#258).

**Architecture:** A torch `AttentionEncoder` numerically identical to the ncnn-verified pure-Python replica (`encoder_forward` in `scripts/make_synthetic_attention.py`), exported to ncnn via the existing production `attention_policy_layers` writer (NOT ONNX/pnnx). A CleanRL PPO trainer reuses `train_cleanrl.py` with the encoder as the agent trunk.

**Tech Stack:** Python 3.13 (`.venv-train` torch 2.12), the `ncnn` pip module, stdlib `unittest` (torch-gated), the existing `scripts/export_statedict_to_ncnn.py` writer, `CleanRLGodotEnv` from godot-rl.

## Global Constraints

- Python: **4-space** indentation; tests are stdlib `unittest` under `test/python/`, auto-discovered by `run_tests.sh`; heavy imports (torch) **lazy inside functions/`main()`** so pure helpers stay importable without torch. Torch tests guard with `@unittest.skipUnless(_HAS_TORCH, ...)`.
- **Mask constant = `-6e4`** (`sd.FP16_SAFE_NEG_MASK`), never `-inf`/`-1e9`.
- **Per-head scale `1/√d`**, `d = embed_dim/num_heads`; `embed_dim % num_heads == 0`.
- **Weight layout = torch `nn.Linear.weight` `[out,in]` row-major**, `.flatten().tolist()` verbatim, no transpose. `emb`=`(D,F)`; `q/k/v/out`=`(D,D)`; `head0`=`(5,D)`.
- Obs reshape is **row-major entity-major**: `flat[:, :N*F].reshape(B,N,F)`, `flags=flat[:, N*F:]`.
- Export via `sd.attention_policy_layers(N, F, D, H, weights, head_dims=[5])` → blobs `in0`/`out0`.
- Full `./test/run_tests.sh` green before push.

---

### Task 1: `scripts/attention_encoder.py` — torch encoder + parity anchor

**Files:**
- Create: `scripts/attention_encoder.py`
- Test: `test/python/test_attention_encoder.py`

**Interfaces:**
- Produces: `class AttentionEncoder(nn.Module)` with `__init__(self, n_entities, feat, embed_dim, num_heads, global_dim=0)`, submodules `emb,q,k,v,out` (`nn.Linear`), `forward(flat: Tensor[B, n_entities*feat + n_entities]) -> Tensor[B, embed_dim]`. Consumed by Tasks 2–3.
- Produces: `NEG_MASK = -6e4` module constant.

- [ ] **Step 1: Write the failing parity test**

Create `test/python/test_attention_encoder.py`. It draws the SAME weights the fixture generator's `write_encoder_fixture` draws (LCG seed 20260705, in the same order) and asserts the torch module matches the ncnn-verified replica `encoder_forward` to 1e-5 on the fixture's four obs cases, plus invariance properties.

```python
import math
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import make_synthetic_attention as msa  # the ncnn-verified replica + LCG

try:
    import torch
    from attention_encoder import AttentionEncoder, NEG_MASK
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


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
        scrambled = base.clone(); scrambled[0, 2 * F:4 * F] = torch.randn(2 * F)
        b = enc(scrambled)
        self.assertTrue(torch.allclose(a, b, atol=1e-5))
        # fully masked -> exact zeros, no NaN (NEG_MASK finite)
        empty = torch.randn(1, N * F + N); empty[0, N * F:] = 0.0
        z = enc(empty)
        self.assertFalse(torch.isnan(z).any())
        self.assertTrue(torch.allclose(z, torch.zeros_like(z), atol=1e-6))

    def test_neg_mask_is_fp16_safe(self):
        self.assertEqual(NEG_MASK, -6e4)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv-train/bin/python -m unittest test.python.test_attention_encoder -v`
Expected: FAIL — `ImportError`/`ModuleNotFoundError` for `attention_encoder` (module not created).

- [ ] **Step 3: Implement `scripts/attention_encoder.py`**

```python
#!/usr/bin/env python3
"""Torch attention encoder for variable-length entity obs (#46/#258).

Numerically identical to the ncnn-verified pure-Python replica `encoder_forward` in
scripts/make_synthetic_attention.py, so a trained model exports byte-for-byte through
scripts/export_statedict_to_ncnn.py::attention_policy_layers. Export-safe primitives only:
one masked multi-head self-attention block over an entity embedding, masked mean-pool.

NEG_MASK is -6e4 (fp16-finite): -1e9 overflows fp16 -> -inf -> NaN on a fully-masked row on
ARM deploys. Per-head scale is 1/sqrt(embed_dim/num_heads). No residual, no LayerNorm.
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
        ents = flat[:, :nf].reshape(b, self.n, self.f)      # (B,N,F) row-major
        flags = flat[:, nf:nf + self.n]                     # (B,N)
        emb = torch.relu(self.emb(ents))                    # (B,N,D)
        q = self.q(emb).reshape(b, self.n, self.h, self.dh).transpose(1, 2)  # (B,H,N,d)
        k = self.k(emb).reshape(b, self.n, self.h, self.dh).transpose(1, 2)
        v = self.v(emb).reshape(b, self.n, self.h, self.dh).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale           # (B,H,N,N)
        addmask = (1.0 - flags) * NEG_MASK                                   # (B,N) on key dim
        scores = scores + addmask[:, None, None, :]
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v).transpose(1, 2).reshape(b, self.n, self.d)  # (B,N,D)
        o = self.out(ctx)                                                    # (B,N,D)
        denom = torch.clamp(flags.sum(dim=1, keepdim=True), min=1.0)         # (B,1)
        pooled = (flags[:, :, None] * o).sum(dim=1) / denom                  # (B,D)
        return pooled
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv-train/bin/python -m unittest test.python.test_attention_encoder -v`
Expected: PASS (all 3 tests). If `test_matches_ncnn_replica` fails, the reshape/scale/mask math diverges from the replica — fix before proceeding (this is the load-bearing anchor).

- [ ] **Step 5: Commit**

```bash
git add scripts/attention_encoder.py test/python/test_attention_encoder.py
git commit -m "feat(#258): torch AttentionEncoder matching the ncnn-verified replica"
```

---

### Task 2: `scripts/spike_attention_ncnn.py` — export-parity gate + pnnx probe

**Files:**
- Create: `scripts/spike_attention_ncnn.py`
- Test: `test/python/test_attention_export_parity.py`

**Interfaces:**
- Produces: `export_encoder_policy(enc, head_w, head_b, outdir, stem) -> (param_path, bin_path)` — maps a trained `AttentionEncoder` + actor head to ncnn via `attention_policy_layers(head_dims=[len(head_b)])`.
- Produces: `encoder_weights_dict(enc) -> dict` — pure `state_dict → weight-key` mapper (keys: `emb_w/emb_b/q_w/...` per the exporter contract).

- [ ] **Step 1: Write the failing export-parity test**

Create `test/python/test_attention_export_parity.py`:

```python
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

try:
    import torch
    import ncnn  # noqa: F401
    from attention_encoder import AttentionEncoder
    from spike_attention_ncnn import encoder_weights_dict, export_encoder_policy
    _HAS = True
except Exception:
    _HAS = False


@unittest.skipUnless(_HAS, "torch/ncnn not available")
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
            net.load_param(str(param)); net.load_model(str(binp))
            # a few obs incl. padded rows
            for flags in ([1, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]):
                ent = torch.randn(1, N * F)
                flat = torch.cat([ent, torch.tensor([flags], dtype=torch.float32)], dim=1)
                with torch.no_grad():
                    ref = head(enc(flat))[0].tolist()
                ex = net.create_extractor()
                mat = ncnn.Mat(flat[0].numpy().copy())
                ex.input("in0", mat)
                _, out = ex.extract("out0")
                got = [out[i] for i in range(A)]
                for a, b in zip(got, ref):
                    self.assertAlmostEqual(a, b, delta=1e-2)
                self.assertEqual(max(range(A), key=lambda i: got[i]),
                                 max(range(A), key=lambda i: ref[i]), "argmax parity")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv-train/bin/python -m unittest test.python.test_attention_export_parity -v`
Expected: FAIL — `spike_attention_ncnn` not importable.

- [ ] **Step 3: Implement `scripts/spike_attention_ncnn.py`**

```python
#!/usr/bin/env python3
"""M2 export-parity gate + pnnx-emission probe (#258).

Primary deploy path (approved): the direct hand-written exporter
(export_statedict_to_ncnn.attention_policy_layers) — already production and byte-pinned to
ncnn-verified fixtures. This script exports a torch AttentionEncoder+head through it and asserts
torch-vs-ncnn parity, then ADDITIONALLY probes whether pnnx can emit the graph (logged, not a
gate). Run: .venv-train/bin/python scripts/spike_attention_ncnn.py
"""
from __future__ import annotations

from pathlib import Path

import export_statedict_to_ncnn as sd


def encoder_weights_dict(enc) -> dict:
    """Map an AttentionEncoder's parameters to the attention_policy_layers weight-key contract."""
    def flat(t):
        return t.detach().cpu().flatten().tolist()
    w = {
        "emb_w": flat(enc.emb.weight), "emb_b": flat(enc.emb.bias),
        "q_w": flat(enc.q.weight), "q_b": flat(enc.q.bias),
        "k_w": flat(enc.k.weight), "k_b": flat(enc.k.bias),
        "v_w": flat(enc.v.weight), "v_b": flat(enc.v.bias),
        "out_w": flat(enc.out.weight), "out_b": flat(enc.out.bias),
    }
    return w


def export_encoder_policy(enc, head_weight, head_bias, outdir, stem="sorter_attention"):
    """Write <outdir>/<stem>.ncnn.{param,bin} for enc + a single linear actor head."""
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


def _probe_pnnx(enc, log) -> str:
    """Trace the encoder and try pnnx; return a status string (logged, not a gate)."""
    try:
        import subprocess
        import tempfile
        import torch
        with tempfile.TemporaryDirectory() as td:
            pt = Path(td) / "enc.pt"
            example = torch.randn(1, enc.n * enc.f + enc.n)
            traced = torch.jit.trace(enc.eval(), example)
            traced.save(str(pt))
            pnnx = Path(".venv/bin/pnnx")
            if not pnnx.exists():
                return "pnnx-absent"
            r = subprocess.run([str(pnnx), str(pt),
                                "inputshape=[1,%d]" % (enc.n * enc.f + enc.n)],
                               cwd=td, capture_output=True, text=True, timeout=180)
            param = Path(td) / "enc.ncnn.param"
            if r.returncode != 0 or not param.exists():
                return "pnnx-failed"
            text = param.read_text()
            return "pnnx-emitted-MHA" if "MultiHeadAttention" in text else "pnnx-decomposed"
    except Exception as e:  # noqa: BLE001
        return "pnnx-error:%s" % type(e).__name__


def main() -> None:
    import torch
    N, F, D, H, A = 6, 4, 16, 2, 5
    from attention_encoder import AttentionEncoder
    enc = AttentionEncoder(N, F, D, H).eval()
    head = torch.nn.Linear(D, A)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        param, binp = export_encoder_policy(enc, head.weight, head.bias, td, "spike")
        import ncnn
        net = ncnn.Net(); net.load_param(str(param)); net.load_model(str(binp))
        worst = 0.0
        for flags in ([1] * 6, [1, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]):
            flat = torch.cat([torch.randn(1, N * F),
                              torch.tensor([flags], dtype=torch.float32)], dim=1)
            with torch.no_grad():
                ref = head(enc(flat))[0].tolist()
            ex = net.create_extractor(); ex.input("in0", ncnn.Mat(flat[0].numpy().copy()))
            _, out = ex.extract("out0")
            worst = max(worst, max(abs(out[i] - ref[i]) for i in range(A)))
        print("DIRECT-EXPORT parity worst abs err: %.2e (gate: <1e-2)" % worst)
        print("PNNX-EMISSION probe:", _probe_pnnx(enc, print))
        assert worst < 1e-2, "direct export parity failed"
        print("SPIKE OK: direct export is the M2 deploy path.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test + the spike script to verify parity**

Run: `.venv-train/bin/python -m unittest test.python.test_attention_export_parity -v`
Expected: PASS.
Run: `.venv-train/bin/python scripts/spike_attention_ncnn.py`
Expected: prints `DIRECT-EXPORT parity worst abs err: <small> (gate: <1e-2)`, a `PNNX-EMISSION probe:` line (record its verdict for the docs), and `SPIKE OK`.

- [ ] **Step 5: Commit**

```bash
git add scripts/spike_attention_ncnn.py test/python/test_attention_export_parity.py
git commit -m "feat(#258): attention export-parity spike + pnnx probe (direct export gate)"
```

---

### Task 3: `scripts/train_sorter_cleanrl.py` + `scripts/train_sorter.sh`

**Files:**
- Create: `scripts/train_sorter_cleanrl.py`
- Create: `scripts/train_sorter.sh`
- Reference (read, do not modify): `scripts/train_cleanrl.py`, `scripts/train_cleanrl.sh`
- Test: `test/python/test_train_sorter.py`

**Interfaces:**
- Consumes: `AttentionEncoder` (Task 1), `export_encoder_policy` (Task 2).
- Produces: a trainer that reads `CleanRLGodotEnv`, uses the encoder trunk, and on completion writes `<stem>.ncnn.{param,bin}` via `export_encoder_policy`.

- [ ] **Step 1: Read the reference trainer to mirror its structure**

Run: `sed -n '1,60p;180,210p;280,480p' scripts/train_cleanrl.py`  (understand `PPOConfig`, `_build_agent`, obs/act dims, the rollout+update loop, and the ONNX export block to replace).

- [ ] **Step 2: Write a pure-helper test (no live env)**

Create `test/python/test_train_sorter.py` — tests the one pure seam that doesn't need a socket: that `build_sorter_agent(obs_dim, n_act)` produces an agent whose `get_action_and_value(obs)` returns logits of width `n_act` and a scalar value, and whose encoder is an `AttentionEncoder` with `n*f+n == obs_dim`.

```python
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

try:
    import torch
    from train_sorter_cleanrl import build_sorter_agent
    _HAS = True
except Exception:
    _HAS = False


@unittest.skipUnless(_HAS, "torch not available")
class TestSorterAgent(unittest.TestCase):
    def test_agent_shapes(self):
        agent = build_sorter_agent(obs_dim=30, n_act=5, embed_dim=16, num_heads=2)
        obs = torch.zeros(4, 30)
        logits = agent.actor(agent.encoder(obs))
        value = agent.critic(agent.encoder(obs))
        self.assertEqual(tuple(logits.shape), (4, 5))
        self.assertEqual(tuple(value.shape), (4, 1))
        self.assertEqual(agent.encoder.n * agent.encoder.f + agent.encoder.n, 30)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run it to verify it fails**

Run: `.venv-train/bin/python -m unittest test.python.test_train_sorter -v`
Expected: FAIL — `train_sorter_cleanrl` not importable.

- [ ] **Step 4: Implement `scripts/train_sorter_cleanrl.py`**

Copy `train_cleanrl.py` as the base. Keep `PPOConfig`, `compute_gae`, `layer_init`, `_split_categoricals`, arg parsing, and the rollout/PPO-update loop verbatim. Make these changes:
- Add `from attention_encoder import AttentionEncoder` (lazy, inside `main`/builders).
- Add a top-level `build_sorter_agent(obs_dim, n_act, embed_dim=16, num_heads=2)` returning an `nn.Module` with `.encoder=AttentionEncoder(n=(obs_dim - n_ent)//f ...)`. Derive `N,F`: for Sorter `obs_dim=30, N=6, F=4` — pass `N,F` explicitly via config (`--n_entities 6 --feat 4`), default those, and assert `N*F+N==obs_dim`. `.actor=layer_init(nn.Linear(embed_dim, n_act), std=0.01)`, `.critic=layer_init(nn.Linear(embed_dim,1), std=1.0)`. `get_action_and_value` mirrors `train_cleanrl.py` but calls `self.encoder(obs)` for the shared features.
- Replace the ONNX export block with: build `weights` via `export_encoder_policy(agent.encoder, agent.actor.weight, agent.actor.bias, outdir, stem)` (Task 2). Save the `.pt` too for debugging. Default `SCENE`/stem for sorter.

(Reuse everything else. The env still yields flat `(B,30)` obs; the encoder reshapes internally.)

- [ ] **Step 5: Run the pure test to verify it passes**

Run: `.venv-train/bin/python -m unittest test.python.test_train_sorter -v`
Expected: PASS.

- [ ] **Step 6: Implement `scripts/train_sorter.sh`**

Clone `scripts/train_cleanrl.sh`; set `SCENE="${SCENE:-res://examples/sorter/sorter_train_parallel.tscn}"`, retarget `SAVE_MODEL_PATH`/export stems to `models/sorter_attention*`, keep the `TIMESTEPS < NUM_STEPS*num_envs` loud-exit guard (mirror `train_pettingzoo.sh`/#119 if present, else add a simple check). `bash -n` clean.

Run: `bash -n scripts/train_sorter.sh && echo OK`

- [ ] **Step 7: Commit**

```bash
git add scripts/train_sorter_cleanrl.py scripts/train_sorter.sh test/python/test_train_sorter.py
git commit -m "feat(#258): CleanRL Sorter trainer with attention encoder trunk + direct ncnn export"
```

---

### Task 4: End-to-end training smoke (short live run) + `run_tests.sh` wiring

**Files:**
- Modify: `test/run_tests.sh`

**Interfaces:** consumes Tasks 1–3.

- [ ] **Step 1: Prove a real short run trains + exports (manual, background)**

Run a ~2000-step run against `sorter_train_parallel.tscn` into a temp dir; confirm it exits 0 and writes `sorter_attention.ncnn.{param,bin}`. (Background it like the #198 smoke; the run takes a couple minutes.) If the encoder mistrains/export fails, debug with `systematic-debugging` before wiring CI.

- [ ] **Step 2: Add a guarded smoke block to `run_tests.sh`**

In the godot_rl-gated cluster (mirror the CleanRL+RND block), add:

```bash
echo "== Sorter attention-encoder trainer smoke (skipped if godot_rl absent in .venv-train) =="
if [ -x .venv-train/bin/python ] && .venv-train/bin/python -c "import godot_rl" >/dev/null 2>&1; then
	SORTER_TMP="$(mktemp -d)"
	TIMESTEPS="${SORTER_SMOKE_TIMESTEPS:-2000}" \
	SAVE_MODEL_PATH="$SORTER_TMP/sorter_attention.pt" \
	OUTDIR="$SORTER_TMP" \
		./scripts/train_sorter.sh
	test -f "$SORTER_TMP/sorter_attention.ncnn.param" || { echo "FAIL: Sorter ncnn .param not produced" >&2; rm -rf "$SORTER_TMP"; exit 1; }
	test -f "$SORTER_TMP/sorter_attention.ncnn.bin"   || { echo "FAIL: Sorter ncnn .bin not produced" >&2; rm -rf "$SORTER_TMP"; exit 1; }
	rm -rf "$SORTER_TMP"
	echo "Sorter attention-encoder trainer smoke OK."
else
	echo "SKIP: godot_rl not installed in .venv-train (run scripts/setup_training.sh to enable the Sorter smoke)."
fi
```

Add `${SORTER_TMP:-}` to the EXIT-trap cleanup line. (Match the actual env-var names `train_sorter.sh` exposes from Step 6 of Task 3 — reconcile `OUTDIR`/`SAVE_MODEL_PATH` if the script uses different names.)

- [ ] **Step 3: Run the full suite green**

Run: `./test/run_tests.sh`
Expected: ends `All tests passed.`, output includes `Sorter attention-encoder trainer smoke OK.` and the new Python tests pass under the `Python helper tests` section.

- [ ] **Step 4: Commit**

```bash
git add test/run_tests.sh
git commit -m "test(#258): guarded Sorter attention-encoder trainer smoke"
```

---

### Task 5: Docs + close

**Files:**
- Modify: `CLAUDE.md`, `docs/godot-rl-gap-analysis-2026-06-02.md` (if it tracks #46)

- [ ] **Step 1: Add a `**Train (sorter, attention encoder):**` command bullet to `CLAUDE.md`**

Describe `./scripts/train_sorter.sh` (CleanRL PPO over the Sorter env with the attention encoder trunk; direct `attention_policy_layers` export, NOT ONNX), and update the `sorter` example line to note M2 shipped (encoder + trainer + spike). Record the pnnx-probe verdict from Task 2 Step 4.

- [ ] **Step 2: Commit + push + draft PR update**

The PR (#359 on this branch, or a new one) should note `Closes #258` and the #46 M2 checkbox. Push and ensure the draft PR body reflects the M2 work.

---

## Self-Review

**Spec coverage:** encoder (spec §Design.1) → Task 1; trainer (§Design.2) → Task 3; spike + pnnx probe + direct-export decision (§Design.3) → Task 2; CI smoke (§Testing) → Task 4; docs (§Design.4) → Task 5. ✅

**Placeholder scan:** the trainer body (Task 3 Step 4) is described structurally rather than as a full code block because it is a near-verbatim clone of `train_cleanrl.py` with three enumerated changes + a read step (Step 1) to ground them — acceptable per "follow existing patterns"; the load-bearing new code (encoder forward, export mapping, parity tests) is shown in full. No TBDs.

**Type consistency:** `AttentionEncoder(n_entities, feat, embed_dim, num_heads)` + `.emb/.q/.k/.v/.out/.n/.f/.d/.h` used identically across Tasks 1–3; `encoder_weights_dict`/`export_encoder_policy` signatures match between Task 2 definition and Task 3 use; weight keys (`emb_w/emb_b/q_w/...head0_w/head0_b`) match `attention_policy_layers`' documented contract; mask constant `-6e4`/`NEG_MASK` consistent; blobs `in0`/`out0` consistent.
