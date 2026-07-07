# Attention Encoder M2 — torch side (#258)

**Status:** approved design
**Issue:** [#258](https://github.com/minigraphx/godot-native-rl/issues/258) (parent [#46](https://github.com/minigraphx/godot-native-rl/issues/46))
**Parent spec:** `docs/superpowers/specs/2026-06-15-attention-entity-encoder-design.md` (§4 encoder math, §5 the ncnn spike)
**Date:** 2026-07-07

## Context — what M2 still needs

The **entire ncnn deploy path is already proven and CI-pinned** (shipped in #307): a hand-authored
`MultiHeadAttention` + additive-mask graph round-trips through `NcnnRunner.run_inference_multi`
(and a single-input variant), matches a pure-Python replica of ncnn's forward loops to 1e-4, and
shows cross-run mask invariance — byte-pinned to committed fixtures at **0.0 error on fp32**
(`test/unit/test_attention_golden_inference.gd`, `models/synthetic_attention*`). The
`state_dict → ncnn` direct exporter (`scripts/export_statedict_to_ncnn.py` +
`attention_policy_layers`) is **already production**. The `examples/sorter/` env is **complete**
(variable 2–6 tiles, `EntitySensor2D` obs, scripted-expert smoke).

What remains is purely the **torch side** that plugs into that proven contract. This session has
torch (`.venv-train` torch 2.12) and pnnx (`.venv/bin/pnnx`), unlike the torch-free sessions that
filed #307.

## Goal & non-goals

- **Goal:** ship `scripts/attention_encoder.py` (a torch encoder numerically identical to the
  ncnn-verified replica), a CleanRL Sorter trainer that solves variable-count episodes, and an
  export-parity spike that gates M3/M4 — closing #46 acceptance checkbox 2 + the spike de-risk.
- **Non-goal:** re-verify the ncnn runtime (done in #307). Non-goal: the M4 trained behavioral
  regression + committed net (that's #260) — M2 proves the *training + export contract*.
- **Non-goal:** a global/self observation block in the exporter (Sorter is entity-only,
  `global_dim=0`); extending the exporter for a global concat is out of scope.

## Design

### 1. `scripts/attention_encoder.py` — the torch encoder

`AttentionEncoder(n_entities, feat, embed_dim, num_heads, global_dim=0)`, a torch `nn.Module`
that is **numerically identical to the pure-Python `encoder_forward()` replica** in
`scripts/make_synthetic_attention.py`, so it exports byte-for-byte through the existing direct
exporter. Forward, for flat obs `(B, N*F + N)` with Sorter `N=6, F=4`:

| Step | Op | Shape |
|---|---|---|
| 0 | split: `entities = flat[:, :N*F].reshape(B,N,F)`, `flags = flat[:, N*F:]` | `(B,N,F)`, `(B,N)` |
| 1 | `ReLU(entities @ emb.Wᵀ + emb.b)` | `(B,N,D)` |
| 2 | Q,K,V per-entity Linear | each `(B,N,D)` |
| 3 | head reshape (contiguous embed split, `d=D/H`) | `(B,H,N,d)` |
| 4 | `scores = Q·Kᵀ / √d + addmask` | `(B,H,N,N)` |
| 5 | `ctx = softmax(scores) · V`, concat heads | `(B,N,D)` |
| 6 | `O = ctx @ out.Wᵀ + out.b` | `(B,N,D)` |
| 7 | masked mean-pool `zₖ = Σⱼ pⱼ·Oⱼₖ / max(Σⱼ pⱼ, 1)` | `(B,D)` |

**Load-bearing invariants (must match the exporter/replica or train/deploy silently diverge):**
- **Mask constant `-6e4`** (`sd.FP16_SAFE_NEG_MASK`), NOT `-inf`/`-1e9` — additive mask
  `addmask_j = (1 − p_j)·(−6e4)`, broadcast across query rows, shared across heads. `-1e9`
  overflows fp16 → -inf → NaN on a fully-masked row on ARM deploys.
- **Per-head scale `1/√d`** (d = D/H), NOT `1/√D`. ncnn omits param 6 → default `1/√(D/H)`.
- **No residual, no LayerNorm; ReLU only after the embedding.** Single attention block.
- **Weight layout = torch `nn.Linear` `[out,in]` row-major** copied verbatim (no transpose):
  `emb (D,F)`; `q/k/v/out (D,D)`.
- `embed_dim % num_heads == 0` (ncnn silently truncates otherwise).
- Fully-masked (empty) group → exact zeros via the `max(Σp,1)` guard + zero flags.

Submodules named `emb`, `q`, `k`, `v`, `out` (all `nn.Linear`) so `.state_dict()` maps directly to
the exporter's weight keys.

### 2. `scripts/train_sorter_cleanrl.py` + `scripts/train_sorter.sh`

Mirror `scripts/train_cleanrl.py` wholesale (`PPOConfig`, `compute_gae`, `layer_init`,
`_split_categoricals`, the rollout/PPO loop, `CleanRLGodotEnv`). **Only two changes:**

- **Agent trunk** = the attention encoder:
  `encoder: AttentionEncoder(6,4,D,H)` → `actor: Linear(D, 5)` + `critic: Linear(D, 1)`. The
  encoder does the flat→`(N,F)`+flags reshape internally, so the CleanRL loop keeps passing flat
  `(B,30)` obs unchanged.
- **Export** via the direct exporter, not ONNX: map the trained `state_dict` into a weights dict
  and call `sd.attention_policy_layers(N=6, F=4, D, H, weights, head_dims=[5])` → write
  `.ncnn.{param,bin}`. Deploy blobs `in0`/`out0` → plain `run_inference` + `ActionDecode` argmax.
  A pure `sorter_export_weights(agent) -> dict` helper (state_dict → weight keys) is unit-testable
  without a live model.

`train_sorter.sh` clones `train_cleanrl.sh`, retargeting
`SCENE=res://examples/sorter/sorter_train_parallel.tscn` (8 tiled worlds) and the save/export
stems.

### 3. `scripts/spike_attention_ncnn.py` — the export-parity gate + pnnx probe

The M2 go/no-go: random-init `AttentionEncoder` → direct export via `attention_policy_layers` →
load with `ncnn.Net` → assert torch-vs-ncnn parity (atol 1e-2, argmax exact) over random obs
including padded rows. **Plus** a logged pnnx-emission probe (trace the module, run pnnx, log
whether it emits a `MultiHeadAttention` layer, a decomposed-but-valid graph, or breaks) — recorded
for the design note, not a gate.

**Decision (approved): direct hand-written export is PRIMARY; pnnx-emission is a logged
experiment.** The direct exporter is already production and byte-pinned (0.0 error), the encoder's
math *is* the exporter's semantics, and it round-trips deterministically with no
toolchain-deprecation exposure. pnnx is recorded per spec §5 but training/deploy do not depend on
it.

### 4. Docs / backlog

`CLAUDE.md` train-command line for the Sorter attention trainer; gap analysis; #258 checkbox +
#46 acceptance checkbox 2.

## Sorter interface (confirmed)

- **Obs width 30** = `[24 entity block][6 presence flags]`; `N=6` entities × `F=4` features
  `[rel_x, rel_y, number/total, visited]`; padded slots zero + flag 0; entities are nearest-6
  ascending by distance.
- **Action:** single discrete head `{"move": {size:5}}` → 5 logits, per-segment argmax on deploy.
- **Train scene:** `res://examples/sorter/sorter_train_parallel.tscn` (8 tiled worlds via
  `ParallelArena2D`; per-world isolation via `EntitySensor2D.scope_root`, so one export works under
  tiling).

## Testing (TDD)

- **Encoder parity anchor** (`test/python/test_attention_encoder.py`, torch-gated): with the same
  weights the fixture generator draws, `AttentionEncoder(obs) ≈ encoder_forward(obs)` to ~1e-5 on
  the `all_present`/`two_present`/`two_present_junk`/`none_present` cases. Plus mask-invariance,
  count-invariance, permutation-invariance, and fully-masked→zeros (no NaN with `-6e4`).
- **Export-parity gate** (the spike + a torch-gated integration test): random-init encoder →
  direct export → torch-vs-ncnn parity atol 1e-2, argmax exact, over padded inputs.
- **`sorter_export_weights` unit test:** state_dict → weight-key dict mapping (no live model).
- **Guarded CI smoke:** a short few-update Sorter training run in the `run_tests.sh` godot_rl-gated
  cluster (mirrors the CleanRL+RND smoke), exiting loud if `TIMESTEPS < NUM_STEPS·num_envs`
  (the #119 precedent), asserting the `.ncnn.{param,bin}` are produced.
- Full `./test/run_tests.sh` green.

## Risks / gotchas

- Mask constant must be `-6e4` (fp16-safe), not `-inf`/`-1e9`.
- Per-head scale `1/√d`, not `1/√D` — a `1/√D` bug passes shape checks and mistrains.
- Weight layout `[out,in]` verbatim — do not transpose.
- Obs reshape must be row-major entity-major (`flat[:, :24].reshape(B,6,4)`, `flags=flat[:,24:]`).
- `embed_dim % num_heads == 0` (pick e.g. `D=16,H=2` or `D=32,H=4`).
- Sparse-ish reward → PPO with entropy; parallel scene (8 worlds) is the throughput lever;
  `num_steps`/`ent_coef` may need tuning to solve variable-count episodes.
- `max_tiles`(6) ↔ `max_entities`(6) coupling defines the fixed 30-wide contract.

## Files touched

| File | Change |
|---|---|
| `scripts/attention_encoder.py` | **new** torch encoder module |
| `scripts/train_sorter_cleanrl.py` | **new** CleanRL PPO w/ attention trunk + direct export |
| `scripts/train_sorter.sh` | **new** orchestrator (clone of `train_cleanrl.sh`) |
| `scripts/spike_attention_ncnn.py` | **new** export-parity gate + pnnx probe |
| `test/python/test_attention_encoder.py` | **new** parity/invariance tests |
| `test/python/test_sorter_export_weights.py` | **new** state_dict→weights mapper test |
| `test/run_tests.sh` | guarded Sorter training smoke |
| `CLAUDE.md`, gap analysis | train-command + status notes |

Closes #258 (parent #46 acceptance checkbox 2 + spike de-risk).
