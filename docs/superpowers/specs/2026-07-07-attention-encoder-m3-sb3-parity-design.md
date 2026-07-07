# Attention Encoder M3 — SB3 FeaturesExtractor parity (#259)

**Status:** approved-by-implementer (autonomous continuation; design grounded by an empirical SB3 probe)
**Issue:** [#259](https://github.com/minigraphx/godot-native-rl/issues/259) (parent [#46](https://github.com/minigraphx/godot-native-rl/issues/46))
**Parent spec:** `docs/superpowers/specs/2026-06-15-attention-entity-encoder-design.md` §2
**Builds on:** M2 (#258) — the torch `AttentionEncoder` + the direct `attention_policy_layers` exporter, both proven and CI-pinned.
**Date:** 2026-07-07

## Goal & non-goals

- **Goal:** wrap the shared `scripts/attention_encoder.py` as an SB3 `BaseFeaturesExtractor` and add
  an SB3 PPO Sorter trainer, so the "both backends" decision holds (CleanRL proven in M2, SB3 here)
  and an SB3-trained attention policy exports to native ncnn through the same direct exporter.
- **Non-goal:** numeric equality between the CleanRL-trained and SB3-trained nets (different runs);
  "parity" here means the SB3 path reaches ncnn correctly via the same encoder + exporter.
- **Non-goal:** a committed trained net + behavioral regression (that's M4/#260).

## Grounding — empirical SB3 probe (verified, not assumed)

A dummy Dict-obs env (`{"obs": Box(30)}`, `MultiDiscrete([5])`) with a custom
`BaseFeaturesExtractor(features_dim=16)` + `net_arch=[]` yields:

```
action_net: Linear(in=16, out=5)      value_net: Linear(in=16, out=1)
features_extractor: <custom>          mlp_extractor.latent_dim_pi: 16 (identity)
features_extractor({"obs":(B,30)}) -> (B,16);  action_net(features) -> (B,5) logits
```

So with `net_arch=[]` the deploy actor is exactly **features_extractor.encoder → action_net**, i.e.
the M2 export contract (`AttentionEncoder` + one linear head). godot_rl's `StableBaselinesGodotEnv`
gives a **Dict** obs (hence `MultiInputPolicy`), so the extractor must read `observations["obs"]`.

## Design

### 1. `scripts/attention_features_extractor.py`

`AttentionFeaturesExtractor(BaseFeaturesExtractor)`:
- `__init__(observation_space, embed_dim=16, num_heads=2, n_entities=6, feat=4)` — `features_dim =
  embed_dim`; holds `self.encoder = AttentionEncoder(n_entities, feat, embed_dim, num_heads)`.
  Resolve the flat obs width from `observation_space` (Dict → its `"obs"` Box `.shape[-1]`; plain Box
  → `.shape[-1]`) and `assert n_entities*feat + n_entities == obs_width`.
- `forward(observations)` — extract the flat tensor (`observations["obs"]` for a Dict, else
  `observations`) and return `self.encoder(flat)` `(B, embed_dim)`.

Pure-ish: the only new logic is the Dict/Box unwrap + the delegation; the math lives in the M2
encoder.

### 2. `scripts/train_sorter_sb3.py`

Mirror `scripts/train_chase.py` (SB3 PPO over `StableBaselinesGodotEnv` + `VecMonitor`). Changes:
- `PPO("MultiInputPolicy", env, policy_kwargs=dict(features_extractor_class=AttentionFeaturesExtractor,
  features_extractor_kwargs=dict(embed_dim, num_heads, n_entities, feat), net_arch=[]))`.
- On completion, export via a helper `export_sb3_sorter_policy(model, outdir, stem)` that pulls
  `model.policy.features_extractor.encoder` + `model.policy.action_net` and calls the M2
  `spike_attention_ncnn.export_encoder_policy(enc, action_net.weight, action_net.bias, outdir, stem)`.
  Also `torch.save` the SB3 zip.
- CLI: `--timesteps/--speedup/--action_repeat/--n_steps/--n_entities/--feat/--embed_dim/--num_heads/
  --save_model_path/--outdir/--stem/--seed`. No VecNormalize (keeps deploy parity clean — the obs is
  raw, matching the encoder's training input).

### 3. `scripts/train_sorter_sb3.sh`

Clone `scripts/train_chase.sh`; `SCENE=res://examples/sorter/sorter_train_parallel.tscn`, retarget
stems to `models/sorter_attention_sb3*`, `NUM_STEPS`/`OUTDIR`/`STEM` passthroughs.

### 4. Tests (TDD)

- **`test/python/test_attention_features_extractor.py`** (torch-gated): a Dict `{"obs": Box(30)}`
  space → `AttentionFeaturesExtractor(embed_dim=16, num_heads=2)`; assert `.features_dim==16`,
  `forward({"obs": zeros(4,30)}).shape == (4,16)`, `.encoder` is an `AttentionEncoder` with
  `n*f+n==30`, and a mismatched obs width raises `AssertionError`.
- **`test/python/test_sb3_attention_export_parity.py`** (torch+ncnn+SB3-gated): build a small
  `PPO("MultiInputPolicy", DummyDictEnv, policy_kwargs=...net_arch=[])` (no Godot), export via
  `export_sb3_sorter_policy`, and assert ncnn `out0` == `action_net(features_extractor(obs))` at
  atol 1e-2, argmax exact, over obs with padded rows. **This is the M3 export gate.**

### 5. CI + manual verification (decision)

The M2 CleanRL sorter smoke already exercises env→encoder→PPO→direct-export end-to-end in CI. The
SB3 delta is purely the **export extraction** (`features_extractor.encoder` + `action_net`), which
the deterministic parity unit test (#4b) covers with no Godot and no flakiness. Adding a second
~2-min Godot training smoke for SB3 would double CI cost for marginal coverage — and the repo does
not smoke the SB3 `train_chase.py` either. **Decision:** the parity unit test is the CI gate; the
SB3 trainer is proven **once manually** (a short real run → export) during implementation, not wired
as a permanent Godot smoke.

### 6. Docs / close

`CLAUDE.md` sorter line + a `Train (sorter, SB3 attention)` note; `Closes #259` + #46 acceptance.

## Files touched

| File | Change |
|---|---|
| `scripts/attention_features_extractor.py` | **new** SB3 BaseFeaturesExtractor wrapping AttentionEncoder |
| `scripts/train_sorter_sb3.py` | **new** SB3 PPO Sorter trainer + `export_sb3_sorter_policy` |
| `scripts/train_sorter_sb3.sh` | **new** orchestrator |
| `test/python/test_attention_features_extractor.py` | **new** extractor shape/Dict/validation test |
| `test/python/test_sb3_attention_export_parity.py` | **new** SB3→ncnn export-parity gate |
| `CLAUDE.md` | sorter line + SB3 train note |

Closes #259 (parent #46 M3).

## Risks / gotchas

- **Dict obs:** godot_rl SB3 wrapper gives `{"obs": Box}` → `MultiInputPolicy`; the extractor MUST
  unwrap the `"obs"` key. A plain-Box path is supported for the dummy-env tests.
- **`net_arch=[]` is load-bearing:** any hidden layer after the features breaks the single-linear-head
  export contract. Assert/keep `net_arch=[]`.
- **No VecNormalize:** raw obs only, so the deployed encoder sees the same distribution it trained on
  (no stats sidecar needed).
- The action head must emit exactly `sum(nvec)=5` raw logits (no trailing activation) — SB3's
  `action_net` is a bare `Linear`, so this holds; the exporter's `head_dims=[5]` matches.
