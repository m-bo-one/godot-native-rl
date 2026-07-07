# Attention Encoder M3 (SB3 parity) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Wrap the M2 `AttentionEncoder` as an SB3 `BaseFeaturesExtractor` + an SB3 PPO Sorter trainer that exports to ncnn via the M2 direct exporter (#259).

**Architecture:** `AttentionFeaturesExtractor` (features_dim=embed_dim) unwraps the Dict obs and delegates to `AttentionEncoder`; PPO with `net_arch=[]` makes `action_net` a single `Linear(embed_dim, n_act)`, so the deploy actor is `encoder → action_net` — the exact M2 export contract. Export pulls those two modules and reuses `spike_attention_ncnn.export_encoder_policy`.

**Tech Stack:** SB3 2.8 (`.venv-train`), torch 2.12, `ncnn` pip module, stdlib `unittest` (gated), godot_rl `StableBaselinesGodotEnv`.

## Global Constraints

- Python 4-space indent; tests torch/ncnn/SB3-gated via `@skipUnless`; heavy imports lazy.
- `net_arch=[]` is load-bearing (single linear action head).
- Dict obs: unwrap `observations["obs"]` (godot_rl SB3 wrapper) — support plain Box for tests.
- Export reuses M2: `spike_attention_ncnn.export_encoder_policy(enc, w, b, outdir, stem)`.
- No VecNormalize. Full `./test/run_tests.sh` green before push.

---

### Task 1: `AttentionFeaturesExtractor` + unit test

**Files:** Create `scripts/attention_features_extractor.py`; Test `test/python/test_attention_features_extractor.py`.

**Interfaces:** Produces `class AttentionFeaturesExtractor(BaseFeaturesExtractor)` with `__init__(observation_space, embed_dim=16, num_heads=2, n_entities=6, feat=4)` and `.encoder: AttentionEncoder`, `forward(observations) -> Tensor[B, embed_dim]`.

- [ ] **Step 1: Failing test** — `test/python/test_attention_features_extractor.py`:

```python
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

try:
    import torch
    from gymnasium import spaces
    import numpy as np
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
```

- [ ] **Step 2: Run → fail.** `.venv-train/bin/python -m unittest test.python.test_attention_features_extractor -v` → ImportError.

- [ ] **Step 3: Implement `scripts/attention_features_extractor.py`:**

```python
#!/usr/bin/env python3
"""SB3 BaseFeaturesExtractor wrapping the shared AttentionEncoder (#46/#259 M3).

With net_arch=[] the SB3 policy's action_net becomes a single Linear(embed_dim, n_act), so the
deploy actor is exactly this extractor's encoder followed by that head — the M2 export contract.
godot_rl's StableBaselinesGodotEnv gives a Dict obs {"obs": Box}, so forward unwraps "obs".
"""
from __future__ import annotations

from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from attention_encoder import AttentionEncoder


def _obs_width(observation_space) -> int:
    if isinstance(observation_space, spaces.Dict):
        return int(observation_space.spaces["obs"].shape[-1])
    return int(observation_space.shape[-1])


class AttentionFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, embed_dim: int = 16, num_heads: int = 2,
                 n_entities: int = 6, feat: int = 4):
        super().__init__(observation_space, features_dim=embed_dim)
        width = _obs_width(observation_space)
        assert n_entities * feat + n_entities == width, (
            "obs width %d != n*f+n (n=%d f=%d)" % (width, n_entities, feat))
        self._is_dict = isinstance(observation_space, spaces.Dict)
        self.encoder = AttentionEncoder(n_entities, feat, embed_dim, num_heads)

    def forward(self, observations):
        flat = observations["obs"] if self._is_dict else observations
        return self.encoder(flat)
```

- [ ] **Step 4: Run → pass.** Same command → OK (3 tests).

- [ ] **Step 5: Commit.** `git add scripts/attention_features_extractor.py test/python/test_attention_features_extractor.py && git commit -m "feat(#259): SB3 AttentionFeaturesExtractor wrapping the M2 encoder"`

---

### Task 2: SB3 export helper + parity gate

**Files:** Create `scripts/train_sorter_sb3.py` (helper first); Test `test/python/test_sb3_attention_export_parity.py`.

**Interfaces:** Produces `export_sb3_sorter_policy(model, outdir, stem="sorter_attention_sb3") -> (param, bin)` pulling `model.policy.features_extractor.encoder` + `model.policy.action_net`.

- [ ] **Step 1: Failing parity test** — `test/python/test_sb3_attention_export_parity.py`:

```python
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

try:
    import numpy as np
    import torch
    import ncnn  # noqa: F401
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    _HAS = True
except Exception:
    _HAS = False

if _HAS:
    from attention_features_extractor import AttentionFeaturesExtractor
    from train_sorter_sb3 import export_sb3_sorter_policy

    class _DummyDictEnv(gym.Env):
        def __init__(self):
            self.observation_space = spaces.Dict({"obs": spaces.Box(-10, 10, (30,), np.float32)})
            self.action_space = spaces.MultiDiscrete([5])
        def reset(self, *, seed=None, options=None):
            return {"obs": np.zeros(30, np.float32)}, {}
        def step(self, a):
            return {"obs": np.zeros(30, np.float32)}, 0.0, False, False, {}


@unittest.skipUnless(_HAS, "torch/ncnn/SB3 not available")
class TestSB3AttentionExportParity(unittest.TestCase):
    def test_sb3_export_matches_ncnn(self):
        import ncnn
        torch.manual_seed(3)
        model = PPO("MultiInputPolicy", _DummyDictEnv(), n_steps=8, batch_size=8, seed=3,
                    policy_kwargs=dict(features_extractor_class=AttentionFeaturesExtractor,
                                       features_extractor_kwargs=dict(embed_dim=16, num_heads=2),
                                       net_arch=[]))
        policy = model.policy.eval()
        with tempfile.TemporaryDirectory() as td:
            param, binp = export_sb3_sorter_policy(model, td, "spike_sb3")
            net = ncnn.Net(); net.load_param(str(param)); net.load_model(str(binp))
            for flags in ([1] * 6, [1, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]):
                ent = torch.randn(1, 24)
                flat = torch.cat([ent, torch.tensor([flags], dtype=torch.float32)], dim=1)
                with torch.no_grad():
                    feats = policy.features_extractor({"obs": flat})
                    ref = policy.action_net(feats)[0].tolist()
                ex = net.create_extractor(); ex.input("in0", ncnn.Mat(flat[0].numpy().copy()))
                _, out = ex.extract("out0")
                got = [out[i] for i in range(5)]
                for a, b in zip(got, ref):
                    self.assertAlmostEqual(a, b, delta=1e-2)
                self.assertEqual(max(range(5), key=lambda i: got[i]),
                                 max(range(5), key=lambda i: ref[i]))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run → fail** (train_sorter_sb3 missing).

- [ ] **Step 3: Implement `scripts/train_sorter_sb3.py`** (helper + trainer). Full module:

```python
#!/usr/bin/env python3
"""SB3 PPO Sorter trainer with the attention features extractor (#46/#259 M3).

Mirrors scripts/train_chase.py (SB3 PPO over StableBaselinesGodotEnv). The policy uses
AttentionFeaturesExtractor + net_arch=[] so the deploy actor is encoder -> single linear head;
export reuses the M2 direct exporter (spike_attention_ncnn.export_encoder_policy). No VecNormalize.
"""
from __future__ import annotations

import argparse
import pathlib
from typing import Sequence


def export_sb3_sorter_policy(model, outdir: str, stem: str = "sorter_attention_sb3"):
    """Export an SB3 attention policy (net_arch=[]) to ncnn via the M2 direct exporter."""
    from spike_attention_ncnn import export_encoder_policy

    policy = model.policy
    enc = policy.features_extractor.encoder
    head = policy.action_net
    return export_encoder_policy(enc, head.weight, head.bias, outdir, stem)


def parse_args(argv: Sequence[str] | None = None):
    p = argparse.ArgumentParser(allow_abbrev=False, description="SB3 PPO Sorter (attention encoder).")
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--speedup", type=int, default=8)
    p.add_argument("--action_repeat", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_steps", type=int, default=256)
    p.add_argument("--n_entities", type=int, default=6)
    p.add_argument("--feat", type=int, default=4)
    p.add_argument("--embed_dim", type=int, default=16)
    p.add_argument("--num_heads", type=int, default=2)
    p.add_argument("--save_model_path", type=str, default="models/sorter_attention_sb3.zip")
    p.add_argument("--outdir", type=str, default="models")
    p.add_argument("--stem", type=str, default="sorter_attention_sb3")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
    from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

    from attention_features_extractor import AttentionFeaturesExtractor

    args = parse_args(argv)
    env = StableBaselinesGodotEnv(env_path=None, show_window=False, seed=args.seed, n_parallel=1,
                                  speedup=args.speedup, action_repeat=args.action_repeat)
    env = VecMonitor(env)

    model = PPO(
        "MultiInputPolicy", env, verbose=1, n_steps=args.n_steps, batch_size=64,
        policy_kwargs=dict(
            features_extractor_class=AttentionFeaturesExtractor,
            features_extractor_kwargs=dict(embed_dim=args.embed_dim, num_heads=args.num_heads,
                                           n_entities=args.n_entities, feat=args.feat),
            net_arch=[]),
    )
    model.learn(args.timesteps)

    zip_path = pathlib.Path(args.save_model_path).with_suffix(".zip")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(zip_path)
    print("Saved SB3 policy to:", zip_path)

    pathlib.Path(args.outdir).mkdir(parents=True, exist_ok=True)
    param, binp = export_sb3_sorter_policy(model, args.outdir, args.stem)
    print("Exported ncnn to:", param, binp)
    env.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run → pass** (the parity gate). If it fails, debug the export extraction with `systematic-debugging` before continuing.

- [ ] **Step 5: Commit.** `git add scripts/train_sorter_sb3.py test/python/test_sb3_attention_export_parity.py && git commit -m "feat(#259): SB3 Sorter trainer + SB3->ncnn export-parity gate"`

---

### Task 3: `train_sorter_sb3.sh` + manual end-to-end proof

**Files:** Create `scripts/train_sorter_sb3.sh`.

- [ ] **Step 1:** Clone `scripts/train_chase.sh` structure; set `SCENE=res://examples/sorter/sorter_train_parallel.tscn`, `NUM_STEPS`/`OUTDIR`/`STEM` env passthroughs to `train_sorter_sb3.py`, stem `sorter_attention_sb3`. `bash -n` clean.

- [ ] **Step 2: Manual proof run (background).** Run a short SB3 Sorter run (small `--n_steps`, `--timesteps` ≥ one batch under 8 envs) into a temp dir; confirm exit 0 + `sorter_attention_sb3.ncnn.{param,bin}` produced, `models/` untouched. (Not wired as a permanent CI smoke — see spec §5.)

- [ ] **Step 3: Commit.** `git add scripts/train_sorter_sb3.sh && git commit -m "feat(#259): SB3 Sorter trainer orchestrator"`

---

### Task 4: Docs + full suite + close

- [ ] **Step 1:** `CLAUDE.md` — extend the sorter line ("M3 SB3 parity done (#259): AttentionFeaturesExtractor + train_sorter_sb3, same direct export; deterministic SB3→ncnn parity unit test gates it") and add a `Train (sorter, SB3 attention)` bullet.
- [ ] **Step 2: Full suite green.** `./test/run_tests.sh` → `All tests passed.` (the new Python parity/extractor tests run under `Python helper tests`).
- [ ] **Step 3: Commit + push; PR #359 body note M3 + `Closes #259`.**

---

## Self-Review

**Spec coverage:** extractor (§Design.1)→T1; trainer+export (§2)→T2; .sh+manual proof (§3,§5)→T3; docs (§6)→T4. ✅
**Placeholder scan:** all code shown in full. **Type consistency:** `AttentionFeaturesExtractor(observation_space, embed_dim, num_heads, n_entities, feat)` + `.encoder`/`.features_dim` consistent T1↔T2; `export_sb3_sorter_policy(model, outdir, stem)` reuses M2 `export_encoder_policy(enc, w, b, outdir, stem)`; blobs `in0`/`out0`; head width 5.
