# Releasing

Releases ship prebuilt binaries via **GitHub Releases**; the **Asset Library** entry points at
the release-asset addon zip (the `Custom` download provider — binaries never enter git).
Background + rationale: `docs/superpowers/specs/2026-06-07-release-shipping-design.md`.

## Cut a release

1. Decide the version (SemVer; pre-1.0 ⇒ API/wire protocol may still break).
2. Bump `addons/godot_native_rl/plugin.cfg` `version="X.Y.Z"` if it isn't already, commit.
3. Tag and push:
   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```
4. `release.yml` runs: version guard → build all platforms (debug+release) → **validate each binary
   loads/resolves its symbols** (shared `validate-binaries.yml`: Windows `--headless` load, Android
   x86_64 emulator `dlopen`, Android arm64 link audit, iOS SDK test-link — publish is gated on it) →
   assemble `godot-native-rl-addon-vX.Y.Z.zip` + `godot-native-rl-examples-vX.Y.Z.zip` → drop-in smoke →
   create the GitHub Release. (A tag/`plugin.cfg` mismatch fails the guard job.)
5. Open the published release; copy the **addon zip sha256** from the notes (or `SHA256SUMS.txt`).

## Update the Asset Library entry

The Asset Library has no write API — this is a manual web edit, once per release.

1. Go to <https://godotengine.org/asset-library> → your asset → **Edit** (first time: **Submit**).
2. Set/confirm:
   - **Repository / Browse URL + Issues URL** → this (public) repo — lets moderators inspect source.
   - **Repository host** → `Custom`.
   - **Download URL** → the addon zip's release-asset link:
     `https://github.com/<owner>/<repo>/releases/download/vX.Y.Z/godot-native-rl-addon-vX.Y.Z.zip`
   - **Download hash** → the addon zip sha256 from step 5.
   - **Version** → `X.Y.Z`; **Godot version** → `4.5`.
3. Submit; wait for moderation approval (first submission only; edits are usually fast).

## Asset Library listing copy

Paste these into the AssetLib form. Keep the wording in sync with the README opener and
`plugin.cfg` `description=` when features change; the URLs are stable across releases.

**Brief description** (one line) — the `plugin.cfg` `description=`:

> GDExtension RL framework: native ncnn inference + godot_rl_agents-compatible training bridge, declarative reward authoring, and sensors.

**Description** (long) — plain text; the AssetLib renders limited formatting, so full URLs (they
auto-link) and `•` bullets are used instead of Markdown:

```
Reinforcement learning for Godot 4.5+ with NATIVE ncnn inference — statically linked C++, no C#/.NET, no external runtime. Train with the standard godot-rl (godot_rl_agents) Python stack; deploy native on web/WASM, console, mobile, desktop, and edge — targets an ONNX/.NET pipeline can't reach.

▶ TRY IT IN YOUR BROWSER (no install): https://minigraphx.github.io/godot-native-rl/
The demo launcher as a single-threaded WASM build. Trained agents run on native ncnn with no Python at runtime — pick "Evolution Lab" and the browser tab TRAINS a neural net in front of you.

▶ Watch the demo: https://youtu.be/Cud1gvbjg0I

WHAT YOU GET
• A GDExtension that runs trained policies natively via ncnn — one NcnnRunner node, no runtime dependency.
• A godot_rl_agents-compatible training bridge: train with the stock godot-rl Python package, convert to ncnn, deploy.
• Declarative reward authoring (Signal→Reward + RewardBuilder) and a sensors library (raycast, relative-position, camera/pixels, grid, navmesh, entity/attention).
• In-engine ES training (ESTrainer) — evolve a policy with NO Python, NO socket, NO backprop; runs on every target including web.
• Deploy extras a Python-server framework can't match: web/WASM, INT8 quantization, async + batched crowd inference, LOD policy switching, recurrent/LSTM state, continuous + multi-discrete actions.

EXAMPLES (all ship a trained net and run standalone — browse them in the launcher above or the examples/ folder)
Chase the target • Seek & Avoid • Rover 3D • Ball Chase (SAC) • Fly By (PPO continuous) • Hide & Seek (multi-policy self-play) • Coop Collect (MA-POCA) • Quadruped Walk / Hurdles / Jump • Hexapod • 3DBall & GridWorld (Unity-parity) • Visual Chase (CNN from pixels) • Memory Chase (LSTM) • plus the live Evolution Lab.

INSTALL
Enable the plugin, then open the demo launcher (F5). Full guide: https://github.com/minigraphx/godot-native-rl/blob/main/docs/guide/getting-started.md

LINKS
• Source & README: https://github.com/minigraphx/godot-native-rl
• Guides (getting started, running examples, training, deploying, sensors): https://github.com/minigraphx/godot-native-rl/tree/main/docs/guide
• ncnn vs ONNX Runtime — honest decision guide: https://github.com/minigraphx/godot-native-rl/blob/main/docs/ncnn_vs_onnx.md
• Issues / questions: https://github.com/minigraphx/godot-native-rl/issues

Pre-1.0: the API and wire protocol may still change between minor versions. MIT licensed.
```

## Prerequisites (one-time)

- The repo must be **public** (Settings → General → Danger Zone → Change visibility) so the
  AssetLib browse/issues URLs resolve and moderators can read the source.
