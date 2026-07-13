# terminated/truncated split — implemented additively, ahead of upstream (#12)

**Status:** approved-by-implementer (autonomous; user direction: "check upstream… we could
integrate PR code, otherwise implement it first")
**Issue:** [#12](https://github.com/minigraphx/godot-native-rl/issues/12) (backlog item 9's last slice)
**Date:** 2026-07-13

## Upstream findings (2026-07-13)

- **PyPI:** `godot-rl` 0.8.2 (2025-02-25) is still the latest release.
- **upstream main** (`edbeeching/godot_rl_agents` @ `207b6f4`, 2026-06-13): recent activity is
  examples (SB3-contrib LSTM) + ONNX-export fixes for torch ≥ 2.9 — the protocol surface is
  unchanged; `godot_env.py::step_recv` still returns `done` in BOTH the terminated and truncated
  slots with the `TODO update API to term, trunc` comment intact.
- **No upstream issue or PR implements the split** (repo search: zero PRs matching; the only
  mentions are incidental). The 0.9 roadmap (#193) does not list it.

→ Nothing to integrate; we implement first. The upstream **compatibility pin** is now recorded in
README + CLAUDE.md: wire-compatible with godot-rl 0.8.2 / upstream main through `207b6f4`.

## Design — additive, zero-breakage

The blocker in #12 was "changing `done` semantics breaks `ep_rew_mean`". So `done` is NOT changed:

- **Godot side:** `NcnnControllerCore` gains `truncated` — set ONLY by the `reset_after` horizon
  (`step()`), never by agent-set task terminals; cleared by `reset()` and by `set_done_false()`
  (the sync's paired read-and-clear). Controllers expose `get_truncated()`; `NcnnSync` emits an
  additive `"truncated"` array in the step message (duck-typed: pre-#12 custom controllers report
  false). `done` stays = terminated OR truncated — stock godot_rl 0.8.2 reads only
  obs/reward/done/info and ignores unknown keys, so every existing trainer is byte-compatible.
- **Python side (our consumers):** `scripts/godot_env_truncation.py` —
  `TruncationAwareGodotEnv(GodotEnv)` overrides `step_recv` to return the real Gymnasium
  (terminated, truncated) split via the pure, unit-tested `split_done_truncated`. Fallback when
  the wire lacks the field (pre-#12 scene): all ends count as TERMINAL (truncated stays False) —
  strictly better than upstream's done-duplication and identical to our adapters' previous
  assumption. Wired into `GodotParallelEnv` (PettingZoo — `truncations` is now the real flag) and
  `GodotRLlibEnv` (RLlib single- and multi-policy).
- Stock-godot_rl paths (SB3/CleanRL/SF trainers) intentionally unchanged: they get the split for
  free the day upstream consumes the field.

## Also fixed here: the startup race, properly

The live smoke exposed the #286 startup-race class again (ray cold-start > Godot's 10s connect
window → HUMAN fallback → `accept()` timeout). The blind `sleep N` in
`train_rllib{,_pettingzoo}.sh` / `train_skrl.sh` is replaced by a **listen-poll** (lsof on
macOS/Linux, `ss` fallback; probing by *connecting* would consume the trainer's accept) that
launches Godot only once the env port is actually bound.

## Verification

- `test_controller_core.gd` — horizon sets truncated (+done); agent terminal does NOT; cleared on
  reset/set_done_false. `test_sync_messages.gd` — the step message carries the array.
- `run_protocol_test.py` — live wire: `truncated` present and `[false]` on a normal step.
- `test_godot_env_truncation.py` — splitter cases + `step_recv` over injected messages (incl. the
  missing-field fallback). `test_godot_pettingzoo_env.py` — per-agent truncations flow through.
- Live: the RLlib multi-policy smoke end-to-end over `TruncationAwareGodotEnv` (real horizon
  truncations during training), both actors exported, parity OK. Full `run_tests.sh` gate.

## When upstream lands it

`TruncationAwareGodotEnv` becomes a no-op (or is retired) and the pin in CLAUDE.md/README gets
bumped. Offering the `step_recv` patch upstream is a natural follow-up PR to godot_rl_agents.
