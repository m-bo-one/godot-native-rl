# Real RecurrentPPO-trained recurrent deploy net — memory chase (#378)

**Status:** approved-by-implementer (autonomous; goal: "finish 378")
**Issue:** [#378](https://github.com/minigraphx/godot-native-rl/issues/378) — the deferred half of
backlog item 22 (deploy infra shipped there, gated by a synthetic LSTM only)
**Date:** 2026-07-13

## Export spike verdict (done first — the known blocker class)

A real `sb3-contrib` 2.8.0 RecurrentPPO `MlpLstmPolicy` actor (Flatten → `lstm_actor` →
`mlp_extractor.policy_net` → `action_net`), wrapped as `forward(obs, h, c) -> (logits, h', c')`
and exported via legacy ONNX (opset 13, `dynamo=False`) → pnnx → ncnn:

- **State blobs preserved**: `in0`=obs, `in1`=h, `in2`=c → `out0`=logits, `out1`=h', `out2`=c' —
  the exact synthetic-golden sidecar contract, so the deploy side needs ZERO changes.
- Parity over a 6-step state-carried sequence: max |diff| 6e-4, argmax exact.
- The wrapper's argmax matches `model.predict(deterministic=True)` step-for-step, so the
  extracted actor IS the policy's actor (episode_starts handled game-side by zero-init on reset,
  which `NcnnControllerCore.init_recurrent_state()` already does).

## The env: memory chase (blinking target)

A partially-observable twist on chase where the LSTM is load-bearing, not decorative:

- **Obs (6 floats):** `[agent_x, agent_y, dir_x, dir_y, dist, visible]`. On HIDDEN decisions the
  target-derived slots (`dir`, `dist`) are zeroed and `visible`=0; agent position is always
  observed. Pure helpers in `chase_memory_obs.gd` (`compute_obs`, `visible_for_step`).
- **Visibility schedule:** deterministic decision-count cycle — visible for `visible_steps`
  (default 2), hidden for `hidden_steps` (default 6). Episodes start visible; the counter resets
  with the episode. The target is static between catches (standard chase), so the optimal blind
  policy is "remember where it was and integrate own motion" — exactly what a feed-forward net
  cannot do (its blind obs are constant given agent pos, so it can't head anywhere useful).
- Game/reward reused from chase (`ChaseGame`, progress shaping + catch bonus − step penalty).
  Reward reads the true distance — training-only, not observed.

## Training + export

- `scripts/train_chase_memory.py` / `train_chase_memory.sh`: sb3-contrib RecurrentPPO
  (`MlpLstmPolicy`, lstm_hidden_size 64, `net_arch pi=[32] vf=[32]`) over
  `StableBaselinesGodotEnv` on the 8-world `chase_memory_train_parallel.tscn`.
  sb3-contrib is an **opt-in add-on** (`requirements-recurrent.txt`, jax/tune/hub pattern);
  the CI smoke is guarded on its importability.
- `scripts/export_recurrent_ppo.py`: checkpoint → actor wrapper → ONNX → `export_to_ncnn.py`
  (3-input shape) → writes the `<stem>.recurrent.json` sidecar → **verifies state-carried
  parity** by replaying a random sequence through ncnn(python) vs torch (the spike's check,
  productized — `--skip-verify` skips pnnx's single-step check, ours is the carried-sequence one).

## Deploy + the memory-is-real regression

- Deploy scene `chase_memory.tscn`: standard chase rendering, target sprite blinks with the
  actual visibility flag, legend + FitCamera2D; agent `control_mode=3` +
  `recurrent_stats_path` → the sidecar. Launcher entry.
- `test/integration/chase_memory_trained_scene.tscn`: 1800 frames; assert the trained LSTM
  catches ≥ threshold **and** the same net with `ablate_memory=true` (a test-only agent flag
  that calls `reset_recurrent_state()` after every decision — same weights, memory zeroed)
  catches meaningfully fewer. Ablation beats a feed-forward baseline as the proof: no second
  net, same graph, only the carried state differs — if memory were decorative the two scores
  would match.
- Unit: pure obs/visibility helpers; the existing synthetic golden already covers the
  controller's recurrent path.

## Honest limits

- Discrete-action MLP-LSTM only (matches the deploy contract's float-obs limit; the image path
  explicitly warns recurrent state is unused there).
- `enable_critic_lstm` stays default (separate critic LSTM) — only the actor is exported.
