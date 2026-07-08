# Record replay to video via MovieWriter (#40)

**Status:** approved-by-implementer (autonomous batch)
**Issue:** [#40](https://github.com/minigraphx/godot-native-rl/issues/40) (Backlog item 35)
**Date:** 2026-07-07
**Builds on:** #39 (episode replay — `ReplayRecorder` / `ReplayPlayer`)

## Problem

We can record and replay episodes (#39), but there is no way to turn a replay into a shareable
**video clip**. Godot 4 ships a `MovieWriter` (enabled by the `--write-movie <path>` cmdline:
built-in MJPEG `.avi` or a PNG frame sequence), which captures every rendered frame at a fixed fps
and finalizes the file when the app quits. The missing pieces are (a) a replay scene that **quits
when the replay ends** so the clip is bounded, and (b) a one-command pipeline that records a replay
from the deployed net and renders it — under a virtual display, since headless Godot can't render.

## Design

### `ReplayPlayer` — quit-on-finish + cmdline path (addon)

- `quit_on_finish: bool` (default false). When the replay ends (and `loop` is false), schedule
  `get_tree().quit()` one short timer later — the last action gets a frame to render into the movie
  before the writer finalizes. Ignored while looping (a looping replay never ends).
- `parse_replay_path_arg(args)` — a pure static helper that pulls `replay_path=<value>` out of the
  user cmdline args (the part after `--`). `_ready()` applies it, so **one** video scene can render
  **any** recorded episode without editing the `.tscn`. Unit-tested (no tree needed).

### `chase_video.tscn` — the render scene

Mirrors `chase_replay.tscn` (ChaseGame + trained-agent-less ChaseAgent + `ReplayPlayer` +
`FitCamera2D`) but with `quit_on_finish = true`. `chase_game._draw()` already renders the arena +
agent + target circles and `FitCamera2D` frames the arena, so the clip is watchable with no new art.

### `record_chase_replay.tscn` / `.gd` — headless replay source

A trained `ChaseAgent` (`control_mode = 3`) driven by a sibling `Sync` (NCNN_INFERENCE mode, no
socket). A driver node creates a `ReplayRecorder` in code, `attach_agent()`s the deployed agent
(inference path — taps `inference_step`, #194), runs `frames=` physics ticks, then
`flush_inference_episodes()` and copies the newest ring file to the requested `out=` path. Cmdline:
`out=user://chase_replay.json frames=900`.

### `scripts/record_replay_video.sh` — the pipeline

1. Record a replay from the trained net (headless) — skipped if `REPLAY=` supplies an existing file.
2. `xvfb-run godot --write-movie <OUT> --fixed-fps <FPS> chase_video.tscn -- replay_path=<json>`.
3. Verify the output is a non-empty file (`.avi`) or a non-empty frame directory (`.png`).

`xvfb-run` gives Godot a virtual framebuffer (rendering needs a display). Codec follows the `OUT`
extension (Godot's built-in choice). Not wired into headless CI (no display); the value is the
one-command clip + the unit-tested player logic.

## Testing

- `test_replay_player.gd` — extended with `parse_replay_path_arg` assertions (pure, in CI).
- `scripts/record_replay_video.sh` — produces a real `.avi` (verified manually under xvfb; documented).
- Full `run_tests.sh` green (no regression from the addon change).

## Files

| File | Change |
|---|---|
| `addons/godot_native_rl/training/replay_player.gd` | `quit_on_finish` + cmdline `replay_path=` |
| `examples/chase_the_target/chase_video.tscn` | **new** render scene (quit_on_finish) |
| `examples/chase_the_target/record_chase_replay.{gd,tscn}` | **new** headless replay source |
| `scripts/record_replay_video.sh` | **new** record → render → verify pipeline |
| `test/unit/test_replay_player.gd` | parse-helper assertions |
| `CLAUDE.md`, `docs/BACKLOG.md` | document + check off item 35 |

Closes #40.
