#!/usr/bin/env bash
# Render a recorded chase replay to a video clip via Godot 4's MovieWriter (#40).
#
#   1. Record ONE episode from the deployed chase ncnn net into a gnrl_replay_v1 JSON
#      (headless; skipped if REPLAY= points at an existing file).
#   2. Play that replay in chase_video.tscn under `godot --write-movie <out>` — the scene
#      QUITS when the replay ends (ReplayPlayer.quit_on_finish), so the clip is bounded.
#   3. Verify the output file exists and is non-empty.
#
# Rendering needs a display; headless Godot cannot render a viewport, so this wraps Godot in
# xvfb-run (a virtual framebuffer). MovieWriter picks the codec from the OUT extension: .avi =>
# built-in MJPEG, .png => a PNG frame sequence directory.
#
# Usage:  scripts/record_replay_video.sh [OUT] [FRAMES]
# Env:    OUT=chase_replay.avi FRAMES=900 FPS=60 REPLAY=<existing replay.json> GODOT=<binary>
set -euo pipefail
cd "$(dirname "$0")/.."

GODOT="${GODOT:-godot}"
OUT="${1:-${OUT:-chase_replay.avi}}"
FRAMES="${2:-${FRAMES:-900}}"
FPS="${FPS:-60}"
REPLAY="${REPLAY:-}"

if ! command -v xvfb-run >/dev/null 2>&1; then
	echo "error: xvfb-run not found — install xvfb (rendering needs a display; headless can't)." >&2
	exit 1
fi

# Intermediate replay path: an ABSOLUTE filesystem path (Godot's FileAccess accepts these), so we
# never have to guess the project's user:// data dir (it is derived from the project name).
REPLAY_ABS="$(mktemp -t chase_replay.XXXXXX.json)"

# --- Step 1: record a replay (unless one was supplied) ---
if [[ -z "$REPLAY" ]]; then
	echo "== Recording a chase replay from the trained net ($FRAMES frames) =="
	"$GODOT" --headless --path . res://examples/chase_the_target/record_chase_replay.tscn \
		-- "out=$REPLAY_ABS" "frames=$FRAMES"
	if [[ ! -s "$REPLAY_ABS" ]]; then
		echo "error: replay was not recorded at $REPLAY_ABS" >&2
		exit 1
	fi
else
	cp "$REPLAY" "$REPLAY_ABS"
fi

# --- Step 2: render the replay to video under a virtual display ---
echo "== Rendering $REPLAY_ABS to $OUT (MovieWriter, ${FPS} fps) =="
xvfb-run -a "$GODOT" --path . --write-movie "$OUT" --fixed-fps "$FPS" \
	res://examples/chase_the_target/chase_video.tscn -- "replay_path=$REPLAY_ABS"

# --- Step 3: verify ---
if [[ -s "$OUT" ]]; then
	echo "== OK: wrote $OUT ($(du -h "$OUT" | cut -f1)) =="
elif [[ -d "$OUT" ]]; then
	# PNG-sequence output is a directory of frames.
	echo "== OK: wrote $OUT ($(find "$OUT" -type f | wc -l) frames) =="
else
	echo "error: no video output produced at $OUT" >&2
	exit 1
fi
