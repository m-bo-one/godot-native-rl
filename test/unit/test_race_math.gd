extends SceneTree
# Unit tests for the #60 M4 race ranking helpers (pure, no scene).

const Harness = preload("res://test/harness.gd")
const R = preload("res://examples/quadruped_walk/race_math.gd")

func _initialize() -> void:
	var h = Harness.new()

	h.assert_eq(R.standings([3.0, 21.0, 10.0]), [1, 2, 0], "ranked by distance desc")
	h.assert_eq(R.standings([5.0, 5.0, 9.0]), [2, 0, 1], "ties break by lower lane id")
	h.assert_eq(R.standings([]), [], "empty -> empty")

	h.assert_eq(R.places([3.0, 21.0, 10.0]), [3, 1, 2], "1-based places (lane1 leads)")
	h.assert_eq(R.places([9.0, 9.0]), [1, 2], "tie places stable")

	h.assert_true(R.finished(40.0, 40.0), "exactly at finish counts")
	h.assert_true(R.finished(41.0, 40.0), "past finish counts")
	h.assert_true(not R.finished(39.9, 40.0), "before finish does not")

	h.assert_eq(R.format_row(1, "gen-6M", 20.9), "1. gen-6M  20.9 m", "row format")

	# Countdown to the next generation switch (frames remaining -> whole seconds, rounded up).
	h.assert_eq(R.countdown_seconds(4000, 60), 67, "4000 frames @60fps rounds up to 67s")
	h.assert_eq(R.countdown_seconds(60, 60), 1, "exactly one second")
	h.assert_eq(R.countdown_seconds(1, 60), 1, "any remaining frame is >=1s")
	h.assert_eq(R.countdown_seconds(0, 60), 0, "no frames left -> 0s")
	h.assert_eq(R.countdown_seconds(-5, 60), 0, "past the deadline clamps to 0s")
	h.assert_eq(R.countdown_seconds(120, 0), 0, "non-positive tick rate -> 0s (no divide-by-zero)")

	# "Now playing" banner for the active generation.
	h.assert_eq(R.format_now(2, 3, "gen-2.5M", 12.3, 41),
		"▶ gen-2.5M  (2/3)  reach 12.3 m  ·  next in 41s", "now-playing banner")

	h.finish(self)
