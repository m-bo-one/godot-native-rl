extends Node
# Real-render verification for CameraSensor (#36 part a). Headless Godot cannot render a
# viewport, so the live capture path (viewport.get_texture().get_image()) was never exercised
# by the CI unit tests. This scene runs NON-headless (under xvfb in CI-less environments) with a
# real rendering context: it fills a SubViewport with a known solid color, waits for the frame to
# be drawn, then asserts CameraSensor.get_image() / get_observation() return a real image of the
# expected size/format/content. Prints PASS/FAIL lines and quits with code 0 (all pass) or 1.
#
# Run:  xvfb-run -a godot --path . res://test/integration/camera_render_check.tscn
# (Deliberately NOT wired into headless run_tests.sh — see the spec: no display in CI.)

const CameraSensor = preload("res://addons/godot_native_rl/sensors/camera_sensor.gd")

const SIZE := 8
# The known fill color (opaque red-ish) the ColorRect paints into the viewport.
const FILL := Color(0.8, 0.2, 0.1, 1.0)

var _failures := 0

func _ok(cond: bool, msg: String) -> void:
	if cond:
		print("  PASS: ", msg)
	else:
		print("  FAIL: ", msg)
		_failures += 1

func _ready() -> void:
	# Give the renderer a few frames to actually draw the SubViewport before capturing.
	await get_tree().process_frame
	await get_tree().process_frame
	await RenderingServer.frame_post_draw
	await get_tree().process_frame

	var viewport: SubViewport = $SubViewport

	# --- RGB deploy path: get_image() -> a real, correctly-sized frame ---
	var rgb_sensor := CameraSensor.new()
	rgb_sensor.viewport = viewport
	rgb_sensor.grayscale = false
	add_child(rgb_sensor)
	var img: Image = rgb_sensor.get_image()
	_ok(img != null, "get_image() returns a non-null Image with a real render context")
	if img != null:
		_ok(not img.is_empty(), "captured image is non-empty")
		_ok(img.get_width() == SIZE and img.get_height() == SIZE,
			"captured image is %dx%d (matches viewport size)" % [SIZE, SIZE])
		# The fill color should dominate — sample the center pixel and check it is close to FILL.
		img.convert(Image.FORMAT_RGB8)
		var px: Color = img.get_pixel(SIZE / 2, SIZE / 2)
		_ok(absf(px.r - FILL.r) < 0.1 and absf(px.g - FILL.g) < 0.1 and absf(px.b - FILL.b) < 0.1,
			"center pixel %s matches the known fill color" % str(px))

	# --- Grayscale obs path: get_observation() hex byte count == W*H*1 ---
	var gray_sensor := CameraSensor.new()
	gray_sensor.viewport = viewport
	gray_sensor.grayscale = true
	add_child(gray_sensor)
	var hex: String = gray_sensor.get_observation()
	_ok(hex.length() == SIZE * SIZE * 1 * 2,
		"grayscale get_observation() hex length == W*H*1*2 (%d)" % (SIZE * SIZE * 2))

	# --- RGB obs path: hex byte count == W*H*3 ---
	var rgb_hex: String = rgb_sensor.get_observation()
	_ok(rgb_hex.length() == SIZE * SIZE * 3 * 2,
		"rgb get_observation() hex length == W*H*3*2 (%d)" % (SIZE * SIZE * 3 * 2))

	if _failures == 0:
		print("Results: all camera render checks passed")
	else:
		print("Results: %d camera render check(s) FAILED" % _failures)
	get_tree().quit(1 if _failures > 0 else 0)
