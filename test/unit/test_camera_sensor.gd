extends SceneTree

const Harness = preload("res://test/harness.gd")
const CameraSensor = preload("res://addons/godot_native_rl/sensors/camera_sensor.gd")

func _make_image(w: int, h: int, fmt: int, fill: Color) -> Image:
	var img := Image.create(w, h, false, fmt)
	img.fill(fill)
	return img

func _initialize() -> void:
	var h := Harness.new()

	# --- RGB path: 2x2 image injected via the test seam ---
	var s = CameraSensor.new()
	var vp := SubViewport.new()
	vp.size = Vector2i(2, 2)
	s.viewport = vp
	# A real RGB8 image with known bytes (all red): each pixel = (255, 0, 0).
	var rgb := _make_image(2, 2, Image.FORMAT_RGB8, Color(1, 0, 0))
	s.set_image_for_test(rgb)

	var obs: String = s.get_observation()
	# 4 pixels * 3 channels = 12 bytes, each pixel "ff0000".
	h.assert_eq(obs, "ff0000ff0000ff0000ff0000", "RGB obs hex == red 2x2")
	h.assert_eq(s.get_obs_shape(), [2, 2, 3], "RGB obs_shape [H,W,3]")
	h.assert_eq(s.get_obs_space_entry(), {"space": "box", "size": [2, 2, 3]}, "RGB obs_space entry")
	h.assert_eq(s.get_observation_key(), "camera_2d", "default observation_key")

	# --- Grayscale path: white image -> L8, 1 channel, deterministic luminance ---
	# Pure white converts to L8 255 regardless of the luminance weights, so the exact
	# hex is stable: 4 pixels * 1 channel * 0xFF = "ffffffff".
	s.grayscale = true
	s.set_image_for_test(_make_image(2, 2, Image.FORMAT_RGB8, Color(1, 1, 1)))
	var gray_obs: String = s.get_observation()
	h.assert_eq(s.get_obs_shape(), [2, 2, 1], "grayscale obs_shape [H,W,1]")
	h.assert_eq(gray_obs, "ffffffff", "grayscale white obs == L8 255 x4")

	s.free()
	vp.free()

	# --- Missing viewport -> stable empty obs, no crash ---
	var s2 = CameraSensor.new()
	s2.set_image_for_test(_make_image(2, 2, Image.FORMAT_RGB8, Color(1, 0, 0)))
	h.assert_eq(s2.get_observation(), "", "missing viewport -> empty obs")
	h.assert_eq(s2.get_obs_shape(), [0, 0, 3], "missing viewport -> zero shape")
	s2.free()

	# --- Key without "2d" is rejected (validation returns false) ---
	var s3 = CameraSensor.new()
	h.assert_true(s3.is_key_valid("camera_2d"), "key with 2d is valid")
	h.assert_true(not s3.is_key_valid("camera"), "key without 2d is invalid")
	s3.free()

	# --- get_image(): returns the raw captured Image (deploy path, no hex) ---
	var s4 = CameraSensor.new()
	var vp4 := SubViewport.new()
	vp4.size = Vector2i(2, 2)
	s4.viewport = vp4
	var src := _make_image(2, 2, Image.FORMAT_RGB8, Color(0, 1, 0))
	s4.set_image_for_test(src)
	var got: Image = s4.get_image()
	h.assert_true(got != null, "get_image returns the injected image")
	h.assert_eq(got.get_width(), 2, "get_image width")
	h.assert_eq(got.get_height(), 2, "get_image height")
	h.assert_eq(got.get_format(), Image.FORMAT_RGB8, "RGB get_image -> FORMAT_RGB8")
	s4.free()
	vp4.free()

	# --- get_image() grayscale deploy path (#36): a grayscale sensor must hand back FORMAT_L8, so
	# NcnnControllerCore auto-detects 1-channel inference. A raw viewport texture is RGBA8, which the
	# controller would mis-route to the 3-channel path — get_image() must coerce like get_observation().
	var s6 = CameraSensor.new()
	var vp6 := SubViewport.new()
	vp6.size = Vector2i(2, 2)
	s6.viewport = vp6
	s6.grayscale = true
	# Inject an RGBA8 frame (what viewport.get_texture().get_image() actually returns at deploy).
	s6.set_image_for_test(_make_image(2, 2, Image.FORMAT_RGBA8, Color(1, 1, 1)))
	var gray_img: Image = s6.get_image()
	h.assert_true(gray_img != null, "grayscale get_image returns an image")
	h.assert_eq(gray_img.get_format(), Image.FORMAT_L8, "grayscale get_image -> FORMAT_L8 (deploy auto-detect)")
	s6.free()
	vp6.free()

	# Missing viewport and no capture fn -> null (no crash).
	var s5 = CameraSensor.new()
	h.assert_true(s5.get_image() == null, "get_image with no viewport/capture -> null")
	s5.free()

	# --- obs_size (#362): render big, observe small — one viewport serves display AND a cheap
	# policy input. obs_size set -> the captured frame is downscaled to it before encoding, and
	# get_obs_shape() reports it. Uniform-color frames survive resampling byte-exactly, so the hex
	# is deterministic. Default (0,0) = no resize (all earlier cases pin that path).
	var s7 = CameraSensor.new()
	var vp7 := SubViewport.new()
	vp7.size = Vector2i(8, 8)
	s7.viewport = vp7
	s7.obs_size = Vector2i(2, 2)
	s7.set_image_for_test(_make_image(8, 8, Image.FORMAT_RGB8, Color(1, 0, 0)))
	h.assert_eq(s7.get_obs_shape(), [2, 2, 3], "obs_size overrides viewport size in obs_shape")
	h.assert_eq(s7.get_obs_space_entry(), {"space": "box", "size": [2, 2, 3]}, "obs_size obs_space entry")
	h.assert_eq(s7.get_observation(), "ff0000ff0000ff0000ff0000", "8x8 red frame downscales to 2x2 red obs")
	var small: Image = s7.get_image()
	h.assert_eq(small.get_width(), 2, "get_image resized to obs_size width")
	h.assert_eq(small.get_height(), 2, "get_image resized to obs_size height")
	h.assert_eq(small.get_format(), Image.FORMAT_RGB8, "resized get_image keeps target format")
	# The grayscale path resizes then converts (deploy parity with training obs).
	s7.grayscale = true
	s7.set_image_for_test(_make_image(8, 8, Image.FORMAT_RGBA8, Color(1, 1, 1)))
	h.assert_eq(s7.get_obs_shape(), [2, 2, 1], "obs_size + grayscale obs_shape")
	h.assert_eq(s7.get_observation(), "ffffffff", "obs_size + grayscale white obs")
	var small_gray: Image = s7.get_image()
	h.assert_eq(small_gray.get_format(), Image.FORMAT_L8, "obs_size + grayscale get_image -> L8")
	h.assert_eq(small_gray.get_width(), 2, "obs_size + grayscale get_image width")
	s7.free()
	vp7.free()

	h.finish(self)
