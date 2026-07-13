# Path-based extends, no bare class_name (headless script-cache gotcha — see CLAUDE.md).
extends CanvasLayer
# Cosmetic "camera view" inset (#229): renders the ACTUAL 36x36x3 frame the CNN policy consumes,
# nearest-neighbor upscaled in a small bordered panel (top-right). It reuses the agent's PUBLIC
# get_inference_image() — the exact Image handed to NcnnRunner.run_inference_image — so it shows
# what the net sees without changing the obs/action flow at all. Read-only + inert under --headless
# (no render context), so it never affects training scenes or the regression checks.

@export var agent_path: NodePath                    ## the VisualChaseAgent (has get_inference_image)
@export var refresh_every := 2                       ## frames between texture refreshes
@export var inset_size := 144                        ## on-screen pixel size of the upscaled frame
@export var caption := "policy input 36x36"

const BORDER := 2
const MARGIN := 12
const CAPTION_H := 20

var _agent: Node = null
var _tex: ImageTexture = ImageTexture.new()
var _texrect: TextureRect = null
var _counter := 0


func _ready() -> void:
	layer = 95  # above the game canvas, alongside the DemoLegend (layer 90)
	_agent = get_node_or_null(agent_path)
	var total := inset_size + BORDER * 2

	# Container anchored to the top-right corner (screen space under the CanvasLayer).
	var box := Control.new()
	box.name = "InsetBox"
	box.anchor_left = 1.0
	box.anchor_right = 1.0
	box.anchor_top = 0.0
	box.anchor_bottom = 0.0
	box.offset_left = -(total + MARGIN)
	box.offset_right = -MARGIN
	box.offset_top = MARGIN
	box.offset_bottom = MARGIN + total + CAPTION_H
	box.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(box)

	# Border: a light rect behind the frame.
	var border := ColorRect.new()
	border.name = "Border"
	border.color = Color(0.85, 0.88, 0.95, 1.0)
	border.position = Vector2.ZERO
	border.size = Vector2(total, total)
	border.mouse_filter = Control.MOUSE_FILTER_IGNORE
	box.add_child(border)

	# The live frame view — nearest-neighbor so pixels read as chunky pixels.
	_texrect = TextureRect.new()
	_texrect.name = "FrameView"
	_texrect.texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST
	_texrect.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	_texrect.stretch_mode = TextureRect.STRETCH_SCALE
	_texrect.position = Vector2(BORDER, BORDER)
	_texrect.size = Vector2(inset_size, inset_size)
	_texrect.mouse_filter = Control.MOUSE_FILTER_IGNORE
	box.add_child(_texrect)

	var label := Label.new()
	label.name = "Caption"
	label.text = caption
	label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	label.add_theme_font_size_override("font_size", 12)
	label.position = Vector2(0, total + 2)
	label.size = Vector2(total, CAPTION_H)
	label.mouse_filter = Control.MOUSE_FILTER_IGNORE
	box.add_child(label)


func _process(_delta: float) -> void:
	if _agent == null or not _agent.has_method("get_inference_image"):
		return
	_counter += 1
	if _counter % maxi(refresh_every, 1) != 0:
		return
	var img: Image = _agent.get_inference_image()
	if img == null:
		return
	if _tex.get_width() != img.get_width() or _tex.get_height() != img.get_height():
		_tex = ImageTexture.create_from_image(img)
	else:
		_tex.update(img)
	if _texrect != null:
		_texrect.texture = _tex
