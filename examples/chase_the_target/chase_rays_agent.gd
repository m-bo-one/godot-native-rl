extends "res://examples/chase_the_target/chase_agent.gd"

# chase_rays (#364): the chase agent + an 8-ray surround RaycastSensor2D appended to the obs
# (5 base floats + 8 closenesses = 13). The sensor is a child of the agent body (rotation 0 ->
# world-fixed fan), matching the twin's ray_closenesses exactly.

@export var ray_sensor_path: NodePath

var _ray_sensor


func _ready() -> void:
	super._ready()
	_ray_sensor = get_node_or_null(ray_sensor_path)
	if _ray_sensor == null:
		# Fail LOUD: the shipped net expects 13 inputs; a silent 5-float obs would feed the ncnn
		# runner a garbage-shaped input and just "behave badly" — a scene-configuration error.
		push_error("ChaseRaysAgent: ray_sensor_path not set/invalid — obs will be 5 floats but the trained net expects 13. Fix the scene wiring.")


func get_obs() -> Dictionary:
	var base: Array = super.get_obs()["obs"]
	if _ray_sensor != null:
		base.append_array(_ray_sensor.get_observation())
	return {"obs": base}
