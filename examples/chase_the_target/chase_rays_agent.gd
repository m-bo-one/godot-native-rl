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
		push_warning("ChaseRaysAgent: ray_sensor_path not set — obs will be the 5 base floats only.")


func get_obs() -> Dictionary:
	var base: Array = super.get_obs()["obs"]
	if _ray_sensor != null:
		base.append_array(_ray_sensor.get_observation())
	return {"obs": base}
