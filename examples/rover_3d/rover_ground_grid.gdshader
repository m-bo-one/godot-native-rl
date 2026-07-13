// Cosmetic procedural grid for the rover_3d demo ground (#229). Purely visual — no effect on
// physics, collision, or observations. Draws subtle lighter lines over a dark base so the
// rover/goal still pop while the moving rover reads clearly against a static grid.
shader_type spatial;
render_mode cull_back, diffuse_burley;

uniform vec3 base_color : source_color = vec3(0.10, 0.17, 0.12);
uniform vec3 line_color : source_color = vec3(0.22, 0.38, 0.30);
uniform float cells = 20.0;       // grid divisions across the mesh UV (0..1)
uniform float line_width = 0.03;  // line thickness as a fraction of a cell

void fragment() {
	vec2 g = fract(UV * cells);
	vec2 d = min(g, 1.0 - g);
	float line = 1.0 - smoothstep(0.0, line_width, min(d.x, d.y));
	ALBEDO = mix(base_color, line_color, line);
	ROUGHNESS = 0.9;
}
