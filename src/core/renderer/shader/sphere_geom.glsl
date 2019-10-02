layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

// in camera space
layout(location = 0) in vec3 in_position[1];
layout(location = 1) in float in_radius[1];
layout(location = 2) flat in uint in_materialIndex[1];

// sphere center in camera space
layout(location = 0) flat out vec3 out_center;
// sphere radius
layout(location = 1) flat out float out_radius;
// view space position
layout(location = 2) out vec3 out_position;
layout(location = 3) flat out uint out_materialIndex;

const vec2 offsets[] = { vec2(-1.0, 1.0), vec2(1.0, 1.0), vec2(-1.0, -1.0), vec2(1.0, -1.0) };

// span a quad in view space that contains the entire sphere
void main() {
	out_radius = in_radius[0];
	out_materialIndex = in_materialIndex[0];
	out_center = in_position[0];

	// get vectors that are perpendicular to in_position
	vec3 planeX = normalize(cross(in_position[0], vec3(0.0, 1.0, 0.0)));
	vec3 planeY = normalize(cross(in_position[0], planeX));

	for(int i = 0; i < 4; ++i) {
		// view space
		//out_position = vec3(in_position[0].xy + offsets[i] * in_radius[0], in_position[0].z);
		out_position = in_position[0] + offsets[i].x * planeX * in_radius[0] + offsets[i].y * planeY * in_radius[0];
		// projection
		gl_Position = u_cam.projection * vec4(out_position, 1.0);
		EmitVertex();
	}
	EndPrimitive();
}