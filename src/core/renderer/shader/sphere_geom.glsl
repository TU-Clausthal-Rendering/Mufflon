layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

// in camera space
layout(location = 0) in vec3 in_position[1];
layout(location = 1) in float in_radius[1];

// location in [-1, 1]² range
layout(location = 0) out vec2 out_location;
// sphere radius
layout(location = 1) flat out float out_radius;
// view space position
layout(location = 2) out vec3 out_position;

layout(binding = 0) uniform u_camTrans
{
	CameraTransforms u_cam;
};

const vec2 offsets[] = { vec2(-1.0, 1.0), vec2(1.0, 1.0), vec2(-1.0, -1.0), vec2(1.0, -1.0) };

// span a quad in view space that contains the entire sphere
void main() {
	out_radius = in_radius[0];
	for(int i = 0; i < 4; ++i) {
		// view space
		out_position = vec3(in_position[0].xy + offsets[i] * in_radius[0], in_position[0].z);
		// projection
		gl_Position = u_cam.projection * vec4(out_position, 1.0);
		// [-1, 1]
		out_location = offsets[i];
		EmitVertex();
	}
	EndPrimitive();
}