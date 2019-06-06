#define RESOLUTION 64

layout(points) in;
layout(line_strip, max_vertices = RESOLUTION) out;

// in camera space
layout(location = 0) in vec3 in_position[1];
layout(location = 1) in float in_radius[1];

layout(binding = 0) uniform u_camTrans
{
	CameraTransforms u_cam;
};

layout(location = 2) uniform vec2 screenSize;

void main() {
	// 2 pi / RESOLUTION
	const float factor = 6.28318530718 / float(RESOLUTION - 1);

	// draw circle in view space
	for(int i = 0; i <= RESOLUTION; ++i) {
		float t = i * factor;

		vec3 p = vec3(cos(t), sin(t), 0.0);
		gl_Position = u_cam.projection * vec4(in_position[0] + p * in_radius[0], 1.0);
		// increase the radius and position a little bit because drawing exactly on the radius will probably be hidden by the depth test
		gl_Position.xy += p.xy * 2.0 / screenSize * gl_Position.w; // move one pixel in the current direction
		EmitVertex();
	}

	EndPrimitive();
}