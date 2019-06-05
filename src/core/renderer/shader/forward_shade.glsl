#define FORWARD_SHADE
layout(location = 0) out vec4 out_fragColor;
layout(location = 1) out vec3 out_position;
layout(location = 2) out vec3 out_albedo;
layout(location = 3) out vec3 out_normal;
layout(location = 4) out vec3 out_lightness;

void shade(vec3 pos, vec3 normal, vec2 texcoord) {
	out_normal = normal;
	out_position = pos;
	out_albedo = vec3(1.0f);
	out_lightness = vec3(0.0f);
	out_fragColor = vec4(normal * 0.5 + vec3(0.5), 1.0);
}