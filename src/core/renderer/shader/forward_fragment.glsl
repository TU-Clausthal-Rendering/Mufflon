layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_texcoord;

layout(location = 0) out vec4 out_fragColor;
layout(location = 1) out vec3 out_position;
layout(location = 2) out vec3 out_albedo;
layout(location = 3) out vec3 out_normal;
layout(location = 4) out vec3 out_lightness;

void main() {
	vec3 N = normalize(in_normal);
	out_normal = N; // mapped to 0,1: N * 0.5 + vec3(0.5)
	out_position = in_position;
	out_albedo = vec3(1.0f);
	out_lightness = vec3(0.0f);
	out_fragColor = vec4(N * 0.5 + vec3(0.5), 1.0);
}