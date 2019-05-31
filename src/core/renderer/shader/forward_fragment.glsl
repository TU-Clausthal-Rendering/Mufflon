layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_texcoord;

layout(location = 0) out vec4 out_fragColor;

void main() {
	vec3 N = normalize(in_normal);
	out_fragColor = vec4(N * 0.5 + vec3(0.5), 1.0);
}